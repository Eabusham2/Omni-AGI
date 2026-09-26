import builtins
import json
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.capability_rehearsal import (
    CapabilityRehearsalPolicy,
    CapabilityScheduleState,
    advance_schedule_state,
    capability_probes,
    due_rehearsal_phase,
)
from omni_core.config import OmniConfig
from omni_core.distributed_runtime import (
    DatasetManifest,
    DistributedRunStore,
    RankCursor,
)
from omni_core.distributed_training import (
    DistributedBrainTrainingModule,
    DistributedGroundUpTrainer,
    DistributedTrainingOptions,
    _canonical_sha256,
    _due_distributed_rehearsal_phase,
    _finite_wave_progress,
    _new_training_optimizer,
    _resolve_strategy,
    _wrap_module,
    _safe_distributed_coverage,
    _safe_resource_telemetry,
    _telemetry_ledger_receipt,
    _validate_distributed_origin_template,
    _validate_promotion_receipt,
    merge_media_training_state,
)
from omni_core.optimizers import PackedOnlyOptimizer
from omni_core.model import (
    PackedAdaptiveBitConv1d,
    PackedAdaptiveBitLinear,
    PackedAdaptiveTernaryEmbedding,
)
from omni_core.ground_up import (
    current_ground_up_training_receipt_contract,
    ground_up_curriculum_manifest,
    seal_ground_up_v3_training_manifest,
    seal_ground_up_v3_training_receipt,
)


class DistributedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-distributed-runtime-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _parquet(self, values):
        try:
            import pyarrow as pa
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("PyArrow is not installed")
        path = self.root / "tiny.parquet"
        parquet.write_table(pa.table({"text": list(values)}), path, row_group_size=1)
        return path

    def test_manifest_shards_every_valid_parquet_row_once_and_detects_mutation(self):
        path = self._parquet(["zero", "one", "one", "three", "four"])
        manifest = DatasetManifest.build(path)
        manifest_path = self.root / "manifest.json"
        manifest.write(manifest_path)
        loaded = DatasetManifest.read(manifest_path)

        rank_zero = [
            entry.ordinal
            for entry, _record in loaded.iter_verified_records(rank=0, world_size=2)
        ]
        rank_one = [
            entry.ordinal
            for entry, _record in loaded.iter_verified_records(rank=1, world_size=2)
        ]
        self.assertEqual(rank_zero, [0, 2, 4])
        self.assertEqual(rank_one, [1, 3])
        self.assertEqual(sorted(rank_zero + rank_one), list(range(5)))
        self.assertEqual(len({entry.record_id for entry in loaded.entries}), 5)
        self.assertFalse(loaded.to_dict()["rawSourceStored"])

        self._parquet(["zero", "changed", "one", "three", "four"])
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            loaded.verify_current_source()
        manifest.close()

    def test_sqlite_manifest_indexes_over_100k_rows_with_bounded_python_memory(self):
        source = self.root / "large.jsonl"
        with source.open("w", encoding="utf-8") as stream:
            for index in range(100_003):
                stream.write(json.dumps({"text": "record-%d" % index}) + "\n")
        pointer = self.root / "large-manifest.json"
        tracemalloc.start()
        manifest = DatasetManifest.build(
            source,
            database_path=pointer.with_suffix(".sqlite3"),
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        manifest.write(pointer)

        self.assertEqual(len(manifest.entries), 100_003)
        self.assertNotIsInstance(manifest.entries, (list, tuple))
        self.assertEqual(manifest.entries[100_002].name, "large.jsonl#line-100003")
        self.assertLess(peak, 64 * 1024 * 1024)
        self.assertLess(pointer.stat().st_size, 64 * 1024)
        loaded = DatasetManifest.read(pointer)
        self.assertEqual(loaded.owned_record_count(0, 2), 50_002)
        self.assertEqual(loaded.owned_record_count(1, 2), 50_001)
        self.assertTrue(loaded.to_dict()["boundedMemoryBuild"])
        resumed_suffix = [
            entry.ordinal
            for entry, _record in loaded.iter_verified_records(
                rank=1,
                world_size=2,
                start_ordinal=99_998,
            )
        ]
        self.assertEqual(resumed_suffix, [99_999, 100_001])

    def test_checkpoint_publishes_rank_cursors_with_matching_native_generation(self):
        dataset = self._parquet(["alpha", "beta"])
        manifest = DatasetManifest.build(dataset)
        store = DistributedRunStore(self.root / "run")
        store.initialize()
        brain_json = self.root / "brain.json"
        brain_json.write_text(
            json.dumps(
                {
                    "mutable_state": {"activeGeneration": "a" * 64},
                    "substrate": {
                        "persistence": {"activeGeneration": "b" * 64}
                    },
                }
            ),
            encoding="utf-8",
        )
        cursors = [
            RankCursor(rank, 2, 0, 2, 1, 1, manifest.content_sha256)
            for rank in range(2)
        ]
        checkpoint = store.publish_checkpoint(
            brain_json_path=brain_json,
            manifest=manifest,
            cursors=cursors,
            epochs_requested=1,
            global_optimizer_steps=1,
            dynamic_high_water=2,
            strategy="ddp",
            telemetry={"rankCount": 2},
            capability_rehearsal={"startCompleted": True},
        )
        loaded = store.load_active_checkpoint(
            manifest_sha256=manifest.content_sha256, world_size=2
        )
        self.assertEqual(loaded, checkpoint)
        self.assertEqual(loaded["dynamicHighWater"], 2)
        self.assertEqual(len(loaded["rankCursors"]), 2)
        self.assertTrue(loaded["capabilityRehearsal"]["startCompleted"])
        with self.assertRaisesRegex(ValueError, "WORLD_SIZE changed"):
            store.load_active_checkpoint(
                manifest_sha256=manifest.content_sha256, world_size=1
            )

    def test_rehearsal_schedule_is_global_not_rank_local(self):
        policy = CapabilityRehearsalPolicy(periodic_global_waves=4)
        state = CapabilityScheduleState()
        self.assertEqual(
            due_rehearsal_phase(
                state, policy, committed_global_waves=0
            ),
            "start",
        )
        state = CapabilityScheduleState(start_completed=True)
        self.assertIsNone(
            due_rehearsal_phase(state, policy, committed_global_waves=3)
        )
        self.assertEqual(
            due_rehearsal_phase(state, policy, committed_global_waves=4),
            "middle",
        )
        self.assertEqual(
            due_rehearsal_phase(
                state, policy, committed_global_waves=4, final=True
            ),
            "final",
        )

    def test_finite_midpoint_rehearsal_uses_completed_waves_and_survives_resume(self):
        policy = CapabilityRehearsalPolicy(periodic_global_waves=128)
        self.assertEqual(
            _finite_wave_progress(
                record_count=7,
                global_batch_records=3,
                epochs=1,
                completed_epochs=0,
                next_global_ordinal=3,
            ),
            (1, 3),
        )
        with self.assertRaisesRegex(ValueError, "not a completed wave"):
            _finite_wave_progress(
                record_count=7,
                global_batch_records=3,
                epochs=1,
                completed_epochs=0,
                next_global_ordinal=4,
            )

        def due(state, *, ordinal, global_waves, final=False):
            return _due_distributed_rehearsal_phase(
                state,
                policy,
                committed_global_waves=global_waves,
                record_count=7,
                global_batch_records=3,
                epochs=1,
                completed_epochs=0 if not final else 1,
                next_global_ordinal=ordinal,
                final=final,
            )

        state = CapabilityScheduleState()
        self.assertEqual(
            due(state, ordinal=0, global_waves=0),
            "start",
        )
        state = advance_schedule_state(
            state,
            {
                "format": "omni-capability-rehearsal",
                "phase": "start",
                "committedGlobalWaves": 0,
                "after": {"minimumExpectedProbability": 0.8},
            },
        )
        self.assertIsNone(
            due(state, ordinal=3, global_waves=1)
        )
        self.assertEqual(
            due(state, ordinal=6, global_waves=2),
            "middle",
        )
        state = advance_schedule_state(
            state,
            {
                "format": "omni-capability-rehearsal",
                "phase": "middle",
                "committedGlobalWaves": 2,
            },
        )
        resumed = CapabilityScheduleState.from_dict(state.to_dict())
        self.assertTrue(resumed.middle_completed)
        self.assertIsNone(
            due(resumed, ordinal=6, global_waves=2)
        )
        self.assertEqual(
            due(resumed, ordinal=0, global_waves=3, final=True),
            "final",
        )
        completed = advance_schedule_state(
            resumed,
            {
                "format": "omni-capability-rehearsal",
                "phase": "final",
                "committedGlobalWaves": 3,
            },
        )
        self.assertIsNone(
            due(completed, ordinal=0, global_waves=3, final=True)
        )

    def test_one_wave_run_has_no_artificial_middle_rehearsal(self):
        state = CapabilityScheduleState(start_completed=True, event_count=1)
        policy = CapabilityRehearsalPolicy(periodic_global_waves=128)
        self.assertIsNone(
            _due_distributed_rehearsal_phase(
                state,
                policy,
                committed_global_waves=1,
                record_count=1,
                global_batch_records=16,
                epochs=1,
                completed_epochs=0,
                next_global_ordinal=1,
            )
        )
        self.assertEqual(
            _due_distributed_rehearsal_phase(
                state,
                policy,
                committed_global_waves=1,
                record_count=1,
                global_batch_records=16,
                epochs=1,
                completed_epochs=1,
                next_global_ordinal=0,
                final=True,
            ),
            "final",
        )

    def test_midpoint_resume_at_zero_optimizer_steps_is_not_replayed(self):
        policy = CapabilityRehearsalPolicy(periodic_global_waves=128)
        state = CapabilityScheduleState(start_completed=True, event_count=1)
        arguments = {
            "committed_global_waves": 0,
            "record_count": 1,
            "global_batch_records": 16,
            "epochs": 2,
            "completed_epochs": 1,
            "next_global_ordinal": 0,
        }
        self.assertEqual(
            _due_distributed_rehearsal_phase(state, policy, **arguments),
            "middle",
        )
        state = advance_schedule_state(
            state,
            {
                "format": "omni-capability-rehearsal",
                "phase": "middle",
                "committedGlobalWaves": 0,
            },
        )
        resumed = CapabilityScheduleState.from_dict(state.to_dict())
        self.assertTrue(resumed.middle_completed)
        self.assertIsNone(
            _due_distributed_rehearsal_phase(resumed, policy, **arguments)
        )

    def test_strategy_auto_uses_ddp_on_cpu_and_fsdp_is_cuda_gated(self):
        context = SimpleNamespace(world_size=2, device=torch.device("cpu"))
        module = torch.nn.Linear(2, 2)
        auto = DistributedTrainingOptions(strategy="auto")
        self.assertEqual(_resolve_strategy(auto, context, module), "ddp")
        forced = DistributedTrainingOptions(strategy="fsdp")
        with self.assertRaisesRegex(ValueError, "FSDP requires"):
            _resolve_strategy(forced, context, module)

    def test_multi_rank_rejects_unsynchronized_packed_mutations_before_wrap(self):
        context = SimpleNamespace(world_size=2, local_rank=0, device=torch.device("cpu"))
        packed_modules = (
            PackedAdaptiveBitLinear(2, 2),
            PackedAdaptiveTernaryEmbedding(3, 2),
            PackedAdaptiveBitConv1d(1, 1, 1),
        )
        for packed in packed_modules:
            packed.online_learning_rate = 0.0
            module = torch.nn.Sequential(packed)
            for strategy in ("ddp", "fsdp"):
                with self.subTest(layer=type(packed).__name__, strategy=strategy):
                    with mock.patch(
                        "omni_core.distributed_training.DistributedDataParallel",
                        side_effect=AssertionError("DDP must not be constructed"),
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "direct backward mutations are not synchronized",
                        ):
                            _wrap_module(module, strategy=strategy, context=context)
            with self.assertRaisesRegex(RuntimeError, "direct backward mutations"):
                _wrap_module(module, strategy="single", context=context)
            single = SimpleNamespace(world_size=1, local_rank=0, device=torch.device("cpu"))
            self.assertIs(_wrap_module(module, strategy="single", context=single), module)

    def test_packed_only_single_rank_has_optimizer_adapter_and_zero_slot(self):
        brain = SimpleNamespace(
            config=SimpleNamespace(learning_rate=0.001, weight_decay=0.0),
            _streaming_experience_parameters=lambda: (),
            _optimizer=PackedOnlyOptimizer(),
        )
        self.assertIsInstance(
            _new_training_optimizer(brain, torch.nn.Identity(), None),
            PackedOnlyOptimizer,
        )
        module = torch.nn.Module()
        module.register_buffer("_autograd_trigger", torch.zeros((), requires_grad=True))
        zero = DistributedBrainTrainingModule._zero_loss(module)
        self.assertEqual(float(zero.detach()), 0.0)
        self.assertTrue(zero.requires_grad)

    def test_distributed_seed_template_rejects_mutable_post_origin_state(self):
        config = OmniConfig.micro(
            origin_kind="ground-up",
        )
        brain = SimpleNamespace(
            config=config,
            ground_up_training_manifest=ground_up_curriculum_manifest(),
            _validate_ground_up_training_manifest=(
                lambda: ground_up_curriculum_manifest()
            ),
            _ground_up_action_origin_verified=True,
            engine_path=self.root / "engine",
            messages=[],
            traces=[],
            training_sources=[],
            ingestion_checkpoints={},
            completed_ingestions=[],
            completed_chat_turns=[],
            replay=[],
            working_memory=[],
            paged_working_memory=SimpleNamespace(count=lambda: 0),
            workspace_items=[],
            recent_token_context=[],
            fresh_attention_boundary=None,
            current_context={
                "tokenCount": 0,
                "recentTokenCount": 0,
                "sensorySlots": 0,
                "tokenHash": "",
            },
            memory_lifecycle=SimpleNamespace(
                scratch_items=[], active_focus=[]
            ),
            installed_modality_packs=[],
            counters={"inference_count": 0},
            conversation=SimpleNamespace(
                summary=lambda: {
                    "totalEntries": 0,
                    "messageCount": 0,
                    "actionCount": 0,
                    "traceCount": 0,
                }
            ),
        )
        with mock.patch(
            "omni_core.distributed_training.eligible_ground_up_rehearsal",
            return_value=True,
        ), mock.patch(
            "omni_core.distributed_training.verify_ternary_shards"
        ):
            _validate_distributed_origin_template(brain, config)
            brain.training_sources = [{"name": "prior-private-dataset"}]
            with self.assertRaisesRegex(RuntimeError, "post-origin learning"):
                _validate_distributed_origin_template(brain, config)
            brain.training_sources = []
            brain.conversation.summary = lambda: {
                "totalEntries": 1,
                "messageCount": 0,
                "actionCount": 1,
                "traceCount": 0,
            }
            with self.assertRaisesRegex(RuntimeError, "post-origin learning"):
                _validate_distributed_origin_template(brain, config)
            brain.conversation.summary = lambda: {
                "totalEntries": 0,
                "messageCount": 0,
                "actionCount": 0,
                "traceCount": 0,
            }
            brain.recent_token_context = [42]
            with self.assertRaisesRegex(RuntimeError, "post-origin learning"):
                _validate_distributed_origin_template(brain, config)
            brain.recent_token_context = []
            mismatched = OmniConfig.micro(
                origin_kind="ground-up",
                d_model=64,
                idea_dim=64,
                n_heads=4,
            )
            with self.assertRaisesRegex(RuntimeError, "requested architecture"):
                _validate_distributed_origin_template(brain, mismatched)

    def test_promotion_receipt_helpers_bind_resources_and_reject_tampering(self):
        telemetry = self.root / "telemetry.jsonl"
        telemetry.write_bytes(b'{"rank":0}\n{"rank":1}\n')
        ledger = _telemetry_ledger_receipt(telemetry)
        self.assertEqual(ledger["records"], 2)
        self.assertEqual(len(ledger["sha256"]), 64)

        coverage = _safe_distributed_coverage(
            {
                "schemaVersion": 1,
                "discoveredFiles": 3,
                "processedFiles": 2,
                "rejectedFiles": 1,
                "discoveredRecords": 9,
                "processedRecords": 8,
                "rejectedRecords": 1,
                "discoveredBytes": 99,
                "processedBytes": 88,
                "shards": 2,
                "modalityCounts": {"image": 1},
                "errors": [{"source": "/private/path", "message": "bad"}],
                "errorLog": "/private/errors.jsonl",
                "complete": True,
            }
        )
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["errorCount"], 1)
        self.assertNotIn("errors", coverage)
        self.assertNotIn("errorLog", coverage)
        resources = _safe_resource_telemetry(
            {
                "rankCount": 1,
                "hosts": ["private-hostname"],
                "minimumDiskFreeBytes": 100,
                "perRank": [
                    {
                        "rank": 0,
                        "host": "private-hostname",
                        "device": "cpu",
                        "processPeakRssBytes": 50,
                    }
                ],
            }
        )
        self.assertNotIn("hosts", resources)
        self.assertNotIn("host", resources["perRank"][0])

        body = {
            "format": "omni-distributed-ground-up-promotion",
            "formatVersion": 2,
            "runIdentitySha256": "a" * 64,
            "runtimeReady": True,
            "originKind": "ground-up",
            "externalWeightFiles": [],
            "rlhf": False,
            "rewardModel": False,
            "preferenceLabels": False,
            "parameterChecksum": "b" * 64,
            "parameterEvidence": {
                "before": "a" * 64,
                "after": "b" * 64,
                "changed": True,
            },
            "parameterAccounting": {"totalNeuralParameters": 1},
            "resources": {
                "telemetryLedger": ledger,
                "finalCheckpoint": resources,
            },
            "dataset": {
                "coverage": coverage,
                "recordsExpected": 8,
                "recordsVisited": 8,
                "dynamicHighWater": 8,
            },
            "packedTernary": {
                "coverageComplete": True,
                "contentSha256": "c" * 64,
                "parameterChecksum": "b" * 64,
            },
        }
        receipt = {**body, "contentSha256": _canonical_sha256(body)}
        self.assertEqual(
            _validate_promotion_receipt(receipt, "a" * 64), receipt
        )
        receipt["dataset"]["coverage"]["processedRecords"] = 7
        with self.assertRaisesRegex(RuntimeError, "content checksum"):
            _validate_promotion_receipt(receipt, "a" * 64)

    def test_promotion_receipt_embeds_complete_portable_provenance(self):
        trainer = object.__new__(DistributedGroundUpTrainer)
        trainer.brain_id = "promotion-fixture"
        trainer.output_path = self.root / "output"
        trainer.options = SimpleNamespace(epochs=1, replace_output=False)
        trainer.context = SimpleNamespace(world_size=1)
        telemetry_path = self.root / "telemetry.jsonl"
        telemetry_path.write_text('{"rank":0}\n', encoding="utf-8")
        curriculum = ground_up_curriculum_manifest()
        capability_body = {
            "phase": "final",
            "appliedRank": 0,
            "regressionGatePassed": True,
            "curriculumId": curriculum["id"],
            "curriculumSha256": curriculum["sha256"],
            "recordsVisited": 33,
            "expectedRecords": 33,
            "action": {"calibrated": True},
            "after": {
                "passed": True,
                "correct": 8,
                "allToolTrajectories": {
                    "passed": True,
                    "routeCount": 27,
                    "correctRoutes": 27,
                    "negativeCount": 6,
                    "negativeNoActionCount": 6,
                },
            },
        }
        capability_receipt = {
            **capability_body,
            "contentSha256": _canonical_sha256(capability_body),
        }
        schedule = CapabilityScheduleState(
            start_completed=True,
            final_completed=True,
            last_receipt=capability_receipt,
        )
        media = merge_media_training_state(None, ())
        cursor = RankCursor(
            rank=0,
            world_size=1,
            epoch=1,
            next_global_ordinal=0,
            owned_records_completed=2,
            optimizer_steps_completed=1,
            manifest_sha256="d" * 64,
        )
        final_telemetry = {
            "rankCount": 1,
            "hosts": ["private-host"],
            "minimumDiskFreeBytes": 100,
            "diskPressure": False,
            "perRank": [
                {
                    "rank": 0,
                    "host": "private-host",
                    "device": "cpu",
                    "processPeakRssBytes": 50,
                }
            ],
        }
        final_checkpoint = {
            "contentSha256": "e" * 64,
            "rankCursors": [cursor.to_dict()],
            "globalOptimizerSteps": 1,
            "dynamicHighWater": 2,
            "capabilityRehearsal": schedule.to_dict(),
            "mediaTraining": media,
            "telemetry": final_telemetry,
        }
        trainer.store = SimpleNamespace(
            path=self.root / "run",
            telemetry_path=telemetry_path,
            load_active_checkpoint=lambda **_kwargs: final_checkpoint,
        )
        manifest = SimpleNamespace(
            entries=(object(), object()),
            content_sha256="d" * 64,
            record_chain_sha256="f" * 64,
            source=str(self.root / "selected.jsonl"),
            requested_kind="jsonl",
            coverage={
                "schemaVersion": 1,
                "discoveredFiles": 1,
                "processedFiles": 1,
                "rejectedFiles": 0,
                "discoveredRecords": 2,
                "processedRecords": 2,
                "rejectedRecords": 0,
                "discoveredBytes": 20,
                "processedBytes": 20,
                "shards": 1,
                "modalityCounts": {},
                "errors": [],
                "complete": True,
            },
        )
        ground_receipt = seal_ground_up_v3_training_receipt({
            **current_ground_up_training_receipt_contract(),
            "recordsVisited": 68,
            "completeCoverage": True,
            "parametersChanged": True,
            "substrateChanged": True,
            "substrateBefore": {
                "neurons": 0,
                "assemblies": 0,
                "synapses": 0,
            },
            "parameterChecksumAfter": "1" * 64,
            "parameterChecksumBefore": "0" * 64,
            "substrateAfter": {
                "neurons": 2,
                "assemblies": 1,
                "synapses": 1,
            },
            "modalityParametersChanged": False,
            "imaginationSelectorParametersChanged": False,
            "modalityParameterChecksumBefore": "4" * 64,
            "modalityParameterChecksumAfter": "4" * 64,
            "imaginationSelectorChecksumBefore": "5" * 64,
            "imaginationSelectorChecksumAfter": "5" * 64,
            "nativeCoreParameterTensors": 2,
            "changedNativeCoreParameterTensors": 1,
            "changedNativeCoreTensorNames": ["decoder.output.weight"],
        })
        ground_manifest = seal_ground_up_v3_training_manifest({
            **curriculum,
            "trainingReceipt": ground_receipt,
        })
        packed_manifest = {
            "contentSha256": "2" * 64,
            "coverage": {
                "complete": True,
                "eligibleTensorCount": 7,
            },
            "metadata": {
                "parameterChecksum": "3" * 64,
                "originKind": "ground-up",
                "baseFrozen": False,
                "groundUpCurriculumSha256": curriculum["sha256"],
                "groundUpTrainingManifestSha256": ground_manifest[
                    "contentSha256"
                ],
                "groundUpTrainingReceiptSha256": ground_receipt[
                    "contentSha256"
                ],
            },
        }
        brain = SimpleNamespace(
            brain_id="promotion-fixture",
            config=OmniConfig.micro(
                origin_kind="ground-up",
            ),
            ground_up_training_manifest=ground_manifest,
            _ground_up_action_origin_verified=True,
            engine_path=self.root / "fake-engine",
            parameter_checksum=lambda: "3" * 64,
            parameter_accounting=lambda: {"totalNeuralParameters": 9},
            memory=SimpleNamespace(
                neurons={"a": {}, "b": {}, "c": {}},
                assemblies=[{"id": "one"}, {"id": "two"}],
                synapses={"one": {}, "two": {}},
            ),
            training_sources=[],
            save=mock.Mock(),
            close=mock.Mock(),
        )
        brain.engine_path.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "capability readiness"):
            trainer._promote_output(
                manifest=manifest,
                cursors=[cursor],
                global_steps=1,
                dynamic_high_water=2,
                strategy="ddp",
                schedule_state=CapabilityScheduleState(
                    start_completed=True,
                    final_completed=True,
                ),
                media_training_state=media,
            )
        with mock.patch(
            "omni_core.distributed_training._copytree_transactional"
        ), mock.patch(
            "omni_core.distributed_training.AdaptiveBrain.load",
            return_value=brain,
        ), mock.patch(
            "omni_core.distributed_training.eligible_ground_up_rehearsal",
            return_value=True,
        ), mock.patch(
            "omni_core.distributed_training.verify_ternary_shards",
            return_value=SimpleNamespace(manifest=packed_manifest),
        ):
            receipt = trainer._promote_output(
                manifest=manifest,
                cursors=[cursor],
                global_steps=1,
                dynamic_high_water=2,
                strategy="ddp",
                schedule_state=schedule,
                media_training_state=media,
            )

        self.assertTrue(receipt["runtimeReady"])
        self.assertEqual(receipt["dataset"]["recordsExpected"], 2)
        self.assertEqual(receipt["dataset"]["recordsVisited"], 2)
        self.assertNotIn(
            "hosts", receipt["resources"]["finalCheckpoint"]
        )
        self.assertEqual(len(brain.training_sources), 1)
        self.assertEqual(
            brain.training_sources[0]["distributed_training_receipt"],
            receipt,
        )
        brain.save.assert_called_once_with()
        brain.close.assert_called_once_with()

        invalid_output = self.root / "invalid-output"
        (invalid_output / "engine").mkdir(parents=True)
        invalid_receipt = invalid_output / "engine" / "distributed-training.json"
        invalid_receipt.write_text(
            json.dumps({
                "format": "omni-distributed-ground-up-promotion",
                "formatVersion": 2,
                "runPath": str(self.root / "run"),
            }),
            encoding="utf-8",
        )
        trainer.output_path = invalid_output
        with self.assertRaisesRegex(RuntimeError, "receipt identity is invalid"):
            trainer._promote_output(
                manifest=manifest,
                cursors=[cursor],
                global_steps=1,
                dynamic_high_water=2,
                strategy="ddp",
                schedule_state=schedule,
                media_training_state=media,
            )
        self.assertTrue(invalid_receipt.is_file())

        occupied_output = self.root / "occupied-output"
        occupied_output.mkdir()
        sentinel = occupied_output / "keep.txt"
        sentinel.write_text("do not overwrite", encoding="utf-8")
        trainer.output_path = occupied_output
        with self.assertRaises(FileExistsError):
            trainer._promote_output(
                manifest=manifest,
                cursors=[cursor],
                global_steps=1,
                dynamic_high_water=2,
                strategy="ddp",
                schedule_state=schedule,
                media_training_state=media,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite")

        trainer.options.replace_output = True
        brain.training_sources.clear()
        with mock.patch(
            "omni_core.distributed_training._copytree_transactional"
        ), mock.patch(
            "omni_core.distributed_training.AdaptiveBrain.load",
            return_value=brain,
        ), mock.patch(
            "omni_core.distributed_training.eligible_ground_up_rehearsal",
            return_value=True,
        ), mock.patch(
            "omni_core.distributed_training.verify_ternary_shards",
            return_value=SimpleNamespace(manifest=packed_manifest),
        ):
            upgraded = trainer._promote_output(
                manifest=manifest,
                cursors=[cursor],
                global_steps=1,
                dynamic_high_water=2,
                strategy="ddp",
                schedule_state=schedule,
                media_training_state=media,
            )
        self.assertEqual(upgraded["formatVersion"], 2)
        self.assertEqual(brain.save.call_count, 2)
        self.assertEqual(brain.close.call_count, 2)

    def test_windows_probes_use_windows_absolute_paths(self):
        import omni_core.capability_rehearsal as rehearsal

        with mock.patch.object(rehearsal.os, "name", "nt"):
            probes = {probe.name: probe for probe in capability_probes()}
        self.assertEqual(
            probes["files"].expected_arguments["path"],
            "C:\\work\\omni-capability-probe.txt",
        )
        self.assertEqual(
            probes["shell"].expected_arguments["cwd"], "C:\\work"
        )
        self.assertIn("PowerShell", probes["shell"].route_text)

    def test_module_import_survives_missing_posix_resource_on_windows(self):
        script = r'''
import builtins, importlib, sys, torch
original = builtins.__import__
sys.modules.pop("resource", None)
def guarded(name, *args, **kwargs):
    if name == "resource":
        raise ImportError("simulated Windows")
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
module = importlib.import_module("omni_core.distributed_runtime")
assert module._resource is None
assert module._process_peak_rss_bytes() >= 0
'''
        environment = {**os.environ, "PYTHONPATH": str(ENGINE)}
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
