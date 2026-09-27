import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import MethodType
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.datasets import DatasetRecord
from worker import Worker


class RecordCheckpointResumeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(307)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-record-checkpoint-"
        )
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _compact(brain: AdaptiveBrain) -> None:
        original = brain._streaming_neural_storage_plan

        def compact_plan(self, source_bytes):
            plan = original(source_bytes)
            plan.update(
                {
                    "detailedRecordAssemblies": False,
                    "corpusRepresentation": (
                        "shared-semantic-field-and-local-synapses"
                    ),
                    "slowGradientMode": (
                        "streaming-microbatch-gradient-accumulation"
                    ),
                }
            )
            return plan

        brain._streaming_neural_storage_plan = MethodType(compact_plan, brain)
        brain._ingestion_checkpoint_records = 2

    def _create(self, folder: str) -> AdaptiveBrain:
        brain = AdaptiveBrain.create(
            "record-recovery",
            self.root / folder,
            OmniConfig.micro(
                max_seq_len=32,
                train_batch_size=2,
                gradient_accumulation=2,
            ),
            initialize_ground_up=True,
        )
        self._compact(brain)
        return brain

    @staticmethod
    def _force_training_schedule(
        brain: AdaptiveBrain, *, physical_batch: int, sequence_tokens: int
    ) -> None:
        original = brain._training_resource_plan

        def forced_plan(self):
            plan = original()
            effective_target = int(plan["effectiveBatchTarget"])
            if effective_target % physical_batch:
                raise AssertionError(
                    "forced physical batch must divide the logical target"
                )
            plan.update(
                {
                    "physicalBatchRecords": physical_batch,
                    "gradientAccumulation": (
                        effective_target // physical_batch
                    ),
                    "windowTokens": sequence_tokens,
                    "pauseBeforeStep": False,
                }
            )
            self._runtime_train_batch_size = physical_batch
            self._runtime_training_max_seq_len = sequence_tokens
            return plan

        brain._training_resource_plan = MethodType(forced_plan, brain)

    def _dataset(self) -> Path:
        path = self.root / "records.jsonl"
        values = [
            "alpha recovery atom",
            "beta recovery atom",
            "gamma recovery atom",
            "delta recovery atom",
            "epsilon recovery atom",
        ]
        path.write_text(
            json.dumps({"text": values[0]})
            + "\n"
            + "{ malformed json row\n"
            + "".join(
                json.dumps({"text": value}) + "\n" for value in values[1:]
            ),
            encoding="utf-8",
        )
        return path

    def test_auto_divisor_schedule_keeps_ssd_safe_checkpoint_cadence(self):
        brain = AdaptiveBrain(
            "record-cadence",
            self.root / "record-cadence",
            OmniConfig.micro(
                train_batch_size=2,
                gradient_accumulation=8,
            ),
        )
        neural_plan = {
            "detailedRecordAssemblies": False,
            "physicalBatchRecords": 4,
            "gradientAccumulation": 4,
            "trainingSequenceTokens": 256,
            "corpusRepresentation": (
                "shared-semantic-field-and-local-synapses"
            ),
            "slowGradientMode": (
                "streaming-microbatch-gradient-accumulation"
            ),
        }
        schedule = brain._ingestion_learning_schedule(
            neural_plan,
            brain._ingestion_checkpoint_records,
        )
        encoded = json.dumps(
            schedule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

        self.assertEqual(schedule["physicalBatchRecords"], 4)
        self.assertEqual(schedule["gradientAccumulation"], 4)
        self.assertEqual(schedule["checkpointRecords"], 512)
        self.assertEqual(schedule["formatVersion"], 2)
        self.assertEqual(
            schedule["localTypedTargetWindowPolicy"],
            "role-bounded-causal-exact-byte-windows-v1",
        )
        self.assertEqual(
            brain._validated_ingestion_learning_schedule(
                schedule,
                hashlib.sha256(encoded).hexdigest(),
            ),
            schedule,
        )
        tampered = dict(schedule)
        tampered["localTypedTargetWindowPolicy"] = "tampered-policy"
        tampered_encoded = json.dumps(
            tampered,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "local typed window policy"):
            brain._validated_ingestion_learning_schedule(
                tampered, hashlib.sha256(tampered_encoded).hexdigest()
            )
        brain.events.close()

    def test_streaming_float_canonicalizer_is_relative_and_nonfinite_safe(self):
        left = torch.tensor([0.0005155043327249587], dtype=torch.float32)
        right = torch.tensor([0.0005155044491402805], dtype=torch.float32)
        AdaptiveBrain._canonicalize_streaming_float_tensor(left)
        AdaptiveBrain._canonicalize_streaming_float_tensor(right)
        self.assertTrue(torch.equal(left, right))

        smallest = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
        values = torch.tensor(
            [
                float(smallest),
                torch.finfo(torch.float32).tiny,
                -1.0e-20,
                1.0e-8,
                -0.0005155044,
                0.75,
                -1024.25,
                1.0e20,
            ],
            dtype=torch.float32,
        )
        before = values.clone()
        maximum_delta = AdaptiveBrain._canonicalize_streaming_float_tensor(
            values
        )
        relative_delta = (values - before).abs() / before.abs().clamp_min(
            torch.finfo(torch.float32).tiny
        )
        self.assertLessEqual(float(relative_delta.max().item()), 2.0**-19)
        self.assertEqual(maximum_delta, float((values - before).abs().max()))
        self.assertNotEqual(float(values[0]), 0.0)

        objective_values = torch.tensor(
            [-0.9, -0.2, 0.1, 0.7], dtype=torch.float32
        )
        targets = torch.tensor([-0.4, 0.0, 0.2, 0.5], dtype=torch.float32)
        loss_before = torch.mean((objective_values - targets) ** 2)
        AdaptiveBrain._canonicalize_streaming_float_tensor(objective_values)
        loss_after = torch.mean((objective_values - targets) ** 2)
        self.assertLessEqual(
            abs(float(loss_after - loss_before)),
            max(1.0, float(loss_before)) * 2.0**-19,
        )

        special = torch.tensor(
            [
                float("nan"),
                float("inf"),
                float("-inf"),
                0.0,
                -0.0,
                float(smallest),
                -float(smallest),
            ],
            dtype=torch.float32,
        )
        special_bits = special.view(torch.int32).clone()
        AdaptiveBrain._canonicalize_streaming_float_tensor(special)
        self.assertTrue(torch.equal(special.view(torch.int32), special_bits))

    @unittest.skipUnless(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "MPS is unavailable",
    )
    def test_streaming_float_canonicalizer_runs_natively_on_mps(self):
        smallest_subnormal = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        )
        smallest_normal = torch.tensor(
            torch.finfo(torch.float32).tiny,
            dtype=torch.float32,
        )
        next_normal = torch.nextafter(
            smallest_normal,
            torch.tensor(float("inf"), dtype=torch.float32),
        )
        last_first_bin = torch.nextafter(
            2.0 * smallest_normal,
            torch.tensor(0.0, dtype=torch.float32),
        )
        cpu_input = torch.tensor(
            [
                0.0,
                -0.0,
                float(smallest_subnormal),
                -float(smallest_subnormal),
                float(smallest_normal),
                -float(smallest_normal),
                float(next_normal),
                1.5 * float(smallest_normal),
                float(last_first_bin),
                2.0 * float(smallest_normal),
                1.4451447615393347e-31,
                -1.4451447615393347e-31,
                0.0005155043327249587,
                0.0005155044491402805,
                -1.0e-20,
                0.75,
                torch.finfo(torch.float32).max,
                -torch.finfo(torch.float32).max,
                float("nan"),
                float("inf"),
                float("-inf"),
            ],
            dtype=torch.float32,
        )
        cpu_expected = cpu_input.clone()
        cpu_delta = AdaptiveBrain._canonicalize_streaming_float_tensor(
            cpu_expected
        )
        values = cpu_input.to("mps")
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            AdaptiveBrain,
            "_unsupported_bitwise_canonicalization",
            side_effect=AssertionError(
                "native MPS bit canonicalization unexpectedly fell back"
            ),
        ):
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            maximum_delta = (
                AdaptiveBrain._canonicalize_streaming_float_tensor(values)
            )
            torch.mps.synchronize()
        after = values.cpu()
        self.assertEqual(maximum_delta, cpu_delta)
        self.assertTrue(
            torch.equal(after.view(torch.int32), cpu_expected.view(torch.int32))
        )
        finite_input = torch.isfinite(cpu_input)
        self.assertTrue(bool(torch.isfinite(after[finite_input]).all()))
        first_application = after.clone()
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(
            AdaptiveBrain,
            "_unsupported_bitwise_canonicalization",
            side_effect=AssertionError(
                "native MPS bit canonicalization unexpectedly fell back"
            ),
        ):
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            second_delta = (
                AdaptiveBrain._canonicalize_streaming_float_tensor(values)
            )
            torch.mps.synchronize()
        self.assertEqual(second_delta, 0.0)
        self.assertTrue(
            torch.equal(
                values.cpu().view(torch.int32),
                first_application.view(torch.int32),
            )
        )

    def test_first_oom_repartition_is_rehashed_before_schedule_commit(self):
        dataset = self._dataset()
        brain = self._create("oom-schedule")
        self._force_training_schedule(
            brain, physical_batch=2, sequence_tokens=32
        )
        original_loss = brain._experience_ids_batch_loss
        allocator_failed = False

        def fail_first_physical_batch(self, encoded, vectors):
            nonlocal allocator_failed
            if not allocator_failed and len(encoded) > 1:
                allocator_failed = True
                raise RuntimeError("cannot allocate memory for forced batch")
            return original_loss(encoded, vectors)

        brain._experience_ids_batch_loss = MethodType(
            fail_first_physical_batch, brain
        )
        original_save = brain.save
        stopped = False

        def stop_after_first_checkpoint():
            nonlocal stopped
            original_save()
            if brain.ingestion_checkpoints and not stopped:
                stopped = True
                raise RuntimeError("stop after OOM-adjusted schedule checkpoint")

        brain.save = stop_after_first_checkpoint
        with self.assertRaisesRegex(RuntimeError, "OOM-adjusted schedule"):
            brain.ingest(path=str(dataset), policy="encode", epoch=0)
        self.assertTrue(allocator_failed)
        brain.events.close()

        restored = AdaptiveBrain.load(self.root / "oom-schedule", "record-recovery")
        checkpoint = next(iter(restored.ingestion_checkpoints.values()))
        schedule = checkpoint["learningSchedule"]
        self.assertEqual(schedule["physicalBatchRecords"], 1)
        self.assertEqual(schedule["gradientAccumulation"], 4)
        self.assertEqual(
            schedule["physicalBatchRecords"]
            * schedule["gradientAccumulation"],
            4,
        )
        self.assertEqual(schedule["trainingSequenceTokens"], 32)
        encoded = json.dumps(
            schedule,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.assertEqual(
            checkpoint["learningScheduleSha256"],
            hashlib.sha256(encoded).hexdigest(),
        )
        self.assertEqual(
            checkpoint["neuralStateChecksum"], restored.parameter_checksum()
        )
        restored.events.close()

    def test_crash_rolls_back_partial_suffix_and_resumes_exactly_once(self):
        dataset = self._dataset()
        interrupted = self._create("interrupted")
        worker = Worker()
        worker.brains[interrupted.brain_id] = interrupted
        original_learn = interrupted.learn_experience
        calls = 0

        def crash_after_third_mutation(*args, **kwargs):
            nonlocal calls
            learned = original_learn(*args, **kwargs)
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated crash after partial suffix mutation")
            return learned

        interrupted.learn_experience = crash_after_third_mutation
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "partial suffix"):
                worker.ingest(
                    {
                        "brainId": interrupted.brain_id,
                        "storagePath": str(interrupted.storage_path),
                        "path": str(dataset),
                        "policy": "encode",
                        "epoch": 0,
                        "jobId": "interrupted-job",
                    },
                    "interrupted-request",
                )

        restored = worker.brains[interrupted.brain_id]
        self._compact(restored)
        self.assertEqual(len(restored.ingestion_checkpoints), 1)
        checkpoint = next(iter(restored.ingestion_checkpoints.values()))
        self.assertEqual(checkpoint["committedRecords"], 2)
        self.assertEqual(checkpoint["visitedRecords"], 3)
        self.assertEqual(checkpoint["processedRecords"], 2)
        self.assertEqual(checkpoint["rejectedRecords"], 1)
        self.assertEqual(
            checkpoint["neuralStateChecksum"], restored.parameter_checksum()
        )
        # The third record appended replay before the simulated crash. Recovery
        # must remove that row above the generation high-water mark.
        committed_replay = int(restored.mutable_state_manifest["replayCount"])
        self.assertEqual(len(restored.replay), committed_replay)
        rollback_events = [
            event
            for event in restored.events.recent(100)
            if event["kind"] == "ingestion-rollback"
        ]
        self.assertEqual(len(rollback_events), 1)
        self.assertFalse(
            rollback_events[0]["payload"][
                "uncommittedLearningRepresentedAsCommitted"
            ]
        )

        with contextlib.redirect_stdout(io.StringIO()):
            resumed = worker.ingest(
                {
                    "brainId": restored.brain_id,
                    "storagePath": str(restored.storage_path),
                    "path": str(dataset),
                    "policy": "encode",
                    "epoch": 0,
                    "jobId": "resumed-job",
                },
                "resumed-request",
            )
        final = worker.brains[restored.brain_id]

        control = self._create("control")
        control_result = control.ingest(
            path=str(dataset), policy="encode", epoch=0
        )
        self.assertEqual(
            final.parameter_checksum(), control.parameter_checksum()
        )
        self.assertEqual(len(final.replay), len(control.replay))
        self.assertEqual(final.counters, control.counters)
        self.assertEqual(resumed["coverage"], control_result["coverage"])
        self.assertEqual(resumed["coverage"]["processedRecords"], 5)
        self.assertEqual(resumed["coverage"]["rejectedRecords"], 1)
        self.assertEqual(resumed["recordRecovery"]["resumedRecords"], 2)
        self.assertEqual(resumed["recordRecovery"]["committedRecords"], 5)
        self.assertFalse(resumed["recordRecovery"]["checkpointActive"])
        source = resumed["source"]
        self.assertGreater(source["memory_synapse_update_events"], 0)
        self.assertGreater(source["synaptic_update_events"], 0)
        self.assertGreater(source["parameter_update_steps"], 0)
        self.assertEqual(
            source["neural_update_events"],
            source["synaptic_update_events"]
            + source["parameter_update_steps"],
        )
        self.assertEqual(
            source["plasticity_events"], source["neural_update_events"]
        )
        self.assertTrue(source["parameter_checksum_changed"])
        self.assertEqual(
            source["parameter_checksum_before"],
            resumed["parameterChecksumBefore"],
        )
        self.assertEqual(
            source["parameter_checksum_after"],
            resumed["parameterChecksumAfter"],
        )
        self.assertFalse(final.ingestion_checkpoints)
        self.assertEqual(len(final.training_sources), 1)
        field = final.memory.assemblies[0]
        self.assertEqual(field["statistical_experiences"], 5)
        self.assertTrue(
            all(
                int(synapse["effective_weight"]) in {-1, 0, 1}
                for synapse in final.memory.synapses.values()
            )
        )
        final_metadata = json.loads(
            (final.engine_path / "brain.json").read_text("utf-8")
        )
        self.assertEqual(final_metadata["ingestion_checkpoints"], {})
        final.events.close()
        control.events.close()

    def test_resume_binds_learning_schedule_across_resource_plan_drift(self):
        dataset = self._dataset()
        interrupted = self._create("schedule-interrupted")
        self._force_training_schedule(
            interrupted, physical_batch=2, sequence_tokens=32
        )
        original_save = interrupted.save
        stopped = False

        def stop_after_first_checkpoint():
            nonlocal stopped
            original_save()
            if interrupted.ingestion_checkpoints and not stopped:
                stopped = True
                raise RuntimeError("stop after schedule checkpoint")

        interrupted.save = stop_after_first_checkpoint
        with self.assertRaisesRegex(RuntimeError, "schedule checkpoint"):
            interrupted.ingest(path=str(dataset), policy="encode", epoch=0)
        interrupted.events.close()

        restored = AdaptiveBrain.load(
            self.root / "schedule-interrupted", "record-recovery"
        )
        self._compact(restored)
        checkpoint = next(iter(restored.ingestion_checkpoints.values()))
        schedule = checkpoint["learningSchedule"]
        self.assertEqual(schedule["physicalBatchRecords"], 2)
        self.assertEqual(schedule["trainingSequenceTokens"], 32)
        self.assertEqual(schedule["checkpointRecords"], 2)
        encoded_schedule = json.dumps(
            schedule, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self.assertEqual(
            checkpoint["learningScheduleSha256"],
            hashlib.sha256(encoded_schedule).hexdigest(),
        )
        self.assertFalse(
            {"text", "tokens", "tokenIds", "embeddings"}.intersection(schedule)
        )

        # A lower live resource recommendation after restart must not silently
        # repartition the already-committed transaction.
        self._force_training_schedule(
            restored, physical_batch=1, sequence_tokens=16
        )
        restored.ingest(path=str(dataset), policy="encode", epoch=0)

        control = self._create("schedule-control")
        self._force_training_schedule(
            control, physical_batch=2, sequence_tokens=32
        )
        control.ingest(path=str(dataset), policy="encode", epoch=0)
        self.assertEqual(
            restored.parameter_checksum(), control.parameter_checksum()
        )
        self.assertEqual(restored.counters, control.counters)
        self.assertFalse(restored.ingestion_checkpoints)
        restored.events.close()
        control.events.close()

    def test_checkpoint_rejects_mismatch_corruption_and_foreign_source(self):
        dataset = self._dataset()
        brain = self._create("mismatch")
        original_save = brain.save
        stopped = False

        def stop_after_committed_save():
            nonlocal stopped
            original_save()
            if brain.ingestion_checkpoints and not stopped:
                stopped = True
                raise RuntimeError("stop after durable record checkpoint")

        brain.save = stop_after_committed_save
        with self.assertRaisesRegex(RuntimeError, "durable record checkpoint"):
            brain.ingest(path=str(dataset), policy="encode", epoch=0)
        brain.events.close()

        active = AdaptiveBrain.load(self.root / "mismatch", "record-recovery")
        with self.assertRaisesRegex(ValueError, "does not match"):
            active.ingest(path=str(dataset), policy="pretrain", epoch=0)
        foreign = self.root / "foreign.jsonl"
        foreign.write_text(json.dumps({"text": "foreign"}) + "\n", "utf-8")
        with self.assertRaisesRegex(RuntimeError, "another record ingestion"):
            active.ingest(path=str(foreign), policy="encode", epoch=0)
        active.events.close()

        altered_root = self.root / "altered-stream"
        shutil.copytree(self.root / "mismatch", altered_root)
        altered = AdaptiveBrain.load(altered_root, "record-recovery")
        self._compact(altered)

        def same_count_changed_stream(_path, requested_kind="", coverage=None, **_kwargs):
            del requested_kind
            assert coverage is not None
            coverage.discovered_files += 1
            values = [
                "changed alpha record",
                "beta recovery atom",
                "gamma recovery atom",
                "delta recovery atom",
                "epsilon recovery atom",
            ]
            for index, value in enumerate(values):
                if index == 1:
                    coverage.reject("records.jsonl:2", "malformed fixture")
                encoded = value.encode("utf-8")
                coverage.discovered_records += 1
                coverage.processed_records += 1
                coverage.processed_bytes += len(encoded)
                yield DatasetRecord(
                    text=value,
                    name="records.jsonl:%d" % (index + 1),
                    bytes_read=len(encoded),
                    provenance={"format": "jsonl", "row": index + 1},
                )
            coverage.completed_files += 1

        with mock.patch(
            "omni_core.brain.iter_dataset_records", same_count_changed_stream
        ):
            with self.assertRaisesRegex(ValueError, "record content"):
                altered.ingest(path=str(dataset), policy="encode", epoch=0)
        altered.events.close()

        generation_root = self.root / "generation-mismatch"
        shutil.copytree(self.root / "mismatch", generation_root)
        generation_metadata_path = generation_root / "engine" / "brain.json"
        generation_metadata = json.loads(
            generation_metadata_path.read_text("utf-8")
        )
        generation_checkpoint = next(
            iter(generation_metadata["ingestion_checkpoints"].values())
        )
        binding = generation_checkpoint["committedGeneration"]
        binding["substrateContentSha256"] = "0" * 64
        generation_checkpoint["committedGenerationSha256"] = hashlib.sha256(
            json.dumps(
                binding, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        generation_metadata_path.write_text(
            json.dumps(
                generation_metadata, sort_keys=True, separators=(",", ":")
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "committed generation"):
            AdaptiveBrain.load(generation_root, "record-recovery")

        schedule_root = self.root / "schedule-mismatch"
        shutil.copytree(self.root / "mismatch", schedule_root)
        schedule_metadata_path = schedule_root / "engine" / "brain.json"
        schedule_metadata = json.loads(
            schedule_metadata_path.read_text("utf-8")
        )
        schedule_checkpoint = next(
            iter(schedule_metadata["ingestion_checkpoints"].values())
        )
        schedule = schedule_checkpoint["learningSchedule"]
        schedule["slowGradientMode"] = "invalid-non-neural-mode"
        schedule_checkpoint["learningScheduleSha256"] = hashlib.sha256(
            json.dumps(
                schedule, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        schedule_metadata_path.write_text(
            json.dumps(
                schedule_metadata, sort_keys=True, separators=(",", ":")
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "learning schedule"):
            AdaptiveBrain.load(schedule_root, "record-recovery")

        corrupt_root = self.root / "corrupt"
        shutil.copytree(self.root / "mismatch", corrupt_root)
        metadata_path = corrupt_root / "engine" / "brain.json"
        metadata = json.loads(metadata_path.read_text("utf-8"))
        checkpoint = next(iter(metadata["ingestion_checkpoints"].values()))
        checkpoint["committedRecords"] = -1
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "committedRecords"):
            AdaptiveBrain.load(corrupt_root, "record-recovery")

    def test_final_save_before_rpc_ack_reuses_completion_receipt(self):
        dataset = self._dataset()
        brain = self._create("completion-receipt")
        brain.consolidate = mock.Mock(
            side_effect=AssertionError(
                "ingestion must not run a separate manual consolidation pass"
            )
        )
        worker = Worker()
        worker.brains[brain.brain_id] = brain
        original_append = brain.events.append
        failed = False

        def fail_ack_event(kind, payload, job_id=None):
            nonlocal failed
            if kind == "ingestion" and not failed:
                failed = True
                raise RuntimeError("simulated transport loss after final save")
            return original_append(kind, payload, job_id=job_id)

        brain.events.append = fail_ack_event
        request = {
            "brainId": brain.brain_id,
            "storagePath": str(brain.storage_path),
            "path": str(dataset),
            # The legacy policy spelling maps to the same continuously settled
            # learning transaction; there is no non-transactional post-pass.
            "policy": "consolidate",
            "epoch": 0,
            "allowReplay": True,
            "jobId": "completion-lost-ack",
        }
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "transport loss"):
                worker.ingest(request, "lost-ack-request")

        committed = worker.brains[brain.brain_id]
        self.assertFalse(committed.ingestion_checkpoints)
        self.assertEqual(len(committed.completed_ingestions), 1)
        checksum = committed.parameter_checksum()
        replay_count = len(committed.replay)
        counters = dict(committed.counters)
        training_epochs = committed.training_sources[0]["training_epochs"]
        with contextlib.redirect_stdout(io.StringIO()):
            reused = worker.ingest(
                {**request, "jobId": "completion-retry"},
                "completion-retry-request",
            )
        after = worker.brains[brain.brain_id]
        self.assertTrue(reused["idempotentCompletion"])
        self.assertTrue(reused["recordRecovery"]["completionReceiptReused"])
        self.assertEqual(reused["transactionId"], after.training_sources[0]["transactionId"])
        self.assertEqual(after.parameter_checksum(), checksum)
        self.assertEqual(len(after.replay), replay_count)
        self.assertEqual(after.counters, counters)
        self.assertEqual(
            after.training_sources[0]["training_epochs"], training_epochs
        )
        self.assertEqual(len(after.completed_ingestions), 1)
        after.events.close()

    def test_source_mutation_before_batch_commit_never_publishes_suffix(self):
        dataset = self._dataset()
        brain = self._create("source-mutation")
        original_learn = brain.learn_experience
        calls = 0

        def mutate_during_second_record(*args, **kwargs):
            nonlocal calls
            learned = original_learn(*args, **kwargs)
            calls += 1
            if calls == 2:
                dataset.write_text(
                    json.dumps({"text": "mutated source with same workflow"})
                    + "\n",
                    encoding="utf-8",
                )
            return learned

        brain.learn_experience = mutate_during_second_record
        with self.assertRaisesRegex(ValueError, "source changed"):
            brain.ingest(path=str(dataset), policy="encode", epoch=0)
        # No checkpoint/source completion may name the old digest after the
        # source changed before its first batch publication.
        durable = json.loads(
            (brain.engine_path / "brain.json").read_text("utf-8")
        )
        self.assertEqual(durable["ingestion_checkpoints"], {})
        self.assertEqual(durable["completed_ingestions"], [])
        self.assertEqual(durable["training_sources"], [])
        brain.events.close()

    def test_checkpoint_aggregates_remain_bounded_for_thousands_of_reports(self):
        brain = self._create("bounded")
        accumulator = brain._empty_media_accumulator()
        huge_warning = "private-warning-" + ("w" * 1_000_000)
        huge_window_ids = ["private-assembly-%d" % index for index in range(100_000)]
        for index in range(5000):
            brain._accumulate_media_report(
                accumulator,
                {
                    "name": "private-media-name-%d" % index,
                    "kind": "image",
                    "contentSha256": "%064x" % index,
                    "provenance": {"caption": "private caption %d" % index},
                    "trained": True,
                    "loss": 1.0 / (index + 1),
                    "steps": 2,
                    "coverage": {
                        "unit": "images",
                        "windowSize": 1,
                        "windows": 1,
                        "discoveredUnits": 1,
                        "processedUnits": 1,
                        "tailUnits": 0,
                        "sensoryAssemblies": 2,
                        "wholeRecordAssemblyId": "private-whole-%d" % index,
                        "complete": True,
                    },
                    "sensory": {
                        "assemblyId": "private-record-%d" % index,
                        "windowAssemblyIds": huge_window_ids if index == 0 else ["private-window"],
                    },
                    "warnings": [huge_warning if index == 0 else "sample warning"],
                },
            )
        safe = brain._checkpoint_media_accumulator(accumulator)
        encoded = json.dumps(safe, sort_keys=True, separators=(",", ":"))
        self.assertEqual(safe["records"], 5000)
        self.assertEqual(safe["trainedRecords"], 5000)
        self.assertEqual(safe["byModality"]["image"]["records"], 5000)
        self.assertEqual(len(safe["recordSamples"]), 64)
        self.assertEqual(len(safe["warnings"]), 64)
        self.assertTrue(safe["recordSamplesTruncated"])
        self.assertTrue(safe["warningsTruncated"])
        self.assertNotIn("private-media-name", encoded)
        self.assertNotIn("private caption", encoded)
        self.assertNotIn("private-warning", encoded)
        self.assertNotIn("private-assembly", encoded)
        self.assertNotIn("windowAssemblyIds", encoded)
        self.assertEqual(
            safe["recordSamples"][0]["sensoryWindowAssemblyCount"],
            100_000,
        )
        self.assertEqual(brain._checkpoint_media_accumulator(safe), safe)
        resumed_shape = brain._media_result_from_accumulator(safe)
        self.assertEqual(resumed_shape["records"], safe["recordSamples"])
        self.assertLess(len(encoded), 100_000)
        brain.events.close()

    def test_large_private_source_streams_without_inline_brain_text(self):
        brain = AdaptiveBrain(
            "private-stream",
            self.root / "private-stream",
            OmniConfig.micro(
                memory_recipe="human-consolidation",
                retain_source_text=False,
                max_seq_len=32,
            ),
        )
        source = self.root / "large-private.jsonl"
        private_sentinel = "private-total-recall-sentinel-"
        with source.open("w", encoding="utf-8") as stream:
            for index in range(4096):
                stream.write(
                    json.dumps(
                        {
                            "text": private_sentinel
                            + str(index)
                            + "-"
                            + ("x" * 4100)
                        }
                    )
                    + "\n"
                )
        self.assertGreater(source.stat().st_size, 16 * 1024 * 1024)

        def fast_learn(self, text, **kwargs):
            del kwargs
            return {
                "assembly_id": hashlib.sha256(text.encode("utf-8")).hexdigest()[:24],
                "training": {"loss": 0.0},
            }

        def fast_streaming_batch(
            self,
            experiences,
            *,
            learning_schedule=None,
            schedule_locked=False,
        ):
            del self, schedule_locked
            schedule = dict(learning_schedule or {})
            return {
                "loss": 0.0,
                "records": float(len(experiences)),
                "optimizer_steps": 0.0,
                "physical_batch_records": float(
                    schedule.get("physicalBatchRecords", 1)
                ),
                "training_sequence_tokens": float(
                    schedule.get("trainingSequenceTokens", 32)
                ),
            }

        def fast_sequence(self, text, *, retain_exact):
            del self, text, retain_exact
            return {
                "associationKind": "multi-stage-temporal-episode",
                "rawTextStored": False,
                "rawTokenIdsStored": False,
                "exactEpisodicSegments": 0,
                "temporalLinks": 0,
                "cueKeysCreated": 0,
                "synapsesCreated": 0,
                "statisticalUpdates": 0,
            }

        brain.learn_experience = MethodType(fast_learn, brain)
        brain._optimize_streaming_experience_batch = MethodType(
            fast_streaming_batch, brain
        )
        brain._learn_sequence_associations = MethodType(
            fast_sequence, brain
        )
        brain._integrate_reading_record = MethodType(
            lambda self, *args, **kwargs: None, brain
        )
        # One bounded v3 checkpoint window covers this synthetic source;
        # source size no longer changes its neural representation.
        brain._ingestion_checkpoint_records = 4096
        result = brain.ingest(path=str(source), policy="encode")

        metadata_path = brain.engine_path / "brain.json"
        metadata = metadata_path.read_text("utf-8")
        self.assertFalse(result["source"]["raw_text_retained"])
        self.assertNotIn("raw_text", result["source"])
        self.assertNotIn(private_sentinel, metadata)
        self.assertLess(metadata_path.stat().st_size, 512 * 1024)
        self.assertEqual(result["coverage"]["processedRecords"], 4096)
        brain.events.close()


if __name__ == "__main__":
    unittest.main()
