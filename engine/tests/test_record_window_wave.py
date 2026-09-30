"""Literal scheduling/seals/storage and constructor-free production stubs.

No brain, neural module, application, model training, or CI is launched.
"""

import ast
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch.nn import functional as F

ENGINE = Path(__file__).resolve().parents[1]
PACKAGE = "omni_literal_window_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ENGINE / "omni_core")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, ENGINE / "omni_core" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spools = load("text_spool")
waves = load("record_window_wave")
seals = load("distributed_seal")
datasets = load("datasets")
runtime = load("distributed_runtime")
buffers = load("window_wave_buffer")


def source_function(file, name, class_name=None, globals_=None):
    tree = ast.parse((ENGINE / "omni_core" / file).read_text())
    scope = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name).body if class_name else tree.body
    node = copy.deepcopy(next(node for node in scope if isinstance(node, ast.FunctionDef) and node.name == name))
    node.decorator_list = []
    namespace = {"__package__": PACKAGE, "copy": copy, "json": json, "hashlib": hashlib,
        "Mapping": Mapping, "torch": torch, **(globals_ or {})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), name, "exec"), namespace)
    return namespace[name], node


class LiteralWindowContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.tokenizer = types.SimpleNamespace(bos_id=1, eos_id=2, human_id=3, brain_id=4, byte_offset=8)

    def giant(self, text):
        builder = spools.TextBuilder()
        builder.write(text)
        payload = builder.finish()
        self.addCleanup(payload.close)
        record = datasets.DatasetRecord(payload.text, "leased record", payload.bytes, text_payload=payload)
        return record, runtime.DatasetManifestEntry.from_record(0, record)

    def test_fixed_waves_and_resume_cover_every_byte_and_eos_once(self):
        text = "Aé😀" * 20_000 + " " * 4_001 + "tail"
        record, entry = self.giant(text)
        for length in (2, 7, 128):
            stream = waves.RecordWindowStream(record, entry, self.tokenizer, length)
            labels = []
            for _ in range(17):
                batch = stream.next_batch(1)
                self.assertLessEqual(len(batch), 1)
                labels.extend(batch[0].ids[1:])
            cursor = stream.state()
            stream.close()
            resumed = waves.RecordWindowStream(record, entry, self.tokenizer, length, cursor)
            while not resumed.complete:
                labels.extend(value for item in resumed.next_batch(1) for value in item.ids[1:])
            self.assertEqual(labels, [value + 8 for value in text.encode()] + [2])
            self.assertEqual(resumed.state()["completedWindows"], (len(text.encode()) + length - 1) // (length - 1))
            resumed.close()

    def test_leased_typed_targets_are_supervised_after_complete_literal_transcript(self):
        human, response = "human é " * 12_000, "response 😀 " * 8_000
        payload, provenance, rejected = spools.parse_spooled_json(io.StringIO(json.dumps({"messages": [
            {"role": "user", "content": human}, {"role": "assistant", "content": response},
            {"role": "assistant", "content": "tail"},
        ]})), (), set())
        self.assertIsNone(rejected)
        self.addCleanup(payload.close)
        record = datasets.DatasetRecord(payload.text, "dialogue", payload.bytes, provenance=provenance, text_payload=payload)
        entry = runtime.DatasetManifestEntry.from_record(0, record)
        stream = waves.RecordWindowStream(record, entry, self.tokenizer, 128)
        text_labels, target_labels = [], []
        restored = False
        while not stream.complete:
            for item in stream.next_batch(1):
                if item.phase == "text": text_labels.extend(item.ids[1:])
                else:
                    self.assertEqual(item.human.sha256, hashlib.sha256(human.strip().encode()).hexdigest())
                    target_labels.extend(item.ids[1:])
            if not restored and stream.state()["phase"] == "targets" and stream.state()["tokenCursor"] > 500:
                cursor = stream.state()
                stream.close()
                stream = waves.RecordWindowStream(record, entry, self.tokenizer, 128, cursor)
                restored = True
        self.assertTrue(restored)
        transcript = ("human: " + human.strip() + "\nbrain: " + response.strip() + "\nbrain: tail").encode()
        self.assertEqual(text_labels, [value + 8 for value in transcript] + [2])
        self.assertEqual(target_labels, [value + 8 for value in response.strip().encode()] + [2] + [value + 8 for value in b"tail"] + [2])
        stream.close()

    def test_record_cursor_rejects_geometry_identity_and_window_count_drift(self):
        record, entry = self.giant("record" * 20_000)
        stream = waves.RecordWindowStream(record, entry, self.tokenizer, 128)
        stream.next_batch(1)
        state = stream.state()
        for key, altered in (("recordId", "1" * 64), ("sequenceTokens", 64), ("completedWindows", 999), ("tokenCursor", record.text_payload.bytes + 2)):
            bad = {**state, key: altered}
            with self.assertRaises(ValueError):
                waves.RecordWindowStream(record, entry, self.tokenizer, 128, bad)
        stream.close()

    def test_admitted_native_nested_dialogue_does_not_join_or_drop_leased_supervision(self):
        human, response = "human" * 30_000, "literal response" * 20_000
        coverage = datasets.DatasetCoverage()
        with patch.object(datasets, "_conversation_training_text", side_effect=AssertionError("no giant joined transcript")):
            record = datasets._structured_record({"messages": [{"role": "user", "content": human},
                {"role": "assistant", "content": response}]}, "native nested scalar", len(human) + len(response), coverage)
        self.addCleanup(record.text_payload.close)
        self.assertEqual(record.provenance["dialoguePairCount"], 1)
        self.assertNotIn("dialoguePairs", record.provenance)
        self.assertEqual(record.text_payload.dialogue.pair(0)[1].sha256, hashlib.sha256(response.encode()).hexdigest())

    def test_single_root_json_dialogue_and_speech_are_records_not_receipt_text(self):
        dialogue = self.root / "dialogue.json"
        dialogue.write_text(json.dumps({"id": 7, "metadata": {"note": "value"}, "messages": [
            {"role": "user", "content": "human" * 20_000}, {"role": "assistant", "content": "literal response"}]}))
        coverage = datasets.DatasetCoverage()
        stream = iter(datasets.iter_dataset_records(dialogue, coverage=coverage))
        record = next(stream)
        self.assertEqual(record.text_payload.dialogue.pair_count, 1)
        self.assertEqual(record.name, "dialogue.json#record-1")
        self.assertIsNone(next(stream, None))
        self.assertEqual(coverage.processed_records, 1)
        self.assertEqual(coverage.processed_bytes, dialogue.stat().st_size)
        audio = self.root / "clip.wav"
        audio.write_bytes(b"parser-only audio fixture; never decoded")
        speech = self.root / "speech.json"
        utterance = " exact whitespace " * 8_000
        speech.write_text(json.dumps({"text": utterance, "audioPath": "clip.wav",
            "audioSha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "format": "omni-speech-pair-1",
            "messages": [{"role": "user", "content": "irrelevant extra metadata"}]}))
        coverage = datasets.DatasetCoverage()
        stream = iter(datasets.iter_dataset_records(speech, coverage=coverage))
        record = next(stream)
        self.assertEqual(record.kind, "audio")
        self.assertEqual(record.provenance["speech_text"], utterance)
        self.assertEqual(record.local_path, str(audio.resolve()))
        self.assertIsNone(next(stream, None))
        self.assertEqual(coverage.processed_records, 1)
        self.assertEqual(coverage.processed_bytes, speech.stat().st_size)

    def seal(self, manifest, cursors, committed, steps):
        return seals.make_distributed_training_seal(manifest_sha256=manifest.content_sha256,
            topology_sha256="a" * 64, training_policy_sha256="b" * 64,
            record_count=len(manifest.entries), epochs=1, cursors=cursors,
            committed_record_stop=committed, global_steps=steps)

    def manifest(self):
        path = self.root / "source.jsonl"
        path.write_text(json.dumps({"text": "alpha"}) + "\n" + json.dumps({"text": "beta"}) + "\n")
        manifest = runtime.DatasetManifest.build(path)
        self.addCleanup(manifest.close)
        return manifest

    def test_native_seal_requires_exact_rank_owned_record_prefixes(self):
        manifest = self.manifest()
        cursors = [runtime.RankCursor(rank, 2, 0, 2, 1, 7, manifest.content_sha256) for rank in range(2)]
        value = self.seal(manifest, cursors, 2, 7)
        self.assertEqual(seals.validate_distributed_training_seal(value), value)
        bad = copy.deepcopy(value)
        bad["rankCursors"][1]["ownedRecordsCompleted"] = 0
        bad["rankCursorsSha256"] = spools.bounded_json_sha256(bad["rankCursors"])
        bad["contentSha256"] = spools.bounded_json_sha256({key: item for key, item in bad.items() if key != "contentSha256"})
        with self.assertRaisesRegex(ValueError, "exact ordinal prefix"):
            seals.validate_distributed_training_seal(bad)

    def test_external_run_writer_lease_is_exclusive_and_releases_without_unlink_race(self):
        store = runtime.DistributedRunStore(self.root / "run")
        first = store.acquire_run_lease()
        self.addCleanup(first.close)
        with self.assertRaises(OSError): store.acquire_run_lease()
        first.close()
        second = store.acquire_run_lease()
        second.close()
        self.assertTrue((store.path / ".run-owner.lock").is_file())

    def test_auto_wave_admission_has_no_fixed_parallel_ceiling_and_preserves_requested_work(self):
        plan_method, _ = source_function("distributed_training.py", "_distributed_window_plan", "DistributedGroundUpTrainer",
            {"math": __import__("math"), "DatasetResourcePause": spools.DatasetResourcePause})
        memory = {"optimizerAndGradientBytes": 0, "packedUpdateScratchBytes": 0,
            "activationBytesPerToken": 1024, "allocatorMarginBytes": 1024, "ramBudgetBytes": 32 * 1024 * 1024}
        captured = []
        admitted = {"batch": 32}
        def training_plan(**values):
            captured.append(values)
            return {"windowTokens": values["max_window_tokens"], "physicalBatchRecords": min(admitted["batch"], values["requested_batch_size"]),
                "pauseBeforeStep": False, "memory": dict(memory)}
        config = types.SimpleNamespace(max_seq_len=64, training_resource_mode="auto", training_ram_budget_bytes=0,
            training_accelerator_budget_bytes=0, training_scratch_budget_bytes=0, storage_bytes_per_second=0, disk_state_offload=True)
        brain = types.SimpleNamespace(config=config, _runtime_training_max_seq_len=64,
            _optimizer=types.SimpleNamespace(state={}), _optimizer_offloaded=False,
            decoder=types.SimpleNamespace(maximum_forward_tokens=lambda batch: 64),
            _training_resource_plan=lambda: {"memory": memory}, resource_policy=types.SimpleNamespace(training_plan=training_plan))
        trainer = types.SimpleNamespace(context=types.SimpleNamespace(rank=0, world_size=2),
            options=types.SimpleNamespace(micro_batch_records=0, gradient_accumulation=0))
        high = plan_method(trainer, brain, 0, 64)
        self.assertEqual(high["physicalBatchRecords"], 32)
        self.assertEqual(high["waveWindowTarget"], 32)
        self.assertEqual(high["accumulationSlots"], 1)
        self.assertEqual(captured[-1]["requested_batch_size"], 32)
        admitted["batch"] = 1
        low = plan_method(trainer, brain, 0, 64, 64)
        self.assertEqual(low["physicalBatchRecords"], 1)
        self.assertEqual(low["waveWindowTarget"], 32)
        self.assertEqual(low["accumulationSlots"], 32)
        trainer.options.micro_batch_records, trainer.options.gradient_accumulation = 4, 3
        explicit = plan_method(trainer, brain, 0, 8, 64)
        self.assertEqual(explicit["waveWindowTarget"], 12)
        self.assertEqual(captured[-1]["requested_batch_size"], 4)

    def test_input_wave_spills_with_bounded_microbatches_and_removes_transient_ids(self):
        reservations = []
        wave = buffers.PreparedWindowWave(physical_batch=7, ram_budget=100,
            directory=self.root, reserve=lambda **values: reservations.append(values))
        expected = []
        for ordinal in range(103):
            ids = torch.tensor([1, ordinal + 8, 2], dtype=torch.int64)
            expected.append(ids.tolist())
            wave.append(ids, torch.tensor([0.5]), torch.tensor([0.0]))
        self.assertEqual(wave.window_count, 103)
        self.assertEqual(wave.label_count, 206)
        path = wave.path
        actual = []
        for batch in wave.batches():
            self.assertLessEqual(len(batch), 7)
            actual.extend(ids.tolist() for ids, _cue, _noise in batch)
        self.assertEqual(actual, expected)
        self.assertTrue(any(values.get("disk_bytes", 0) for values in reservations))
        wave.close()
        self.assertFalse(path.exists())

    def test_production_loss_weights_real_labels_across_heterogeneous_padding(self):
        forward, _ = source_function("distributed_training.py", "forward", "DistributedBrainTrainingModule", {"F": F})
        decoder = types.SimpleNamespace(embedding=lambda ids: torch.zeros((*ids.shape, 2)),
            global_workspace=types.SimpleNamespace(summarize=lambda embedded, **_: torch.zeros((embedded.shape[0], 2))))
        class DecoderStub:
            embedding = staticmethod(decoder.embedding)
            global_workspace = decoder.global_workspace
            def __call__(self, ids, **_): return {"logits": torch.zeros((*ids.shape, 16))}
        module = types.SimpleNamespace(memory_bridge=lambda value: value, idea_adapter=lambda value: value,
            liquid=lambda idea, **_: (idea, {}), decoder=DecoderStub(),
            brain=types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_id=0), liquid_state=torch.zeros((1, 2))))
        ids = torch.tensor([[1, 8, 9, 2, 0], [1, 2, 0, 0, 0]])
        loss, measured = forward(module, ids, ids.ne(0), torch.zeros((2, 2)), torch.zeros((2, 2)),
            world_size=1, global_window_count=3, global_label_count=6, include_stability=False)
        self.assertAlmostEqual(loss.item(), 4 / 6 * __import__("math").log(16), places=6)
        self.assertEqual(measured[4:].tolist(), [2.0, 4.0])

    def test_immutable_native_publication_couples_progress_and_survives_failed_next_copy(self):
        manifest = self.manifest()
        store = runtime.DistributedRunStore(self.root / "run")
        store.initialize()
        native = self.root / "native"
        (native / "engine").mkdir(parents=True)
        blob = native / "engine" / "tiny-neural-fixture.bin"
        blob.write_bytes(b"first exact committed fixture")
        metadata = native / "engine" / "brain.json"
        cursors = [runtime.RankCursor(0, 1, 0, 1, 1, 1, manifest.content_sha256)]
        seal = self.seal(manifest, cursors, 1, 1)
        metadata.write_text(json.dumps({"distributed_training_seal": seal}))
        kwargs = dict(brain_json_path=metadata, manifest=manifest, cursors=cursors,
            epochs_requested=1, global_optimizer_steps=1, dynamic_high_water=1,
            strategy="single", telemetry={}, native_brain_path=native)
        first = store.publish_checkpoint(**kwargs)
        self.assertEqual((store.published_native_path(first) / "engine" / blob.name).read_bytes(), blob.read_bytes())
        blob.write_bytes(b"uncommitted next fixture")
        next_cursors = [runtime.RankCursor(0, 1, 0, 2, 2, 2, manifest.content_sha256)]
        metadata.write_text(json.dumps({"distributed_training_seal": self.seal(manifest, next_cursors, 2, 2)}))
        with patch.object(runtime.shutil, "copytree", side_effect=OSError("injected snapshot copy failure")):
            with self.assertRaisesRegex(OSError, "copy failure"):
                store.publish_checkpoint(**{**kwargs, "cursors": next_cursors, "global_optimizer_steps": 2, "dynamic_high_water": 2})
        loaded = store.load_active_checkpoint(manifest_sha256=manifest.content_sha256, world_size=1)
        self.assertEqual(loaded, first)
        self.assertEqual((store.published_native_path(loaded) / "engine" / blob.name).read_bytes(), b"first exact committed fixture")

    def test_external_cursor_rehash_cannot_change_the_independent_native_seal(self):
        manifest = self.manifest()
        store = runtime.DistributedRunStore(self.root / "run")
        store.initialize()
        native = self.root / "native"
        (native / "engine").mkdir(parents=True)
        cursors = [runtime.RankCursor(0, 1, 0, 1, 1, 1, manifest.content_sha256)]
        metadata = native / "engine" / "brain.json"
        metadata.write_text(json.dumps({"distributed_training_seal": self.seal(manifest, cursors, 1, 1)}))
        value = store.publish_checkpoint(brain_json_path=metadata, manifest=manifest, cursors=cursors,
            epochs_requested=1, global_optimizer_steps=1, dynamic_high_water=1,
            strategy="single", telemetry={}, native_brain_path=native)
        directory = store.checkpoints_path / value["contentSha256"]
        changed = copy.deepcopy(value)
        changed["rankCursors"][0].update(nextGlobalOrdinal=2, ownedRecordsCompleted=2)
        changed["dynamicHighWater"] = 2
        changed["contentSha256"] = spools.bounded_json_sha256({key: item for key, item in changed.items() if key != "contentSha256"})
        (directory / "checkpoint.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "committed native seal"):
            store._read_checkpoint_directory(directory)

    def test_public_save_restores_staged_seal_only_before_commit(self):
        manifest = self.manifest()
        initial = self.seal(manifest, [runtime.RankCursor(0, 1, 0, 0, 0, 0, manifest.content_sha256)], 0, 0)
        next_seal = self.seal(manifest, [runtime.RankCursor(0, 1, 0, 1, 1, 1, manifest.content_sha256)], 1, 1)
        stage, _ = source_function("brain.py", "stage_distributed_training_seal", "AdaptiveBrain", {"validate_distributed_training_seal": seals.validate_distributed_training_seal})
        save, _ = source_function("brain.py", "save", "AdaptiveBrain")
        for post_commit in (False, True):
            brain = types.SimpleNamespace(distributed_training_seal=initial,
                memory=types.SimpleNamespace(persistence_manifest={"old": "substrate"}), mutable_state_manifest={"old": "mutable"})
            stage(brain, next_seal)
            def write(**options):
                brain.memory.persistence_manifest = {"new": "substrate"}
                brain.mutable_state_manifest = {"new": "mutable"}
                if post_commit: options["commit_callback"]()
                raise OSError("injected before/after pointer publication")
            brain._save_checkpoint_impl = write
            with self.assertRaises(OSError): save(brain)
            self.assertEqual(brain.distributed_training_seal, next_seal if post_commit else initial)
            self.assertEqual(brain.memory.persistence_manifest, {"new": "substrate"} if post_commit else {"old": "substrate"})
            self.assertIsNone(brain._distributed_seal_stage_previous)

    def test_canonical_fast_replay_receives_every_literal_section_not_synthetic_labels(self):
        function, _ = source_function("distributed_training.py", "apply_authoritative_source_updates")
        fake_vsa = types.ModuleType(PACKAGE + ".vsa")
        fake_vsa._now = lambda: 0
        record, entry = self.giant("literal é 😀 data " * 20_000)
        seen = []
        brain = types.SimpleNamespace(learn_experience=lambda text, **options: seen.append((text, options)))
        with patch.dict(sys.modules, {PACKAGE + ".vsa": fake_vsa}):
            committed, reports = function(brain, [(0, entry, record)])
        self.assertEqual(committed, 1)
        self.assertEqual(reports, [])
        self.assertEqual("".join(text for text, _ in seen), "literal é 😀 data " * 20_000)
        self.assertTrue(all(options["steps"] == 0 for _, options in seen))
        self.assertTrue(all(len(text) <= spools.LEARNING_WINDOW_CHARS for text, _ in seen))

    def test_leased_supervised_production_method_uses_masked_decoder_targets(self):
        class FakePackedOptimizer:
            param_groups = []
            zero_grad = MagicMock()
            step = MagicMock()
        class FakeLoss:
            @classmethod
            def __torch_function__(cls, function, types_, args=(), kwargs=None):
                if function is torch.isfinite: return torch.tensor(True)
                return NotImplemented
            def backward(self): self.backwards += 1
            def detach(self): return self
            def item(self): return 0.125
            backwards = 0
        loss, observed = FakeLoss(), []
        method, _ = source_function("brain.py", "_optimize_leased_dialogue_window", "AdaptiveBrain", {"PackedOnlyOptimizer": FakePackedOptimizer})
        def decoder(ids, **options):
            observed.append((ids.tolist(), options["labels"].tolist(), options["memory_bias"]))
            return {"loss": loss}
        brain = types.SimpleNamespace(_optimizer=FakePackedOptimizer(), device=torch.device("cpu"),
            tokenizer=types.SimpleNamespace(pad_id=0), counters={"training_steps": 9},
            _ensure_optimizer_resident=lambda: None, _idea_model_vector=lambda cue: cue,
            idea_adapter=lambda cue: cue, decoder=decoder, _accumulate_slow_importance=lambda _: None,
            _commit_slow_anchors=lambda **_: None, _maintain_neural_state_resources=lambda: None)
        result = method(brain, [4, 73, 2], "all-human-neural-cue-stub")
        self.assertEqual(observed, [([[4, 73, 2]], [[0, 73, 2]], "all-human-neural-cue-stub")])
        self.assertEqual(result["target_tokens"], 2)
        self.assertEqual(brain.counters["training_steps"], 10)
        self.assertEqual(loss.backwards, 1)
        brain._optimizer.step.assert_called_once()

    def test_production_driver_stub_cancel_resume_never_replays_committed_literal_windows(self):
        self._driver_fixture(["literal é😀 " * 10_000], cancel_step=32, group_size=16,
            expected_records_at_pause=0, expected_windows_at_pause=32 * 16)

    def test_requested_record_group_really_batches_short_records_before_advancing(self):
        self._driver_fixture(["a", "bb", "ccc", "dddd", "last"], cancel_step=1, group_size=4,
            expected_records_at_pause=4, expected_windows_at_pause=None)

    def _driver_fixture(self, texts, *, cancel_step, group_size, expected_records_at_pause, expected_windows_at_pause):
        # Only scheduling is exercised. The learner/checkpoint/promotion are
        # explicit stubs, not neural construction or a neural acceptance run.
        path = self.root / "driver.jsonl"
        path.write_text("\n".join(json.dumps({"text": text}) for text in texts))
        manifest = runtime.DatasetManifest.build(path)
        self.addCleanup(manifest.close)
        class PackedMarker: pass
        class ModuleStub:
            def to(self, _device): return self
            def modules(self): return (PackedMarker(),)
        context = types.SimpleNamespace(rank=0, world_size=1, distributed=False, backend="single",
            device=torch.device("cpu"), is_rank_zero=True, status=lambda: {"fixture": True})
        namespace = {"RecordWindowStream": waves.RecordWindowStream, "RankCursor": runtime.RankCursor,
            "contextlib": contextlib,
            "PreparedWindowWave": buffers.PreparedWindowWave,
            "PACKED_AUTHORITATIVE_PROJECTION_TYPES": (PackedMarker,),
            "DistributedBrainTrainingModule": lambda _: ModuleStub(),
            "_collect_objects": lambda _context, value: [value],
            "_collective_cancelled": lambda _context, value: value,
            "_resolve_strategy": lambda *_: "single", "_wrap_module": lambda value, **_: value,
            "_new_training_optimizer": lambda *_: None, "_make_grad_scaler": lambda _: None,
            "CapabilityRehearsalPolicy": lambda **_: {}, "_due_distributed_rehearsal_phase": lambda *_, **__: None,
            "MonotonicManifestReplay": lambda *_: types.SimpleNamespace(resource_admission=None, close=lambda: None),
            "DatasetResourcePause": spools.DatasetResourcePause, "_broadcast_rank_zero_error": lambda *_: None,
            "_failure_payload": lambda error: {"message": str(error)},
            "_raise_phase_failures": lambda values, _: None if not any(values) else (_ for _ in ()).throw(RuntimeError(str(values))),
        }
        run, _ = source_function("distributed_training.py", "run", "DistributedGroundUpTrainer", namespace)
        saved, labels, statuses = {}, [], []
        brain = types.SimpleNamespace(tokenizer=self.tokenizer, device=context.device,
            config=types.SimpleNamespace(max_seq_len=64), close=lambda: None,
            _training_resource_plan=lambda: {"windowTokens": 64, "pauseBeforeStep": False},
            resource_policy=types.SimpleNamespace(status=lambda **_: {"memoryPressure": False, "diskPressure": False}))
        trainer = types.SimpleNamespace(context=context, store=types.SimpleNamespace(
            path=self.root / "run",
            active_path=types.SimpleNamespace(is_file=lambda: bool(saved)),
            cancel_path=self.root / "cancel", cancel_requested=lambda: trainer.cancel and trainer.steps >= cancel_step,
            write_status=lambda **values: statuses.append(values), recent_failures=lambda: [], record_failure=lambda *_: None),
            options=types.SimpleNamespace(epochs=1, amp="off", learning_rate=None, capability_rehearsal_waves=128,
                global_batch_records=group_size, checkpoint_steps=16, micro_batch_records=0, gradient_accumulation=0), brain_id="primitive-driver", output_path=self.root / "output",
            _signal_cancelled=False, _install_signal_handlers=lambda: None, _restore_signal_handlers=lambda: None,
            _acquire_run_leases=lambda: (),
            _prepare_manifest=lambda: manifest, _bind_actual_native_device=lambda _: None,
            _refresh_native_replica=lambda current, _: current, _failure_injection_matches=lambda _: False,
            _promote_output=lambda **_: {"fixtureOnly": True}, cancel=True, steps=0)
        brain.config.disk_state_offload = True
        trainer._distributed_window_plan = lambda *_: {"windowTokens": 64, "physicalBatchRecords": 8,
            "waveWindowTarget": 16, "inputRamBudgetBytes": 100_000}
        trainer._rank_records_for_wave = lambda _brain, _manifest, _epoch, literal: ([(entry, [(
            torch.tensor(item.ids, dtype=torch.int64), torch.tensor([0.5]), torch.tensor([0.0]))]) for entry, item in literal], [])
        schedule = types.SimpleNamespace(to_dict=lambda: {"fixtureOnly": True})
        def prepare(_):
            cursors = [runtime.RankCursor.from_dict(value) for value in saved["cursors"]] if saved else runtime.initial_rank_cursors(world_size=1, manifest_sha256=manifest.content_sha256)
            return brain, cursors, saved.get("committed", 0), saved.get("steps", 0), bool(saved), schedule, {}
        def learn(**values):
            for batch in values["local_records"].batches():
                for ids, _cue, _noise in batch:
                    labels.extend(ids.tolist()[1:])
            trainer.steps += 1
            return {"loss": 0.0}, [], bool(values["local_records"])
        def checkpoint(**values):
            saved.update(cursors=[cursor.to_dict() for cursor in values["cursors"]], steps=values["global_steps"],
                committed=min(cursor.epoch * len(manifest.entries) + cursor.next_global_ordinal for cursor in values["cursors"]))
            return saved["committed"], schedule, {}
        trainer._prepare_brains, trainer._train_wave, trainer._checkpoint = prepare, learn, checkpoint
        first = run(trainer)
        self.assertEqual(first["state"], "cancelled")
        self.assertEqual(saved["cursors"][0]["ownedRecordsCompleted"], expected_records_at_pause)
        if expected_windows_at_pause is not None:
            self.assertEqual(saved["cursors"][0]["recordWindow"]["completedWindows"], expected_windows_at_pause)
            # Only physical batching changes after a durable boundary. The
            # next literal byte/target cursor remains exactly the same.
            trainer._distributed_window_plan = lambda *_: {"windowTokens": 64, "physicalBatchRecords": 2,
                "waveWindowTarget": 16, "inputRamBudgetBytes": 100_000}
        trainer.cancel = False
        final = run(trainer)
        self.assertEqual(final["state"], "complete")
        self.assertEqual(labels, [value for text in texts for value in [*(byte + 8 for byte in text.strip().encode()), 2]])
        self.assertEqual(saved["cursors"][0]["epoch"], 1)
        self.assertEqual(saved["cursors"][0]["ownedRecordsCompleted"], len(texts))
        self.assertTrue(all(value.get("progress", 0) < 1 for value in statuses if value["state"] != "complete"))


if __name__ == "__main__":
    unittest.main()
