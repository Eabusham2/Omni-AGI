"""Execute extracted production routing with stubs; never import/build a brain."""

import ast
import contextlib
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch


ENGINE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ingestion_contract_fixture", ENGINE / "omni_core" / "ingestion_contract.py"
)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def production_method(file_name, class_name, method_name, globals_=None):
    """Compile only a method's AST with deferred annotations and stub globals."""
    tree = ast.parse((ENGINE / file_name).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    namespace = {
        "hashlib": hashlib, "Path": Path, "re": re,
        "Mapping": Mapping,
        "parser_admission": lambda _: contextlib.nullcontext(),
        "DatasetResourcePause": StubResourcePause,
        "MediaDecodeError": contract.MediaDecodeError,
        "NoAudioTrackError": contract.NoAudioTrackError,
        "ingestion_transaction_id": contract.ingestion_transaction_id,
        **(globals_ or {}),
    }
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), file_name, "exec"), namespace)
    return namespace[method_name], method, namespace


class StubFault(RuntimeError):
    def __init__(self, code, message, data=None):
        super().__init__(message)
        self.code = code
        self.data = data


class StubResourcePause(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status or {}


class IngestionIdentityTests(unittest.TestCase):
    def test_explicit_manifest_key_is_distinct_and_legacy_identity_is_preserved(self):
        fields = dict(content_hash="c" * 64, epoch=2, policy="pretrain", resolved_kind="text", source_bytes=5)
        self.assertEqual(contract.ingestion_transaction_id(**fields, transaction_key="a" * 64), "a" * 64)
        self.assertEqual(contract.ingestion_transaction_id(**fields, transaction_key="b" * 64), "b" * 64)
        legacy = hashlib.sha256(("%s\0%d\0%s\0%s\0%d" % (
            fields["content_hash"], 2, "pretrain", "text", 5
        )).encode("utf-8")).hexdigest()
        self.assertEqual(contract.ingestion_transaction_id(**fields), legacy)
        for invalid in (None, 0, [], "a" * 63, "A" * 64):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                contract.ingestion_transaction_id(**fields, transaction_key=invalid)

    def test_native_same_key_is_idempotent_but_new_manifest_reaches_learning(self):
        # Stop at resource planning before any tensor/model method can run.
        ingest, _, _ = production_method("omni_core/brain.py", "AdaptiveBrain", "ingest", {
            "DatasetCoverage": MagicMock,
            "dataset_format": lambda *_: "text",
            "dataset_record_count_hint": lambda *_: 1,
        })
        with tempfile.TemporaryDirectory() as directory:
            path = (Path(directory) / "source.txt").resolve()
            path.write_text("valid input", encoding="utf-8")
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            source = {"id": "source-a", "content_hash": content_hash}
            receipt = {
                "transactionId": "a" * 64,
                "contentHash": content_hash, "epoch": 0, "policy": "pretrain",
                "sourceIdentity": hashlib.sha256(("file\0" + str(path)).encode("utf-8")).hexdigest(),
                "sourceNameHash": hashlib.sha256(path.name.encode("utf-8")).hexdigest(),
                "sourceId": "source-a", "coverage": {"complete": True},
                "parameterChecksumAfter": "d" * 64,
            }
            planning = MagicMock(side_effect=RuntimeError("learning boundary reached without a model"))
            brain = types.SimpleNamespace(
                brain_id="fixture-brain", ingestion_checkpoints={},
                completed_ingestions=[receipt], training_sources=[source],
                _ingestion_v3_manifest_hashes=lambda **_: ("e" * 64, "f" * 64),
                _streaming_neural_storage_plan=planning, metrics=lambda: {},
                _validated_completed_ingestions=lambda value: value,
            )
            repeated = ingest(brain, path=str(path), policy="pretrain", allow_replay=True, transaction_key="a" * 64)
            self.assertTrue(repeated["idempotentCompletion"])
            self.assertEqual(repeated["transactionKey"], "a" * 64)
            planning.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "learning boundary reached"):
                ingest(brain, path=str(path), policy="pretrain", allow_replay=True, transaction_key="b" * 64)
            planning.assert_called_once()

    def worker_fixture(self, brain):
        ingest, _, _ = production_method("worker.py", "Worker", "ingest", {
            "RpcFault": StubFault,
            "NeuralStateResourcePause": StubResourcePause,
            "is_allocator_oom_error": lambda error: "out of memory" in str(error),
            "AdaptiveBrain": types.SimpleNamespace(load=MagicMock()),
        })
        worker = types.SimpleNamespace(
            _job=MagicMock(return_value=(brain, "fixture-job", MagicMock())),
            _job_complete=MagicMock(), brains={}, notify=MagicMock(),
        )
        return ingest, worker

    def test_worker_forwards_key_and_rejects_invalid_or_disagreeing_keys_before_load(self):
        brain = types.SimpleNamespace(ingest=MagicMock(return_value={"okay": True}))
        ingest, worker = self.worker_fixture(brain)
        params = {"transactionKey": "a" * 64, "idempotencyKey": "a" * 64, "epoch": 3, "allowReplay": True}
        self.assertEqual(ingest(worker, params, "request-a"), {"okay": True})
        self.assertEqual(brain.ingest.call_args.kwargs["transaction_key"], "a" * 64)
        self.assertEqual(brain.ingest.call_args.kwargs["epoch"], 3)
        self.assertTrue(brain.ingest.call_args.kwargs["allow_replay"])
        for params in ({"transactionKey": "short"}, {"transactionKey": None}, {"transactionKey": 0},
                       {"transactionKey": "a" * 64, "idempotencyKey": "b" * 64}):
            worker._job.reset_mock()
            with self.subTest(params=params), self.assertRaises(StubFault):
                ingest(worker, params, "invalid")
            worker._job.assert_not_called()

    def test_worker_restores_counters_and_committed_state_after_partial_media_failure(self):
        restored = types.SimpleNamespace(ingestion_checkpoints={}, events=MagicMock(), counters={"training_steps": 7})
        loader = MagicMock(return_value=restored)
        ingest, _, _ = production_method("worker.py", "Worker", "ingest", {
            "RpcFault": StubFault, "NeuralStateResourcePause": StubResourcePause,
            "is_allocator_oom_error": lambda _: False,
            "AdaptiveBrain": types.SimpleNamespace(load=loader),
        })
        brain = types.SimpleNamespace(brain_id="fixture-brain", storage_path=Path("fixture-storage"),
                                     counters={"training_steps": 7}, close=MagicMock())
        def failing_ingest(**_):
            brain.counters["training_steps"] += 1
            raise RuntimeError("non-finite modality training loss")
        brain.ingest = failing_ingest
        worker = types.SimpleNamespace(_job=lambda *_: (brain, "fixture-job", MagicMock()),
                                       brains={brain.brain_id: brain}, _job_complete=MagicMock(), notify=MagicMock())
        with self.assertRaisesRegex(RuntimeError, "non-finite modality"):
            ingest(worker, {"transactionKey": "a" * 64}, "failing-request")
        self.assertIs(worker.brains[brain.brain_id], restored)
        self.assertEqual(restored.counters["training_steps"], 7)
        brain.close.assert_called_once()
        worker._job_complete.assert_not_called()
        loader.assert_called_once_with(Path("fixture-storage"), expected_brain_id=brain.brain_id)


class MediaFailureRoutingTests(unittest.TestCase):
    def media_ingest_branch(self):
        _, method, namespace = production_method("omni_core/brain.py", "AdaptiveBrain", "ingest")
        branch = next(node for node in ast.walk(method) if isinstance(node, ast.Try)
                      and any(isinstance(handler.type, ast.Name) and handler.type.id == "MediaDecodeError"
                              for handler in node.handlers))
        function = ast.FunctionDef(
            name="media_record", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.parse("record_path='fixture.mp3'\nrecord_kind='audio'\nrecord_name='fixture'\npolicy='encode'\nprogress=None\nrecord=None\nmedia_training_steps_before=self.counters['training_steps']\n").body
                 + [branch, ast.Return(value=ast.Name(id="trained_media", ctx=ast.Load()))],
            decorator_list=[],
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "media-ingest-branch", "exec"), namespace)
        return namespace["media_record"]

    def test_only_unmutated_invalid_media_is_rejected_by_ingest(self):
        branch = self.media_ingest_branch()
        brain = types.SimpleNamespace(counters={"training_steps": 7},
                                      _empty_media_coverage=lambda _: {"complete": False})
        brain._train_media = MagicMock(side_effect=contract.MediaDecodeError("invalid source"))
        self.assertFalse(branch(brain)["trained"])
        for failure in (RuntimeError("out of memory"), RuntimeError("non-finite modality loss"),
                        ValueError("learner shape failure"), OSError("runtime unavailable")):
            brain._train_media = MagicMock(side_effect=failure)
            with self.subTest(failure=failure), self.assertRaises(type(failure)):
                branch(brain)
        def partial_stream(**_):
            brain.counters["training_steps"] += 1
            raise contract.MediaDecodeError("truncated suffix")
        brain._train_media = lambda *_args, **kwargs: partial_stream(**kwargs)
        with self.assertRaisesRegex(RuntimeError, "after neural mutation"):
            branch(brain)

    def embedded_branch(self):
        _, method, namespace = production_method("omni_core/brain.py", "AdaptiveBrain", "_train_media")
        # Execute the production embedded-audio branch only; no optimizer,
        # tensor, decoder, constructor or neural training is executed.
        branch = next(node for node in method.body if isinstance(node, ast.If)
                      and "_include_embedded_audio" in ast.unparse(node.test))
        function = ast.FunctionDef(
            name="embedded", args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self")], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.parse("kind='video'\n_include_embedded_audio=True\npath='fixture.mp4'\nsource_name='fixture'\nsteps_per_window=2\nprogress=None\nfingerprint_base='hash'\nembedded_audio=None\nembedded_steps=0\nembedded_loss=0.0\nmedia_coverage={}\n").body
                 + [branch, ast.Return(value=ast.Name(id="embedded_audio", ctx=ast.Load()))],
            decorator_list=[],
        )
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), "embedded-media-branch", "exec"), namespace)
        return namespace["embedded"]

    def test_only_verified_absent_audio_is_treated_as_silence(self):
        branch = self.embedded_branch()
        brain = types.SimpleNamespace(config=types.SimpleNamespace(audio_enabled=True),
                                      _empty_media_coverage=lambda _: {"complete": False})
        brain._train_media = MagicMock(side_effect=contract.NoAudioTrackError("no audio stream"))
        self.assertEqual(branch(brain)["detected"], False)
        for failure in (RuntimeError("allocator out of memory"), RuntimeError("non-finite loss"),
                        contract.MediaDecodeError("damaged audio"), OSError("decoder unavailable"),
                        StubResourcePause("reserve reached")):
            brain._train_media = MagicMock(side_effect=failure)
            with self.subTest(failure=failure), self.assertRaises(type(failure)):
                branch(brain)
        brain._train_media = MagicMock(return_value={"trained": False, "coverage": {"complete": False}})
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            branch(brain)

    def test_audio_track_probe_requires_explicit_missing_stream_diagnostic(self):
        run = MagicMock()
        probe, _, _ = production_method("omni_core/brain.py", "AdaptiveBrain", "_assert_embedded_audio_track", {
            "subprocess": types.SimpleNamespace(run=run, DEVNULL=-3, PIPE=-1),
        })
        decoder = types.SimpleNamespace(get_ffmpeg_exe=lambda: "fixture-ffmpeg")
        brain = types.SimpleNamespace(_video_runtime_executable=lambda *_args, **_kwargs: contextlib.nullcontext("fixture-ffmpeg"))
        with patch.dict(sys.modules, {"imageio_ffmpeg": decoder}):
            run.return_value = types.SimpleNamespace(returncode=0, stderr=b"")
            probe(brain, "fixture.mp4")
            command = run.call_args.args[0]
            self.assertIn("0:a:0", command)
            self.assertIn("0", command)
            run.return_value = types.SimpleNamespace(returncode=1, stderr=b"Stream map '0:a:0' matches no streams.")
            with self.assertRaises(contract.NoAudioTrackError):
                probe(brain, "fixture.mp4")
            run.return_value = types.SimpleNamespace(returncode=1, stderr=b"invalid input, allocation failed")
            with self.assertRaisesRegex(RuntimeError, "inspection failed"):
                probe(brain, "fixture.mp4")
            run.side_effect = subprocess.TimeoutExpired("fixture-ffmpeg", 30)
            with self.assertRaises(subprocess.TimeoutExpired):
                probe(brain, "fixture.mp4")

    def test_audio_decoder_does_not_replay_a_prefix_or_swallow_allocator_failure(self):
        popen = MagicMock(side_effect=AssertionError("fallback must not replay a consumed prefix"))
        windows, _, _ = production_method("omni_core/brain.py", "AdaptiveBrain", "_iter_audio_windows", {
            "torch": types.SimpleNamespace(from_numpy=lambda _: types.SimpleNamespace(float=lambda: types.SimpleNamespace(mean=lambda **_: "samples"))),
            "subprocess": types.SimpleNamespace(Popen=popen),
            "is_allocator_oom_error": lambda error: "out of memory" in str(error),
        })
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.read.side_effect = [types.SimpleNamespace(shape=(4, 1)), RuntimeError("late decoder failure")]
        soundfile = types.SimpleNamespace(SoundFile=lambda *_args, **_kwargs: reader)
        brain = types.SimpleNamespace(config=types.SimpleNamespace(audio_samples=4), _bounded_audio_target=lambda values: (values, 4))
        with patch.dict(sys.modules, {"soundfile": soundfile}):
            stream = windows(brain, "fixture.mp3")
            self.assertEqual(next(stream), ("samples", 4))
            with self.assertRaises(contract.MediaDecodeError):
                next(stream)
            reader.read.side_effect = RuntimeError("allocator out of memory")
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                next(windows(brain, "fixture.mp3"))
            reader.read.side_effect = [types.SimpleNamespace(shape=(4, 1))]
            unavailable = RuntimeError("device became unavailable")
            brain._bounded_audio_target = MagicMock(side_effect=unavailable)
            with self.assertRaises(RuntimeError) as raised:
                next(windows(brain, "fixture.mp3"))
            self.assertIs(raised.exception, unavailable)
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
