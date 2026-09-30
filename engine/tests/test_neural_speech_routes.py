"""Constructor-free speech transport, exact pairing and primitive loss fixtures.

No model/brain constructor, training/backward, live media decoder or playback.
"""
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import Worker, RpcFault, ModalityGenerationCancelled
from omni_core.datasets import DatasetCoverage, _structured_record, _iter_jsonl, _iter_json
from omni_core.speech_training import paired_speech_waveform_loss, StreamingSpeechResampler
from omni_core.text_spool import DatasetResourcePause, parser_admission
from omni_core.brain import AdaptiveBrain


class NativeSpeechRouteTests(unittest.TestCase):
    def worker(self, pairs=0):
        worker = Worker.__new__(Worker)
        worker.generate_modality = Mock(return_value={"brainId": "mind", "modality": "audio",
            "mimeType": "audio/wav", "speechPairedExamples": pairs, "randomlyInitialized": False})
        return worker

    def test_exact_text_goes_to_same_audio_region_not_external_synthesis(self):
        worker = self.worker()
        text = " literal reply \n"
        result = worker.generate_neural_speech({"brainId": "mind", "jobId": "job", "speechRequestId": "speech", "text": text, "rate": 1.35}, "rpc")
        params = worker.generate_modality.call_args.args[0]
        self.assertEqual(params["prompt"], text)
        self.assertEqual(params["modality"], "audio")
        self.assertEqual(params["inputPath"], "")
        self.assertEqual(params["settings"]["durationMs"], 2 * 1000 / 3)
        self.assertEqual(result["speech"]["textSha256"], hashlib.sha256(text.encode()).hexdigest())
        self.assertFalse(result["speech"]["externalModelUsed"])
        self.assertFalse(result["speech"]["intelligibilityVerified"])
        self.assertEqual(result["speech"]["trainingState"], "needs-speech-training")

    def test_paired_training_count_is_not_intelligibility_proof(self):
        result = self.worker(4).generate_neural_speech({"speechRequestId": "speech", "text": "actual"}, "rpc")
        self.assertEqual(result["speech"]["trainingState"], "speech-quality-unverified")
        self.assertEqual(result["speech"]["pairedExamples"], 4)
        self.assertFalse(result["speech"]["intelligibilityVerified"])

    def test_no_old_length_cap_or_hidden_truncation(self):
        worker = self.worker()
        text = "exact " * 20_000
        with patch("worker.require_parser_resources") as admit:
            worker.generate_neural_speech({"speechRequestId": "speech", "text": text}, "rpc")
        self.assertEqual(worker.generate_modality.call_args.args[0]["prompt"], text)
        self.assertGreater(admit.call_args.kwargs["ram_bytes"], len(text))

    def test_measured_allocation_pause_does_not_dispatch_waveform(self):
        worker = self.worker()
        with patch("worker.require_parser_resources", side_effect=DatasetResourcePause("physical RAM limit")):
            with self.assertRaises(RpcFault) as raised:
                worker.generate_neural_speech({"speechRequestId": "speech", "text": "actual"}, "rpc")
        self.assertEqual(raised.exception.code, -32020)
        worker.generate_modality.assert_not_called()

    def test_native_speech_cancellation_is_selected_cooperative_not_chat_control(self):
        worker = Worker.__new__(Worker)
        worker._active_request_lock = threading.RLock()
        worker._active_request = ("generate_neural_speech", "owned-rpc")
        worker._cooperative_cancel = threading.Event()
        self.assertTrue(worker.request_cooperative_cancel())
        self.assertTrue(worker._cooperative_cancel.is_set())
        worker._active_request = ("health", "unrelated")
        worker._cooperative_cancel.clear()
        self.assertFalse(worker.request_cooperative_cancel())
        self.assertFalse(worker._cooperative_cancel.is_set())

    def test_actual_generation_handler_passes_selected_cancel_check_and_never_publishes_after_cancel(self):
        worker = Worker.__new__(Worker)
        worker._cooperative_cancel = threading.Event()
        worker._cooperative_cancel.set()
        worker.cancelled_jobs = set()
        brain = SimpleNamespace(brain_id="mind", modality_training={"audio_speech_pairs": 0})
        def actual_generation(**kwargs):
            self.assertTrue(kwargs["cancel_check"]())
            raise ModalityGenerationCancelled("owned speech cancelled")
        brain.generate_modality = actual_generation
        worker._job = Mock(return_value=(brain, "job", Mock()))
        worker._claim_inline_generation = Mock(return_value=None)
        worker._job_complete = Mock()
        with self.assertRaises(RpcFault) as raised:
            worker.generate_neural_speech({"speechRequestId": "speech", "text": "actual", "jobId": "job"}, "rpc")
        self.assertEqual(raised.exception.code, -32800)
        worker._job_complete.assert_not_called()


class SpeechPairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="omni-speech-pair-")
        self.root = Path(self.temp.name)
        self.audio = self.root / "clip.wav"
        self.audio.write_bytes(b"hash-only WAV fixture; no live decoding")
        self.audio_hash = hashlib.sha256(self.audio.read_bytes()).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def row(self, text=" literal recorded words \n"):
        return {"format": "omni-speech-pair-1", "audioPath": "clip.wav", "audioSha256": self.audio_hash, "text": text}

    def test_pair_binds_exact_transcript_and_clip_as_audio_not_generic_text(self):
        coverage = DatasetCoverage()
        row = self.row()
        record = _structured_record(row, "pair", 100, coverage, source_directory=self.root)
        self.assertEqual(record.kind, "audio")
        self.assertEqual(record.provenance["speech_text"], row["text"])
        self.assertEqual(record.local_path, str(self.audio.resolve()))
        self.assertEqual(coverage.modality_counts, {"audio": 1})
        changed = _structured_record(self.row("different words"), "pair", 100, DatasetCoverage(), source_directory=self.root)
        self.assertNotEqual(record.content_sha256, changed.content_sha256)

    def test_clip_checksum_and_source_directory_are_enforced_not_fabricated(self):
        row = self.row()
        row["audioSha256"] = "0" * 64
        coverage = DatasetCoverage()
        self.assertIsNone(_structured_record(row, "pair", 100, coverage, source_directory=self.root))
        self.assertEqual(coverage.rejected_records, 1)
        row = self.row()
        row["audioPath"] = "../outside.wav"
        self.assertIsNone(_structured_record(row, "pair", 100, DatasetCoverage(), source_directory=self.root))

    def test_actual_json_array_and_jsonl_readers_forward_audio_pair_ownership(self):
        for extension, content in (("jsonl", json.dumps(self.row()) + "\n"), ("json", json.dumps([self.row()]))):
            path = self.root / ("pairs." + extension)
            path.write_text(content)
            iterator = _iter_jsonl(path, DatasetCoverage()) if extension == "jsonl" else _iter_json(path, DatasetCoverage())
            try:
                record = next(iterator)
                self.assertEqual(record.kind, "audio")
                self.assertEqual(record.provenance["speech_text"], self.row()["text"])
            finally:
                iterator.close()

    def test_spooled_literal_utterance_is_preserved_in_full_with_resource_admission(self):
        text = "  " + "paired utterance \n" * 5_000 + "  "
        source = self.root / "giant-pair.jsonl"
        source.write_text(json.dumps(self.row(text)) + "\n")
        admissions = []
        with parser_admission(lambda stage, ram, disk: admissions.append((stage, ram, disk))):
            iterator = _iter_jsonl(source, DatasetCoverage())
            try:
                record = next(iterator)
                self.assertEqual(record.kind, "audio")
                self.assertEqual(record.provenance["speech_text"], text)
                self.assertEqual(record.provenance["speech_text_sha256"], hashlib.sha256(text.encode()).hexdigest())
            finally:
                iterator.close()
        self.assertTrue(any(stage == "speech utterance text conditioning" and ram > len(text) for stage, ram, _disk in admissions))

    def test_physical_speech_conditioning_pause_is_not_a_rejected_or_shortened_record(self):
        def pause(stage, _ram, _disk):
            if stage == "speech utterance text conditioning":
                raise DatasetResourcePause("physical speech allocation denied")
        coverage = DatasetCoverage()
        with parser_admission(pause):
            with self.assertRaises(DatasetResourcePause):
                _structured_record(self.row(), "pair", 100, coverage, source_directory=self.root)
        self.assertEqual(coverage.rejected_records, 0)

    def test_spooled_pair_without_literal_text_never_trains_another_column_as_speech(self):
        row = self.row()
        del row["text"]
        row["response"] = "unrelated content column " * 5000
        source = self.root / "invalid-pair.jsonl"
        source.write_text(json.dumps(row) + "\n")
        coverage = DatasetCoverage()
        self.assertEqual(list(_iter_jsonl(source, coverage)), [])
        self.assertEqual(coverage.rejected_records, 1)

    def test_direct_generator_objective_uses_actual_tail_and_keeps_gradient_graph(self):
        idea = torch.tensor([[0.25, 0.75]])
        predicted = torch.tensor([[0.0, 0.5, 100.0, 100.0]], requires_grad=True)
        codec = SimpleNamespace(generate=Mock(return_value=predicted))
        target = torch.tensor([[[1.0, 0.5, 0.0, 0.0]]])
        loss = paired_speech_waveform_loss(codec, idea, target, 2, seed=3)
        self.assertEqual(loss.item(), 0.5)
        self.assertTrue(loss.requires_grad) # no backward or parameter training is run
        self.assertIs(codec.generate.call_args.args[0], idea)
        self.assertIsInstance(codec.generate.call_args.args[1], torch.Generator)

    def test_streaming_resampling_preserves_phase_across_windows_and_true_duration(self):
        source = torch.arange(7, dtype=torch.float32)
        def collect(parts):
            resampler = StreamingSpeechResampler(3, 4, 2)
            output = []
            for part in parts:
                output.extend(resampler.push(part))
            output.extend(resampler.push(torch.empty(0), final=True))
            self.assertEqual(resampler.source_samples, 7)
            self.assertEqual(resampler.output_samples, 10)
            self.assertTrue(all(piece.numel() <= 2 for piece in output))
            return torch.cat(output)
        full, split = collect([source]), collect([source[:2], source[2:5], source[5:]])
        self.assertTrue(torch.equal(full, split))
        self.assertTrue(torch.allclose(full, torch.tensor([0., .75, 1.5, 2.25, 3., 3.75, 4.5, 5.25, 6., 6.])))

    def test_same_rate_resampling_never_loses_or_duplicates_source_samples(self):
        resampler = StreamingSpeechResampler(16000, 16000, 2)
        output = [*resampler.push(torch.tensor([0.25, -0.25])), *resampler.push(torch.tensor([0.5])),
            *resampler.push(torch.empty(0), final=True)]
        self.assertTrue(torch.equal(torch.cat(output), torch.tensor([0.25, -0.25, 0.5])))

    def test_actual_paired_window_route_normalizes_stub_pcm_without_a_codec_or_brain(self):
        owner = SimpleNamespace(config=SimpleNamespace(audio_samples=2), device="cpu")
        owner._bounded_audio_target = lambda values: AdaptiveBrain._bounded_audio_target(owner, values)
        owner._iter_audio_windows = lambda _path: (value for value in [
            (torch.tensor([[[0.0, 1.0]]]), 2), (torch.tensor([[[2.0, 3.0]]]), 2)])
        source = MagicMock()
        source.__enter__.return_value.getframerate.return_value = 8000
        with patch("omni_core.brain.wave.open", return_value=source):
            windows = list(AdaptiveBrain._iter_paired_speech_windows(owner, "stub.wav"))
        self.assertEqual([actual for _target, actual in windows], [2, 2, 2, 2])
        output = torch.cat([target.flatten()[:actual] for target, actual in windows])
        self.assertTrue(torch.equal(output, torch.tensor([0., .5, 1., 1.5, 2., 2.5, 3., 3.])))


if __name__ == "__main__":
    unittest.main()
