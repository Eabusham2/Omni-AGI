"""Reviewed sensory sampling: bounded decoders and an actual tiny native brain.

No saved user brain, initial curriculum, corpus, foreign model or provider.
Codec-only fixtures are distinguished from actual native perception proofs.
"""
import base64
import hashlib
import io
import math
import os
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import wave
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from omni_core.mcp_sampling import parse_approved_sampling, sample_native_mcp, NativeSamplingCancelled
from omni_core.mcp_sampling_media import media_block, private_media_source, sensory_windows
from omni_core.streamed_vsa_activity import activity_reduction_scratch_bytes
from omni_core.offload import NeuralStateResourcePause
from codec_gateway import CodecOwner, CodecRuntimeCancelled, _SCOPE
from worker import Worker, ChatSteeringState
from worker_mcp_sampling import SamplingRecord
import tests.test_mcp_sampling as native_fixture


def image_bytes(format="PNG", animated=False):
    buffer = io.BytesIO()
    image = Image.new("RGB", (8, 8), (180, 40, 80))
    if animated:
        image.save(buffer, format=format, save_all=True, append_images=[Image.new("RGB", (8, 8), (20, 200, 70))],
                   duration=[40, 60], loop=0)
    else:
        image.save(buffer, format=format)
    return buffer.getvalue()


def wav_bytes(frames=137, rate=16000, width=2):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as source:
        source.setnchannels(1); source.setsampwidth(width); source.setframerate(rate)
        maximum = (1 << (width * 8 - 1)) - 1
        source.writeframes(b"".join((int(math.sin(i / 9.) * maximum * .6) + (128 if width == 1 else 0)).to_bytes(
            width, "little", signed=width != 1) for i in range(frames)))
    return buffer.getvalue()


def content(kind, mime, payload):
    return {"type": kind, "mimeType": mime, "data": base64.b64encode(payload).decode()}


class Lease:
    def __init__(self, policy):
        self.policy, self.path, self.released = policy, None, False
        policy.live.append(self)
    def __enter__(self): return self
    def __exit__(self, *_args): self.release()
    def release(self): self.released = True
    def bind_path(self, path): self.path = Path(path)
    def commit(self, **_kwargs): return self


class DecoderPolicy:
    def __init__(self, *, spill=False, reject_pixels=False):
        self.spill, self.reject_pixels, self.live, self.operations = spill, reject_pixels, [], []
    def reserve_ram(self, size, operation):
        self.operations.append((size, operation))
        if self.spill and operation == "private MCP source RAM buffer" or self.reject_pixels and "decoded pixels" in operation:
            raise NeuralStateResourcePause("fixture admission refusal", {"memoryPressure": True})
        return Lease(self)
    def allocate_spill_file(self, *_args, **_kwargs): return None
    def reserve_spill(self, _size, _operation): return Lease(self)


class MediaDecoderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="omni-private-mcp-decoder-")
        self.policy = DecoderPolicy()
        self.brain = SimpleNamespace(resource_policy=self.policy, device=torch.device("cpu"),
            config=SimpleNamespace(audio_samples=64, image_size=8), _live_paging_cache_directory=Path(self.directory.name))
    def tearDown(self):
        self.directory.cleanup()

    def decode(self, block):
        receipt = {}
        output = list(sensory_windows(self.brain, media_block(block, 0, 0), lambda: False, receipt))
        return output, receipt

    def test_base64_and_mime_reject_without_pretending_unsupported_blocks_were_read(self):
        for block in ({"type": "image", "mimeType": "image/svg+xml", "data": "AA=="},
                      {"type": "audio", "mimeType": "audio/midi", "data": "AA=="},
                      {"type": "image", "mimeType": "image/png", "data": "http://image"},
                      {"type": "image", "mimeType": "image/png", "data": "AB=="},
                      {"type": "video", "mimeType": "video/mp4", "data": "AA=="}):
            with self.subTest(block=block), self.assertRaises(ValueError):
                media_block(block, 0, 0)

    def test_ram_first_source_and_authorized_spill_fallback_remove_owned_bytes(self):
        block = media_block(content("image", "image/png", image_bytes()), 0, 0)
        with private_media_source(self.brain, block, lambda: False) as source:
            self.assertIsNone(source.path)
            self.assertEqual(source.sha256, hashlib.sha256(image_bytes()).hexdigest())
        self.assertTrue(all(lease.released for lease in self.policy.live))
        self.policy.spill = True
        with private_media_source(self.brain, block, lambda: False) as source:
            path = source.path
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), image_bytes())
        self.assertFalse(path.exists())
        self.assertTrue(all(lease.released for lease in self.policy.live))

    def test_all_image_frames_use_admitted_native_resolution_and_no_caption(self):
        for format, mime, animated in (("PNG", "image/png", False), ("JPEG", "image/jpeg", False),
                                       ("WEBP", "image/webp", False), ("GIF", "image/gif", True)):
            with self.subTest(format=format):
                output, receipt = self.decode(content("image", mime, image_bytes(format, animated)))
                self.assertEqual(len(output), 2 if animated else 1)
                self.assertEqual(receipt["sourceFrames"], len(output))
                self.assertTrue(receipt["wholeInputTraversed"])
                self.assertFalse(receipt["nativeSensory"], "decoder-only fixture must not claim native perception")
                self.assertTrue(all(tensor.shape == (1, 3, 8, 8) for tensor, _elapsed in output))

    def test_spill_source_replacement_is_rejected_and_never_deleted(self):
        self.policy.spill = True
        block = media_block(content("image", "IMAGE/PNG", image_bytes()), 0, 0)
        with self.assertRaisesRegex(RuntimeError, "replacement was not deleted"):
            with private_media_source(self.brain, block, lambda: False) as source:
                path = source.path
                original = path.with_suffix(".original")
                path.rename(original)
                path.write_bytes(b"not an owned source")
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    source.ensure_path()
        self.assertEqual(path.read_bytes(), b"not an owned source")
        self.assertEqual(original.read_bytes(), image_bytes())
        self.assertTrue(all(lease.released for lease in self.policy.live))

    def test_pixel_admission_precedes_conversion_and_mismatched_containers_fail(self):
        self.policy.reject_pixels = True
        with patch.object(Image.Image, "convert", side_effect=AssertionError("unadmitted pixels")), self.assertRaises(NeuralStateResourcePause):
            self.decode(content("image", "image/png", image_bytes()))
        with self.assertRaisesRegex(ValueError, "container signature"):
            self.decode(content("image", "image/png", wav_bytes()))
        self.assertTrue(all(lease.released for lease in self.policy.live))

    def test_audio_visits_every_pcm_frame_resamples_and_retains_the_tail_including_24_bit(self):
        for width, rate in ((1, 16000), (2, 8000), (3, 16000), (4, 44100)):
            with self.subTest(width=width, rate=rate):
                output, receipt = self.decode(content("audio", "audio/wav", wav_bytes(rate=rate, width=width)))
                expected = math.ceil(137 * 16000 / rate)
                self.assertEqual(receipt["sourceSampleFrames"], 137)
                self.assertEqual(receipt["modelSamples"], expected)
                self.assertEqual(len(output), math.ceil(expected / 64))
                self.assertEqual(receipt["decodedWindows"], len(output))
                self.assertTrue(receipt["wholeInputTraversed"])
                self.assertTrue(all(tensor.shape == (1, 1, 64) for tensor, _elapsed in output))

    def test_corrupt_truncated_or_unsupported_audio_is_not_partial_success(self):
        for payload in (wav_bytes()[:-10], b"#EXTM3U\nhttp://127.0.0.1/private\n", b"not a codec"):
            with self.subTest(payload=payload[:10]), self.assertRaises(ValueError):
                self.decode(content("audio", "audio/wav", payload))

    def test_cancel_between_audio_windows_releases_every_source_and_lease(self):
        self.policy.spill = True
        stop, receipt = [False], {}
        windows = sensory_windows(self.brain, media_block(content("audio", "audio/wav", wav_bytes(frames=200)), 0, 0), lambda: stop[0], receipt)
        next(windows); stop[0] = True
        with self.assertRaises(InterruptedError): next(windows)
        self.assertFalse(receipt["wholeInputTraversed"])
        self.assertTrue(all(lease.released for lease in self.policy.live))
        self.assertFalse(list(Path(self.directory.name).rglob("*.mcp-media")))

    def test_real_local_codec_fallback_is_streamed_local_only_and_cleans_its_seekable_source(self):
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            self.skipTest("No local FFmpeg fixture binary; no download is performed")
        wav, mp3 = Path(self.directory.name) / "fixture.wav", Path(self.directory.name) / "fixture.mp3"
        wav.write_bytes(wav_bytes(frames=205, rate=16000))
        encoded = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(wav), "-codec:a", "libmp3lame", str(mp3)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        if encoded.returncode:
            self.skipTest("Local codec build has no MP3 encoder fixture")
        @contextmanager
        def codec(_purpose): yield executable
        self.brain._video_runtime_executable = codec
        calls, actual_popen = [], subprocess.Popen
        def observed(arguments, **kwargs):
            calls.append(arguments)
            return actual_popen(arguments, **kwargs)
        with patch.dict("sys.modules", {"soundfile": None}), patch("omni_core.mcp_sampling_media.subprocess.Popen", side_effect=observed):
            output, receipt = self.decode(content("audio", "audio/mpeg", mp3.read_bytes()))
        self.assertTrue(output)
        self.assertEqual(receipt["modelSamples"], 205)
        self.assertTrue(receipt["wholeInputTraversed"])
        self.assertEqual(calls[0][calls[0].index("-protocol_whitelist") + 1], "file,pipe")
        self.assertIn("-xerror", calls[0])
        self.assertEqual(calls[0][calls[0].index("-map") + 1], "0:a", "do not silently select just one audio track")
        self.assertTrue(all(lease.released for lease in self.policy.live))
        self.assertFalse(list(Path(self.directory.name).rglob("*.mcp-media")))

    def test_flac_and_single_ogg_sources_traverse_their_full_decoded_samples(self):
        import numpy as np
        import soundfile
        for format, subtype, mime in (("FLAC", "PCM_16", "audio/flac"), ("OGG", "VORBIS", "audio/ogg")):
            with self.subTest(format=format):
                encoded = io.BytesIO()
                soundfile.write(encoded, np.sin(np.arange(141) / 9.).astype("float32") * .5, 16000,
                                format=format, subtype=subtype)
                output, receipt = self.decode(content("audio", mime, encoded.getvalue()))
                self.assertEqual(receipt["sourceSampleFrames"], 141)
                self.assertEqual(receipt["modelSamples"], 141)
                self.assertEqual(len(output), 3)
                self.assertTrue(receipt["wholeInputTraversed"])

    def test_aac_and_mp4_codec_inputs_decode_all_actual_output_and_multitrack_is_not_silently_dropped(self):
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            self.skipTest("No local FFmpeg fixture binary; no download is performed")
        wav = Path(self.directory.name) / "aac-source.wav"; wav.write_bytes(wav_bytes(frames=205))
        @contextmanager
        def codec(_purpose): yield executable
        self.brain._video_runtime_executable = codec
        for suffix, format, mime in (("aac", "adts", "audio/aac"), ("m4a", "ipod", "audio/mp4")):
            with self.subTest(format=format):
                path = Path(self.directory.name) / ("encoded." + suffix)
                encoded = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(wav), "-c:a", "aac", "-f", format, str(path)],
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                self.assertEqual(encoded.returncode, 0, encoded.stderr)
                reference = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                self.assertEqual(reference.returncode, 0, reference.stderr)
                with patch.dict("sys.modules", {"soundfile": None}):
                    output, receipt = self.decode(content("audio", mime, path.read_bytes()))
                self.assertEqual(receipt["modelSamples"], len(reference.stdout) // 4)
                self.assertEqual(len(output), math.ceil(receipt["modelSamples"] / 64))
                self.assertTrue(receipt["wholeInputTraversed"])
        multiple = Path(self.directory.name) / "multiple.m4a"
        encoded = subprocess.run([executable, "-nostdin", "-v", "error", "-i", str(wav), "-i", str(wav),
            "-map", "0:a", "-map", "1:a", "-c:a", "aac", str(multiple)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(encoded.returncode, 0, encoded.stderr)
        with patch.dict("sys.modules", {"soundfile": None}), self.assertRaisesRegex(ValueError, "multiple audio tracks"):
            self.decode(content("audio", "audio/mp4", multiple.read_bytes()))
        self.assertTrue(all(lease.released for lease in self.policy.live))

    def test_blocked_codec_reader_is_cancelled_and_owned_child_is_terminated(self):
        stop, exited, receipt = [False], threading.Event(), {}
        class Process:
            returncode = None
            def __init__(self): self.stdout = self
            def read(self, _size): exited.wait(3); return b""
            def poll(self): return self.returncode
            def terminate(self): self.returncode = -15; exited.set()
            def kill(self): self.terminate()
            def wait(self, timeout=None): exited.wait(timeout); return self.returncode
            def close(self): pass
        process = Process()
        @contextmanager
        def codec(_purpose): yield "owned-fixture-codec"
        self.brain._video_runtime_executable = codec
        timer = threading.Timer(.03, lambda: stop.__setitem__(0, True)); timer.start()
        try:
            with patch.dict("sys.modules", {"soundfile": None}), patch("omni_core.mcp_sampling_media.subprocess.Popen", return_value=process), self.assertRaises(InterruptedError):
                list(sensory_windows(self.brain, media_block(content("audio", "audio/mpeg", b"ID3" + b"\0" * 20), 0, 0), lambda: stop[0], receipt))
        finally:
            timer.join()
        self.assertTrue(exited.is_set())
        self.assertFalse(receipt["wholeInputTraversed"])
        self.assertTrue(all(lease.released for lease in self.policy.live))
        self.assertFalse(list(Path(self.directory.name).rglob("*.mcp-media")))

    def test_codec_read_failure_is_retained_under_backpressure_not_lost_into_an_infinite_wait(self):
        stop, receipt = [False], {}
        class Process:
            returncode = None
            def __init__(self): self.stdout = self; self.calls = 0
            def read(self, size):
                self.calls += 1
                if self.calls <= 2: return b"\0" * size
                self.returncode = 0
                raise OSError("fixture codec pipe failed")
            def poll(self): return self.returncode
            def terminate(self): self.returncode = -15
            def kill(self): self.terminate()
            def wait(self, timeout=None): return self.returncode
            def close(self): pass
        @contextmanager
        def codec(_purpose): yield "owned-fixture-codec"
        self.brain._video_runtime_executable = codec
        timer = threading.Timer(2, lambda: stop.__setitem__(0, True)); timer.start()
        try:
            with patch.dict("sys.modules", {"soundfile": None}), patch("omni_core.mcp_sampling_media.subprocess.Popen", return_value=Process()):
                windows = sensory_windows(self.brain, media_block(content("audio", "audio/mpeg", b"ID3" + b"\0" * 20), 0, 0), lambda: stop[0], receipt)
                next(windows); time.sleep(.1); next(windows)
                with self.assertRaisesRegex(RuntimeError, "codec read failed"): next(windows)
        finally:
            timer.cancel(); timer.join()
        self.assertFalse(receipt["wholeInputTraversed"])
        self.assertTrue(all(lease.released for lease in self.policy.live))


class NativeSensorySamplingTests(unittest.TestCase):
    snapshot = native_fixture.NativeSamplingFixtureTests.snapshot
    assert_unchanged = native_fixture.NativeSamplingFixtureTests.assert_unchanged
    @classmethod
    def setUpClass(cls):
        native_fixture.NativeSamplingFixtureTests.setUpClass()
        cls.brain = native_fixture.NativeSamplingFixtureTests.brain
    @classmethod
    def tearDownClass(cls):
        native_fixture.NativeSamplingFixtureTests.tearDownClass()

    def test_real_same_brain_image_and_audio_perception_without_learning_or_hidden_caption(self):
        brain = self.brain
        media = [content("image", "image/gif", image_bytes("GIF", True)), content("audio", "audio/wav", wav_bytes())]
        request = parse_approved_sampling(native_fixture.approved(maxTokens=1, messages=[{"role": "user", "content": [
            {"type": "text", "text": "see hear"}, *media]}]))
        observed, prompts = [], []
        perception, step = brain.modalities.perception_embedding, brain.decoder._generation_step
        def actual_perception(kind, tensor):
            observed.append((kind, tensor.shape))
            return perception(kind, tensor)
        def actual_step(window, bias, cache):
            prompts.append(window.clone())
            return step(window, bias, cache)
        before = self.snapshot()
        with patch.object(brain.modalities, "perception_embedding", side_effect=actual_perception), \
             patch.object(brain.decoder, "_generation_step", side_effect=actual_step), \
             patch.object(brain, "learn_experience", side_effect=AssertionError("private media learning")), \
             patch.object(brain, "_admit_sensory_embedding", side_effect=AssertionError("private media ingestion")), \
             patch.object(brain, "save", side_effect=AssertionError("private media save")), \
             patch.object(brain.events, "append", side_effect=AssertionError("private media event")):
            result = sample_native_mcp(brain, request)
        self.assertEqual([kind for kind, _shape in observed], ["image", "image", "audio", "audio", "audio"])
        self.assertEqual(prompts[0].tolist(), [[1, 259, *brain.tokenizer.encode("see hear"), 260]])
        self.assertEqual(result["mediaInput"][0]["sourceFrames"], 2)
        self.assertEqual(result["mediaInput"][1]["sourceSampleFrames"], 137)
        self.assertTrue(all(row["nativeSensory"] and row["wholeInputTraversed"] for row in result["mediaInput"]))
        self.assert_unchanged(before)
        self.assertEqual(brain.decoder.working_attention_status()["livePages"], 0)

    def test_media_only_task_uses_role_tokens_not_a_fabricated_prompt(self):
        brain = self.brain
        request = parse_approved_sampling(native_fixture.approved(maxTokens=1, messages=[{"role": "user",
            "content": content("image", "image/png", image_bytes())}]))
        seen, step = [], brain.decoder._generation_step
        def capture(window, bias, cache):
            seen.append(window.clone())
            return step(window, bias, cache)
        before = self.snapshot()
        with patch.object(brain.decoder, "_generation_step", side_effect=capture):
            result = sample_native_mcp(brain, request)
        self.assertEqual(seen[0].tolist(), [[1, 259, 260]])
        self.assertEqual(result["mediaInput"][0]["sourceFrames"], 1)
        self.assert_unchanged(before)

    def test_media_interrupt_and_disabled_region_return_no_partial_sample_or_mutation(self):
        brain = self.brain
        request = parse_approved_sampling(native_fixture.approved(maxTokens=1, messages=[{"role": "user",
            "content": content("audio", "audio/wav", wav_bytes())}]))
        before, stop = self.snapshot(), [False]
        perception = brain.modalities.perception_embedding
        def cancelled_perception(*args):
            result = perception(*args)
            stop[0] = True
            return result
        with patch.object(brain.modalities, "perception_embedding", side_effect=cancelled_perception), self.assertRaises(NativeSamplingCancelled):
            sample_native_mcp(brain, request, cancelled=lambda: stop[0])
        self.assert_unchanged(before)
        with patch.object(brain.config, "audio_enabled", False), patch.object(brain.modalities, "perception_embedding", side_effect=AssertionError("disabled native pack")), \
                self.assertRaisesRegex(ValueError, "disabled"):
            sample_native_mcp(brain, request)
        self.assert_unchanged(before)

    def test_text_sampling_admits_shared_streamed_cue_scratch_not_a_whole_label_matrix(self):
        brain = self.brain
        text = "a b c d e f g h i j k l m"
        labels = brain.memory.extract_concepts(text)
        self.assertGreater(len(labels), 32)
        admitted, reserve = [], brain.resource_policy.reserve_ram
        def record(amount, operation):
            if operation == "native MCP bounded VSA reduction and weighted-row decoding":
                admitted.append(amount)
            return reserve(amount, operation)
        request = parse_approved_sampling(native_fixture.approved(maxTokens=1,
            messages=[{"role": "user", "content": {"type": "text", "text": text}}]))
        before = self.snapshot()
        with patch.object(brain.resource_policy, "reserve_ram", side_effect=record):
            sample_native_mcp(brain, request)
        dimensions = brain.memory.space.dimensions
        self.assertEqual(admitted, [activity_reduction_scratch_bytes(dimensions, len(labels)) + dimensions * 256 + 8192])
        self.assert_unchanged(before)


class CodecSamplingOwnershipTests(unittest.TestCase):
    def test_private_codec_scope_keeps_real_rpc_correlation_without_cancelling_main_turn(self):
        worker = Worker()
        brain = SimpleNamespace(brain_id="mcp-native-fixture", resource_policy=DecoderPolicy())
        worker.brains[brain.brain_id] = brain
        session = ChatSteeringState(brain.brain_id, "owned-turn", claimed=True)
        worker._chat_steering[(brain.brain_id, session.stream_id)] = session
        approved = parse_approved_sampling(native_fixture.approved(originatingTurnId=session.stream_id))
        record = SamplingRecord(brain.brain_id, approved.request_id, approved.input_sha256, approved.turn_id, approved)
        owner = CodecOwner("actual-outer-rpc", brain.brain_id, stream_id=session.stream_id)
        def native_fixture_sample(_brain, _request, **_kwargs):
            self.assertIs(worker._codec_gateway.current_owner(), owner)
            self.assertFalse(_SCOPE.get()[2]())
            record.cancel.set()
            self.assertTrue(_SCOPE.get()[2](), "codec must observe this exact private sample's cancellation")
            raise CodecRuntimeCancelled("private codec setup cancelled")
        try:
            with worker._codec_gateway.scope(owner, lambda: False), patch("worker_mcp_sampling.sample_native_mcp", side_effect=native_fixture_sample):
                receipt = worker._execute_native_mcp_sample(record, brain)
                self.assertIs(worker._codec_gateway.current_owner(), owner)
                self.assertFalse(_SCOPE.get()[2](), "private cancellation leaked into outer codec ownership")
            self.assertEqual(receipt["state"], "cancelled")
            self.assertNotIn("result", receipt)
            self.assertFalse(worker._cooperative_cancel.is_set())
            self.assertFalse(session.requested.is_set())
            self.assertIsNone(record.request)
        finally:
            worker.brains.clear(); worker.shutdown({}, None)


if __name__ == "__main__": unittest.main()
