"""NEW real file/canvas/codec source boundaries, no model/brain constructor.

Only tiny raw CPU tensors, fake resource leases and inert decoder callbacks.
No packed kernel, learning, FFmpeg process, device capture or provider runs.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import io
import os
import sqlite3
import tempfile
import types
import unittest
import wave
from pathlib import Path

import torch
from PIL import Image

from test_operation_auto_boundaries import PACKAGE, MODAL

CANVAS = importlib.import_module(PACKAGE + ".media_canvas")
GENERATE = importlib.import_module(PACKAGE + ".media_canvas_generation")
CODEC = importlib.import_module(PACKAGE + ".media_streaming_codec")
RANDOM = importlib.import_module(PACKAGE + ".media_randomness")
PLAN = importlib.import_module(PACKAGE + ".media_planning")


class Lease:
    def __init__(self, policy, size): self.policy, self.size, self.closed, self.path = policy, size, False, None
    def bind_path(self, path): self.path = Path(path); return self
    def commit(self, *args, **kwargs): return self
    def mark_allocated(self, *args): return self
    def refresh_physical(self): return self
    def release(self): self.closed = True
    def backing_closed(self): self.release()
    def __enter__(self): return self
    def __exit__(self, *_args): self.release()


class Policy:
    def __init__(self): self.leases, self.demands, self.fail_ram, self.fail_disk = [], [], False, False
    def reserve_ram(self, size, purpose):
        self.demands.append(("ram", size, purpose))
        if self.fail_ram: raise RuntimeError("tiny selected RAM ceiling")
        lease = Lease(self, size); self.leases.append(lease); return lease
    def reserve_spill(self, size, purpose):
        self.require_disk(size, purpose); lease = Lease(self, size); self.leases.append(lease); return lease
    def allocate_spill_file(self, *_args, **_kwargs): return None
    def require_disk(self, size, purpose):
        self.demands.append(("disk", size, purpose))
        if self.fail_disk: raise RuntimeError("tiny selected spill pool")
        return {}


def plan(modality, *, width=6, height=5, samples=17, frames=3, file=True):
    windows = PLAN.NeuralMediaWindows(4, 8, 2, 2)
    measurements = PLAN.MediaGenerationMeasurements.for_modality(modality,
        available_memory_bytes=1 << 28, available_storage_bytes=1 << 30, native_unit_ms=1, source="inert-source-fixture")
    request = PLAN.MediaOutputRequest(width=width, height=height) if modality == "image" else (
        PLAN.MediaOutputRequest(duration_ms=samples / 16 * 1000, sample_rate=16) if modality == "audio" else
        PLAN.MediaOutputRequest(width=width, height=height, duration_ms=frames / 8 * 1000, fps=8))
    value = PLAN.plan_media_output(modality, windows, measurements, request)
    return dataclasses.replace(value, canvas_storage="file" if file else "device")


def hub():
    def image(idea, generator, **_kw): return torch.full((idea.shape[0], 3, 4, 4), .25)
    def audio(idea, generator, **_kw): return torch.linspace(-.5, .5, 8).reshape(1, 8).expand(idea.shape[0], -1)
    def video(idea, generator, **_kw): return torch.full((idea.shape[0], 3, 2, 4, 4), .25)
    encode = lambda value: torch.zeros((value.shape[0], 2))
    value = types.SimpleNamespace(image=types.SimpleNamespace(image_size=4, generate=image),
        audio=types.SimpleNamespace(samples=8, generate=audio, encode=encode),
        video=types.SimpleNamespace(image_size=4, frames=2, generate=video, encode=encode), eval=lambda: None)
    value._admit_plan, value._scaled_metadata = MODAL.ModalityHub._admit_plan, MODAL.ModalityHub._scaled_metadata
    for name in ("_unit_demand", "_generate_scaled_image", "_generate_scaled_audio", "_generate_scaled_video"):
        setattr(value, name, types.MethodType(getattr(MODAL.ModalityHub, name), value))
    return value


class MediaCanvasStreamingCases(unittest.TestCase):
    def scope(self, callback): return CANVAS.media_storage_lifetime(callback)()
    def canvas(self, root, policy, modality="image", shape=(1, 3, 2, 3), cancelled=None):
        return CANVAS.MediaCanvas(shape, modality=modality, dtype=torch.float32, directory=root, policy=policy, cancelled=cancelled)

    def test_overlap_neural_values_and_weight_coverage_use_real_file_bytes_and_release_exact_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            policy, paths = Policy(), []
            def run():
                canvas = self.canvas(root, policy); paths.extend([canvas.output.path, canvas.weights.path])
                region = (slice(0, 1), slice(0, 3), slice(0, 2), slice(0, 3))
                canvas.add(region, torch.ones((1, 3, 2, 3)) * .25, torch.ones((2, 3)))
                self.assertEqual(canvas.covered_fraction(), 1.0)
                first = canvas.view().neural_sha256()
                canvas.add(region, torch.ones((1, 3, 2, 3)) * .75, torch.ones((2, 3)))
                self.assertTrue(torch.equal(canvas.read(region), torch.full((1, 3, 2, 3), .5)))
                self.assertNotEqual(first, canvas.view().neural_sha256())
            self.scope(run)
            self.assertTrue(all(not path.exists() for path in paths)); self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_png_stream_decodes_the_exact_native_pixels_and_refuses_replaced_payload(self):
        with tempfile.TemporaryDirectory() as root:
            policy = Policy()
            def run():
                canvas = self.canvas(root, policy)
                canvas.add((slice(0, 1), slice(0, 3), slice(0, 2), slice(0, 3)), torch.full((1, 3, 2, 3), .5), torch.ones((2, 3)))
                payload = CODEC.encode_png_file(canvas.view(), directory=root, policy=policy)
                raw = payload.inline_bytes(65536)
                with Image.open(io.BytesIO(raw)) as image:
                    self.assertEqual(image.size, (3, 2)); self.assertEqual(image.getpixel((1, 1)), (191, 191, 191))
                destination = Path(root) / "saved.png"; payload.copy_to(destination)
                self.assertEqual(hashlib.sha256(destination.read_bytes()).hexdigest(), payload.sha256)
                replacement = Path(root) / "replacement"; replacement.write_bytes(b"wrong")
                os.replace(replacement, payload.path)
                with self.assertRaisesRegex(RuntimeError, "backing changed"): payload.inline_bytes(65536)
                payload.close(); self.assertEqual(payload.path.read_bytes(), b"wrong")
                payload.path.unlink()
            self.scope(run)
            self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_wav_and_apng_stream_exact_timeline_without_whole_output_materialization(self):
        with tempfile.TemporaryDirectory() as root:
            policy = Policy()
            def run():
                audio = self.canvas(root, policy, "audio", (1, 7))
                audio.add((slice(0, 1), slice(0, 7)), torch.full((1, 7), .5), torch.ones(7))
                payload = CODEC.encode_wav_file(audio.view(end=5), directory=root, policy=policy, sample_rate=16)
                with wave.open(io.BytesIO(payload.inline_bytes(65536)), "rb") as opened:
                    self.assertEqual(opened.getnframes(), 5); self.assertEqual(opened.getframerate(), 16)
                    self.assertEqual(opened.readframes(5), b"\xff?" * 5)
                video = self.canvas(root, policy, "video", (1, 3, 2, 2, 3))
                video.add((slice(0, 1), slice(0, 3), slice(0, 2), slice(0, 2), slice(0, 3)), torch.full((1, 3, 2, 2, 3), -.5), torch.ones((2, 2, 3)))
                movie = CODEC.encode_png_file(video.view(), directory=root, policy=policy, fps=8)
                with Image.open(io.BytesIO(movie.inline_bytes(65536))) as opened:
                    self.assertEqual(opened.n_frames, 2); opened.seek(1)
                    self.assertEqual(opened.getpixel((1, 1)), (63, 63, 63))
            self.scope(run)
            self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_center_first_order_matches_existing_math_without_materializing_a_coordinate_list(self):
        with tempfile.TemporaryDirectory() as root:
            policy = Policy()
            def run():
                canvas = self.canvas(root, policy)
                actual = list(GENERATE.spatial_starts(canvas, 13, 17, 4, 1))
                expected = MODAL._ordered_spatial_positions(13, 17, 4, 1)
                self.assertEqual(actual, expected); self.assertEqual(len(set(actual)), len(expected))
            self.scope(run)

    def test_actual_scaled_source_uses_file_canvas_for_all_modalities_and_keeps_exact_unit_progress(self):
        with tempfile.TemporaryDirectory() as root:
            for modality in ("image", "audio", "video"):
                with self.subTest(modality=modality):
                    policy, previews, native = Policy(), [], hub()
                    current = plan(modality)
                    def run():
                        def factory(selected, idea, cancelled):
                            shape = (1, selected.total_samples) if modality == "audio" else (
                                (1, 3, selected.total_frames, selected.height, selected.width) if modality == "video" else
                                (1, 3, selected.height, selected.width))
                            return self.canvas(root, policy, modality, shape, cancelled)
                        result = MODAL.ModalityHub.generate_scaled(native, current, torch.ones((1, 2)), seed=3,
                            preview_callback=lambda progress, output, details: previews.append((progress, details)), canvas_factory=factory)
                        self.assertIsInstance(result.tensor, CANVAS.MediaCanvasView)
                        self.assertEqual(previews[-1][0], 1.0)
                        self.assertEqual(previews[-1][1]["completedUnits"], current.work_units)
                        self.assertEqual(result.metadata["activityStorage"], "file")
                        self.assertEqual(len(result.tensor.neural_sha256()), 64)
                    self.scope(run)
                    self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_small_output_keeps_original_in_memory_consumer_and_no_spill_files(self):
        current = plan("image", file=False); native = hub()
        result = MODAL.ModalityHub.generate_scaled(native, current, torch.ones((1, 2)), seed=3, canvas_factory=lambda *_args: None)
        self.assertIsInstance(result.tensor, torch.Tensor); self.assertEqual(tuple(result.tensor.shape), (1, 3, 5, 6))
        self.assertTrue(torch.equal(result.tensor, torch.full((1, 3, 5, 6), .25)))

    def test_spill_plan_keeps_explicit_dimensions_and_never_enables_storage_without_the_option(self):
        windows = PLAN.NeuralMediaWindows(4, 8, 2, 2)
        measured = PLAN.MediaGenerationMeasurements.for_modality("image", available_memory_bytes=12*1024*1024,
            available_storage_bytes=1 << 30, native_unit_ms=1, preview_interval_ms=1e9, source="inert-source-fixture")
        request = PLAN.MediaOutputRequest(width=2048, height=2048)
        ordinary = PLAN.plan_media_output("image", windows, measured, request)
        spilled = PLAN.plan_media_output("image", windows, measured, request, allow_canvas_spill=True)
        self.assertFalse(ordinary.admitted); self.assertTrue(spilled.admitted)
        self.assertEqual((spilled.width, spilled.height), (2048, 2048)); self.assertEqual(spilled.canvas_storage, "file")
        self.assertEqual(spilled.canvas_bytes, 2048*2048*16)

    def test_stop_and_resource_refusal_do_not_leave_live_generated_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            policy, stopped = Policy(), [False]
            def run():
                canvas = self.canvas(root, policy, cancelled=lambda: stopped[0])
                stopped[0] = True
                with self.assertRaises(InterruptedError): canvas.covered_fraction()
            self.scope(run); self.assertEqual(list(Path(root).iterdir()), [])
            policy = Policy(); policy.fail_disk = True
            with self.assertRaisesRegex(RuntimeError, "spill pool"):
                self.scope(lambda: self.canvas(root, policy))
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_home_rng_and_codec_sampling_match_original_cpu_law_without_retained_generator_list(self):
        scope = RANDOM.MediaRandomScope("cpu")
        expected = torch.Generator().manual_seed(7)
        first = torch.randn((1, 2), generator=expected)
        logits = torch.tensor([[[.2, -.5, .9]]])
        sampled = torch.multinomial(logits.softmax(-1).reshape(-1, 3), 1, generator=expected).reshape(1, 1)
        with scope.activate():
            generator = RANDOM.media_generator("cpu", 7)
            self.assertTrue(torch.equal(RANDOM.media_randn((1, 2), generator=generator, device="cpu", dtype=torch.float32), first))
            self.assertTrue(torch.equal(RANDOM.media_codec_sample(logits, generator), sampled))
        self.assertEqual(scope.summary()["draws"], 2); self.assertEqual(scope.summary()["generators"], 1)
        self.assertIsInstance(scope.generators, int)


if __name__ == "__main__": unittest.main()
