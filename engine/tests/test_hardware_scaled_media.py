import base64
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.config import OmniConfig
from omni_core.brain import AdaptiveBrain
from omni_core.media_planning import (
    INLINE_MEDIA_BINARY_BYTES,
    MediaGenerationMeasurements,
    MediaOutputRequest,
    MediaPlanError,
    MediaResourcePause,
    NeuralMediaWindows,
    inline_media_data_url,
    plan_media_output,
)
from omni_core.modalities import ModalityGenerationCancelled, ModalityHub


GIB = 1024**3


def measurements(
    *,
    memory: int = 2 * GIB,
    storage: int = 16 * GIB,
    latency_ms: float = 30_000.0,
    image_ms: float = 278.0,
    audio_ms: float = 276.0,
    video_ms: float = 203.0,
) -> MediaGenerationMeasurements:
    return MediaGenerationMeasurements(
        available_memory_bytes=memory,
        available_storage_bytes=storage,
        image_tile_ms=image_ms,
        audio_chunk_ms=audio_ms,
        video_tile_window_ms=video_ms,
        target_latency_ms=latency_ms,
        preview_interval_ms=500.0,
        parallel_units=1,
        source="measured-m1-disposable-test",
    )


class HardwareScaledMediaPlanningTests(unittest.TestCase):
    def test_inline_playback_fallback_is_bounded_without_capping_output(self):
        ordinary_auto_wav = b"R" * 183_852
        embedded = inline_media_data_url("audio/wav", ordinary_auto_wav)
        self.assertEqual(INLINE_MEDIA_BINARY_BYTES, 512 * 1024)
        self.assertIsNotNone(embedded)
        self.assertEqual(
            base64.b64decode(str(embedded).split(",", 1)[1], validate=True),
            ordinary_auto_wav,
        )
        self.assertIsNone(
            inline_media_data_url(
                "audio/wav", b"R" * (INLINE_MEDIA_BINARY_BYTES + 1)
            )
        )

    def test_single_modality_benchmark_cannot_be_relabelled(self):
        measured = MediaGenerationMeasurements.for_modality(
            "audio",
            available_memory_bytes=GIB,
            available_storage_bytes=GIB,
            native_unit_ms=25.0,
            source="measured-audio-probe",
        )
        plan = plan_media_output(
            "audio",
            NeuralMediaWindows(32, 512, 6, 24),
            measured,
        )
        self.assertEqual(plan.measurement_source, "measured-audio-probe")
        with self.assertRaises(MediaPlanError):
            plan_media_output(
                "image",
                NeuralMediaWindows(32, 512, 6, 24),
                measured,
            )

    def test_measured_m1_gpu_windows_admit_useful_size_floors(self):
        windows = NeuralMediaWindows(
            image_patch=32,
            audio_chunk_samples=512,
            video_window_frames=6,
            channels=24,
        )
        measured = measurements()
        image = plan_media_output("image", windows, measured)
        audio = plan_media_output("audio", windows, measured)
        video = plan_media_output("video", windows, measured)

        self.assertTrue(image.admitted)
        self.assertGreaterEqual(image.width or 0, 256)
        self.assertGreaterEqual(image.height or 0, 256)
        self.assertTrue(image.useful_floor_met)
        self.assertTrue(audio.admitted)
        self.assertGreaterEqual(audio.duration_ms or 0.0, 2_000.0)
        self.assertTrue(audio.useful_floor_met)
        self.assertTrue(video.admitted)
        self.assertGreaterEqual(video.width or 0, 128)
        self.assertGreaterEqual(video.height or 0, 128)
        self.assertGreaterEqual(video.total_frames or 0, 16)
        self.assertTrue(video.useful_floor_met)
        for plan in (image, audio, video):
            self.assertIsNone(plan.model_defined_maximum)
            self.assertIsNone(plan.as_dict()["modelDefinedMaximum"])
            self.assertEqual(
                plan.as_dict()["measurementSource"],
                "measured-m1-disposable-test",
            )
            self.assertLessEqual(
                plan.estimated_peak_memory_bytes,
                measured.available_memory_bytes,
            )
            self.assertLessEqual(
                plan.estimated_peak_storage_bytes,
                measured.available_storage_bytes,
            )

    def test_lower_measured_throughput_scales_auto_down_without_false_floor(self):
        windows = NeuralMediaWindows(32, 512, 6, 24)
        constrained = measurements(
            memory=64 * 1024**2,
            storage=64 * 1024**2,
            latency_ms=1_000,
            image_ms=500,
            audio_ms=500,
            video_ms=500,
        )
        image = plan_media_output("image", windows, constrained)
        audio = plan_media_output("audio", windows, constrained)
        video = plan_media_output("video", windows, constrained)
        self.assertTrue(image.admitted)
        self.assertLess(image.width or 0, 256)
        self.assertFalse(image.useful_floor_met)
        self.assertTrue(audio.admitted)
        self.assertLess(audio.duration_ms or 0.0, 2_000.0)
        self.assertFalse(audio.useful_floor_met)
        self.assertTrue(video.admitted)
        self.assertLess(video.width or 0, 128)
        self.assertFalse(video.useful_floor_met)

    def test_faster_hardware_scales_above_m1_and_has_no_model_ceiling(self):
        windows = NeuralMediaWindows(32, 512, 6, 24)
        baseline = plan_media_output("image", windows, measurements())
        faster = plan_media_output(
            "image",
            windows,
            measurements(image_ms=40.0),
        )
        self.assertGreater(faster.width or 0, baseline.width or 0)
        self.assertIsNone(faster.model_defined_maximum)

        exact = plan_media_output(
            "image",
            windows,
            measurements(latency_ms=1.0, memory=8 * GIB, storage=32 * GIB),
            MediaOutputRequest(width=1_024, height=768),
        )
        self.assertTrue(exact.explicit_request)
        self.assertTrue(exact.admitted)
        self.assertEqual((exact.width, exact.height), (1_024, 768))
        self.assertFalse(exact.within_latency_budget)
        self.assertIsNone(exact.model_defined_maximum)

    def test_explicit_resource_failure_is_reported_without_silent_clamping(self):
        windows = NeuralMediaWindows(32, 512, 6, 24)
        requested = MediaOutputRequest(width=8_192, height=8_192)
        plan = plan_media_output(
            "image",
            windows,
            measurements(memory=8 * 1024**2, storage=8 * 1024**2),
            requested,
        )
        self.assertFalse(plan.admitted)
        self.assertEqual((plan.width, plan.height), (8_192, 8_192))
        self.assertIn(plan.limiting_resource, {"memory", "storage"})


class HardwareScaledMediaCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(301)
        torch.set_num_threads(1)
        self.config = OmniConfig.micro()
        self.windows = NeuralMediaWindows.from_config(self.config)
        self.hub = ModalityHub(self.config)
        self.idea = torch.randn(1, self.config.idea_dim)
        self.measured = MediaGenerationMeasurements(
            available_memory_bytes=512 * 1024**2,
            available_storage_bytes=512 * 1024**2,
            image_tile_ms=1.0,
            audio_chunk_ms=1.0,
            video_tile_window_ms=1.0,
            target_latency_ms=10_000.0,
            preview_interval_ms=2.0,
            source="disposable-unit-benchmark",
        )

    def test_tiled_image_is_actual_positioned_decode_and_preserves_state_dict(self):
        before = {
            key: value.detach().clone()
            for key, value in self.hub.state_dict().items()
        }
        plan = plan_media_output(
            "image",
            self.windows,
            self.measured,
            MediaOutputRequest(width=16, height=12),
        )
        previews = []
        result = self.hub.generate_scaled(
            plan,
            self.idea,
            seed=17,
            training_steps=0,
            preview_callback=lambda progress, tensor, detail: previews.append(
                (progress, tensor.detach().clone(), detail)
            ),
        )
        self.assertEqual(tuple(result.tensor.shape), (1, 3, 12, 16))
        self.assertTrue(torch.isfinite(result.tensor).all())
        self.assertGreater(plan.work_units, 1)
        self.assertEqual(previews[0][0], 1.0 / plan.work_units)
        self.assertEqual(previews[-1][0], 1.0)
        self.assertEqual(len(previews), plan.estimated_previews)
        self.assertTrue(previews[0][2]["partialCoverage"])
        self.assertFalse(previews[-1][2]["partialCoverage"])
        self.assertEqual(previews[-1][2]["trainingState"], "untrained-diagnostic")
        self.assertEqual(result.metadata["trainingState"], "untrained-diagnostic")
        self.assertFalse(result.metadata["semanticQualityClaimed"])
        self.assertTrue(result.metadata["legacyCheckpointCompatible"])
        self.assertEqual(set(before), set(self.hub.state_dict()))
        for key, expected in before.items():
            self.assertTrue(torch.equal(self.hub.state_dict()[key], expected), key)

        reloaded = ModalityHub(self.config)
        reloaded.load_state_dict(before, strict=True)
        repeated = reloaded.generate_scaled(plan, self.idea, seed=17)
        self.assertTrue(torch.equal(result.tensor, repeated.tensor))

    def test_audio_uses_overlap_recurrent_carry_and_growing_stable_previews(self):
        plan = plan_media_output(
            "audio",
            self.windows,
            self.measured,
            MediaOutputRequest(duration_ms=20.0, sample_rate=8_000),
        )
        previews = []
        watermark = []
        result = self.hub.generate_scaled(
            plan,
            self.idea,
            seed=23,
            training_steps=7,
            resource_watermark=lambda demand: watermark.append(demand) or True,
            preview_callback=lambda progress, tensor, detail: previews.append(
                (progress, tensor.shape[-1], detail)
            ),
        )
        self.assertEqual(tuple(result.tensor.shape), (1, 160))
        self.assertTrue(torch.isfinite(result.tensor).all())
        self.assertEqual(len(watermark), plan.work_units + 1)
        self.assertEqual(watermark[0].stage, "output-allocation")
        self.assertEqual([value[1] for value in previews], sorted(value[1] for value in previews))
        self.assertEqual(previews[-1][1], 160)
        self.assertEqual(len(previews), plan.estimated_previews)
        self.assertTrue(any(value[2]["recurrentCarryUsed"] for value in previews[1:]))
        self.assertEqual(
            previews[-1][2]["trainingState"],
            "trained-unverified-quality",
        )
        self.assertEqual(result.metadata["trainingState"], "trained-unverified-quality")
        self.assertFalse(result.metadata["semanticQualityClaimed"])

    def test_video_rolls_temporal_windows_and_positioned_spatial_tiles(self):
        plan = plan_media_output(
            "video",
            self.windows,
            self.measured,
            MediaOutputRequest(width=12, height=12, duration_ms=750.0, fps=4),
        )
        previews = []
        result = self.hub.generate_scaled(
            plan,
            self.idea,
            seed=31,
            training_steps=3,
            preview_callback=lambda progress, tensor, detail: previews.append(
                (progress, tuple(tensor.shape), detail)
            ),
        )
        self.assertEqual(tuple(result.tensor.shape), (1, 3, 3, 12, 12))
        self.assertTrue(torch.isfinite(result.tensor).all())
        self.assertGreater(plan.work_units, 1)
        self.assertEqual(previews[-1][0], 1.0)
        self.assertEqual(previews[-1][1], (1, 3, 3, 12, 12))
        self.assertEqual(len(previews), plan.estimated_previews)
        self.assertTrue(any(value[2]["recurrentCarryUsed"] for value in previews))
        self.assertFalse(previews[-1][2]["partialCoverage"])

    def test_cancel_and_live_watermark_stop_between_native_units(self):
        plan = plan_media_output(
            "image",
            self.windows,
            self.measured,
            MediaOutputRequest(width=16, height=16),
        )
        admitted = 0

        def watermark(_demand):
            nonlocal admitted
            admitted += 1
            return admitted < 3

        with self.assertRaises(MediaResourcePause):
            self.hub.generate_scaled(
                plan,
                self.idea,
                resource_watermark=watermark,
            )

        visible_previews = []
        with self.assertRaises(ModalityGenerationCancelled):
            self.hub.generate_scaled(
                plan,
                self.idea,
                preview_callback=lambda progress, _tensor, _detail: (
                    visible_previews.append(progress)
                ),
                cancel_check=lambda: bool(visible_previews),
            )
        self.assertEqual(visible_previews, [1.0 / plan.work_units])


class HardwareScaledAdaptiveBrainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(401)
        torch.set_num_threads(1)

    def test_exact_image_generation_persists_scaled_plan_and_truthful_previews(self):
        with tempfile.TemporaryDirectory(prefix="omni-scaled-brain-image-") as root:
            brain = AdaptiveBrain("scaled-image", Path(root), OmniConfig.micro())
            previews = []
            try:
                result = brain.generate_modality(
                    "image",
                    prompt="disposable tiled image",
                    seed=71,
                    settings={
                        "outputMode": "exact",
                        "width": 16,
                        "height": 12,
                        "targetLatencyMs": 10_000,
                        "previewIntervalMs": 1,
                    },
                    preview_callback=lambda progress, mime, payload, detail: (
                        previews.append((progress, mime, payload, detail))
                    ),
                )
                self.assertEqual(result["shape"], [1, 3, 12, 16])
                self.assertEqual(
                    (result["mediaOutputPlan"]["width"], result["mediaOutputPlan"]["height"]),
                    (16, 12),
                )
                self.assertTrue(result["mediaOutputPlan"]["explicitRequest"])
                self.assertIsNone(result["mediaOutputPlan"]["modelDefinedMaximum"])
                self.assertEqual(
                    result["mediaOutput"]["trainingState"],
                    "untrained-diagnostic",
                )
                self.assertFalse(result["mediaOutput"]["semanticQualityClaimed"])
                self.assertTrue(Path(result["path"]).is_file())
                self.assertGreater(len(previews), 1)
                self.assertEqual(previews[-1][0], 1.0)
                self.assertEqual(previews[-1][1], "image/png")
                self.assertEqual(previews[-1][2][:8], b"\x89PNG\r\n\x1a\n")
                self.assertTrue(previews[-1][3]["hardwareScaled"])
                self.assertEqual(
                    result["generationPerformance"]["progressivePreviews"],
                    len(previews),
                )
            finally:
                brain.events.close()

    def test_exact_audio_duration_and_sample_rate_are_not_silently_clamped(self):
        with tempfile.TemporaryDirectory(prefix="omni-scaled-brain-audio-") as root:
            brain = AdaptiveBrain("scaled-audio", Path(root), OmniConfig.micro())
            previews = []
            try:
                brain.modality_training["audio"] = 2
                result = brain.generate_modality(
                    "audio",
                    seed=73,
                    settings={
                        "outputMode": "exact",
                        "durationMs": 20.0,
                        "sampleRate": 8_000,
                        "targetLatencyMs": 10_000,
                        "previewIntervalMs": 1,
                    },
                    preview_callback=lambda _progress, mime, payload, detail: (
                        previews.append((mime, payload, detail))
                    ),
                )
                self.assertEqual(result["shape"], [1, 160])
                self.assertEqual(result["mediaOutputPlan"]["sampleRate"], 8_000)
                self.assertEqual(result["mediaOutputPlan"]["totalSamples"], 160)
                self.assertEqual(result["mediaOutputPlan"]["durationMs"], 20.0)
                self.assertEqual(
                    result["mediaOutput"]["trainingState"],
                    "trained-unverified-quality",
                )
                self.assertTrue(all(mime == "audio/wav" for mime, _payload, _detail in previews))
                self.assertTrue(all(payload[:4] == b"RIFF" for _mime, payload, _detail in previews))
                self.assertEqual(previews[-1][2]["sampleRate"], 8_000)
                self.assertEqual(previews[-1][2]["durationMs"], 20.0)
            finally:
                brain.events.close()


if __name__ == "__main__":
    unittest.main()
