import base64
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.modalities import (
    ModalityGenerationCancelled,
    ModalityHub,
)
from omni_core.media_planning import INLINE_MEDIA_BINARY_BYTES
from worker import Worker


class ProgressiveImaginationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(101)
        torch.set_num_threads(1)

    def test_real_decoder_revisions_refine_image_and_grow_audio_video(self):
        config = OmniConfig.micro()
        idea = torch.randn(1, config.idea_dim)
        hub = ModalityHub(config)

        previews = {}
        outputs = {}
        for modality in ("image", "audio", "video"):
            revisions = []
            outputs[modality] = hub.generate(
                modality,
                idea,
                seed=17,
                maximum_previews=2,
                preview_callback=lambda progress, tensor, values=revisions: (
                    values.append((progress, tensor.detach().clone()))
                ),
            )
            previews[modality] = revisions
            self.assertEqual(len(revisions), 2)
            self.assertEqual(revisions[-1][0], 1.0)
            self.assertFalse(torch.equal(revisions[0][1], revisions[-1][1]))

        image_revisions = previews["image"]
        self.assertTrue(
            all(
                tuple(tensor.shape) == (1, 3, config.image_size, config.image_size)
                for _progress, tensor in image_revisions
            )
        )
        self.assertEqual(
            tuple(outputs["image"].shape),
            (1, 3, config.image_size, config.image_size),
        )

        audio_lengths = [
            int(tensor.shape[-1]) for _progress, tensor in previews["audio"]
        ]
        self.assertEqual(audio_lengths, sorted(audio_lengths))
        self.assertLess(audio_lengths[0], audio_lengths[-1])
        self.assertEqual(audio_lengths[-1], config.audio_samples)
        self.assertEqual(
            tuple(outputs["audio"].shape), (1, config.audio_samples)
        )

        video_frames = [
            int(tensor.shape[2]) for _progress, tensor in previews["video"]
        ]
        self.assertEqual(video_frames, sorted(video_frames))
        self.assertLess(video_frames[0], video_frames[-1])
        self.assertEqual(video_frames[-1], config.video_frames)
        self.assertTrue(
            all(
                tuple(tensor.shape[-2:]) == (config.image_size, config.image_size)
                for _progress, tensor in previews["video"]
            )
        )
        self.assertEqual(
            tuple(outputs["video"].shape),
            (
                1,
                3,
                config.video_frames,
                config.image_size,
                config.image_size,
            ),
        )

    def test_hardware_tier_changes_only_preview_cadence(self):
        idea = torch.randn(1, OmniConfig.micro().idea_dim)
        counts = {}
        shapes = {}
        for tier in ("micro", "workstation"):
            config = OmniConfig.micro(hardware_tier=tier)
            with tempfile.TemporaryDirectory(
                prefix="omni-progressive-cadence-"
            ) as temporary:
                brain = AdaptiveBrain("cadence-" + tier, Path(temporary), config)
                revisions = []
                result = brain.generate_modality(
                    "image",
                    prompt="measured cadence",
                    seed=5,
                    preview_callback=lambda _progress, _mime, _data, detail: (
                        revisions.append(detail)
                    ),
                )
                counts[tier] = len(revisions)
                shapes[tier] = tuple(result["shape"])
                self.assertTrue(
                    all(detail["spatialResolutionReduced"] is False for detail in revisions)
                )
                self.assertTrue(
                    all(detail["actualDecoderOutput"] is True for detail in revisions)
                )
                brain.events.close()
        self.assertLess(counts["micro"], counts["workstation"])
        self.assertEqual(shapes["micro"], shapes["workstation"])

    def test_blank_prompt_uses_active_same_brain_state_and_can_cancel(self):
        with tempfile.TemporaryDirectory(
            prefix="omni-progressive-internal-"
        ) as temporary:
            config = OmniConfig.micro()
            brain = AdaptiveBrain("internal-seed", Path(temporary), config)
            brain._append_working_memory(
                torch.linspace(-0.8, 0.8, config.idea_dim),
                assembly_id="measured-active-idea",
                source="test",
                salience=0.9,
            )
            details = []
            result = brain.generate_modality(
                "audio",
                prompt="",
                concept_ids=[],
                seed=23,
                preview_callback=lambda _progress, mime, payload, detail: (
                    details.append((mime, payload, detail))
                ),
            )
            self.assertEqual(result["ideaSeed"]["source"], "active-working-memory")
            self.assertFalse(result["ideaSeed"]["promptProvided"])
            self.assertTrue(result["ideaSeed"]["sameBrain"])
            self.assertEqual(len(details), 2)
            self.assertTrue(all(mime == "audio/wav" for mime, _payload, _detail in details))
            self.assertTrue(all(payload[:4] == b"RIFF" for _mime, payload, _detail in details))
            self.assertLess(
                details[0][2]["sampleCount"], details[-1][2]["sampleCount"]
            )

            cancelled_previews = []
            with self.assertRaises(ModalityGenerationCancelled):
                brain.modalities.generate(
                    "image",
                    brain._modality_idea("", []),
                    seed=31,
                    maximum_previews=2,
                    preview_callback=lambda progress, _tensor: (
                        cancelled_previews.append(progress)
                    ),
                    cancel_check=lambda: len(cancelled_previews) >= 1,
                )
            self.assertEqual(len(cancelled_previews), 1)
            brain.events.close()

    def test_worker_revisions_are_hash_bound_to_actual_decoder_bytes(self):
        with tempfile.TemporaryDirectory(
            prefix="omni-progressive-protocol-"
        ) as temporary:
            config = OmniConfig.micro()
            brain = AdaptiveBrain("protocol-preview", Path(temporary), config)
            worker = Worker()
            worker.brains[brain.brain_id] = brain
            notifications = []
            worker.notify = lambda event_type, **values: notifications.append(
                (event_type, values)
            )
            try:
                result = worker.generate_modality(
                    {
                        "brainId": brain.brain_id,
                        "storagePath": str(Path(temporary)),
                        "jobId": "preview-job",
                        "modality": "image",
                        "prompt": "hash-bound decoder evidence",
                        "seed": 29,
                    },
                    "preview-request",
                )
                events = [
                    values
                    for event_type, values in notifications
                    if event_type == "modality-preview"
                ]
                self.assertEqual(len(events), 2)
                self.assertEqual(
                    [event["sequence"] for event in events], [0, 1]
                )
                self.assertEqual(
                    [event["data"]["preview"]["revision"] for event in events],
                    [0, 1],
                )
                self.assertEqual(
                    [
                        event["data"]["preview"]["completedUnits"]
                        for event in events
                    ],
                    [1, 4],
                )
                for event in events:
                    preview = event["data"]["preview"]
                    self.assertEqual(preview["schemaVersion"], 1)
                    self.assertEqual(preview["producer"], "same-brain-decoder")
                    self.assertTrue(preview["actualDecoderOutput"])
                    self.assertFalse(preview["spatialResolutionReduced"])
                    encoded = preview["dataUrl"].split(",", 1)[1]
                    payload = base64.b64decode(encoded, validate=True)
                    self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")
                    self.assertEqual(
                        hashlib.sha256(payload).hexdigest(),
                        preview["payloadSha256"],
                    )
                self.assertEqual(result["generationPerformance"]["progressivePreviews"], 2)
                self.assertTrue(
                    Path(result["path"]).name.startswith(
                        result["artifactSha256"]
                    )
                )
                self.assertFalse(
                    result["generationPerformance"][
                        "generationResolutionReducedForPreview"
                    ]
                )
            finally:
                worker._shutdown_inline_generations()
                brain.events.close()

    def test_large_preview_crosses_protocol_as_file_reference_not_base64(self):
        class Events:
            def append(self, *_args, **_kwargs):
                return "event"

        class LargePreviewBrain:
            def __init__(self, root):
                self.brain_id = "large-preview"
                self.storage_path = root
                self.engine_path = root / "engine"
                self.engine_path.mkdir(parents=True)
                self.events = Events()

            def generate_modality(self, **values):
                payload = (
                    b"\x89PNG\r\n\x1a\n"
                    + b"x" * INLINE_MEDIA_BINARY_BYTES
                )
                values["preview_callback"](
                    0.5,
                    "image/png",
                    payload,
                    {
                        "schemaVersion": 1,
                        "modality": "image",
                        "stage": "diffusion-vq-decode",
                        "completedUnits": 2,
                        "totalUnits": 4,
                        "actualDecoderOutput": True,
                        "spatialResolutionReduced": False,
                    },
                )
                return {"path": str(self.engine_path / "unused.png")}

        temporary = tempfile.TemporaryDirectory(prefix="omni-large-preview-")
        root = Path(temporary.name).resolve()
        brain = LargePreviewBrain(root)
        worker = Worker()
        worker.brains[brain.brain_id] = brain
        notifications = []
        worker.notify = lambda event_type, **values: notifications.append(
            (event_type, values)
        )
        try:
            worker.generate_modality(
                {
                    "brainId": brain.brain_id,
                    "storagePath": str(root),
                    "jobId": "large-preview-job",
                    "modality": "image",
                },
                "large-preview-request",
            )
            preview = next(
                values["data"]["preview"]
                for event_type, values in notifications
                if event_type == "modality-preview"
            )
            self.assertNotIn("dataUrl", preview)
            artifact = Path(preview["artifactPath"])
            self.assertTrue(artifact.is_file())
            self.assertEqual(
                hashlib.sha256(artifact.read_bytes()).hexdigest(),
                preview["payloadSha256"],
            )
        finally:
            worker._shutdown_inline_generations()
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
