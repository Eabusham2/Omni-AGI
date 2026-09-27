import base64
import array
import io
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.modalities import (
    ModalityGenerationCancelled,
    ModalityHub,
)
from worker import Worker


class LiveObservationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-live-observation-"
        )

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def config(**overrides):
        return OmniConfig.micro(
            vision_enabled=True,
            image_enabled=True,
            audio_enabled=True,
            video_enabled=True,
            **overrides,
        )

    def test_imagination_selector_is_learned_ternary_and_generation_cancels(self):
        hub = ModalityHub(self.config())
        with torch.no_grad():
            levels = torch.zeros(
                hub.imagination_selector.ternary_weight_shape,
                dtype=torch.int8,
            )
            levels[2].fill_(1)
            hub.imagination_selector.set_ternary_weight_(levels)
        idea = torch.ones(1, hub.config.idea_dim)
        selected, scores = hub.select_imagination(
            idea, enabled=("image", "audio", "video")
        )
        self.assertEqual(selected, "video")
        self.assertGreater(scores["video"], scores["image"])
        self.assertTrue(
            set(
                hub.imagination_selector.effective_weight()
                .detach()
                .cpu()
                .unique()
                .tolist()
            ).issubset({-1.0, 0.0, 1.0})
        )

        previews = []
        with self.assertRaises(ModalityGenerationCancelled):
            hub.generate(
                "image",
                idea,
                seed=7,
                preview_callback=lambda progress, _value: previews.append(
                    progress
                ),
                cancel_check=lambda: len(previews) >= 1,
            )
        self.assertEqual(len(previews), 1)

    def test_worker_reports_same_brain_audio_without_claiming_speech(self):
        root = Path(self.temporary.name)
        brain = AdaptiveBrain("audio-capabilities", root / "brain", self.config())
        brain.modality_training["audio"] = 3
        brain.modality_training["video"] = 2
        worker = Worker()
        worker.brains[brain.brain_id] = brain
        try:
            result = worker.modality_capabilities(
                {
                    "brainId": brain.brain_id,
                    "storagePath": str(root / "brain"),
                },
                "capabilities",
            )
            self.assertTrue(result["audioPerception"])
            self.assertTrue(result["audioGeneration"])
            self.assertTrue(result["sameBrainSubstrate"])
            self.assertTrue(result["synchronizedVideoAudioGeneration"])
            self.assertFalse(result["neuralSpeechRecognition"])
            self.assertFalse(result["neuralSpeechSynthesis"])
            self.assertFalse(result["hiddenBehavioralPrompt"])
            self.assertIn("same neural substrate", result["detail"])
        finally:
            brain.events.close()
            worker.brains.clear()
            worker.shutdown({}, "shutdown")

    def test_installed_safe_audio_pack_counts_as_trained_live_input(self):
        root = Path(self.temporary.name)
        brain = AdaptiveBrain("installed-audio", root / "installed", self.config())
        brain.modality_training["audio"] = 0
        brain.installed_modality_packs.append(
            {"id": "safe-audio", "modalities": ["audio"]}
        )
        readiness = Worker._trained_modality_capabilities(brain)
        self.assertTrue(readiness["audioNeural"])
        self.assertTrue(readiness["audioGeneration"])
        brain.events.close()

    def test_inline_video_snapshot_copies_linked_audio_decoder(self):
        root = Path(self.temporary.name)
        brain = AdaptiveBrain("inline-av", root / "inline-av", self.config())
        brain.modality_training["video"] = 1
        brain.modality_training["audio"] = 1
        worker = Worker()
        record = worker._start_inline_generation(
            brain,
            "a" * 32,
            "stream-inline-av",
            {
                "kind": "imagine",
                "toolId": "modality.imagine",
                "action": "generate",
                "arguments": {
                    "modality": "video",
                    "prompt": "linked motion and abstract sound",
                    # This test verifies that the isolated video snapshot also
                    # receives its linked audio decoder. Keep it on the native
                    # fixture dimensions; hardware-scaled Auto/Exact output is
                    # covered independently and can legitimately take longer
                    # than this unit test's bounded future deadline.
                    "settings": {
                        "outputMode": "legacy",
                        "fps": 8,
                        "sampleRate": 16_000,
                    },
                    "seed": 51,
                },
            },
            lambda _record, _preview: None,
        )
        try:
            self.assertIsNotNone(record)
            self.assertIsNotNone(record.future)
            result = record.future.result(timeout=30)
            self.assertTrue(result["synchronizedAudio"]["supported"])
            self.assertTrue(result["synchronizedAudio"]["sameBrainIdea"])
            self.assertFalse(result["synchronizedAudio"]["speechSynthesis"])
            if result["mimeType"] == "video/mp4":
                self.assertTrue(result["synchronizedAudio"]["generated"])
                self.assertIn(b"soun", Path(result["path"]).read_bytes())
        finally:
            worker.shutdown({}, "shutdown")
            brain.events.close()

    def test_high_resolution_frame_uses_global_and_distinct_position_tiles(self):
        try:
            from PIL import Image, ImageDraw
        except ImportError as error:  # pragma: no cover - required dependency
            self.skipTest(str(error))
        root = Path(self.temporary.name)
        brain = AdaptiveBrain("multires", root / "brain", self.config())
        image = Image.new("RGB", (64, 48), (12, 18, 28))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 31, 23), fill=(250, 20, 10))
        draw.rectangle((32, 0, 63, 23), fill=(10, 240, 30))
        draw.rectangle((0, 24, 31, 47), fill=(20, 40, 250))
        draw.line((0, 47, 63, 0), fill=(255, 255, 255), width=3)
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        payload = encoded.getvalue()

        with torch.no_grad():
            global_only = brain.modalities.perception_embedding(
                "image", brain._live_image_tensor(payload)
            )
            multires, diagnostics = brain._live_visual_embedding(
                payload,
                "image",
                {"width": 64, "height": 48, "resolutionMode": "native"},
            )
        self.assertGreater(diagnostics["tilesEncoded"], 1)
        self.assertEqual(diagnostics["sourceWidth"], 64)
        self.assertEqual(diagnostics["sourceHeight"], 48)
        self.assertEqual(diagnostics["tileBinding"], "ternary-position-vsa")
        self.assertFalse(diagnostics["rawTilesStored"])
        self.assertFalse(torch.allclose(multires, global_only, atol=1e-6))

        before_assemblies = len(brain.memory.assemblies)
        result = brain.observe_live_packet(
            modality="image",
            mime_type="image/png",
            payload=payload,
            session_id="camera-session",
            sequence=0,
            timestamp_ms=1.0,
            retention="neural",
            settings={
                "width": 64,
                "height": 48,
                "resolutionMode": "native",
                "observationControlId": "snapshot-one",
                "burstIndex": 0,
                "burstCount": 1,
            },
            permission_source="camera",
        )
        self.assertGreater(len(brain.memory.assemblies), before_assemblies)
        self.assertIsNotNone(result["assemblyId"])
        self.assertGreater(result["perception"]["tilesEncoded"], 1)
        self.assertFalse(result["rawPacketStored"])
        self.assertFalse(result["datasetCoverageCommitted"])
        self.assertTrue(result["sameBrainSharedIdeaSpace"])

        generated = brain.generate_modality("image", seed=19)
        self.assertGreater(generated["generationPerformance"]["elapsedMs"], 0)
        self.assertGreater(
            generated["generationPerformance"]["stepsPerSecond"], 0
        )
        self.assertFalse(
            generated["generationPerformance"]["hiddenBehavioralPrompt"]
        )
        Path(generated["path"]).unlink(missing_ok=True)

    def test_worker_accepts_packet_above_old_fixed_limit_when_resources_allow(self):
        worker = Worker()
        root = Path(self.temporary.name) / "worker-brain"
        brain = MagicMock()
        brain.brain_id = "worker-live"
        brain.storage_path = root.resolve()
        brain.config = SimpleNamespace(
            vision_enabled=True,
            image_enabled=True,
            audio_enabled=True,
            video_enabled=True,
        )
        brain.modality_training = {
            "vision": 1,
            "image": 1,
            "audio": 1,
            "video": 1,
        }
        brain.observe_live_packet.return_value = {
            "assemblyId": "assembly-worker",
            "spikeRate": 0.1,
            "novelty": 0.2,
            "packetSha256": "unused-by-worker",
            "rawPacketStored": False,
            "datasetCoverageCommitted": False,
            "sameBrainSharedIdeaSpace": True,
            "hiddenBehavioralPrompt": False,
        }
        brain.idle_cycle.return_value = {"ran": False, "actions": []}
        worker.brains[brain.brain_id] = brain
        start = {
            "sessionId": "session-large",
            "brainId": brain.brain_id,
            "storagePath": str(root),
            "modalities": ["image"],
            "permission": {
                "source": "camera",
                "granted": True,
                "scope": "session",
                "grantedAt": "2026-08-22T12:00:00Z",
            },
            "retention": "neural",
            "toolSchemas": [],
            "maxPacketBytes": 8 * 1024 * 1024,
        }
        with patch("worker._available_memory_bytes", return_value=1024**3):
            session = worker.start_observation(start, None)
            self.assertEqual(session["maxPacketBytes"], 8 * 1024 * 1024)
            payload = b"z" * (5 * 1024 * 1024)
            result = worker.observe_packet(
                {
                    "sessionId": "session-large",
                    "modality": "image",
                    "sequence": 0,
                    "timestampMs": 1,
                    "mimeType": "image/png",
                    "dataBase64": base64.b64encode(payload).decode("ascii"),
                    "settings": {},
                },
                None,
            )
        self.assertEqual(result["session"]["bytesAccepted"], len(payload))
        observed_payload = brain.observe_live_packet.call_args.kwargs["payload"]
        self.assertEqual(len(observed_payload), len(payload))
        self.assertGreater(len(observed_payload), 4 * 1024 * 1024)
        worker.stop_observation({"sessionId": "session-large"}, None)
        worker.shutdown({}, None)

    def test_live_pcm_audio_forms_an_assembly_that_can_drive_same_brain_imagination(self):
        root = Path(self.temporary.name)
        brain = AdaptiveBrain("live-audio", root / "audio-brain", self.config())
        sample_count = max(64, brain.config.audio_samples // 2)
        samples = array.array(
            "h",
            [
                int(12_000 * math.sin(index * 2.0 * math.pi / 32.0))
                for index in range(sample_count)
            ],
        )
        before_assemblies = len(brain.memory.assemblies)
        before_selector = tuple(
            tensor.detach().clone()
            for tensor in brain.modalities.imagination_selector.authoritative_packed_tensors()
        )
        observed = brain.observe_live_packet(
            modality="audio",
            mime_type="audio/pcm-s16le",
            payload=samples.tobytes(),
            session_id="microphone-session",
            sequence=0,
            timestamp_ms=1.0,
            retention="neural",
            settings={"sampleRate": 16_000, "channels": 1},
            permission_source="microphone",
        )
        self.assertGreater(len(brain.memory.assemblies), before_assemblies)
        self.assertIsNotNone(observed["assemblyId"])
        self.assertTrue(observed["sameBrainSharedIdeaSpace"])
        self.assertFalse(observed["rawPacketStored"])
        self.assertFalse(observed["datasetCoverageCommitted"])
        self.assertTrue(any(
            not torch.equal(before, after)
            for before, after in zip(
                before_selector,
                brain.modalities.imagination_selector.authoritative_packed_tensors(),
            )
        ))

        generated = brain.generate_modality(
            "audio",
            concept_ids=[observed["assemblyId"]],
            seed=23,
        )
        self.assertEqual(generated["mimeType"], "audio/wav")
        self.assertTrue(Path(generated["path"]).is_file())
        self.assertFalse(
            generated["generationPerformance"]["hiddenBehavioralPrompt"]
        )
        Path(generated["path"]).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
