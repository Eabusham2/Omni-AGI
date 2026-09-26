import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.distributed_runtime import DatasetManifest
from omni_core.distributed_training import (
    MonotonicManifestReplay,
    apply_media_updates,
    merge_media_training_state,
)
from omni_core.persistence import tensor_checksum


class DistributedMediaTrainingTests(unittest.TestCase):
    def setUp(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required")
        torch.manual_seed(811)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-distributed-media-"
        )
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()

        image = Image.new("RGB", (8, 8), (220, 30, 40))
        image.save(self.dataset / "01-image.png")

        with wave.open(str(self.dataset / "02-audio.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8_000)
            samples = [
                int(12_000 * math.sin(2.0 * math.pi * index / 16.0))
                for index in range(64)
            ]
            output.writeframes(b"".join(struct.pack("<h", value) for value in samples))

        first = Image.new("RGB", (8, 8), (20, 40, 210))
        second = Image.new("RGB", (8, 8), (30, 210, 80))
        first.save(
            self.dataset / "03-video.gif",
            save_all=True,
            append_images=[second],
            duration=50,
            loop=0,
        )
        self.brain = AdaptiveBrain.create(
            "distributed-media",
            self.root / "brain",
            OmniConfig.micro(
                max_seq_len=16,
                train_batch_size=1,
                gradient_accumulation=1,
            ),
            initialize_ground_up=True,
        )
        self.manifest = DatasetManifest.build(
            self.dataset,
            database_path=self.root / "manifest.sqlite3",
        )

    def tearDown(self):
        self.brain.events.close()
        self.temporary.cleanup()

    @staticmethod
    def _checksum(module):
        return tensor_checksum(
            parameter.detach().cpu() for parameter in module.parameters()
        )

    def test_monotonic_replay_trains_each_real_pack_once_with_full_coverage(self):
        before = {
            "image": self._checksum(self.brain.modalities.image),
            "audio": self._checksum(self.brain.modalities.audio),
            "video": self._checksum(self.brain.modalities.video),
        }
        replay = MonotonicManifestReplay(self.manifest, 0)
        reports = apply_media_updates(
            self.brain, replay.consume_until(len(self.manifest.entries))
        )
        after = {
            "image": self._checksum(self.brain.modalities.image),
            "audio": self._checksum(self.brain.modalities.audio),
            "video": self._checksum(self.brain.modalities.video),
        }

        self.assertEqual(replay.position, 3)
        self.assertEqual([value["kind"] for value in reports], ["image", "audio", "video"])
        self.assertTrue(all(value["coverage"]["complete"] for value in reports))
        self.assertTrue(
            all(
                value["parameterChecksumBefore"]
                != value["parameterChecksumAfter"]
                for value in reports
            )
        )
        self.assertNotEqual(before["image"], after["image"])
        self.assertNotEqual(before["audio"], after["audio"])
        self.assertNotEqual(before["video"], after["video"])

        state = merge_media_training_state(None, reports)
        self.assertEqual(state["trainedRecords"], 3)
        self.assertEqual(state["byModality"], {"audio": 1, "image": 1, "video": 1})
        self.assertTrue(state["allParameterChecksumsChanged"])
        self.assertEqual(
            apply_media_updates(self.brain, replay.consume_until(3)),
            [],
            "the monotonic high-water cannot replay a committed media row",
        )

    def test_resume_cursor_scans_prefix_once_and_yields_only_suffix(self):
        replay = MonotonicManifestReplay(self.manifest, 2)
        values = list(replay.consume_until(3))
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], 2)
        self.assertEqual(values[0][1].name, "03-video.gif")
        self.assertEqual(replay.position, 3)

    def test_disabled_media_pack_fails_closed_instead_of_advancing_a_receipt(self):
        self.brain.config.audio_enabled = False
        before = self._checksum(self.brain.modalities.audio)
        replay = MonotonicManifestReplay(self.manifest, 1)
        with self.assertRaisesRegex(RuntimeError, "did not mutate its modality pack"):
            apply_media_updates(self.brain, replay.consume_until(2))
        self.assertEqual(before, self._checksum(self.brain.modalities.audio))


if __name__ == "__main__":
    unittest.main()
