import unittest
from media_timeline import encoded_media_timeline


class MeasuredMediaTimelineTests(unittest.TestCase):
    def test_progressive_range_does_not_claim_sample_stability(self):
        value = encoded_media_timeline("samples", 320, 16000)
        self.assertEqual(value["count"], 320)
        self.assertEqual(value["layout"], "replacement-from-zero")
        self.assertNotIn("stableCount", value)
        self.assertEqual(encoded_media_timeline("samples", 320, 16000, stable_count=160)["stableCount"], 160)

    def test_container_audio_requires_actual_completed_paired_output(self):
        audio = {"sampleRate": 16000, "sampleCount": 8000, "startSample": 0}
        with self.assertRaises(ValueError):
            encoded_media_timeline("frames", 4, 8, container_audio=audio)
        self.assertEqual(encoded_media_timeline("frames", 4, 8, complete=True, container_audio=audio)["containerAudio"], audio)

    def test_bad_counts_and_invented_stability_are_rejected(self):
        for values in [("samples", True, 16000), ("image", 2, 1), ("frames", 4, 0)]:
            with self.assertRaises(ValueError):
                encoded_media_timeline(*values)
        with self.assertRaises(ValueError):
            encoded_media_timeline("frames", 2, 8, stable_count=3)
