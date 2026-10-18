"""Tiny typed-file parser contracts; no brain, tensor, decoder, codec or learning."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from omni_core.datasets import DatasetCoverage, _structured_record, iter_dataset_records
from omni_core.text_spool import INLINE_TEXT_BYTES


class TeacherMediaPairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "assets").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def pair(self, kind="image", condition="The bird sings and moves."):
        raw, mime, extension = {
            "image": (b"\x89PNG\r\n\x1a\nfixture", "image/png", "png"),
            "audio": (b"RIFF0000WAVEfixture", "audio/wav", "wav"),
            "video": (b"0000ftypisomfixture", "video/mp4", "mp4"),
        }[kind]
        digest = hashlib.sha256(raw).hexdigest()
        (self.root / "assets" / (digest + "." + extension)).write_bytes(raw)
        return {"format": "omni-teacher-media-pair-1", "mediaPath": "assets/" + digest + "." + extension,
            "mediaSha256": digest, "kind": kind, "mimeType": mime, "conditioningText": condition}

    def test_all_actual_modalities_produce_binary_records_not_caption_text(self):
        for kind in ("image", "audio", "video"):
            coverage = DatasetCoverage()
            record = _structured_record(self.pair(kind), kind, 100, coverage, self.root)
            self.assertEqual(record.kind, kind)
            self.assertEqual(record.text, "")
            self.assertTrue(Path(record.local_path).is_file())
            self.assertEqual(record.provenance["conditioning_text"], "The bird sings and moves.")
            self.assertEqual(coverage.modality_counts, {kind: 1})
            self.assertNotIn("speech_text", record.provenance)

    def test_missing_hash_wrong_mime_path_and_extra_fields_do_not_fall_back_to_text(self):
        value = self.pair()
        for patch in ({"mediaSha256": "0" * 64}, {"mediaPath": "../outside.png"}, {"mimeType": "video/mp4"}, {"unexpected": True}):
            with self.assertRaises(ValueError):
                _structured_record(dict(value, **patch), "bad", 100, DatasetCoverage(), self.root)

    def test_large_condition_uses_the_typed_jsonl_route_without_losing_media(self):
        value = self.pair(condition="A" * (INLINE_TEXT_BYTES + 128))
        path = self.root / "lesson.jsonl"
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        coverage = DatasetCoverage()
        records = list(iter_dataset_records(path, "jsonl", coverage))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].kind, "image")
        self.assertEqual(len(records[0].provenance["conditioning_text"]), INLINE_TEXT_BYTES + 128)
        self.assertEqual(coverage.modality_counts, {"image": 1})


if __name__ == "__main__":
    unittest.main()
