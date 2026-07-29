import hashlib
import json
import os
import sqlite3
import tarfile
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from omni_core.brain import AdaptiveBrain
from omni_core.datasets import DatasetCoverage, iter_dataset_records


class DatasetStreamingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def records(self, path: Path):
        coverage = DatasetCoverage()
        records = list(iter_dataset_records(path, coverage=coverage))
        return records, coverage

    def test_csv_jsonl_sqlite_and_manifest_visit_every_record(self):
        csv_path = self.root / "rows.csv"
        csv_path.write_text("name,value\nalpha,1\nbeta,2\n", encoding="utf-8")
        csv_records, csv_coverage = self.records(csv_path)
        self.assertEqual(len(csv_records), 3)
        self.assertEqual(csv_coverage.processed_records, 3)
        self.assertEqual(csv_coverage.rejected_records, 0)

        jsonl_path = self.root / "rows.jsonl"
        jsonl_path.write_text(
            "\n".join(json.dumps({"index": index}) for index in range(11)),
            encoding="utf-8",
        )
        jsonl_records, jsonl_coverage = self.records(jsonl_path)
        self.assertEqual(len(jsonl_records), 11)
        self.assertEqual(jsonl_coverage.processed_records, 11)

        sqlite_path = self.root / "rows.sqlite"
        connection = sqlite3.connect(sqlite_path)
        connection.execute("CREATE TABLE facts (name TEXT, value INTEGER)")
        connection.executemany(
            "INSERT INTO facts VALUES (?, ?)",
            [("fact-%d" % index, index) for index in range(9)],
        )
        connection.commit()
        connection.close()
        sqlite_records, sqlite_coverage = self.records(sqlite_path)
        self.assertEqual(len(sqlite_records), 9)
        self.assertEqual(sqlite_coverage.completed_files, 1)

        manifest_path = self.root / "training.hf.json"
        manifest_path.write_text(
            json.dumps({"data_files": ["rows.csv", "rows.jsonl"]}),
            encoding="utf-8",
        )
        manifest_records, manifest_coverage = self.records(manifest_path)
        self.assertEqual(len(manifest_records), 14)
        self.assertEqual(manifest_coverage.processed_records, 14)
        self.assertEqual(manifest_coverage.completed_files, 3)

    def test_webdataset_tar_streams_text_and_binary_modality_members(self):
        text_a = self.root / "a.txt"
        text_b = self.root / "b.json"
        image = self.root / "pixels.png"
        audio = self.root / "sample.wav"
        video = self.root / "clip.mp4"
        binary = self.root / "pixels.bin"
        text_a.write_text("first record", encoding="utf-8")
        text_b.write_text('{"second":"record"}', encoding="utf-8")
        image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
        audio.write_bytes(b"RIFF\x08\x00\x00\x00WAVEfixture")
        video.write_bytes(b"\x00\x00\x00\x18ftypmp42fixture")
        binary.write_bytes(b"\x00\xff\x00")
        archive_path = self.root / "fixture.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(text_a, arcname="0001.txt")
            archive.add(text_b, arcname="0002.json")
            archive.add(image, arcname="0003.png")
            archive.add(audio, arcname="0004.wav")
            archive.add(video, arcname="0005.mp4")
            archive.add(binary, arcname="0006.bin")
        records, coverage = self.records(archive_path)
        self.assertEqual(len(records), 5)
        self.assertEqual([record.kind for record in records[2:]], ["image", "audio", "video"])
        self.assertTrue(all(record.local_path for record in records[2:]))
        self.assertTrue(all(len(record.content_sha256) == 64 for record in records[2:]))
        self.assertEqual(coverage.shards, 1)
        self.assertEqual(coverage.discovered_files, 7)
        self.assertEqual(coverage.rejected_records, 1)
        self.assertEqual(coverage.modality_counts["image"], 1)
        self.assertEqual(coverage.modality_counts["audio"], 1)
        self.assertEqual(coverage.modality_counts["video"], 1)

    def test_binary_member_local_path_is_leased_until_iterator_advances(self):
        image_bytes = b"\x89PNG\r\n\x1a\nleased-fixture"
        image = self.root / "leased.png"
        image.write_bytes(image_bytes)
        archive_path = self.root / "leased.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(image, arcname="sample.png")

        coverage = DatasetCoverage()
        iterator = iter(iter_dataset_records(archive_path, coverage=coverage))
        record = next(iterator)
        self.assertEqual(record.kind, "image")
        self.assertIsNotNone(record.local_path)
        leased_path = Path(str(record.local_path))
        self.assertTrue(leased_path.exists())
        self.assertEqual(leased_path.read_bytes(), image_bytes)
        self.assertEqual(record.content_sha256, hashlib.sha256(image_bytes).hexdigest())
        with self.assertRaises(StopIteration):
            next(iterator)
        self.assertFalse(leased_path.exists())

    def test_remote_manifest_streams_shard_and_records_provenance(self):
        remote_path = self.root / "remote.jsonl"
        body = "\n".join(json.dumps({"index": index}) for index in range(7)) + "\n"
        remote_path.write_text(body, encoding="utf-8")
        digest = hashlib.sha256(body.encode("utf-8")).hexdigest()

        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                del format, args

        handler = partial(QuietHandler, directory=str(self.root))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        prior_local_setting = os.environ.get("OMNI_ALLOW_LOCAL_URLS")
        os.environ["OMNI_ALLOW_LOCAL_URLS"] = "1"
        try:
            remote_url = "http://127.0.0.1:%d/remote.jsonl" % server.server_port
            manifest_path = self.root / "remote.hf.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {
                                "data_files": [
                                    {
                                        "url": remote_url,
                                        "sha256": digest,
                                        "license": "fixture-only",
                                    }
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            records, coverage = self.records(manifest_path)
        finally:
            if prior_local_setting is None:
                os.environ.pop("OMNI_ALLOW_LOCAL_URLS", None)
            else:
                os.environ["OMNI_ALLOW_LOCAL_URLS"] = prior_local_setting
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        self.assertEqual(len(records), 7)
        self.assertEqual(coverage.processed_records, 7)
        self.assertEqual(coverage.rejected_records, 0)
        self.assertTrue(records[0].name.startswith(remote_url))
        self.assertEqual(records[0].provenance["requested_url"], remote_url)
        self.assertEqual(records[0].provenance["final_url"], remote_url)
        self.assertEqual(records[0].provenance["shard_sha256"], digest)
        self.assertEqual(records[0].provenance["downloaded_bytes"], len(body.encode("utf-8")))
        self.assertEqual(records[0].provenance["declared"]["license"], "fixture-only")

    def test_remote_manifest_rejects_private_network_by_default(self):
        prior_local_setting = os.environ.pop("OMNI_ALLOW_LOCAL_URLS", None)
        try:
            manifest_path = self.root / "unsafe.hf.json"
            manifest_path.write_text(
                json.dumps({"data_files": ["https://127.0.0.1/private.jsonl"]}),
                encoding="utf-8",
            )
            records, coverage = self.records(manifest_path)
        finally:
            if prior_local_setting is not None:
                os.environ["OMNI_ALLOW_LOCAL_URLS"] = prior_local_setting

        self.assertEqual(records, [])
        self.assertEqual(coverage.rejected_records, 1)
        self.assertIn("private or reserved", coverage.errors[0]["message"])

    def test_experience_chunking_has_no_sixty_four_chunk_ceiling(self):
        text = "\n\n".join(
            "Sentence %d contains a unique exhaustive training fact." % index
            for index in range(1_200)
        )
        chunks = AdaptiveBrain._experience_chunks(text)
        self.assertGreater(len(chunks), 64)
        self.assertIn("Sentence 1199", chunks[-1])

    def test_single_long_record_is_partitioned_without_losing_content(self):
        source = "x" * 12_345
        chunks = AdaptiveBrain._experience_chunks(source)
        self.assertGreater(len(chunks), 3)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(len(chunk) <= 4_000 for chunk in chunks))

    def test_parquet_when_pyarrow_is_available(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")
        path = self.root / "rows.parquet"
        parquet.write_table(
            arrow.table({"name": ["alpha", "beta", "gamma"], "value": [1, 2, 3]}),
            path,
        )
        records, coverage = self.records(path)
        self.assertEqual(len(records), 3)
        self.assertEqual(coverage.processed_records, 3)
        self.assertEqual(coverage.rejected_records, 0)


if __name__ == "__main__":
    unittest.main()
