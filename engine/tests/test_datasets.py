import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import sqlite3
import tarfile
import tempfile
import threading
import unittest
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from omni_core.brain import AdaptiveBrain
from omni_core.datasets import (
    DatasetCoverage,
    dataset_format,
    dataset_record_count_hint,
    iter_dataset_records,
    sqlite_consistent_snapshot,
    sqlite_consistent_snapshot_sha256,
)


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

    def test_desktop_media_extensions_reach_the_neural_decoder_path(self):
        for name in ("frame.avif", "photo.heic", "photo.heif"):
            self.assertEqual(dataset_format(Path(name)), "image", msg=name)
        for name in ("voice.oga", "voice.aiff", "voice.aif", "voice.wma"):
            self.assertEqual(dataset_format(Path(name)), "audio", msg=name)
        for name in ("movie.m4v", "movie.wmv", "movie.flv"):
            self.assertEqual(dataset_format(Path(name)), "video", msg=name)

    def test_rejection_diagnostics_are_sampled_without_capping_coverage(self):
        coverage = DatasetCoverage()
        for index in range(300):
            coverage.reject("row-%d" % index, "invalid fixture %d" % index)
        self.assertEqual(coverage.discovered_records, 300)
        self.assertEqual(coverage.rejected_records, 300)
        self.assertEqual(coverage.error_count, 300)
        self.assertEqual(len(coverage.errors), 256)
        self.assertTrue(coverage.errors_truncated)
        self.assertTrue(coverage.as_dict()["complete"])

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
        self.assertIsNone(
            dataset_record_count_hint(jsonl_path, "jsonl"),
            "streaming JSONL must not be double-scanned merely to draw a total",
        )

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

    def test_folder_rejects_hidden_and_incomplete_download_artifacts(self):
        folder = self.root / "folder"
        folder.mkdir()
        (folder / "records.txt").write_text("trainable record", encoding="utf-8")
        (folder / ".hidden.jsonl").write_text(
            json.dumps({"text": "must not train"}), encoding="utf-8"
        )
        (folder / "shard.parquet.incomplete").write_text(
            "unfinished download", encoding="utf-8"
        )
        hidden_cache = folder / ".cache" / "downloads"
        hidden_cache.mkdir(parents=True)
        (hidden_cache / "cached.jsonl").write_text(
            json.dumps({"text": "cache must not train"}), encoding="utf-8"
        )

        records, coverage = self.records(folder)

        self.assertEqual([record.text for record in records], ["trainable record"])
        self.assertEqual(coverage.discovered_files, 3)
        self.assertEqual(coverage.completed_files, 1)
        self.assertEqual(coverage.rejected_files, 2)
        self.assertEqual(coverage.discovered_records, 3)
        self.assertEqual(coverage.processed_records, 1)
        self.assertEqual(coverage.rejected_records, 2)
        self.assertEqual(coverage.error_count, 2)
        self.assertTrue(coverage.as_dict()["complete"])
        self.assertNotIn("cache must not train", "\n".join(record.text for record in records))

    def test_sqlite_snapshot_includes_wal_and_orders_tables_and_composite_keys(self):
        sqlite_path = self.root / "live.sqlite3"
        writer = sqlite3.connect(sqlite_path)
        try:
            self.assertEqual(writer.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE zeta (text TEXT)")
            writer.execute(
                "CREATE TABLE alpha (group_id INTEGER, item_id INTEGER, text TEXT, "
                "PRIMARY KEY (group_id, item_id))"
            )
            writer.executemany(
                "INSERT INTO alpha VALUES (?, ?, ?)",
                [(2, 1, "alpha-2-1"), (1, 2, "alpha-1-2"), (1, 1, "alpha-1-1")],
            )
            writer.executemany(
                "INSERT INTO zeta VALUES (?)",
                [("zeta-first",), ("zeta-second",)],
            )
            writer.commit()
            wal_path = Path(str(sqlite_path) + "-wal")
            self.assertTrue(wal_path.is_file())
            self.assertGreater(wal_path.stat().st_size, 0)

            main_file_digest = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()
            first_identity = sqlite_consistent_snapshot_sha256(sqlite_path)
            first, first_coverage = self.records(sqlite_path)
            self.assertEqual(
                [record.text for record in first],
                [
                    "alpha-1-1",
                    "alpha-1-2",
                    "alpha-2-1",
                    "zeta-first",
                    "zeta-second",
                ],
            )
            self.assertEqual(
                [record.name.split("#", 1)[1].split("-", 1)[0] for record in first],
                ["alpha", "alpha", "alpha", "zeta", "zeta"],
            )
            self.assertTrue(
                all(
                    record.provenance["sqlite_snapshot_sha256"] == first_identity
                    for record in first
                )
            )
            self.assertEqual(first[0].provenance["sqlite_order"], ["group_id", "item_id"])
            self.assertEqual(first[-1].provenance["sqlite_order"], ["rowid"])
            self.assertTrue(first_coverage.as_dict()["complete"])

            writer.execute("INSERT INTO alpha VALUES (0, 9, 'wal-only-new-row')")
            writer.commit()
            self.assertEqual(
                hashlib.sha256(sqlite_path.read_bytes()).hexdigest(),
                main_file_digest,
                "the fixture must prove the new committed row exists only in WAL",
            )
            second_identity = sqlite_consistent_snapshot_sha256(sqlite_path)
            second, _ = self.records(sqlite_path)
            self.assertNotEqual(second_identity, first_identity)
            self.assertEqual(second[0].text, "wal-only-new-row")
            self.assertTrue(
                all(
                    record.provenance["sqlite_snapshot_sha256"] == second_identity
                    for record in second
                )
            )
        finally:
            writer.close()

    def test_committed_desktop_sqlite_snapshot_preserves_its_exact_manifest_hash(self):
        source = self.root / "desktop-source.sqlite3"
        writer = sqlite3.connect(source)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, text TEXT)")
            writer.executemany(
                "INSERT INTO facts VALUES (?, ?)",
                [(2, "second"), (1, "first")],
            )
            writer.commit()
            with sqlite_consistent_snapshot(source) as snapshot:
                self.assertEqual(
                    hashlib.sha256(snapshot.path.read_bytes()).hexdigest(),
                    snapshot.sha256,
                )
                coverage = DatasetCoverage()
                records = list(
                    iter_dataset_records(
                        snapshot.path,
                        requested_kind="sqlite",
                        coverage=coverage,
                        _committed_sqlite_snapshot_sha256=snapshot.sha256,
                    )
                )
                self.assertEqual([record.text for record in records], ["first", "second"])
                self.assertTrue(coverage.as_dict()["complete"])
                self.assertTrue(
                    all(
                        record.provenance["sqlite_snapshot_sha256"]
                        == snapshot.sha256
                        for record in records
                    )
                )
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    list(
                        iter_dataset_records(
                            snapshot.path,
                            requested_kind="sqlite",
                            coverage=DatasetCoverage(),
                            _committed_sqlite_snapshot_sha256="0" * 64,
                        )
                    )
        finally:
            writer.close()

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
        self.assertEqual(coverage.completed_files, 6)
        self.assertEqual(coverage.rejected_files, 1)
        self.assertEqual(coverage.rejected_records, 1)
        self.assertEqual(coverage.modality_counts["image"], 1)
        self.assertEqual(coverage.modality_counts["audio"], 1)
        self.assertEqual(coverage.modality_counts["video"], 1)
        self.assertTrue(coverage.as_dict()["complete"])

    def test_corrupt_file_and_unsupported_member_complete_as_explicit_rejections(self):
        corrupt = self.root / "corrupt.tar"
        corrupt.write_bytes(b"not a tar, zip, or readable archive")
        corrupt_records, corrupt_coverage = self.records(corrupt)
        self.assertEqual(corrupt_records, [])
        self.assertEqual(
            corrupt_coverage.as_dict(),
            {
                "discoveredFiles": 1,
                "completedFiles": 0,
                "processedFiles": 0,
                "rejectedFiles": 1,
                "discoveredRecords": 1,
                "processedRecords": 0,
                "rejectedRecords": 1,
                "processedBytes": 0,
                "shards": 1,
                "modalityCounts": {},
                "errors": [
                    {
                        "source": str(corrupt.resolve()),
                        "message": corrupt_coverage.errors[0]["message"],
                    }
                ],
                "errorCount": 1,
                "errorsTruncated": False,
                "complete": True,
            },
        )

        unsupported_source = self.root / "opaque.bin"
        unsupported_source.write_bytes(b"\x00\xff\x00\xff")
        archive_path = self.root / "unsupported-member.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(unsupported_source, arcname="opaque.bin")
        member_records, member_coverage = self.records(archive_path)
        self.assertEqual(member_records, [])
        self.assertEqual(member_coverage.discovered_files, 2)
        self.assertEqual(member_coverage.completed_files, 1)
        self.assertEqual(member_coverage.rejected_files, 1)
        self.assertEqual(member_coverage.discovered_records, 1)
        self.assertEqual(member_coverage.processed_records, 0)
        self.assertEqual(member_coverage.rejected_records, 1)
        self.assertTrue(member_coverage.as_dict()["complete"])
        self.assertIn(
            "unsupported binary archive member",
            member_coverage.errors[0]["message"],
        )

    def test_office_and_opendocument_containers_stream_every_content_part(self):
        long_docx = "".join(
            "<w:p><w:r><w:t>docx-row-%d</w:t></w:r></w:p>" % index
            for index in range(4_000)
        )
        fixtures = {
            ".docx": {
                "word/document.xml": (
                    '<w:document xmlns:w="urn:word"><w:body>'
                    + long_docx
                    + "<w:p><w:r><w:t>docx-tail-sentinel</w:t></w:r></w:p>"
                    + "</w:body></w:document>"
                )
            },
            ".pptx": {
                "ppt/slides/slide1.xml": (
                    '<p:sld xmlns:p="urn:presentation" xmlns:a="urn:drawing">'
                    "<a:p><a:r><a:t>pptx-tail-sentinel</a:t></a:r></a:p>"
                    "</p:sld>"
                )
            },
            ".xlsx": {
                "xl/sharedStrings.xml": (
                    '<sst xmlns="urn:spreadsheet"><si><t>xlsx-tail-sentinel</t></si></sst>'
                ),
                "xl/worksheets/sheet1.xml": (
                    '<worksheet xmlns="urn:spreadsheet"><sheetData><row>'
                    '<c t="s"><v>0</v></c><c><v>42</v></c>'
                    "</row></sheetData></worksheet>"
                ),
            },
            ".odt": {
                "content.xml": (
                    '<office:document-content xmlns:office="urn:office" '
                    'xmlns:text="urn:text"><office:body><office:text>'
                    "<text:p>odt-tail-sentinel</text:p>"
                    "</office:text></office:body></office:document-content>"
                )
            },
            ".ods": {
                "content.xml": (
                    '<office:document-content xmlns:office="urn:office" '
                    'xmlns:text="urn:text" xmlns:table="urn:table">'
                    "<office:body><office:spreadsheet><table:table><table:table-row>"
                    '<table:table-cell office:value-type="float" office:value="42">'
                    "<text:p>ods-tail-sentinel</text:p></table:table-cell>"
                    "</table:table-row></table:table></office:spreadsheet></office:body>"
                    "</office:document-content>"
                )
            },
            ".odp": {
                "content.xml": (
                    '<office:document-content xmlns:office="urn:office" '
                    'xmlns:text="urn:text"><office:body><office:presentation>'
                    "<text:p>odp-tail-sentinel</text:p>"
                    "</office:presentation></office:body></office:document-content>"
                )
            },
        }

        for suffix, members in fixtures.items():
            with self.subTest(suffix=suffix):
                path = self.root / ("fixture" + suffix)
                with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    for member_name, xml in members.items():
                        archive.writestr(member_name, xml.encode("utf-8"))
                    archive.writestr("ignored/binary.bin", b"\x00\xff")
                records, coverage = self.records(path)
                combined = "\n".join(record.text for record in records)
                self.assertIn(suffix[1:] + "-tail-sentinel", combined)
                self.assertEqual(coverage.rejected_records, 0, coverage.errors)
                self.assertEqual(coverage.shards, 1)
                self.assertGreaterEqual(coverage.completed_files, 2)
                if suffix == ".docx":
                    self.assertGreater(len(records), 1)
                    self.assertIn("docx-row-3999", combined)
                if suffix in {".xlsx", ".ods"}:
                    self.assertIn("42", combined)

    def test_standalone_gzip_bzip2_and_xz_stream_without_a_byte_cap(self):
        payload = "\n".join(
            ["compressed-row-%d" % index for index in range(6_000)]
            + ["compressed-tail-sentinel"]
        )
        for suffix, opener in (
            (".gz", gzip.open),
            (".bz2", bz2.open),
            (".xz", lzma.open),
        ):
            with self.subTest(suffix=suffix):
                path = self.root / ("corpus.txt" + suffix)
                with opener(path, "wt", encoding="utf-8", newline="") as stream:
                    stream.write(payload)
                records, coverage = self.records(path)
                combined = "\n".join(record.text for record in records)
                self.assertGreater(len(records), 1)
                self.assertIn("compressed-row-0", combined)
                self.assertIn("compressed-row-5999", combined)
                self.assertIn("compressed-tail-sentinel", combined)
                self.assertEqual(coverage.rejected_records, 0, coverage.errors)
                self.assertEqual(coverage.shards, 1)
                self.assertEqual(coverage.completed_files, 2)

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
        # Use exact bytes so Windows text-mode newline conversion cannot make
        # the served shard differ from the manifest checksum.
        remote_path.write_bytes(body.encode("utf-8"))
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

        self.assertEqual(len(records), 7, coverage.errors)
        self.assertEqual(coverage.processed_records, 7)
        self.assertEqual(coverage.rejected_records, 0)
        self.assertTrue(records[0].name.startswith(remote_url))
        self.assertEqual(records[0].provenance["requested_url"], remote_url)
        self.assertEqual(records[0].provenance["final_url"], remote_url)
        self.assertEqual(records[0].provenance["shard_sha256"], digest)
        self.assertEqual(records[0].provenance["downloaded_bytes"], len(body.encode("utf-8")))
        self.assertEqual(records[0].provenance["declared"]["license"], "fixture-only")

    def test_remote_manifest_requires_and_verifies_declared_sha256(self):
        missing_hash = self.root / "missing-hash.hf.json"
        missing_hash.write_text(
            json.dumps({"data_files": ["https://example.com/mutable.jsonl"]}),
            encoding="utf-8",
        )
        missing_records, missing_coverage = self.records(missing_hash)
        self.assertEqual(missing_records, [])
        self.assertIn("requires a declared sha256", missing_coverage.errors[0]["message"])

        remote_path = self.root / "changed.jsonl"
        original = b'{"text":"original"}\n'
        changed = b'{"text":"changed after manifest"}\n'
        remote_path.write_bytes(changed)
        expected = hashlib.sha256(original).hexdigest()

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
            remote_url = "http://127.0.0.1:%d/changed.jsonl" % server.server_port
            changed_manifest = self.root / "changed-remote.hf.json"
            changed_manifest.write_text(
                json.dumps(
                    {"data_files": [{"url": remote_url, "sha256": expected}]}
                ),
                encoding="utf-8",
            )
            changed_records, changed_coverage = self.records(changed_manifest)
        finally:
            if prior_local_setting is None:
                os.environ.pop("OMNI_ALLOW_LOCAL_URLS", None)
            else:
                os.environ["OMNI_ALLOW_LOCAL_URLS"] = prior_local_setting
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        self.assertEqual(changed_records, [])
        self.assertIn("checksum mismatch", changed_coverage.errors[0]["message"])

    def test_remote_manifest_rejects_private_network_by_default(self):
        prior_local_setting = os.environ.pop("OMNI_ALLOW_LOCAL_URLS", None)
        try:
            manifest_path = self.root / "unsafe.hf.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "data_files": [
                            {
                                "url": "https://127.0.0.1/private.jsonl",
                                "sha256": "0" * 64,
                            }
                        ]
                    }
                ),
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
        self.assertEqual(dataset_record_count_hint(path, "parquet"), 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(coverage.processed_records, 3)
        self.assertEqual(coverage.rejected_records, 0)

    def test_parquet_footer_detects_a_silently_short_record_stream(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")

        path = self.root / "short-reader.parquet"
        parquet.write_table(
            arrow.table({"text": ["first", "second", "third"]}), path
        )
        actual = parquet.ParquetFile(path)

        class ShortReader:
            metadata = actual.metadata
            schema_arrow = actual.schema_arrow

            def iter_batches(self, batch_size):
                yield next(actual.iter_batches(batch_size=2))

        with patch.object(parquet, "ParquetFile", return_value=ShortReader()):
            records, coverage = self.records(path)
        self.assertEqual([record.text for record in records], ["first", "second"])
        self.assertEqual(coverage.processed_records, 2)
        self.assertEqual(coverage.discovered_records, 3)
        self.assertTrue(coverage.traversal_incomplete)
        self.assertFalse(coverage.as_dict()["complete"])
        with self.assertRaisesRegex(RuntimeError, "dataset traversal stopped"):
            AdaptiveBrain._require_complete_ingestion_coverage(
                coverage, policy="pretrain", resolved_kind="parquet", record_count_hint=3
            )

    def test_jsonl_io_failure_after_valid_prefix_is_not_complete(self):
        path = self.root / "partial.jsonl"
        path.write_text(
            '{"text":"first"}\n{"text":"second"}\n', encoding="utf-8"
        )
        from omni_core import datasets as dataset_module

        original = dataset_module._iter_jsonl

        def failed_after_first(source, coverage):
            iterator = original(source, coverage)
            try:
                yield next(iterator)
            finally:
                iterator.close()
            raise OSError("simulated mid-stream read failure")

        with patch.object(dataset_module, "_iter_jsonl", failed_after_first):
            records, coverage = self.records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(coverage.processed_records, 1)
        self.assertTrue(coverage.traversal_incomplete)
        self.assertFalse(coverage.as_dict()["complete"])
        with self.assertRaisesRegex(RuntimeError, "dataset traversal stopped"):
            AdaptiveBrain._require_complete_ingestion_coverage(
                coverage, policy="encode", resolved_kind="jsonl", record_count_hint=None
            )

    def test_large_parquet_and_arrow_batches_visit_every_row_in_order(self):
        try:
            import pyarrow as arrow
            import pyarrow.ipc as ipc
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")

        row_count = 1_037
        values = ["columnar-row-%04d" % index for index in range(row_count)]
        table = arrow.table(
            {
                "text": values,
                "id": ["id-%04d" % index for index in range(row_count)],
            }
        )
        parquet_path = self.root / "large-rows.parquet"
        parquet.write_table(table, parquet_path, row_group_size=113)
        parquet_records, parquet_coverage = self.records(parquet_path)
        self.assertEqual(dataset_record_count_hint(parquet_path, "parquet"), row_count)
        self.assertEqual(len(parquet_records), row_count)
        self.assertEqual(parquet_records[0].text, values[0])
        self.assertEqual(parquet_records[-1].text, values[-1])
        self.assertEqual(parquet_coverage.processed_records, row_count)
        self.assertEqual(parquet_coverage.rejected_records, 0)
        self.assertTrue(parquet_coverage.as_dict()["complete"])
        repeated_records, repeated_coverage = self.records(parquet_path)
        self.assertEqual(
            [(record.name, record.text) for record in repeated_records],
            [(record.name, record.text) for record in parquet_records],
        )
        # A persisted record cursor can replay and validate the prefix, then
        # continue at the first uncommitted record without a gap or duplicate.
        committed_records = 513
        resumed = parquet_records[:committed_records] + repeated_records[committed_records:]
        self.assertEqual(
            [(record.name, record.text) for record in resumed],
            [(record.name, record.text) for record in parquet_records],
        )
        self.assertTrue(repeated_coverage.as_dict()["complete"])

        truncated_path = self.root / "truncated.parquet"
        truncated_path.write_bytes(parquet_path.read_bytes()[:-4])
        truncated_records, truncated_coverage = self.records(truncated_path)
        self.assertEqual(truncated_records, [])
        self.assertEqual(truncated_coverage.rejected_files, 1)
        self.assertEqual(truncated_coverage.rejected_records, 1)
        self.assertTrue(truncated_coverage.as_dict()["complete"])

        for suffix, writer_factory in (
            (".arrow", ipc.new_file),
            (".ipc", ipc.new_stream),
        ):
            with self.subTest(suffix=suffix):
                arrow_path = self.root / ("large-rows" + suffix)
                with arrow_path.open("wb") as output:
                    with writer_factory(output, table.schema) as writer:
                        for batch in table.to_batches(max_chunksize=127):
                            writer.write_batch(batch)
                arrow_records, arrow_coverage = self.records(arrow_path)
                self.assertEqual(len(arrow_records), row_count)
                self.assertEqual(arrow_records[0].text, values[0])
                self.assertEqual(arrow_records[-1].text, values[-1])
                self.assertEqual(arrow_coverage.processed_records, row_count)
                self.assertEqual(arrow_coverage.rejected_records, 0)
                self.assertTrue(arrow_coverage.as_dict()["complete"])

    def test_large_webdataset_tar_visits_every_member_without_a_member_ceiling(self):
        member_count = 513
        archive_path = self.root / "large-webdataset.tar"
        with tarfile.open(archive_path, "w") as archive:
            for index in range(member_count):
                payload = ("webdataset-row-%04d" % index).encode("utf-8")
                info = tarfile.TarInfo("records/%04d.txt" % index)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

        records, coverage = self.records(archive_path)
        self.assertEqual(len(records), member_count)
        self.assertEqual(records[0].text, "webdataset-row-0000")
        self.assertEqual(records[-1].text, "webdataset-row-0512")
        self.assertEqual(coverage.processed_records, member_count)
        self.assertEqual(coverage.rejected_records, 0)
        self.assertEqual(coverage.discovered_files, member_count + 1)
        self.assertEqual(coverage.completed_files, member_count + 1)
        self.assertTrue(coverage.as_dict()["complete"])

    def test_archive_member_error_after_valid_prefix_cannot_complete(self):
        from omni_core import datasets as dataset_module

        original = dataset_module._iter_text_stream

        def failed_member(stream, name, coverage, chunk_chars=32_768):
            if name.endswith("broken.txt"):
                prefix = original(stream, name, coverage, chunk_chars=4)
                try:
                    yield next(prefix)
                finally:
                    prefix.close()
                raise OSError("simulated member read failure")
            yield from original(stream, name, coverage, chunk_chars=chunk_chars)

        for extension in ("zip", "tar"):
            with self.subTest(extension=extension):
                archive_path = self.root / ("partial-members." + extension)
                if extension == "zip":
                    with zipfile.ZipFile(archive_path, "w") as archive:
                        archive.writestr("broken.txt", "broken member tail")
                        archive.writestr("good.txt", "good member")
                else:
                    with tarfile.open(archive_path, "w") as archive:
                        for name, value in (
                            ("broken.txt", b"broken member tail"),
                            ("good.txt", b"good member"),
                        ):
                            member = tarfile.TarInfo(name)
                            member.size = len(value)
                            archive.addfile(member, io.BytesIO(value))
                with patch.object(dataset_module, "_iter_text_stream", failed_member):
                    records, coverage = self.records(archive_path)
                self.assertEqual(len(records), 2)
                self.assertEqual(coverage.rejected_files, 1)
                self.assertTrue(coverage.traversal_incomplete)
                self.assertFalse(coverage.as_dict()["complete"])
                with self.assertRaisesRegex(RuntimeError, "dataset traversal stopped"):
                    AdaptiveBrain._require_complete_ingestion_coverage(
                        coverage,
                        policy="pretrain",
                        resolved_kind="archive",
                        record_count_hint=None,
                    )

    def test_hugging_face_content_schemas_do_not_train_row_metadata(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")

        fineweb = self.root / "fineweb.parquet"
        parquet.write_table(
            arrow.table(
                {
                    "text": ["The trainable document body."],
                    "id": ["doc-1"],
                    "url": ["https://example.test/source"],
                    "score": [5.0],
                }
            ),
            fineweb,
        )
        records, coverage = self.records(fineweb)
        self.assertEqual([record.text for record in records], ["The trainable document body."])
        self.assertEqual(records[0].provenance["selectedField"], "text")
        self.assertEqual(records[0].provenance["id"], "doc-1")
        self.assertNotIn("doc-1", records[0].text)
        self.assertTrue(coverage.as_dict()["complete"])

        metadata_only = self.root / "python-edu.parquet"
        parquet.write_table(
            arrow.table(
                {
                    "blob_id": ["abc"],
                    "repo_name": ["owner/repo"],
                    "path": ["src/example.py"],
                    "length_bytes": [123],
                    "score": [4.0],
                }
            ),
            metadata_only,
        )
        rejected, rejected_coverage = self.records(metadata_only)
        self.assertEqual(rejected, [])
        self.assertEqual(rejected_coverage.discovered_records, 1)
        self.assertEqual(rejected_coverage.rejected_records, 1)
        self.assertIn("metadata-only", rejected_coverage.errors[0]["message"])
        self.assertTrue(rejected_coverage.as_dict()["complete"])

    def test_large_metadata_only_parquet_is_classified_from_schema_without_error_growth(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")

        path = self.root / "python-edu-large.parquet"
        rows = 4_097
        parquet.write_table(
            arrow.table(
                {
                    "blob_id": ["blob-%d" % index for index in range(rows)],
                    "repo_name": ["owner/repo"] * rows,
                    "path": ["src/example.py"] * rows,
                    "length_bytes": [123] * rows,
                    "score": [4.0] * rows,
                }
            ),
            path,
        )

        records, coverage = self.records(path)
        self.assertEqual(records, [])
        self.assertEqual(coverage.discovered_records, rows)
        self.assertEqual(coverage.rejected_records, rows)
        self.assertEqual(coverage.error_count, rows)
        self.assertEqual(len(coverage.errors), 1)
        self.assertEqual(coverage.errors[0]["count"], rows)
        self.assertIn("referenced source/blob payload", coverage.errors[0]["message"])
        self.assertTrue(coverage.as_dict()["complete"])

    def test_typed_dialogue_excludes_system_text_and_marks_brain_spans(self):
        path = self.root / "dialogue.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt_id": "sample-1",
                    "messages": [
                        {"role": "system", "content": "hidden persona text"},
                        {"role": "user", "content": "Can you explain rain?"},
                        {"role": "assistant", "content": "Rain begins when water condenses."},
                    ],
                }
            ),
            encoding="utf-8",
        )

        records, coverage = self.records(path)
        self.assertEqual(len(records), 1)
        self.assertNotIn("hidden persona text", records[0].text)
        self.assertEqual(
            records[0].text,
            "human: Can you explain rain?\nbrain: Rain begins when water condenses.",
        )
        spans = records[0].provenance["assistantSpans"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(records[0].text[spans[0]["start"] : spans[0]["end"]],
                         "Rain begins when water condenses.")
        self.assertEqual(records[0].provenance["excludedRoles"], {"system": 1})
        self.assertEqual(
            records[0].provenance["dialoguePairs"],
            [
                {
                    "human": "Can you explain rain?",
                    "brain": "Rain begins when water condenses.",
                }
            ],
        )
        self.assertTrue(coverage.as_dict()["complete"])

    def test_ultrachat_parquet_becomes_typed_dialogue_without_persona_text(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")

        path = self.root / "ultrachat.parquet"
        parquet.write_table(
            arrow.Table.from_pylist(
                [
                    {
                        "prompt": "Duplicate prompt metadata must not be trained.",
                        "prompt_id": "dialogue-1",
                        "messages": [
                            {"role": "system", "content": "Adopt a hidden persona."},
                            {"role": "user", "content": "What is rain?"},
                            {
                                "role": "assistant",
                                "content": "Rain is liquid water falling from clouds.",
                            },
                        ],
                    }
                ]
            ),
            path,
        )

        records, coverage = self.records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].text,
            "human: What is rain?\nbrain: Rain is liquid water falling from clouds.",
        )
        self.assertNotIn("hidden persona", records[0].text.lower())
        self.assertNotIn("Duplicate prompt metadata", records[0].text)
        self.assertEqual(records[0].provenance["selectedField"], "messages")
        self.assertEqual(records[0].provenance["excludedRoles"], {"system": 1})
        self.assertEqual(coverage.processed_records, 1)
        self.assertTrue(coverage.as_dict()["complete"])

    def test_system_only_dialogue_is_rejected_instead_of_serialized(self):
        path = self.root / "system-only.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt_id": "persona-only",
                    "messages": [
                        {"role": "system", "content": "Never train this persona text."}
                    ],
                }
            ),
            encoding="utf-8",
        )

        records, coverage = self.records(path)
        self.assertEqual(records, [])
        self.assertEqual(coverage.discovered_records, 1)
        self.assertEqual(coverage.rejected_records, 1)
        self.assertIn("no trainable human or brain", coverage.errors[0]["message"])
        self.assertTrue(coverage.as_dict()["complete"])


if __name__ == "__main__":
    unittest.main()
