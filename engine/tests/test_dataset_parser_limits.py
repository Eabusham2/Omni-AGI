"""Parser-only coverage: no brain construction, training, or resource changes."""

import csv
import datetime
import errno
import gzip
import io
import json
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_core import datasets


class _RowOnlyBatch:
    """Expose native scalars while rejecting an eager whole-batch conversion."""

    def __init__(self, native):
        self.native = native
        self.schema = native.schema
        self.num_columns = native.num_columns
        self.num_rows = native.num_rows

    def column(self, index):
        return self.native.column(index)

    def to_pylist(self):
        raise AssertionError("parser attempted a whole-batch Python conversion")


class DatasetParserLimitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def records(self, path):
        coverage = datasets.DatasetCoverage()
        records = []
        self.last_parser_pause = None
        try:
            for record in datasets.iter_dataset_records(path, coverage=coverage):
                # Fixture-only materialization verifies the entire leased
                # content before advancing its iterator. Production does not.
                if record.text_payload is not None:
                    record.text = "".join(piece for piece, _ in record.text_payload.windows())
                    record.text_payload = None
                records.append(record)
        except datasets.DatasetResourcePause as error:
            self.last_parser_pause = error
        return records, coverage

    def test_csv_and_tsv_preserve_fields_above_the_library_default(self):
        previous_limit = csv.field_size_limit()
        self.addCleanup(csv.field_size_limit, previous_limit)
        # Includes Unicode, quoting, a delimiter and a physical newline. The
        # entire valid field must survive, not just its first parser window.
        large_field = '雪🙂,"quoted"\n' * 20_000
        rows = [["header", "value"], ["large", large_field], ["tail", "last"]]
        for suffix, delimiter in ((".csv", ","), (".tsv", "\t")):
            with self.subTest(suffix=suffix):
                path = self.root / ("large-fields" + suffix)
                with path.open("w", encoding="utf-8", newline="") as output:
                    csv.writer(output, delimiter=delimiter).writerows(rows)
                csv.field_size_limit(64)
                records, coverage = self.records(path)
                self.assertEqual([json.loads(record.text) for record in records], rows)
                self.assertEqual(
                    [record.name for record in records],
                    ["%s#row-%d" % (path.name, index + 1) for index in range(3)],
                )
                self.assertEqual(coverage.processed_records, 3)
                self.assertEqual(coverage.rejected_records, 0)
                self.assertTrue(coverage.as_dict()["complete"])
                repeated, repeated_coverage = self.records(path)
                self.assertEqual(repeated, records)
                self.assertTrue(repeated_coverage.as_dict()["complete"])

    def test_csv_runtime_integer_overflow_finds_the_exact_supported_boundary(self):
        accepted = []
        maximum = 255

        def field_limit(candidate):
            if candidate > maximum:
                raise OverflowError("narrow fixture C integer")
            accepted.append(candidate)
            return 0

        with patch.object(
            datasets.csv, "field_size_limit", side_effect=field_limit
        ) as setter:
            datasets._configure_csv_field_limit()
        self.assertEqual(setter.call_args_list[0].args, (sys.maxsize,))
        self.assertEqual(accepted[-1], maximum)

    def test_columnar_scalar_conversion_is_lazy_and_preserves_nested_values(self):
        converted = []
        values = [
            {"text": "first", "details": {"items": [1, None]}},
            {"text": "second", "details": {"items": [2, 3]}},
        ]

        class Scalar:
            def __init__(self, key, row_index):
                self.key, self.row_index = key, row_index

            def as_py(self):
                converted.append((self.row_index, self.key))
                return values[self.row_index][self.key]

        class Column:
            def __init__(self, key):
                self.key = key

            def __getitem__(self, row_index):
                return Scalar(self.key, row_index)

        class Batch:
            schema = SimpleNamespace(names=["text", "details"])
            num_columns = 2
            num_rows = 2

            def column(self, index):
                return Column(self.schema.names[index])

            def to_pylist(self):
                raise AssertionError("whole-batch conversion is forbidden")

        iterator = datasets._iter_columnar_rows(Batch())
        self.assertEqual(converted, [])
        self.assertEqual(next(iterator), values[0])
        self.assertEqual(converted, [(0, "text"), (0, "details")])
        self.assertEqual(next(iterator), values[1])
        self.assertEqual(converted[-2:], [(1, "text"), (1, "details")])
        self.assertEqual(list(iterator), [])

    def test_incremental_serialized_bytes_match_previous_json_accounting(self):
        for value in (
            None,
            {"text": "雪🙂" * 40_000, "id": "quoted\"\n"},
            {"nested": [1, None, True, {"value": "\u0000\\tab\t"}]},
            {"timestamp": datetime.datetime(2026, 9, 29, 12, 3), "binary": b"raw"},
        ):
            expected = len(
                json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
            )
            original_dumps = json.dumps
            def bounded_dump(piece, **options):
                self.assertNotIsInstance(piece, (dict, list, tuple))
                if isinstance(piece, str):
                    self.assertLessEqual(len(piece), datasets.TEXT_BLOCK_CHARS)
                return original_dumps(piece, **options)
            with patch.object(
                datasets.json, "dumps", side_effect=bounded_dump
            ):
                self.assertEqual(datasets._serialized_row_bytes(value), expected)

    def test_parquet_and_arrow_do_not_convert_complete_batches_to_python(self):
        try:
            import pyarrow as arrow
            import pyarrow.ipc as ipc
            import pyarrow.parquet as parquet
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")

        row_count = 513
        values = [
            {
                "text": "row-%04d-%s"
                % (index, "雪🙂" * (20_000 if index == 257 else 1)),
                "id": str(index),
                "details": {"items": [index, None]},
            }
            for index in range(row_count)
        ]
        table = arrow.Table.from_pylist(values)
        path = self.root / "rows.parquet"
        parquet.write_table(table, path, row_group_size=113)
        native = parquet.ParquetFile(path)

        class ParquetReader:
            metadata = native.metadata
            schema_arrow = native.schema_arrow

            def iter_batches(self, batch_size, **options):
                for batch in native.iter_batches(batch_size=batch_size, **options):
                    yield _RowOnlyBatch(batch)

        with patch.object(parquet, "ParquetFile", return_value=ParquetReader()):
            records, coverage = self.records(path)
            repeated, repeated_coverage = self.records(path)
        self.assertEqual(
            [record.text for record in records], [row["text"] for row in values]
        )
        self.assertEqual(
            [record.name for record in records],
            [
                "rows.parquet#batch-%d-row-%d" % (index // 256 + 1, index % 256 + 1)
                for index in range(row_count)
            ],
        )
        self.assertEqual(
            [record.bytes_read for record in records],
            [len(json.dumps(row, ensure_ascii=False).encode("utf-8")) for row in values],
        )
        self.assertEqual(repeated, records)
        self.assertEqual(records[:257] + repeated[257:], records)
        self.assertTrue(coverage.as_dict()["complete"])
        self.assertTrue(repeated_coverage.as_dict()["complete"])

        original_open_file, original_open_stream = ipc.open_file, ipc.open_stream

        class FileReader:
            def __init__(self, source):
                self.native = original_open_file(source)
                self.num_record_batches = self.native.num_record_batches

            def get_batch(self, index):
                return _RowOnlyBatch(self.native.get_batch(index))

        class StreamReader:
            def __init__(self, source):
                self.native = original_open_stream(source)

            def __iter__(self):
                for batch in self.native:
                    yield _RowOnlyBatch(batch)

            def read_next_batch(self):
                return _RowOnlyBatch(self.native.read_next_batch())

        for suffix, writer_factory, opener, reader_type in (
            (".arrow", ipc.new_file, "open_file", FileReader),
            (".ipc", ipc.new_stream, "open_stream", StreamReader),
        ):
            with self.subTest(suffix=suffix):
                path = self.root / ("rows" + suffix)
                with path.open("wb") as output:
                    with writer_factory(output, table.schema) as writer:
                        writer.write_batch(table.to_batches()[0])
                with patch.object(ipc, opener, side_effect=reader_type):
                    records, coverage = self.records(path)
                    repeated, repeated_coverage = self.records(path)
                self.assertEqual(
                    [record.text for record in records], [row["text"] for row in values]
                )
                self.assertEqual(
                    [record.name for record in records],
                    [
                        "%s#batch-1-row-%d" % (path.name, index + 1)
                        for index in range(row_count)
                    ],
                )
                self.assertEqual(repeated, records)
                self.assertTrue(coverage.as_dict()["complete"])
                self.assertTrue(repeated_coverage.as_dict()["complete"])

    def test_resource_failures_before_or_after_prefix_are_incomplete_and_replayable(self):
        path = self.root / "resume.jsonl"
        path.write_text(
            "\n".join(
                json.dumps({"text": value}) for value in ("first", "second", "tail")
            ),
            encoding="utf-8",
        )
        original = datasets._iter_jsonl
        expected, _ = self.records(path)
        errors = [
            MemoryError(),
            OSError(errno.ENOMEM, "injected memory pressure"),
            OSError(errno.ENOSPC, "injected full spool volume"),
        ]
        if hasattr(errno, "EDQUOT"):
            errors.append(OSError(errno.EDQUOT, "injected spool quota"))
        for error in errors:
            for committed in (0, 1):
                with self.subTest(error=type(error).__name__, committed=committed):

                    def failed_parser(source, coverage):
                        iterator = original(source, coverage)
                        try:
                            for _ in range(committed):
                                yield next(iterator)
                        finally:
                            iterator.close()
                        raise error

                    with patch.object(datasets, "_iter_jsonl", failed_parser):
                        records, coverage = self.records(path)
                    self.assertEqual(records, expected[:committed])
                    self.assertEqual(coverage.processed_records, committed)
                    self.assertEqual(coverage.completed_files, 0)
                    self.assertEqual(coverage.rejected_files, 0)
                    self.assertTrue(coverage.traversal_incomplete)
                    self.assertFalse(coverage.as_dict()["complete"])
                    self.assertIsNotNone(self.last_parser_pause)
                    self.assertTrue(str(self.last_parser_pause))
                    replayed, replay_coverage = self.records(path)
                    self.assertEqual(records + replayed[committed:], expected)
                    self.assertTrue(replay_coverage.as_dict()["complete"])

    def test_nested_archive_resource_failures_stop_without_claiming_completion(self):
        payload = b"trainable member body"
        zip_path = self.root / "member.zip"
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("member.txt", payload)
        tar_path = self.root / "member.tar"
        with tarfile.open(tar_path, "w") as archive:
            member = tarfile.TarInfo("member.txt")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        compressed_path = self.root / "member.txt.gz"
        with gzip.open(compressed_path, "wb") as output:
            output.write(payload)
        office_path = self.root / "member.docx"
        with zipfile.ZipFile(office_path, "w") as archive:
            archive.writestr(
                "word/document.xml",
                '<w:document xmlns:w="urn:word"><w:body><w:p><w:r>'
                "<w:t>trainable member body</w:t></w:r></w:p></w:body></w:document>",
            )
        for path in (zip_path, tar_path, compressed_path, office_path):
            with self.subTest(source=path.suffix):
                parser = (
                    "_iter_streamed_xml" if path == office_path else "_iter_text_stream"
                )
                expected, expected_coverage = self.records(path)
                self.assertEqual(len(expected), 1)
                self.assertTrue(expected_coverage.as_dict()["complete"])
                for error in (
                    MemoryError("injected member pressure"),
                    OSError(errno.ENOSPC, "injected full member spool"),
                ):
                    with patch.object(datasets, parser, side_effect=error):
                        records, coverage = self.records(path)
                    self.assertEqual(records, [])
                    self.assertTrue(coverage.traversal_incomplete)
                    self.assertFalse(coverage.as_dict()["complete"])
                    repeated, repeated_coverage = self.records(path)
                    self.assertEqual(repeated, expected)
                    self.assertTrue(repeated_coverage.as_dict()["complete"])

    def test_remote_shard_spool_failure_is_not_reported_as_completed_rejection(self):
        path = self.root / "remote.manifest.json"
        path.write_text(
            json.dumps(
                {
                    "shards": [
                        {
                            "url": "https://example.test/data.jsonl",
                            "sha256": "0" * 64,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        with patch.object(
            datasets,
            "_download_remote_shard",
            side_effect=OSError(errno.ENOSPC, "injected full shard spool"),
        ) as download:
            records, coverage = self.records(path)
        download.assert_called_once()
        self.assertEqual(records, [])
        self.assertTrue(coverage.traversal_incomplete)
        self.assertFalse(coverage.as_dict()["complete"])

    def test_arrow_header_memory_failure_is_not_retried_as_another_format(self):
        try:
            import pyarrow as arrow
            import pyarrow.ipc as ipc
        except ImportError:
            self.skipTest("pyarrow is optional in the lightweight developer environment")
        path = self.root / "header.arrow"
        with path.open("wb") as output, ipc.new_file(output, arrow.schema([("text", arrow.string())])):
            pass
        with patch.object(
            ipc,
            "open_file",
            side_effect=arrow.ArrowMemoryError("injected header pressure"),
        ), patch.object(ipc, "open_stream") as stream_fallback:
            records, coverage = self.records(path)
        stream_fallback.assert_not_called()
        self.assertEqual(records, [])
        self.assertTrue(coverage.traversal_incomplete)
        self.assertFalse(coverage.as_dict()["complete"])


if __name__ == "__main__":
    unittest.main()
