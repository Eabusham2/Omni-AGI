"""Parser/spool/source-route contracts without importing a neural package."""

import ast
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import tracemalloc
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch


ENGINE = Path(__file__).resolve().parents[1]
PACKAGE = "omni_giant_record_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ENGINE / "omni_core")]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, ENGINE / "omni_core" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


spools = load("text_spool")
admission = load("columnar_admission")
datasets = load("datasets")
schema = load("ingestion_schedule_v3")


def source_method(name, globals_=None):
    source = ast.parse((ENGINE / "omni_core" / "brain.py").read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = {"hashlib": hashlib, "json": json, "Mapping": Mapping, **(globals_ or {})}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), name, "exec"), namespace)
    return namespace[name], method, namespace


class GiantRecordSpoolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def consume(self, path):
        coverage = datasets.DatasetCoverage()
        values = []
        with spools.parser_admission(lambda *_: None):
            for record in datasets.iter_dataset_records(path, coverage=coverage):
                text = record.text
                if record.text_payload is not None:
                    text = "".join(piece for piece, _ in record.text_payload.windows())
                values.append((text, record.name, record.bytes_read))
        return values, coverage

    def test_csv_rows_keep_quoted_newlines_escaped_quotes_empty_fields_and_physical_bytes(self):
        rows = [["long quoted\n" + ('a"é' * 45_000), "tail"], ["", "end"], [], ["", ""]]
        for delimiter, suffix in ((",", ".csv"), ("\t", ".tsv")):
            path = self.root / ("rows" + suffix)
            with path.open("w", encoding="utf-8-sig", newline="") as output:
                csv.writer(output, delimiter=delimiter).writerows(rows)
            with patch.object(datasets.csv, "reader", side_effect=AssertionError("no giant row list")):
                records, coverage = self.consume(path)
            self.assertEqual([value[0] for value in records], [json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows])
            self.assertEqual(sum(value[2] for value in records), path.stat().st_size)
            self.assertEqual(coverage.processed_bytes, path.stat().st_size)
            self.assertTrue(coverage.as_dict()["complete"])
            self.assertEqual(coverage.processed_records, len(rows))

    def test_giant_jsonl_selects_actual_text_without_full_line_json_loads_and_leases_cleanup(self):
        content = "  " + ("Aé😀\\\"\n" * 30_000) + "\x00 tail  "
        path = self.root / "rows.jsonl"
        path.write_text(json.dumps({"id": 7, "text": content, "score": 1.0}, ensure_ascii=False) + "\n", encoding="utf-8")
        coverage = datasets.DatasetCoverage()
        iterator = datasets.iter_dataset_records(path, coverage=coverage)
        with patch.object(datasets.json, "loads", side_effect=AssertionError("no whole giant JSON load")):
            record = next(iterator)
            self.assertEqual(record.text, "")
            self.assertTrue(record.text_payload.path)
            lease_path = record.text_payload.path
            expected = content.replace("\x00", "").strip()
            digest = hashlib.sha256()
            observed = 0
            for piece, end in record.text_payload.windows():
                self.assertLessEqual(len(piece), spools.LEARNING_WINDOW_CHARS)
                encoded = piece.encode("utf-8")
                digest.update(encoded)
                observed += len(encoded)
                self.assertEqual(end, observed)
            self.assertEqual(digest.hexdigest(), hashlib.sha256(expected.encode("utf-8")).hexdigest())
            self.assertEqual(record.text_payload.sha256, digest.hexdigest())
            self.assertEqual(record.bytes_read, path.stat().st_size)
            with self.assertRaises(StopIteration):
                next(iterator)
        self.assertFalse(os.path.exists(lease_path))
        self.assertEqual(coverage.discovered_records, 1)
        self.assertEqual(coverage.processed_records, 1)
        self.assertTrue(coverage.as_dict()["complete"])

    def test_giant_scalar_parser_peak_does_not_scale_to_a_whole_field_copy(self):
        path = self.root / "memory.jsonl"
        block = "x" * 32_768
        with path.open("w", encoding="utf-8") as output:
            output.write('{"text":"')
            for _ in range(96): output.write(block)
            output.write('"}\n')
        coverage = datasets.DatasetCoverage()
        tracemalloc.start()
        try:
            for record in datasets.iter_dataset_records(path, coverage=coverage):
                self.assertEqual(record.text, "")
                self.assertEqual(record.text_payload.bytes, 96 * len(block))
                for piece, _ in record.text_payload.windows():
                    self.assertLessEqual(len(piece), 4_000)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The fixture is 3 MiB; parsing never creates a row/field-sized Python
        # str/list/UTF8 copy. This observes Python allocations, not native Arrow.
        self.assertLess(peak, 2 * 1024 * 1024)
        self.assertTrue(coverage.as_dict()["complete"])

    def test_giant_json_array_is_serialized_incrementally_without_list_materialization(self):
        path = self.root / "list.jsonl"
        path.write_text("[" + ",".join(str(index) for index in range(30_000)) + "]\n", encoding="utf-8")
        records, coverage = self.consume(path)
        self.assertEqual(records[0][0], path.read_text().strip())
        self.assertEqual(coverage.processed_records, 1)
        self.assertEqual(coverage.processed_bytes, path.stat().st_size)

    def test_plain_json_preserves_logical_array_items_map_entries_and_giant_scalar(self):
        giant = "giant é😀" * 20_000
        fixtures = [
            ("array.json", [{"text": giant}, {"text": "tail"}], [giant, "tail"]),
            ("map.json", {"first": giant, "second": 7}, [
                json.dumps({"key": "first", "value": giant}, ensure_ascii=False, separators=(",", ":")),
                '{"key":"second","value":7}',
            ]),
            ("scalar.json", giant, [giant]),
            ("small.json", [1, "word", {"text": "actual content"}, None], ["1", "word", "actual content", "null"]),
        ]
        for name, value, expected in fixtures:
            path = self.root / name
            path.write_text(json.dumps(value, ensure_ascii=False) + " \n", encoding="utf-8-sig")
            with self.subTest(name=name):
                records, coverage = self.consume(path)
                self.assertEqual([text for text, _, _ in records], expected)
                self.assertEqual(coverage.processed_bytes, path.stat().st_size)
                self.assertTrue(coverage.as_dict()["complete"])

    def test_spool_allocation_failure_is_not_an_invalid_record_or_completion(self):
        path = self.root / "pressure.jsonl"
        path.write_text(json.dumps({"text": "x" * 200_000}), encoding="utf-8")
        coverage = datasets.DatasetCoverage()
        with patch.object(spools.TextBuilder, "write", side_effect=MemoryError("injected spool pressure")):
            with self.assertRaises(spools.DatasetResourcePause):
                list(datasets.iter_dataset_records(path, coverage=coverage))
        self.assertEqual(coverage.rejected_records, 0)
        self.assertEqual(coverage.rejected_files, 0)
        self.assertFalse(coverage.as_dict()["complete"])

    def test_giant_typed_dialogue_keeps_literal_leased_targets_and_all_response_labels(self):
        path = self.root / "dialogue.jsonl"
        human, response = "é" * 55_000, "😀A" * 20_000
        path.write_text(json.dumps({"messages": [
            {"role": "system", "content": "excluded"}, {"role": "user", "content": human},
            {"role": "assistant", "content": response}, {"role": "assistant", "content": "tail"},
        ]}), encoding="utf-8")
        coverage = datasets.DatasetCoverage()
        stream = iter(datasets.iter_dataset_records(path, coverage=coverage))
        record = next(stream)
        lease = record.text_payload.dialogue
        self.assertEqual(lease.pair_count, 2)
        self.assertEqual(record.provenance["excludedRoles"], {"system": 1})
        self.assertNotIn("dialoguePairs", record.provenance)
        first_human, first_response = lease.pair(0)
        self.assertEqual(first_human.bytes, len(human.encode()))
        tokenizer = types.SimpleNamespace(brain_id=4, eos_id=2, byte_offset=8)
        for length in (2, 7, 128):
            windows = list(spools.leased_dialogue_windows(first_response, tokenizer, length))
            labels = [value for ids, _ in windows for value in ids[1:]]
            self.assertEqual(labels, [value + 8 for value in response.encode()] + [2])
            boundary = windows[31][1]
            resumed = list(spools.leased_dialogue_windows(first_response, tokenizer, length, start_target=boundary))
            self.assertEqual(resumed, windows[32:])
        owned_paths = (lease.path, lease.index_path, record.text_payload.path)
        self.assertEqual(next(stream, None), None)
        self.assertTrue(all(not Path(item).exists() for item in owned_paths))
        self.assertEqual(coverage.rejected_records, 0)
        self.assertEqual(coverage.rejected_files, 0)
        self.assertTrue(coverage.as_dict()["complete"])

    def test_spool_unicode_byte_resume_matches_uninterrupted_windows_and_hash(self):
        builder = spools.TextBuilder(clean=True)
        builder.write(" \n" + "α😀 text " * 20_000 + " \n")
        payload = builder.finish()
        self.addCleanup(payload.close)
        first = list(payload.windows())
        split = 17
        boundary = first[split - 1][1]
        second = list(payload.windows(start_bytes=boundary))
        self.assertEqual(second, first[split:])
        self.assertEqual(second[-1][1], payload.bytes)
        self.assertIsNotNone(spools.previous_payload_byte(payload, boundary))
        digest = hashlib.sha256()
        for piece, _ in first:
            digest.update(piece.encode("utf-8"))
        self.assertEqual(digest.hexdigest(), payload.sha256)

    def test_spool_token_windows_overlap_context_but_never_repeat_or_omit_source_labels(self):
        tokenizer = types.SimpleNamespace(bos_id=1, eos_id=2, byte_offset=8)
        text = "Aé😀" * 900 + " " * 8_100 + "tail"
        builder = spools.TextBuilder(clean=True)
        builder.write(text * 4)
        payload = builder.finish()
        self.addCleanup(payload.close)
        for max_length in (2, 3, 7, 128):
            labels = []
            previous = None
            context = None
            position = 0
            for piece, end in payload.windows():
                adapted = spools.SpoolLearningWindow(piece, context, position == 0, end == payload.bytes)
                for window in adapted.token_windows(tokenizer, max_length):
                    self.assertTrue(2 <= len(window) <= max_length)
                    if previous is not None:
                        self.assertEqual(window[0], previous)
                    labels.extend(window[1:])
                    previous = window[-1]
                context = piece.encode("utf-8")[-1]
                position = end
            expected = [byte + tokenizer.byte_offset for byte in (text * 4).encode("utf-8")] + [tokenizer.eos_id]
            self.assertEqual(labels, expected)

    def test_production_ingest_window_commit_and_resume_never_replays_committed_labels(self):
        class FakeTensor:
            def __init__(self, ids): self.ids, self.shape = ids, (1, len(ids[0]))
            def __getitem__(self, index): return self.ids[index]
        fake_torch = types.SimpleNamespace(tensor=lambda ids, **_: FakeTensor(ids), long="int64")
        tokenizer = types.SimpleNamespace(bos_id=1, eos_id=2, byte_offset=8,
            window_tensors=MagicMock(side_effect=AssertionError("spooled pieces use the bounded token adapter")))
        batched, _, _ = source_method("_streaming_experience_window_batches", {
            "torch": fake_torch, "SpoolLearningWindow": spools.SpoolLearningWindow,
        })
        _, ingest, namespace = source_method("ingest", {
            "DatasetResourcePause": spools.DatasetResourcePause,
            "ReadingWindowHierarchy": spools.ReadingWindowHierarchy,
            "LEARNING_WINDOW_CHARS": spools.LEARNING_WINDOW_CHARS,
            "SpoolLearningWindow": spools.SpoolLearningWindow,
            "previous_payload_byte": spools.previous_payload_byte,
            "validate_active_record_window": spools.validate_active_record_window,
            "active_record_window_sha256": spools.active_record_window_sha256,
            "schedule_sha256": schema.schedule_sha256,
            "INGESTION_CHECKPOINT_FORMAT": "omni-record-ingestion-checkpoint",
            "INGESTION_CHECKPOINT_VERSION": 2, "INGESTION_PARSER_CONTRACT": "omni-dataset-record-stream-v1",
            "_iso_now": lambda: "2026-09-29T00:00:00Z",
        })
        commit = next(node for node in ast.walk(ingest) if isinstance(node, ast.FunctionDef) and node.name == "commit_record_checkpoint")
        branch = next(node for node in ast.walk(ingest) if isinstance(node, ast.If)
                      and "payload.dialogue is not None" in ast.unparse(node.test))
        preface = ast.parse('''
checkpoint = self.saved
commit_sequence = checkpoint["commitSequence"] if checkpoint else 0
record_cursor = 1
last_committed_record_cursor = 0
last_committed_window_count = resume["completedWindows"] if resume else 0
resumed_active_window = resume
active_record_window = None
record_count_hint = 1
current_record_source_bytes = record.bytes_read
record_prefix_sha256 = "9" * 64
previous_record_prefix_sha256 = self.empty_prefix
coverage = self.coverage
learning_schedule = self.schedule
neural_storage_plan = {"detailedRecordAssemblies": True}
compact_streaming = True
policy = "pretrain"
record_name = record.name
record_kind = "text"
source_bytes = record.bytes_read
source_snapshot = {"device": 1, "inode": 1, "size": source_bytes, "mtimeNs": 1}
source_manifest_sha256 = "a" * 64
parser_manifest_sha256 = "c" * 64
content_hash = "b" * 64
source_name_hash = "3" * 64
checkpoint_key = "4" * 64
transaction_id = "5" * 64
epoch = 0
resolved_kind = "jsonl"
loss_total = 0.0
learned_chunks = resume["completedWindows"] if resume else 0
reading_report_count = 0
streaming_gradient_records = 0
streaming_gradient_optimizer_steps = 0
media_accumulator = {}
before_checksum = "6" * 64
before_concepts = before_ideas = before_events = 0
before_memory_neurons = before_memory_synapses = before_memory_synaptic_uses = 0
before_training_steps = before_statistical_experiences = 0
capability_rehearsal_enabled = False
capability_rehearsal_state = None
capability_rehearsal_cadence = {"middleWave": None}
progress = self.report
flush_streaming_local = self.flush
queue_streaming_local = self.queue
verify_source_snapshot = lambda: None
restore_committed_coverage = lambda **_: None
record_progress_value = lambda ordinal, within=0.0: min(0.99, within)
record_position = lambda ordinal: str(ordinal)
def dataset_progress_data(checkpoint_committed=False):
    return {"datasetProgress": {"checkpointCommitted": checkpoint_committed, "activeRecordWindow": active_record_window}}
payload = record.text_payload
''').body
        function = ast.FunctionDef(name="native_route",
            args=ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="record"), ast.arg(arg="resume")], kwonlyargs=[], kw_defaults=[], defaults=[ast.Constant(value=None)]),
            body=preface + [commit, ast.For(target=ast.Name(id="record", ctx=ast.Store()),
                iter=ast.List(elts=[ast.Name(id="record", ctx=ast.Load())], ctx=ast.Load()), body=[branch], orelse=[])]
                + ast.parse("return checkpoint, learned_chunks, reading_report_count\n").body,
            decorator_list=[])
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "production-ingest-window-route", "exec"), namespace)
        route = namespace["native_route"]
        labels, pending = [], []
        brain = types.SimpleNamespace(saved=None, tokenizer=tokenizer, device="cpu",
            empty_prefix=schema.EMPTY_RECORD_PREFIX_SHA256, counter=0, fail_at=64,
            ingestion_checkpoints={}, events=MagicMock(), _clear_allocator_recovery_pause=lambda: None,
            _checkpoint_media_accumulator=lambda value: value,
            parameter_checksum=lambda: hashlib.sha256(str(len(labels)).encode()).hexdigest(),
            _reading_chunk_importances=lambda pieces: [0.5] * len(pieces))
        brain.schedule = schema.make_ingestion_schedule_v3(source_manifest_sha256="a" * 64,
            source_content_sha256="b" * 64, parser_manifest_sha256="c" * 64,
            physical_batch_records=1, gradient_accumulation=1, training_sequence_tokens=2,
            checkpoint_records=512, assembly_page_records=128)
        brain.coverage = datasets.DatasetCoverage(discovered_files=1, discovered_records=1,
            processed_records=1, processed_bytes=320_000)
        def learn(text, **_):
            brain.counter += 1
            return {"assembly_id": "section-%d" % brain.counter, "training": {"loss": 0}}
        brain.learn_experience = learn
        brain._integrate_reading_window_group = lambda children, weights, fingerprint: hashlib.sha256(fingerprint.encode()).hexdigest()
        def flush():
            if not pending: return
            for windows, _ in batched(brain, pending, physical_batch=1, sequence_tokens=2):
                for ids in windows:
                    labels.extend(ids[1:])
            pending.clear()
        brain.flush = flush
        def queue(text, **options):
            self.assertTrue(options["preserve_whitespace"])
            pending.append((options["token_window"], 0))
            if len(pending) >= 2: flush()
        brain.queue = queue
        def save():
            brain.saved = copy.deepcopy(next(iter(brain.ingestion_checkpoints.values())))
            cursor = {"committedRecords": brain.saved["committedRecords"],
                      "recordPrefixSha256": brain.saved["recordPrefixSha256"]}
            if "activeRecordWindowSha256" in brain.saved:
                cursor["activeRecordWindowSha256"] = brain.saved["activeRecordWindowSha256"]
            brain.saved["pagedCheckpointBinding"] = schema.make_checkpoint_binding_v3(
                schedule=brain.schedule, schedule_sha256_value=schema.schedule_sha256(brain.schedule),
                neural_state_sha256=brain.saved["neuralStateChecksum"], checkpoint_sequence=brain.saved["commitSequence"], cursor=cursor,
                coverage={"visitedRecords": 1, "processedRecords": 1, "rejectedRecords": 0,
                          "processedBytes": 320_000, "expectedRecords": 1, "sourceStreamExhausted": False,
                          "sourceContentReverifiedSha256": None},
                vector_generation={"generationId": "e" * 64, "contentSha256": "f" * 64, "recordCount": 1},
                index_generation={"generationId": "1" * 64, "contentSha256": "2" * 64, "recordCount": 1, "highWaterSequence": 1})
        brain.save = save
        def report(_progress, _message, data):
            event = data["datasetProgress"]
            active = event["activeRecordWindow"]
            if event["checkpointCommitted"] and active and active["completedWindows"] == brain.fail_at:
                raise RuntimeError("interrupt after durable window checkpoint")
        brain.report = report
        content = "x" * 320_000
        builder = spools.TextBuilder(clean=True)
        builder.write(content)
        payload = builder.finish()
        record = types.SimpleNamespace(text_payload=payload, bytes_read=320_000, name="giant-row")
        with self.assertRaisesRegex(RuntimeError, "durable window"):
            route(brain, record)
        saved = copy.deepcopy(brain.saved)
        self.assertEqual(saved["committedRecords"], 0)
        self.assertEqual(saved["activeRecordWindow"]["committedTextBytes"], 256_000)
        self.assertEqual(len(labels), 256_000)
        self.assertEqual(saved["pagedCheckpointBinding"]["cursor"]["activeRecordWindowSha256"],
                         saved["activeRecordWindowSha256"])
        payload.close()
        builder = spools.TextBuilder(clean=True)
        builder.write(content)
        payload = builder.finish()
        self.addCleanup(payload.close)
        record.text_payload = payload
        brain.fail_at = -1
        final, learned, integrated = route(brain, record, saved["activeRecordWindow"])
        self.assertEqual(final["committedRecords"], 1)
        self.assertNotIn("activeRecordWindow", final)
        self.assertEqual(learned, 80)
        self.assertEqual(integrated, 1)
        self.assertEqual(brain.counter, 80)
        self.assertEqual(labels, [ord("x") + tokenizer.byte_offset] * 320_000 + [tokenizer.eos_id])
        self.assertFalse(pending)
        tokenizer.window_tensors.assert_not_called()

    def test_hierarchy_is_bounded_and_restores_all_committed_sections(self):
        edges = {}
        def merge(children, weights, fingerprint):
            self.assertLessEqual(len(children), 128)
            identity = hashlib.sha256(fingerprint.encode()).hexdigest()
            edges[identity] = list(children)
            return identity
        tree = spools.ReadingWindowHierarchy("a" * 64, merge)
        for index in range(20_000):
            tree.add("section-%d" % index, 0.5)
            if index == 12_500:
                tree = spools.ReadingWindowHierarchy("a" * 64, merge, state=tree.state())
            self.assertTrue(all(len(bucket) < 128 for bucket in tree.levels))
        root = tree.finish()
        stack, leaves = [root], []
        while stack:
            identity = stack.pop()
            if identity in edges:
                stack.extend(reversed(edges[identity]))
            else:
                leaves.append(identity)
        self.assertEqual(leaves, ["section-%d" % index for index in range(20_000)])

    def test_columnar_giant_strings_use_buffer_views_not_whole_python_scalar_copies(self):
        content = "xé" * 50_000
        scalar = types.SimpleNamespace(type="string", is_valid=True,
            as_buffer=lambda: memoryview(content.encode("utf-8")),
            as_py=MagicMock(side_effect=AssertionError("no giant as_py copy")))
        column = types.SimpleNamespace(__getitem__=lambda _: scalar)
        class Column:
            def __getitem__(self, _): return scalar
        batch = types.SimpleNamespace(schema=types.SimpleNamespace(names=["text"]), num_columns=1, num_rows=1, column=lambda _: Column())
        arrow = types.SimpleNamespace(types=types.SimpleNamespace(is_string=lambda _: True, is_large_string=lambda _: False))
        value = next(datasets._bounded_columnar_rows(batch, arrow))
        coverage = datasets.DatasetCoverage()
        record = datasets._structured_record(value, "native-row", datasets._serialized_row_bytes(value), coverage)
        self.addCleanup(record.text_payload.close)
        self.assertTrue(record.text_payload.path)
        self.assertEqual(record.text_payload.sha256, hashlib.sha256(content.encode("utf-8")).hexdigest())
        scalar.as_py.assert_not_called()

    def test_native_parquet_group_is_admitted_before_a_batch_can_allocate(self):
        path = self.root / "native.parquet"
        path.write_bytes(b"PAR1xx" + struct.pack("<I", 2) + b"PAR1")
        reader = types.SimpleNamespace(schema_arrow=types.SimpleNamespace(names=["text"]),
            metadata=types.SimpleNamespace(num_rows=1, num_row_groups=1,
                row_group=lambda _: types.SimpleNamespace(total_byte_size=200_000_000)),
            iter_batches=MagicMock(side_effect=AssertionError("unsafe native allocation")))
        parquet = types.ModuleType("pyarrow.parquet")
        parquet.ParquetFile = lambda _: reader
        arrow = types.ModuleType("pyarrow")
        arrow.__path__ = []
        arrow.parquet = parquet
        def admit(stage, ram, _disk):
            if ram > 64 * 1024 * 1024:
                raise spools.DatasetResourcePause("native group cannot fit", {"stage": stage})
        coverage = datasets.DatasetCoverage()
        with patch.dict(sys.modules, {"pyarrow": arrow, "pyarrow.parquet": parquet}), spools.parser_admission(admit):
            with self.assertRaisesRegex(spools.DatasetResourcePause, "cannot fit"):
                list(datasets.iter_dataset_records(path, coverage=coverage))
        reader.iter_batches.assert_not_called()
        self.assertEqual(coverage.rejected_records, 0)
        self.assertFalse(coverage.as_dict()["complete"])

    def test_real_parquet_arrow_file_and_stream_keep_giant_scalar_and_tail_rows(self):
        try:
            import pyarrow as arrow
            import pyarrow.parquet as parquet
            import pyarrow.ipc as ipc
        except ImportError:
            self.skipTest("native columnar dependency is unavailable")
        content = "gé😀" * 40_000
        table = arrow.table({"text": [content, "tail"], "id": [1, 2]})
        parquet_path = self.root / "rows.parquet"
        parquet.write_table(table, parquet_path, row_group_size=1)
        paths = [parquet_path]
        for compression in (None, "lz4"):
            for name, writer in (("file", ipc.new_file), ("stream", ipc.new_stream)):
                path = self.root / ("%s-%s.arrow" % (name, compression or "plain"))
                with path.open("wb") as output, writer(output, table.schema,
                    options=ipc.IpcWriteOptions(compression=compression)) as sink:
                    sink.write_table(table)
                paths.append(path)
        for path in paths:
            with self.subTest(path=path.name):
                records, coverage = self.consume(path)
                self.assertEqual([text for text, _, _ in records], [content, "tail"])
                self.assertEqual(coverage.processed_records, 2)
                self.assertTrue(coverage.as_dict()["complete"])

    def test_active_window_hash_is_independently_part_of_the_v3_generation_cursor(self):
        hierarchy = {"leaves": 1, "levels": [[["section-a", 0.5]]], "groups": [0]}
        active = {"recordOrdinal": 1, "textSha256": "a" * 64, "textBytes": 20_000,
                  "committedTextBytes": 4_000, "completedWindows": 1, "windowChars": 4_000, "hierarchy": hierarchy}
        window_hash = spools.active_record_window_sha256(active)
        schedule = schema.make_ingestion_schedule_v3(source_manifest_sha256="a" * 64,
            source_content_sha256="b" * 64, parser_manifest_sha256="c" * 64,
            physical_batch_records=1, gradient_accumulation=1, training_sequence_tokens=2,
            checkpoint_records=512, assembly_page_records=128)
        binding = schema.make_checkpoint_binding_v3(schedule=schedule, schedule_sha256_value=schema.schedule_sha256(schedule),
            neural_state_sha256="d" * 64, checkpoint_sequence=1,
            cursor={"committedRecords": 0, "recordPrefixSha256": schema.EMPTY_RECORD_PREFIX_SHA256,
                    "activeRecordWindowSha256": window_hash},
            coverage={"visitedRecords": 1, "processedRecords": 1, "rejectedRecords": 0, "processedBytes": 20_000,
                      "expectedRecords": 1, "sourceStreamExhausted": False, "sourceContentReverifiedSha256": None},
            vector_generation={"generationId": "e" * 64, "contentSha256": "f" * 64, "recordCount": 1},
            index_generation={"generationId": "1" * 64, "contentSha256": "2" * 64, "recordCount": 1, "highWaterSequence": 1})
        self.assertEqual(binding["cursor"]["activeRecordWindowSha256"], window_hash)
        changed = {**active, "committedTextBytes": 8_000, "completedWindows": 2,
                   "hierarchy": {"leaves": 2, "levels": [[["section-a", 0.5], ["section-b", 0.5]]], "groups": [0]}}
        self.assertNotEqual(spools.active_record_window_sha256(changed), window_hash)
        prefix, _, _ = source_method("_record_prefix_digest", {"bounded_json_sha256": spools.bounded_json_sha256})
        builder = spools.TextBuilder(clean=True)
        builder.write("source" * 20_000)
        payload = builder.finish()
        self.addCleanup(payload.close)
        record = types.SimpleNamespace(text="", text_payload=payload, provenance={}, kind="text", name="fixture", content_sha256="")
        observed = prefix(schema.EMPTY_RECORD_PREFIX_SHA256, 1, record)
        same_text = "".join(piece for piece, _ in payload.windows())
        record.text, record.text_payload = same_text, None
        self.assertEqual(prefix(schema.EMPTY_RECORD_PREFIX_SHA256, 1, record), observed)


if __name__ == "__main__":
    unittest.main()
