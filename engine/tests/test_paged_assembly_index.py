"""Storage-only checks for the paged assembly index; no model is built."""

import base64
import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from omni_core.paged_assembly_index import (
    AssemblyIndexResourcePause,
    PagedAssemblyIndex,
)
from omni_core.paged_packed_vectors import PagedVectorCacheNeedsRebuild
from omni_core.paged_vector_scoring import prepare_exact_paged_similarity


def _assembly(number, *, rehearsals=1):
    return {
        "id": "assembly-%04d" % number,
        "fingerprint": "fingerprint-%04d" % number,
        "neuron_ids": ["neuron-%04d" % number],
        "rehearsals": rehearsals,
        "importance": 0.25,
    }


def _packed(levels):
    row = bytearray([0x55] * ((len(levels) + 3) // 4))
    for index, level in enumerate(levels):
        if level not in (-1, 0, 1):
            raise ValueError("test vector must be ternary")
        shift = (index % 4) * 2
        row[index // 4] = (row[index // 4] & ~(3 << shift)) | ((level + 1) << shift)
    return bytes(row)


class PagedAssemblyIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-assembly-index-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "assembly.sqlite3"
        self.index = PagedAssemblyIndex(self.path)

    def test_persistent_exact_lookup_and_reexposure_preserves_sequence(self):
        original = _assembly(1)
        self.assertTrue(self.index.upsert(original))
        self.assertFalse(self.index.upsert(dict(original)))
        updated = _assembly(1, rehearsals=2)
        self.assertFalse(self.index.upsert(updated))
        self.assertEqual(self.index.get_by_id(original["id"]), updated)
        self.assertEqual(self.index.get_by_fingerprint(original["fingerprint"]), updated)
        self.assertIsNone(self.index.get_by_id("missing"))
        self.assertEqual(self.index.count(), 1)
        self.assertEqual(self.index.status()["highWaterSequence"], 1)
        reopened = PagedAssemblyIndex(self.path)
        self.assertEqual(reopened.get_by_id(original["id"]), updated)
        self.assertEqual(reopened.count(), 1)

    def test_id_and_fingerprint_collisions_never_overwrite_or_advance_checkpoint(self):
        self.index.upsert(_assembly(1), checkpoint=("source", {"row": 1}))
        duplicate_id = {**_assembly(2), "id": _assembly(1)["id"]}
        duplicate_fingerprint = {
            **_assembly(2), "fingerprint": _assembly(1)["fingerprint"]
        }
        for conflicting in (duplicate_id, duplicate_fingerprint):
            with self.subTest(conflicting=conflicting):
                with self.assertRaisesRegex(ValueError, "collision"):
                    self.index.upsert(
                        conflicting, checkpoint=("source", {"row": 99})
                    )
                self.assertEqual(self.index.load_checkpoint("source"), {"row": 1})
                self.assertEqual(self.index.count(), 1)
        self.assertEqual(self.index.get_by_id("assembly-0001"), _assembly(1))

    def test_keyset_pages_are_bounded_deterministic_and_resume_after_reopen(self):
        for number in range(25):
            self.index.upsert(_assembly(number))
        first = self.index.page(page_size=7)
        self.assertEqual([item["id"] for item in first.records], [
            "assembly-%04d" % number for number in range(7)
        ])
        self.assertTrue(first.has_more)
        self.assertEqual(first.through_sequence, 25)
        self.index.upsert(_assembly(25))
        reopened = PagedAssemblyIndex(self.path)
        pages = [first]
        cursor = first.cursor
        while pages[-1].has_more:
            pages.append(reopened.page(page_size=7, cursor=cursor))
            cursor = pages[-1].cursor
        seen = [item["id"] for page in pages for item in page.records]
        self.assertEqual(seen, ["assembly-%04d" % number for number in range(25)])
        self.assertEqual([len(page.records) for page in pages], [7, 7, 7, 4])
        self.assertFalse(pages[-1].has_more)
        self.assertEqual(reopened.page(7, cursor=cursor).records, ())
        self.assertEqual([item["id"] for page in reopened.iter_pages(9)
                          for item in page.records], [
            "assembly-%04d" % number for number in range(26)
        ])

    def test_cursor_rejects_tampering_or_different_store(self):
        self.index.upsert(_assembly(1))
        cursor = self.index.page(1).cursor
        other_path = self.path.parent / "other.sqlite3"
        other = PagedAssemblyIndex(other_path)
        other.upsert(_assembly(1))
        with self.assertRaisesRegex(ValueError, "cursor"):
            other.page(1, cursor=cursor)
        with self.assertRaisesRegex(ValueError, "cursor"):
            self.index.page(1, cursor=cursor[:-1] + ("0" if cursor[-1] != "0" else "1"))
        with self.assertRaisesRegex(ValueError, "page size"):
            self.index.page(0)
        with self.assertRaisesRegex(ValueError, "page size"):
            self.index.page(4097)

    def test_checksum_mismatch_fails_lookup_page_and_duplicate_admission(self):
        self.index.upsert(_assembly(1))
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE assembly_records SET record_json = ? WHERE assembly_id = ?",
                (b'{"id":"corrupt"}', "assembly-0001"),
            )
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.index.get_by_id("assembly-0001")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.index.page(1)
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.index.upsert(_assembly(1))

    def test_checkpoint_is_atomic_with_upsert_and_checked_on_load(self):
        self.index.upsert(_assembly(1), checkpoint=("source", {"row": 1}))
        self.assertEqual(self.index.load_checkpoint("source"), {"row": 1})
        self.assertFalse(self.index.upsert(
            _assembly(1), checkpoint=("source", {"row": 2})
        ))
        self.assertEqual(self.index.load_checkpoint("source"), {"row": 2})
        self.index.save_checkpoint("source", {"row": 3, "scheduleVersion": 2})
        self.assertEqual(self.index.load_checkpoint("source"), {
            "row": 3, "scheduleVersion": 2
        })
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE progress_checkpoints SET payload_json = ? WHERE name = ?",
                (b'{"row":999}', "source"),
            )
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.index.load_checkpoint("source")

    def test_reserve_hooks_pause_before_mutation_or_page_materialization(self):
        observed_disk = []
        observed_memory = []

        def allow_disk(estimated, operation):
            observed_disk.append((estimated, operation))
            return True

        def allow_memory(estimated, operation):
            observed_memory.append((estimated, operation))
            return True

        guarded = PagedAssemblyIndex(
            self.path, disk_reserve=allow_disk, memory_reserve=allow_memory
        )
        guarded.upsert(_assembly(1))
        guarded.page(1)
        self.assertTrue(any("upsert" in operation for _, operation in observed_disk))
        self.assertTrue(any("page" in operation for _, operation in observed_memory))
        self.assertTrue(all(estimated > 0 for estimated, _ in observed_disk + observed_memory))

        def deny_disk(_estimated, _operation):
            return False

        denied_write = PagedAssemblyIndex(self.path, disk_reserve=deny_disk)
        with self.assertRaises(AssemblyIndexResourcePause):
            denied_write.upsert(_assembly(2), checkpoint=("source", {"row": 2}))
        self.assertIsNone(self.index.get_by_id("assembly-0002"))
        self.assertIsNone(self.index.load_checkpoint("source"))

        def deny_memory(_estimated, _operation):
            return False

        denied_read = PagedAssemblyIndex(self.path, memory_reserve=deny_memory)
        with self.assertRaises(AssemblyIndexResourcePause):
            denied_read.page(1)
        with self.assertRaises(AssemblyIndexResourcePause):
            denied_read.get_by_id("assembly-0001")

    def test_checkpoint_reserve_failure_rolls_back_prior_record_insert(self):
        def deny_only_checkpoint(_estimated, operation):
            return "checkpoint" not in operation

        guarded = PagedAssemblyIndex(
            self.path, disk_reserve=deny_only_checkpoint
        )
        with self.assertRaises(AssemblyIndexResourcePause):
            guarded.upsert(_assembly(1), checkpoint=("source", {"row": 1}))
        self.assertEqual(self.index.count(), 0)
        self.assertIsNone(self.index.load_checkpoint("source"))

    def test_existing_incomplete_schema_fails_closed(self):
        with sqlite3.connect(self.path) as connection:
            connection.execute("DROP TABLE progress_checkpoints")
        with self.assertRaisesRegex(ValueError, "schema is incomplete"):
            PagedAssemblyIndex(self.path)

    def test_sqlite_uses_exact_and_sequence_indexes(self):
        with sqlite3.connect(self.path) as connection:
            id_plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT sequence FROM assembly_records "
                "WHERE assembly_id = ?", ("assembly-0001",)
            ).fetchall()
            fp_plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT sequence FROM assembly_records "
                "WHERE fingerprint = ?", ("fingerprint-0001",)
            ).fetchall()
            page_plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT sequence FROM assembly_records "
                "WHERE sequence > ? AND sequence <= ? ORDER BY sequence LIMIT ?",
                (0, 10, 5),
            ).fetchall()
        self.assertTrue(any("INDEX" in row[-1] for row in id_plan), id_plan)
        self.assertTrue(any("INDEX" in row[-1] for row in fp_plan), fp_plan)
        self.assertTrue(any("INTEGER PRIMARY KEY" in row[-1] for row in page_plan), page_plan)

    def test_explicit_packed_row_is_atomic_and_metadata_only_never_defaults(self):
        self.index.upsert(_assembly(1))
        self.assertIsNone(self.index.get_packed_vector("assembly-0001"))
        self.assertEqual(self.index.status()["packedVectorRows"], 0)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.index.current_snapshot()
        vector_index = PagedAssemblyIndex(self.path, dimensions=5)
        row = _packed([1, 0, -1, 1, 0])
        self.assertFalse(vector_index.upsert(
            _assembly(1), packed_vector=row,
            checkpoint=("source", {"row": 1}),
        ))
        self.assertEqual(vector_index.get_packed_vector("assembly-0001"), row)
        self.assertEqual(vector_index.load_checkpoint("source"), {"row": 1})
        self.assertEqual(vector_index.status()["vectorRevision"], 1)
        self.assertEqual(vector_index.status()["packedVectorRows"], 1)
        self.assertFalse(vector_index.upsert(_assembly(1, rehearsals=2)))
        self.assertEqual(vector_index.get_packed_vector("assembly-0001"), row)
        self.assertEqual(vector_index.status()["vectorRevision"], 1)
        self.assertEqual(PagedAssemblyIndex(self.path).get_packed_vector("assembly-0001"), row)
        with self.assertRaisesRegex(ValueError, "disagree"):
            PagedAssemblyIndex(self.path, dimensions=4)

    def test_vector_page_provider_skips_metadata_only_and_scores_all_matches(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        expected_ids = []
        for number in range(31):
            packed = _packed([1, 0, 0, 0]) if number % 4 else None
            index.upsert(_assembly(number), packed_vector=packed)
            if packed is not None:
                expected_ids.append("assembly-%04d" % number)
        snapshot = index.current_snapshot()
        cursor = None
        rows = []
        while True:
            page = index.page_rows(snapshot, cursor, 5)
            self.assertLessEqual(len(page.rows), 5)
            self.assertEqual(page.snapshot_id, snapshot)
            rows.extend(page.rows)
            if not page.has_more:
                break
            cursor = page.next_cursor
        self.assertEqual([row.assembly_id for row in rows], expected_ids)
        self.assertEqual([row.sequence for row in rows], sorted(row.sequence for row in rows))
        scan = prepare_exact_paged_similarity(
            index, [1, 0, 0, 0], page_size=5, workspace_slots=2
        )
        self.assertEqual(scan.summary.positive_count, len(expected_ids))
        self.assertEqual(
            [match.assembly_id for match in scan.iter_matches()], expected_ids
        )

    def test_vector_page_joins_one_authoritative_row_and_checks_coverage(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        for number in range(40):
            index.upsert(_assembly(number))
        index.upsert(
            _assembly(41), packed_vector=_packed([1, 0, 0, 0])
        )
        with sqlite3.connect(self.path) as connection:
            plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT a.sequence FROM assembly_records AS a "
                "JOIN paged_vector_rows AS v ON v.vector_id=a.assembly_id "
                "WHERE a.sequence > ? AND a.sequence <= ? "
                "ORDER BY a.sequence LIMIT ?",
                (0, 41, 5),
            ).fetchall()
            self.assertTrue(any(
                "paged_vector_rows" in row[-1]
                for row in plan
            ), plan)
            snapshot = index.current_snapshot()
            connection.execute(
                "DELETE FROM paged_vector_rows WHERE vector_id='assembly-0041'"
            )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            index.page_rows(snapshot, None, 5)

    def test_scorer_rejects_vector_change_between_its_two_passes(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        index.upsert(_assembly(1), packed_vector=_packed([1, 0, 0, 0]))
        scan = prepare_exact_paged_similarity(index, [1, 0, 0, 0])
        index.upsert(_assembly(1), packed_vector=_packed([-1, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "generation drift"):
            list(scan.iter_matches())

    def test_vector_generation_drift_and_cursor_mismatch_fail_closed(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        index.upsert(_assembly(1), packed_vector=_packed([1, 0, 0, 0]))
        index.upsert(_assembly(2), packed_vector=_packed([0, 1, 0, 0]))
        snapshot = index.current_snapshot()
        first = index.page_rows(snapshot, None, 1)
        self.assertTrue(first.has_more)
        index.upsert(_assembly(1, rehearsals=2))  # Metadata only: same vector generation.
        self.assertEqual(len(index.page_rows(snapshot, first.next_cursor, 1).rows), 1)
        index.upsert(_assembly(3))  # Changes the next snapshot bound, not vectors.
        other_snapshot = index.current_snapshot()
        with self.assertRaisesRegex(ValueError, "cursor"):
            index.page_rows(other_snapshot, first.next_cursor, 1)
        index.upsert(_assembly(2), packed_vector=_packed([0, -1, 0, 0]))
        with self.assertRaisesRegex(ValueError, "generation drift"):
            index.page_rows(snapshot, first.next_cursor, 1)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            index.page_rows(other_snapshot, None, 1)

    def test_invalid_or_corrupt_packed_rows_are_never_silent(self):
        index = PagedAssemblyIndex(self.path, dimensions=5)
        for bad in (b"\x55", b"\xff\x55", b"\x56\x59"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    index.upsert(
                        _assembly(1), packed_vector=bad,
                        checkpoint=("source", {"row": 1}),
                    )
                self.assertEqual(index.count(), 0)
                self.assertIsNone(index.load_checkpoint("source"))
        index.upsert(_assembly(1), packed_vector=_packed([1, 0, 0, 0, 0]))
        snapshot = index.current_snapshot()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE paged_vector_rows SET packed = ? WHERE vector_id = ?",
                (_packed([-1, 0, 0, 0, 0]), "assembly-0001"),
            )
        with self.assertRaisesRegex(ValueError, "checksum"):
            index.get_packed_vector("assembly-0001")
        with self.assertRaisesRegex(ValueError, "checksum"):
            index.page_rows(snapshot, None, 1)
        with self.assertRaisesRegex(ValueError, "checksum"):
            index.upsert(_assembly(1))

    def test_packed_row_rolls_back_if_checkpoint_reserve_denies(self):
        def deny_checkpoint(_estimated, operation):
            return "checkpoint" not in operation

        index = PagedAssemblyIndex(
            self.path, dimensions=4, disk_reserve=deny_checkpoint
        )
        with self.assertRaises(AssemblyIndexResourcePause):
            index.upsert(
                _assembly(1), packed_vector=_packed([1, 0, 0, 0]),
                checkpoint=("source", {"row": 1}),
            )
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.status()["vectorRevision"], 0)
        self.assertIsNone(index.load_checkpoint("source"))

    def test_v1_metadata_index_upgrades_without_losing_rows_or_cursor(self):
        legacy_path = self.path.parent / "legacy-v1.sqlite3"
        record = _assembly(7)
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE assembly_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    assembly_id TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL UNIQUE,
                    record_json BLOB NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE index_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE progress_checkpoints (
                    name TEXT PRIMARY KEY,
                    payload_json BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID;
                PRAGMA user_version = 1;
                """
            )
            connection.execute(
                "INSERT INTO index_metadata(key,value) VALUES ('store_id',?)",
                ("legacy-store",),
            )
            connection.execute(
                "INSERT INTO assembly_records "
                "(assembly_id,fingerprint,record_json,record_sha256,updated_at) "
                "VALUES (?,?,?,?,?)",
                (record["id"], record["fingerprint"], payload,
                 hashlib.sha256(payload).hexdigest(), 1.0),
            )
        cursor_body = json.dumps({
            "format": "omni-assembly-page-cursor",
            "version": 1,
            "storeId": "legacy-store",
            "throughSequence": 1,
            "afterSequence": 0,
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        cursor = (
            base64.urlsafe_b64encode(cursor_body).rstrip(b"=").decode("ascii")
            + "." + hashlib.sha256(cursor_body).hexdigest()
        )
        upgraded = PagedAssemblyIndex(legacy_path, dimensions=4)
        self.assertEqual(upgraded.status()["formatVersion"], 3)
        self.assertEqual(upgraded.get_by_id(record["id"]), record)
        self.assertEqual(upgraded.page(2, cursor=cursor).records, (record,))
        self.assertIsNone(upgraded.get_packed_vector(record["id"]))
        upgraded.upsert(record, packed_vector=_packed([1, 0, 0, 0]))
        self.assertEqual(
            PagedAssemblyIndex(legacy_path).get_packed_vector(record["id"]),
            _packed([1, 0, 0, 0]),
        )

    def test_v2_vector_blob_migrates_into_single_authoritative_row(self):
        legacy_path = self.path.parent / "legacy-v2.sqlite3"
        record = _assembly(9)
        payload = json.dumps(
            record, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        packed = _packed([1, 0, -1, 0])
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE assembly_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    assembly_id TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL UNIQUE,
                    record_json BLOB NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    packed_vector BLOB,
                    vector_sha256 TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE index_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE progress_checkpoints (
                    name TEXT PRIMARY KEY,
                    payload_json BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID;
                PRAGMA user_version = 2;
                """
            )
            connection.executemany(
                "INSERT INTO index_metadata(key,value) VALUES (?,?)",
                [("store_id", "legacy-v2"), ("vector_dimensions", "4"),
                 ("vector_revision", "1")],
            )
            connection.execute(
                "INSERT INTO assembly_records "
                "(assembly_id,fingerprint,record_json,record_sha256,"
                "packed_vector,vector_sha256,updated_at) VALUES (?,?,?,?,?,?,?)",
                (record["id"], record["fingerprint"], payload,
                 hashlib.sha256(payload).hexdigest(), packed,
                 hashlib.sha256(packed).hexdigest(), 1.0),
            )
        migrated = PagedAssemblyIndex(legacy_path)
        self.assertEqual(migrated.status()["formatVersion"], 3)
        self.assertEqual(migrated.get_packed_vector(record["id"]), packed)
        self.assertEqual(migrated._vectors.packed_row(record["id"]), packed)
        with sqlite3.connect(legacy_path) as connection:
            legacy_copy = connection.execute(
                "SELECT packed_vector,vector_sha256 FROM assembly_records "
                "WHERE assembly_id=?", (record["id"],)
            ).fetchone()
            self.assertEqual(legacy_copy, (None, None))
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE assembly_records SET packed_vector=? WHERE assembly_id=?",
                    (packed, record["id"]),
                )

    def test_direct_upsert_rejects_raw_source_and_credential_fields(self):
        for extra in (
            {"source_text": "a private passage"},
            {"raw_passage": "a private passage"},
            {"token_ids": [1, 2, 3]},
            {"api_key": "sk-secret"},
            {"password": "secret"},
            {"source_label": "api_key=sk-secret"},
            {"source_label": "first line\nprivate second line"},
            {"source_provenance": {"token=secret": 1}},
        ):
            with self.subTest(extra=extra):
                with self.assertRaises(ValueError):
                    self.index.upsert({**_assembly(1), **extra})
                self.assertEqual(self.index.count(), 0)
        safe = {
            **_assembly(1),
            "source": "sha256:document-abc",
            "source_label": "document import",
            "source_provenance": {"sha256:document-abc": 1},
        }
        self.assertTrue(self.index.upsert(safe))
        self.assertEqual(self.index.get_by_id(safe["id"]), safe)

    def test_long_safe_provenance_is_hashed_not_rejected_or_stored_verbatim(self):
        long_path = "dataset/" + "safe-folder/" * 40 + "records.parquet"
        record = {
            **_assembly(1),
            "source": long_path,
            "source_label": long_path,
            "source_provenance": {long_path: 1},
        }
        self.assertTrue(self.index.upsert(record))
        stored = self.index.get_by_id(record["id"])
        self.assertTrue(stored["source"].startswith(long_path[:160]))
        self.assertIn("#sha256:", stored["source"])
        self.assertNotEqual(stored["source"], long_path)
        self.assertEqual(list(stored["source_provenance"]), [stored["source"]])
        self.assertFalse(self.index.upsert(record))
        with sqlite3.connect(self.path) as connection:
            payload = connection.execute(
                "SELECT record_json FROM assembly_records WHERE assembly_id=?",
                (record["id"],),
            ).fetchone()[0]
        self.assertNotIn(long_path.encode(), payload)

    def test_sequence_lookup_and_numeric_page_after_remain_bounded(self):
        for number in range(8):
            self.index.upsert(_assembly(number))
        self.assertEqual(self.index.get_by_sequence(1), _assembly(0))
        self.assertIsNone(self.index.get_by_sequence(100))
        first = self.index.page_after(0, page_size=3)
        self.assertEqual(
            [record["id"] for record in first.records],
            ["assembly-%04d" % number for number in range(3)],
        )
        self.index.upsert(_assembly(8))
        rest = self.index.page_after(
            3, page_size=20, through_sequence=first.through_sequence
        )
        self.assertEqual(
            [record["id"] for record in rest.records],
            ["assembly-%04d" % number for number in range(3, 8)],
        )
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.index.get_by_sequence(0)

    def test_uncommitted_working_cursor_cannot_resume_past_brain_pointer(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        index.upsert(
            _assembly(1), packed_vector=_packed([1, 0, 0, 0]),
            checkpoint=("source", {"row": 1}),
        )
        generation = "b" * 64
        before = index.status()
        bound = index.bind_committed_generation(
            generation,
            expected_index_revision=before["indexRevision"],
            expected_vector_revision=before["vectorRevision"],
        )
        self.assertEqual(bound["generationSha256"], generation)
        self.assertEqual(
            index.load_committed_checkpoint("source", generation), {"row": 1}
        )
        index.save_checkpoint("source", {"row": 99})
        self.assertEqual(index.load_checkpoint("source"), {"row": 99})
        self.assertTrue(index.status()["dirtySinceCommit"])
        with self.assertRaises(PagedVectorCacheNeedsRebuild):
            index.load_committed_checkpoint("source", generation)
        with self.assertRaises(PagedVectorCacheNeedsRebuild):
            index.discard_or_reconcile_uncommitted(generation)

    def test_bounded_assembly_batch_commits_metadata_vectors_and_cursor_once(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        with index.batch(max_rows=3, max_payload_bytes=4096) as batch:
            self.assertTrue(batch.upsert(
                _assembly(1), packed_vector=_packed([1, 0, 0, 0])
            ))
            self.assertTrue(batch.upsert(
                _assembly(2), packed_vector=_packed([0, 1, 0, 0])
            ))
            self.assertTrue(batch.upsert(_assembly(3)))
            batch.save_checkpoint("source", {"row": 3})
            with sqlite3.connect(self.path) as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM assembly_records"
                ).fetchone()[0], 0)
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM paged_vector_rows"
                ).fetchone()[0], 0)
        self.assertEqual(index.count(), 3)
        self.assertEqual(index.status()["packedVectorRows"], 2)
        self.assertEqual(index.load_checkpoint("source"), {"row": 3})
        self.assertEqual(index.get_packed_vector("assembly-0001"), _packed([1, 0, 0, 0]))
        self.assertEqual(index.status()["vectorRevision"], 2)

    def test_assembly_batch_window_or_caught_privacy_error_rolls_back_all(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        with self.assertRaises(AssemblyIndexResourcePause):
            with index.batch(max_rows=1, max_payload_bytes=4096) as batch:
                batch.upsert(_assembly(1), packed_vector=_packed([1, 0, 0, 0]))
                batch.upsert(_assembly(2), packed_vector=_packed([0, 1, 0, 0]))
        self.assertEqual(index.count(), 0)
        self.assertEqual(index.status()["vectorRevision"], 0)

        with self.assertRaises(AssemblyIndexResourcePause):
            with index.batch(max_rows=2, max_payload_bytes=4096) as batch:
                batch.upsert(_assembly(1))
                try:
                    batch.upsert({**_assembly(2), "source_text": "private passage"})
                except ValueError:
                    pass
        self.assertEqual(index.count(), 0)

    def test_assembly_batch_cursor_reserve_failure_rolls_back_rows(self):
        def deny_cursor(_estimated, operation):
            return "assembly batch cursor" not in operation

        index = PagedAssemblyIndex(
            self.path, dimensions=4, disk_reserve=deny_cursor
        )
        with self.assertRaises(AssemblyIndexResourcePause):
            with index.batch(max_rows=2, max_payload_bytes=4096) as batch:
                batch.upsert(_assembly(1), packed_vector=_packed([1, 0, 0, 0]))
                batch.save_checkpoint("source", {"row": 1})
        self.assertEqual(index.count(), 0)
        self.assertIsNone(index.load_checkpoint("source"))
        self.assertEqual(index.status()["vectorRevision"], 0)

    def test_assembly_batch_requires_cursor_hint_to_be_final(self):
        index = PagedAssemblyIndex(self.path, dimensions=4)
        with self.assertRaisesRegex(ValueError, "final"):
            with index.batch(max_rows=2, max_payload_bytes=4096) as batch:
                batch.upsert(
                    _assembly(1),
                    checkpoint=("source", {"row": 1}),
                )
                batch.upsert(_assembly(2))
        self.assertEqual(index.count(), 0)
        self.assertIsNone(index.load_checkpoint("source"))

        with index.batch(max_rows=1, max_payload_bytes=4096) as batch:
            batch.upsert(
                _assembly(1), packed_vector=_packed([1, 0, 0, 0]),
                checkpoint=("source", {"row": 1}),
            )
        self.assertEqual(index.count(), 1)
        self.assertEqual(index.load_checkpoint("source"), {"row": 1})

    def test_legacy_raw_source_row_is_rejected_before_v3_migration(self):
        legacy_path = self.path.parent / "unsafe-v1.sqlite3"
        record = {**_assembly(3), "source_text": "private raw source"}
        payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        with sqlite3.connect(legacy_path) as connection:
            connection.executescript(
                """
                CREATE TABLE assembly_records (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    assembly_id TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL UNIQUE,
                    record_json BLOB NOT NULL,
                    record_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE index_metadata (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                ) WITHOUT ROWID;
                CREATE TABLE progress_checkpoints (
                    name TEXT PRIMARY KEY,
                    payload_json BLOB NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                ) WITHOUT ROWID;
                PRAGMA user_version = 1;
                """
            )
            connection.execute(
                "INSERT INTO index_metadata(key,value) VALUES ('store_id','unsafe-v1')"
            )
            connection.execute(
                "INSERT INTO assembly_records "
                "(assembly_id,fingerprint,record_json,record_sha256,updated_at) "
                "VALUES (?,?,?,?,?)",
                (record["id"], record["fingerprint"], payload,
                 hashlib.sha256(payload).hexdigest(), 1.0),
            )
        with self.assertRaisesRegex(ValueError, "non-structural"):
            PagedAssemblyIndex(legacy_path)
        with sqlite3.connect(legacy_path) as connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
