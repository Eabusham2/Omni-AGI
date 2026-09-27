"""Storage-only tests for the disk-authoritative ternary vector mapping."""

import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core.packed_vsa_vectors import PackedTernaryVectors
from omni_core.paged_packed_vectors import (
    PagedPackedVectors,
    PagedVectorCacheNeedsRebuild,
    PagedVectorResourcePause,
)
from omni_core.paged_vector_scoring import prepare_exact_paged_similarity


class PagedPackedVectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-paged-vectors-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "vectors.sqlite3"
        self.vectors = PagedPackedVectors(
            self.path, 5, seed=17, cache_bytes=10
        )

    def test_mapping_is_disk_backed_with_bounded_independent_decodes(self):
        for index in range(13):
            levels = [1 if index % 2 else -1, 0, 1, 0, -1]
            self.vectors["row-%02d" % index] = torch.tensor(levels)
        self.assertEqual(len(self.vectors), 13)
        self.assertEqual(list(self.vectors), ["row-%02d" % i for i in range(13)])
        self.assertEqual([key for key, _value in self.vectors.items()], list(self.vectors))
        self.assertEqual(len(list(self.vectors.values())), 13)
        self.assertIn("row-01", self.vectors)
        self.assertNotIn("missing", self.vectors)
        self.assertIsNone(self.vectors.get("missing"))
        decoded = self.vectors["row-01"]
        decoded[0] = 0.0
        self.assertAlmostEqual(float(self.vectors["row-01"][0]), 1 / math.sqrt(3))
        levels = self.vectors.levels("row-01")
        self.assertEqual(levels.dtype, torch.int8)
        levels[0] = 0
        self.assertEqual(int(self.vectors.levels("row-01")[0]), 1)
        self.assertLessEqual(self.vectors.status()["cacheBytes"], 10)
        self.assertEqual(self.vectors.status()["packedRowBytes"], 2)
        self.assertEqual(self.vectors.decode_rows(["row-00", "row-01"]).shape, (2, 5))
        with self.assertRaisesRegex(ValueError, "bounded window"):
            self.vectors.decode_rows(["row-00"] * 65)
        reopened = PagedPackedVectors(self.path, 5)
        self.assertEqual(reopened.seed, 17)
        self.assertEqual(reopened.packed_row("row-01"), self.vectors.packed_row("row-01"))
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PagedPackedVectors(self.path, 4)

    def test_adapt_matches_in_memory_reference_and_resumes_counter(self):
        reference = PackedTernaryVectors(5, seed=17)
        original = torch.tensor([1.0, -1.0, 0.0, 1.0, 0.0])
        self.vectors["idea"] = original
        reference["idea"] = original
        targets = [
            torch.tensor([-1.0, 1.0, 0.0, 0.0, 1.0]),
            torch.tensor([1.0, 0.0, -1.0, 1.0, 0.0]),
        ]
        for index in range(18):
            target = targets[index % 2]
            rate = 0.23 if index % 3 else 0.61
            actual = self.vectors.adapt("idea", target, rate)
            expected = reference.adapt("idea", target, rate)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(self.vectors.packed_row("idea"), reference.packed_row("idea"))
            self.assertEqual(self.vectors.update_count("idea"), reference.update_count("idea"))
        reopened = PagedPackedVectors(self.path, 5, seed=17)
        result = reopened.adapt("idea", targets[0], 0.35)
        expected = reference.adapt("idea", targets[0], 0.35)
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual(reopened.packed_row("idea"), reference.packed_row("idea"))
        self.assertEqual(reopened.update_count("idea"), reference.update_count("idea"))

    def test_pages_feed_exact_scorer_and_drift_fails_closed(self):
        for index in range(27):
            self.vectors["row-%02d" % index] = torch.tensor([1, 0, 0, 0, 0])
        snapshot = self.vectors.current_snapshot()
        page = self.vectors.page_rows(snapshot, None, 7)
        self.assertEqual(len(page.rows), 7)
        self.assertTrue(page.has_more)
        self.assertEqual(page.rows[0].assembly_id, "row-00")
        scan = prepare_exact_paged_similarity(
            self.vectors, [1, 0, 0, 0, 0], page_size=7, workspace_slots=2
        )
        self.assertEqual(scan.summary.positive_count, 27)
        self.assertEqual(len(list(scan.iter_matches())), 27)
        self.vectors["row-00"] = torch.tensor([-1, 0, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, "generation drift"):
            self.vectors.page_rows(snapshot, page.next_cursor, 7)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            list(scan.iter_matches())

    def test_deleted_rows_do_not_reappear_or_shift_keyset_order(self):
        for index in range(10):
            self.vectors["row-%02d" % index] = torch.tensor([1, 0, 0, 0, 0])
        del self.vectors["row-03"]
        with self.assertRaises(KeyError):
            self.vectors["row-03"]
        self.assertEqual(len(self.vectors), 9)
        snapshot = self.vectors.current_snapshot()
        cursor = None
        rows = []
        while True:
            page = self.vectors.page_rows(snapshot, cursor, 3)
            rows.extend(page.rows)
            if not page.has_more:
                break
            cursor = page.next_cursor
        self.assertEqual(
            [row.assembly_id for row in rows],
            ["row-%02d" % i for i in range(10) if i != 3],
        )

    def test_checksum_and_ternary_validation_reject_corrupt_rows(self):
        self.vectors["idea"] = torch.tensor([1, 0, -1, 0, 1])
        with self.assertRaisesRegex(ValueError, "width"):
            self.vectors.set_packed("bad", b"\x55")
        with self.assertRaisesRegex(ValueError, "reserved"):
            self.vectors.set_packed("bad", b"\xff\x55")
        with self.assertRaisesRegex(ValueError, "padding"):
            self.vectors.set_packed("bad", b"\x55\x59")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE paged_vector_rows SET packed=? WHERE vector_id='idea'",
                (b"\x55\x55",),
            )
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.vectors.packed_row("idea")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.vectors.page_rows(self.vectors.current_snapshot(), None, 1)

    def test_external_writer_invalidates_decoded_cache(self):
        self.vectors["idea"] = torch.tensor([1, 0, 0, 0, 0])
        self.assertEqual(int(self.vectors.levels("idea")[0]), 1)
        other = PagedPackedVectors(self.path, 5, seed=17)
        other["idea"] = torch.tensor([-1, 0, 0, 0, 0])
        self.assertEqual(int(self.vectors.levels("idea")[0]), -1)

    def test_reserve_refusal_preserves_row_and_revision(self):
        self.vectors["idea"] = torch.tensor([1, 0, 0, 0, 0])
        before = self.vectors.status()["revision"]

        def deny_disk(_bytes, _label):
            return False

        denied = PagedPackedVectors(
            self.path, 5, seed=17, disk_reserve=deny_disk
        )
        with self.assertRaises(PagedVectorResourcePause):
            denied["idea"] = torch.tensor([-1, 0, 0, 0, 0])
        self.assertEqual(self.vectors.status()["revision"], before)
        self.assertEqual(int(self.vectors.levels("idea")[0]), 1)

        def deny_memory(_bytes, label):
            return "page" not in label

        denied_page = PagedPackedVectors(
            self.path, 5, seed=17, memory_reserve=deny_memory
        )
        with self.assertRaises(PagedVectorResourcePause):
            denied_page.page_rows(denied_page.current_snapshot(), None, 1)

    def test_bounded_batch_commits_once_and_rolls_back_on_window_or_caught_error(self):
        with self.vectors.batch(max_rows=3, max_payload_bytes=4096) as batch:
            batch.set("a", torch.tensor([1, 0, 0, 0, 0]))
            batch.set("b", torch.tensor([0, 1, 0, 0, 0]))
            batch.adapt("a", torch.tensor([-1, 0, 0, 0, 0]), 0.5)
            with sqlite3.connect(self.path) as connection:
                self.assertEqual(connection.execute(
                    "SELECT COUNT(*) FROM paged_vector_rows"
                ).fetchone()[0], 0)
        self.assertEqual(len(self.vectors), 2)
        self.assertEqual(self.vectors.update_count("a"), 1)
        self.assertEqual(self.vectors.status()["revision"], 3)

        with self.assertRaises(PagedVectorResourcePause):
            with self.vectors.batch(max_rows=1, max_payload_bytes=4096) as batch:
                batch.set("c", torch.tensor([1, 0, 0, 0, 0]))
                batch.set("d", torch.tensor([1, 0, 0, 0, 0]))
        self.assertNotIn("c", self.vectors)
        self.assertNotIn("d", self.vectors)
        self.assertEqual(self.vectors.status()["revision"], 3)

        with self.assertRaises(PagedVectorResourcePause):
            with self.vectors.batch(max_rows=2, max_payload_bytes=4096) as batch:
                batch.set("c", torch.tensor([1, 0, 0, 0, 0]))
                try:
                    batch.set_packed("bad", b"\xff\x55")
                except ValueError:
                    pass
        self.assertNotIn("c", self.vectors)

    def test_export_state_is_bounded_and_matches_existing_shard_format(self):
        for index in range(5):
            self.vectors["row-%d" % index] = torch.tensor([1, index % 2, 0, 0, -1])
        self.vectors.adapt("row-1", torch.tensor([-1, 0, 0, 0, 1]), 0.6)
        self.assertEqual(self.vectors.storage_bytes, 5 * (self.vectors.row_bytes + 8))
        metadata, tensors = self.vectors.export_state(
            prefix="neurons_", keys=["row-3", "row-1"]
        )
        restored = PackedTernaryVectors.from_state(
            metadata, tensors, prefix="neurons_"
        )
        self.assertEqual(list(restored), ["row-3", "row-1"])
        for identifier in restored:
            self.assertEqual(restored.packed_row(identifier), self.vectors.packed_row(identifier))
            self.assertEqual(restored.update_count(identifier), self.vectors.update_count(identifier))
        with mock.patch("omni_core.paged_packed_vectors.MAX_EXPORT_ROWS", 4):
            with self.assertRaisesRegex(ValueError, "bounded shard"):
                self.vectors.export_state()
            with self.assertRaisesRegex(ValueError, "unbounded"):
                self.vectors.export_state(keys=list(self.vectors))
        small = PagedPackedVectors(self.path, 5, seed=17)
        metadata, tensors = small.export_state(keys=["row-0"])
        self.assertEqual(metadata["rowCount"], 1)
        self.assertEqual(tuple(tensors["packed_rows"].shape), (1, 2))

    def test_committed_generation_binding_rejects_dirty_cache(self):
        generation = "a" * 64
        with self.assertRaises(PagedVectorCacheNeedsRebuild):
            self.vectors.discard_or_reconcile_uncommitted(generation)
        self.vectors["idea"] = torch.tensor([1, 0, 0, 0, 0])
        revision = self.vectors.status()["revision"]
        with self.assertRaisesRegex(ValueError, "revision changed"):
            self.vectors.bind_committed_generation(
                generation, expected_revision=revision - 1
            )
        bound = self.vectors.bind_committed_generation(
            generation, expected_revision=revision
        )
        self.assertEqual(bound["revision"], revision)
        self.assertFalse(self.vectors.status()["dirtySinceCommit"])
        self.assertEqual(
            self.vectors.discard_or_reconcile_uncommitted(generation)["state"],
            "clean",
        )
        self.vectors["idea"] = torch.tensor([-1, 0, 0, 0, 0])
        self.assertTrue(self.vectors.status()["dirtySinceCommit"])
        with self.assertRaises(PagedVectorCacheNeedsRebuild):
            self.vectors.discard_or_reconcile_uncommitted(generation)
        with self.assertRaises(PagedVectorCacheNeedsRebuild):
            PagedPackedVectors(self.path, 5).discard_or_reconcile_uncommitted(generation)

    def test_bounded_shard_import_rebuilds_fresh_cache_in_page_order(self):
        for index in range(6):
            self.vectors["row-%d" % index] = torch.tensor([
                1, 0, index % 2, 0, -1
            ])
        self.vectors.adapt(
            "row-2", torch.tensor([-1, 0, 0, 0, 1]), 0.7
        )
        rebuilt = PagedPackedVectors(
            self.path.parent / "rebuilt.sqlite3", 5, seed=17
        )
        for ids in (["row-0", "row-1", "row-2"],
                    ["row-3", "row-4", "row-5"]):
            metadata, tensors = self.vectors.export_state(keys=ids)
            self.assertEqual(rebuilt.import_state(metadata, tensors), len(ids))
        self.assertEqual(list(rebuilt), list(self.vectors))
        for identifier in self.vectors:
            self.assertEqual(rebuilt.packed_row(identifier), self.vectors.packed_row(identifier))
            self.assertEqual(rebuilt.update_count(identifier), self.vectors.update_count(identifier))
        metadata, tensors = self.vectors.export_state(keys=["row-0"])
        before = rebuilt.status()["revision"]
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            rebuilt.import_state(metadata, tensors)
        self.assertEqual(rebuilt.status()["revision"], before)


if __name__ == "__main__":
    unittest.main()
