"""Storage-only assembly scoring checks; no brain or model is constructed."""

import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.paged_assembly_scoring import PagedAssemblyVectorProvider
from omni_core.paged_packed_vectors import PagedPackedVectors
from omni_core.paged_vector_scoring import prepare_exact_paged_similarity


class PagedAssemblyScoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-paged-assembly-score-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.index = PagedAssemblyIndex(root / "assemblies.sqlite3")
        self.vectors = PagedPackedVectors(root / "neurons.sqlite3", 16, seed=7)

    def _add(self, number, value):
        identifier = "assembly-%04d" % number
        self.index.upsert({
            "id": identifier,
            "fingerprint": "fingerprint-%04d" % number,
            "neuron_ids": [identifier],
        })
        self.vectors[identifier] = torch.tensor(value, dtype=torch.float32)
        return identifier

    def test_exact_two_pass_scores_only_assemblies_and_no_top_k(self):
        expected = [self._add(number, [1] + [0] * 15) for number in range(73)]
        self.vectors["ordinary-neuron"] = torch.tensor([-1] + [0] * 15)
        provider = PagedAssemblyVectorProvider(self.index, self.vectors)
        scan = prepare_exact_paged_similarity(
            provider, [1] + [0] * 15, page_size=7, workspace_slots=2
        )
        self.assertEqual(scan.summary.rows_seen, 73)
        self.assertEqual(scan.summary.positive_count, 73)
        self.assertEqual([match.assembly_id for match in scan.iter_matches()], expected)
        provider.assert_unchanged(scan.summary.snapshot_id)

    def test_metadata_and_neural_drift_both_fail_closed(self):
        identifier = self._add(1, [1] + [0] * 15)
        provider = PagedAssemblyVectorProvider(self.index, self.vectors)
        scan = prepare_exact_paged_similarity(provider, [1] + [0] * 15)
        self.index.upsert({
            "id": identifier,
            "fingerprint": "fingerprint-0001",
            "neuron_ids": [identifier],
            "rehearsals": 2,
        })
        with self.assertRaisesRegex(ValueError, "generation drift"):
            list(scan.iter_matches())
        with self.assertRaisesRegex(ValueError, "generation drift"):
            provider.assert_unchanged(scan.summary.snapshot_id)

        provider = PagedAssemblyVectorProvider(self.index, self.vectors)
        scan = prepare_exact_paged_similarity(provider, [1] + [0] * 15)
        self.vectors[identifier] = torch.tensor([-1] + [0] * 15)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            list(scan.iter_matches())

    def test_missing_authoritative_assembly_row_is_not_a_silent_zero(self):
        self.index.upsert({
            "id": "assembly-without-vector",
            "fingerprint": "missing-vector",
            "neuron_ids": [],
        })
        provider = PagedAssemblyVectorProvider(self.index, self.vectors)
        with self.assertRaisesRegex(ValueError, "lacks its authoritative"):
            prepare_exact_paged_similarity(provider, [1] + [0] * 15)

    def test_cursor_is_generation_bound_and_paging_is_bounded(self):
        for number in range(9):
            self._add(number, [1] + [0] * 15)
        provider = PagedAssemblyVectorProvider(self.index, self.vectors)
        snapshot = provider.current_snapshot()
        first = provider.page_rows(snapshot, None, 3)
        self.assertEqual(len(first.rows), 3)
        self.assertTrue(first.has_more)
        with self.assertRaisesRegex(ValueError, "cursor"):
            provider.page_rows(snapshot, first.next_cursor + "x", 3)
        seen = list(first.rows)
        cursor = first.next_cursor
        while True:
            page = provider.page_rows(snapshot, cursor, 3)
            self.assertLessEqual(len(page.rows), 3)
            seen.extend(page.rows)
            if not page.has_more:
                break
            cursor = page.next_cursor
        self.assertEqual(len(seen), 9)
        with self.assertRaisesRegex(ValueError, "page size"):
            provider.page_rows(snapshot, None, 4097)

    def test_verified_shared_db_uses_one_authoritative_vector_object(self):
        shared = PagedAssemblyIndex(
            self.index.path.parent / "shared.sqlite3", dimensions=16, seed=7
        )
        shared.upsert({
            "id": "assembly-one",
            "fingerprint": "assembly-one",
            "neuron_ids": ["assembly-one"],
        })
        assert shared._vectors is not None
        shared._vectors["assembly-one"] = torch.tensor([1] + [0] * 15)
        shared._vectors["ordinary-neuron"] = torch.tensor([-1] + [0] * 15)
        provider = PagedAssemblyVectorProvider(shared, shared._vectors)
        scan = prepare_exact_paged_similarity(provider, [1] + [0] * 15)
        self.assertEqual(
            [match.assembly_id for match in scan.iter_matches()], ["assembly-one"]
        )
        duplicate_handle = PagedPackedVectors(shared.path, 16, seed=7)
        with self.assertRaisesRegex(ValueError, "second vector authority"):
            PagedAssemblyVectorProvider(shared, duplicate_handle)


if __name__ == "__main__":
    unittest.main()
