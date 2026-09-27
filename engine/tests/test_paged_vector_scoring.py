"""Code-only checks for exact packed-row paging; no brain is constructed."""

import math
import unittest

import torch
from torch.nn import functional as F

from omni_core.paged_vector_scoring import (
    PackedVectorPage,
    PackedVectorRow,
    PagedVectorScoringCancelled,
    PagedVectorScoringPaused,
    prepare_exact_paged_similarity,
)


def _pack(levels):
    row = bytearray([0x55] * ((len(levels) + 3) // 4))
    for index, level in enumerate(levels):
        if level not in (-1, 0, 1):
            raise ValueError("test row is not ternary")
        shift = (index % 4) * 2
        row[index // 4] = (row[index // 4] & ~(3 << shift)) | ((level + 1) << shift)
    return bytes(row)


class _Pages:
    def __init__(self, vectors, *, snapshot="vectors-generation-1"):
        self.dimensions = len(vectors[0]) if vectors else 4
        self.rows = [
            PackedVectorRow(index + 1, "assembly-%04d" % index, _pack(levels))
            for index, levels in enumerate(vectors)
        ]
        self.snapshot = snapshot
        self.calls = []

    def current_snapshot(self):
        return self.snapshot

    def page_rows(self, snapshot_id, cursor, page_size):
        self.calls.append((snapshot_id, cursor, page_size))
        start = int(cursor) if cursor is not None else 0
        end = min(len(self.rows), start + page_size)
        return PackedVectorPage(
            rows=tuple(self.rows[start:end]),
            next_cursor=str(end),
            has_more=end < len(self.rows),
            snapshot_id=self.snapshot,
        )


class PagedVectorScoringTests(unittest.TestCase):
    def test_two_pass_scores_match_direct_cosine_and_floor(self):
        cue = torch.tensor([0.8, 0.4, -0.2, 0.1], dtype=torch.float32)
        vectors = [
            [1, 1, -1, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-1, -1, 1, 0],
            [0, 0, 0, 0],
            [1, -1, 0, 0],
        ]
        provider = _Pages(vectors)
        scan = prepare_exact_paged_similarity(
            provider, cue, page_size=2, workspace_slots=8
        )
        direct = [
            float(F.cosine_similarity(
                cue.reshape(1, -1),
                torch.tensor(row, dtype=torch.float32).reshape(1, -1),
            ).item())
            for row in vectors
        ]
        positive = [score for score in direct if score > 0.0]
        expected_floor = max(
            0.01,
            max(positive) / (2.0 + math.log2(scan.summary.workspace_slots + 1.0)),
        )
        self.assertEqual(scan.summary.rows_seen, len(vectors))
        self.assertEqual(scan.summary.positive_count, len(positive))
        self.assertAlmostEqual(scan.summary.best_score, max(positive), places=6)
        self.assertAlmostEqual(scan.summary.adaptive_floor, expected_floor, places=6)
        matches = list(scan.iter_matches())
        expected_indices = [
            index for index, score in enumerate(direct)
            if score > 0.0 and score >= expected_floor
        ]
        self.assertEqual(
            [match.assembly_id for match in matches],
            ["assembly-%04d" % index for index in expected_indices],
        )
        for match, index in zip(matches, expected_indices):
            self.assertAlmostEqual(match.score, direct[index], places=6)
            self.assertEqual(match.packed, _pack(vectors[index]))
        self.assertEqual(len(provider.calls), 8)
        self.assertTrue(all(page_size == 2 for _, _, page_size in provider.calls))

    def test_no_top_k_truncation_even_when_matches_exceed_workspace_slots(self):
        provider = _Pages([[1, 0, 0, 0] for _ in range(103)])
        scan = prepare_exact_paged_similarity(
            provider, [1.0, 0.0, 0.0, 0.0],
            page_size=7, workspace_slots=2,
        )
        self.assertEqual(scan.summary.positive_count, 103)
        self.assertEqual(len(list(scan.iter_matches())), 103)
        self.assertEqual(len(provider.calls), 30)  # 15 bounded pages per pass.
        self.assertEqual(provider.calls[0][1], None)
        self.assertEqual(provider.calls[15][1], None)

    def test_empty_and_zero_cue_have_no_positive_matches(self):
        empty = _Pages([])
        scan = prepare_exact_paged_similarity(empty, [1, 0, 0, 0])
        self.assertEqual(scan.summary.rows_seen, 0)
        self.assertEqual(list(scan.iter_matches()), [])
        zero = prepare_exact_paged_similarity(
            _Pages([[1, 0, 0, 0]]), [0, 0, 0, 0]
        )
        self.assertEqual(zero.summary.positive_count, 0)
        self.assertEqual(list(zero.iter_matches()), [])

    def test_tiny_nonzero_cue_matches_cosine_epsilon_semantics(self):
        cue = torch.tensor([1e-9, 0, 0, 0], dtype=torch.float32)
        provider = _Pages([[1, 0, 0, 0]])
        scan = prepare_exact_paged_similarity(provider, cue)
        direct = float(F.cosine_similarity(
            cue.reshape(1, -1),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        ).item())
        self.assertAlmostEqual(scan.summary.best_score, direct, places=6)
        self.assertEqual(len(list(scan.iter_matches())), 1)

    def test_resource_and_cancellation_hooks_run_before_fetch_and_per_row(self):
        provider = _Pages([[1, 0, 0, 0] for _ in range(10)])
        observed = []

        def deny_first(estimate, stage):
            observed.append((estimate, stage))
            return False

        with self.assertRaises(PagedVectorScoringPaused):
            prepare_exact_paged_similarity(
                provider, [1, 0, 0, 0], reserve_memory=deny_first
            )
        self.assertEqual(provider.calls, [])
        self.assertGreater(observed[0][0], 0)

        checks = 0

        def cancel_soon():
            nonlocal checks
            checks += 1
            return checks == 5

        with self.assertRaises(PagedVectorScoringCancelled):
            prepare_exact_paged_similarity(
                provider, [1, 0, 0, 0], page_size=10, cancelled=cancel_soon
            )
        self.assertEqual(len(provider.calls), 1)

        pass_two = prepare_exact_paged_similarity(
            provider, [1, 0, 0, 0], page_size=3,
            reserve_memory=lambda _estimate, stage: "pass 2" not in stage,
        )
        calls_before_second_pass = len(provider.calls)
        with self.assertRaises(PagedVectorScoringPaused):
            list(pass_two.iter_matches())
        self.assertEqual(len(provider.calls), calls_before_second_pass)

    def test_mutating_generation_is_detected_at_end_of_second_pass(self):
        provider = _Pages([[1, 0, 0, 0], [0, 1, 0, 0]])
        scan = prepare_exact_paged_similarity(provider, [1, 0, 0, 0], page_size=1)
        provider.rows[1] = PackedVectorRow(2, "assembly-0001", _pack([0, -1, 0, 0]))
        with self.assertRaisesRegex(ValueError, "snapshot changed"):
            list(scan.iter_matches())

    def test_provider_contract_rejects_bad_order_generation_and_packing(self):
        provider = _Pages([[1, 0, 0, 0], [0, 1, 0, 0]])
        provider.rows[1] = PackedVectorRow(1, "assembly-0001", provider.rows[1].packed)
        with self.assertRaisesRegex(ValueError, "order"):
            prepare_exact_paged_similarity(provider, [1, 0, 0, 0], page_size=1)

        provider = _Pages([[1, 0, 0, 0]])
        provider.rows[0] = PackedVectorRow(1, "assembly-0000", b"\xff")
        with self.assertRaisesRegex(ValueError, "reserved"):
            prepare_exact_paged_similarity(provider, [1, 0, 0, 0])

        provider = _Pages([[1, 0, 0, 0, 0]])
        provider.rows[0] = PackedVectorRow(1, "assembly-0000", b"\x56\x59")
        with self.assertRaisesRegex(ValueError, "padding"):
            prepare_exact_paged_similarity(provider, [1, 0, 0, 0, 0])

        provider = _Pages([[1, 0, 0, 0]])
        scan = prepare_exact_paged_similarity(provider, [1, 0, 0, 0])
        provider.snapshot = "different-generation"
        with self.assertRaisesRegex(ValueError, "invalid page"):
            list(scan.iter_matches())

    def test_invalid_cue_and_page_size_fail_before_provider_fetch(self):
        provider = _Pages([[1, 0, 0, 0]])
        for cue in ([1, 0], [1, float("nan"), 0, 0], [True, 0, 0, 0]):
            with self.subTest(cue=cue):
                with self.assertRaises(ValueError):
                    prepare_exact_paged_similarity(provider, cue)
        with self.assertRaisesRegex(ValueError, "page size"):
            prepare_exact_paged_similarity(provider, [1, 0, 0, 0], page_size=4097)
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
