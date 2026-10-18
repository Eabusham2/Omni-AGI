"""NEW pure allocator/tile range cases; no Torch/cache/model/hardware work."""
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "omni_core/cached_operation_layout.py"
spec = importlib.util.spec_from_file_location("owned_cached_range_fixture", path)
LAYOUT = importlib.util.module_from_spec(spec); spec.loader.exec_module(LAYOUT)


def row(length, *, ends=None, page=16, token_bytes=32, budget=65536, key=64, safe=64, owner="actual.key"):
    capacity = min(page, max(1, budget // (4 * token_bytes)))
    return {"owner": owner, "axis": 2, "shape": [1, 2, length - 1, 4], "dtype": "float32", "tokenBytes": token_bytes,
        "pageTokens": page, "tileBudgetBytes": budget, "previousPageTokens": capacity, "ends": ends or [length - 1],
        "configuredKeyTokens": key, "maximumSafeKeyTokens": safe}


class CachedOperationLayoutCases(unittest.TestCase):
    def test_fresh_lengths_within_real_tail_range_match_and_actual_allocation_transitions_do_not(self):
        five = LAYOUT.cached_operation_range(5, 256, [row(5)])
        six = LAYOUT.cached_operation_range(6, 256, [row(6)])
        self.assertEqual(five, six)
        self.assertEqual((five["minimumTokens"], five["maximumTokens"]), (2, 16))
        allocate = LAYOUT.cached_operation_range(17, 256, [row(17)])
        self.assertEqual((allocate["minimumTokens"], allocate["maximumTokens"]), (17, 17))
        next_tail = LAYOUT.cached_operation_range(18, 256, [row(18, ends=[16, 17])])
        self.assertEqual((next_tail["minimumTokens"], next_tail["maximumTokens"]), (18, 32))
        self.assertNotEqual(five["layoutSha256"], next_tail["layoutSha256"])

    def test_all_real_owner_transitions_intersect_not_an_arbitrary_context_bucket(self):
        value = LAYOUT.cached_operation_range(5, 256, [row(5), row(5, page=8, owner="other.value")])
        self.assertEqual((value["minimumTokens"], value["maximumTokens"]), (2, 8))
        value = LAYOUT.cached_operation_range(21, 256, [row(21, page=64, key=8, safe=8)])
        self.assertEqual((value["minimumTokens"], value["maximumTokens"]), (17, 24))

    def test_q_one_actual_halving_regime_and_known_byte_formula_bound_the_range(self):
        maximum = LAYOUT.maximum_safe_key_tokens(256, budget=6400, cost=lambda k: 128 + 64 * k)
        self.assertEqual(maximum, 98)
        first = LAYOUT.cached_operation_range(105, 1024, [row(105, page=128, key=256, safe=maximum)])
        later = LAYOUT.cached_operation_range(106, 1024, [row(106, page=128, key=256, safe=maximum)])
        self.assertEqual(first, later)
        self.assertEqual((first["minimumTokens"], first["maximumTokens"]), (99, 128))
        for length in range(first["minimumTokens"], first["maximumTokens"] + 1):
            actual, halves = length, 0
            while 128 + 64 * actual > 6400: actual //= 2; halves += 1
            self.assertEqual(halves, 1)

    def test_opaque_stale_or_impossible_pages_and_insufficient_cell_admission_refuse(self):
        stale = row(5); stale["previousPageTokens"] = 8
        bad_ends = row(5, ends=[3, 2, 4])
        extra = row(5); extra["guessedBucket"] = 128
        for item in (stale, bad_ends, extra):
            with self.assertRaises(ValueError): LAYOUT.cached_operation_range(5, 256, [item])
        with self.assertRaises(RuntimeError): LAYOUT.maximum_safe_key_tokens(64, budget=1, cost=lambda k: 8 * k)
        with self.assertRaises(ValueError): LAYOUT.cached_operation_range(257, 256, [row(257)])


if __name__ == "__main__": unittest.main()
