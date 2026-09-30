"""Constructor-free numerical/storage contracts for paged recurrence."""

import math
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from omni_core.paged_recurrent_spreading import PagedRecurrentState


def reference(seeds, edges, *, slots, floor):
    incoming = defaultdict(int)
    adjacency = defaultdict(list)
    for source, target, level in edges:
        if level:
            incoming[target] += 1
            adjacency[source].append((target, level))
    activation = dict(seeds)
    admitted = set(seeds)
    rounds = 0
    while activation:
        pressure = max(1.0, len(activation) / float(slots))
        pressure_floor = max(1e-5, floor * (0.02 + 0.03 * math.log2(pressure + 1)))
        drives = defaultdict(float)
        for source, value in activation.items():
            if abs(value) > 1e-12:
                for target, level in adjacency[source]:
                    drives[target] += value * level
        settled = {}
        for target in admitted.union(seeds).union(drives):
            value = max(-1.0, min(1.0, seeds.get(target, 0) + 0.52 * drives[target] / max(1, incoming[target])))
            if target in admitted or abs(value) >= pressure_floor:
                admitted.add(target)
                settled[target] = value
        delta = max((abs(settled.get(key, 0) - activation.get(key, 0))
                     for key in set(settled).union(activation)), default=0)
        activation = settled
        rounds += 1
        if delta <= max(1e-7, pressure_floor * 0.001):
            break
    return activation, rounds, delta


class PagedRecurrentTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="omni-recurrence-math-")
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)

    def state(self):
        state = PagedRecurrentState(self.directory)
        self.addCleanup(state.close)
        return state

    def test_recurrent_cycle_inhibition_and_continuous_values_match_original_math(self):
        seeds = {"a": 0.9, "b": 0.4}
        edges = [("a", "b", 1), ("b", "a", -1), ("a", "c", 1), ("c", "a", 0)]
        expected, rounds, delta = reference(seeds, edges, slots=8, floor=0.05)
        state = self.state()
        state.load(seeds.items(), edges, node_exists=lambda _identifier: True)
        state.settle(slots=8, adaptive_floor=0.05)
        actual = dict(state.iter_activation())
        self.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key], places=12)
        self.assertEqual(state.rounds, rounds)
        self.assertAlmostEqual(state.convergence_delta, delta, places=12)

    def test_many_seeds_and_long_recurrence_have_no_top_k_or_hop_limit(self):
        seeds = {"seed-%d" % index: 0.6 for index in range(2048)}
        edges = [("seed-%d" % index, "tail-%d" % index, 1) for index in range(2048)]
        edges += [("tail-%d" % index, "seed-%d" % index, 1) for index in range(2048)]
        state = self.state()
        state.load(seeds.items(), edges, node_exists=lambda _identifier: True)
        state.settle(slots=2, adaptive_floor=0.01)
        self.assertEqual(state.active_count, 4096)
        self.assertGreater(state.rounds, 8)
        self.assertEqual(len(list(state.iter_activation(positive_only=True))), 4096)

    def test_missing_neurons_zero_edges_and_reserve_failure_do_not_invent_firing(self):
        state = self.state()
        state.load([("a", 0.6)], [("a", "b", 0), ("a", "missing", 1)],
                   node_exists=lambda identifier: identifier != "missing")
        state.settle(slots=1, adaptive_floor=0.01)
        self.assertEqual(dict(state.iter_activation()), {"a": 0.6})
        self.assertEqual(state.eligible_edges, 0)
        state.reserve = lambda _size, _operation: False
        with self.assertRaises(RuntimeError):
            state.settle(slots=1, adaptive_floor=0.01)

    def test_recalled_results_remain_addressable_without_a_resident_list(self):
        state = self.state()
        for index in range(3000):
            state.append_recalled({"assembly_id": "assembly-%d" % index, "score": index / 3000})
        recalled = state.recalled()
        self.assertEqual(len(recalled), 3000)
        self.assertEqual(recalled[-1]["assembly_id"], "assembly-2999")
        self.assertEqual(len(recalled[10:20]), 10)
        self.assertEqual(sum(1 for _record in recalled), 3000)


if __name__ == "__main__":
    unittest.main()
