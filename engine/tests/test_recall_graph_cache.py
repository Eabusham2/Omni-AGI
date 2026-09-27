import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core.vsa import (
    LazyPersistedSynapses,
    NeuralSubstrate,
    _unpack_ternary_level,
)


def _edge(source: str, target: str, weight: int, kind: str = "test") -> dict:
    identifier = f"{source}>{target}:{kind}"
    return {
        "id": identifier,
        "source_id": source,
        "target_id": target,
        "kind": kind,
        "effective_weight": weight,
        "eligibility": 0.0,
        "plasticity": 1.0,
        "uses": 0,
        "stability": 0.0,
        "last_updated_at": 0.0,
    }


def _node(identifier: str) -> dict:
    return {
        "id": identifier,
        "activation": 0.0,
        "uncertainty": 0.5,
        "exposures": 1,
    }


def _small_graph() -> tuple[NeuralSubstrate, torch.Tensor]:
    memory = NeuralSubstrate(16, seed=11)
    vectors = {
        "a": torch.tensor([1.0, 0.0] + [0.0] * 14),
        "b": torch.tensor([0.8, 0.2] + [0.0] * 14),
        "c": torch.tensor([-0.2, 0.8] + [0.0] * 14),
    }
    for identifier, vector in vectors.items():
        memory.neurons[identifier] = _node(identifier)
        memory.assemblies.append({"id": identifier, "neuron_ids": []})
        memory.assembly_vectors[identifier] = vector
    for source, target, weight in (
        ("a", "b", 1),
        ("b", "a", -1),
        ("a", "c", 1),
        ("c", "a", 0),
    ):
        record = _edge(source, target, weight)
        memory.synapses[record["id"]] = record
    return memory, vectors["a"]


def _recall(memory: NeuralSubstrate, cue: torch.Tensor):
    signal, recalled = memory.recall_vector(
        cue, workspace_slots=8, record_activity=False
    )
    return signal, recalled, dict(memory._last_recall_audit)


def _uncached(memory: NeuralSubstrate, cue: torch.Tensor):
    original_nodes = memory.neurons
    original_edges = memory.synapses
    memory.neurons = dict(original_nodes)
    memory.synapses = {key: dict(value) for key, value in original_edges.items()}
    try:
        return _recall(memory, cue)
    finally:
        memory.neurons = original_nodes
        memory.synapses = original_edges


def _reference_lazy_edges(lazy: LazyPersistedSynapses) -> list:
    """The original shard/overlay merge, to pin exact edge order in tests."""

    result = []
    for key in lazy._ordered_keys:
        replacements = {
            index: lazy._dirty[record_id]
            for record_id, (location_key, index) in lazy._dirty_locations.items()
            if location_key == key and record_id in lazy._dirty
        }
        deleted = {
            index
            for _record_id, (location_key, index) in lazy._deleted_locations.items()
            if location_key == key
        }
        structures, packed = lazy._forward_by_shard.get(key, ([], b""))
        base = {
            index: (source, target, _unpack_ternary_level(packed, offset))
            for offset, (index, _record_id, source, target) in enumerate(structures)
        }
        for index in sorted(set(base).union(replacements)):
            if index in deleted:
                continue
            replacement = replacements.get(index)
            if replacement is None:
                result.append(base[index])
                continue
            weight = NeuralSubstrate.exact_effective_weight(
                replacement.get("effective_weight", 0)
            )
            if weight:
                result.append(
                    (
                        str(replacement["source_id"]),
                        str(replacement["target_id"]),
                        weight,
                    )
                )
    for record_id, record in lazy._dirty.items():
        if record_id in lazy._new_ids:
            weight = NeuralSubstrate.exact_effective_weight(
                record.get("effective_weight", 0)
            )
            if weight:
                result.append(
                    (str(record["source_id"]), str(record["target_id"]), weight)
                )
    return result


class RecallGraphCacheTests(unittest.TestCase):
    def assert_same_recall(self, left, right):
        self.assertTrue(torch.equal(left[0], right[0]))
        self.assertEqual(left[1], right[1])
        self.assertEqual(left[2], right[2])

    def test_eager_cache_preserves_exact_recurrence_and_survives_activity(self):
        memory, cue = _small_graph()
        first = _recall(memory, cue)
        cache = memory._recall_graph_cache
        self.assertIsNotNone(cache)
        self.assert_same_recall(first, _recall(memory, cue))
        self.assertIs(cache, memory._recall_graph_cache)

        memory.record_recall_activity(first[1])
        self.assert_same_recall(first, _recall(memory, cue))
        self.assertIs(cache, memory._recall_graph_cache)
        self.assert_same_recall(first, _uncached(memory, cue))

    def test_first_exposure_updates_stored_tracked_synapse_not_a_detached_copy(self):
        memory = NeuralSubstrate(16, seed=23)
        memory.learn("one fresh pathway through a new substrate")
        self.assertTrue(memory.synapses)
        self.assertTrue(all(record["uses"] >= 1 for record in memory.synapses.values()))
        self.assertTrue(
            any(int(record["effective_weight"]) == 1 for record in memory.synapses.values())
        )

    def test_eager_cache_invalidates_every_forward_mutation(self):
        memory, cue = _small_graph()
        _recall(memory, cue)

        def checked_change(change):
            previous = memory._recall_graph_cache
            change()
            actual = _recall(memory, cue)
            self.assertIsNot(previous, memory._recall_graph_cache)
            self.assert_same_recall(actual, _uncached(memory, cue))
            _recall(memory, cue)

        checked_change(
            lambda: memory.synapses["a>b:test"].__setitem__(
                "effective_weight", -1
            )
        )
        checked_change(
            lambda: memory.synapses["a>b:test"].update(
                {"source_id": "c", "target_id": "b"}
            )
        )
        checked_change(
            lambda: memory.synapses.__setitem__(
                "b>c:extra", _edge("b", "c", 1, "extra")
            )
        )
        checked_change(lambda: memory.synapses.__delitem__("b>c:extra"))
        checked_change(lambda: memory.neurons.__delitem__("c"))
        checked_change(lambda: memory.neurons.__setitem__("c", _node("c")))

        # Decay can cross the ternary boundary without changing topology.
        memory.synapses["a>c:test"]["eligibility"] = -0.249
        memory.mark_attention_synapse("a>c:test")
        memory.synapses["a>c:test"]["uses"] = 0
        checked_change(lambda: memory.decay(0.02))
        self.assertEqual(memory.synapses["a>c:test"]["effective_weight"], 0)

    def test_zero_weight_population_does_not_disqualify_small_live_graph(self):
        memory, cue = _small_graph()
        for index in range(100):
            record = _edge("a", "c", 0, f"dormant-{index}")
            memory.synapses[record["id"]] = record
        with mock.patch("omni_core.vsa._RECALL_GRAPH_CACHE_BYTES", 1_024):
            actual = _recall(memory, cue)
            self.assertIsNotNone(memory._recall_graph_cache)
            self.assert_same_recall(actual, _uncached(memory, cue))

    def test_assembly_lookup_tracks_appends_replacement_and_explicit_reorder(self):
        memory, cue = _small_graph()
        first = memory.assembly_by_id
        self.assertEqual(set(first), {"a", "b", "c"})
        added = {"id": "d", "fingerprint": "repeat", "neuron_ids": []}
        memory.assemblies.append(added)
        memory.assembly_vectors["d"] = cue.clone()
        memory.neurons["d"] = _node("d")
        self.assertIs(memory.assembly_by_id["d"], added)
        self.assertIn("d", {row["assembly_id"] for row in _recall(memory, cue)[1]})

        duplicate = {"id": "a", "fingerprint": "repeat", "neuron_ids": ["d"]}
        memory.assemblies.append(duplicate)
        memory.assemblies.append({"id": "", "neuron_ids": []})
        memory.assemblies.append({"neuron_ids": []})
        self.assertIs(memory.assembly_by_id["a"], duplicate)
        self.assertIs(memory.assembly_by_fingerprint["repeat"], added)
        self.assertNotIn("", memory.assembly_by_id)
        memory.assemblies = [{"id": "replacement", "fingerprint": "new", "neuron_ids": []}]
        self.assertEqual(set(memory.assembly_by_id), {"replacement"})
        self.assertEqual(set(memory.assembly_by_fingerprint), {"new"})
        memory.assemblies[0] = {"id": "same-length", "fingerprint": "newer", "neuron_ids": []}
        memory.invalidate_assembly_index()
        self.assertEqual(set(memory.assembly_by_id), {"same-length"})
        self.assertEqual(set(memory.assembly_by_fingerprint), {"newer"})

    def test_lazy_forward_index_cache_keeps_cold_shards_cold_and_invalidates(self):
        with tempfile.TemporaryDirectory(prefix="omni-recall-graph-") as folder:
            store = Path(folder) / "substrate"
            source = NeuralSubstrate(32, seed=19)
            source.learn("alpha beta gamma delta exact recurrent recall")
            source.learn("beta gamma inhibitor delta persistent graph")
            source.save_sharded(store, records_per_shard=3)
            memory = NeuralSubstrate.load_sharded(
                store, source.metadata(include_records=False), lazy_synapses=True
            )
            self.assertIsInstance(memory.synapses, LazyPersistedSynapses)
            cue = memory.vector_for_text("alpha beta recurrent recall")
            self.assertEqual(
                list(memory.synapses.iter_effective_edges()),
                _reference_lazy_edges(memory.synapses),
            )
            original_iter = memory.synapses.iter_effective_edges
            with mock.patch.object(
                memory.synapses,
                "iter_effective_edges",
                wraps=memory.synapses.iter_effective_edges,
            ) as scanned:
                first = _recall(memory, cue)
                second = _recall(memory, cue)
                self.assertEqual(scanned.call_count, 1)
                self.assert_same_recall(first, second)
                self.assertEqual(memory.synapses.resident_record_count, 0)

                edge_id = next(iter(memory.synapses))
                record = memory.synapses[edge_id]
                record["effective_weight"] = -int(record["effective_weight"] or 1)
                self.assertEqual(
                    list(original_iter()),
                    _reference_lazy_edges(memory.synapses),
                )
                after = _recall(memory, cue)
                self.assertEqual(scanned.call_count, 2)
                self.assert_same_recall(after, _uncached(memory, cue))

                record["source_id"] = str(record["target_id"])
                self.assertEqual(
                    list(original_iter()),
                    _reference_lazy_edges(memory.synapses),
                )
                _recall(memory, cue)
                self.assertEqual(scanned.call_count, 3)

                added = _edge(
                    str(memory.assemblies[0]["id"]),
                    str(memory.assemblies[1]["id"]),
                    -1,
                    "new-lazy",
                )
                memory.synapses[added["id"]] = added
                self.assertEqual(
                    list(original_iter()),
                    _reference_lazy_edges(memory.synapses),
                )
                changed = _recall(memory, cue)
                self.assertEqual(scanned.call_count, 4)
                self.assert_same_recall(changed, _uncached(memory, cue))
                del memory.synapses[added["id"]]
                self.assertEqual(
                    list(original_iter()),
                    _reference_lazy_edges(memory.synapses),
                )
                _recall(memory, cue)
                self.assertEqual(scanned.call_count, 5)

    def test_new_native_attention_is_tracked_and_direct_load_is_legacy_safe(self):
        memory = NeuralSubstrate(16, seed=5)
        self.assertFalse(memory.attention_legacy_raw_active)
        memory.mark_attention_neuron("a")
        self.assertEqual(memory.attention_active_neuron_ids, {"a"})
        with tempfile.TemporaryDirectory(prefix="omni-attention-load-") as folder:
            memory.learn("tracked new native neural state")
            store = Path(folder) / "substrate"
            memory.save_sharded(store, records_per_shard=3)
            loaded = NeuralSubstrate.load_sharded(
                store, memory.metadata(include_records=False), lazy_synapses=False
            )
            self.assertTrue(loaded.attention_legacy_raw_active)


if __name__ == "__main__":
    unittest.main()
