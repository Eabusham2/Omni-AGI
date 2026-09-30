"""Constructor-free source/index math checks; no brain/model execution."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.paged_recall_graph import PagedRecallGraph
from omni_core.paged_recurrent_spreading import PagedRecurrentState
from omni_core.vsa import _RevisionedSynapses


class PagedRecallGraphTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="omni-graph-index-storage-")
        self.addCleanup(folder.cleanup)
        self.directory = Path(folder.name)
        self.nodes = PagedNeuronMetadata(self.directory / "nodes.sqlite3")
        for identifier in ("a", "b", "c"):
            self.nodes[identifier] = {"id": identifier, "activation": 0.5, "uncertainty": 0.2}
        self.edges = _RevisionedSynapses({
            "ab": {"id": "ab", "source_id": "a", "target_id": "b", "effective_weight": 1},
            "cb": {"id": "cb", "source_id": "c", "target_id": "b", "effective_weight": -1},
            "missing": {"id": "missing", "source_id": "a", "target_id": "missing", "effective_weight": 1},
        })
        self.graph = PagedRecallGraph(self.directory, reserve=lambda *_args: True)
        self.addCleanup(self.graph.close)
        self.graph.load(((key, row["source_id"], row["target_id"], row["effective_weight"])
                         for key, row in self.edges.items()), node_exists=lambda key: key in self.nodes,
                        synapse_revision=self.edges.graph_revision, neuron_revision=self.nodes.graph_revision)
        self.edges._recall_graph_observer = self.graph.queue_change
        self.nodes._recall_graph_observer = self.graph.queue_membership_change

    def eligible(self):
        return list(self.graph.connection.execute("SELECT source_id,target_id,level FROM edges ORDER BY ordinal"))

    def test_sign_endpoint_and_delete_deltas_never_scan_the_source_graph(self):
        self.edges["ab"]["effective_weight"] = -1
        self.edges["cb"]["target_id"] = "c"
        del self.edges["missing"]
        with mock.patch.object(self.edges, "items", side_effect=AssertionError("full source scan")), \
             mock.patch.object(self.edges, "values", side_effect=AssertionError("full source scan")):
            self.assertTrue(self.graph.refresh(self.edges, self.nodes))
        self.assertEqual(self.eligible(), [("a", "b", -1), ("c", "c", -1)])
        self.assertEqual(self.graph.eligible_edges, 2)
        self.assertEqual(self.graph.inhibitory_edges, 2)
        self.assertEqual(dict(self.graph.connection.execute("SELECT * FROM incoming")), {"b": 1, "c": 1})

    def test_membership_add_remove_updates_all_and_only_incident_signs(self):
        self.nodes["missing"] = {"id": "missing"}
        self.assertTrue(self.graph.refresh(self.edges, self.nodes))
        self.assertEqual(self.graph.eligible_edges, 3)
        del self.nodes["b"]
        self.assertTrue(self.graph.refresh(self.edges, self.nodes))
        self.assertEqual(self.eligible(), [("a", "missing", 1)])
        self.assertEqual(self.graph.eligible_edges, 1)
        self.nodes["b"] = {"id": "b"}
        self.assertTrue(self.graph.refresh(self.edges, self.nodes))
        self.assertEqual(self.graph.eligible_edges, 3)

    def test_missing_authenticated_delta_or_missed_revision_requires_rebuild(self):
        self.edges["ab"]["effective_weight"] = -1
        self.graph.connection.execute("DELETE FROM authenticated_merkle_nodes")
        self.graph.connection.commit()
        self.assertFalse(self.graph.refresh(self.edges, self.nodes))
        self.assertTrue(self.graph.invalid)

    def test_source_mutation_survives_optional_cache_reserve_failure(self):
        self.graph.reserve = lambda *_args: False
        self.edges["ab"]["effective_weight"] = -1
        self.assertEqual(self.edges["ab"]["effective_weight"], -1)
        self.assertTrue(self.graph.invalid)
        self.assertFalse(self.graph.refresh(self.edges, self.nodes))

    def test_frontier_uses_actual_read_only_index_and_exact_fan_in(self):
        state = PagedRecurrentState(self.directory)
        self.addCleanup(state.close)
        state.use_graph(self.graph)
        state.load_seeds([("a", 0.9), ("c", 0.4)])
        state.settle(slots=1, adaptive_floor=0.05)
        self.assertAlmostEqual(state.activation("b"), 0.13, places=12)
        with self.assertRaisesRegex(Exception, "readonly"):
            state.connection.execute("DELETE FROM graph_index.raw_edges")


if __name__ == "__main__":
    unittest.main()
