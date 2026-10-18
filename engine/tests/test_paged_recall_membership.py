"""Constructor-free checks for the bounded cold-recall membership component."""

from contextlib import ExitStack
import hashlib
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omni_core.paged_neuron_metadata import PagedNeuronMetadata, NeuronMetadataResourcePause
from omni_core.paged_recall_graph import PagedRecallGraph
from omni_core.connected_transaction import _ACTIVE, _Participant
from omni_core.paged_read_session import ReadSession


def neuron(index):
    return {"id": "n%d" % index, "neuron_id": "n%d" % index,
            "label": "node %d" % index, "region": "semantic"}


def graph_rows(graph):
    return list(graph.connection.execute("SELECT * FROM raw_edges ORDER BY ordinal"))


class PagedRecallMembershipTests(unittest.TestCase):
    def test_bulk_read_obeys_the_real_connected_write_transaction_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = PagedNeuronMetadata(Path(folder) / "nodes.sqlite3")
            nodes.import_page([neuron(0)], max_rows=1)
            connection = sqlite3.connect(nodes.path, isolation_level=None)
            participant = _Participant(connection)
            key = str(nodes.path.resolve())
            token = _ACTIVE.set(SimpleNamespace(allowed_databases={key}, databases={key: participant}))
            try:
                nodes["n1"] = neuron(1)
                revision = nodes.graph_revision
                with ReadSession([nodes]) as kept:
                    self.assertEqual(nodes.contains_many(("n0", "n1", "absent"),
                                                         expected_graph_revision=revision),
                                     frozenset({"n0", "n1"}))
                    self.assertEqual(kept.opened_connections, 0)
                connection.rollback()
            finally:
                _ACTIVE.reset(token)
                connection.close()
            self.assertNotIn("n1", nodes)

    def test_exact_order_sign_absence_duplicates_and_sql_reduction(self):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as cleanup:
            root = Path(folder)
            nodes = PagedNeuronMetadata(root / "nodes.sqlite3")
            nodes.import_page([neuron(index) for index in range(19)], max_rows=19)
            revision = nodes.graph_revision
            before = hashlib.sha256(nodes.path.read_bytes()).hexdigest()
            edges = [("e%d" % index, "n%d" % (index % 21), "n%d" % ((index + 1) % 21),
                      (-1 if index % 3 else 1) * (1 + index % 16)) for index in range(143)]
            edges += [("e3", "n2", "n7", -16), ("e0", "n0", "n1", 0)]
            graphs = [PagedRecallGraph(root / name, reserve=lambda *_args: True)
                      for name in ("original", "batched")]
            for graph in graphs:
                cleanup.callback(graph.close)
            observed = []
            connect = nodes._connect

            def traced_connection():
                connection = connect()
                connection.set_trace_callback(observed.append)
                return connection

            with patch.object(nodes, "_connect", traced_connection):
                graphs[0].load(iter(edges), node_exists=lambda key: key in nodes,
                               synapse_revision=1, neuron_revision=revision, read_stores=(nodes,))
                original_queries = sum("FROM paged_neuron_records WHERE neuron_id=" in sql
                                       for sql in observed)
                observed.clear()
                pages = []

                def membership(identifiers):
                    pages.append(identifiers)
                    return nodes.contains_many(identifiers, expected_graph_revision=revision)

                graphs[1].load(iter(edges), node_exists=lambda _key: self.fail("single-ID fallback"),
                               node_exists_many=membership, synapse_revision=1,
                               neuron_revision=revision, read_stores=(nodes,))
                batched_queries = sum("FROM paged_neuron_records WHERE neuron_id IN (" in sql
                                      for sql in observed)
            self.assertGreater(original_queries, 250)
            self.assertEqual(batched_queries, 3)
            self.assertTrue(all(len(page) <= 128 for page in pages))
            self.assertEqual(graph_rows(graphs[0]), graph_rows(graphs[1]))
            self.assertEqual(list(graphs[0].connection.execute("SELECT * FROM incoming ORDER BY node_id")),
                             list(graphs[1].connection.execute("SELECT * FROM incoming ORDER BY node_id")))
            self.assertEqual((graphs[0].eligible_edges, graphs[0].inhibitory_edges, graphs[0].next_ordinal),
                             (graphs[1].eligible_edges, graphs[1].inhibitory_edges, graphs[1].next_ordinal))
            self.assertEqual(hashlib.sha256(nodes.path.read_bytes()).hexdigest(), before)

    def test_revision_change_after_checked_page_cannot_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nodes = PagedNeuronMetadata(root / "nodes.sqlite3")
            nodes.import_page([neuron(0), neuron(1)], max_rows=2)
            revision = nodes.graph_revision
            graph = PagedRecallGraph(root / "graph", reserve=lambda *_args: True)

            def stale_membership(identifiers):
                present = nodes.contains_many(identifiers, expected_graph_revision=revision)
                with sqlite3.connect(nodes.path) as outside:
                    outside.execute("UPDATE paged_neuron_meta SET value=CAST(value AS INTEGER)+1 "
                                    "WHERE key='graph_revision'")
                return present

            try:
                with self.assertRaisesRegex(ValueError, "node generation changed"):
                    graph.load([("e", "n0", "n1", 1)], node_exists=lambda key: key in nodes,
                               node_exists_many=stale_membership, synapse_revision=1,
                               neuron_revision=revision, read_stores=(nodes,))
                self.assertIsNone(graph.applied_neuron_revision)
                self.assertEqual(graph_rows(graph), [])
                with self.assertRaisesRegex(ValueError, "node generation changed"):
                    nodes.contains_many(("n0",), expected_graph_revision=revision)
            finally:
                graph.close()

    def test_cancellation_during_page_fetch_stops_before_membership_or_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = PagedRecallGraph(root / "graph", reserve=lambda *_args: True)
            cancelled = [False]
            read = []

            def edges():
                for index in range(100):
                    read.append(index)
                    yield ("e%d" % index, "a", "b", 1)
                    if index == 6:
                        cancelled[0] = True

            try:
                with self.assertRaises(InterruptedError):
                    graph.load(edges(), node_exists=lambda _key: True,
                               node_exists_many=lambda _ids: self.fail("cancelled page was queried"),
                               synapse_revision=1, neuron_revision=1,
                               cancelled=lambda: cancelled[0])
                self.assertLessEqual(len(read), 8)
                self.assertEqual(graph_rows(graph), [])
                self.assertIsNone(graph.applied_neuron_revision)
            finally:
                graph.close()

    def test_membership_bound_and_ram_refusal_precede_sql_payload_query(self):
        with tempfile.TemporaryDirectory() as folder:
            nodes = PagedNeuronMetadata(Path(folder) / "nodes.sqlite3")
            nodes.import_page([neuron(0)], max_rows=1)
            revision = nodes.graph_revision
            with self.assertRaisesRegex(ValueError, "membership page is invalid"):
                nodes.contains_many(tuple("n%d" % index for index in range(129)),
                                    expected_graph_revision=revision)
            nodes._memory_reserve = lambda _size, _label: False
            with patch.object(nodes, "_connect", side_effect=AssertionError("refused page opened SQL")):
                with self.assertRaises(NeuronMetadataResourcePause):
                    nodes.contains_many(("n0",), expected_graph_revision=revision)

    def test_unknown_callback_membership_and_page_resource_refusal_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            graph = PagedRecallGraph(root / "invalid", reserve=lambda *_args: True)
            refused = PagedRecallGraph(root / "refused", reserve=lambda _size, label:
                                       label != "recall graph endpoint page")
            try:
                with self.assertRaisesRegex(ValueError, "returned invalid IDs"):
                    graph.load([("e", "a", "b", 1)], node_exists=lambda _key: True,
                               node_exists_many=lambda _ids: frozenset({"unrequested"}),
                               synapse_revision=1, neuron_revision=1)
                self.assertEqual(graph_rows(graph), [])
                with self.assertRaisesRegex(RuntimeError, "paused at resource reserve"):
                    refused.load(iter([("e", "a", "b", 1)]), node_exists=lambda _key: True,
                                 node_exists_many=lambda _ids: self.fail("refused page queried"),
                                 synapse_revision=1, neuron_revision=1)
                self.assertEqual(graph_rows(refused), [])
                self.assertIsNone(refused.applied_neuron_revision)
            finally:
                graph.close()
                refused.close()
