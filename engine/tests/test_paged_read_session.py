from pathlib import Path
import sqlite3
import copy
import tempfile
from types import SimpleNamespace
import unittest

import torch

from omni_core.connected_transaction import _ACTIVE, _Participant
from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.paged_packed_vectors import PagedPackedVectors
from omni_core.paged_read_session import ReadSession
from omni_core.paged_recall_graph import PagedRecallGraph


def neuron(index):
    return {'id': 'n%d' % index, 'neuron_id': 'n%d' % index, 'label': 'item%d' % index,
        'region': 'semantic', 'activation': 0., 'importance': .2, 'uncertainty': .5,
        'exposures': 1, 'aliases': []}


class PagedReadSessionTests(unittest.TestCase):
    def test_actual_graph_entry_rejects_changed_node_generation_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); nodes = PagedNeuronMetadata(root / 'nodes.sqlite3')
            nodes.import_page([neuron(0), neuron(1)], max_rows=2)
            revision = nodes.graph_revision
            graph = PagedRecallGraph(root / 'graph', reserve=lambda *_args: True)
            def edges():
                yield ('e0', 'n0', 'n1', 1)
                with sqlite3.connect(nodes.path) as outside:
                    outside.execute("UPDATE paged_neuron_meta SET value=CAST(value AS INTEGER)+1 WHERE key='graph_revision'")
                yield ('e1', 'n1', 'n0', -1)
            try:
                with self.assertRaisesRegex(ValueError, 'node generation changed'):
                    graph.load(edges(), node_exists=lambda key: key in nodes, synapse_revision=1,
                        neuron_revision=revision, read_stores=(nodes,))
                self.assertIsNone(graph.applied_neuron_revision)
            finally:
                graph.close()

    def test_ordinary_native_readonly_recall_and_export_have_exact_affected_path_parity(self):
        from omni_core.vsa import NeuralSubstrate
        from omni_core.paged_assembly_index import PagedAssemblyIndex
        from omni_core.memory_inference_export import vector_tensor_spec
        from omni_core.ternary_packing import export_ternary_shards, verify_ternary_shards
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); memory = NeuralSubstrate(16, seed=31)
            memory.learn('alpha red language sequence')
            memory.learn('beta green language pattern')
            memory.learn('gamma blue related sequence')
            nodes = PagedNeuronMetadata(root / 'nodes.sqlite3')
            nodes.import_page(list(memory.neurons.values()), max_rows=len(memory.neurons))
            memory.neurons = nodes
            memory.enable_paged_vectors(root / 'vectors.sqlite3', shard_rows=8)
            index = PagedAssemblyIndex(root / 'assemblies.sqlite3')
            memory.enable_paged_assemblies(index, page_size=2)
            cue = memory.vector_for_text('alpha language pattern')
            try:
                first, recalled = memory._recall_vector_impl(cue, workspace_slots=4, record_activity=False)
                first_rows, first_audit = list(recalled), copy.deepcopy(memory._last_recall_audit)
                memory.release_paged_recall_scratch()
                with ReadSession([nodes, index, memory.neuron_vectors]) as read:
                    second, recalled = memory.recall_vector(cue, workspace_slots=4, record_activity=False)
                    second_rows, second_audit = list(recalled), copy.deepcopy(memory._last_recall_audit)
                    self.assertEqual(read.opened_connections, 3)
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first_rows, second_rows); self.assertEqual(first_audit, second_audit)
                with self.assertRaises(InterruptedError):
                    memory.recall_vector(cue, workspace_slots=4, record_activity=False, cancelled=lambda: True)
                base, binding = vector_tensor_spec(memory.neuron_vectors)
                export_ternary_shards(root / 'plain', [base], metadata=binding)
                with ReadSession([memory.neuron_vectors]):
                    cached, cached_binding = vector_tensor_spec(memory.neuron_vectors)
                    export_ternary_shards(root / 'cached', [cached], metadata=cached_binding)
                self.assertEqual(verify_ternary_shards(root / 'plain').manifest,
                    verify_ternary_shards(root / 'cached').manifest)
            finally:
                memory.release_paged_recall_scratch()

    def test_actual_graph_load_node_validation_matches_and_preserves_exact_edge_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); nodes = PagedNeuronMetadata(root / 'nodes.sqlite3')
            nodes.import_page([neuron(i) for i in range(12)], max_rows=12)
            edges = [('e%d' % i, 'n%d' % (i % 13), 'n%d' % ((i + 1) % 13), (-1 if i % 3 else 1) * (1 + i % 16)) for i in range(96)]
            initial = nodes.status()
            baseline = PagedRecallGraph(root / 'baseline', reserve=lambda *_args: True)
            reused = PagedRecallGraph(root / 'reused', reserve=lambda *_args: True)
            try:
                baseline.load(iter(edges), node_exists=lambda key: key in nodes,
                    synapse_revision=1, neuron_revision=nodes.graph_revision)
                with ReadSession([nodes]) as scope:
                    reused.load(iter(edges), node_exists=lambda key: key in nodes,
                        synapse_revision=1, neuron_revision=nodes.graph_revision)
                    self.assertEqual(scope.opened_connections, 1)
                self.assertEqual(list(baseline.connection.execute('SELECT * FROM raw_edges ORDER BY ordinal')),
                    list(reused.connection.execute('SELECT * FROM raw_edges ORDER BY ordinal')))
                self.assertEqual((baseline.eligible_edges, baseline.inhibitory_edges),
                    (reused.eligible_edges, reused.inhibitory_edges))
                self.assertEqual(nodes.status(), initial)
            finally:
                baseline.close(); reused.close()

    def test_stale_generation_cancel_replacement_and_write_controls_remain_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); vectors = PagedPackedVectors(root / 'vectors.sqlite3', 4)
            vectors['a'] = torch.tensor([1, 0, -1, 1])
            cancel = [False]
            with ReadSession([vectors], cancelled=lambda: cancel[0]):
                snapshot = vectors.current_snapshot()
                vectors.page_rows(snapshot, None, 1)
                # An independent SQLite writer remains visible at the next
                # operation; no keeper-long stale snapshot masks revision.
                with sqlite3.connect(vectors.path) as outside:
                    outside.execute("UPDATE paged_vector_meta SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")
                with self.assertRaisesRegex(ValueError, 'generation drift'):
                    vectors.page_rows(snapshot, None, 1)
                with self.assertRaisesRegex(RuntimeError, 'read-only'):
                    vectors['a'] = torch.ones(4)
                with vectors._transaction() as connection:
                    with self.assertRaisesRegex(RuntimeError, 'read-only'):
                        connection.execute('PRAGMA query_only=OFF')
                    with self.assertRaises(sqlite3.DatabaseError):
                        connection.execute('-- a comment cannot disable native read authority\nPRAGMA query_only=OFF')
                cancel[0] = True
                with self.assertRaises(InterruptedError):
                    vectors.current_snapshot()
            # A real replacement must not keep reading the unlinked old file.
            replacement = root / 'replacement.sqlite3'
            with sqlite3.connect(replacement) as incoming:
                incoming.execute('CREATE TABLE unrelated(value)')
            with ReadSession([vectors]):
                vectors.current_snapshot()
                replacement.replace(vectors.path)
                with self.assertRaisesRegex(ValueError, 'replaced'):
                    vectors.current_snapshot()

    def test_primary_connected_write_transaction_takes_precedence_and_rolls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'nodes.sqlite3'; nodes = PagedNeuronMetadata(path)
            nodes['n0'] = neuron(0)
            connection = sqlite3.connect(path, isolation_level=None)
            participant = _Participant(connection)
            scope = SimpleNamespace(allowed_databases={str(path.resolve())}, databases={str(path.resolve()): participant})
            token = _ACTIVE.set(scope)
            try:
                with ReadSession([nodes]) as read:
                    nodes['n1'] = neuron(1)
                    self.assertIn('n1', nodes)
                    self.assertEqual(read.opened_connections, 0)
                connection.rollback()
            finally:
                _ACTIVE.reset(token); connection.close()
            self.assertNotIn('n1', nodes)

    def test_keeper_ram_lease_is_admitted_and_released_on_exception(self):
        class Policy:
            def __init__(self): self.reserved, self.released = [], []
            def require_disk(self, *_args): return None
            def check_active_pressure(self, *_args): return None
            def reserve_ram(self, size, label):
                self.reserved.append((size, label)); owner = self
                return SimpleNamespace(release=lambda: owner.released.append(size))
        with tempfile.TemporaryDirectory() as folder:
            policy = Policy(); nodes = PagedNeuronMetadata(Path(folder) / 'nodes.sqlite3', resource_policy=policy)
            nodes['n0'] = neuron(0)
            with self.assertRaisesRegex(ValueError, 'leave safely'):
                with ReadSession([nodes]):
                    self.assertIn('n0', nodes)
                    raise ValueError('leave safely')
            self.assertEqual(policy.reserved, [(320 * 1024, 'native read keeper cache')])
            self.assertEqual(policy.released, [320 * 1024])

    def test_native_export_keeper_does_not_escape_generator_yield_or_hide_revision_change(self):
        from omni_core.memory_inference_export import vector_tensor_spec
        with tempfile.TemporaryDirectory() as folder:
            vectors = PagedPackedVectors(Path(folder) / 'vectors.sqlite3', 4)
            vectors['a'] = torch.tensor([1, 0, -1, 1]); vectors['b'] = torch.ones(4)
            spec, _ = vector_tensor_spec(vectors)
            pages = spec.values.pages()
            next(pages)
            # This is a genuine independent source mutation, not a fake stale
            # token. The read keeper must not leak into the consumer's writer.
            vectors['b'] = -torch.ones(4)
            with self.assertRaisesRegex(ValueError, 'generation changed'):
                next(pages)


if __name__ == '__main__': unittest.main()
