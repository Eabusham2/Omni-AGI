"""Small memory-graph components only; no cortical model or user corpus."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from omni_core.vsa import NeuralSubstrate
from omni_core.weighted_substrate_state import synapse_contribution
from omni_core.paged_recall_graph import PagedRecallGraph
from omni_core.paged_recurrent_spreading import PagedRecurrentState
from omni_core.packed_vsa_vectors import PackedTernaryVectors
from omni_core.paged_packed_vectors import PagedPackedVectors, PagedVectorResourcePause
from omni_core.weighted_vsa_row import CONTACT, payload_parts
from omni_core.paged_vector_scoring import prepare_exact_paged_similarity


class WeightedMemorySubstrateTests(unittest.TestCase):
    def graph(self):
        memory = NeuralSubstrate(16, seed=211)
        source = memory._ensure_neuron('first', 1.)
        target = memory._ensure_neuron('second', 1.)
        for moment in range(18):
            memory._strengthen_synapse(source, target, float(moment + 2), kind='association', amount=10.)
        return memory, source, target

    def test_repeated_coactivity_learns_independent_strength_and_contacts(self):
        memory, source, target = self.graph()
        primary = memory.synapses[f'{source}>{target}:association']
        contacts = list(memory._parallel_synapse_records(primary))
        self.assertEqual([synapse_contribution(record) for record in contacts], [16, 2])
        self.assertEqual(primary['parallel_count'], 1)
        self.assertEqual({record['effective_weight'] for record in contacts}, {1})
        self.assertTrue(all('latent_weight' not in record for record in contacts))

    def test_lazy_reload_preserves_gain_and_does_not_page_records_to_recur(self):
        memory, source, target = self.graph()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory.save_sharded(root, records_per_shard=2)
            restored = NeuralSubstrate.load_sharded(root, memory.metadata(include_records=False), lazy_synapses=True)
            with patch.object(restored.synapses, '_read_shard', side_effect=AssertionError('cold source records must stay paged')):
                edges = list(restored.synapses.iter_weighted_edges())
            self.assertEqual(edges, [(source, target, 16), (source, target, 2)])
            self.assertEqual(set(restored.synapses.observed_effective_levels()), {1})

    def test_weighted_recurrence_preserves_contraction_and_relative_input_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = PagedRecallGraph(Path(directory), reserve=lambda *_: True)
            state = PagedRecurrentState(Path(directory))
            try:
                graph.load([('a>b', 'a', 'b', 8), ('c>b', 'c', 'b', -1)], node_exists=lambda _: True,
                    synapse_revision=0, neuron_revision=0)
                self.assertEqual(dict(graph.connection.execute('SELECT * FROM incoming')), {'b': 9})
                state.use_graph(graph)
                state.load_seeds([('a', .9), ('c', .4)])
                state.settle(slots=2, adaptive_floor=.05)
                self.assertGreater(state.activation('b'), .3)
                self.assertLess(state.activation('b'), 1.)
            finally:
                state.close()
                graph.close()

    def test_integer_contact_activity_and_cosine_use_strength_and_opposing_contacts(self):
        rows = PackedTernaryVectors(4, seed=211)
        rows['one'] = torch.tensor([1, 1, 0, 0])
        rows.set_connection_state('one', bytes([7, 0, 0, 0]), CONTACT.pack(0, 0x54, 2))
        self.assertEqual(rows.weighted_levels('one').tolist(), [5, 1, 0, 0])
        self.assertTrue(torch.allclose(rows['one'], torch.nn.functional.normalize(torch.tensor([5., 1., 0., 0.]), dim=0)))
        metadata, tensors = rows.export_state()
        self.assertTrue(all(tensor.dtype == torch.uint8 for tensor in tensors.values()))
        with tempfile.TemporaryDirectory() as directory:
            paged = PagedPackedVectors(Path(directory) / 'vectors.sqlite3', 4, seed=211)
            paged.import_state(metadata, tensors)
            matches = list(prepare_exact_paged_similarity(paged, torch.tensor([1., 0., 0., 0.])).iter_matches())
            self.assertEqual(len(matches), 1)
            self.assertAlmostEqual(matches[0].score, 5. / (26. ** .5))
            self.assertEqual(paged.row_payload('one'), rows.row_payload('one'))

    def test_weighted_adaptation_is_seeded_and_reload_preserves_resistance(self):
        rows = PackedTernaryVectors(16, seed=211)
        rows['one'] = torch.ones(16)
        target = torch.tensor([3.] + [1.] * 15)
        for _ in range(15):
            rows.adapt('one', target, .6)
        self.assertGreater(rows.weighted_levels('one')[0].item(), rows.weighted_levels('one')[1].item())
        self.assertTrue(any(value >> 4 for value in rows.connection_state('one')))
        metadata, tensors = rows.export_state()
        loaded = PackedTernaryVectors.from_state(metadata, tensors)
        for _ in range(8):
            rows.adapt('one', -target, .2)
            loaded.adapt('one', -target, .2)
        self.assertEqual(rows.row_payload('one'), loaded.row_payload('one'))
        self.assertEqual(rows.update_count('one'), loaded.update_count('one'))

    def test_paged_batch_budgets_full_weighted_payload_and_preserves_on_refusal(self):
        rows = PackedTernaryVectors(16)
        rows['one'] = torch.ones(16)
        rows.set_connection_state('one', bytes(16), CONTACT.pack(0, 0x56, 15) * 100)
        with tempfile.TemporaryDirectory() as directory:
            paged = PagedPackedVectors(Path(directory) / 'vectors.sqlite3', 16)
            paged.import_state(*rows.export_state())
            before = paged.row_payload('one')
            with self.assertRaises(PagedVectorResourcePause):
                with paged.batch(max_rows=1, max_payload_bytes=256) as batch:
                    batch.adapt('one', torch.ones(16), .3)
            self.assertEqual(paged.row_payload('one'), before)
            self.assertEqual(paged.update_count('one'), 0)


if __name__ == '__main__':
    unittest.main()
