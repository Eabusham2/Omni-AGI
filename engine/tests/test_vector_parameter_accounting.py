import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from omni_core.brain import AdaptiveBrain
from omni_core.packed_vsa_vectors import PackedTernaryVectors
from omni_core.paged_packed_vectors import PagedPackedVectors
from omni_core.weighted_vsa_row import CONTACT


class VectorParameterAccountingTests(unittest.TestCase):
    def vectors(self):
        result = PackedTernaryVectors(5)
        result['a'] = torch.ones(5)
        result['b'] = torch.ones(5)
        result.set_connection_state('a', bytes(5), CONTACT.pack(0, 0x56, 2) * 3)
        return result

    def test_resident_totals_follow_mutation_without_scanning_contacts(self):
        rows = self.vectors()
        class NoScan(dict):
            def values(self):
                raise AssertionError('telemetry must not scan contact buffers')
            def __iter__(self):
                raise AssertionError('telemetry must not scan contact buffers')
        rows._parallel_records = NoScan(rows._parallel_records)
        self.assertEqual(rows.parameter_accounting()['logicalParameters'], 13)
        self.assertEqual(rows.storage_bytes, 60)
        rows.set_connection_state('a', bytes(5), CONTACT.pack(0, 0x54, 1))
        self.assertEqual(rows.parameter_accounting()['repeatedContacts'], 1)
        restored = PackedTernaryVectors.from_state(*rows.export_state())
        self.assertEqual(restored.parameter_accounting(), rows.parameter_accounting())
        del rows['a']
        self.assertEqual(rows.parameter_accounting()['logicalParameters'], 5)
        rows['c'] = torch.ones(5)
        self.assertEqual(rows.parameter_accounting()['logicalParameters'], 10)
        rows.clear()
        self.assertEqual(rows.storage_bytes, 0)

    def test_paged_totals_are_transactional_and_status_reads_only_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = PagedPackedVectors(Path(folder) / 'rows.sqlite3', 5)
            rows.import_state(*self.vectors().export_state())
            self.assertEqual(rows.parameter_accounting()['logicalParameters'], 13)
            self.assertEqual(rows.storage_bytes, 84)
            before = rows.parameter_accounting()
            with self.assertRaises(RuntimeError):
                with rows.batch() as batch:
                    batch.set('a', torch.zeros(5))
                    raise RuntimeError('interrupted')
            self.assertEqual(rows.parameter_accounting(), before)
            statements = []
            original = rows._connect
            def connect():
                connection = original()
                connection.set_trace_callback(statements.append)
                return connection
            with patch.object(rows, '_connect', side_effect=connect):
                for _ in range(4):
                    self.assertEqual(rows.parameter_accounting(), before)
            self.assertFalse(any('FROM paged_vector_rows' in query for query in statements), statements)
            del rows['a']
            self.assertEqual(rows.parameter_accounting()['logicalParameters'], 5)
            self.assertEqual(PagedPackedVectors(rows.path, 5).parameter_accounting(), rows.parameter_accounting())

    def test_old_cache_counters_rebuild_once_then_triggers_remain_required(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = PagedPackedVectors(Path(folder) / 'rows.sqlite3', 5)
            rows.import_state(*self.vectors().export_state())
            before = rows.parameter_accounting()
            with sqlite3.connect(rows.path) as connection:
                connection.execute("DELETE FROM paged_vector_meta WHERE key IN ('vector_payload_bytes','vector_weighted_rows','vector_repeated_contacts')")
                for event in ('insert', 'delete', 'update'):
                    connection.execute('DROP TRIGGER weighted_vector_count_' + event)
            reopened = PagedPackedVectors(rows.path, 5)
            self.assertEqual(reopened.parameter_accounting(), before)
            with sqlite3.connect(rows.path) as connection:
                connection.execute('DROP TRIGGER weighted_vector_count_update')
            with self.assertRaisesRegex(ValueError, 'trigger'):
                PagedPackedVectors(rows.path, 5)

    def test_brain_counts_repeated_vector_contacts_and_connected_aliases_once(self):
        rows = self.vectors()
        brain = object.__new__(AdaptiveBrain)
        for name in ('decoder', 'memory_bridge', 'idea_adapter', 'router', 'liquid', 'modalities'):
            setattr(brain, name, nn.Identity())
        brain.memory = SimpleNamespace(neuron_vectors=rows, synapses={}, space=SimpleNamespace(dimensions=5))
        other = SimpleNamespace(neuron_vectors=PackedTernaryVectors(3), synapses={}, space=SimpleNamespace(dimensions=3))
        other.neuron_vectors['c'] = torch.ones(3)
        brain._connected_regions = SimpleNamespace(iter_memories=lambda: iter([('alias', brain.memory), ('other', other)]))
        result = brain.parameter_accounting()
        self.assertEqual(result['substrateBaseVectorParameters'], 13)
        self.assertEqual(result['substrateRepeatedVectorContacts'], 3)
        self.assertEqual(result['substrateVectorParameters'], 16)
        self.assertEqual(result['totalNeuralParameters'], 16)
        self.assertEqual(result['substrateVectorLearnedStateBytes'], 24)
        self.assertEqual(result['substrateVectorStructuralStateBytes'], 48)


if __name__ == '__main__':
    unittest.main()
