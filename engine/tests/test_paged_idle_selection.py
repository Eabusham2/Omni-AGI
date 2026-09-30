"""Constructor-free scheduling metadata/index fixtures, not cognition runs."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.paged_assembly_vector_view import PagedAssemblyVectorView
from omni_core.paged_assembly_view import PagedAssemblyView
from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.paged_idle_selection import PagedIdleSelector
from omni_core.bounded_idle_selection import select_idle_workspace


class PagedIdleSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="omni-idle-query-storage-")
        self.addCleanup(directory.cleanup)
        index = PagedAssemblyIndex(Path(directory.name) / "source.sqlite3", dimensions=16, seed=9)
        self.memory = SimpleNamespace(assemblies=PagedAssemblyView(index), neurons=PagedNeuronMetadata(index.path),
                                      neuron_vectors=index._vectors, attention_legacy_raw_active=True,
                                      attention_active_neuron_ids=set())
        self.memory.assembly_vectors = PagedAssemblyVectorView(index, index._vectors)
        for number in range(400):
            identifier = "assembly-%04d" % number
            self.memory.assemblies.append({"id": identifier, "fingerprint": identifier,
                                           "neuron_ids": [], "importance": (number % 13) / 13.0,
                                           "rehearsals": number % 7})
            self.memory.neurons[identifier] = {"id": identifier, "activation": (number % 9) / 9.0,
                                               "uncertainty": (number % 5) / 5.0, "exposures": number % 11}
            index._vectors.set_packed(identifier, b"\xaa" * 4)
        # SimpleNamespace is not weak-referenceable; this shell owns only
        # source storage, no neural class/model constructor is used.
        class StorageShell:
            pass
        shell = StorageShell()
        shell.__dict__.update(self.memory.__dict__)
        self.memory = shell
        self.selector = PagedIdleSelector(shell)
        self.addCleanup(self.selector.close)

    def expected(self, capacity):
        def score(row):
            node = self.memory.neurons.get(row["id"], {})
            activation = node.get("activation", 0.0) if self.memory.attention_legacy_raw_active or row["id"] in self.memory.attention_active_neuron_ids else 0.0
            return activation * 0.35 + row.get("importance", 0.0) * 0.30 + node.get("uncertainty", 0.5) * 0.20 + 1.0 / (1.0 + row.get("rehearsals", 0)) * 0.15
        return [row["id"] for row in select_idle_workspace(self.memory.assemblies, capacity=capacity, score=score,
                                                          eligible=lambda row: row["id"] in self.memory.assembly_vectors)]

    def test_exact_ranking_and_warm_selection_decode_no_source_pages(self):
        actual = [row["id"] for row in self.selector.select(9)]
        self.assertEqual(actual, self.expected(9))
        self.assertEqual(self.selector.last_source_rows_read, 0)

    def test_metadata_vector_and_deletion_changes_update_only_affected_groups(self):
        self.memory.neurons.edit_by_id("assembly-0000", lambda row: row.update(activation=4.0, uncertainty=0.3))
        with self.memory.assemblies.transaction(max_rows=1) as edit:
            edit.edit_by_id("assembly-0001", lambda row: row.update(importance=3.0))
        del self.memory.neuron_vectors["assembly-0002"]
        actual = [row["id"] for row in self.selector.select(5)]
        self.assertEqual(actual, self.expected(5))
        self.assertLessEqual(self.selector.last_source_rows_read, self.selector.PAGE_ROWS)
        self.assertIn("assembly-0000", actual)

    def test_global_decay_and_attention_mask_preserve_exact_old_math_without_rewrite(self):
        self.memory.neurons.decay(0.37)
        self.memory.neurons.edit_by_id("assembly-0150", lambda row: row.update(activation=0.73))
        self.memory.neurons.decay(0.81)
        self.memory.attention_legacy_raw_active = False
        self.memory.attention_active_neuron_ids.update({"assembly-0150", "assembly-0151"})
        self.assertEqual([row["id"] for row in self.selector.select(13)], self.expected(13))
        self.memory.neurons.decay(1.0)
        self.assertEqual([row["id"] for row in self.selector.select(13)], self.expected(13))
        self.assertEqual(self.selector.last_source_rows_read, 0)

    def test_missing_authenticated_change_causes_full_actual_source_rebuild(self):
        self.memory.neurons.edit_by_id("assembly-0000", lambda row: row.update(activation=4.0))
        self.selector.connection.execute("DELETE FROM authenticated_merkle_nodes WHERE namespace='idle-priority-changes'")
        self.selector.connection.commit()
        self.assertEqual([row["id"] for row in self.selector.select(3)], self.expected(3))
        self.assertGreater(self.selector.last_source_rows_read, self.selector.PAGE_ROWS)

    def test_refused_optional_index_write_never_discards_valid_source_changes(self):
        old = self.selector.source._reserve_disk
        self.selector.source._reserve_disk = lambda *_args: (_ for _ in ()).throw(RuntimeError("reserve"))
        try:
            self.memory.neurons.edit_by_id("assembly-0000", lambda row: row.update(activation=4.0))
        finally:
            self.selector.source._reserve_disk = old
        self.assertEqual(self.memory.neurons["assembly-0000"]["activation"], 4.0)
        self.assertTrue(self.selector.invalid)
        self.assertEqual([row["id"] for row in self.selector.select(3)], self.expected(3))


if __name__ == "__main__":
    unittest.main()
