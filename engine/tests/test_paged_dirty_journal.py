"""Pure SQLite/storage checks for the generation-bound dirty journal."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.live_paging_migration import migrate_live_substrate_to_paged
from omni_core.paged_dirty_journal import (
    DirtyJournalResourcePause,
    install_generation_journal,
    iter_dirty_ids,
    journal_status,
)
from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.vsa import NeuralSubstrate


class PagedDirtyJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(prefix="omni-dirty-journal-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.store = self.root / "substrate"
        self.memory = NeuralSubstrate(16, seed=41)
        self.row = torch.tensor([1, 0, -1, 0] * 4, dtype=torch.int8)
        for number in range(2):
            neuron_id = "neuron-%d" % number
            assembly_id = "assembly-%d" % number
            for identifier in (neuron_id, assembly_id):
                self.memory.neurons[identifier] = {
                    "id": identifier, "label": identifier,
                }
                self.memory.neuron_vectors[identifier] = self.row
            self.memory.assemblies.append({
                "id": assembly_id,
                "fingerprint": "fingerprint-%d" % number,
                "neuron_ids": [neuron_id],
                "rehearsals": 1,
            })
            self.memory.assembly_vectors.link(assembly_id)
        self.pointer = self.memory.save_sharded(self.store, records_per_shard=2)
        migrated = migrate_live_substrate_to_paged(self.memory, self.root / "cache")
        self.path = Path(migrated["path"])
        self.neurons = PagedNeuronMetadata(self.path)
        self.neurons.import_page(list(self.memory.neurons.values()))
        self.memory.assemblies.index.bind_committed_generation(
            self.pointer["activeGeneration"]
        )
        self.neurons.bind_committed_generation(self.pointer["activeGeneration"])

    def test_triggers_capture_same_transaction_changes_and_prior_placement(self) -> None:
        ready = install_generation_journal(self.path, self.store, self.pointer)
        self.assertEqual(ready["state"], "ready")
        self.assertEqual(ready["dirtyIds"], 0)
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE paged_neuron_records SET record_json=record_json "
                "WHERE neuron_id='neuron-0'"
            )
            connection.rollback()
        self.assertEqual(journal_status(self.path, self.pointer)["dirtyIds"], 0)
        self.neurons.edit_by_id("neuron-0", lambda row: row.update(label="revised"))
        self.memory.neuron_vectors.set_packed("neuron-1", b"\x55" * 4)
        with self.memory.assemblies.transaction(max_rows=1) as edit:
            edit.edit_by_id("assembly-0", lambda row: row.update(rehearsals=2))
        dirty = list(iter_dirty_ids(self.path, self.pointer))
        self.assertEqual([(kind, identifier) for kind, identifier, _, _ in dirty], [
            ("assemblies", "assembly-0"),
            ("neurons", "neuron-0"),
            ("neurons", "neuron-1"),
        ])
        self.assertTrue(all(bucket is not None and part is not None
                            for _, _, bucket, part in dirty))

    def test_interrupted_base_build_remains_fail_closed_and_preserves_new_dirty_id(self) -> None:
        def deny_membership(_size, operation):
            return operation != "dirty journal base membership"

        with self.assertRaises(DirtyJournalResourcePause):
            install_generation_journal(
                self.path, self.store, self.pointer,
                disk_reserve=deny_membership,
            )
        self.assertEqual(journal_status(self.path, self.pointer)["state"], "building")
        with self.assertRaisesRegex(ValueError, "building"):
            list(iter_dirty_ids(self.path, self.pointer))
        self.neurons.edit_by_id("neuron-0", lambda row: row.update(label="after-pause"))
        resumed = install_generation_journal(self.path, self.store, self.pointer)
        self.assertEqual(resumed["state"], "ready")
        self.assertIn(("neurons", "neuron-0"), [
            (kind, identifier) for kind, identifier, _, _
            in iter_dirty_ids(self.path, self.pointer)
        ])

    def test_new_record_has_no_prior_part_and_decay_flags_full_neuron_rewrite(self) -> None:
        install_generation_journal(self.path, self.store, self.pointer)
        self.neurons["new-neuron"] = {"id": "new-neuron", "label": "new"}
        self.assertIn(("neurons", "new-neuron", None, None),
                      list(iter_dirty_ids(self.path, self.pointer)))
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO paged_neuron_meta(key,value) VALUES('decay_epoch','1') "
                "ON CONFLICT(key) DO UPDATE SET value='1'"
            )
        self.assertTrue(journal_status(self.path, self.pointer)["fullNeuronsDirty"])
        with self.assertRaisesRegex(ValueError, "all neuron shards"):
            list(iter_dirty_ids(self.path, self.pointer))

    def test_install_requires_one_clean_shared_three_store_boundary(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE paged_neuron_meta SET value='-1' "
                "WHERE key='committed_revision'"
            )
        with self.assertRaisesRegex(ValueError, "cleanly bound"):
            install_generation_journal(self.path, self.store, self.pointer)
        with sqlite3.connect(self.path) as connection:
            self.assertIsNone(connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='substrate_dirty_meta'"
            ).fetchone())


if __name__ == "__main__":
    unittest.main()
