"""Pure storage checks for all-or-nothing live packed/assembly paging."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.live_paging_migration import (
    migrate_live_substrate_to_paged,
    prune_abandoned_live_paging_cache,
)
from omni_core.committed_paged_cache import rebuild_committed_paged_cache
from omni_core.persistence import atomic_write_json
from omni_core.packed_vsa_vectors import PackedTernaryVectors
from omni_core.paged_assembly_vector_view import PagedAssemblyVectorView
from omni_core.paged_assembly_view import PagedAssemblyView
from omni_core.paged_packed_vectors import PagedPackedVectors
from omni_core.vsa import NeuralSubstrate, SubstrateResourcePause


class LivePagingMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(prefix="omni-live-page-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.cache = self.root / "cache"
        self.memory = NeuralSubstrate(16, seed=31)
        row = torch.tensor([1, 0, -1, 0] * 4, dtype=torch.int8)
        for number in range(7):
            neuron = "neuron-%d" % number
            assembly = "assembly-%d" % number
            for identifier in (neuron, assembly):
                self.memory.neurons[identifier] = {
                    "id": identifier, "label": identifier,
                }
                self.memory.neuron_vectors[identifier] = row
            self.memory.assemblies.append({
                "id": assembly,
                "fingerprint": "fingerprint-%d" % number,
                "neuron_ids": [neuron],
                "source_provenance": {"sensor": 1},
                "rehearsals": 1,
            })
            self.memory.assembly_vectors.link(assembly)

    def test_migration_uses_one_shared_cache_and_paged_views(self) -> None:
        original_rows = {
            identifier: self.memory.neuron_vectors.packed_row(identifier)
            for identifier in self.memory.neurons
        }
        result = migrate_live_substrate_to_paged(self.memory, self.cache)
        self.assertFalse(result["alreadyPaged"])
        self.assertEqual(result["neuronCount"], 14)
        self.assertEqual(result["assemblyCount"], 7)
        self.assertIsInstance(self.memory.neuron_vectors, PagedPackedVectors)
        self.assertIsInstance(self.memory.assemblies, PagedAssemblyView)
        self.assertIsInstance(self.memory.assembly_vectors, PagedAssemblyVectorView)
        self.assertIs(self.memory.assemblies.index._vectors, self.memory.neuron_vectors)
        self.assertIs(self.memory.assembly_vectors.backing, self.memory.neuron_vectors)
        self.assertEqual(list(self.memory.assemblies)[-1]["id"], "assembly-6")
        for identifier, packed in original_rows.items():
            self.assertEqual(self.memory.neuron_vectors.packed_row(identifier), packed)
        self.assertFalse((self.cache / ".live-paged-staging").exists())
        receipt = json.loads(
            (self.cache / "live-paged" / "receipt.json").read_text("utf-8")
        )
        self.assertFalse(receipt["authoritative"])
        self.assertTrue(migrate_live_substrate_to_paged(self.memory, self.cache)["alreadyPaged"])
        pointer = self.memory.save_sharded(self.root / "substrate", records_per_shard=2)
        restored = NeuralSubstrate.load_sharded(
            self.root / "substrate",
            self.memory.metadata(include_records=False),
            lazy_synapses=False,
        )
        self.assertEqual(pointer["counts"]["assemblies"], 7)
        self.assertEqual(len(restored.assemblies), 7)

    def test_denied_disk_reserve_keeps_live_state_and_creates_no_orphan(self) -> None:
        source_vectors = self.memory.neuron_vectors
        source_assemblies = self.memory.assemblies
        with self.assertRaises(SubstrateResourcePause):
            migrate_live_substrate_to_paged(
                self.memory,
                self.cache,
                disk_reserve=lambda _size, _operation: False,
            )
        self.assertIs(self.memory.neuron_vectors, source_vectors)
        self.assertIs(self.memory.assemblies, source_assemblies)
        self.assertFalse(self.cache.exists())

    def test_invalid_late_assembly_rolls_back_and_removes_only_own_stage(self) -> None:
        self.memory.assemblies[-1]["source_text"] = "private source passage"
        source_vectors = self.memory.neuron_vectors
        source_assemblies = self.memory.assemblies
        with self.assertRaisesRegex(ValueError, "non-structural"):
            migrate_live_substrate_to_paged(self.memory, self.cache)
        self.assertIs(self.memory.neuron_vectors, source_vectors)
        self.assertIs(self.memory.assemblies, source_assemblies)
        self.assertIsInstance(self.memory.neuron_vectors, PackedTernaryVectors)
        self.assertFalse((self.cache / "live-paged").exists())
        self.assertFalse((self.cache / ".live-paged-staging").exists())

    def test_stale_owned_path_fails_closed_without_creating_new_staging(self) -> None:
        self.cache.mkdir()
        stale = self.cache / ".live-paged-staging"
        stale.mkdir()
        (stale / "do-not-adopt").write_text("stale", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            migrate_live_substrate_to_paged(self.memory, self.cache)
        self.assertEqual((stale / "do-not-adopt").read_text("utf-8"), "stale")
        self.assertEqual([path.name for path in self.cache.iterdir()], [stale.name])

    def test_explicit_prune_requires_verified_rebuild_and_no_consumers(self) -> None:
        migrate_live_substrate_to_paged(self.memory, self.cache)
        engine = self.root / "engine"
        self.memory.save_sharded(engine / "substrate", records_per_shard=2)
        atomic_write_json(engine / "brain.json", {
            "substrate": self.memory.metadata(include_records=False)
        })
        rebuilt = rebuild_committed_paged_cache(engine, self.root / "rebuilt")
        with self.assertRaisesRegex(RuntimeError, "consumers"):
            prune_abandoned_live_paging_cache(
                self.cache, engine, verified_rebuild=rebuilt, target="live",
                no_consumers=lambda _path: False,
            )
        self.assertTrue((self.cache / "live-paged").is_dir())
        # Simulate the host releasing the old working cache after recovering
        # from the committed shards and choosing the verified replacement.
        self.memory = NeuralSubstrate.load_sharded(
            engine / "substrate",
            json.loads((engine / "brain.json").read_text("utf-8"))["substrate"],
            lazy_synapses=False,
        )
        result = prune_abandoned_live_paging_cache(
            self.cache, engine, verified_rebuild=rebuilt, target="live",
            no_consumers=lambda _path: True,
        )
        self.assertTrue(result["removed"])
        self.assertFalse((self.cache / "live-paged").exists())
        self.assertTrue(rebuilt.path.is_file())


if __name__ == "__main__":
    unittest.main()
