"""Source contract for moving one packed VSA authority to paged rows."""

import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.paged_packed_vectors import PagedPackedVectors
from omni_core.vsa import NeuralSubstrate


class PagedVSASwitchTest(unittest.TestCase):
    @staticmethod
    def substrate() -> NeuralSubstrate:
        memory = NeuralSubstrate(16, seed=13)
        memory.neurons["assembly"] = {"id": "assembly"}
        memory.neuron_vectors["assembly"] = torch.tensor(
            [1, 0, -1, 0, 1, 0, -1, 0] * 2, dtype=torch.int8
        )
        memory.assemblies.append({"id": "assembly", "neuron_ids": []})
        memory.assembly_vectors.link("assembly")
        return memory

    def test_paging_preserves_one_packed_assembly_neuron_row(self) -> None:
        memory = self.substrate()
        before = memory.neuron_vectors.packed_row("assembly")
        with tempfile.TemporaryDirectory(prefix="omni-paged-vsa-") as folder:
            path = Path(folder) / "vectors.sqlite3"
            result = memory.enable_paged_vectors(path, shard_rows=1)
            self.assertFalse(result["alreadyPaged"])
            self.assertEqual(result["vectorCount"], 1)
            self.assertIsInstance(memory.neuron_vectors, PagedPackedVectors)
            self.assertIs(memory.assembly_vectors.backing, memory.neuron_vectors)
            self.assertEqual(memory.neuron_vectors.packed_row("assembly"), before)
            self.assertEqual(memory.assembly_vectors.backing.packed_row("assembly"), before)
            self.assertTrue(memory.enable_paged_vectors(path)["alreadyPaged"])

    def test_existing_unverified_cache_cannot_be_adopted(self) -> None:
        memory = self.substrate()
        original = memory.neuron_vectors
        with tempfile.TemporaryDirectory(prefix="omni-paged-vsa-") as folder:
            path = Path(folder) / "vectors.sqlite3"
            path.write_bytes(b"unverified")
            with self.assertRaises(FileExistsError):
                memory.enable_paged_vectors(path)
        self.assertIs(memory.neuron_vectors, original)

    def test_sharded_load_streams_vectors_into_empty_paged_cache(self) -> None:
        memory = NeuralSubstrate(16, seed=23)
        memory.learn("alpha beta gamma delta")
        with tempfile.TemporaryDirectory(prefix="omni-paged-vsa-") as folder:
            root = Path(folder)
            memory.save_sharded(root / "substrate", records_per_shard=2)
            original_rows = {
                identifier: memory.neuron_vectors.packed_row(identifier)
                for identifier in memory.neuron_vectors
            }
            paged = PagedPackedVectors(
                root / "paged.sqlite3", memory.space.dimensions,
                seed=memory.space.seed,
                zero_deadband=memory.neuron_vectors.zero_deadband,
            )
            restored = NeuralSubstrate.load_sharded(
                root / "substrate", memory.metadata(include_records=False),
                paged_vectors=paged,
            )
            self.assertIs(restored.neuron_vectors, paged)
            self.assertIs(restored.assembly_vectors.backing, paged)
            self.assertEqual(len(paged), len(original_rows))
            for identifier, expected in original_rows.items():
                self.assertEqual(paged.packed_row(identifier), expected)


if __name__ == "__main__":
    unittest.main()
