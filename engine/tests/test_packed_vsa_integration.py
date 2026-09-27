"""Source/storage contracts for VSA's one packed neuron/assembly authority.

These checks are not a trained-brain, recall-quality, or live app acceptance run.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.packed_vsa_vectors import PackedTernaryVectors, PackedTernaryVectorView
from omni_core.persistence import load_tensors
from omni_core.vsa import NeuralSubstrate


class PackedVSAIntegrationTests(unittest.TestCase):
    def test_assembly_and_neuron_views_share_one_packed_row(self):
        memory = NeuralSubstrate(dimensions=32, seed=271)
        learned = memory.learn("Copper lanterns crossed the violet harbor.")
        identifier = learned["assembly_id"]
        self.assertIsInstance(memory.neuron_vectors, PackedTernaryVectors)
        self.assertIsInstance(memory.assembly_vectors, PackedTernaryVectorView)
        self.assertIs(memory.assembly_vectors.backing, memory.neuron_vectors)
        self.assertEqual(len(memory.neuron_vectors), len(memory.neurons))
        self.assertEqual(len(memory.assembly_vectors), len(memory.assemblies))
        self.assertTrue(torch.equal(
            memory.neuron_vectors[identifier], memory.assembly_vectors[identifier]
        ))
        self.assertTrue(torch.allclose(
            memory.assembly_vectors[identifier].norm(), torch.tensor(1.0)
        ))
        levels = memory.neuron_vectors.levels(identifier)
        self.assertEqual(levels.dtype, torch.int8)
        self.assertTrue(bool(((levels >= -1) & (levels <= 1)).all()))
        self.assertFalse(any(
            isinstance(value, torch.Tensor)
            for value in vars(memory.neuron_vectors).values()
        ))

        memory._adapt_assembly_vector(identifier, -memory.assembly_vectors[identifier], 1.0)
        self.assertTrue(torch.equal(
            memory.neuron_vectors.levels(identifier), -levels
        ))
        self.assertTrue(torch.equal(
            memory.neuron_vectors[identifier], memory.assembly_vectors[identifier]
        ))

    def test_non_sharded_state_roundtrips_only_packed_rows_and_rejects_float(self):
        memory = NeuralSubstrate(dimensions=32, seed=273)
        learned = memory.learn("Violet trams crossed the copper bridge.")
        identifier = learned["assembly_id"]
        memory._adapt_assembly_vector(identifier, -memory.assembly_vectors[identifier], 1.0)
        metadata = memory.metadata(include_records=True)
        tensors = memory.tensor_state()
        self.assertEqual(set(tensors), {
            "substrate.vectors.packed_rows",
            "substrate.vectors.update_counters_le",
        })
        self.assertTrue(all(value.dtype == torch.uint8 for value in tensors.values()))
        restored = NeuralSubstrate.from_state(metadata, tensors)
        self.assertEqual(
            restored.neuron_vectors.packed_row(identifier),
            memory.neuron_vectors.packed_row(identifier),
        )
        self.assertEqual(
            restored.neuron_vectors.update_count(identifier),
            memory.neuron_vectors.update_count(identifier),
        )
        self.assertIs(restored.assembly_vectors.backing, restored.neuron_vectors)

        with self.assertRaisesRegex(ValueError, "legacy float VSA"):
            NeuralSubstrate.from_state(
                metadata,
                dict(tensors, **{"substrate.neuron_vectors": torch.zeros(1, 32)}),
            )
        with self.assertRaisesRegex(ValueError, "legacy higher-precision VSA"):
            NeuralSubstrate.from_state(dict(metadata, schema="neural-substrate-1"), tensors)

    def test_sharded_generation_has_no_float_vector_tensor_or_duplicate_assembly_row(self):
        memory = NeuralSubstrate(dimensions=32, seed=277)
        learned = memory.learn("A copper tram reached the violet harbor.")
        identifier = learned["assembly_id"]
        memory._adapt_assembly_vector(identifier, -memory.assembly_vectors[identifier], 1.0)
        with tempfile.TemporaryDirectory(prefix="omni-packed-vsa-integrated-") as folder:
            store = Path(folder) / "substrate"
            pointer = memory.save_sharded(store, records_per_shard=3)
            self.assertEqual(pointer["formatVersion"], 3)
            generation = json.loads(
                (store / pointer["generationManifest"]).read_text("utf-8")
            )
            self.assertEqual(generation["schema"], "neural-substrate-2")
            for shard in generation["shards"]:
                if shard["kind"] == "neurons":
                    payload = json.loads(
                        (store / shard["records"]["path"]).read_text("utf-8")
                    )
                    self.assertEqual(payload["vectorIds"], payload["ids"])
                    self.assertEqual(
                        payload["packedVectorState"]["ids"], payload["ids"]
                    )
                    values = load_tensors(store / shard["tensors"]["path"])
                    self.assertEqual(set(values), {"packed_rows", "update_counters_le"})
                    self.assertTrue(all(value.dtype == torch.uint8 for value in values.values()))
                elif shard["kind"] == "assemblies":
                    self.assertIsNone(shard["tensors"])
            restored = NeuralSubstrate.load_sharded(
                store, memory.metadata(include_records=False)
            )
            self.assertEqual(
                restored.neuron_vectors.packed_row(identifier),
                memory.neuron_vectors.packed_row(identifier),
            )
            self.assertEqual(
                restored.neuron_vectors.update_count(identifier),
                memory.neuron_vectors.update_count(identifier),
            )
            self.assertIs(restored.assembly_vectors.backing, restored.neuron_vectors)

            old_pointer = dict(pointer, formatVersion=2)
            old_metadata = memory.metadata(include_records=False)
            old_metadata["persistence"] = old_pointer
            with self.assertRaisesRegex(ValueError, "legacy higher-precision VSA"):
                NeuralSubstrate.load_sharded(store, old_metadata)


if __name__ == "__main__":
    unittest.main()
