import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.packed_vsa_vectors import PackedTernaryVectors
from omni_core.memory_inference_export import vector_tensor_spec
from omni_core.ternary_packing import export_ternary_shards, verify_ternary_shards
from omni_core.weighted_vsa_row import CONTACT


class MemoryInferenceExportTests(unittest.TestCase):
    def vectors(self):
        values = PackedTernaryVectors(5)
        values["second"] = torch.tensor([1, -1, 0, 1, 1])
        values["first"] = torch.tensor([-1, 0, 1, 1, -1])
        values.set_connection_state("second", bytes([15, 3, 0, 6, 1]),
            CONTACT.pack(4, 0x56, 2) + CONTACT.pack(0, 0x54, 7))
        return values

    def test_sign_strength_and_unordered_repeat_storage_export_exactly(self):
        vectors = self.vectors()
        spec, binding = vector_tensor_spec(vectors)
        with tempfile.TemporaryDirectory() as directory:
            export_ternary_shards(Path(directory), [spec], metadata={"vectors": binding})
            result = verify_ternary_shards(Path(directory))
            reconstructed = result.tensors[spec.name].to(torch.int64) * (
                (result.connection_states[spec.name].to(torch.int64) & 15) + 1)
            for row, column, _ordinal, sign, strength in result.parallel_connections[spec.name]:
                reconstructed[row, column] += ((sign & 3) - 1) * ((strength & 15) + 1)
            self.assertTrue(torch.equal(reconstructed, torch.stack([vectors.weighted_levels(key) for key in vectors])))
            self.assertEqual(binding["repeatedContacts"], 2)

    def test_change_after_capture_rejects_each_companion(self):
        vectors = self.vectors()
        spec, _ = vector_tensor_spec(vectors)
        vectors.set_connection_state("first", bytes([1, 2, 3, 4, 5]))
        for generate in (spec.values.pages, spec.connection_state.pages, spec.parallel_connections.records):
            with self.assertRaisesRegex(ValueError, "generation changed"):
                list(generate())

    def test_empty_vector_table_roundtrips(self):
        spec, _ = vector_tensor_spec(PackedTernaryVectors(5))
        with tempfile.TemporaryDirectory() as directory:
            export_ternary_shards(Path(directory), [spec])
            result = verify_ternary_shards(Path(directory))
            self.assertEqual(tuple(result.tensors[spec.name].shape), (0, 5))
            self.assertEqual(tuple(result.connection_states[spec.name].shape), (0, 5))


if __name__ == "__main__":
    unittest.main()
