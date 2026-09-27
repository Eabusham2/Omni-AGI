"""Source-level contracts for authoritative packed adaptive VSA vectors.

This module does not alter the current VSA integration or prove live recall.
"""

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from omni_core.packed_vsa_vectors import (
    MAX_DECODE_ROWS,
    PackedTernaryVectors,
    PackedTernaryVectorView,
    quantize_vsa_vector,
)


class PackedVSAVectorTests(unittest.TestCase):
    def test_relative_quantization_is_exact_deterministic_and_shape_safe(self):
        values = torch.tensor([4.0, 1.0, -1.0, -4.0, 0.0])
        expected = torch.tensor([1, 0, 0, -1, 0], dtype=torch.int8)
        self.assertTrue(torch.equal(quantize_vsa_vector(values, 5), expected))
        self.assertTrue(torch.equal(quantize_vsa_vector(values.double(), 5), expected))
        self.assertTrue(torch.equal(
            quantize_vsa_vector(torch.tensor([-1, 0, 1], dtype=torch.int8), 3),
            torch.tensor([-1, 0, 1], dtype=torch.int8),
        ))
        self.assertTrue(torch.equal(
            quantize_vsa_vector(torch.zeros(5), 5),
            torch.zeros(5, dtype=torch.int8),
        ))
        with self.assertRaises(ValueError):
            quantize_vsa_vector(torch.tensor([float("nan")] * 5), 5)
        with self.assertRaises(ValueError):
            quantize_vsa_vector(torch.tensor([float("inf")] * 5), 5)
        with self.assertRaises(ValueError):
            quantize_vsa_vector(torch.ones(4), 5)
        with self.assertRaises(ValueError):
            quantize_vsa_vector(torch.ones(5), 5, zero_deadband=1.0)

    def test_mapping_decodes_only_copies_and_recycles_wiped_slots(self):
        vectors = PackedTernaryVectors(5, seed=17)
        vectors["first"] = torch.tensor([1, 0, -1, 0, 1])
        vectors.update({"second": torch.tensor([-1, 1, 0, -1, 0])})
        self.assertEqual(list(vectors), ["first", "second"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors.storage_bytes, 2 * (2 + 8))
        self.assertEqual(vectors["first"].dtype, torch.float32)
        self.assertTrue(torch.equal(
            vectors.levels("first"),
            torch.tensor([1, 0, -1, 0, 1], dtype=torch.int8),
        ))
        self.assertTrue(torch.allclose(vectors.get("first").norm(), torch.tensor(1.0)))
        self.assertEqual(len(list(vectors.items())), 2)
        self.assertEqual(len(list(vectors.values())), 2)
        read = vectors["first"]
        read.fill_(0)
        self.assertGreater(vectors["first"][0].item(), 0.0)
        self.assertEqual(vectors.levels("first").dtype, torch.int8)
        self.assertEqual(len(vectors.packed_row("first")), 2)
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in vars(vectors).values()))

        del vectors["first"]
        vectors["third"] = torch.tensor([0, 0, 0, 0, -1])
        self.assertEqual(list(vectors), ["second", "third"])
        self.assertEqual(vectors.storage_bytes, 2 * (2 + 8))
        self.assertEqual(vectors.update_count("third"), 0)
        self.assertTrue(torch.equal(
            vectors.levels("third"), torch.tensor([0, 0, 0, 0, -1], dtype=torch.int8)
        ))
        vectors.clear()
        self.assertEqual(vectors.storage_bytes, 0)
        self.assertEqual(len(vectors), 0)

    def test_small_rate_discrete_adaptation_resumes_exactly_after_safetensors(self):
        original = PackedTernaryVectors(16, seed=43)
        original["assembly"] = torch.ones(16)
        target = -torch.ones(16)
        first = original.adapt("assembly", target, 0.05)
        self.assertEqual(first.dtype, torch.float32)
        self.assertEqual(original.update_count("assembly"), 1)
        metadata, tensors = original.export_state(prefix="vsa.")
        self.assertEqual(set(tensors), {"vsa.packed_rows", "vsa.update_counters_le"})
        self.assertEqual(tensors["vsa.packed_rows"].dtype, torch.uint8)
        self.assertEqual(tensors["vsa.packed_rows"].shape, (1, 4))
        self.assertEqual(tensors["vsa.update_counters_le"].shape, (1, 8))
        self.assertFalse(any(tensor.is_floating_point() for tensor in tensors.values()))

        with tempfile.TemporaryDirectory(prefix="omni-packed-vsa-") as folder:
            path = Path(folder) / "vectors.safetensors"
            save_file(tensors, str(path))
            restored = PackedTernaryVectors.from_state(
                metadata, load_file(str(path)), prefix="vsa."
            )
        self.assertEqual(restored.packed_row("assembly"), original.packed_row("assembly"))
        self.assertEqual(restored.update_count("assembly"), 1)
        for _ in range(128):
            original.adapt("assembly", target, 0.05)
            restored.adapt("assembly", target, 0.05)
        self.assertEqual(restored.packed_row("assembly"), original.packed_row("assembly"))
        self.assertEqual(restored.update_count("assembly"), original.update_count("assembly"))
        self.assertFalse(torch.equal(original.levels("assembly"), torch.ones(16, dtype=torch.int8)))

        # The export is an independent snapshot, not an alias of the live row.
        snapshot = original.packed_row("assembly")
        tensors["vsa.packed_rows"].fill_(0)
        self.assertEqual(original.packed_row("assembly"), snapshot)

    def test_bounded_decode_and_empty_roundtrip(self):
        vectors = PackedTernaryVectors(5)
        self.assertEqual(tuple(vectors.decode_rows([]).shape), (0, 5))
        metadata, tensors = vectors.export_state()
        self.assertEqual(tuple(tensors["packed_rows"].shape), (0, 2))
        self.assertEqual(len(PackedTernaryVectors.from_state(metadata, tensors)), 0)
        for index in range(MAX_DECODE_ROWS + 1):
            vectors[str(index)] = torch.ones(5)
        self.assertEqual(tuple(vectors.decode_rows(list(vectors)[:MAX_DECODE_ROWS]).shape), (MAX_DECODE_ROWS, 5))
        with self.assertRaisesRegex(ValueError, "bounded window"):
            vectors.decode_rows(list(vectors))

    def test_subset_export_and_shared_view_have_one_authoritative_row(self):
        neurons = PackedTernaryVectors(5, seed=9)
        neurons["semantic"] = torch.tensor([1, -1, 0, 1, 0])
        assemblies = PackedTernaryVectorView(neurons)
        assemblies["idea"] = torch.tensor([-1, 0, 1, 0, -1])
        self.assertIn("idea", neurons)
        self.assertNotIn("semantic", assemblies)
        self.assertIs(assemblies.backing, neurons)
        self.assertEqual(neurons.storage_bytes, 2 * (2 + 8))
        assemblies.adapt("idea", torch.ones(5), 1.0)
        self.assertTrue(torch.equal(neurons.levels("idea"), torch.ones(5, dtype=torch.int8)))
        self.assertTrue(torch.equal(neurons["idea"], assemblies["idea"]))

        subset_meta, subset_tensors = neurons.export_state(keys=["idea"])
        subset = PackedTernaryVectors.from_state(subset_meta, subset_tensors)
        self.assertEqual(list(subset), ["idea"])
        merged = PackedTernaryVectors(5, seed=9)
        merged.update_packed(subset)
        self.assertEqual(merged.packed_row("idea"), neurons.packed_row("idea"))
        self.assertEqual(merged.update_count("idea"), neurons.update_count("idea"))
        with self.assertRaisesRegex(ValueError, "duplicate IDs"):
            merged.update_packed(subset)
        with self.assertRaisesRegex(ValueError, "parameters"):
            PackedTernaryVectors(5, seed=10).update_packed(subset)

    def test_import_rejects_hash_shape_reserved_padding_counter_and_duplicate_ids(self):
        vectors = PackedTernaryVectors(5)
        vectors["a"] = torch.tensor([1, 0, -1, 0, 1])
        vectors["b"] = torch.zeros(5)
        metadata, tensors = vectors.export_state()

        wrong_shape = dict(tensors, packed_rows=tensors["packed_rows"][:, :1])
        with self.assertRaisesRegex(ValueError, "tensor is invalid"):
            PackedTernaryVectors.from_state(metadata, wrong_shape)
        wrong_type = dict(tensors, packed_rows=tensors["packed_rows"].to(torch.int8))
        with self.assertRaisesRegex(ValueError, "tensor is invalid"):
            PackedTernaryVectors.from_state(metadata, wrong_type)
        altered = tensors["packed_rows"].clone()
        altered[0, 0] = int(altered[0, 0]) ^ 1
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            PackedTernaryVectors.from_state(metadata, dict(tensors, packed_rows=altered))

        reserved = tensors["packed_rows"].clone()
        reserved[0, 0] = (int(reserved[0, 0]) & 0xFC) | 0x03
        reserved_meta = dict(
            metadata, packedSha256=hashlib.sha256(reserved.numpy().tobytes()).hexdigest()
        )
        with self.assertRaisesRegex(ValueError, "reserved"):
            PackedTernaryVectors.from_state(
                reserved_meta, dict(tensors, packed_rows=reserved)
            )

        bad_padding = tensors["packed_rows"].clone()
        bad_padding[0, 1] = int(bad_padding[0, 1]) & 0xF3
        padding_meta = dict(
            metadata,
            packedSha256=hashlib.sha256(bad_padding.numpy().tobytes()).hexdigest(),
        )
        with self.assertRaisesRegex(ValueError, "padding"):
            PackedTernaryVectors.from_state(
                padding_meta, dict(tensors, packed_rows=bad_padding)
            )

        bad_counters = tensors["update_counters_le"].clone()
        bad_counters[0].fill_(255)
        counter_meta = dict(
            metadata,
            countersSha256=hashlib.sha256(bad_counters.numpy().tobytes()).hexdigest(),
        )
        with self.assertRaisesRegex(ValueError, "counter is invalid"):
            PackedTernaryVectors.from_state(
                counter_meta, dict(tensors, update_counters_le=bad_counters)
            )

        duplicated = dict(metadata, ids=["a", "a"])
        with self.assertRaisesRegex(ValueError, "IDs or row count"):
            PackedTernaryVectors.from_state(duplicated, tensors)
        with self.assertRaisesRegex(ValueError, "format is incompatible"):
            PackedTernaryVectors.from_state(dict(metadata, formatVersion=1.0), tensors)


if __name__ == "__main__":
    unittest.main()
