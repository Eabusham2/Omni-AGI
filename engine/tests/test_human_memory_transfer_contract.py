"""No-Build tests for lossy but persistent neural experience.

These check structural retention and reuse, not human equivalence or a
question-to-answer memory table. Spoken recall needs a trained-brain run.
"""

import hashlib
import json
import unittest
import tempfile
from pathlib import Path

import torch

from omni_core.persistence import atomic_save_tensors, atomic_write_json, load_tensors
from omni_core.vsa import (
    NeuralSubstrate,
    _unpack_persisted_synapse_weights,
    _unpack_ternary_level,
)


class HumanMemoryTransferContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_decay_reduces_influence_without_deleting_neural_structure(self):
        memory = NeuralSubstrate(dimensions=64, seed=227)
        learned = memory.learn(
            "A copper tram crossed the violet harbor beside three lanterns.",
            source="conversation",
            retain_source_text=False,
        )
        assembly_ids = {str(item["id"]) for item in memory.assemblies}
        neuron_ids = set(memory.neurons)
        synapse_ids = set(memory.synapses)
        vectors = {
            key: value.clone() for key, value in memory.assembly_vectors.items()
        }
        activation_before = sum(
            memory.effective_activation(node) for node in memory.neurons.values()
        )

        memory.decay(0.75)

        self.assertIn(str(learned["assembly_id"]), memory.assembly_vectors)
        self.assertEqual({str(item["id"]) for item in memory.assemblies}, assembly_ids)
        self.assertEqual(set(memory.neurons), neuron_ids)
        self.assertEqual(set(memory.synapses), synapse_ids)
        for key, vector in vectors.items():
            self.assertTrue(torch.equal(memory.assembly_vectors[key], vector))
        self.assertLessEqual(
            sum(memory.effective_activation(node) for node in memory.neurons.values()),
            activation_before,
        )

    def test_related_reuse_warms_activity_without_restoring_raw_text(self):
        memory = NeuralSubstrate(dimensions=64, seed=229)
        text = "A faint paper kite crossed the empty lower window."
        learned = memory.learn(text, retain_source_text=False, importance=0.05)
        assembly_id = str(learned["assembly_id"])
        memory.decay(0.85)
        cooled = memory.effective_activation(memory.neurons[assembly_id])
        structure = (len(memory.neurons), len(memory.assemblies), len(memory.synapses))

        for _ in range(4):
            memory.learn(text, retain_source_text=False, importance=0.90)

        warmed = memory.effective_activation(memory.neurons[assembly_id])
        self.assertGreater(warmed, cooled)
        self.assertEqual(
            (len(memory.neurons), len(memory.assemblies), len(memory.synapses)),
            structure,
        )
        self.assertFalse(any("source_text" in row for row in memory.assemblies))

    def test_reexposure_reactivates_decayed_ternary_membership(self):
        memory = NeuralSubstrate(dimensions=64, seed=233)
        text = "A paper kite drifted above the violet harbor."
        learned = memory.learn(text, retain_source_text=False, importance=0.9)
        assembly_id = str(learned["assembly_id"])
        member_id = str(learned["neuron_ids"][0])
        edge_id = f"{assembly_id}>{member_id}:contains"
        self.assertEqual(memory.synapses[edge_id]["effective_weight"], 1)

        for _ in range(5):
            memory.decay(1.0)
        self.assertEqual(memory.synapses[edge_id]["effective_weight"], 0)
        existing_structure = (len(memory.neurons), len(memory.assemblies), len(memory.synapses))
        prior_uses = int(memory.synapses[edge_id]["uses"])

        for _ in range(6):
            memory.learn(text, retain_source_text=False, importance=0.9)

        self.assertEqual(memory.synapses[edge_id]["effective_weight"], 1)
        self.assertGreater(int(memory.synapses[edge_id]["uses"]), prior_uses)
        self.assertEqual(
            (len(memory.neurons), len(memory.assemblies), len(memory.synapses)),
            existing_structure,
        )
        self.assertFalse(any("source_text" in row for row in memory.assemblies))

    def test_opposing_timing_moves_one_synapse_through_exact_ternary_levels(self):
        memory = NeuralSubstrate(dimensions=64, seed=239)
        learned = memory.learn("Copper lanterns crossed the harbor.")
        assembly_id = str(learned["assembly_id"])
        member_id = str(learned["neuron_ids"][0])
        edge_id = f"{assembly_id}>{member_id}:contains"
        edge = memory.synapses[edge_id]
        self.assertEqual(edge["effective_weight"], 1)
        timestamp = float(edge["last_updated_at"]) + 1.0

        levels = []
        for _ in range(8):
            memory._strengthen_synapse(
                assembly_id, member_id, timestamp,
                kind="contains", amount=-0.4,
            )
            levels.append(memory.synapses[edge_id]["effective_weight"])
        self.assertIn(0, levels)
        self.assertEqual(levels[-1], -1)
        self.assertTrue(all(level in {-1, 0, 1} for level in levels))

        for _ in range(12):
            memory._strengthen_synapse(
                assembly_id, member_id, timestamp,
                kind="contains", amount=0.4,
            )
        self.assertEqual(memory.synapses[edge_id]["effective_weight"], 1)
        self.assertNotIn("latent_weight", memory.synapses[edge_id])

    def test_saved_sparse_synapses_have_one_ternary_weight_and_reload(self):
        memory = NeuralSubstrate(dimensions=64, seed=241)
        memory.learn("A copper lantern crossed the violet harbor.")
        with tempfile.TemporaryDirectory(prefix="omni-ternary-synapse-") as directory:
            root = Path(directory) / "substrate"
            pointer = memory.save_sharded(root, records_per_shard=32)
            self.assertEqual(pointer["formatVersion"], 3)
            generation = NeuralSubstrate._safe_store_path(
                root, pointer["generationManifest"]
            )
            shards = json.loads(generation.read_text("utf-8"))["shards"]
            for shard in shards:
                if shard["kind"] != "synapses":
                    continue
                tensors = load_tensors(
                    NeuralSubstrate._safe_store_path(
                        root, shard["tensors"]["path"]
                    ),
                    device="cpu",
                )
                self.assertNotIn("latent_weight", tensors)
                self.assertNotIn("effective_weight", tensors)
                self.assertEqual(
                    int(tensors["packed_effective_weight"].numel()),
                    (int(shard["count"]) + 3) // 4,
                )
                packed = bytes(tensors["packed_effective_weight"].tolist())
                self.assertTrue(all(
                    _unpack_ternary_level(packed, index) in {-1, 0, 1}
                    for index in range(int(shard["count"]))
                ))
            restored = NeuralSubstrate.load_sharded(
                root, memory.metadata(include_records=False)
            )
            self.assertEqual(len(restored.synapses), len(memory.synapses))
            self.assertTrue(all(
                "latent_weight" not in record
                for record in restored.synapses.values()
            ))

    def test_packed_sparse_shard_rejects_reserved_and_bad_padding_codes(self):
        with self.assertRaisesRegex(ValueError, "reserved code"):
            _unpack_persisted_synapse_weights(
                {"packed_effective_weight": torch.tensor([0xFF], dtype=torch.uint8)},
                1,
                2,
            )
        with self.assertRaisesRegex(ValueError, "noncanonical"):
            _unpack_persisted_synapse_weights(
                {"packed_effective_weight": torch.tensor([0x00], dtype=torch.uint8)},
                1,
                2,
            )
        with self.assertRaisesRegex(ValueError, "duplicate weight"):
            _unpack_persisted_synapse_weights(
                {
                    "packed_effective_weight": torch.tensor([0x55], dtype=torch.uint8),
                    "effective_weight": torch.tensor([0], dtype=torch.int8),
                },
                1,
                2,
            )

    def test_native_v1_with_float_vector_state_is_rejected_without_migration(self):
        memory = NeuralSubstrate(dimensions=32, seed=251)
        memory.learn("Copper lanterns drifted across the violet harbor.")
        with tempfile.TemporaryDirectory(prefix="omni-native-v1-migration-") as directory:
            root = Path(directory) / "substrate"
            pointer = memory.save_sharded(root, records_per_shard=32)
            generation = json.loads(
                (root / pointer["generationManifest"]).read_text("utf-8")
            )
            for shard in generation["shards"]:
                if shard["kind"] != "synapses":
                    continue
                tensors = load_tensors(
                    root / shard["tensors"]["path"], device="cpu"
                )
                packed = bytes(tensors.pop("packed_effective_weight").tolist())
                tensors["effective_weight"] = torch.tensor(
                    [
                        _unpack_ternary_level(packed, index)
                        for index in range(int(shard["count"]))
                    ],
                    dtype=torch.int8,
                )
                tensors["latent_weight"] = (
                    tensors["effective_weight"].to(torch.float64) * 0.75
                )
                staged = root / "blobs" / "native-v1-staged.safetensors"
                atomic_save_tensors(
                    staged,
                    tensors,
                    metadata={
                        "format": "omni-substrate-shards",
                        "formatVersion": "1",
                    },
                )
                checksum = hashlib.sha256(staged.read_bytes()).hexdigest()
                target = root / "blobs" / f"{checksum}.safetensors"
                staged.replace(target)
                shard["tensors"] = {
                    "path": f"blobs/{checksum}.safetensors",
                    "sha256": checksum,
                    "bytes": target.stat().st_size,
                }
            generation["formatVersion"] = 1
            body = {
                key: value for key, value in generation.items()
                if key != "contentSha256"
            }
            generation_id = hashlib.sha256(
                NeuralSubstrate._canonical_json(body)
            ).hexdigest()
            generation["contentSha256"] = generation_id
            relative = f"generations/{generation_id}/manifest.json"
            manifest = root / relative
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest_bytes = NeuralSubstrate._canonical_json(generation)
            manifest.write_bytes(manifest_bytes)
            v1_pointer = {
                **pointer,
                "formatVersion": 1,
                "activeGeneration": generation_id,
                "generationManifest": relative,
                "generationManifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "contentSha256": generation_id,
            }
            atomic_write_json(root / "manifest.json", v1_pointer)
            metadata = memory.metadata(include_records=False)
            metadata["persistence"] = v1_pointer

            with self.assertRaisesRegex(ValueError, "legacy higher-precision VSA"):
                NeuralSubstrate.load_sharded(
                    root, metadata, lazy_synapses=True
                )


if __name__ == "__main__":
    unittest.main()
