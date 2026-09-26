import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.vsa import LazyPersistedSynapses, NeuralSubstrate


class LazySubstrateLoadingTests(unittest.TestCase):
    def saved_memory(self, root: Path) -> tuple[NeuralSubstrate, dict]:
        memory = NeuralSubstrate(32, seed=71)
        for text in (
            "alpha beta gamma delta epsilon zeta eta theta",
            "gamma delta durable recall follows exact ternary pathways",
            "theta alpha recurrent memory preserves inhibitory competition",
            "hot assemblies retain cold persisted synapses without truncation",
        ):
            memory.learn(text)
        memory.save_sharded(root, records_per_shard=3)
        return memory, memory.metadata(include_records=False)

    def test_hot_first_load_preserves_exact_recall_without_resident_cold_dicts(self):
        with tempfile.TemporaryDirectory(prefix="omni-lazy-substrate-") as folder:
            store = Path(folder) / "substrate"
            _memory, metadata = self.saved_memory(store)
            eager = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=False
            )
            lazy = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=True
            )

            self.assertIsInstance(lazy.synapses, LazyPersistedSynapses)
            status = lazy.synapses.paging_status()
            self.assertEqual(status["totalSynapses"], len(eager.synapses))
            self.assertEqual(status["residentSynapseRecords"], 0)
            self.assertEqual(status["persistedColdSynapses"], len(eager.synapses))
            self.assertTrue(status["allRecordsAddressable"])
            self.assertEqual(status["forwardWeightsPerByte"], 4)
            self.assertFalse(status["denseForwardWeightsMaterialized"])
            self.assertLessEqual(
                status["packedForwardBytes"],
                (status["forwardEdges"] + 3) // 4 + status["shards"],
            )

            cue = eager.vector_for_text("alpha recurrent memory")
            expected_signal, expected_recall = eager.recall_vector(
                cue, workspace_slots=16, record_activity=False
            )
            actual_signal, actual_recall = lazy.recall_vector(
                cue, workspace_slots=16, record_activity=False
            )
            self.assertTrue(torch.equal(actual_signal, expected_signal))
            self.assertEqual(actual_recall, expected_recall)
            self.assertEqual(
                lazy._last_recall_audit, eager._last_recall_audit
            )
            self.assertEqual(lazy.synapses.resident_record_count, 0)
            eager_edges = sorted(
                (
                    str(record["source_id"]),
                    str(record["target_id"]),
                    eager.exact_effective_weight(record["effective_weight"]),
                )
                for record in eager.synapses.values()
                if eager.exact_effective_weight(record["effective_weight"])
            )
            self.assertEqual(
                sorted(lazy.synapses.iter_effective_edges()), eager_edges
            )

    def test_generation_bound_forward_index_skips_every_cold_shard_on_reload(self):
        with tempfile.TemporaryDirectory(prefix="omni-forward-index-") as folder:
            store = Path(folder) / "substrate"
            _memory, metadata = self.saved_memory(store)
            pointer = metadata["persistence"]
            index_path = (
                store
                / "forward-index"
                / "generations"
                / (pointer["activeGeneration"] + ".json")
            )
            self.assertTrue(index_path.is_file())
            manifest = json.loads(index_path.read_text("utf-8"))
            self.assertEqual(
                manifest["sourceGeneration"], pointer["activeGeneration"]
            )
            self.assertEqual(
                manifest["sourceGenerationManifestSha256"],
                pointer["generationManifestSha256"],
            )
            with mock.patch.object(
                LazyPersistedSynapses,
                "_read_shard",
                side_effect=AssertionError("indexed load must not open a cold shard"),
            ):
                restored = NeuralSubstrate.load_sharded(
                    store, metadata, lazy_synapses=True
                )
                packed, count, order_sha, order_basis = (
                    restored.synapses.dynamic_pack_state()
                )
                use_count = restored.synaptic_use_count()
            self.assertEqual(len(restored.synapses), len(_memory.synapses))
            self.assertEqual(restored.synapses.resident_record_count, 0)
            self.assertEqual(count, len(_memory.synapses))
            self.assertEqual(int(packed.numel()), len(_memory.synapses))
            self.assertEqual(
                int(packed.ne(0).sum().item()),
                sum(
                    int(record["effective_weight"] != 0)
                    for record in _memory.synapses.values()
                ),
            )
            self.assertRegex(order_sha, r"^[a-f0-9]{64}$")
            self.assertEqual(order_basis, "substrate-shard-record-sha256-v1")
            self.assertEqual(
                use_count,
                sum(int(record["uses"]) for record in _memory.synapses.values()),
            )

    def test_valid_v1_index_is_atomically_enriched_once(self):
        with tempfile.TemporaryDirectory(prefix="omni-forward-upgrade-") as folder:
            store = Path(folder) / "substrate"
            memory, metadata = self.saved_memory(store)
            pointer = metadata["persistence"]
            index_path = (
                store
                / "forward-index"
                / "generations"
                / (pointer["activeGeneration"] + ".json")
            )
            legacy = json.loads(index_path.read_text("utf-8"))
            legacy["formatVersion"] = 1
            legacy.pop("synapticUses")
            for entry in legacy["shards"]:
                entry.pop("synapticUses")
            body = {
                key: value
                for key, value in legacy.items()
                if key != "contentSha256"
            }
            legacy["contentSha256"] = hashlib.sha256(
                NeuralSubstrate._canonical_json(body)
            ).hexdigest()
            index_path.write_bytes(NeuralSubstrate._canonical_json(legacy))

            original = LazyPersistedSynapses._read_shard
            opened = 0

            def counted(instance, key, *, validate=False):
                nonlocal opened
                opened += 1
                return original(instance, key, validate=validate)

            with mock.patch.object(LazyPersistedSynapses, "_read_shard", counted):
                restored = NeuralSubstrate.load_sharded(
                    store, metadata, lazy_synapses=True
                )
            upgraded = json.loads(index_path.read_text("utf-8"))
            self.assertGreater(opened, 0)
            self.assertEqual(upgraded["formatVersion"], 2)
            self.assertEqual(
                restored.synaptic_use_count(),
                sum(int(record["uses"]) for record in memory.synapses.values()),
            )

    def test_stale_or_tampered_forward_index_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="omni-forward-tamper-") as folder:
            store = Path(folder) / "substrate"
            _memory, metadata = self.saved_memory(store)
            pointer = metadata["persistence"]
            index_path = (
                store
                / "forward-index"
                / "generations"
                / (pointer["activeGeneration"] + ".json")
            )
            manifest = json.loads(index_path.read_text("utf-8"))
            manifest["sourceGeneration"] = "0" * 64
            body = {
                key: value
                for key, value in manifest.items()
                if key != "contentSha256"
            }
            manifest["contentSha256"] = hashlib.sha256(
                NeuralSubstrate._canonical_json(body)
            ).hexdigest()
            index_path.write_bytes(NeuralSubstrate._canonical_json(manifest))
            with self.assertRaisesRegex(ValueError, "stale or corrupt"):
                NeuralSubstrate.load_sharded(
                    store, metadata, lazy_synapses=True
                )

    def test_touched_shard_checkpoint_reuses_cold_blobs_and_roundtrips_mutation(self):
        with tempfile.TemporaryDirectory(prefix="omni-lazy-checkpoint-") as folder:
            store = Path(folder) / "substrate"
            _memory, metadata = self.saved_memory(store)
            lazy = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=True
            )
            before_pointer = dict(lazy.persistence_manifest or {})
            before_generation = json.loads(
                (store / before_pointer["generationManifest"]).read_text("utf-8")
            )
            before_shards = {
                (item["kind"], item["bucket"], item["part"]): item
                for item in before_generation["shards"]
                if item["kind"] == "synapses"
            }

            target_id = next(iter(lazy.synapses))
            target = lazy.synapses[target_id]
            uses_before = lazy.synaptic_use_count()
            target["uses"] = int(target["uses"]) + 9
            target["effective_weight"] = -1
            self.assertEqual(lazy.synaptic_use_count(), uses_before + 9)
            self.assertEqual(lazy.synapses.resident_record_count, 1)
            lazy.save_sharded(store, records_per_shard=3)
            self.assertEqual(lazy.synapses.resident_record_count, 0)

            after_generation = json.loads(
                (
                    store
                    / str(lazy.persistence_manifest["generationManifest"])
                ).read_text("utf-8")
            )
            after_shards = {
                (item["kind"], item["bucket"], item["part"]): item
                for item in after_generation["shards"]
                if item["kind"] == "synapses"
            }
            reused = sum(
                1
                for key, value in before_shards.items()
                if key in after_shards and after_shards[key] == value
            )
            self.assertGreaterEqual(reused, len(before_shards) - 1)

            restored = NeuralSubstrate.load_sharded(
                store,
                lazy.metadata(include_records=False),
                lazy_synapses=False,
            )
            self.assertEqual(restored.synapses[target_id]["uses"], target["uses"])
            self.assertNotIn("latent_weight", restored.synapses[target_id])
            self.assertEqual(restored.synapses[target_id]["effective_weight"], -1)
            self.assertEqual(len(restored.synapses), len(lazy.synapses))
            self.assertEqual(restored.synaptic_use_count(), uses_before + 9)
            with mock.patch.object(
                LazyPersistedSynapses,
                "_read_shard",
                side_effect=AssertionError(
                    "mutated generation index must not open a cold shard"
                ),
            ):
                indexed = NeuralSubstrate.load_sharded(
                    store,
                    lazy.metadata(include_records=False),
                    lazy_synapses=True,
                )
            self.assertIn(
                (str(target["source_id"]), str(target["target_id"]), -1),
                set(indexed.synapses.iter_effective_edges()),
            )

    def test_clean_export_checkpoint_reuses_lazy_generation_without_paging_cold_records(self):
        """The checkpoint preceding .omni export stays O(shards + index), not O(records)."""

        with tempfile.TemporaryDirectory(prefix="omni-lazy-export-") as folder:
            store = Path(folder) / "substrate"
            memory, metadata = self.saved_memory(store)
            lazy = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=True
            )
            before = dict(lazy.persistence_manifest or {})
            with mock.patch.object(
                LazyPersistedSynapses,
                "_read_shard",
                side_effect=AssertionError(
                    "clean export checkpoint must not page a cold record shard"
                ),
            ):
                after = lazy.save_sharded(store, records_per_shard=3)

            self.assertEqual(after, before)
            self.assertEqual(lazy.synapses.resident_record_count, 0)
            self.assertEqual(len(lazy.synapses), len(memory.synapses))

    def test_forward_index_publication_groups_hot_locations_in_one_pass(self):
        with tempfile.TemporaryDirectory(prefix="omni-forward-linear-") as folder:
            store = Path(folder) / "substrate"
            _memory, metadata = self.saved_memory(store)
            lazy = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=True
            )
            pointer = metadata["persistence"]
            generation = json.loads(
                (store / pointer["generationManifest"]).read_text("utf-8")
            )
            descriptors = [
                value
                for value in generation["shards"]
                if value["kind"] == "synapses"
            ]

            class CountingLocations(dict):
                def __init__(self, values):
                    super().__init__(values)
                    self.items_calls = 0

                def items(self):
                    self.items_calls += 1
                    return super().items()

            locations = CountingLocations(lazy.synapses._hot_locations)
            lazy.synapses._hot_locations = locations
            entries = lazy.synapses.forward_index_entries(
                descriptors,
                {},
                {
                    str(record.get("id", ""))
                    for record in lazy.assemblies
                    if record.get("id")
                },
            )

            self.assertEqual(len(entries), len(descriptors))
            self.assertEqual(locations.items_calls, 1)

    def test_lazy_load_still_rejects_a_corrupt_cold_shard(self):
        with tempfile.TemporaryDirectory(prefix="omni-lazy-integrity-") as folder:
            store = Path(folder) / "substrate"
            _memory, metadata = self.saved_memory(store)
            pointer = metadata["persistence"]
            generation = json.loads(
                (store / pointer["generationManifest"]).read_text("utf-8")
            )
            shard = next(
                item for item in generation["shards"] if item["kind"] == "synapses"
            )
            path = store / shard["records"]["path"]
            record_id = json.loads(path.read_text("utf-8"))["ids"][0]
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            lazy = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=True
            )
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                lazy.synapses[record_id]
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                lazy.synapses.scrub_persisted_shards()

    def test_explicit_scrub_proves_complete_index_parity(self):
        with tempfile.TemporaryDirectory(prefix="omni-lazy-scrub-") as folder:
            store = Path(folder) / "substrate"
            memory, metadata = self.saved_memory(store)
            lazy = NeuralSubstrate.load_sharded(
                store, metadata, lazy_synapses=True
            )
            progress = []
            result = lazy.synapses.scrub_persisted_shards(
                lambda done, total, key: progress.append((done, total, key))
            )
            self.assertTrue(result["verified"])
            self.assertEqual(result["synapses"], len(memory.synapses))
            self.assertEqual(result["shards"], progress[-1][1])
            self.assertEqual(progress[-1][0], progress[-1][1])
            self.assertEqual(
                lazy.synapses.paging_status()["pendingScrubShards"], 0
            )


if __name__ == "__main__":
    unittest.main()
