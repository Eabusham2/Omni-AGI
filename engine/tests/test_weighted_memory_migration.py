"""Tiny real storage generations; no brain constructor or training corpus."""
import copy
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch

from omni_core.committed_paged_cache import prepare_committed_paged_cache, finish_verified_index_from_loaded_vectors
from omni_core.persistence import atomic_write_bytes, atomic_write_json, load_tensors
from omni_core.vsa import NeuralSubstrate
from omni_core.weighted_vsa_row import CONTACT


class WeightedMemoryMigrationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="weighted-memory-proof-")
        self.addCleanup(folder.cleanup)
        self.engine = Path(folder.name)
        self.store = self.engine / "substrate"
        self.memory = NeuralSubstrate(16, seed=91)
        self.a = self.memory._ensure_neuron("harbor", 1.)
        self.b = self.memory._ensure_neuron("lantern", 1.)
        self.memory._strengthen_synapse(self.a, self.b, 2., kind="association", amount=1.)
        self.pointer = self.memory.save_sharded(self.store, records_per_shard=2)

    def publish(self, body):
        body = {key: value for key, value in body.items() if key != "contentSha256"}
        digest = hashlib.sha256(NeuralSubstrate._canonical_json(body)).hexdigest()
        relative = "generations/" + digest + "/manifest.json"
        payload = NeuralSubstrate._canonical_json({**body, "contentSha256": digest})
        atomic_write_bytes(self.store / relative, payload)
        return {**self.pointer, "activeGeneration": digest, "contentSha256": digest,
                "generationManifest": relative, "generationManifestSha256": hashlib.sha256(payload).hexdigest(),
                "counts": body["counts"], "shardCount": len(body["shards"])}

    def unweighted(self, pointer):
        generation = json.loads((self.store / pointer["generationManifest"]).read_bytes())
        generation.pop("nativeWeightedRepresentation", None)
        for shard in generation["shards"]:
            if shard["kind"] == "assemblies":
                continue
            tensors = load_tensors(self.store / shard["tensors"]["path"])
            if shard["kind"] == "neurons":
                payload = json.loads((self.store / shard["records"]["path"]).read_bytes())
                metadata = payload["packedVectorState"]
                metadata["formatVersion"] = 1
                for name, field in (("connection_state", "connectionStateSha256"),
                                    ("parallel_offsets_le", "parallelOffsetsSha256"),
                                    ("parallel_records", "parallelRecordsSha256")):
                    tensors.pop(name)
                    metadata.pop(field)
                shard["records"] = self.memory._store_json_blob(self.store, payload)
            else:
                tensors.pop("packed_connection_strength")
            shard["tensors"] = self.memory._store_tensor_blob(self.store, tensors)
        return self.publish(generation)

    def metadata(self, pointer):
        return {**self.memory.metadata(include_records=False), "persistence": pointer}

    def test_native_unit_conversion_proves_every_record_and_is_not_repeated(self):
        source = self.unweighted(self.pointer)
        restored = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        self.assertTrue(restored.native_weighted_migration_required)
        self.assertEqual(restored.persistence_manifest, source)
        before = restored.neuron_vectors.row_payload(self.a)
        target = restored.save_sharded(self.store, records_per_shard=2)
        receipt = NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["preservedNeuronCount"], 2)
        self.assertEqual(receipt["preservedSynapseCount"], 1)
        self.assertFalse(restored.native_weighted_migration_required)
        self.assertEqual(restored.neuron_vectors.row_payload(self.a), before)
        reopened = NeuralSubstrate.load_sharded(self.store, self.metadata(target), lazy_synapses=True)
        self.assertFalse(reopened.native_weighted_migration_required)
        # An interrupted parent publication restores the original pointer;
        # its migration need must not have been consumed by staging the target.
        reopened.persistence_manifest = source
        self.assertTrue(reopened.native_weighted_migration_required)

    def test_conversion_rejects_changed_native_row_even_when_sign_is_unchanged(self):
        source = self.unweighted(self.pointer)
        restored = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        restored.neuron_vectors.set_connection_state(self.a, bytes([1]) + bytes(15))
        target = restored.save_sharded(self.store, records_per_shard=2)
        with self.assertRaisesRegex(ValueError, "nonunit"):
            NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)

    def test_conversion_rejects_new_repeated_contact(self):
        source = self.unweighted(self.pointer)
        restored = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        restored.neuron_vectors.set_connection_state(self.a, bytes(16), CONTACT.pack(0, 0x56, 0))
        target = restored.save_sharded(self.store, records_per_shard=2)
        with self.assertRaisesRegex(ValueError, "nonunit"):
            NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)

    def test_conversion_requires_actual_target_companions_not_only_a_marker(self):
        source = self.unweighted(self.pointer)
        generation = json.loads((self.store / source["generationManifest"]).read_bytes())
        generation["nativeWeightedRepresentation"] = 2
        target = self.publish(generation)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)

    def test_existing_strengths_contacts_counters_and_controls_remain_exact(self):
        self.memory.neuron_vectors.set_connection_state(self.a, bytes([0x93]) + bytes(15), CONTACT.pack(2, 0x54, 0x12))
        for moment in range(19):
            self.memory._strengthen_synapse(self.a, self.b, moment + 3., kind="association", amount=10.)
        source = self.memory.save_sharded(self.store, records_per_shard=2)
        restored = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        target = restored.save_sharded(self.store, records_per_shard=2)
        self.assertTrue(NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)["verified"])
        self.assertEqual(restored.neuron_vectors.row_payload(self.a), self.memory.neuron_vectors.row_payload(self.a))
        self.assertEqual([weight for _, _, weight in restored.synapses.iter_weighted_edges()], [16, 4])

    def paged(self, pointer):
        metadata = self.metadata(pointer)
        atomic_write_json(self.engine / "brain.json", {"brain_id": "fixture", "substrate": metadata})
        with prepare_committed_paged_cache(self.engine, self.engine / "caches") as prepared:
            loaded = NeuralSubstrate.load_sharded(self.store, metadata,
                paged_vectors=prepared.vectors, paged_neurons=prepared.neurons,
                defer_paged_assemblies=True, lazy_synapses=True)
            finish_verified_index_from_loaded_vectors(prepared, loaded)
        return loaded

    def test_real_paged_cache_rebuild_preserves_forward_gains(self):
        for moment in range(19):
            self.memory._strengthen_synapse(self.a, self.b, moment + 3., kind="association", amount=10.)
        source = self.memory.save_sharded(self.store, records_per_shard=2)
        loaded = self.paged(source)
        self.assertEqual([weight for _, _, weight in loaded.synapses.iter_weighted_edges()], [16, 4])
        self.assertFalse(loaded.native_weighted_migration_required)

    def test_paged_conversion_rewrites_clean_native_v1_groups_without_learning(self):
        source = self.unweighted(self.pointer)
        loaded = self.paged(source)
        self.assertTrue(loaded.native_weighted_migration_required)
        target = loaded.save_sharded(self.store, records_per_shard=2)
        self.assertTrue(NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)["verified"])

    def test_gain_export_has_same_order_as_signs_and_refuses_midstream_mutation(self):
        source = self.memory.save_sharded(self.store, records_per_shard=2)
        loaded = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        identifier = next(iter(loaded.synapses))
        loaded.synapses[identifier]["connection_strength"] = 7
        signs, count, order, basis = loaded.synapses.stream_dynamic_pack_state()
        state, other_count, other_order, other_basis = loaded.synapses.stream_dynamic_connection_state()
        self.assertEqual((count, order, basis), (other_count, other_order, other_basis))
        self.assertEqual(state.shape, (1, count))
        self.assertEqual(torch.cat(list(state.pages())).tolist(), [6])
        stale, *_ = loaded.synapses.stream_dynamic_connection_state()
        loaded.synapses[identifier]["connection_strength"] = 8
        with self.assertRaisesRegex(ValueError, "changed"):
            list(stale.pages())

    def test_saved_blob_change_after_row_comparison_invalidates_proof(self):
        from omni_core import weighted_substrate_migration as implementation
        source = self.unweighted(self.pointer)
        restored = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        target = restored.save_sharded(self.store, records_per_shard=2)
        actual = implementation._records
        generation = json.loads((self.store / source['generationManifest']).read_bytes())
        old_blob = self.store / generation['shards'][0]['records']['path']

        def changing(*args, **kwargs):
            yield from actual(*args, **kwargs)
            if kwargs.get('require_weighted'):
                # Preserve byte length but corrupt a blob that was already
                # compared. The root pointer itself remains unchanged.
                payload = old_blob.read_bytes()
                atomic_write_bytes(old_blob, b' ' + payload[1:])

        with patch.object(implementation, '_records', side_effect=changing):
            with self.assertRaisesRegex(ValueError, 'changed'):
                NeuralSubstrate.verify_saved_weighted_generation(self.store, target, source_pointer=source)

    def test_old_derived_index_without_strength_contract_rebuilds_from_neural_state(self):
        for moment in range(19):
            self.memory._strengthen_synapse(self.a, self.b, moment + 3., kind='association', amount=10.)
        source = self.memory.save_sharded(self.store, records_per_shard=2)
        index_path = self.store / 'forward-index' / 'generations' / (source['activeGeneration'] + '.json')
        index = json.loads(index_path.read_bytes())
        for descriptor in index['shards']:
            descriptor.pop('connectionStateVersion')
            descriptor.pop('strengthBytes')
        index.pop('contentSha256')
        index['contentSha256'] = hashlib.sha256(NeuralSubstrate._canonical_json(index)).hexdigest()
        atomic_write_json(index_path, index)
        loaded = NeuralSubstrate.load_sharded(self.store, self.metadata(source), lazy_synapses=True)
        self.assertEqual([value for _, _, value in loaded.synapses.iter_weighted_edges()], [16, 4])
        repaired = json.loads(index_path.read_bytes())
        self.assertTrue(all(row['connectionStateVersion'] == 1 for row in repaired['shards']))


if __name__ == "__main__":
    unittest.main()
