"""Filesystem/storage-only contracts for committed v3 paged-cache rebuild."""

import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omni_core.committed_paged_cache import (
    CommittedCacheResourcePause,
    finish_verified_index_from_loaded_vectors,
    prepare_committed_paged_cache,
    prune_derived_caches,
    rebuild_committed_paged_cache,
)
from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.vsa import NeuralSubstrate


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


class CommittedPagedCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-committed-cache-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.engine = self.root / "engine"
        self.engine.mkdir()
        self.store = self.engine / "substrate"
        self.caches = self.root / "caches"
        self.memory = NeuralSubstrate(16, seed=29)
        self.memory.learn("Amber lanterns light the cedar harbor.")
        self.memory.learn("Violet trams reach the copper bridge.")
        self.pointer = self.commit_brain()

    def commit_brain(self):
        pointer = self.memory.save_sharded(self.store, records_per_shard=2)
        (self.engine / "brain.json").write_bytes(canonical({
            "brain_id": "committed-cache-test",
            "substrate": self.memory.metadata(include_records=False),
            "ingestion_checkpoints": [{"cursor": "only-brain-owns-this"}],
        }))
        return pointer

    def published_paths(self):
        if not self.caches.exists():
            return []
        return sorted(self.caches.glob("paged-*/working.sqlite3"))

    def test_streams_shards_into_bound_cache_without_loading_whole_substrate(self):
        assembly_ids = [record["id"] for record in self.memory.assemblies]
        with mock.patch.object(
            NeuralSubstrate, "load_sharded",
            side_effect=AssertionError("whole-substrate loader must not run"),
        ):
            result = rebuild_committed_paged_cache(self.engine, self.caches)
        self.assertEqual(result.generation_sha256, self.pointer["activeGeneration"])
        self.assertEqual(result.neurons, len(self.memory.neurons))
        self.assertEqual(result.assemblies, len(self.memory.assemblies))
        self.assertEqual(result.synapses, len(self.memory.synapses))
        self.assertEqual([path.resolve() for path in self.published_paths()], [result.path])
        index = PagedAssemblyIndex(result.path, dimensions=16, seed=29)
        self.assertEqual(
            index.discard_or_reconcile_uncommitted(result.generation_sha256)["state"],
            "clean",
        )
        self.assertEqual(index.status()["count"], len(assembly_ids))
        self.assertEqual(index.status()["packedVectorRows"], len(assembly_ids))
        self.assertEqual(
            [record["id"] for page in index.iter_pages(page_size=1)
             for record in page.records],
            assembly_ids,
        )
        self.assertIsNone(index.load_committed_checkpoint(
            "source-cursor", result.generation_sha256
        ))
        for identifier in self.memory.neuron_vectors:
            self.assertEqual(
                index._vectors.packed_row(identifier),
                self.memory.neuron_vectors.packed_row(identifier),
            )
            self.assertEqual(
                index._vectors.update_count(identifier),
                self.memory.neuron_vectors.update_count(identifier),
            )

    def test_composable_cold_load_imports_vectors_only_once_into_same_db(self):
        metadata = json.loads((self.engine / "brain.json").read_text("utf-8"))["substrate"]
        with prepare_committed_paged_cache(self.engine, self.caches) as prepared:
            self.assertEqual(len(prepared.vectors), 0)
            with mock.patch.object(
                prepared.vectors, "import_state",
                wraps=prepared.vectors.import_state,
            ) as import_shard:
                loaded = NeuralSubstrate.load_sharded(
                    self.store, metadata, paged_vectors=prepared.vectors,
                    lazy_synapses=True,
                )
                calls_after_load = import_shard.call_count
                self.assertGreater(calls_after_load, 0)
                result = finish_verified_index_from_loaded_vectors(prepared, loaded)
                self.assertEqual(import_shard.call_count, calls_after_load)
            self.assertEqual(result.generation_sha256, self.pointer["activeGeneration"])
            self.assertEqual(prepared.vectors.path, result.path)
            first_neuron = next(iter(self.memory.neuron_vectors))
            self.assertEqual(
                loaded.neuron_vectors.packed_row(first_neuron),
                self.memory.neuron_vectors.packed_row(first_neuron),
            )
        index = PagedAssemblyIndex(result.path, dimensions=16, seed=29)
        self.assertEqual(index.status()["count"], len(self.memory.assemblies))
        self.assertEqual(
            index.discard_or_reconcile_uncommitted(result.generation_sha256)["state"],
            "clean",
        )

    def test_deferred_cold_load_never_builds_resident_neuron_or_assembly_maps(self):
        metadata = json.loads((self.engine / "brain.json").read_text("utf-8"))["substrate"]
        with prepare_committed_paged_cache(self.engine, self.caches) as prepared:
            loaded = NeuralSubstrate.load_sharded(
                self.store, metadata,
                paged_vectors=prepared.vectors,
                paged_neurons=prepared.neurons,
                defer_paged_assemblies=True,
                lazy_synapses=True,
            )
            self.assertIs(loaded.neurons, prepared.neurons)
            self.assertEqual(len(loaded.assemblies), 0)
            self.assertEqual(len(loaded.synapses), 0)
            self.assertTrue(loaded._paged_load_incomplete)
            self.assertEqual(len(loaded._persistence_record_groups), 0)
            result = finish_verified_index_from_loaded_vectors(prepared, loaded)
            self.assertFalse(loaded._paged_load_incomplete)
            self.assertIs(loaded.assemblies.index, prepared.index)
            self.assertIs(loaded.assembly_vectors.backing, prepared.vectors)
            self.assertEqual(len(loaded.neurons), len(self.memory.neurons))
            self.assertEqual(len(loaded.assemblies), len(self.memory.assemblies))
            self.assertEqual(len(loaded.synapses), len(self.memory.synapses))
            self.assertFalse(loaded.neurons.status()["dirtySinceCommit"])
            self.assertEqual(loaded.neurons.path, result.path)

    def test_deferred_load_fails_closed_without_verified_forward_index(self):
        metadata = json.loads((self.engine / "brain.json").read_text("utf-8"))["substrate"]
        forward = self.store / "forward-index" / "generations" / (
            self.pointer["activeGeneration"] + ".json"
        )
        self.assertTrue(forward.is_file())
        forward.unlink()
        with self.assertRaisesRegex(ValueError, "requires a verified forward index"):
            with prepare_committed_paged_cache(self.engine, self.caches) as prepared:
                staged = prepared.staged_directory
                loaded = NeuralSubstrate.load_sharded(
                    self.store, metadata,
                    paged_vectors=prepared.vectors,
                    paged_neurons=prepared.neurons,
                    defer_paged_assemblies=True,
                    lazy_synapses=True,
                )
                finish_verified_index_from_loaded_vectors(prepared, loaded)
        self.assertFalse(staged.exists())
        self.assertEqual(self.published_paths(), [])

    def test_composable_failure_discards_only_private_staging(self):
        metadata = json.loads((self.engine / "brain.json").read_text("utf-8"))["substrate"]
        old = PagedAssemblyIndex(self.caches / "old.sqlite3", dimensions=16, seed=29)
        old.upsert({"id": "dirty", "fingerprint": "dirty-fingerprint"})
        old_status = old.status()
        with self.assertRaisesRegex(ValueError, "verified vector import"):
            with prepare_committed_paged_cache(self.engine, self.caches) as prepared:
                staged = prepared.staged_directory
                wrong = NeuralSubstrate(16, seed=29)
                finish_verified_index_from_loaded_vectors(prepared, wrong)
        self.assertFalse(staged.exists())
        self.assertEqual(old.status(), old_status)
        with self.assertRaisesRegex(ValueError, "checksum"):
            with prepare_committed_paged_cache(self.engine, self.caches) as prepared:
                staged = prepared.staged_directory
                generation = json.loads(
                    (self.store / self.pointer["generationManifest"]).read_text("utf-8")
                )
                tensor = next(
                    shard for shard in generation["shards"] if shard["kind"] == "neurons"
                )["tensors"]
                (self.store / tensor["path"]).write_bytes(b"corrupt")
                NeuralSubstrate.load_sharded(
                    self.store, metadata, paged_vectors=prepared.vectors
                )
        self.assertFalse(staged.exists())
        self.assertEqual(self.published_paths(), [])
        self.assertEqual(old.status(), old_status)

    def test_ahead_substrate_manifest_and_dirty_old_cache_are_never_adopted(self):
        old = PagedAssemblyIndex(
            self.caches / "old.sqlite3", dimensions=16, seed=29
        )
        old.upsert({"id": "dirty", "fingerprint": "dirty-fingerprint"})
        old.save_checkpoint("source-cursor", {"offset": 999})
        old_status = old.status()
        old_cursor = old.load_checkpoint("source-cursor")
        old_pointer = (self.engine / "brain.json").read_bytes()
        self.memory.learn("A third uncommitted sentence changes the substrate.")
        ahead = self.memory.save_sharded(self.store, records_per_shard=2)
        self.assertNotEqual(ahead["activeGeneration"], self.pointer["activeGeneration"])
        self.assertEqual((self.engine / "brain.json").read_bytes(), old_pointer)

        result = rebuild_committed_paged_cache(self.engine, self.caches)
        self.assertEqual(result.generation_sha256, self.pointer["activeGeneration"])
        self.assertEqual(result.assemblies, 2)
        self.assertEqual(old.status(), old_status)
        self.assertEqual(old.load_checkpoint("source-cursor"), old_cursor)
        self.assertEqual((self.engine / "brain.json").read_bytes(), old_pointer)

    def test_bad_blob_hash_and_reserve_refusal_leave_existing_state_untouched(self):
        old = PagedAssemblyIndex(self.caches / "old.sqlite3", dimensions=16, seed=29)
        old.upsert({"id": "dirty", "fingerprint": "dirty-fingerprint"})
        before_old = old.status()
        before_brain = (self.engine / "brain.json").read_bytes()
        with self.assertRaises(CommittedCacheResourcePause):
            rebuild_committed_paged_cache(
                self.engine, self.caches,
                memory_reserve=lambda _size, _label: False,
            )
        self.assertEqual(self.published_paths(), [])
        self.assertEqual(old.status(), before_old)
        self.assertEqual((self.engine / "brain.json").read_bytes(), before_brain)

        generation = json.loads(
            (self.store / self.pointer["generationManifest"]).read_text("utf-8")
        )
        record_path = self.store / generation["shards"][0]["records"]["path"]
        record_path.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum|size mismatch"):
            rebuild_committed_paged_cache(self.engine, self.caches)
        self.assertEqual(self.published_paths(), [])
        self.assertEqual(old.status(), before_old)
        self.assertEqual((self.engine / "brain.json").read_bytes(), before_brain)

    def test_self_consistent_blob_hash_cannot_hide_mismatched_ids(self):
        generation_path = self.store / self.pointer["generationManifest"]
        generation = json.loads(generation_path.read_text("utf-8"))
        descriptor = next(
            shard for shard in generation["shards"] if shard["kind"] == "neurons"
        )
        old_spec = descriptor["records"]
        payload = json.loads((self.store / old_spec["path"]).read_text("utf-8"))
        payload["ids"][0] += "-wrong"
        blob = canonical(payload)
        blob_sha = hashlib.sha256(blob).hexdigest()
        new_spec = {
            "path": "blobs/%s.json" % blob_sha,
            "sha256": blob_sha,
            "bytes": len(blob),
        }
        (self.store / new_spec["path"]).write_bytes(blob)
        descriptor["records"] = new_spec
        body = {key: value for key, value in generation.items() if key != "contentSha256"}
        generation_id = hashlib.sha256(canonical(body)).hexdigest()
        generation["contentSha256"] = generation_id
        generation_blob = canonical(generation)
        relative = "generations/%s/manifest.json" % generation_id
        new_generation_path = self.store / relative
        new_generation_path.parent.mkdir(parents=True)
        new_generation_path.write_bytes(generation_blob)
        pointer = dict(self.pointer)
        pointer.update({
            "activeGeneration": generation_id,
            "contentSha256": generation_id,
            "generationManifest": relative,
            "generationManifestSha256": hashlib.sha256(generation_blob).hexdigest(),
        })
        brain = json.loads((self.engine / "brain.json").read_text("utf-8"))
        brain["substrate"]["persistence"] = pointer
        (self.engine / "brain.json").write_bytes(canonical(brain))
        with self.assertRaisesRegex(ValueError, "identifiers"):
            rebuild_committed_paged_cache(self.engine, self.caches)
        self.assertEqual(self.published_paths(), [])

    def test_concurrent_brain_pointer_change_aborts_before_publication(self):
        original = (self.engine / "brain.json").read_bytes()
        changed = False

        def mutate_once(_size, _label):
            nonlocal changed
            if not changed:
                changed = True
                brain = json.loads(original)
                brain["test_writer_marker"] = "new commit"
                (self.engine / "brain.json").write_bytes(canonical(brain))
            return True

        with self.assertRaisesRegex(ValueError, "pointer changed"):
            rebuild_committed_paged_cache(
                self.engine, self.caches, memory_reserve=mutate_once
            )
        self.assertEqual(self.published_paths(), [])

    def test_prune_only_helper_created_old_cache_after_clean_switch(self):
        first = rebuild_committed_paged_cache(self.engine, self.caches)
        second = rebuild_committed_paged_cache(self.engine, self.caches)
        foreign = self.caches / ("paged-" + first.generation_sha256[:16] + "-" + "f" * 32)
        foreign.mkdir()
        (foreign / "keep.txt").write_text("user data", "utf-8")
        dirty = PagedAssemblyIndex(second.path, dimensions=16, seed=29)
        dirty.upsert({"id": "dirty", "fingerprint": "dirty-fingerprint"})
        with self.assertRaisesRegex(ValueError, "not verified and clean"):
            prune_derived_caches(self.caches, second.path, second.generation_sha256)
        self.assertTrue(first.path.exists())
        third = rebuild_committed_paged_cache(self.engine, self.caches)
        removed = prune_derived_caches(
            self.caches, third.path, third.generation_sha256
        )
        self.assertEqual(
            set(removed["removedPaths"]),
            {str(first.path.parent), str(second.path.parent)},
        )
        self.assertFalse(first.path.exists())
        self.assertFalse(second.path.exists())
        self.assertTrue(third.path.exists())
        self.assertEqual((foreign / "keep.txt").read_text("utf-8"), "user data")


if __name__ == "__main__":
    unittest.main()
