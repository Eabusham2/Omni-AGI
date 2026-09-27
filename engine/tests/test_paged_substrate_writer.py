"""Pure storage checks for the bounded paged v3 substrate writer."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.paged_assembly_view import PagedAssemblyView
from omni_core.paged_substrate_writer import (
    _verified_identity_sha,
    write_paged_substrate_generation,
)
from omni_core.packed_vsa_vectors import PackedTernaryVectorView
from omni_core.vsa import NeuralSubstrate, SubstrateResourcePause


def _assembly(number: int) -> dict:
    return {
        "id": "assembly-%04d" % number,
        "fingerprint": "fingerprint-%04d" % number,
        "neuron_ids": ["neuron-%04d" % number],
        "source_provenance": {"sensor": 1},
        "rehearsals": 1,
        "importance": 0.25,
    }


class PagedSubstrateWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(prefix="omni-paged-writer-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.store = self.root / "substrate"
        self.memory = NeuralSubstrate(16, seed=13)
        self.vector = torch.tensor([1, 0, -1, 0] * 4, dtype=torch.int8)

    def _populate(self, count: int, *, page_assemblies: bool = True) -> None:
        for number in range(count):
            for identifier in ("neuron-%04d" % number, "assembly-%04d" % number):
                self.memory.neurons[identifier] = {"id": identifier, "label": identifier}
                self.memory.neuron_vectors[identifier] = self.vector
            record = _assembly(number)
            self.memory.assemblies.append(record)
            self.memory.assembly_vectors.link(record["id"])
        if page_assemblies:
            self.memory.enable_paged_assemblies(
                PagedAssemblyIndex(self.root / "assembly.sqlite3"), page_size=3
            )

    def _generation(self, pointer: dict) -> dict:
        path = self.store / pointer["generationManifest"]
        return json.loads(path.read_text("utf-8"))

    def test_emits_exact_v3_bounded_alias_shards_loadable_by_current_reader(self) -> None:
        self._populate(27)
        self.memory.neurons["neuron-0000"].update({
            "retention_score": 0.7,
            "activity_score": 0.2,
            "unfinished": False,
            "settling_signals": {"novelty": 0.3},
            "last_settled_at": "2026-09-26T12:00:00Z",
        })
        pointer = self.memory.save_sharded(self.store, records_per_shard=2)
        generation = self._generation(pointer)
        self.assertEqual(generation["formatVersion"], 3)
        self.assertEqual(generation["counts"], {
            "neurons": 54, "assemblies": 27, "synapses": 0,
        })
        self.assertTrue(all(0 < shard["count"] <= 2 for shard in generation["shards"]))
        for shard in generation["shards"]:
            payload = json.loads((self.store / shard["records"]["path"]).read_text("utf-8"))
            self.assertEqual(payload["ids"], sorted(payload["ids"]))
            if shard["kind"] == "assemblies":
                self.assertIsNone(shard["tensors"])
                self.assertEqual(payload["vectorStorage"], "shared-neuron-packed")
                self.assertTrue(all("source_text" not in record for record in payload["records"]))
            else:
                self.assertEqual(payload["vectorIds"], payload["ids"])
                self.assertEqual(set(payload["packedVectorState"]["ids"]), set(payload["ids"]))
                self.assertIsNotNone(shard["tensors"])
        restored = NeuralSubstrate.load_sharded(
            self.store, self.memory.metadata(include_records=False), lazy_synapses=False
        )
        self.assertEqual(len(restored.neurons), 54)
        self.assertEqual(len(restored.assemblies), 27)
        self.assertEqual(restored.neurons["neuron-0000"]["retention_score"], 0.7)
        self.assertEqual(restored.neuron_vectors.packed_row("assembly-0026"),
                         self.memory.neuron_vectors.packed_row("assembly-0026"))

    def test_unchanged_blobs_are_reused_and_one_edit_changes_only_local_content(self) -> None:
        self._populate(6)
        first = write_paged_substrate_generation(
            self.memory, self.store, records_per_shard=2
        )
        first_shards = self._generation(first)["shards"]
        blob_inodes = {
            shard["records"]["path"]: (self.store / shard["records"]["path"]).stat().st_ino
            for shard in first_shards
        }
        second = write_paged_substrate_generation(
            self.memory, self.store, records_per_shard=2
        )
        self.assertEqual(first, second)
        for relative, inode in blob_inodes.items():
            self.assertEqual((self.store / relative).stat().st_ino, inode)
        with self.memory.assemblies.transaction(max_rows=1) as edit:
            edit.edit_by_id("assembly-0001", lambda row: row.update(rehearsals=2))
        third = write_paged_substrate_generation(
            self.memory, self.store, records_per_shard=2
        )
        self.assertNotEqual(third["activeGeneration"], first["activeGeneration"])
        third_shards = self._generation(third)["shards"]
        old_by_key = {(item["kind"], item["bucket"], item["part"]): item for item in first_shards}
        new_by_key = {(item["kind"], item["bucket"], item["part"]): item for item in third_shards}
        changed = [key for key in new_by_key if old_by_key[key]["records"] != new_by_key[key]["records"]]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0][0], "assemblies")

    def test_paged_packed_vectors_emit_exact_reloadable_rows(self) -> None:
        self._populate(5)
        self.memory.enable_paged_vectors(self.root / "vectors.sqlite3", shard_rows=2)
        pointer = write_paged_substrate_generation(
            self.memory, self.store, records_per_shard=2
        )
        restored = NeuralSubstrate.load_sharded(
            self.store, self.memory.metadata(include_records=False), lazy_synapses=False
        )
        self.assertEqual(pointer["counts"]["neurons"], 10)
        for identifier in self.memory.neurons:
            self.assertEqual(
                restored.neuron_vectors.packed_row(identifier),
                self.memory.neuron_vectors.packed_row(identifier),
            )

    def test_shared_sqlite_index_and_vector_object_is_one_authority(self) -> None:
        self._populate(2, page_assemblies=False)
        source_assemblies = list(self.memory.assemblies)
        path = self.root / "shared.sqlite3"
        shared_index = PagedAssemblyIndex(
            path,
            dimensions=self.memory.space.dimensions,
            seed=self.memory.space.seed,
            zero_deadband=self.memory.neuron_vectors.zero_deadband,
        )
        identifiers = list(self.memory.neuron_vectors)
        for offset in range(0, len(identifiers), 2):
            metadata, tensors = self.memory.neuron_vectors.export_state(
                keys=identifiers[offset : offset + 2]
            )
            shared_index._vectors.import_state(metadata, tensors)
        self.memory.neuron_vectors = shared_index._vectors
        view = PackedTernaryVectorView(self.memory.neuron_vectors)
        paged_assemblies = PagedAssemblyView(shared_index)
        for record in source_assemblies:
            view.link(record["id"])
            paged_assemblies.append(record)
        self.memory.assembly_vectors = view
        self.memory.assemblies = paged_assemblies
        pointer = self.memory.save_sharded(self.store, records_per_shard=2)
        self.assertEqual(pointer["counts"]["assemblies"], 2)
        reloaded = NeuralSubstrate.load_sharded(
            self.store, self.memory.metadata(include_records=False), lazy_synapses=False
        )
        self.assertEqual(len(reloaded.assemblies), 2)

    def test_existing_unchanged_lazy_synapses_are_referenced_not_rewritten(self) -> None:
        self._populate(2, page_assemblies=False)
        record_id = "assembly-0000>neuron-0000:semantic"
        self.memory.synapses[record_id] = {
            "id": record_id,
            "source_id": "assembly-0000",
            "target_id": "neuron-0000",
            "kind": "semantic",
            "effective_weight": 1,
            "eligibility": 0.0,
            "plasticity": 1.0,
            "uses": 2,
            "stability": 0.0,
            "last_updated_at": 0.0,
        }
        original = self.memory.save_sharded(self.store, records_per_shard=2)
        metadata = self.memory.metadata(include_records=False)
        restored = NeuralSubstrate.load_sharded(
            self.store, metadata, lazy_synapses=True
        )
        restored.enable_paged_assemblies(
            PagedAssemblyIndex(self.root / "assembly.sqlite3"), page_size=2
        )
        pointer = write_paged_substrate_generation(
            restored, self.store, records_per_shard=2
        )
        cached_before = _verified_identity_sha.cache_info().hits
        write_paged_substrate_generation(restored, self.store, records_per_shard=2)
        self.assertGreater(_verified_identity_sha.cache_info().hits, cached_before)
        prior_synapses = [
            item for item in self._generation(original)["shards"]
            if item["kind"] == "synapses"
        ]
        current_synapses = [
            item for item in self._generation(pointer)["shards"]
            if item["kind"] == "synapses"
        ]
        self.assertEqual(current_synapses, prior_synapses)
        reloaded = NeuralSubstrate.load_sharded(
            self.store, restored.metadata(include_records=False), lazy_synapses=True
        )
        self.assertEqual(reloaded.synapses[record_id]["effective_weight"], 1)

    def test_corrupt_reused_blob_blocks_new_pointer(self) -> None:
        self._populate(2)
        first = write_paged_substrate_generation(
            self.memory, self.store, records_per_shard=2
        )
        prior_pointer = (self.store / "manifest.json").read_bytes()
        record = self._generation(first)["shards"][0]["records"]
        blob = self.store / record["path"]
        payload = blob.read_bytes()
        blob.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            write_paged_substrate_generation(
                self.memory, self.store, records_per_shard=2
            )
        self.assertEqual((self.store / "manifest.json").read_bytes(), prior_pointer)

    def test_denied_reserve_or_interrupted_pointer_preserves_prior_generation(self) -> None:
        self._populate(3)
        first = write_paged_substrate_generation(self.memory, self.store, records_per_shard=2)
        before = (self.store / "manifest.json").read_bytes()
        with self.memory.assemblies.transaction(max_rows=1) as edit:
            edit.edit_by_id("assembly-0001", lambda row: row.update(rehearsals=2))
        with self.assertRaises(SubstrateResourcePause):
            write_paged_substrate_generation(
                self.memory, self.store, records_per_shard=2,
                disk_reserve=lambda _size, _operation: False,
            )
        self.assertEqual((self.store / "manifest.json").read_bytes(), before)
        with mock.patch(
            "omni_core.paged_substrate_writer._replace_pointer",
            side_effect=RuntimeError("interrupted before pointer"),
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                write_paged_substrate_generation(
                    self.memory, self.store, records_per_shard=2
                )
        self.assertEqual((self.store / "manifest.json").read_bytes(), before)
        previous = NeuralSubstrate.load_sharded(
            self.store,
            {**self.memory.metadata(include_records=False), "persistence": first},
            lazy_synapses=False,
        )
        self.assertEqual(previous.assemblies[1]["rehearsals"], 1)

    def test_raw_text_is_rejected_and_dynamic_synapses_reload_exactly(self) -> None:
        self._populate(1)
        self.memory.neurons["neuron-0000"]["source_text"] = "private passage"
        with self.assertRaisesRegex(ValueError, "non-structural text"):
            write_paged_substrate_generation(self.memory, self.store)
        self.assertFalse((self.store / "manifest.json").exists())
        del self.memory.neurons["neuron-0000"]["source_text"]
        self.memory.synapses["assembly-0000>neuron-0000:semantic"] = {
            "id": "assembly-0000>neuron-0000:semantic",
            "source_id": "assembly-0000",
            "target_id": "neuron-0000",
            "effective_weight": 1,
        }
        pointer = self.memory.save_sharded(self.store, records_per_shard=2)
        self.assertEqual(pointer["counts"]["synapses"], 1)
        restored = NeuralSubstrate.load_sharded(
            self.store, self.memory.metadata(include_records=False), lazy_synapses=True
        )
        synapse = restored.synapses["assembly-0000>neuron-0000:semantic"]
        self.assertEqual(synapse["effective_weight"], 1)
        self.assertEqual(synapse["source_id"], "assembly-0000")

    def test_changed_lazy_synapse_rewrites_one_shard_without_losing_old_pointer(self) -> None:
        self._populate(2)
        key = "assembly-0000>neuron-0000:semantic"
        self.memory.synapses[key] = {
            "id": key,
            "source_id": "assembly-0000",
            "target_id": "neuron-0000",
            "kind": "semantic",
            "effective_weight": 1,
            "eligibility": 0.0,
            "plasticity": 1.0,
            "uses": 1,
            "stability": 0.0,
            "last_updated_at": 0.0,
        }
        first = self.memory.save_sharded(self.store, records_per_shard=2)
        self.assertEqual(self.memory.synapses[key]["uses"], 1)
        self.memory.synapses[key]["uses"] = 3
        with mock.patch(
            "omni_core.paged_substrate_writer._replace_pointer",
            side_effect=RuntimeError("interrupted before pointer"),
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.memory.save_sharded(self.store, records_per_shard=2)
        self.assertEqual(
            json.loads((self.store / "manifest.json").read_text("utf-8")), first
        )
        self.assertEqual(self.memory.persistence_manifest, first)
        self.assertEqual(self.memory.synapses[key]["uses"], 3)
        second = self.memory.save_sharded(self.store, records_per_shard=2)
        self.assertNotEqual(first["activeGeneration"], second["activeGeneration"])
        reloaded = NeuralSubstrate.load_sharded(
            self.store, self.memory.metadata(include_records=False), lazy_synapses=True
        )
        self.assertEqual(reloaded.synapses[key]["uses"], 3)
        old = NeuralSubstrate.load_sharded(
            self.store,
            {**self.memory.metadata(include_records=False), "persistence": first},
            lazy_synapses=True,
        )
        self.assertEqual(old.synapses[key]["uses"], 1)


if __name__ == "__main__":
    unittest.main()
