"""Pure storage checks: no cortical model or training is constructed."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omni_core.bounded_synapse_persistence import (
    BoundedSynapseShardPlan,
    PagedHotNodeIds,
)
from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.paged_assembly_view import PagedAssemblyView
from omni_core.persistence import atomic_save_tensors
from omni_core.vsa import (
    LazyPersistedSynapses, NeuralSubstrate, _forward_hot_ids_sha256,
)


class BoundedSynapsePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory(prefix="omni-synapse-shards-")
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        (self.root / "blobs").mkdir()
        self.source = SimpleNamespace(
            synapses={},
            assemblies=[{"id": "a"}],
            neurons={"a": {"region": "semantic"}, "b": {"region": "semantic"}},
            state_revision=0,
            growth_guard=None,
        )

    def _callbacks(self):
        def write_json(value):
            payload = NeuralSubstrate._canonical_json(value)
            checksum = hashlib.sha256(payload).hexdigest()
            path = self.root / "blobs" / (checksum + ".json")
            path.write_bytes(payload)
            return {"path": "blobs/" + path.name, "sha256": checksum, "bytes": len(payload)}

        def write_tensor(values, _reusable):
            scratch = self.root / "tensor-scratch.safetensors"
            atomic_save_tensors(scratch, values, metadata={"format": "omni-substrate-shards", "formatVersion": "3"})
            data = scratch.read_bytes()
            checksum = hashlib.sha256(data).hexdigest()
            path = self.root / "blobs" / (checksum + ".safetensors")
            path.write_bytes(data)
            scratch.unlink()
            return {"path": "blobs/" + path.name, "sha256": checksum, "bytes": len(data)}

        return write_json, write_tensor

    def _plan(self):
        write_json, write_tensor = self._callbacks()
        return BoundedSynapseShardPlan(
            self.source, self.root, records_per_shard=3,
            write_json_blob=write_json, write_tensor_blob=write_tensor,
        )

    def _save(self, label: str):
        plan = self._plan()
        descriptors = list(plan.iter_descriptors())
        generation = hashlib.sha256(label.encode()).hexdigest()
        manifest_sha = hashlib.sha256((label + "-manifest").encode()).hexdigest()
        plan.publish_index(
            generation, manifest_sha,
            [str(item["id"]) for item in self.source.assemblies],
        )
        pointer = {"activeGeneration": generation}
        plan.prepare_commit(pointer)
        plan.commit(pointer)
        return descriptors

    @staticmethod
    def _record(number: int, weight: int = 1):
        identifier = "a>b:association-%04d" % number
        return identifier, {
            "id": identifier,
            "source_id": "a",
            "target_id": "b",
            "kind": "association",
            "effective_weight": weight,
            "eligibility": 0.1,
            "plasticity": 1.0,
            "uses": number,
            "stability": 0.2,
            "last_updated_at": float(number),
        }

    def test_fresh_resident_map_streams_bounded_packed_shards_and_becomes_lazy(self):
        self.source.synapses.update(dict(self._record(n, (n % 3) - 1) for n in range(41)))
        descriptors = self._save("first")
        self.assertEqual(sum(item["count"] for item in descriptors), 41)
        self.assertTrue(all(1 <= item["count"] <= 3 for item in descriptors))
        self.assertIsInstance(self.source.synapses, LazyPersistedSynapses)
        self.assertEqual(len(self.source.synapses), 41)
        self.assertEqual(self.source.synapses.observed_effective_levels(), {-1, 0, 1})
        self.assertFalse(list((self.root / "staging").glob("*.sqlite3")))
        for descriptor in descriptors:
            record_payload = json.loads(
                (self.root / descriptor["records"]["path"]).read_text("utf-8")
            )
            self.assertEqual(record_payload["ids"], sorted(record_payload["ids"]))
            self.assertTrue(all("effective_weight" not in row for row in record_payload["records"]))

    def test_lazy_edit_add_delete_rewrites_only_affected_shards(self):
        self.source.synapses.update(dict(self._record(n) for n in range(21)))
        before = self._save("before")
        mapping = self.source.synapses
        assert isinstance(mapping, LazyPersistedSynapses)
        edit_id, _ = self._record(4)
        delete_id, _ = self._record(8)
        mapping[edit_id]["effective_weight"] = -1
        del mapping[delete_id]
        new_id, new_record = self._record(200, 0)
        mapping[new_id] = new_record
        after = self._save("after")
        self.assertEqual(len(self.source.synapses), 21)
        self.assertEqual(self.source.synapses[edit_id]["effective_weight"], -1)
        self.assertNotIn(delete_id, self.source.synapses)
        self.assertEqual(self.source.synapses[new_id]["effective_weight"], 0)
        old = {(item["bucket"], item["part"]): item for item in before}
        new = {(item["bucket"], item["part"]): item for item in after}
        self.assertTrue(any(old.get(key) == item for key, item in new.items()))
        self.assertTrue(any(old.get(key) != item for key, item in new.items()))
        self.assertFalse(list((self.root / "staging").glob("*.sqlite3")))

    def test_rejects_fractional_forward_weight_before_publication(self):
        identifier, record = self._record(1)
        record["effective_weight"] = 0.25
        self.source.synapses[identifier] = record
        plan = self._plan()
        with self.assertRaises(ValueError):
            list(plan.iter_descriptors())
        self.assertFalse(list((self.root / "staging").glob("*.sqlite3")))
        self.assertIsInstance(self.source.synapses, dict)

    def test_rejects_raw_source_text_in_synapse_metadata(self):
        identifier, record = self._record(1)
        record["source_text"] = "private training passage"
        self.source.synapses[identifier] = record
        with self.assertRaisesRegex(ValueError, "non-structural text"):
            list(self._plan().iter_descriptors())
        self.assertIsInstance(self.source.synapses, dict)

    def test_disk_denial_does_not_create_scratch_or_blobs(self):
        identifier, record = self._record(1)
        self.source.synapses[identifier] = record
        write_json, write_tensor = self._callbacks()
        plan = BoundedSynapseShardPlan(
            self.source, self.root, records_per_shard=3,
            write_json_blob=write_json, write_tensor_blob=write_tensor,
            disk_reserve=lambda _size, _operation: False,
        )
        with self.assertRaisesRegex(Exception, "resource reserve|disk reserve"):
            list(plan.iter_descriptors())
        self.assertFalse((self.root / "staging").exists())
        self.assertFalse(list((self.root / "blobs").iterdir()))

    def test_paged_hot_ids_match_resident_hash_and_indexed_membership(self):
        view = PagedAssemblyView(PagedAssemblyIndex(self.root / "assemblies.sqlite3"))
        for name in ("z", "a", "m"):
            view.append({"id": name, "fingerprint": "fingerprint-" + name})
        paged = PagedHotNodeIds(view)
        self.assertEqual(list(paged), ["a", "m", "z"])
        self.assertIn("m", paged)
        self.assertNotIn("unknown", paged)
        self.assertEqual(_forward_hot_ids_sha256(paged), _forward_hot_ids_sha256(["z", "a", "m"]))

    def test_new_assembly_reindexes_reused_synapse_endpoints(self):
        self.source.synapses.update(dict(self._record(n) for n in range(8)))
        first = self._save("hot-before")
        self.source.assemblies.append({"id": "b"})
        second = self._save("hot-after")
        self.assertEqual(first, second)
        lazy = self.source.synapses
        assert isinstance(lazy, LazyPersistedSynapses)
        self.assertEqual(len(lazy.connected_records(["b"])), 8)

    def test_paged_hot_ids_load_lazy_index_without_resident_id_set(self):
        view = PagedAssemblyView(PagedAssemblyIndex(self.root / "hot-index.sqlite3"))
        view.append({"id": "a", "fingerprint": "assembly-a"})
        view.append({"id": "b", "fingerprint": "assembly-b"})
        self.source.assemblies = view
        self.source.synapses.update(dict(self._record(n) for n in range(11)))
        plan = self._plan()
        self.assertEqual(sum(item["count"] for item in plan.iter_descriptors()), 11)
        generation = hashlib.sha256(b"paged-hot").hexdigest()
        manifest_sha = hashlib.sha256(b"paged-hot-manifest").hexdigest()
        plan.publish_index(generation, manifest_sha, PagedHotNodeIds(view))
        pointer = {"activeGeneration": generation}
        plan.prepare_commit(pointer)
        plan.commit(pointer)
        lazy = self.source.synapses
        assert isinstance(lazy, LazyPersistedSynapses)
        self.assertIsInstance(lazy._hot_node_ids, PagedHotNodeIds)
        self.assertEqual(len(lazy.connected_records(["b"])), 11)


if __name__ == "__main__":
    unittest.main()
