"""Storage-only incremental checkpoint tests: no brain/model constructors."""

import json
import hashlib
import os
import sqlite3
import torch
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from omni_core.committed_paged_cache import _cache_has_clean_binding, rebuild_committed_paged_cache
from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.paged_assembly_vector_view import PagedAssemblyVectorView
from omni_core.paged_assembly_view import PagedAssemblyView
from omni_core.bounded_synapse_persistence import PagedHotNodeIds
from omni_core.paged_assembly_membership import checksum_for_ids
from omni_core.authenticated_paged_cache import cache_session, canonical
from omni_core.paged_dirty_journal import journal_status
from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.paged_forward_index import iter_entries, publish_entries, BlobForwardTopology
from omni_core.paged_substrate_writer import (
    _identity_proof_supported, _verified_file,
    commit_paged_substrate_generation, forget_paged_blob_proof, write_paged_substrate_generation,
)
from omni_core.persistence import atomic_write_json
from omni_core.vsa import LazyPersistedSynapses, NeuralSubstrate, SubstrateResourcePause, _forward_hot_ids_sha256


class IncrementalPagedCheckpointTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(prefix="omni-incremental-storage-")
        self.addCleanup(folder.cleanup)
        self.engine = Path(folder.name)
        self.store = self.engine / "substrate"
        self.path = self.engine / "cache" / "working.sqlite3"
        index = PagedAssemblyIndex(self.path, dimensions=16, seed=19)
        # Constructor-free substrate shell used ONLY as the writer's storage
        # protocol. No model, training, cognition, application or load_sharded.
        self.substrate = object.__new__(NeuralSubstrate)
        self.addCleanup(self.substrate.release_paged_recall_scratch)
        self.substrate.space = SimpleNamespace(dimensions=16, seed=19)
        self.substrate.state_revision = 0
        self.substrate.growth_events = 0
        self.substrate.growth_pauses = 0
        self.substrate.growth_guard = None
        self.substrate.persistence_manifest = None
        self.substrate.synapses = {}
        self.substrate.neurons = PagedNeuronMetadata(self.path)
        self.substrate.neuron_vectors = index._vectors
        self.substrate.assemblies = PagedAssemblyView(index)
        self.substrate.assembly_vectors = PagedAssemblyVectorView(index, index._vectors)
        for number in range(24):
            neuron = "neuron-%04d" % number
            assembly = "assembly-%04d" % number
            self._add_neuron(neuron)
            self._add_neuron(assembly)
            self.substrate.assemblies.append({
                "id": assembly, "fingerprint": "fingerprint-%04d" % number,
                "neuron_ids": [neuron], "rehearsals": 1,
            })
        self.first = self._write()
        self._commit(self.first)

    def _add_neuron(self, identifier):
        self.substrate.neurons[identifier] = {
            "id": identifier, "label": identifier,
            "activation": 1.0, "uncertainty": 0.1, "exposures": 1,
        }
        self.substrate.neuron_vectors.set_packed(identifier, b"\x55" * 4)

    def _write(self, **options):
        return write_paged_substrate_generation(
            self.substrate, self.store, records_per_shard=2, **options,
        )

    def _brain_commit(self, pointer):
        atomic_write_json(self.engine / "brain.json", {"substrate": {
            "schema": NeuralSubstrate.SCHEMA, "dimensions": 16,
            "seed": 19, "persistence": pointer,
        }, "brain_id": "storage-test"})

    def _commit(self, pointer):
        self._brain_commit(pointer)
        return commit_paged_substrate_generation(
            self.substrate, self.store, self.engine / "brain.json",
        )

    def _descriptors(self, pointer):
        generation = json.loads((self.store / pointer["generationManifest"]).read_bytes())
        return {
            (row["kind"], row["bucket"], row["part"]): row
            for row in generation["shards"]
        }

    def _placements(self, pointer):
        result = {}
        for key, row in self._descriptors(pointer).items():
            payload = json.loads((self.store / row["records"]["path"]).read_bytes())
            for identifier in payload["ids"]:
                result[(key[0], identifier)] = key[1:]
        return result

    def test_one_edit_reads_only_complete_changed_groups_and_reuses_other_descriptors(self):
        self.substrate.neurons.edit_by_id(
            "neuron-0001", lambda row: row.update(label="revised")
        )
        self.substrate.neuron_vectors.set_packed("neuron-0002", b"\x00" * 4)
        with self.substrate.assemblies.transaction(max_rows=1) as edit:
            edit.edit_by_id("assembly-0003", lambda row: row.update(rehearsals=2))
        with mock.patch.object(PagedNeuronMetadata, "items", side_effect=AssertionError("full neuron scan")), \
             mock.patch.object(PagedAssemblyView, "__iter__", side_effect=AssertionError("full assembly scan")):
            second = self._write()
        before, after = self._descriptors(self.first), self._descriptors(second)
        changed = {key for key in after if before[key] != after[key]}
        self.assertLessEqual(len(changed), 3)
        self.assertGreaterEqual(len(changed), 2)
        self.assertEqual(self._placements(self.first), self._placements(second))
        self.assertEqual(self._commit(second)["incremental"], True)
        self.assertEqual(journal_status(self.path, second)["dirtyIds"], 0)
        self.assertEqual(self.substrate.neurons.status()["committedGenerationSha256"],
                         second["activeGeneration"])

    def test_new_and_deleted_ids_keep_stable_bounded_parts(self):
        old = self._placements(self.first)
        candidates = ["new-%04d" % number for number in range(200)]
        selected = [value for value in candidates
                    if NeuralSubstrate._bucket("neurons", value) == "a"][:8]
        self.assertEqual(len(selected), 8)
        for identifier in selected:
            self._add_neuron(identifier)
        del self.substrate.neurons["neuron-0005"]
        del self.substrate.neuron_vectors["neuron-0005"]
        second = self._write()
        new = self._placements(second)
        for key, placement in old.items():
            if key != ("neurons", "neuron-0005"):
                self.assertEqual(new[key], placement)
        self.assertNotIn(("neurons", "neuron-0005"), new)
        self.assertTrue(all(row["count"] <= 2 for row in self._descriptors(second).values()))
        self.assertEqual(second["counts"]["neurons"], self.first["counts"]["neurons"] + 7)
        self._commit(second)
        # A later small append retains every already committed placement too.
        self._add_neuron("another-new-neuron")
        third = self._write()
        third_placements = self._placements(third)
        for key, placement in new.items():
            self.assertEqual(third_placements[key], placement)
        self._commit(third)

    def test_global_decay_rewrites_every_neuron_group_exactly(self):
        self.substrate.neurons.decay(0.2)
        self.assertTrue(journal_status(self.path, self.first)["fullNeuronsDirty"])
        second = self._write()
        before, after = self._descriptors(self.first), self._descriptors(second)
        for key, row in before.items():
            if key[0] == "neurons":
                self.assertNotEqual(row["records"], after[key]["records"])
                payload = json.loads((self.store / after[key]["records"]["path"]).read_bytes())
                self.assertTrue(all(abs(record["activation"] - 0.8) < 1e-12
                                    for record in payload["records"]))
                self.assertEqual(row["tensors"], after[key]["tensors"])
            else:
                self.assertEqual(row, after[key])
        self._commit(second)
        self.assertFalse(journal_status(self.path, second)["fullNeuronsDirty"])

    def test_uncommitted_convenience_pointer_never_clears_journal_or_controls_recovery(self):
        self.substrate.neurons.edit_by_id(
            "neuron-0001", lambda row: row.update(label="not committed")
        )
        second = self._write()
        self.assertNotEqual(second, self.first)
        with self.assertRaisesRegex(ValueError, "has not committed"):
            commit_paged_substrate_generation(
                self.substrate, self.store, self.engine / "brain.json",
            )
        self.assertEqual(journal_status(self.path, self.first)["dirtyIds"], 1)
        recovered = rebuild_committed_paged_cache(self.engine, self.engine / "recovered")
        neurons = PagedNeuronMetadata(recovered.path)
        self.assertEqual(neurons["neuron-0001"]["label"], "neuron-0001")
        self.assertEqual(journal_status(recovered.path, self.first)["dirtyIds"], 0)
        self.assertEqual(recovered.generation_sha256, self.first["activeGeneration"])

    def test_rebase_revision_drift_and_reserve_pause_do_not_clear_learning(self):
        self.substrate.neurons.edit_by_id(
            "neuron-0001", lambda row: row.update(label="first edit")
        )
        second = self._write()
        self.substrate.neurons.edit_by_id(
            "neuron-0001", lambda row: row.update(label="later edit")
        )
        self._brain_commit(second)
        with self.assertRaisesRegex(ValueError, "revision|epoch"):
            commit_paged_substrate_generation(
                self.substrate, self.store, self.engine / "brain.json",
            )
        self.assertEqual(journal_status(self.path, self.first)["dirtyIds"], 1)

        self.substrate.persistence_manifest = dict(self.first)
        before = (self.store / "manifest.json").read_bytes()
        with self.assertRaises(SubstrateResourcePause):
            self._write(disk_reserve=lambda _size, operation: operation != "substrate immutable blob")
        self.assertEqual((self.store / "manifest.json").read_bytes(), before)
        self.assertEqual(journal_status(self.path, self.first)["dirtyIds"], 1)

    def test_post_commit_rebase_reserve_pause_is_retryable_without_losing_dirty_ids(self):
        self.substrate.neurons.edit_by_id(
            "neuron-0001", lambda row: row.update(label="committed edit")
        )
        second = self._write()
        self._brain_commit(second)
        with self.assertRaises(SubstrateResourcePause):
            commit_paged_substrate_generation(
                self.substrate, self.store, self.engine / "brain.json",
                disk_reserve=lambda _size, _operation: False,
            )
        self.assertEqual(journal_status(self.path, self.first)["dirtyIds"], 1)
        self.assertEqual(json.loads((self.engine / "brain.json").read_bytes())
                         ["substrate"]["persistence"], second)
        result = commit_paged_substrate_generation(
            self.substrate, self.store, self.engine / "brain.json",
        )
        self.assertTrue(result["incremental"])
        self.assertEqual(journal_status(self.path, second)["dirtyIds"], 0)

    def test_store_counts_roll_back_and_status_never_counts_source_tables(self):
        statements = []
        tracked = (
            self.substrate.neurons, self.substrate.neuron_vectors,
            self.substrate.assemblies.index,
        )
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM paged_neuron_records WHERE neuron_id='neuron-0001'")
            connection.rollback()
        self.assertEqual(len(self.substrate.neurons), 48)
        patches = [mock.patch.object(store, "_connect", (lambda original=store._connect: self._traced(original, statements)))
                   for store in tracked]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        for store in tracked:
            store.status()
        len(self.substrate.neurons)
        len(self.substrate.neuron_vectors)
        len(self.substrate.assemblies)
        self.assertFalse(any("COUNT(" in statement.upper() for statement in statements))

    def test_new_assembly_uses_merkle_delta_without_an_all_id_or_record_walk(self):
        self.substrate.assemblies.append({
            "id": "neuron-0007", "fingerprint": "new-assembly-fingerprint",
            "neuron_ids": ["neuron-0008"], "rehearsals": 1,
        })
        with mock.patch.object(PagedHotNodeIds, "iter_sorted_ids", side_effect=AssertionError("all ID scan")), \
             mock.patch.object(PagedNeuronMetadata, "items", side_effect=AssertionError("all neuron scan")), \
             mock.patch.object(PagedAssemblyView, "__iter__", side_effect=AssertionError("all assembly scan")):
            second = self._write()
        self.assertEqual(second["counts"]["assemblies"], 25)
        self._commit(second)

    def test_merkle_root_is_order_independent_exact_on_delete_and_rollback(self):
        index = self.substrate.assemblies.index
        ids = ["assembly-%04d" % number for number in range(24)]
        original = index.ids_sha256()
        self.assertEqual(original, checksum_for_ids(reversed(ids)))
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM assembly_records WHERE assembly_id='assembly-0007'")
            connection.rollback()
        self.assertEqual(index.ids_sha256(), original)
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM assembly_records WHERE assembly_id='assembly-0007'")
        self.assertEqual(index.ids_sha256(), checksum_for_ids(
            identifier for identifier in ids if identifier != "assembly-0007"
        ))

    def test_merkle_node_tamper_fails_closed(self):
        index = self.substrate.assemblies.index
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE assembly_membership_nodes SET sha256=? WHERE prefix="
                "(SELECT value FROM index_metadata WHERE key='assembly_membership_root')",
                ("0" * 64,),
            )
        with self.assertRaisesRegex(ValueError, "Merkle|checksum"):
            index.ids_sha256()

    def test_added_hot_member_rebuilds_zero_weight_incident_synapse_locations(self):
        key = "neuron-0007>neuron-0008:semantic"
        self.substrate.synapses[key] = {
            "id": key, "source_id": "neuron-0007", "target_id": "neuron-0008",
            "kind": "semantic", "effective_weight": 0,
        }
        second = self._write()
        self._commit(second)
        previous_synapses = {key: value for key, value in self._descriptors(second).items()
                             if key[0] == "synapses"}
        self.substrate.assemblies.append({
            "id": "neuron-0007", "fingerprint": "hot-neuron-0007",
            "neuron_ids": ["neuron-0008"], "rehearsals": 1,
        })
        third = self._write()
        current_synapses = {key: value for key, value in self._descriptors(third).items()
                            if key[0] == "synapses"}
        self.assertEqual(previous_synapses, current_synapses)
        forward = json.loads((self.store / "forward-index" / "generations" /
                              (third["activeGeneration"] + ".json")).read_bytes())
        self.assertEqual(forward["formatVersion"], 4)
        self.assertEqual(forward["hotNodeIdsChecksumAlgorithm"], "sha256-patricia-id-set-v1")
        self.assertIn(["neuron-0007", key, 0], next(iter_entries(self.store, forward))["hotLocations"])
        self._commit(third)

    def test_checked_local_immutable_proofs_avoid_hashing_untouched_blob_bodies(self):
        first_blob = next(iter(self._descriptors(self.first).values()))["records"]
        path = self.store / first_blob["path"]
        if not _identity_proof_supported(path, path.stat()):
            self.skipTest("filesystem requires conservative full content rehash")
        original_hash = NeuralSubstrate._file_sha256
        self.substrate.neuron_vectors.set_packed("neuron-0001", b"\x00" * 4)
        with mock.patch.object(NeuralSubstrate, "_file_sha256", wraps=original_hash) as hashing:
            second = self._write()
        touched = [Path(call.args[0]).resolve() for call in hashing.call_args_list]
        self.assertTrue(touched)
        old_blobs = {
            (self.store / spec["path"]).resolve()
            for descriptor in self._descriptors(self.first).values()
            for spec in (descriptor["records"], descriptor.get("tensors")) if spec is not None
        }
        self.assertFalse(old_blobs.intersection(touched))
        self._commit(second)

    def test_restored_mtime_tamper_is_rejected_and_weak_fs_never_reuses_identity_proof(self):
        spec = next(iter(self._descriptors(self.first).values()))["records"]
        path = self.store / spec["path"]
        _verified_file(path, spec["sha256"], spec["bytes"])
        original_hash = NeuralSubstrate._file_sha256
        with mock.patch("omni_core.paged_substrate_writer._identity_proof_supported", return_value=False), \
             mock.patch.object(NeuralSubstrate, "_file_sha256", wraps=original_hash) as hashing:
            _verified_file(path, spec["sha256"], spec["bytes"])
            _verified_file(path, spec["sha256"], spec["bytes"])
        self.assertEqual(hashing.call_count, 2)
        before = path.stat()
        content = path.read_bytes()
        path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        pointer_before = (self.store / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "checksum"):
            self._write()
        self.assertEqual((self.store / "manifest.json").read_bytes(), pointer_before)

    def test_same_authoritative_generation_upgrades_a_verified_sorted_id_forward_index(self):
        path = self.store / "forward-index" / "generations" / (self.first["activeGeneration"] + ".json")
        prior = json.loads(path.read_bytes())
        ids = ["assembly-%04d" % number for number in range(24)]
        prior["hotNodeIdsChecksumAlgorithm"] = "sha256-sorted-ids-v1"
        prior["hotNodeIdsSha256"] = _forward_hot_ids_sha256(ids, "sha256-sorted-ids-v1")
        body = {key: value for key, value in prior.items() if key != "contentSha256"}
        prior["contentSha256"] = hashlib.sha256(NeuralSubstrate._canonical_json(body)).hexdigest()
        atomic_write_json(path, prior)
        self.substrate.synapses._forward_index_manifest = dict(prior)
        second = self._write()
        self.assertEqual(second, self.first)
        current = json.loads(path.read_bytes())
        self.assertEqual(current["hotNodeIdsChecksumAlgorithm"], "sha256-patricia-id-set-v1")
        self._commit(second)

    def test_neuron_only_drift_is_not_a_clean_derived_cache_binding(self):
        self.assertTrue(_cache_has_clean_binding(self.path, self.first["activeGeneration"]))
        self.substrate.neurons.edit_by_id(
            "neuron-0001", lambda row: row.update(label="metadata-only update")
        )
        self.assertFalse(_cache_has_clean_binding(self.path, self.first["activeGeneration"]))

    @staticmethod
    def _traced(original, statements):
        connection = original()
        connection.set_trace_callback(statements.append)
        return connection

    def _cold_edges(self, count=80):
        keys = []
        for source in range(24):
            for target in range(24):
                if len(keys) == count:
                    break
                identifier = "neuron-%04d>neuron-%04d:semantic" % (source, target)
                self.substrate.synapses[identifier] = {
                    "id": identifier, "source_id": "neuron-%04d" % source,
                    "target_id": "neuron-%04d" % target, "kind": "semantic",
                    "effective_weight": 0, "uses": target % 3,
                }
                keys.append(identifier)
            if len(keys) == count:
                break
        pointer = self._write()
        self._commit(pointer)
        return pointer, keys

    def test_many_zero_weight_groups_reindex_only_exact_incident_old_groups(self):
        before, keys = self._cold_edges()
        endpoint = "neuron-0022"
        index = self.substrate.assemblies.index._synapse_endpoint_index
        incident = set(index.incident_groups(endpoint))
        all_groups = {key[1:] for key in self._descriptors(before) if key[0] == "synapses"}
        self.assertTrue(incident)
        self.assertLess(len(incident), len(all_groups))
        self.substrate.assemblies.append({
            "id": endpoint, "fingerprint": "promoted-cold-endpoint", "neuron_ids": ["neuron-0023"],
        })
        lazy = self.substrate.synapses
        with mock.patch.object(lazy, "_group_records", wraps=lazy._group_records) as reading:
            after = self._write()
        self.assertEqual({call.args[0] for call in reading.call_args_list}, incident)
        result = self._commit(after)
        self.assertTrue(result["endpointLookupSelective"])
        self.assertEqual(result["synapseHotGroupsReindexed"], len(incident))
        hot = [value for entry in iter_entries(self.store, self.substrate.synapses._forward_index_manifest)
               for value in entry["hotLocations"] if value[0] == endpoint]
        expected_edges = [key for key in keys if ">" + endpoint + ":" in key
                          or key.startswith(endpoint + ">")]
        self.assertEqual({value[1] for value in hot}, set(expected_edges))

    def test_hot_endpoint_absence_reads_no_old_synapse_record_groups(self):
        self._cold_edges()
        self._add_neuron("brand-new-hot")
        self.substrate.assemblies.append({
            "id": "brand-new-hot", "fingerprint": "brand-new-hot-fingerprint", "neuron_ids": ["neuron-0001"],
        })
        lazy = self.substrate.synapses
        with mock.patch.object(lazy, "_group_records", side_effect=AssertionError("unrelated synapse read")):
            pointer = self._write()
        self.assertEqual(self._commit(pointer)["synapseHotGroupsReindexed"], 0)

    def test_endpoint_missing_member_and_missing_merkle_leaf_fail_closed(self):
        self._cold_edges()
        index = self.substrate.assemblies.index._synapse_endpoint_index
        endpoint = "neuron-0022"
        key = next(iter(index.incident_groups(endpoint)))
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM synapse_endpoint_members WHERE endpoint=? AND bucket=? AND part=?",
                               (endpoint, *key))
        with self.assertRaisesRegex(ValueError, "coverage|corrupt"):
            list(index.incident_groups(endpoint))
        # A deleted entire proof leaf is not interpreted as an empty lookup.
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM authenticated_merkle_nodes WHERE namespace=? AND item_key=?",
                               ("synapse-endpoint-members", endpoint))
        with self.assertRaises(ValueError):
            list(index.incident_groups(endpoint))

    def test_changed_and_deleted_synapse_groups_rebase_endpoint_coverage_exactly(self):
        pointer, keys = self._cold_edges(count=24)
        removed = keys[0]
        surviving = keys[1]
        del self.substrate.synapses[removed]
        self.substrate.synapses[surviving]["uses"] = 7
        after = self._write()
        self._commit(after)
        index = self.substrate.assemblies.index._synapse_endpoint_index
        endpoint = "neuron-0000"
        incident = set(index.incident_groups(endpoint))
        expected = set()
        for key, descriptor in self._descriptors(after).items():
            if key[0] != "synapses":
                continue
            rows = json.loads((self.store / descriptor["records"]["path"]).read_bytes())["records"]
            if any(endpoint in (row["source_id"], row["target_id"]) for row in rows):
                expected.add(key[1:])
        self.assertEqual(incident, expected)
        self.assertEqual(len(self.substrate.synapses), len(keys) - 1)

    def test_endpoint_rebase_is_in_same_transaction_as_neuron_journal_cleaning(self):
        before, _keys = self._cold_edges(count=16)
        self.substrate.neurons.edit_by_id("neuron-0001", lambda row: row.update(label="next commit"))
        pointer = self._write()
        self._brain_commit(pointer)
        endpoint_plan = self.substrate._pending_paged_checkpoint[5]
        original = endpoint_plan.rebase_in_connection
        def fail_after_rebase(connection, committed):
            original(connection, committed)
            raise RuntimeError("injected atomic endpoint rebase failure")
        with mock.patch.object(endpoint_plan, "rebase_in_connection", side_effect=fail_after_rebase):
            with self.assertRaisesRegex(RuntimeError, "atomic endpoint"):
                commit_paged_substrate_generation(self.substrate, self.store, self.engine / "brain.json")
        self.assertEqual(journal_status(self.path, before)["dirtyIds"], 1)
        with endpoint_plan.session.transaction() as connection:
            self.assertEqual(endpoint_plan.index._meta(connection)["generation"], before["activeGeneration"])
        commit_paged_substrate_generation(self.substrate, self.store, self.engine / "brain.json")
        self.assertEqual(journal_status(self.path, pointer)["dirtyIds"], 0)

    def test_cold_rebuild_primes_endpoint_and_file_proofs_for_next_small_save(self):
        committed, _keys = self._cold_edges()
        rebuilt = rebuild_committed_paged_cache(self.engine, self.engine / "fresh-cache")
        index = PagedAssemblyIndex(rebuilt.path, dimensions=16, seed=19)
        rebuilt.adopt_authenticated_session(index)
        self.path = rebuilt.path
        self.substrate.assemblies = PagedAssemblyView(index)
        self.substrate.neuron_vectors = index._vectors
        self.substrate.assembly_vectors = PagedAssemblyVectorView(index, index._vectors)
        self.substrate.neurons = PagedNeuronMetadata(rebuilt.path)
        self.substrate.persistence_manifest = dict(committed)
        descriptors = [value for key, value in sorted(self._descriptors(committed).items()) if key[0] == "synapses"]
        forward = json.loads((self.store / "forward-index" / "generations" /
                              (committed["activeGeneration"] + ".json")).read_bytes())
        self.substrate.synapses = LazyPersistedSynapses(
            self.store, descriptors, committed["counts"]["synapses"], 2,
            PagedHotNodeIds(self.substrate.assemblies), forward, store_version=3,
        )
        self._add_neuron("post-recovery-new-hot")
        self.substrate.assemblies.append({
            "id": "post-recovery-new-hot", "fingerprint": "post-recovery-fingerprint", "neuron_ids": ["neuron-0001"],
        })
        with mock.patch.object(self.substrate.synapses, "_group_records", side_effect=AssertionError("full synapse reread")):
            pointer = self._write()
        self.assertTrue(self._commit(pointer)["endpointLookupSelective"])

    def test_disk_backed_authenticated_proofs_do_not_evict_at_old_65536_entry_limit(self):
        session = cache_session(self.substrate.assemblies.index)
        spec = next(iter(self._descriptors(self.first).values()))["records"]
        path = self.store / spec["path"]
        info = path.stat()
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        encoded = canonical(identity)
        with session.transaction(write=True) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO authenticated_blob_proofs VALUES(?,?,?,?)",
                (("/cache-capacity-fixture/%d" % number, encoded, spec["sha256"],
                  session.sign("immutable-file-proof-v1", ["/cache-capacity-fixture/%d" % number,
                                                           list(identity), spec["sha256"]]))
                 for number in range(70000)),
            )
        self.assertEqual(session.lookup_blob(path, identity), spec["sha256"])
        self.assertEqual(session.lookup_blob(Path("/cache-capacity-fixture/0"), identity), spec["sha256"])
        self.assertEqual(session.lookup_blob(Path("/cache-capacity-fixture/69999"), identity), spec["sha256"])

    def test_forged_blob_proof_never_certifies_corrupt_immutable_content(self):
        session = cache_session(self.substrate.assemblies.index)
        spec = next(iter(self._descriptors(self.first).values()))["records"]
        path = (self.store / spec["path"]).resolve()
        raw = path.read_bytes()
        path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
        info = path.stat()
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO authenticated_blob_proofs VALUES(?,?,?,?)",
                (str(path), canonical(identity), spec["sha256"], "0" * 64),
            )
        with session.blob_scope():
            with self.assertRaisesRegex(ValueError, "checksum"):
                _verified_file(path, spec["sha256"], spec["bytes"])

    def test_stale_endpoint_authentication_rebuilds_then_next_append_is_selective(self):
        _pointer, _keys = self._cold_edges()
        with sqlite3.connect(self.path) as connection:
            connection.execute("UPDATE synapse_endpoint_meta SET authentication=?", ("0" * 64,))
        self._add_neuron("after-stale-index")
        self.substrate.assemblies.append({
            "id": "after-stale-index", "fingerprint": "after-stale-index-fingerprint", "neuron_ids": ["neuron-0001"],
        })
        pointer = self._write()
        self.assertFalse(self._commit(pointer)["endpointLookupSelective"])
        self._add_neuron("next-selective-append")
        self.substrate.assemblies.append({
            "id": "next-selective-append", "fingerprint": "next-selective-fingerprint", "neuron_ids": ["neuron-0001"],
        })
        with mock.patch.object(self.substrate.synapses, "_group_records", side_effect=AssertionError("full reread")):
            pointer = self._write()
        self.assertTrue(self._commit(pointer)["endpointLookupSelective"])

    def test_cold_recovery_repairs_incomplete_self_consistent_forward_cache_from_actual_shards(self):
        self._cold_edges(count=24)
        self.substrate.assemblies.append({
            "id": "neuron-0007", "fingerprint": "cold-recovery-hot", "neuron_ids": ["neuron-0001"],
        })
        pointer = self._write()
        self._commit(pointer)
        path = self.store / "forward-index" / "generations" / (pointer["activeGeneration"] + ".json")
        forward = json.loads(path.read_bytes())
        entries = list(iter_entries(self.store, forward))
        self.assertTrue(any(row["hotLocations"] for row in entries))
        for row in entries:
            row["hotLocations"] = []
        forward["shards"] = publish_entries(self.store, entries, None)
        body = {key: value for key, value in forward.items() if key != "contentSha256"}
        forward["contentSha256"] = hashlib.sha256(canonical(body)).hexdigest()
        atomic_write_json(path, forward)
        brain_before = (self.engine / "brain.json").read_bytes()
        rebuilt = rebuild_committed_paged_cache(self.engine, self.engine / "repaired-cache")
        repaired = json.loads(path.read_bytes())
        self.assertTrue(any(row["hotLocations"] for row in iter_entries(self.store, repaired)))
        self.assertEqual(rebuilt.generation_sha256, pointer["activeGeneration"])
        self.assertEqual((self.engine / "brain.json").read_bytes(), brain_before)
        path.unlink()
        rebuild_committed_paged_cache(self.engine, self.engine / "missing-forward-cache")
        self.assertTrue(path.is_file())
        self.assertEqual((self.engine / "brain.json").read_bytes(), brain_before)

    def test_synapse_revision_drift_cannot_clear_a_neuron_or_endpoint_journal(self):
        before, keys = self._cold_edges(count=8)
        self.substrate.neurons.edit_by_id("neuron-0001", lambda row: row.update(label="written snapshot"))
        pointer = self._write()
        self._brain_commit(pointer)
        self.substrate.synapses[keys[0]]["uses"] = 999
        with self.assertRaisesRegex(ValueError, "not committed|changed"):
            commit_paged_substrate_generation(self.substrate, self.store, self.engine / "brain.json")
        self.assertEqual(journal_status(self.path, before)["dirtyIds"], 1)

    def test_existing_generation_gc_removes_only_proofs_for_already_removed_blobs(self):
        old_paths = {(self.store / value["records"]["path"]).resolve()
                     for value in self._descriptors(self.first).values()}
        self.substrate.neurons.edit_by_id("neuron-0001", lambda row: row.update(label="middle"))
        middle = self._write()
        self._commit(middle)
        self.substrate.neurons.edit_by_id("neuron-0001", lambda row: row.update(label="current"))
        current = self._write()
        self._commit(current)
        removed = []
        def forget(path):
            removed.append(path.resolve())
            forget_paged_blob_proof(self.substrate, path)
        result = self.substrate.prune_unreferenced(self.store, [current, middle], on_blob_removed=forget)
        self.assertTrue(result["completed"])
        self.assertTrue(old_paths.intersection(removed))
        with sqlite3.connect(self.path) as connection:
            for path in removed:
                self.assertIsNone(connection.execute(
                    "SELECT 1 FROM authenticated_blob_proofs WHERE path=?", (str(path),),
                ).fetchone())
        self.assertTrue((self.store / current["generationManifest"]).is_file())
        self.assertTrue((self.store / middle["generationManifest"]).is_file())

    def test_optional_proof_reserve_refusal_falls_back_to_real_hashes(self):
        session = cache_session(self.substrate.assemblies.index)
        spec = next(iter(self._descriptors(self.first).values()))["records"]
        path = (self.store / spec["path"]).resolve()
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM authenticated_blob_proofs WHERE path=?", (str(path),))
        previous_reserve = session.disk_reserve
        session.disk_reserve = lambda _size, _operation: False
        try:
            original = NeuralSubstrate._file_sha256
            with session.blob_scope(), mock.patch.object(NeuralSubstrate, "_file_sha256", wraps=original) as hashing:
                _verified_file(path, spec["sha256"], spec["bytes"])
                _verified_file(path, spec["sha256"], spec["bytes"])
            self.assertEqual(hashing.call_count, 2)
            self.assertGreater(session.proof_write_pauses, 0)
        finally:
            session.disk_reserve = previous_reserve

    def test_ephemeral_file_authentication_is_not_trusted_by_a_new_process_session(self):
        from omni_core.authenticated_paged_cache import AuthenticatedCacheSession
        owner = self.substrate.assemblies.index
        old = cache_session(owner)
        spec = next(iter(self._descriptors(self.first).values()))["records"]
        path = (self.store / spec["path"]).resolve()
        info = path.stat()
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        self.assertEqual(old.lookup_blob(path, identity), spec["sha256"])
        fresh = AuthenticatedCacheSession(owner)
        self.assertIsNone(fresh.lookup_blob(path, identity))
        with fresh.blob_scope():
            _verified_file(path, spec["sha256"], spec["bytes"])
        self.assertEqual(fresh.lookup_blob(path, identity), spec["sha256"])

    def test_indexed_inspection_pages_do_not_reopen_cold_shards_after_initial_build(self):
        from omni_core.substrate_inspection import PersistedSubstrateView, query_persisted_substrate
        with mock.patch("omni_core.substrate_inspection.INSPECTION_DISK_RESERVE_BYTES", 0):
            first = query_persisted_substrate(self.engine, "storage-test", {"entity": "neurons", "zoom": 1, "pageSize": 3})
            self.assertEqual(first["returned"], 3)
            with mock.patch.object(PersistedSubstrateView, "records", side_effect=AssertionError("cold shard reopened")), \
                 mock.patch.object(PersistedSubstrateView, "tensors", side_effect=AssertionError("cold tensor reopened")):
                second = query_persisted_substrate(self.engine, "storage-test", {
                    "entity": "neurons", "zoom": 1, "pageSize": 3, "cursor": first["nextCursor"],
                })
            self.assertEqual(second["returned"], 3)
            self.assertFalse({row["id"] for row in first["neurons"]}.intersection(row["id"] for row in second["neurons"]))

    def test_inspection_uses_real_assembly_firing_not_confidence_and_ignores_ahead_pointer(self):
        from omni_core.substrate_inspection import query_persisted_substrate
        self.substrate.neurons.edit_by_id("assembly-0000", lambda row: row.update(activation=0.2))
        with self.substrate.assemblies.transaction(max_rows=1) as edit:
            edit.edit_by_id("assembly-0000", lambda row: row.update(confidence=0.98))
        pointer = self._write()
        self._commit(pointer)
        self.substrate.neurons.edit_by_id("assembly-0000", lambda row: row.update(activation=0.9))
        self._write()  # convenience pointer ahead; brain.json still selects 0.2.
        with mock.patch("omni_core.substrate_inspection.INSPECTION_DISK_RESERVE_BYTES", 0):
            page = query_persisted_substrate(self.engine, "storage-test", {
                "entity": "assemblies", "zoom": 1, "search": "assembly-0000", "pageSize": 1,
            })
        self.assertEqual(page["assemblies"][0]["confidence"], 0.98)
        self.assertAlmostEqual(page["assemblies"][0]["activation"], 0.2)
        self.assertTrue(page["assemblies"][0]["activationObserved"])

    def test_self_consistent_forged_inspection_cache_rebuilds_from_canonical_rows(self):
        from omni_core.substrate_inspection import query_persisted_substrate
        with mock.patch("omni_core.substrate_inspection.INSPECTION_DISK_RESERVE_BYTES", 0):
            query_persisted_substrate(self.engine, "storage-test", {"entity": "neurons", "zoom": 1, "pageSize": 1})
            directory = self.store / "inspection" / "query-generations" / self.first["activeGeneration"]
            database = directory / "records.sqlite3"
            manifest = directory / "manifest.json"
            with sqlite3.connect(database) as connection:
                row = json.loads(connection.execute(
                    "SELECT payload FROM records WHERE kind='neurons' AND id='neuron-0000'"
                ).fetchone()[0])
                row["activation"] = 0.01
                row["label"] = "forged cache label"
                connection.execute("UPDATE records SET payload=? WHERE kind='neurons' AND id='neuron-0000'", (canonical(row),))
            metadata = json.loads(manifest.read_bytes())
            metadata["sha256"] = NeuralSubstrate._file_sha256(database)
            metadata["bytes"] = database.stat().st_size
            body = {key: value for key, value in metadata.items() if key not in {"authentication", "contentSha256"}}
            metadata["contentSha256"] = hashlib.sha256(canonical(body)).hexdigest()
            atomic_write_json(manifest, metadata)
            page = query_persisted_substrate(self.engine, "storage-test", {
                "entity": "neurons", "zoom": 1, "search": "neuron-0000", "pageSize": 1,
            })
        self.assertEqual(page["neurons"][0]["label"], "neuron-0000")
        self.assertEqual(page["neurons"][0]["activation"], 1.0)

    def test_new_inspection_process_session_rebuilds_instead_of_trusting_old_cache_origin(self):
        from omni_core import paged_inspection_queries
        from omni_core.substrate_inspection import PersistedSubstrateView, query_persisted_substrate
        with mock.patch("omni_core.substrate_inspection.INSPECTION_DISK_RESERVE_BYTES", 0):
            query_persisted_substrate(self.engine, "storage-test", {"entity": "neurons", "zoom": 1, "pageSize": 1})
            paged_inspection_queries._OWNERS.pop(str(self.engine.resolve()))
            original = PersistedSubstrateView.records
            checked = []
            def records(view, shard):
                checked.append(shard["kind"])
                return original(view, shard)
            with mock.patch.object(PersistedSubstrateView, "records", records):
                query_persisted_substrate(self.engine, "storage-test", {"entity": "neurons", "zoom": 1, "pageSize": 1})
        self.assertTrue(checked)

    def test_production_paged_recall_reuses_disk_graph_and_returns_all_assemblies(self):
        for number in range(24):
            self.substrate.neuron_vectors.set_packed("assembly-%04d" % number, b"\xaa" * 4)
        cue = torch.ones(16)
        source = self.substrate.synapses
        with mock.patch.object(source, "iter_effective_edge_records", wraps=source.iter_effective_edge_records) as scanned:
            first_signal, first = self.substrate.recall_vector(cue, workspace_slots=1, record_activity=False)
            second_signal, second = self.substrate.recall_vector(cue, workspace_slots=1, record_activity=False)
        self.assertEqual(scanned.call_count, 1)
        self.assertEqual(len(first), 24)
        self.assertEqual(len(second), 24)
        self.assertTrue(torch.equal(first_signal, second_signal))
        self.assertEqual(self.substrate._last_recall_audit["activationStorage"], "paged-recurrent-frontier")

    def test_forward_group_bodies_are_not_reopened_for_small_neuron_only_save_or_stats(self):
        self._cold_edges(count=24)
        self.assertIsInstance(self.substrate.synapses._forward_by_shard, BlobForwardTopology)
        self.substrate.neurons.edit_by_id("neuron-0000", lambda row: row.update(label="small change"))
        with mock.patch("omni_core.paged_forward_index.group_payload", side_effect=AssertionError("untouched forward body reopened")):
            pointer = self._write()
            self._commit(pointer)
            status = self.substrate.synapses.paging_status()
        self.assertEqual(status["forwardTopologyResidency"], "paged-immutable-groups")

    def test_sparse_streamed_packing_preserves_all_zero_edges_exact_order_and_hashes(self):
        from omni_core.streamed_ternary import source_chunks, source_hashes
        from omni_core.ternary_packing import encode_ternary_2bit, _tensor_sha256
        self._cold_edges(count=31)
        source = self.substrate.synapses
        identifier = next(iter(source))
        source[identifier]["effective_weight"] = -1
        expected, count, order, basis = source.dynamic_pack_state()  # Small fixture reference only.
        streamed, streamed_count, streamed_order, streamed_basis = source.stream_dynamic_pack_state()
        self.assertEqual((streamed_count, streamed_order, streamed_basis), (count, order, basis))
        self.assertEqual(b"".join(source_chunks(streamed)), encode_ternary_2bit(expected))
        self.assertEqual(source_hashes(streamed)[1], _tensor_sha256(expected))
        source[identifier]["effective_weight"] = 1
        with self.assertRaisesRegex(ValueError, "changed"):
            list(source_chunks(streamed))


if __name__ == "__main__":
    unittest.main()
