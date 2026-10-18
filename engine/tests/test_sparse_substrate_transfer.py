"""Constructor-free packed/SQLite contracts. No brain/model/decoder/training."""
import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omni_core.sparse_substrate_transfer import (
    CONTROL, CommittedSparseSnapshot, SparseOverlayStale, _decode, _guarded_json, _open, _row,
    apply_sparse_overlay, canonical, export_sparse_bundle, inspect_sparse_snapshot,
    prepare_sparse_overlay, prepare_sparse_rollback, sha, sparse_bundle_descriptor, validate_sparse_bundle,
)
from omni_core.weighted_vsa_row import CONTACT, make_payload

COMPATIBILITY = {"dimensions": 4, "seed": 41, "zeroDeadbandHex": float(0.0).hex()}
PAYLOAD = make_payload(bytes([0x64]), bytes([5, 16, 0, 3]), CONTACT.pack(2, 0x56, 0x23), 4)
OTHER = make_payload(bytes([0x66]), bytes([1, 2, 3, 4]), b"", 4)


class Snapshot:
    def __init__(self, rows, brain_id="source"):
        self.data = rows; self.brain_id = brain_id; self.generation = "a" * 64
        self.metadata_sha = "b" * 64; self.compatibility = dict(COMPATIBILITY)
    def rows(self):
        for kind in ("neurons", "assemblies", "synapses"):
            for row in self.data:
                if row[0] == kind:
                    yield row


class RawVectors:
    dimensions = 4; seed = 41; zero_deadband = 0.0
    def __init__(self):
        self.data = {}
    def row_payload(self, identifier):
        return self.data[identifier][0]
    def update_count(self, identifier):
        return self.data[identifier][1]
    def __contains__(self, identifier):
        return identifier in self.data
    def __delitem__(self, identifier):
        del self.data[identifier]
    def append_row_payload(self, identifier, payload, updates):
        self.data[identifier] = (payload, updates)
    def __getitem__(self, identifier):
        raise AssertionError("no decoded vector access is permitted")


def rows():
    result = []
    for identifier in ("a", "b", "memory", "outside"):
        result.append(("neurons", *_row("neurons", {"id": identifier, "label": identifier, "region": "working"}, payload=PAYLOAD, updates=17)))
    result.append(("assemblies", *_row("assemblies", {"id": "memory", "fingerprint": "source-memory", "neuron_ids": ["a", "b"], "child_assembly_ids": [], "source_label": "fixture native source"})))
    for identifier, target in (("a>b:associates", "b"), ("a>outside:associates", "outside")):
        result.append(("synapses", *_row("synapses", {"id": identifier, "source_id": "a", "target_id": target, "kind": "associates", "parallel_count": 2},
            controls=CONTROL.pack(-1, 16, 0.25, 0.75, 4.5, 29, 1234.5))))
    return result


def fake_owner(root, native_rows):
    memory = SimpleNamespace(neurons={}, neuron_vectors=RawVectors(), assemblies=[], synapses={}, state_revision=0,
        persistence_manifest={"activeGeneration": "d" * 64}, invalidate_assembly_index=lambda: None, release_paged_recall_scratch=lambda: None)
    from omni_core.sparse_substrate_transfer import _install
    for _kind, _identifier, body, _checksum in native_rows:
        _install(memory, _decode(body, COMPATIBILITY))
    brain = SimpleNamespace(brain_id="target", memory=memory, engine_path=root, sparse_transfer_head=None, _connected_regions=None, _connected_recovery_required=False)
    commits = []
    def publication(head):
        commits.append(copy.deepcopy(head)); brain.sparse_transfer_head = head
        # Explicitly synthetic atomic metadata receipt, not a native save proof.
        (root / "brain.json").write_text(canonical({"brain_id": "target", "sparse_transfer_head": head, "substrate": {"persistence": {"activeGeneration": "e" * 64}}}), encoding="utf-8")
        memory.persistence_manifest["activeGeneration"] = "e" * 64
    return brain, publication, commits


class SparseTransferContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-sparse-contract-")
        self.root = Path(self.temporary.name)
    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_weighted_bytes_counter_and_closed_dependency_export(self):
        path = self.root / "bundle.sqlite3"
        receipt = export_sparse_bundle(Snapshot(rows()), {"assemblies": ["memory"]}, path)
        manifest = validate_sparse_bundle(path, receipt["sha256"])
        self.assertEqual(manifest["rows"], 5)
        self.assertEqual(manifest["excludedBoundaryContacts"], 1)
        connection = _open(path)
        try:
            packed = connection.execute("SELECT body FROM rows WHERE kind='neurons' AND id='a'").fetchone()[0]
            decoded = _decode(packed, COMPATIBILITY)
            self.assertEqual(decoded[3:5], (PAYLOAD, 17))
            self.assertNotIn("outside", [row[0] for row in connection.execute("SELECT id FROM rows")])
        finally:
            connection.close()
        self.assertEqual(sparse_bundle_descriptor(path, receipt["sha256"])["manifestJson"], receipt["manifestJson"])

    def test_inspection_pages_bind_generation_without_decoded_activity(self):
        snapshot = Snapshot(rows())
        first = inspect_sparse_snapshot(snapshot, {"entity": "neurons", "maxBytes": 1024})
        self.assertTrue(first["hasMore"])
        second = inspect_sparse_snapshot(snapshot, {"entity": "neurons", "maxBytes": 1024, "cursor": first["nextCursor"]})
        self.assertNotEqual(first["rows"][0]["id"], second["rows"][0]["id"])
        snapshot.generation = "c" * 64
        with self.assertRaisesRegex(ValueError, "stale"):
            inspect_sparse_snapshot(snapshot, {"cursor": first["nextCursor"], "entity": "neurons"})

    def test_replace_exact_changes_real_bytes_and_duplicate_does_not_repeat(self):
        bundle = self.root / "bundle.sqlite3"; plan = self.root / "plan.sqlite3"
        source_rows = rows(); target_rows = copy.deepcopy(source_rows)
        target_rows[0] = ("neurons", *_row("neurons", {"id": "a", "label": "a", "region": "working"}, payload=OTHER, updates=2))
        receipt = export_sparse_bundle(Snapshot(source_rows), {"all": True}, bundle)
        preview = prepare_sparse_overlay(Snapshot(target_rows, "target"), bundle, receipt["sha256"], plan, "replace-exact")
        self.assertEqual(preview["replacements"], 1)
        brain, publish, commits = fake_owner(self.root, target_rows)
        params = {"jobId": "one", "requestSha256": "c" * 64, "planSha256": preview["sha256"]}
        result = apply_sparse_overlay(brain, plan, params, publication=publish)
        self.assertEqual(brain.memory.neuron_vectors.data["a"], (PAYLOAD, 17))
        self.assertEqual(result["appliedRows"], 1)
        duplicate = apply_sparse_overlay(brain, plan, params, publication=publish)
        self.assertTrue(duplicate["duplicate"]); self.assertEqual(len(commits), 1)
        self.assertFalse(result["cortexParametersChanged"])

    def test_append_conflict_and_changed_selected_preimage_never_overwrite(self):
        bundle = self.root / "bundle.sqlite3"; plan = self.root / "plan.sqlite3"
        source_rows = rows(); exported = export_sparse_bundle(Snapshot(source_rows), {"all": True}, bundle)
        preview = prepare_sparse_overlay(Snapshot(source_rows, "target"), bundle, exported["sha256"], plan, "replace-exact")
        brain, publish, commits = fake_owner(self.root, source_rows)
        brain.memory.neuron_vectors.data["a"] = (OTHER, 22)
        with self.assertRaises(SparseOverlayStale):
            apply_sparse_overlay(brain, plan, {"jobId": "stale", "requestSha256": "c" * 64, "planSha256": preview["sha256"]}, publication=publish)
        self.assertEqual(len(commits), 0); self.assertEqual(brain.memory.neuron_vectors.data["a"], (OTHER, 22))
        append = prepare_sparse_overlay(Snapshot([("neurons", *_row("neurons", {"id": "a", "label": "a", "region": "working"}, payload=OTHER, updates=22))], "target"), bundle, exported["sha256"], self.root / "append.sqlite3", "append")
        self.assertEqual(append["conflicts"], 1)

    def test_resource_refusal_precedes_metadata_decoding_and_cancel_removes_spool(self):
        metadata = self.root / "brain.json"; metadata.write_text("not JSON", encoding="utf-8")
        def refused(_size, _label):
            raise MemoryError("explicit resource refusal before parse")
        with self.assertRaisesRegex(MemoryError, "before parse"):
            _guarded_json(metadata, reserve=refused)
        path = self.root / "cancel.sqlite3"
        with self.assertRaises(InterruptedError):
            export_sparse_bundle(Snapshot(rows()), {"all": True}, path, cancelled=lambda: True)
        self.assertFalse(path.exists()); self.assertFalse(path.with_name(path.name + ".selection").exists())

    def test_external_schema_and_missing_closure_are_rejected(self):
        path = self.root / "bundle.sqlite3"; exported = export_sparse_bundle(Snapshot(rows()), {"all": True}, path)
        import sqlite3
        connection = sqlite3.connect(path)
        connection.execute("CREATE VIEW malicious AS SELECT 1"); connection.commit(); connection.close()
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_sparse_bundle(path, sha(path.read_bytes()))

    def test_reviewed_inverse_restores_replacement_bytes_and_refuses_unproven_removals(self):
        bundle = self.root / "bundle.sqlite3"; plan = self.root / "plan.sqlite3"
        source_rows = rows(); target_rows = copy.deepcopy(source_rows)
        target_rows[0] = ("neurons", *_row("neurons", {"id": "a", "label": "a", "region": "working"}, payload=OTHER, updates=2))
        exported = export_sparse_bundle(Snapshot(source_rows), {"all": True}, bundle)
        preview = prepare_sparse_overlay(Snapshot(target_rows, "target"), bundle, exported["sha256"], plan, "replace-exact")
        brain, publish, _commits = fake_owner(self.root, target_rows)
        applied = apply_sparse_overlay(brain, plan, {"jobId": "original", "requestSha256": "c" * 64, "planSha256": preview["sha256"]}, publication=publish)
        inverse_path = self.root / "inverse.sqlite3"
        inverse = prepare_sparse_rollback(Snapshot(source_rows, "target"), plan, preview["sha256"], applied["receiptJson"], applied["receiptSha256"], inverse_path)
        self.assertEqual(inverse["conflicts"], 0)
        apply_sparse_overlay(brain, inverse_path, {"jobId": "inverse", "requestSha256": "d" * 64, "planSha256": inverse["sha256"],
            "acknowledgedReceiptSha256": applied["receiptSha256"]}, publication=publish)
        self.assertEqual(brain.memory.neuron_vectors.data["a"], (OTHER, 2))
        appended_plan = self.root / "append.sqlite3"
        append = prepare_sparse_overlay(Snapshot([], "target"), bundle, exported["sha256"], appended_plan, "append")
        empty, publish_empty, _ = fake_owner(self.root, [])
        appended = apply_sparse_overlay(empty, appended_plan, {"jobId": "append", "requestSha256": "e" * 64, "planSha256": append["sha256"]}, publication=publish_empty)
        unsafe = prepare_sparse_rollback(Snapshot(source_rows, "target"), appended_plan, append["sha256"], appended["receiptJson"], appended["receiptSha256"], self.root / "unsafe-inverse.sqlite3")
        self.assertGreater(unsafe["conflicts"], 0); self.assertEqual(unsafe["unprovenRemovalRows"], 7)

    def test_native_metadata_and_fingerprint_collisions_are_rejected_before_mutation(self):
        with self.assertRaisesRegex(ValueError, "non-structural"):
            _row("assemblies", {"id": "bad", "fingerprint": "bad", "answer_key": {"cue": "answer"}})
        source_rows = rows(); bundle = self.root / "bundle.sqlite3"; plan = self.root / "plan.sqlite3"
        exported = export_sparse_bundle(Snapshot(source_rows), {"all": True}, bundle)
        foreign = ("assemblies", *_row("assemblies", {"id": "foreign-owner", "fingerprint": "source-memory", "neuron_ids": []}))
        collision = prepare_sparse_overlay(Snapshot([foreign], "target"), bundle, exported["sha256"], self.root / "conflict.sqlite3", "replace-exact")
        self.assertEqual(collision["fingerprintConflicts"], 1)
        preview = prepare_sparse_overlay(Snapshot([], "target"), bundle, exported["sha256"], plan, "append")
        brain, publication, commits = fake_owner(self.root, [foreign])
        with self.assertRaisesRegex(SparseOverlayStale, "fingerprint"):
            apply_sparse_overlay(brain, plan, {"jobId": "raced", "requestSha256": "c" * 64, "planSha256": preview["sha256"]}, publication=publication)
        self.assertEqual(commits, []); self.assertEqual(brain.memory.neuron_vectors.data, {})


if __name__ == "__main__":
    unittest.main()
