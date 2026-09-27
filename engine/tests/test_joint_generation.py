"""Filesystem-only joint checkpoint tests; no brain or model is built."""

import hashlib
import json
import os
import sqlite3
import struct
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest import mock

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.joint_generation import (
    JointGenerationError,
    VerifiedArtifactCache,
    orphan_stage_paths,
    recover_joint_generation,
    scrub_joint_generation,
    stage_joint_generation,
)


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_safetensors(path: Path, payload: bytes) -> None:
    header = {
        "value": {
            "dtype": "U8", "shape": [len(payload)],
            "data_offsets": [0, len(payload)],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class JointGenerationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-joint-generation-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = self.root / "joint"
        self.neural_store = self.root / "state"
        (self.neural_store / "blobs").mkdir(parents=True)
        neural_roles = {}
        self.neural = {}
        for role in ("core", "plasticity", "optimizer"):
            path = self.root / (role + ".safetensors")
            _write_safetensors(path, role.encode("ascii"))
            digest = _hash(path.read_bytes())
            blob = self.neural_store / "blobs" / (digest + ".safetensors")
            blob.write_bytes(path.read_bytes())
            neural_roles[role] = {
                "path": "blobs/%s.safetensors" % digest,
                "sha256": digest, "bytes": blob.stat().st_size,
                "tensorCount": 1,
            }
            self.neural[role] = blob
        neural_body = {
            "format": "omni-mutable-state", "formatVersion": 1,
            "brainId": "pure-fixture", "roles": neural_roles,
        }
        neural_id = _hash(_canonical(neural_body))
        neural_manifest = {**neural_body, "contentSha256": neural_id}
        neural_relative = "generations/%s/manifest.json" % neural_id
        neural_path = self.neural_store / neural_relative
        neural_path.parent.mkdir(parents=True)
        neural_path.write_bytes(_canonical(neural_manifest))
        self.neural_pointer = {
            "format": "omni-mutable-state", "formatVersion": 1,
            "activeGeneration": neural_id, "contentSha256": neural_id,
            "generationManifest": neural_relative,
            "generationManifestSha256": _hash(neural_path.read_bytes()),
            "replayCount": 0,
        }
        self.substrate_store = self.root / "substrate"
        (self.substrate_store / "blobs").mkdir(parents=True)
        records = _canonical({
            "kind": "assemblies", "ids": ["a1", "a2"],
            "records": [{"id": "a1"}, {"id": "a2"}],
            "vectorIds": ["a1", "a2"],
            "vectorStorage": "shared-neuron-packed",
        })
        records_hash = _hash(records)
        records_path = self.substrate_store / "blobs" / (records_hash + ".json")
        records_path.write_bytes(records)
        substrate_counts = {"neurons": 0, "assemblies": 2, "synapses": 0}
        substrate_body = {
            "format": "omni-substrate-shards", "formatVersion": 3,
            "schema": "pure-fixture", "dimensions": 4, "seed": 1,
            "counts": substrate_counts,
            "shards": [{
                "kind": "assemblies", "bucket": "a", "part": 0,
                "count": 2,
                "records": {
                    "path": "blobs/%s.json" % records_hash,
                    "sha256": records_hash, "bytes": len(records),
                },
                "tensors": None,
            }],
        }
        substrate_id = _hash(_canonical(substrate_body))
        substrate_manifest = {**substrate_body, "contentSha256": substrate_id}
        substrate_relative = "generations/%s/manifest.json" % substrate_id
        substrate_path = self.substrate_store / substrate_relative
        substrate_path.parent.mkdir(parents=True)
        substrate_path.write_bytes(_canonical(substrate_manifest))
        self.substrate_pointer = {
            "format": "omni-substrate-shards", "formatVersion": 3,
            "activeGeneration": substrate_id, "contentSha256": substrate_id,
            "generationManifest": substrate_relative,
            "generationManifestSha256": _hash(substrate_path.read_bytes()),
            "counts": substrate_counts, "shardCount": 1,
        }
        self.substrate_records = records_path
        self.db_path = self.root / "live.sqlite3"
        self.db = sqlite3.connect(self.db_path)
        self.addCleanup(self.db.close)
        self.assertEqual(
            self.db.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal",
        )
        self.db.execute(
            "CREATE TABLE assembly_records (id INTEGER PRIMARY KEY, packed BLOB)"
        )
        self.db.commit()
        self.db.execute(
            "INSERT INTO assembly_records (id, packed) VALUES (1, ?)",
            (b"\x55\x66",),
        )
        self.db.execute(
            "INSERT INTO assembly_records (id, packed) VALUES (2, ?)",
            (b"\x11\x22",),
        )
        self.db.commit()
        self.source_hash = _hash(b"source manifest")
        self.parser_hash = _hash(b"parser manifest")
        self.content_hash = _hash(b"source file bytes")
        self.cursor = {
            "committedRecords": 2,
            "recordPrefixSha256": _hash(b"committed prefix"),
        }
        self.coverage = {
            "visitedRecords": 3,
            "processedRecords": 2,
            "rejectedRecords": 1,
            "processedBytes": 42,
            "expectedRecords": 3,
            "sourceStreamExhausted": True,
            "sourceContentReverifiedSha256": self.content_hash,
        }

    def _backup(self, destination: Path) -> None:
        with closing(sqlite3.connect(destination)) as target:
            self.db.backup(target)

    _DEFAULT_BACKUP = object()

    def _stage(self, *, store=None, backup=_DEFAULT_BACKUP, **changes):
        arguments = {
            "neural_store_root": self.neural_store,
            "neural_pointer": self.neural_pointer,
            "substrate_store_root": self.substrate_store,
            "substrate_pointer": self.substrate_pointer,
            "sqlite_backup": self._backup if backup is self._DEFAULT_BACKUP else backup,
            "source_manifest_sha256": self.source_hash,
            "parser_manifest_sha256": self.parser_hash,
            "source_content_sha256": self.content_hash,
            "checkpoint_sequence": 1,
            "cursor": self.cursor,
            "coverage": self.coverage,
        }
        arguments.update(changes)
        return stage_joint_generation(store or self.store, **arguments)

    def _commit(self, reference, *, store=None):
        store = store or self.store
        temporary = store / "brain.json.next"
        with temporary.open("xb") as handle:
            handle.write(_canonical({"jointCheckpoint": reference}))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, store / "brain.json")
        descriptor = os.open(store, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _recover(self, *, store=None, **changes):
        store = store or self.store
        arguments = {
            "neural_store_root": self.neural_store,
            "substrate_store_root": self.substrate_store,
            "source_manifest_sha256": self.source_hash,
            "parser_manifest_sha256": self.parser_hash,
            "source_content_sha256": self.content_hash,
        }
        arguments.update(changes)
        brain_metadata = store / "brain.json"
        reference = (
            json.loads(brain_metadata.read_text(encoding="utf-8"))[
                "jointCheckpoint"
            ]
            if brain_metadata.exists() else None
        )
        return recover_joint_generation(store, reference, **arguments)

    def test_wal_backup_and_neural_files_share_one_validated_pointer(self):
        self.assertTrue(Path(str(self.db_path) + "-wal").exists())
        reference = self._stage()
        self.assertIsNone(self._recover())
        self._commit(reference)
        recovered = self._recover()
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.generation_id, reference["generationId"])
        self.assertEqual(recovered.manifest["cursor"], self.cursor)
        self.assertEqual(recovered.manifest["coverage"], self.coverage)
        self.assertEqual(recovered.manifest["checkpointSequence"], 1)
        with closing(sqlite3.connect(recovered.sqlite_snapshot)) as db:
            self.assertEqual(
                db.execute("SELECT packed FROM assembly_records WHERE id=1").fetchone(),
                (b"\x55\x66",),
            )
        self.assertEqual(
            recovered.manifest["neuralGenerationSha256"],
            self.neural_pointer["activeGeneration"],
        )
        self.assertEqual(
            recovered.manifest["substrateGenerationSha256"],
            self.substrate_pointer["activeGeneration"],
        )
        self.assertEqual(
            recovered.manifest["sqliteSnapshot"]["sha256"],
            _hash(recovered.sqlite_snapshot.read_bytes()),
        )
        pointer = (self.store / "brain.json").read_text(encoding="utf-8")
        manifest = (recovered.directory / "manifest.json").read_text(
            encoding="utf-8",
        )
        self.assertNotIn(str(self.root), pointer + manifest)
        self.assertNotIn("source file bytes", pointer + manifest)
        self.assertNotIn("credential", pointer + manifest)
        self.assertEqual(
            {path.name for path in recovered.directory.iterdir()},
            {"manifest.json", "packed-vector-index.sqlite3"},
        )
        self.assertEqual(recovered.neural_files, self.neural)
        self.assertEqual(orphan_stage_paths(self.store), ())

    def test_without_sqlite_backup_keeps_only_shard_backed_authority(self):
        reference = self._stage(backup=None)
        self._commit(reference)
        recovered = self._recover()
        self.assertIsNone(recovered.sqlite_snapshot)
        self.assertIsNone(recovered.manifest["sqliteSnapshot"])
        self.assertEqual(
            {path.name for path in recovered.directory.iterdir()},
            {"manifest.json"},
        )

    def test_incremental_stage_reuses_only_previously_hashed_unchanged_blobs(self):
        cache = VerifiedArtifactCache()
        first = self._stage(backup=None, verified_cache=cache)
        self._commit(first)
        self.assertEqual(cache.hashed_files, 4)
        self.assertEqual(cache.reused_files, 0)
        second = self._stage(
            backup=None, previous_reference=first, checkpoint_sequence=2,
            verified_cache=cache,
        )
        self.assertEqual(cache.hashed_files, 4)
        self.assertEqual(cache.reused_files, 4)
        self._commit(second)
        recovered = self._recover(verified_cache=cache)
        self.assertEqual(
            recovered.manifest["previousManifestSha256"], first["sha256"],
        )
        # Recovery deliberately ignores cached stat identities and hashes all.
        self.assertEqual(cache.hashed_files, 8)
        self.assertEqual(cache.reused_files, 4)
        scrub_joint_generation(
            self.store, second,
            neural_store_root=self.neural_store,
            substrate_store_root=self.substrate_store,
            source_manifest_sha256=self.source_hash,
            parser_manifest_sha256=self.parser_hash,
            source_content_sha256=self.content_hash,
            verified_cache=cache,
        )
        self.assertEqual(cache.hashed_files, 12)

    def test_incremental_stage_hashes_new_substrate_blob_only(self):
        cache = VerifiedArtifactCache()
        first = self._stage(backup=None, verified_cache=cache)
        self._commit(first)
        old_manifest_path = (
            self.substrate_store / self.substrate_pointer["generationManifest"]
        )
        updated = json.loads(old_manifest_path.read_bytes())
        updated.pop("contentSha256")
        new_records = _canonical({
            "kind": "assemblies", "ids": ["a1", "a2"],
            "records": [{"id": "a1", "revision": 2}, {"id": "a2"}],
            "vectorIds": ["a1", "a2"],
            "vectorStorage": "shared-neuron-packed",
        })
        new_hash = _hash(new_records)
        (self.substrate_store / "blobs" / (new_hash + ".json")).write_bytes(
            new_records,
        )
        updated["shards"][0]["records"] = {
            "path": "blobs/%s.json" % new_hash,
            "sha256": new_hash, "bytes": len(new_records),
        }
        new_id = _hash(_canonical(updated))
        new_manifest_path = (
            self.substrate_store / "generations" / new_id / "manifest.json"
        )
        new_manifest_path.parent.mkdir()
        new_manifest_path.write_bytes(_canonical({
            **updated, "contentSha256": new_id,
        }))
        self.substrate_pointer = {
            **self.substrate_pointer,
            "activeGeneration": new_id,
            "contentSha256": new_id,
            "generationManifest": "generations/%s/manifest.json" % new_id,
            "generationManifestSha256": _hash(new_manifest_path.read_bytes()),
        }
        second = self._stage(
            backup=None, previous_reference=first, checkpoint_sequence=2,
            verified_cache=cache,
        )
        self.assertEqual(cache.hashed_files, 5)
        self.assertEqual(cache.reused_files, 3)
        self._commit(second)
        self.assertEqual(
            self._recover().manifest["substrateGenerationSha256"], new_id,
        )

    def test_orphan_stage_is_never_recovered_or_promoted(self):
        orphan = self.store / "staging" / ("a" * 32 + ".stage")
        orphan.mkdir(parents=True)
        (orphan / "manifest.json").write_text("{}", encoding="utf-8")
        self.assertEqual(orphan_stage_paths(self.store), (orphan,))
        self.assertIsNone(self._recover())
        reference = self._stage()
        self.assertIsNone(self._recover())
        self._commit(reference)
        self.assertEqual(self._recover().generation_id, reference["generationId"])
        self.assertEqual(orphan_stage_paths(self.store), (orphan,))

    def test_backup_failure_preserves_old_pointer_and_reports_orphan_stage(self):
        active = self._stage()
        self._commit(active)
        old_pointer = (self.store / "brain.json").read_bytes()

        def interrupted(_destination):
            raise RuntimeError("backup interrupted")

        with self.assertRaisesRegex(RuntimeError, "backup interrupted"):
            self._stage(backup=interrupted, checkpoint_sequence=2)
        self.assertEqual((self.store / "brain.json").read_bytes(), old_pointer)
        self.assertEqual(self._recover().generation_id, active["generationId"])
        self.assertEqual(len(orphan_stage_paths(self.store)), 1)

    def test_complete_but_uncommitted_generation_cannot_advance_cursor(self):
        active = self._stage()
        self._commit(active)
        old_pointer = (self.store / "brain.json").read_bytes()
        self.db.execute(
            "INSERT INTO assembly_records (id, packed) VALUES (3, ?)",
            (b"\x55",),
        )
        self.db.commit()
        uncommitted = self._stage(checkpoint_sequence=2)
        self.assertEqual((self.store / "brain.json").read_bytes(), old_pointer)
        self.assertEqual(self._recover().generation_id, active["generationId"])
        self.assertNotEqual(uncommitted["generationId"], active["generationId"])
        self.assertEqual(
            len(list((self.store / "generations").iterdir())), 2,
        )

    def test_recovery_rejects_tampered_references_sqlite_manifest_and_pointer(self):
        for damaged in ("core", "substrate", "sqlite", "manifest", "pointer"):
            with self.subTest(damaged=damaged):
                store = self.root / ("store-" + damaged)
                reference = self._stage(store=store)
                self._commit(reference, store=store)
                active = self._recover(store=store)
                path = {
                    "core": active.neural_files["core"],
                    "substrate": self.substrate_records,
                    "sqlite": active.sqlite_snapshot,
                    "manifest": active.directory / "manifest.json",
                    "pointer": store / "brain.json",
                }[damaged]
                original = path.read_bytes()
                try:
                    if damaged == "pointer":
                        metadata = json.loads(original)
                        metadata["jointCheckpoint"]["sha256"] = _hash(b"wrong")
                        path.write_bytes(_canonical(metadata))
                    else:
                        body = bytearray(original)
                        body[-1] ^= 1
                        path.write_bytes(body)
                    with self.assertRaises(JointGenerationError):
                        self._recover(store=store)
                finally:
                    path.write_bytes(original)

    def test_recovery_requires_fresh_matching_source_and_parser_hashes(self):
        self._commit(self._stage())
        for field in (
            "source_manifest_sha256", "parser_manifest_sha256",
            "source_content_sha256",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(JointGenerationError, "observed source"):
                    self._recover(**{field: _hash(field.encode("ascii"))})

    def test_neural_paths_reject_traversal_and_symlinked_blobs(self):
        altered = {
            **self.neural_pointer,
            "generationManifest": "../outside/manifest.json",
        }
        with self.assertRaisesRegex(JointGenerationError, "pointer"):
            self._stage(neural_pointer=altered)
        self.assertFalse(self.store.exists())

        blob = self.neural["core"]
        original = blob.read_bytes()
        target = self.root / "core-backup.safetensors"
        target.write_bytes(original)
        blob.unlink()
        try:
            blob.symlink_to(target)
            with self.assertRaisesRegex(JointGenerationError, "symlink"):
                self._stage()
        finally:
            blob.unlink()
            blob.write_bytes(original)

    def test_joint_store_symlinked_generation_parent_is_rejected(self):
        store = self.root / "symlinked-joint"
        store.mkdir()
        outside = self.root / "outside-generations"
        outside.mkdir()
        (store / "generations").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(JointGenerationError, "symlink"):
            self._stage(store=store)
        self.assertEqual(list(outside.iterdir()), [])

    def test_substrate_manifest_or_blob_loss_fails_closed(self):
        reference = self._stage(backup=None)
        self._commit(reference)
        active = self._recover()
        substrate_manifest = (
            self.substrate_store
            / self.substrate_pointer["generationManifest"]
        )
        original = substrate_manifest.read_bytes()
        try:
            substrate_manifest.write_bytes(original + b" ")
            with self.assertRaisesRegex(JointGenerationError, "substrate manifest"):
                self._recover()
        finally:
            substrate_manifest.write_bytes(original)
        self.substrate_records.unlink()
        with self.assertRaisesRegex(JointGenerationError, "substrate shard blob"):
            self._recover()
        self.assertTrue(active.directory.is_dir())

    def test_invalid_coverage_or_extra_text_fields_fail_before_staging(self):
        invalid = [
            ({**self.cursor, "sourceText": "do not persist"}, self.coverage),
            (self.cursor, {**self.coverage, "credential": "secret"}),
            (self.cursor, {**self.coverage, "visitedRecords": 4}),
            (self.cursor, {**self.coverage,
                           "sourceContentReverifiedSha256": _hash(b"wrong")}),
        ]
        for cursor, coverage in invalid:
            with self.subTest(cursor=cursor, coverage=coverage):
                backup = mock.Mock()
                with self.assertRaises(JointGenerationError):
                    self._stage(backup=backup, cursor=cursor, coverage=coverage)
                backup.assert_not_called()
        self.assertFalse(self.store.exists())

    def test_incomplete_sqlite_backup_is_not_committed(self):
        def invalid_backup(destination):
            destination.write_bytes(b"not a SQLite backup")

        with self.assertRaisesRegex(JointGenerationError, "SQLite snapshot"):
            self._stage(backup=invalid_backup)
        self.assertIsNone(self._recover())
        self.assertEqual(len(orphan_stage_paths(self.store)), 1)

    def test_missing_or_corrupt_brain_reference_never_scans_generations(self):
        reference = self._stage()
        self._commit(reference)
        active = self._recover()
        (self.store / "brain.json").unlink()
        self.assertIsNone(self._recover())
        self.assertTrue(active.directory.is_dir())
        (self.store / "brain.json").write_text(
            '{"jointCheckpoint":{}}', encoding="utf-8",
        )
        with self.assertRaises(JointGenerationError):
            self._recover()


if __name__ == "__main__":
    unittest.main()
