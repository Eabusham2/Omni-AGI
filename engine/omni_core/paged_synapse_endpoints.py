"""Complete, generation-bound structural endpoint-to-synapse-group index.

All edges are indexed, including zero-weight cold edges. Rows contain no
learned weights or answers. Checked record blobs establish the initial cache;
per-group authentication and an anchored Patricia root prove complete endpoint
lookups (including absence). Plans update only changed complete groups and
advance the cache inside the same post-brain-commit journal transaction.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

from .authenticated_paged_cache import AuthenticatedCacheSession, canonical
from .paged_merkle_kv import MerkleRoot, SqliteMerkleMap


_FORMAT = "omni-derived-synapse-endpoint-index"
_VERSION = 1
_MAX_BLOB_BYTES = 64 * 1024 * 1024
_MEMBERS = "synapse-endpoint-members"
_INCIDENT = "synapse-endpoint-plan-incident"
_INCIDENT_VALUE = hashlib.sha256(b"omni-incident-synapse-group-v1").hexdigest()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def descriptor_row(descriptor: Mapping[str, Any]) -> tuple[str, int, str, str, int]:
    bucket, part, count = descriptor.get("bucket"), descriptor.get("part"), descriptor.get("count")
    records, tensors = descriptor.get("records"), descriptor.get("tensors")
    if (
        descriptor.get("kind") != "synapses" or not isinstance(bucket, str)
        or len(bucket) != 1 or bucket not in "0123456789abcdef"
        or type(part) is not int or part < 0 or type(count) is not int or not 1 <= count <= 512
        or not isinstance(records, Mapping) or not isinstance(tensors, Mapping)
        or not _valid_sha(records.get("sha256")) or not _valid_sha(tensors.get("sha256"))
        or records.get("path") != "blobs/%s.json" % records["sha256"]
        or tensors.get("path") != "blobs/%s.safetensors" % tensors["sha256"]
    ):
        raise ValueError("endpoint index source descriptor is invalid")
    return bucket, part, records["sha256"], tensors["sha256"], count


def descriptors_digest(descriptors: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256(b"omni-endpoint-descriptor-list-v1\0")
    previous: Optional[tuple[str, int]] = None
    for descriptor in descriptors:
        row = descriptor_row(descriptor)
        if previous is not None and row[:2] <= previous:
            raise ValueError("endpoint index descriptors are not sorted and unique")
        previous = row[:2]
        digest.update(canonical(row) + b"\n")
    return digest.hexdigest()


class SynapseEndpointIndex:
    """Authenticated working index; committed blobs remain sole authority."""

    def __init__(self, session: AuthenticatedCacheSession) -> None:
        self.session = session
        self.merkle = SqliteMerkleMap(_MEMBERS)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.session.reserve_disk(128 * 1024, "complete endpoint index schema")
        with self.session.transaction(write=True) as connection:
            SqliteMerkleMap.ensure_schema(connection)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_meta ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1),payload BLOB NOT NULL,"
                "authentication TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_groups ("
                "bucket TEXT NOT NULL,part INTEGER NOT NULL,records_sha256 TEXT NOT NULL,"
                "tensors_sha256 TEXT NOT NULL,count INTEGER NOT NULL,endpoints_sha256 TEXT NOT NULL,"
                "endpoint_count INTEGER NOT NULL,authentication TEXT NOT NULL,"
                "PRIMARY KEY(bucket,part)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_members ("
                "endpoint TEXT NOT NULL,bucket TEXT NOT NULL,part INTEGER NOT NULL,"
                "records_sha256 TEXT NOT NULL,tensors_sha256 TEXT NOT NULL,count INTEGER NOT NULL,"
                "PRIMARY KEY(endpoint,bucket,part)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS synapse_endpoint_member_group "
                "ON synapse_endpoint_members(bucket,part,endpoint)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_plan_groups ("
                "bucket TEXT NOT NULL,part INTEGER NOT NULL,records_sha256 TEXT NOT NULL,"
                "tensors_sha256 TEXT NOT NULL,count INTEGER NOT NULL,endpoints_sha256 TEXT NOT NULL,"
                "endpoint_count INTEGER NOT NULL,authentication TEXT NOT NULL,"
                "PRIMARY KEY(bucket,part)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_plan_members ("
                "endpoint TEXT NOT NULL,bucket TEXT NOT NULL,part INTEGER NOT NULL,"
                "PRIMARY KEY(endpoint,bucket,part)) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS synapse_endpoint_plan_member_group "
                "ON synapse_endpoint_plan_members(bucket,part,endpoint)"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_plan_affected ("
                "endpoint TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            connection.execute(
                "CREATE TABLE IF NOT EXISTS synapse_endpoint_plan_meta ("
                "singleton INTEGER PRIMARY KEY CHECK(singleton=1),nonce TEXT NOT NULL)"
            )

    def _store_ids(self, connection: sqlite3.Connection) -> list[str]:
        result = []
        for table in ("index_metadata", "paged_vector_meta", "paged_neuron_meta"):
            row = connection.execute("SELECT value FROM %s WHERE key='store_id'" % table).fetchone()
            if row is None or not isinstance(row[0], str) or not row[0]:
                raise ValueError("endpoint index needs one shared three-store cache")
            result.append(row[0])
        return result

    @staticmethod
    def _group_auth_value(row: tuple) -> list[Any]:
        return list(row[:7])

    def _meta(self, connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute(
            "SELECT payload,authentication FROM synapse_endpoint_meta WHERE singleton=1"
        ).fetchone()
        if row is None or not isinstance(row[0], bytes):
            raise ValueError("endpoint index has no authenticated committed base")
        value = json.loads(row[0])
        if (
            not isinstance(value, dict) or canonical(value) != row[0]
            or value.get("format") != _FORMAT or value.get("version") != _VERSION
            or value.get("state") != "ready"
            or value.get("storeIds") != self._store_ids(connection)
            or not self.session.authentic("endpoint-index-base-v1", value, row[1])
        ):
            raise ValueError("endpoint index base is stale, malformed or unauthenticated")
        return value

    def _root(self, meta: Mapping[str, Any]) -> MerkleRoot:
        root = meta.get("root")
        if (
            not isinstance(root, dict) or set(root) != {"prefix", "sha256", "count"}
            or not isinstance(root["prefix"], str) or not _valid_sha(root["sha256"])
            or type(root["count"]) is not int or root["count"] < 0
        ):
            raise ValueError("endpoint index authenticated root is malformed")
        return MerkleRoot(root["prefix"], root["sha256"], root["count"])

    def _write_meta(
        self, connection: sqlite3.Connection, pointer: Optional[Mapping[str, Any]],
        descriptors: Sequence[Mapping[str, Any]], root: MerkleRoot,
        forward_sha256: Optional[str] = None,
    ) -> dict[str, Any]:
        value = {
            "format": _FORMAT, "version": _VERSION, "state": "ready",
            "generation": pointer["activeGeneration"] if pointer is not None else "",
            "manifestSha256": pointer["generationManifestSha256"] if pointer is not None else "",
            "descriptorSha256": descriptors_digest(descriptors),
            "synapseCount": sum(descriptor["count"] for descriptor in descriptors),
            "storeIds": self._store_ids(connection),
            "verifiedForwardSha256": forward_sha256,
            "root": {"prefix": root.prefix, "sha256": root.sha256, "count": root.count},
        }
        connection.execute(
            "INSERT OR REPLACE INTO synapse_endpoint_meta(singleton,payload,authentication) VALUES(1,?,?)",
            (canonical(value), self.session.sign("endpoint-index-base-v1", value)),
        )
        return value

    def _checked_group(
        self, connection: sqlite3.Connection, key: tuple[str, int],
    ) -> tuple:
        row = connection.execute(
            "SELECT bucket,part,records_sha256,tensors_sha256,count,endpoints_sha256,"
            "endpoint_count,authentication FROM synapse_endpoint_groups WHERE bucket=? AND part=?",
            key,
        ).fetchone()
        if row is None or (
            not _valid_sha(row[2]) or not _valid_sha(row[3]) or not _valid_sha(row[5])
            or type(row[4]) is not int or not 1 <= row[4] <= 512
            or type(row[6]) is not int or not 1 <= row[6] <= 2 * row[4]
            or not self.session.authentic("endpoint-index-group-v1", self._group_auth_value(row), row[7])
        ):
            raise ValueError("endpoint index group is missing or unauthenticated")
        return row

    def validate_base(
        self, pointer: Optional[Mapping[str, Any]], descriptors: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        with self.session.transaction() as connection:
            meta = self._meta(connection)
            if (
                meta["generation"] != (pointer["activeGeneration"] if pointer is not None else "")
                or meta["manifestSha256"] != (pointer["generationManifestSha256"] if pointer is not None else "")
                or meta["descriptorSha256"] != descriptors_digest(descriptors)
                or meta["synapseCount"] != sum(value["count"] for value in descriptors)
            ):
                raise ValueError("endpoint index does not match the verified generation")
            self.merkle.validate_root(connection, self._root(meta))
            for descriptor in descriptors:
                spec = descriptor_row(descriptor)
                if self._checked_group(connection, spec[:2])[:5] != spec:
                    raise ValueError("endpoint index group references stale descriptor bytes")
            return meta

    def _group_endpoints(self, connection: sqlite3.Connection, key: tuple[str, int]) -> list[str]:
        group = self._checked_group(connection, key)
        rows = connection.execute(
            "SELECT endpoint FROM synapse_endpoint_members WHERE bucket=? AND part=? ORDER BY endpoint LIMIT ?",
            (*key, 2 * group[4] + 1),
        ).fetchall()
        endpoints = [value[0] for value in rows]
        if len(endpoints) != group[6] or _sha(canonical(endpoints)) != group[5]:
            raise ValueError("endpoint index complete group coverage is corrupt")
        return endpoints

    def _endpoint_value(self, connection: sqlite3.Connection, endpoint: str) -> tuple[str, int]:
        digest = hashlib.sha256(b"omni-endpoint-complete-groups-v1\0")
        count = 0
        for row in connection.execute(
            "SELECT bucket,part,records_sha256,tensors_sha256,count FROM synapse_endpoint_members "
            "WHERE endpoint=? ORDER BY bucket,part", (endpoint,),
        ):
            if (
                not isinstance(row[0], str) or len(row[0]) != 1 or row[0] not in "0123456789abcdef"
                or type(row[1]) is not int or row[1] < 0
                or not _valid_sha(row[2]) or not _valid_sha(row[3])
                or type(row[4]) is not int or not 1 <= row[4] <= 512
                or self._checked_group(connection, row[:2])[:5] != row
            ):
                raise ValueError("endpoint index member descriptor is invalid")
            digest.update(canonical(row) + b"\n")
            count += 1
        digest.update(b"count\0" + str(count).encode("ascii"))
        return digest.hexdigest(), count

    def incident_groups(self, endpoint: str) -> Iterator[tuple[str, int]]:
        with self.session.transaction() as connection:
            meta = self._meta(connection)
            expected = self.merkle.get(connection, self._root(meta), endpoint)
            observed, count = self._endpoint_value(connection, endpoint)
            if expected is None:
                if count:
                    raise ValueError("endpoint index has unauthenticated extra members")
                return
            if not count or observed != expected:
                raise ValueError("endpoint index complete lookup coverage is corrupt")
            # Two streamed passes, one snapshot: no unbounded incident-key list.
            for row in connection.execute(
                "SELECT bucket,part FROM synapse_endpoint_members WHERE endpoint=? ORDER BY bucket,part",
                (endpoint,),
            ):
                yield row

    @staticmethod
    def checked_endpoints(descriptor: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> list[str]:
        from .vsa import NeuralSubstrate, _synapse_id_matches_endpoints
        row = descriptor_row(descriptor)
        ids = [value.get("id") for value in records]
        if (
            len(records) != row[4] or any(not isinstance(value, str) or not value for value in ids)
            or ids != sorted(set(ids))
        ):
            raise ValueError("endpoint source record IDs are invalid")
        endpoints: set[str] = set()
        for record in records:
            source, target = record.get("source_id"), record.get("target_id")
            if (
                not _synapse_id_matches_endpoints(record["id"], source, target)
                or NeuralSubstrate._bucket("synapses", record["id"]) != row[0]
            ):
                raise ValueError("endpoint source record identity is invalid")
            endpoints.update((source, target))
        return sorted(endpoints)

    def import_checked_group(
        self, connection: sqlite3.Connection, descriptor: Mapping[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        row = descriptor_row(descriptor)
        endpoints = self.checked_endpoints(descriptor, records)
        self.session.reserve_disk(65536 + 8 * sum(len(value) for value in endpoints),
                                  "complete endpoint checked group")
        group = (*row, _sha(canonical(endpoints)), len(endpoints))
        connection.execute(
            "INSERT INTO synapse_endpoint_groups VALUES(?,?,?,?,?,?,?,?)",
            (*group, self.session.sign("endpoint-index-group-v1", list(group))),
        )
        connection.executemany(
            "INSERT INTO synapse_endpoint_members VALUES(?,?,?,?,?,?)",
            ((endpoint, *row) for endpoint in endpoints),
        )

    def begin_rebuild(self) -> None:
        with self.session.transaction(write=True) as connection:
            connection.execute("DELETE FROM synapse_endpoint_meta")
            connection.execute("DELETE FROM synapse_endpoint_groups")
            connection.execute("DELETE FROM synapse_endpoint_members")
            self.merkle.clear(connection)

    def finish_rebuild(
        self, pointer: Optional[Mapping[str, Any]], descriptors: Sequence[Mapping[str, Any]],
        *, forward_sha256: Optional[str] = None,
    ) -> None:
        with self.session.transaction(write=True) as connection:
            root = self.merkle.empty()
            for (endpoint,) in connection.execute(
                "SELECT DISTINCT endpoint FROM synapse_endpoint_members ORDER BY endpoint"
            ):
                self.session.reserve_disk(8192 + 512 * len(endpoint), "endpoint Merkle rebuild path")
                value, _count = self._endpoint_value(connection, endpoint)
                root = self.merkle.put(connection, root, endpoint, value)
            self._write_meta(connection, pointer, descriptors, root, forward_sha256)
        self.validate_base(pointer, descriptors)

    def rebuild(
        self, store: Path, pointer: Optional[Mapping[str, Any]],
        descriptors: Sequence[Mapping[str, Any]],
    ) -> None:
        from .vsa import NeuralSubstrate
        from .paged_substrate_writer import remember_verified_blob
        self.begin_rebuild()
        for descriptor in descriptors:
            descriptor_row(descriptor)
            spec = descriptor["records"]
            size = spec.get("bytes")
            if type(size) is not int or not 0 <= size <= _MAX_BLOB_BYTES:
                raise ValueError("endpoint rebuild source exceeds bounded read window")
            self.session.owner._reserve_memory(3 * size + 4096, "complete endpoint checked shard read")
            path = NeuralSubstrate._safe_store_path(store, spec["path"])
            before = path.stat()
            if path.is_symlink() or before.st_size != size:
                raise ValueError("endpoint rebuild source size/type changed")
            payload = path.read_bytes()
            if len(payload) != size or _sha(payload) != spec["sha256"]:
                raise ValueError("endpoint rebuild source checksum mismatch")
            remember_verified_blob(path, spec["sha256"], before=before)
            value = json.loads(payload)
            records = value.get("records") if isinstance(value, dict) else None
            if (
                not isinstance(value, dict) or value.get("kind") != "synapses"
                or not isinstance(records, list) or value.get("ids") != [row.get("id") for row in records]
            ):
                raise ValueError("endpoint rebuild source structure is invalid")
            with self.session.transaction(write=True) as connection:
                self.import_checked_group(connection, descriptor, records)
        self.finish_rebuild(pointer, descriptors)


class SynapseEndpointPlan:
    """Exact incident planning and atomic changed-group cache rebasing."""

    def __init__(
        self, index: SynapseEndpointIndex, store: Path, pointer: Optional[Mapping[str, Any]],
        descriptors: Sequence[Mapping[str, Any]], *, incremental_membership: bool,
        forward_manifest: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.index = index
        self.session = index.session
        self.pointer = dict(pointer) if pointer is not None else None
        self.descriptors = descriptors
        self.nonce = uuid.uuid4().hex
        self.incident = SqliteMerkleMap(_INCIDENT)
        self.rebuilt = False
        try:
            self.base_meta = index.validate_base(pointer, descriptors)
        except (ValueError, sqlite3.DatabaseError):
            index.rebuild(store, pointer, descriptors)
            self.base_meta = index.validate_base(pointer, descriptors)
            self.rebuilt = True
        forward = dict(forward_manifest) if isinstance(forward_manifest, Mapping) else None
        forward_body = {key: value for key, value in forward.items() if key != "contentSha256"} if forward else None
        self.full_forward_rebuild = not bool(
            forward is not None and pointer is not None
            and forward.get("sourceGeneration") == pointer["activeGeneration"]
            and forward.get("sourceGenerationManifestSha256") == pointer["generationManifestSha256"]
            and _valid_sha(self.base_meta.get("verifiedForwardSha256"))
            and forward.get("contentSha256") == self.base_meta["verifiedForwardSha256"]
            and _sha(canonical(forward_body)) == forward["contentSha256"]
        )
        self.selective = incremental_membership and not self.full_forward_rebuild
        self.incident_root = self.incident.empty()
        self.stage_count = 0
        self.reindexed_groups = 0
        self.prepared_pointer: Optional[dict[str, Any]] = None
        self.stage_digest: Optional[str] = None
        with self.session.transaction(write=True) as connection:
            for table in ("synapse_endpoint_plan_groups", "synapse_endpoint_plan_members",
                          "synapse_endpoint_plan_affected"):
                connection.execute("DELETE FROM %s" % table)
            connection.execute(
                "INSERT OR REPLACE INTO synapse_endpoint_plan_meta(singleton,nonce) VALUES(1,?)", (self.nonce,)
            )
            self.incident.clear(connection)
        if self.selective:
            with self.session.transaction() as connection:
                changes = connection.execute(
                    "SELECT record_id FROM substrate_plan_membership WHERE kind='assemblies' ORDER BY record_id"
                )
                for (endpoint,) in changes:
                    for key in index.incident_groups(endpoint):
                        self.session.reserve_disk(8192, "incident endpoint group proof")
                        with self.session.transaction(write=True) as writer:
                            self.incident_root = self.incident.put(
                                writer, self.incident_root, "%s:%d" % key, _INCIDENT_VALUE,
                            )

    def assert_unchanged(self, connection: sqlite3.Connection) -> None:
        row = connection.execute("SELECT nonce FROM synapse_endpoint_plan_meta WHERE singleton=1").fetchone()
        if row is None or row[0] != self.nonce or self.index._meta(connection) != self.base_meta:
            raise ValueError("endpoint index base or publication plan changed")
        self.incident.validate_root(connection, self.incident_root)

    def requires_reindex(self, key: tuple[str, int]) -> bool:
        if not self.selective:
            return True
        with self.session.transaction() as connection:
            self.assert_unchanged(connection)
            value = self.incident.get(connection, self.incident_root, "%s:%d" % key)
            if value is not None and value != _INCIDENT_VALUE:
                raise ValueError("incident endpoint group proof is corrupt")
            return value is not None

    def stage_group(
        self, descriptor: Mapping[str, Any], group: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> None:
        row = descriptor_row(descriptor)
        endpoints = self.index.checked_endpoints(descriptor, [record for _identifier, record in group])
        self._stage(row, endpoints)

    def stage_deleted(self, key: tuple[str, int]) -> None:
        self._stage((*key, "", "", 0), [])

    def _stage(self, row: tuple, endpoints: list[str]) -> None:
        self.session.reserve_disk(65536 + 8 * sum(len(value) for value in endpoints), "endpoint group delta")
        value = (*row, _sha(canonical(endpoints)), len(endpoints))
        with self.session.transaction(write=True) as connection:
            self.assert_unchanged(connection)
            if connection.execute(
                "SELECT 1 FROM synapse_endpoint_plan_groups WHERE bucket=? AND part=?", row[:2],
            ).fetchone() is not None:
                raise ValueError("endpoint group was staged twice")
            connection.execute(
                "INSERT INTO synapse_endpoint_plan_groups VALUES(?,?,?,?,?,?,?,?)",
                (*value, self.session.sign("endpoint-plan-group-v1", [self.nonce, list(value)])),
            )
            connection.executemany(
                "INSERT INTO synapse_endpoint_plan_members VALUES(?,?,?)",
                ((endpoint, *row[:2]) for endpoint in endpoints),
            )
        self.stage_count += 1

    def _stage_rows(self, connection: sqlite3.Connection) -> Iterator[tuple]:
        observed = 0
        for row in connection.execute(
            "SELECT bucket,part,records_sha256,tensors_sha256,count,endpoints_sha256,"
            "endpoint_count,authentication FROM synapse_endpoint_plan_groups ORDER BY bucket,part"
        ):
            if not self.session.authentic("endpoint-plan-group-v1", [self.nonce, list(row[:7])], row[7]):
                raise ValueError("endpoint group delta is unauthenticated")
            endpoints = [value[0] for value in connection.execute(
                "SELECT endpoint FROM synapse_endpoint_plan_members WHERE bucket=? AND part=? "
                "ORDER BY endpoint LIMIT ?", (*row[:2], 2 * max(1, row[4]) + 1),
            )]
            if len(endpoints) != row[6] or _sha(canonical(endpoints)) != row[5]:
                raise ValueError("endpoint group delta is incomplete")
            observed += 1
            yield row
        if observed != self.stage_count:
            raise ValueError("endpoint group delta coverage changed")

    def prepare_commit(
        self, pointer: Mapping[str, Any], descriptors: Sequence[Mapping[str, Any]],
        forward_manifest: Mapping[str, Any],
    ) -> None:
        body = {key: value for key, value in forward_manifest.items() if key != "contentSha256"}
        if (
            forward_manifest.get("sourceGeneration") != pointer["activeGeneration"]
            or forward_manifest.get("sourceGenerationManifestSha256") != pointer["generationManifestSha256"]
            or _sha(canonical(body)) != forward_manifest.get("contentSha256")
        ):
            raise ValueError("endpoint plan has no matching checked forward generation")
        with self.session.transaction() as connection:
            self.assert_unchanged(connection)
            digest = hashlib.sha256(b"omni-endpoint-plan-v1\0")
            for row in self._stage_rows(connection):
                digest.update(canonical(row) + b"\n")
            for descriptor in descriptors:
                expected = descriptor_row(descriptor)
                staged = connection.execute(
                    "SELECT bucket,part,records_sha256,tensors_sha256,count FROM "
                    "synapse_endpoint_plan_groups WHERE bucket=? AND part=?", expected[:2],
                ).fetchone()
                actual = staged if staged is not None else self.index._checked_group(connection, expected[:2])[:5]
                if actual != expected:
                    raise ValueError("endpoint delta does not cover published synapse descriptors")
            self.stage_digest = digest.hexdigest()
        self.prepared_pointer = dict(pointer)
        self.next_descriptors = descriptors
        self.next_forward_sha256 = forward_manifest["contentSha256"]

    def rebase_in_connection(self, connection: sqlite3.Connection, pointer: Mapping[str, Any]) -> None:
        if self.prepared_pointer != pointer or self.stage_digest is None:
            raise ValueError("endpoint index generation was not prepared")
        self.assert_unchanged(connection)
        digest = hashlib.sha256(b"omni-endpoint-plan-v1\0")
        for row in self._stage_rows(connection):
            digest.update(canonical(row) + b"\n")
        if digest.hexdigest() != self.stage_digest:
            raise ValueError("endpoint index deltas changed before committed rebase")
        root = self.index._root(self.base_meta)
        for row in self._stage_rows(connection):
            key = row[:2]
            prior = connection.execute("SELECT 1 FROM synapse_endpoint_groups WHERE bucket=? AND part=?", key).fetchone()
            if prior is not None:
                for endpoint in self.index._group_endpoints(connection, key):
                    connection.execute("INSERT OR IGNORE INTO synapse_endpoint_plan_affected VALUES(?)", (endpoint,))
            for (endpoint,) in connection.execute(
                "SELECT endpoint FROM synapse_endpoint_plan_members WHERE bucket=? AND part=?", key,
            ):
                connection.execute("INSERT OR IGNORE INTO synapse_endpoint_plan_affected VALUES(?)", (endpoint,))
            connection.execute("DELETE FROM synapse_endpoint_members WHERE bucket=? AND part=?", key)
            connection.execute("DELETE FROM synapse_endpoint_groups WHERE bucket=? AND part=?", key)
            if row[4]:
                connection.execute(
                    "INSERT INTO synapse_endpoint_groups VALUES(?,?,?,?,?,?,?,?)",
                    (*row[:7], self.session.sign("endpoint-index-group-v1", list(row[:7]))),
                )
                connection.execute(
                    "INSERT INTO synapse_endpoint_members(endpoint,bucket,part,records_sha256,tensors_sha256,count) "
                    "SELECT endpoint,bucket,part,?,?,? FROM synapse_endpoint_plan_members WHERE bucket=? AND part=?",
                    (*row[2:5], *key),
                )
        for (endpoint,) in connection.execute("SELECT endpoint FROM synapse_endpoint_plan_affected ORDER BY endpoint"):
            self.session.reserve_disk(8192 + 512 * len(endpoint), "endpoint committed Merkle path")
            value, count = self.index._endpoint_value(connection, endpoint)
            present = self.index.merkle.get(connection, root, endpoint)
            if count:
                root = self.index.merkle.put(connection, root, endpoint, value)
            elif present is not None:
                root = self.index.merkle.delete(connection, root, endpoint)
        for descriptor in self.next_descriptors:
            expected = descriptor_row(descriptor)
            if self.index._checked_group(connection, expected[:2])[:5] != expected:
                raise ValueError("rebased endpoint descriptors are incomplete")
        count = connection.execute("SELECT COALESCE(SUM(count),0) FROM synapse_endpoint_groups").fetchone()[0]
        if count != pointer["counts"]["synapses"]:
            raise ValueError("rebased endpoint index synapse count differs")
        self.index._write_meta(connection, pointer, self.next_descriptors, root, self.next_forward_sha256)
        for table in ("synapse_endpoint_plan_groups", "synapse_endpoint_plan_members",
                      "synapse_endpoint_plan_affected", "synapse_endpoint_plan_meta"):
            connection.execute("DELETE FROM %s" % table)
        self.incident.clear(connection)
