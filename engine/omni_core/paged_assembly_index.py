"""Bounded, durable structural index for neural assembly metadata.

This is an address/index for assembly records, not a second memory or an
answer-retrieval path. Neural activity and generation must continue to use the
substrate's learned vectors and synapses. When configured with dimensions it
references the same live ``PagedPackedVectors`` row used by the neuron view;
no independently writable assembly-vector copy exists. Committed v3 shards
and the brain pointer remain recovery authority. Records are read one at a
time or in keyset pages; no operation reconstructs an in-memory assembly list.

An optional checkpoint can be committed in the same SQLite transaction as an
upsert. That makes *index admission* and its source cursor atomic; callers
must still coordinate this checkpoint with any separate neural-state commit.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Tuple

from .paged_packed_vectors import (
    PagedPackedVectors,
    PagedVectorCacheNeedsRebuild,
    _decoded_row,
)
from .paged_vector_scoring import PackedVectorPage, PackedVectorRow


_FORMAT = "omni-paged-assembly-index"
_VERSION = 3
_CURSOR_FORMAT = "omni-assembly-page-cursor"
_VECTOR_SNAPSHOT_FORMAT = "omni-assembly-vector-snapshot"
_VECTOR_CURSOR_FORMAT = "omni-assembly-vector-page-cursor"
_WRITE_OVERHEAD_BYTES = 64 * 1024
_READ_OVERHEAD_BYTES = 4 * 1024
_MAX_PAGE_RECORDS = 4096  # I/O window, never a neural assembly/cardinality limit.
_ASSEMBLY_METADATA_FIELDS = frozenset({
    "id", "assembly_neuron_id", "fingerprint", "neuron_ids", "concept_ids",
    "child_assembly_ids", "statistical_anchor_ids", "kind", "source",
    "kind_provenance", "source_provenance", "confidence", "importance",
    "rehearsals", "statistical_experiences", "compressed_field",
    "created_at", "last_recalled_at", "source_label",
})
_SAFE_ID = re.compile(r"^[^\s\x00-\x1f\x7f]{1,512}$")
_CREDENTIAL_MARKER = re.compile(
    r"(?i)(?:bearer\s+\S+|(?:api[_-]?key|token|password|secret|authorization)\s*[:=])"
)


def _safe_display_label(value: Any) -> str:
    if (
        not isinstance(value, str)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or _CREDENTIAL_MARKER.search(value) is not None
    ):
        raise ValueError("assembly metadata provenance label is unsafe")
    if len(value) <= 256:
        return value
    # This is only a display hint. Full provenance belongs in a separate
    # source ledger; the neural index retains a stable hash without storing a
    # long raw path, URL, or passage and without dropping the learned record.
    digest = _sha256(value.encode("utf-8"))
    return value[:160] + "…#sha256:" + digest


class AssemblyIndexResourcePause(RuntimeError):
    """A reserve callback refused a bounded read or write."""


@dataclass(frozen=True)
class AssemblyPage:
    records: Tuple[Dict[str, Any], ...]
    cursor: str
    has_more: bool
    through_sequence: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _encode_checked_token(value: Mapping[str, Any]) -> str:
    body = _canonical_json(dict(value))
    encoded = base64.urlsafe_b64encode(body).rstrip(b"=").decode("ascii")
    return encoded + "." + _sha256(body)


def _decode_checked_token(token: str, label: str) -> Dict[str, Any]:
    if not isinstance(token, str) or len(token) > 2048:
        raise ValueError("%s is invalid" % label)
    try:
        encoded, digest = token.split(".", 1)
        padding = "=" * (-len(encoded) % 4)
        body = base64.b64decode(
            encoded + padding, altchars=b"-_", validate=True
        )
        value = json.loads(body)
    except (ValueError, TypeError, UnicodeError, binascii.Error) as error:
        raise ValueError("%s is invalid" % label) from error
    if (
        not isinstance(value, dict)
        or _sha256(body) != digest
        or _canonical_json(value) != body
    ):
        raise ValueError("%s is invalid" % label)
    return value


def _record_payload(record: Mapping[str, Any]) -> Tuple[str, str, bytes, str]:
    if not isinstance(record, Mapping):
        raise ValueError("assembly metadata must be a mapping")
    identifier = record.get("id")
    fingerprint = record.get("fingerprint")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("assembly metadata requires a nonempty string id")
    if not isinstance(fingerprint, str) or not fingerprint.strip():
        raise ValueError("assembly metadata requires a nonempty fingerprint")
    unknown = set(record) - _ASSEMBLY_METADATA_FIELDS
    if unknown:
        raise ValueError("assembly metadata contains a non-structural field")
    projected = dict(record)
    for value in (identifier, fingerprint, record.get("assembly_neuron_id", identifier)):
        if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
            raise ValueError("assembly metadata structural ID is invalid")
    for field in (
        "neuron_ids", "concept_ids", "child_assembly_ids",
        "statistical_anchor_ids",
    ):
        if field not in record:
            continue
        values = record[field]
        if (
            not isinstance(values, list)
            or any(not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None
                   for value in values)
        ):
            raise ValueError("assembly metadata structural IDs are invalid")
    for field in ("kind", "source", "source_label"):
        if field not in record:
            continue
        projected[field] = _safe_display_label(record[field])
    for field in ("kind_provenance", "source_provenance"):
        if field not in record:
            continue
        values = record[field]
        if not isinstance(values, dict):
            raise ValueError("assembly provenance counts are invalid")
        projected_values = {}
        for label, count in values.items():
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("assembly provenance counts are unsafe")
            projected_values[_safe_display_label(label)] = count
        projected[field] = projected_values
    for field in ("confidence", "importance", "created_at", "last_recalled_at"):
        if field not in record or (field == "last_recalled_at" and record[field] is None):
            continue
        value = record[field]
        if (
            isinstance(value, (bool, str, bytes))
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("assembly numeric metadata is invalid")
    for field in ("rehearsals", "statistical_experiences"):
        if field in record and (
            isinstance(record[field], bool)
            or not isinstance(record[field], int)
            or record[field] < 0
        ):
            raise ValueError("assembly count metadata is invalid")
    if "compressed_field" in record and not isinstance(record["compressed_field"], bool):
        raise ValueError("assembly field marker is invalid")
    try:
        payload = _canonical_json(projected)
    except (TypeError, ValueError) as error:
        raise ValueError("assembly metadata must be finite JSON") from error
    return identifier, fingerprint, payload, _sha256(payload)


def _checkpoint_payload(
    checkpoint: Tuple[str, Mapping[str, Any]],
) -> Tuple[str, bytes, str]:
    if not isinstance(checkpoint, tuple) or len(checkpoint) != 2:
        raise ValueError("assembly checkpoint must be a (name, mapping) pair")
    name, value = checkpoint
    if not isinstance(name, str) or not name.strip():
        raise ValueError("assembly checkpoint requires a nonempty name")
    if not isinstance(value, Mapping):
        raise ValueError("assembly checkpoint value must be a mapping")
    try:
        payload = _canonical_json(dict(value))
    except (TypeError, ValueError) as error:
        raise ValueError("assembly checkpoint must be finite JSON") from error
    return name, payload, _sha256(payload)


def _validated_dimensions(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("packed assembly vector dimensions must be positive")
    return value


def _packed_payload(value: bytes, dimensions: int) -> Tuple[bytes, str]:
    """Validate exact 2-bit {-1,0,+1} codes and canonical zero padding."""

    if not isinstance(value, bytes) or len(value) != (dimensions + 3) // 4:
        raise ValueError("packed assembly vector width is invalid")
    for index in range(dimensions):
        if ((value[index // 4] >> ((index % 4) * 2)) & 3) == 3:
            raise ValueError("packed assembly vector contains a reserved code")
    for index in range(dimensions, len(value) * 4):
        if ((value[index // 4] >> ((index % 4) * 2)) & 3) != 1:
            raise ValueError("packed assembly vector padding is invalid")
    return value, _sha256(value)


def _decoded_packed_payload(
    value: Optional[bytes], digest: Optional[str], dimensions: Optional[int]
) -> Optional[bytes]:
    if value is None and digest is None:
        return None
    if dimensions is None or value is None or not isinstance(digest, str):
        raise ValueError("packed assembly vector row is incomplete")
    if _sha256(value) != digest:
        raise ValueError("packed assembly vector checksum mismatch")
    packed, _expected = _packed_payload(value, dimensions)
    return packed


class PagedAssemblyIndex:
    """SQLite-backed assembly metadata with exact indexes and keyset pages.

    ``resource_policy`` may be the existing ``ResourcePolicy`` (its
    ``require_disk`` method is used). ``disk_reserve`` and ``memory_reserve``
    are optional callbacks of ``(estimated_bytes, operation)``. They may
    raise a project-specific resource pause, or return exactly ``False`` to
    raise :class:`AssemblyIndexResourcePause`. Reads and writes never evade a
    configured reserve; the estimates include SQLite and Python overhead.
    """

    def __init__(
        self,
        path: Path,
        *,
        dimensions: Optional[int] = None,
        seed: Optional[int] = None,
        zero_deadband: Optional[float] = None,
        resource_policy: Optional[Any] = None,
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
        memory_reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> None:
        if resource_policy is not None and disk_reserve is not None:
            raise ValueError("provide either resource_policy or disk_reserve")
        requested_dimensions = _validated_dimensions(dimensions)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("paged assembly vector seed must be an integer")
        self.path = Path(path)
        self._disk_reserve = (
            resource_policy.require_disk
            if resource_policy is not None
            else disk_reserve
        )
        self._memory_reserve = memory_reserve
        existed = self.path.is_file()
        if not existed:
            self._reserve_disk(_WRITE_OVERHEAD_BYTES * 2, "assembly index create")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            journal_mode = str(
                connection.execute("PRAGMA journal_mode").fetchone()[0]
            ).lower()
            if journal_mode != "wal":
                self._reserve_disk(_WRITE_OVERHEAD_BYTES * 2, "assembly index WAL setup")
                connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                version = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if version not in (0, 1, 2, _VERSION):
                    raise ValueError("unsupported paged assembly index version")
                if version == 0:
                    if existed:
                        self._reserve_disk(
                            _WRITE_OVERHEAD_BYTES * 2, "assembly index create"
                        )
                    prior_tables = {
                        name for (name,) in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table' "
                            "AND name NOT LIKE 'sqlite_%'"
                        )
                    }
                    if existed and prior_tables:
                        raise ValueError("unversioned paged assembly index is not safe to open")
                    connection.execute(
                        """
                        CREATE TABLE assembly_records (
                            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                            assembly_id TEXT NOT NULL UNIQUE,
                            fingerprint TEXT NOT NULL UNIQUE,
                            record_json BLOB NOT NULL,
                            record_sha256 TEXT NOT NULL,
                            updated_at REAL NOT NULL
                        )
                        """
                    )
                    connection.execute(
                        """
                        CREATE TABLE index_metadata (
                            key TEXT PRIMARY KEY,
                            value TEXT NOT NULL
                        ) WITHOUT ROWID
                        """
                    )
                    connection.execute(
                        """
                        CREATE TABLE progress_checkpoints (
                            name TEXT PRIMARY KEY,
                            payload_json BLOB NOT NULL,
                            payload_sha256 TEXT NOT NULL,
                            updated_at REAL NOT NULL
                        ) WITHOUT ROWID
                        """
                    )
                else:
                    tables = {
                        name for (name,) in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        )
                    }
                    if not {"assembly_records", "index_metadata", "progress_checkpoints"} <= tables:
                        raise ValueError("paged assembly index schema is incomplete")
                columns = {
                    row[1] for row in connection.execute(
                        "PRAGMA table_info(assembly_records)"
                    )
                }
                legacy_vector_columns = {
                    "packed_vector", "vector_sha256"
                } <= columns
                if version == 2 and not legacy_vector_columns:
                    raise ValueError("paged assembly v2 vector schema is incomplete")
                existing = connection.execute(
                    "SELECT value FROM index_metadata WHERE key = 'store_id'"
                ).fetchone()
                if existing is None:
                    if version != 0:
                        raise ValueError("paged assembly index store identity is missing")
                    connection.execute(
                        "INSERT INTO index_metadata(key, value) VALUES ('store_id', ?)",
                        (uuid.uuid4().hex,),
                    )
                elif not isinstance(existing[0], str) or not existing[0]:
                    raise ValueError("paged assembly index store identity is invalid")
                for key, initial in (
                    ("index_revision", "0"),
                    ("committed_generation_sha256", ""),
                    ("committed_index_revision", "-1"),
                    ("committed_vector_revision", "-1"),
                ):
                    exists_key = connection.execute(
                        "SELECT 1 FROM index_metadata WHERE key=?", (key,)
                    ).fetchone()
                    if exists_key is None:
                        self._reserve_disk(
                            _WRITE_OVERHEAD_BYTES, "assembly index generation metadata"
                        )
                        connection.execute(
                            "INSERT INTO index_metadata(key,value) VALUES (?,?)",
                            (key, initial),
                        )
                if version in (0, 1):
                    connection.execute(
                        "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
                        ("vector_dimensions", str(requested_dimensions or "")),
                    )
                dimensions_row = connection.execute(
                    "SELECT value FROM index_metadata WHERE key = 'vector_dimensions'"
                ).fetchone()
                if dimensions_row is None:
                    raise ValueError("paged assembly vector metadata is incomplete")
                raw_dimensions = dimensions_row[0]
                if raw_dimensions == "":
                    stored_dimensions = None
                else:
                    try:
                        stored_dimensions = int(raw_dimensions)
                    except (TypeError, ValueError) as error:
                        raise ValueError("packed assembly vector dimensions are invalid") from error
                    if (
                        stored_dimensions < 1
                        or str(stored_dimensions) != raw_dimensions
                    ):
                        raise ValueError("packed assembly vector dimensions are invalid")
                if requested_dimensions is not None:
                    if stored_dimensions is not None and requested_dimensions != stored_dimensions:
                        raise ValueError("packed assembly vector dimensions disagree with store")
                    if stored_dimensions is None:
                        if legacy_vector_columns and connection.execute(
                            "SELECT 1 FROM assembly_records WHERE packed_vector IS NOT NULL "
                            "OR vector_sha256 IS NOT NULL LIMIT 1"
                        ).fetchone() is not None:
                            raise ValueError("packed assembly vectors have no dimensions")
                        self._reserve_disk(
                            _WRITE_OVERHEAD_BYTES, "assembly vector dimensions"
                        )
                        connection.execute(
                            "UPDATE index_metadata SET value = ? "
                            "WHERE key = 'vector_dimensions'",
                            (str(requested_dimensions),),
                        )
                        stored_dimensions = requested_dimensions
                if legacy_vector_columns and stored_dimensions is None:
                    if connection.execute(
                        "SELECT 1 FROM assembly_records WHERE packed_vector IS NOT NULL "
                        "OR vector_sha256 IS NOT NULL LIMIT 1"
                    ).fetchone() is not None:
                        raise ValueError("packed assembly vectors have no dimensions")
                if stored_dimensions is not None:
                    chosen_seed, _deadband = PagedPackedVectors.ensure_schema(
                        connection,
                        dimensions=stored_dimensions,
                        seed=seed,
                        zero_deadband=zero_deadband,
                        reserve_disk=self._reserve_disk,
                    )
                    self._vector_seed = chosen_seed
                else:
                    self._vector_seed = seed
                if version == 2 and legacy_vector_columns:
                    assert stored_dimensions is not None or connection.execute(
                        "SELECT 1 FROM assembly_records WHERE packed_vector IS NOT NULL "
                        "OR vector_sha256 IS NOT NULL LIMIT 1"
                    ).fetchone() is None
                    after = 0
                    while True:
                        sizes = connection.execute(
                            "SELECT sequence,COALESCE(LENGTH(packed_vector),0) "
                            "FROM assembly_records WHERE sequence>? AND "
                            "(packed_vector IS NOT NULL OR vector_sha256 IS NOT NULL) "
                            "ORDER BY sequence LIMIT 128",
                            (after,),
                        ).fetchall()
                        if not sizes:
                            break
                        self._reserve_memory(
                            _READ_OVERHEAD_BYTES * len(sizes)
                            + 4 * sum(int(size) for _, size in sizes),
                            "assembly vector migration page",
                        )
                        last = int(sizes[-1][0])
                        rows = connection.execute(
                            "SELECT sequence,assembly_id,packed_vector,vector_sha256 "
                            "FROM assembly_records WHERE sequence>? AND sequence<=? "
                            "AND (packed_vector IS NOT NULL OR vector_sha256 IS NOT NULL) "
                            "ORDER BY sequence",
                            (after, last),
                        ).fetchall()
                        for sequence, identifier, blob, digest in rows:
                            packed = _decoded_packed_payload(
                                blob, digest, stored_dimensions
                            )
                            existing_vector = connection.execute(
                                "SELECT sequence,vector_id,packed,updates,row_sha256 "
                                "FROM paged_vector_rows WHERE vector_id=?",
                                (identifier,),
                            ).fetchone()
                            if existing_vector is not None and _decoded_row(
                                existing_vector, stored_dimensions
                            )[2] != packed:
                                raise ValueError("legacy assembly vector conflicts with authoritative row")
                            PagedPackedVectors.write_packed_in_connection(
                                connection, identifier, packed,
                                dimensions=stored_dimensions,
                                reserve_disk=self._disk_reserve,
                                preserve_counter_if_same=True,
                            )
                            connection.execute(
                                "UPDATE assembly_records SET packed_vector=NULL,"
                                "vector_sha256=NULL WHERE sequence=?",
                                (sequence,),
                            )
                        after = last
                if legacy_vector_columns:
                    remaining = connection.execute(
                        "SELECT 1 FROM assembly_records WHERE packed_vector IS NOT NULL "
                        "OR vector_sha256 IS NOT NULL LIMIT 1"
                    ).fetchone()
                    if remaining is not None:
                        raise ValueError("legacy assembly vector copy remains after migration")
                    connection.execute(
                        "CREATE TRIGGER IF NOT EXISTS assembly_no_legacy_vector_insert "
                        "BEFORE INSERT ON assembly_records WHEN "
                        "NEW.packed_vector IS NOT NULL OR NEW.vector_sha256 IS NOT NULL "
                        "BEGIN SELECT RAISE(ABORT,'legacy assembly vector copy forbidden'); END"
                    )
                    connection.execute(
                        "CREATE TRIGGER IF NOT EXISTS assembly_no_legacy_vector_update "
                        "BEFORE UPDATE OF packed_vector,vector_sha256 ON assembly_records "
                        "WHEN NEW.packed_vector IS NOT NULL OR NEW.vector_sha256 IS NOT NULL "
                        "BEGIN SELECT RAISE(ABORT,'legacy assembly vector copy forbidden'); END"
                    )
                if version < _VERSION:
                    after_metadata = 0
                    while True:
                        sizes = connection.execute(
                            "SELECT sequence,LENGTH(record_json) FROM assembly_records "
                            "WHERE sequence>? ORDER BY sequence LIMIT 128",
                            (after_metadata,),
                        ).fetchall()
                        if not sizes:
                            break
                        self._reserve_memory(
                            _READ_OVERHEAD_BYTES * len(sizes)
                            + 3 * sum(int(size) for _, size in sizes),
                            "assembly metadata privacy migration",
                        )
                        last_metadata = int(sizes[-1][0])
                        rows = connection.execute(
                            "SELECT sequence,assembly_id,fingerprint,record_json,"
                            "record_sha256 FROM assembly_records WHERE sequence>? "
                            "AND sequence<=? ORDER BY sequence",
                            (after_metadata, last_metadata),
                        ).fetchall()
                        for row in rows:
                            _id, _fp, safe_payload, safe_digest = _record_payload(
                                self._decode_record(row)
                            )
                            if safe_payload != row[3]:
                                self._reserve_disk(
                                    _WRITE_OVERHEAD_BYTES + 4 * len(safe_payload),
                                    "assembly metadata privacy projection",
                                )
                                connection.execute(
                                    "UPDATE assembly_records SET record_json=?,"
                                    "record_sha256=? WHERE sequence=?",
                                    (safe_payload, safe_digest, row[0]),
                                )
                        after_metadata = last_metadata
                    connection.execute("PRAGMA user_version = %d" % _VERSION)
                self._dimensions = stored_dimensions
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        self._vectors = (
            PagedPackedVectors(
                self.path,
                self._dimensions,
                seed=self._vector_seed,
                zero_deadband=zero_deadband,
                resource_policy=resource_policy,
                disk_reserve=disk_reserve,
                memory_reserve=memory_reserve,
            )
            if self._dimensions is not None else None
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        return connection

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _reserve(
        callback: Optional[Callable[[int, str], Any]],
        estimated_bytes: int,
        operation: str,
    ) -> None:
        if callback is not None and callback(max(1, int(estimated_bytes)), operation) is False:
            raise AssemblyIndexResourcePause("%s paused at resource reserve" % operation)

    def _reserve_disk(self, estimated_bytes: int, operation: str) -> None:
        self._reserve(self._disk_reserve, estimated_bytes, operation)

    def _reserve_memory(self, estimated_bytes: int, operation: str) -> None:
        self._reserve(self._memory_reserve, estimated_bytes, operation)

    @staticmethod
    def _store_id(connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'store_id'"
        ).fetchone()
        if row is None or not isinstance(row[0], str) or not row[0]:
            raise ValueError("paged assembly index store identity is invalid")
        return row[0]

    @staticmethod
    def _vector_revision(connection: sqlite3.Connection) -> int:
        return PagedPackedVectors.revision_in_connection(connection)

    @staticmethod
    def _index_revision(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM index_metadata WHERE key='index_revision'"
        ).fetchone()
        if row is None or not isinstance(row[0], str) or not row[0].isdecimal():
            raise ValueError("paged assembly index revision is invalid")
        revision = int(row[0])
        if str(revision) != row[0]:
            raise ValueError("paged assembly index revision is invalid")
        return revision

    @classmethod
    def _bump_index_revision(cls, connection: sqlite3.Connection) -> int:
        revision = cls._index_revision(connection) + 1
        connection.execute(
            "UPDATE index_metadata SET value=? WHERE key='index_revision'",
            (str(revision),),
        )
        return revision

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise ValueError("packed assembly vector dimensions are not configured")
        return self._dimensions

    @staticmethod
    def _decode_record(row: Tuple[Any, ...]) -> Dict[str, Any]:
        sequence, identifier, fingerprint, payload, digest = row
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 1
            or not isinstance(identifier, str)
            or not identifier
            or not isinstance(fingerprint, str)
            or not fingerprint
            or not isinstance(payload, bytes)
            or not isinstance(digest, str)
            or _sha256(payload) != digest
        ):
            raise ValueError("assembly metadata checksum or row identity mismatch")
        try:
            record = json.loads(payload)
        except (UnicodeError, ValueError, TypeError) as error:
            raise ValueError("assembly metadata JSON is invalid") from error
        if (
            not isinstance(record, dict)
            or record.get("id") != identifier
            or record.get("fingerprint") != fingerprint
            or _canonical_json(record) != payload
        ):
            raise ValueError("assembly metadata row identity mismatch")
        return record

    @staticmethod
    def _write_checkpoint(
        connection: sqlite3.Connection,
        checkpoint: Tuple[str, bytes, str],
    ) -> None:
        name, payload, digest = checkpoint
        connection.execute(
            """
            INSERT INTO progress_checkpoints
                (name, payload_json, payload_sha256, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                payload_json=excluded.payload_json,
                payload_sha256=excluded.payload_sha256,
                updated_at=excluded.updated_at
            """,
            (name, payload, digest, time.time()),
        )

    def upsert(
        self,
        record: Mapping[str, Any],
        *,
        packed_vector: Optional[bytes] = None,
        checkpoint: Optional[Tuple[str, Mapping[str, Any]]] = None,
    ) -> bool:
        """Insert or update one assembly; return whether it was newly created.

        An ID and fingerprint may only refer to the same row. Re-exposure
        updates that row without changing its insertion sequence, while an
        idempotent identical record leaves it untouched. The optional source
        checkpoint commits even for an idempotent duplicate, but never on a
        collision, checksum failure, reserve pause, or other failed write.
        ``packed_vector`` is explicitly optional: omitting it never invents
        or erases a vector. A provided exact 2-bit row is admitted to the
        *same live* all-ID vector table in this SQLite transaction.
        """

        identifier, fingerprint, payload, digest = _record_payload(record)
        vector = (
            _packed_payload(packed_vector, self.dimensions)[0]
            if packed_vector is not None
            else None
        )
        cursor = _checkpoint_payload(checkpoint) if checkpoint is not None else None
        self._reserve_memory(
            _READ_OVERHEAD_BYTES
            + 3 * (
                len(payload)
                + (len(vector) if vector else 0)
                + (len(cursor[1]) if cursor else 0)
            ),
            "assembly index write buffer",
        )
        with self._transaction(write=True) as connection:
            created, vector_revision = self._upsert_in_connection(
                connection,
                identifier,
                fingerprint,
                payload,
                digest,
                vector,
                cursor,
            )
        if vector_revision is not None and self._vectors is not None:
            self._vectors.invalidate(identifier, vector_revision)
        return created

    def batch(
        self, *, max_rows: int = 256, max_payload_bytes: int = 8 * 1024 * 1024
    ) -> "PagedAssemblyBatch":
        """Open one bounded metadata/vector/cursor WAL transaction.

        This is the ingestion path for many assembly admissions. The window
        is an I/O batch boundary, never a learned-record cardinality limit;
        callers open successive batches until every valid source row is
        visited, and checkpoint the brain only after the batch commits.
        """

        return PagedAssemblyBatch(
            self, max_rows=max_rows, max_payload_bytes=max_payload_bytes
        )

    def _upsert_in_connection(
        self,
        connection: sqlite3.Connection,
        identifier: str,
        fingerprint: str,
        payload: bytes,
        digest: str,
        vector: Optional[bytes],
        cursor: Optional[Tuple[str, bytes, str]],
    ) -> Tuple[bool, Optional[int]]:
        """Admission primitive shared by single-row and bounded batch paths."""

        matches = connection.execute(
            "SELECT sequence,assembly_id,fingerprint,LENGTH(record_json) "
            "FROM assembly_records WHERE assembly_id=? OR fingerprint=?",
            (identifier, fingerprint),
        ).fetchall()
        self._reserve_memory(
            _READ_OVERHEAD_BYTES + 3 * sum(int(row[3]) for row in matches),
            "assembly index duplicate check",
        )
        existing_payload: Optional[bytes] = None
        for sequence, *_ in matches:
            existing = connection.execute(
                "SELECT sequence,assembly_id,fingerprint,record_json,record_sha256 "
                "FROM assembly_records WHERE sequence=?",
                (sequence,),
            ).fetchone()
            self._decode_record(existing)
            existing_payload = existing[3]
        if len(matches) > 1 or (
            matches and (matches[0][1] != identifier or matches[0][2] != fingerprint)
        ):
            raise ValueError("assembly id/fingerprint collision")
        if matches and vector is None and self._vectors is not None:
            authoritative = connection.execute(
                "SELECT sequence,vector_id,packed,updates,row_sha256 "
                "FROM paged_vector_rows WHERE vector_id=?",
                (identifier,),
            ).fetchone()
            if authoritative is not None:
                _decoded_row(authoritative, self.dimensions)
        created = not matches
        metadata_changed = created or existing_payload != payload
        if metadata_changed:
            self._reserve_disk(
                _WRITE_OVERHEAD_BYTES + 4 * len(payload)
                + 2 * (len(identifier) + len(fingerprint)),
                "assembly index upsert",
            )
            if created:
                connection.execute(
                    "INSERT INTO assembly_records "
                    "(assembly_id,fingerprint,record_json,record_sha256,updated_at) "
                    "VALUES (?,?,?,?,?)",
                    (identifier, fingerprint, payload, digest, time.time()),
                )
            else:
                connection.execute(
                    "UPDATE assembly_records SET record_json=?,record_sha256=?,"
                    "updated_at=? WHERE sequence=?",
                    (payload, digest, time.time(), matches[0][0]),
                )
            self._bump_index_revision(connection)
        vector_revision: Optional[int] = None
        if vector is not None:
            _changed, vector_revision = PagedPackedVectors.write_packed_in_connection(
                connection,
                identifier,
                vector,
                dimensions=self.dimensions,
                reserve_disk=self._disk_reserve,
                preserve_counter_if_same=True,
            )
        if cursor is not None:
            self._reserve_disk(
                _WRITE_OVERHEAD_BYTES + 4 * len(cursor[1]),
                "assembly index checkpoint",
            )
            self._write_checkpoint(connection, cursor)
            self._bump_index_revision(connection)
        return created, vector_revision

    def _lookup(self, column: str, key: str) -> Optional[Dict[str, Any]]:
        if not isinstance(key, str) or not key:
            return None
        if column not in {"assembly_id", "fingerprint"}:
            raise ValueError("unsupported assembly lookup column")
        with self._transaction() as connection:
            size = connection.execute(
                "SELECT sequence, LENGTH(record_json) FROM assembly_records "
                "WHERE %s = ?" % column,
                (key,),
            ).fetchone()
            if size is None:
                return None
            self._reserve_memory(
                _READ_OVERHEAD_BYTES + 3 * int(size[1]),
                "assembly index lookup",
            )
            row = connection.execute(
                """
                SELECT sequence, assembly_id, fingerprint,
                       record_json, record_sha256
                FROM assembly_records WHERE sequence = ?
                """,
                (size[0],),
            ).fetchone()
            if row is None:
                raise ValueError("assembly metadata changed during lookup")
            return self._decode_record(row)

    def get_by_id(self, assembly_id: str) -> Optional[Dict[str, Any]]:
        return self._lookup("assembly_id", assembly_id)

    def get_by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        return self._lookup("fingerprint", fingerprint)

    def get_by_sequence(self, sequence: int) -> Optional[Dict[str, Any]]:
        """Exact stable insertion-sequence lookup for paged list adapters."""

        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            raise ValueError("assembly sequence must be positive")
        with self._transaction() as connection:
            size = connection.execute(
                "SELECT LENGTH(record_json) FROM assembly_records WHERE sequence=?",
                (sequence,),
            ).fetchone()
            if size is None:
                return None
            self._reserve_memory(
                _READ_OVERHEAD_BYTES + 3 * int(size[0]),
                "assembly sequence lookup",
            )
            row = connection.execute(
                "SELECT sequence,assembly_id,fingerprint,record_json,"
                "record_sha256 FROM assembly_records WHERE sequence=?",
                (sequence,),
            ).fetchone()
            if row is None:
                raise ValueError("assembly sequence changed during lookup")
            return self._decode_record(row)

    def get_packed_vector(self, assembly_id: str) -> Optional[bytes]:
        """Read the single authoritative row for an indexed assembly ID."""

        if not isinstance(assembly_id, str) or not assembly_id:
            return None
        if self._vectors is None:
            return None
        with self._transaction() as connection:
            exists = connection.execute(
                "SELECT 1 FROM assembly_records WHERE assembly_id = ?",
                (assembly_id,),
            ).fetchone()
        if exists is None:
            return None
        try:
            return self._vectors.packed_row(assembly_id)
        except KeyError:
            return None

    def current_snapshot(self) -> str:
        """Capture a vector revision token for exact two-pass scoring.

        This store fails closed on revision drift rather than retaining old
        vector generations. A caller should finish both passes before neural
        writes, or use a future immutable generation-backed provider.
        """

        dimensions = self.dimensions
        with self._transaction() as connection:
            store_id = self._store_id(connection)
            revision = self._vector_revision(connection)
            high_water = int(connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM assembly_records"
            ).fetchone()[0])
            vector_rows = int(connection.execute(
                "SELECT COUNT(*) FROM assembly_records AS a "
                "JOIN paged_vector_rows AS v ON v.vector_id=a.assembly_id"
            ).fetchone()[0])
        return _encode_checked_token({
            "format": _VECTOR_SNAPSHOT_FORMAT,
            "formatVersion": _VERSION,
            "storeId": store_id,
            "vectorRevision": revision,
            "throughSequence": high_water,
            "vectorRows": vector_rows,
            "dimensions": dimensions,
        })

    @staticmethod
    def _decode_vector_snapshot(snapshot_id: str) -> Dict[str, Any]:
        value = _decode_checked_token(snapshot_id, "assembly vector snapshot")
        if (
            set(value) != {
                "format", "formatVersion", "storeId", "vectorRevision",
                "throughSequence", "vectorRows", "dimensions",
            }
            or value.get("format") != _VECTOR_SNAPSHOT_FORMAT
            or value.get("formatVersion") != _VERSION
            or not isinstance(value.get("storeId"), str)
            or not value["storeId"]
            or any(
                isinstance(value.get(field), bool)
                or not isinstance(value.get(field), int)
                or value[field] < minimum
                for field, minimum in (
                    ("vectorRevision", 0),
                    ("throughSequence", 0),
                    ("vectorRows", 0),
                    ("dimensions", 1),
                )
            )
        ):
            raise ValueError("assembly vector snapshot is invalid")
        return value

    @staticmethod
    def _encode_vector_cursor(
        snapshot_id: str, after: int, rows_seen: int
    ) -> str:
        return _encode_checked_token({
            "format": _VECTOR_CURSOR_FORMAT,
            "formatVersion": _VERSION,
            "snapshotSha256": _sha256(snapshot_id.encode("ascii")),
            "afterSequence": after,
            "rowsSeen": rows_seen,
        })

    @staticmethod
    def _decode_vector_cursor(
        cursor: str, snapshot_id: str, through: int, vector_rows: int
    ) -> Tuple[int, int]:
        value = _decode_checked_token(cursor, "assembly vector page cursor")
        if (
            set(value) != {
                "format", "formatVersion", "snapshotSha256",
                "afterSequence", "rowsSeen",
            }
            or value.get("format") != _VECTOR_CURSOR_FORMAT
            or value.get("formatVersion") != _VERSION
            or value.get("snapshotSha256") != _sha256(snapshot_id.encode("ascii"))
            or isinstance(value.get("afterSequence"), bool)
            or not isinstance(value.get("afterSequence"), int)
            or not 0 <= value["afterSequence"] <= through
            or isinstance(value.get("rowsSeen"), bool)
            or not isinstance(value.get("rowsSeen"), int)
            or not 0 <= value["rowsSeen"] <= vector_rows
        ):
            raise ValueError("assembly vector page cursor is invalid")
        return value["afterSequence"], value["rowsSeen"]

    def page_rows(
        self, snapshot_id: str, cursor: Optional[str], page_size: int
    ) -> PackedVectorPage:
        """Read a bounded, revision-bound packed vector page by sequence.

        A vector insert/adaptation after ``current_snapshot`` changes the
        revision; the next page fails rather than mixing generations. Source
        metadata without a vector is never emitted as an invented zero row.
        """

        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= _MAX_PAGE_RECORDS
        ):
            raise ValueError("assembly vector page size is invalid")
        snapshot = self._decode_vector_snapshot(snapshot_id)
        through = snapshot["throughSequence"]
        after, rows_seen = (
            self._decode_vector_cursor(
                cursor, snapshot_id, through, snapshot["vectorRows"]
            )
            if cursor is not None else (0, 0)
        )
        with self._transaction() as connection:
            if (
                snapshot["storeId"] != self._store_id(connection)
                or snapshot["dimensions"] != self.dimensions
                or snapshot["vectorRevision"] != self._vector_revision(connection)
            ):
                raise ValueError("assembly vector snapshot generation drift")
            latest = int(connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM assembly_records"
            ).fetchone()[0])
            if through > latest:
                raise ValueError("assembly vector snapshot is ahead of this store")
            sizes = connection.execute(
                """
                SELECT a.sequence, LENGTH(v.packed)
                FROM assembly_records AS a
                JOIN paged_vector_rows AS v ON v.vector_id=a.assembly_id
                WHERE a.sequence > ? AND a.sequence <= ?
                ORDER BY a.sequence LIMIT ?
                """,
                (after, through, page_size + 1),
            ).fetchall()
            has_more = len(sizes) > page_size
            selected = sizes[:page_size]
            self._reserve_memory(
                _READ_OVERHEAD_BYTES * max(1, len(selected))
                + 3 * sum(int(size) for _, size in selected),
                "assembly packed vector page",
            )
            if selected:
                last_sequence = int(selected[-1][0])
                rows = connection.execute(
                    """
                    SELECT a.sequence, a.assembly_id,
                           v.sequence, v.vector_id, v.packed,
                           v.updates, v.row_sha256
                    FROM assembly_records AS a
                    JOIN paged_vector_rows AS v ON v.vector_id=a.assembly_id
                    WHERE a.sequence > ? AND a.sequence <= ?
                    ORDER BY a.sequence
                    """,
                    (after, last_sequence),
                ).fetchall()
                if [row[0] for row in rows] != [row[0] for row in selected]:
                    raise ValueError("assembly vector page changed during read")
            else:
                last_sequence = after
                rows = []
            packed_rows = tuple(
                PackedVectorRow(
                    sequence=int(assembly_sequence),
                    assembly_id=str(identifier),
                    packed=_decoded_row(
                        (vector_sequence, vector_id, packed, updates, digest),
                        self.dimensions,
                    )[2],
                )
                for (
                    assembly_sequence, identifier, vector_sequence,
                    vector_id, packed, updates, digest,
                ) in rows
            )
            next_rows_seen = rows_seen + len(packed_rows)
            if (
                next_rows_seen > snapshot["vectorRows"]
                or (not has_more and next_rows_seen != snapshot["vectorRows"])
            ):
                raise ValueError("assembly vector snapshot coverage mismatch")
            return PackedVectorPage(
                rows=packed_rows,
                next_cursor=self._encode_vector_cursor(
                    snapshot_id, last_sequence, next_rows_seen
                ),
                has_more=has_more,
                snapshot_id=snapshot_id,
            )

    @staticmethod
    def _encode_cursor(store_id: str, through: int, after: int) -> str:
        return _encode_checked_token({
            "format": _CURSOR_FORMAT,
            "version": _VERSION,
            "storeId": store_id,
            "throughSequence": through,
            "afterSequence": after,
        })

    @staticmethod
    def _decode_cursor(token: str, store_id: str) -> Tuple[int, int]:
        value = _decode_checked_token(token, "assembly page cursor")
        if (
            value.get("format") != _CURSOR_FORMAT
            or value.get("version") not in (1, _VERSION)
            or value.get("storeId") != store_id
            or isinstance(value.get("throughSequence"), bool)
            or not isinstance(value.get("throughSequence"), int)
            or isinstance(value.get("afterSequence"), bool)
            or not isinstance(value.get("afterSequence"), int)
            or value["throughSequence"] < 0
            or not 0 <= value["afterSequence"] <= value["throughSequence"]
        ):
            raise ValueError("assembly page cursor is invalid or belongs to another store")
        return value["throughSequence"], value["afterSequence"]

    def page(self, page_size: int = 128, cursor: Optional[str] = None) -> AssemblyPage:
        """Read one stable-insertion-bound page without an OFFSET/full scan.

        The first page captures the current high-water sequence. Later inserts
        are excluded from that sweep; updates to as-yet-unread rows remain
        visible. This is a bounded traversal, not a multi-call SQLite snapshot.
        The returned cursor is durable even on the final or empty page.
        """

        if (
            isinstance(page_size, bool)
            or not isinstance(page_size, int)
            or not 1 <= page_size <= _MAX_PAGE_RECORDS
        ):
            raise ValueError("assembly page size must be between 1 and %d" % _MAX_PAGE_RECORDS)
        with self._transaction() as connection:
            store_id = self._store_id(connection)
            latest = int(connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM assembly_records"
            ).fetchone()[0])
            if cursor is None:
                through, after = latest, 0
            else:
                through, after = self._decode_cursor(cursor, store_id)
                if through > latest:
                    raise ValueError("assembly page cursor is ahead of this store")
            sizes = connection.execute(
                """
                SELECT sequence, LENGTH(record_json)
                FROM assembly_records
                WHERE sequence > ? AND sequence <= ?
                ORDER BY sequence LIMIT ?
                """,
                (after, through, page_size + 1),
            ).fetchall()
            has_more = len(sizes) > page_size
            selected = sizes[:page_size]
            self._reserve_memory(
                _READ_OVERHEAD_BYTES * max(1, len(selected))
                + 3 * sum(int(size) for _, size in selected),
                "assembly index page",
            )
            if selected:
                last_sequence = int(selected[-1][0])
                rows = connection.execute(
                    """
                    SELECT sequence, assembly_id, fingerprint,
                           record_json, record_sha256
                    FROM assembly_records
                    WHERE sequence > ? AND sequence <= ?
                    ORDER BY sequence
                    """,
                    (after, last_sequence),
                ).fetchall()
                if [row[0] for row in rows] != [row[0] for row in selected]:
                    raise ValueError("assembly page changed during read")
            else:
                last_sequence = after
                rows = []
            return AssemblyPage(
                records=tuple(self._decode_record(row) for row in rows),
                cursor=self._encode_cursor(store_id, through, last_sequence),
                has_more=has_more,
                through_sequence=through,
            )

    def page_after(
        self,
        sequence: int,
        *,
        page_size: int = 128,
        through_sequence: Optional[int] = None,
    ) -> AssemblyPage:
        """Read after a numeric sequence, optionally pinned to a high-water."""

        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("assembly page sequence is invalid")
        if through_sequence is not None and (
            isinstance(through_sequence, bool)
            or not isinstance(through_sequence, int)
            or through_sequence < sequence
        ):
            raise ValueError("assembly page high-water is invalid")
        with self._transaction() as connection:
            store_id = self._store_id(connection)
            latest = int(connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM assembly_records"
            ).fetchone()[0])
            through = latest if through_sequence is None else through_sequence
            if through > latest or sequence > through:
                raise ValueError("assembly page sequence is ahead of this store")
        return self.page(
            page_size=page_size,
            cursor=self._encode_cursor(store_id, through, sequence),
        )

    def iter_pages(
        self, page_size: int = 128, cursor: Optional[str] = None
    ) -> Iterator[AssemblyPage]:
        """Yield nonempty pages until the captured insertion bound is exhausted."""

        while True:
            page = self.page(page_size=page_size, cursor=cursor)
            if page.records:
                yield page
            if not page.has_more:
                return
            cursor = page.cursor

    def count(self) -> int:
        with self._transaction() as connection:
            return int(connection.execute(
                "SELECT COUNT(*) FROM assembly_records"
            ).fetchone()[0])

    def save_checkpoint(self, name: str, value: Mapping[str, Any]) -> None:
        """Atomically save a source progress cursor without admitting a record."""

        checkpoint = _checkpoint_payload((name, value))
        self._reserve_memory(
            _READ_OVERHEAD_BYTES + 3 * len(checkpoint[1]),
            "assembly index checkpoint buffer",
        )
        self._reserve_disk(
            _WRITE_OVERHEAD_BYTES + 4 * len(checkpoint[1]),
            "assembly index checkpoint",
        )
        with self._transaction(write=True) as connection:
            self._write_checkpoint(connection, checkpoint)
            self._bump_index_revision(connection)

    def load_checkpoint(self, name: str) -> Optional[Dict[str, Any]]:
        """Read a mutable working cursor, not an authoritative resume point.

        Startup/resume callers must use ``load_committed_checkpoint`` with
        the verified brain-pointer generation hash.
        """

        if not isinstance(name, str) or not name.strip():
            raise ValueError("assembly checkpoint requires a nonempty name")
        with self._transaction() as connection:
            return self._read_checkpoint(connection, name)

    def _read_checkpoint(
        self, connection: sqlite3.Connection, name: str
    ) -> Optional[Dict[str, Any]]:
        size = connection.execute(
            "SELECT LENGTH(payload_json) FROM progress_checkpoints WHERE name = ?",
            (name,),
        ).fetchone()
        if size is None:
            return None
        self._reserve_memory(
            _READ_OVERHEAD_BYTES + 3 * int(size[0]),
            "assembly index checkpoint read",
        )
        row = connection.execute(
            "SELECT payload_json, payload_sha256 "
            "FROM progress_checkpoints WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None or not isinstance(row[0], bytes) or _sha256(row[0]) != row[1]:
            raise ValueError("assembly checkpoint checksum mismatch")
        try:
            value = json.loads(row[0])
        except (TypeError, ValueError, UnicodeError) as error:
            raise ValueError("assembly checkpoint JSON is invalid") from error
        if not isinstance(value, dict) or _canonical_json(value) != row[0]:
            raise ValueError("assembly checkpoint is invalid")
        return value

    def bind_committed_generation(
        self,
        generation_sha256: str,
        *,
        expected_index_revision: Optional[int] = None,
        expected_vector_revision: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Bind working index/vector revisions to a verified brain pointer.

        This is a cache-generation assertion, not a new checkpoint authority.
        Call only after the v3 shard manifest and ``brain.json`` pointer have
        been durably published and independently verified.
        """

        if (
            not isinstance(generation_sha256, str)
            or len(generation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in generation_sha256)
        ):
            raise ValueError("committed brain generation hash is invalid")
        for value in (expected_index_revision, expected_vector_revision):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("expected cache revision is invalid")
        self._reserve_disk(_WRITE_OVERHEAD_BYTES * 2, "assembly generation binding")
        with self._transaction(write=True) as connection:
            index_revision = self._index_revision(connection)
            vector_revision = (
                self._vector_revision(connection) if self._dimensions is not None else 0
            )
            if (
                expected_index_revision is not None
                and expected_index_revision != index_revision
            ) or (
                expected_vector_revision is not None
                and expected_vector_revision != vector_revision
            ):
                raise ValueError("cache revision changed before generation binding")
            for key, value in (
                ("committed_generation_sha256", generation_sha256),
                ("committed_index_revision", str(index_revision)),
                ("committed_vector_revision", str(vector_revision)),
            ):
                connection.execute(
                    "UPDATE index_metadata SET value=? WHERE key=?",
                    (value, key),
                )
            if self._dimensions is not None:
                connection.execute(
                    "UPDATE paged_vector_meta SET value=? "
                    "WHERE key='committed_generation_sha256'",
                    (generation_sha256,),
                )
                connection.execute(
                    "UPDATE paged_vector_meta SET value=? "
                    "WHERE key='committed_revision'",
                    (str(vector_revision),),
                )
        return {
            "generationSha256": generation_sha256,
            "indexRevision": index_revision,
            "vectorRevision": vector_revision,
        }

    def _assert_committed_in_connection(
        self, connection: sqlite3.Connection, committed_generation_sha256: str
    ) -> Dict[str, Any]:
        rows = dict(connection.execute(
                "SELECT key,value FROM index_metadata WHERE key IN "
                "('committed_generation_sha256','committed_index_revision',"
                "'committed_vector_revision')"
        ))
        current_index = self._index_revision(connection)
        current_vector = (
            self._vector_revision(connection) if self._dimensions is not None else 0
        )
        expected = (
            rows.get("committed_generation_sha256") == committed_generation_sha256
            and rows.get("committed_index_revision") == str(current_index)
            and rows.get("committed_vector_revision") == str(current_vector)
        )
        if expected and self._dimensions is not None:
            vector_generation, vector_revision = (
                PagedPackedVectors._committed_binding_in_connection(connection)
            )
            expected = (
                vector_generation == committed_generation_sha256
                and vector_revision == current_vector
            )
        if not expected:
            raise PagedVectorCacheNeedsRebuild(
                "assembly/vector cache must be rebuilt from committed substrate shards"
            )
        return {
            "state": "clean",
            "generationSha256": committed_generation_sha256,
            "indexRevision": current_index,
            "vectorRevision": current_vector,
        }

    def discard_or_reconcile_uncommitted(
        self, committed_generation_sha256: str
    ) -> Dict[str, Any]:
        """Fail closed when mutable cache/cursor outruns the brain pointer.

        Reconciliation needs the committed shard reader, which lives outside
        this storage primitive. On mismatch, callers must rebuild derived
        SQLite rows from that generation before accepting any source cursor.
        """

        try:
            with self._transaction() as connection:
                return self._assert_committed_in_connection(
                    connection, committed_generation_sha256
                )
        except PagedVectorCacheNeedsRebuild:
            if self._vectors is not None:
                self._vectors.invalidate_all()
            raise

    def load_committed_checkpoint(
        self, name: str, committed_generation_sha256: str
    ) -> Optional[Dict[str, Any]]:
        """Return a cursor only when the cache matches the verified pointer."""

        if not isinstance(name, str) or not name.strip():
            raise ValueError("assembly checkpoint requires a nonempty name")
        with self._transaction() as connection:
            self._assert_committed_in_connection(
                connection, committed_generation_sha256
            )
            return self._read_checkpoint(connection, name)

    def status(self) -> Dict[str, Any]:
        with self._transaction() as connection:
            count, high_water = connection.execute(
                "SELECT COUNT(*), COALESCE(MAX(sequence), 0) "
                "FROM assembly_records"
            ).fetchone()
            store_id = self._store_id(connection)
            index_revision = self._index_revision(connection)
            bindings = dict(connection.execute(
                "SELECT key,value FROM index_metadata WHERE key IN "
                "('committed_generation_sha256','committed_index_revision',"
                "'committed_vector_revision')"
            ))
            if self._dimensions is None:
                vector_rows, vector_revision = 0, 0
            else:
                vector_rows = int(connection.execute(
                    "SELECT COUNT(*) FROM assembly_records AS a "
                    "JOIN paged_vector_rows AS v ON v.vector_id=a.assembly_id"
                ).fetchone()[0])
                vector_revision = self._vector_revision(connection)
            committed_generation = bindings.get("committed_generation_sha256") or None
            dirty = (
                committed_generation is None
                or bindings.get("committed_index_revision") != str(index_revision)
                or bindings.get("committed_vector_revision") != str(vector_revision)
            )
            if self._dimensions is not None:
                vector_generation, vector_bound_revision = (
                    PagedPackedVectors._committed_binding_in_connection(connection)
                )
                dirty = dirty or (
                    vector_generation != committed_generation
                    or vector_bound_revision != vector_revision
                )
        durable_bytes = sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                Path(str(self.path) + "-wal"),
                Path(str(self.path) + "-shm"),
            )
            if candidate.is_file()
        )
        return {
            "format": _FORMAT,
            "formatVersion": _VERSION,
            "count": int(count),
            "highWaterSequence": int(high_water),
            "storeId": store_id,
            "vectorDimensions": self._dimensions,
            "packedVectorRows": int(vector_rows),
            "vectorRevision": vector_revision,
            "indexRevision": index_revision,
            "committedGenerationSha256": committed_generation,
            "dirtySinceCommit": dirty,
            "bytes": durable_bytes,
            "bytesSharedWithVectorStore": self._dimensions is not None,
            "transactional": True,
            "storage": "disk",
        }


class PagedAssemblyBatch:
    """One bounded transaction for metadata, shared vectors, and a cursor hint.

    No row dictionary is buffered; each upsert writes within one SQLite WAL
    transaction. Any failed operation marks the batch abort-only, including
    errors caught by a caller inside the context. A source cursor here remains
    a *working hint* until a verified brain generation is bound later.
    """

    def __init__(
        self,
        index: PagedAssemblyIndex,
        *,
        max_rows: int,
        max_payload_bytes: int,
    ) -> None:
        if (
            isinstance(max_rows, bool) or not isinstance(max_rows, int)
            or not 1 <= max_rows <= _MAX_PAGE_RECORDS
        ):
            raise ValueError("assembly batch row window is invalid")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or not 1 <= max_payload_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("assembly batch byte window is invalid")
        self.index = index
        self.max_rows = max_rows
        self.max_payload_bytes = max_payload_bytes
        self.rows_attempted = 0
        self.payload_bytes = 0
        self._connection: Optional[sqlite3.Connection] = None
        self._vector_revision: Optional[int] = None
        self._checkpoint_written = False
        self._aborted = False

    def __enter__(self) -> "PagedAssemblyBatch":
        if self._connection is not None:
            raise RuntimeError("assembly batch is already open")
        self.index._reserve_disk(_WRITE_OVERHEAD_BYTES, "assembly batch begin")
        connection = self.index._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return self

    def _connection_or_raise(self) -> sqlite3.Connection:
        if self._connection is None or self._aborted:
            raise RuntimeError("assembly batch is not active")
        return self._connection

    def _charge(self, *, rows: int, payload_bytes: int) -> None:
        next_rows = self.rows_attempted + rows
        next_bytes = self.payload_bytes + 4 * payload_bytes + 128
        if next_rows > self.max_rows or next_bytes > self.max_payload_bytes:
            self._aborted = True
            raise AssemblyIndexResourcePause(
                "assembly batch window reached; retry in a new transaction"
            )
        self.rows_attempted = next_rows
        self.payload_bytes = next_bytes

    def upsert(
        self,
        record: Mapping[str, Any],
        *,
        packed_vector: Optional[bytes] = None,
        checkpoint: Optional[Tuple[str, Mapping[str, Any]]] = None,
    ) -> bool:
        connection = self._connection_or_raise()
        try:
            if self._checkpoint_written:
                raise ValueError("assembly batch cursor hint must be final")
            identifier, fingerprint, payload, digest = _record_payload(record)
            vector = (
                _packed_payload(packed_vector, self.index.dimensions)[0]
                if packed_vector is not None else None
            )
            cursor = (
                _checkpoint_payload(checkpoint)
                if checkpoint is not None else None
            )
            if cursor is not None and self._checkpoint_written:
                raise ValueError("assembly batch accepts one final cursor hint")
            self._charge(
                rows=1,
                payload_bytes=(
                    len(payload) + len(identifier) + len(fingerprint)
                    + (len(vector) if vector is not None else 0)
                    + (len(cursor[1]) if cursor is not None else 0)
                ),
            )
            self.index._reserve_memory(
                _READ_OVERHEAD_BYTES
                + 3 * (
                    len(payload)
                    + (len(vector) if vector is not None else 0)
                    + (len(cursor[1]) if cursor is not None else 0)
                ),
                "assembly batch row buffer",
            )
            created, revision = self.index._upsert_in_connection(
                connection,
                identifier,
                fingerprint,
                payload,
                digest,
                vector,
                cursor,
            )
            if revision is not None:
                self._vector_revision = revision
            if cursor is not None:
                self._checkpoint_written = True
            return created
        except BaseException:
            self._aborted = True
            raise

    def save_checkpoint(self, name: str, value: Mapping[str, Any]) -> None:
        connection = self._connection_or_raise()
        try:
            if self._checkpoint_written:
                raise ValueError("assembly batch accepts one final cursor hint")
            cursor = _checkpoint_payload((name, value))
            self._charge(rows=0, payload_bytes=len(cursor[1]) + len(name))
            self.index._reserve_memory(
                _READ_OVERHEAD_BYTES + 3 * len(cursor[1]),
                "assembly batch cursor buffer",
            )
            self.index._reserve_disk(
                _WRITE_OVERHEAD_BYTES + 4 * len(cursor[1]),
                "assembly batch cursor",
            )
            self.index._write_checkpoint(connection, cursor)
            self.index._bump_index_revision(connection)
            self._checkpoint_written = True
        except BaseException:
            self._aborted = True
            raise

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        connection = self._connection
        if connection is None:
            raise RuntimeError("assembly batch was not opened")
        self._connection = None
        committed = False
        try:
            if error_type is not None or self._aborted:
                connection.rollback()
            else:
                connection.commit()
                committed = True
        finally:
            connection.close()
        if committed and self._vector_revision is not None and self.index._vectors is not None:
            self.index._vectors.invalidate_all(self._vector_revision)
        elif self._aborted and error_type is None:
            raise AssemblyIndexResourcePause("assembly batch aborted")
        return False
