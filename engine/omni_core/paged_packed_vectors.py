"""Disk-backed live ternary vector rows for the VSA substrate.

One SQLite row owns each neuron/assembly ID's exact two-bit vector and
deterministic update counter. An assembly is also a neuron, so both views must
address this same row. Decoded tensors live only in a byte-bounded int8 LRU;
iteration and similarity pages never reconstruct a whole vector dictionary.

The live page provider binds both scorer passes to one vector revision and
fails closed if a write occurs. Committed v3 substrate shards and their single
``brain.json`` generation pointer remain checkpoint authority; this SQLite
working cache is verified or rebuilt from them on recovery, not independently
promoted as a second model or copied at every ingestion checkpoint.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import sqlite3
import struct
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Tuple

import torch

from .paged_store_counts import ensure_row_count, row_count
from .packed_vsa_vectors import (
    FORMAT as SHARD_FORMAT,
    FORMAT_VERSION as SHARD_FORMAT_VERSION,
    _MAX_UPDATES,
    _decode_exact_row,
    _pack_exact_row,
    _splitmix64,
    _valid_deadband,
    _validate_packed_row,
    PackedTernaryVectors,
    quantize_vsa_vector,
)
from .paged_vector_scoring import PackedVectorPage, PackedVectorRow


FORMAT = "omni-paged-packed-vectors"
FORMAT_VERSION = 1
MAX_PAGE_ROWS = 4096
MAX_DECODE_ROWS = 64
MAX_EXPORT_ROWS = 4096
_SNAPSHOT_FORMAT = "omni-paged-vector-snapshot"
_CURSOR_FORMAT = "omni-paged-vector-cursor"
_WRITE_OVERHEAD_BYTES = 64 * 1024
_READ_OVERHEAD_BYTES = 4 * 1024


class PagedVectorResourcePause(RuntimeError):
    """A configured RAM/disk reserve refused the next operation."""


class PagedVectorCacheNeedsRebuild(RuntimeError):
    """SQLite cache differs from the authoritative committed shard generation."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _token(value: dict[str, Any]) -> str:
    payload = _canonical_json(value)
    return (
        base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
        + "." + _sha256(payload)
    )


def _untoken(token: str, label: str) -> dict[str, Any]:
    if not isinstance(token, str) or len(token) > 2048:
        raise ValueError("%s is invalid" % label)
    try:
        encoded, digest = token.split(".", 1)
        payload = base64.b64decode(
            encoded + "=" * (-len(encoded) % 4),
            altchars=b"-_", validate=True,
        )
        value = json.loads(payload)
    except (ValueError, TypeError, UnicodeError, binascii.Error) as error:
        raise ValueError("%s is invalid" % label) from error
    if (
        not isinstance(value, dict)
        or _sha256(payload) != digest
        or _canonical_json(value) != payload
    ):
        raise ValueError("%s is invalid" % label)
    return value


def _identifier(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("paged vector IDs must be nonempty strings")
    return value


def _counter(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_UPDATES:
        raise ValueError("paged vector update counter is invalid")
    return value


def _row_sha(packed: bytes, updates: int) -> str:
    return _sha256(packed + struct.pack("<Q", updates))


def _valid_rate(rate: Any) -> float:
    if isinstance(rate, (bool, str, bytes)):
        raise ValueError("paged vector adaptation rate must be in [0, 1]")
    try:
        alpha = float(rate)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("paged vector adaptation rate must be in [0, 1]") from error
    if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("paged vector adaptation rate must be in [0, 1]")
    return alpha


def _activity(levels: torch.Tensor) -> torch.Tensor:
    """Match the live packed mapping's normalized float activity view."""

    activity = levels.to(torch.float32)
    norm = activity.norm()
    return activity / norm if float(norm) > 0.0 else activity


def _decoded_row(row: Tuple[Any, ...], dimensions: int) -> Tuple[int, str, bytes, int]:
    sequence, identifier, packed, updates, digest = row
    if (
        isinstance(sequence, bool) or not isinstance(sequence, int)
        or sequence < 1 or not isinstance(identifier, str) or not identifier
        or not isinstance(packed, bytes) or not isinstance(digest, str)
    ):
        raise ValueError("paged vector row is invalid")
    updates = _counter(updates)
    if _row_sha(packed, updates) != digest:
        raise ValueError("paged vector row checksum mismatch")
    _validate_packed_row(packed, dimensions)
    return sequence, identifier, packed, updates


class PagedPackedVectors(MutableMapping[str, torch.Tensor]):
    """One live packed row per ID, with a bounded decoded LRU.

    ``dimensions`` is required and immutable. Existing stores adopt their
    persisted seed/deadband when those arguments are omitted. ``cache_bytes``
    is a residency budget, not a row-count or neural-capacity limit.
    """

    packed_authoritative = True

    def __init__(
        self,
        path: Path,
        dimensions: int,
        *,
        seed: Optional[int] = None,
        zero_deadband: Optional[float] = None,
        cache_bytes: int = 8 * 1024 * 1024,
        resource_policy: Optional[Any] = None,
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
        memory_reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
            raise ValueError("paged vector dimensions must be positive")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("paged vector seed must be an integer")
        if zero_deadband is not None:
            zero_deadband = _valid_deadband(zero_deadband)
        if isinstance(cache_bytes, bool) or not isinstance(cache_bytes, int) or cache_bytes < 0:
            raise ValueError("paged vector cache budget must be nonnegative")
        if resource_policy is not None and disk_reserve is not None:
            raise ValueError("provide either resource_policy or disk_reserve")
        self.path = Path(path)
        self.dimensions = dimensions
        self.row_bytes = (dimensions + 3) // 4
        self._disk_reserve = (
            resource_policy.require_disk if resource_policy is not None
            else disk_reserve
        )
        self._memory_reserve = memory_reserve
        self.cache_capacity_bytes = cache_bytes
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._cache_bytes = 0
        self._cache_revision: Optional[int] = None
        self._cache_lock = threading.RLock()
        if not self.path.is_file():
            self._reserve_disk(_WRITE_OVERHEAD_BYTES * 2, "paged vector create")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            if mode != "wal":
                self._reserve_disk(_WRITE_OVERHEAD_BYTES * 2, "paged vector WAL setup")
                connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                self.seed, self.zero_deadband = self.ensure_schema(
                    connection,
                    dimensions=dimensions,
                    seed=seed,
                    zero_deadband=zero_deadband,
                    reserve_disk=self._reserve_disk,
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def ensure_schema(
        connection: sqlite3.Connection,
        *,
        dimensions: int,
        seed: Optional[int],
        zero_deadband: Optional[float],
        reserve_disk: Optional[Callable[[int, str], None]] = None,
    ) -> Tuple[int, float]:
        """Initialize/validate tables inside the caller's SQLite transaction."""

        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='paged_vector_meta'"
        ).fetchone()
        if exists is None:
            if reserve_disk is not None:
                reserve_disk(_WRITE_OVERHEAD_BYTES * 2, "paged vector schema")
            chosen_seed = 0 if seed is None else seed
            chosen_deadband = 0.25 if zero_deadband is None else float(zero_deadband)
            connection.execute(
                """
                CREATE TABLE paged_vector_meta (
                    key TEXT PRIMARY KEY, value TEXT NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                """
                CREATE TABLE paged_vector_rows (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id TEXT NOT NULL UNIQUE,
                    packed BLOB NOT NULL,
                    updates INTEGER NOT NULL,
                    row_sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.executemany(
                "INSERT INTO paged_vector_meta(key,value) VALUES (?,?)",
                (
                    ("format", FORMAT),
                    ("format_version", str(FORMAT_VERSION)),
                    ("store_id", uuid.uuid4().hex),
                    ("dimensions", str(dimensions)),
                    ("seed", str(chosen_seed)),
                    ("zero_deadband", repr(chosen_deadband)),
                    ("revision", "0"),
                    ("committed_generation_sha256", ""),
                    ("committed_revision", "-1"),
                ),
            )
            ensure_row_count(connection, "paged_vector_rows", reserve_disk)
            return chosen_seed, chosen_deadband
        rows = dict(connection.execute("SELECT key,value FROM paged_vector_meta"))
        if (
            rows.get("format") != FORMAT
            or rows.get("format_version") != str(FORMAT_VERSION)
            or rows.get("dimensions") != str(dimensions)
            or not rows.get("store_id")
        ):
            raise ValueError("paged vector store metadata is incompatible")
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='paged_vector_rows'"
        ).fetchone()
        if table is None:
            raise ValueError("paged vector row table is missing")
        try:
            stored_seed = int(rows["seed"])
            stored_deadband = float(rows["zero_deadband"])
            revision = int(rows["revision"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("paged vector store metadata is invalid") from error
        if (
            str(stored_seed) != rows["seed"]
            or not math.isfinite(stored_deadband)
            or not 0.0 <= stored_deadband < 1.0
            or repr(stored_deadband) != rows["zero_deadband"]
            or revision < 0
            or str(revision) != rows["revision"]
            or (seed is not None and seed != stored_seed)
            or (zero_deadband is not None and float(zero_deadband) != stored_deadband)
        ):
            raise ValueError("paged vector store metadata is incompatible")
        if (
            "committed_generation_sha256" not in rows
            or "committed_revision" not in rows
        ) and reserve_disk is not None:
            reserve_disk(_WRITE_OVERHEAD_BYTES, "paged vector binding schema")
        if "committed_generation_sha256" not in rows:
            connection.execute(
                "INSERT INTO paged_vector_meta(key,value) VALUES "
                "('committed_generation_sha256','')"
            )
        if "committed_revision" not in rows:
            connection.execute(
                "INSERT INTO paged_vector_meta(key,value) VALUES "
                "('committed_revision','-1')"
            )
        ensure_row_count(connection, "paged_vector_rows", reserve_disk)
        return stored_seed, stored_deadband

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        from .paged_idle_selection import observe_idle_source_connection
        observe_idle_source_connection(connection, self.path)
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
        estimated: int,
        operation: str,
    ) -> None:
        if callback is not None and callback(max(1, int(estimated)), operation) is False:
            raise PagedVectorResourcePause("%s paused at resource reserve" % operation)

    def _reserve_disk(self, estimated: int, operation: str) -> None:
        self._reserve(self._disk_reserve, estimated, operation)

    def _reserve_memory(self, estimated: int, operation: str) -> None:
        self._reserve(self._memory_reserve, estimated, operation)

    @staticmethod
    def revision_in_connection(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM paged_vector_meta WHERE key='revision'"
        ).fetchone()
        if row is None or not isinstance(row[0], str) or not row[0].isdecimal():
            raise ValueError("paged vector revision is invalid")
        revision = int(row[0])
        if str(revision) != row[0]:
            raise ValueError("paged vector revision is invalid")
        return revision

    @staticmethod
    def store_id_in_connection(connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT value FROM paged_vector_meta WHERE key='store_id'"
        ).fetchone()
        if row is None or not isinstance(row[0], str) or not row[0]:
            raise ValueError("paged vector store identity is invalid")
        return row[0]

    @staticmethod
    def _committed_binding_in_connection(
        connection: sqlite3.Connection,
    ) -> Tuple[Optional[str], Optional[int]]:
        rows = dict(connection.execute(
            "SELECT key,value FROM paged_vector_meta WHERE key IN "
            "('committed_generation_sha256','committed_revision')"
        ))
        generation = rows.get("committed_generation_sha256")
        raw_revision = rows.get("committed_revision")
        if generation == "" and raw_revision == "-1":
            return None, None
        if (
            not isinstance(generation, str)
            or len(generation) != 64
            or any(character not in "0123456789abcdef" for character in generation)
            or not isinstance(raw_revision, str)
            or not raw_revision.isdecimal()
        ):
            raise ValueError("paged vector committed binding is invalid")
        revision = int(raw_revision)
        if str(revision) != raw_revision:
            raise ValueError("paged vector committed binding is invalid")
        return generation, revision

    def bind_committed_generation(
        self, generation_sha256: str, *, expected_revision: Optional[int] = None
    ) -> dict[str, Any]:
        """Bind this cache revision to an externally verified brain pointer.

        The caller must first verify that the v3 shard generation, cursor, and
        ``brain.json`` pointer are committed together. This method records
        the assertion; it does not turn SQLite into checkpoint authority.
        """

        if (
            not isinstance(generation_sha256, str)
            or len(generation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in generation_sha256)
        ):
            raise ValueError("committed brain generation hash is invalid")
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected vector revision is invalid")
        self._reserve_disk(_WRITE_OVERHEAD_BYTES, "paged vector generation binding")
        with self._transaction(write=True) as connection:
            revision = self.revision_in_connection(connection)
            if expected_revision is not None and revision != expected_revision:
                raise ValueError("paged vector revision changed before generation binding")
            connection.execute(
                "UPDATE paged_vector_meta SET value=? "
                "WHERE key='committed_generation_sha256'",
                (generation_sha256,),
            )
            connection.execute(
                "UPDATE paged_vector_meta SET value=? WHERE key='committed_revision'",
                (str(revision),),
            )
        return {"generationSha256": generation_sha256, "revision": revision}

    def committed_binding(self) -> dict[str, Any]:
        with self._transaction() as connection:
            generation, committed_revision = self._committed_binding_in_connection(
                connection
            )
            current_revision = self.revision_in_connection(connection)
        return {
            "generationSha256": generation,
            "committedRevision": committed_revision,
            "currentRevision": current_revision,
            "dirtySinceCommit": (
                generation is None or committed_revision != current_revision
            ),
        }

    def discard_or_reconcile_uncommitted(
        self, committed_generation_sha256: str
    ) -> dict[str, Any]:
        """Fail closed unless cache matches the verified brain generation.

        Selectively discarding dirty rows is impossible without the committed
        shard reader; callers must rebuild this *derived* cache from bounded
        authoritative shard pages, then bind the verified generation. No
        ingestion cursor may be accepted from a dirty cache.
        """

        binding = self.committed_binding()
        if (
            binding["generationSha256"] != committed_generation_sha256
            or binding["dirtySinceCommit"]
        ):
            self.invalidate_all()
            raise PagedVectorCacheNeedsRebuild(
                "paged vector cache must be rebuilt from committed substrate shards"
            )
        return {"state": "clean", **binding}

    @staticmethod
    def _bump_revision(connection: sqlite3.Connection) -> int:
        revision = PagedPackedVectors.revision_in_connection(connection) + 1
        connection.execute(
            "UPDATE paged_vector_meta SET value=? WHERE key='revision'",
            (str(revision),),
        )
        return revision

    def _decode_row(self, row: Tuple[Any, ...]) -> Tuple[int, str, bytes, int]:
        return _decoded_row(row, self.dimensions)

    def _cache_clear_if_revision_changed(self, revision: int) -> None:
        with self._cache_lock:
            if self._cache_revision != revision:
                self._cache.clear()
                self._cache_bytes = 0
                self._cache_revision = revision

    def _cache_put(self, identifier: str, levels: torch.Tensor) -> None:
        if self.cache_capacity_bytes < self.dimensions:
            return
        with self._cache_lock:
            if identifier in self._cache:
                self._cache.pop(identifier)
                self._cache_bytes -= self.dimensions
            while self._cache and self._cache_bytes + self.dimensions > self.cache_capacity_bytes:
                self._cache.popitem(last=False)
                self._cache_bytes -= self.dimensions
            self._cache[identifier] = levels.clone()
            self._cache_bytes += self.dimensions

    def invalidate(self, identifier: str, revision: Optional[int] = None) -> None:
        """Discard one hot copy after another same-DB transactional writer."""

        with self._cache_lock:
            if identifier in self._cache:
                self._cache.pop(identifier)
                self._cache_bytes -= self.dimensions
            if revision is not None:
                self._cache_revision = revision

    def invalidate_all(self, revision: Optional[int] = None) -> None:
        """Drop the bounded hot cache after a committed multi-row batch."""

        with self._cache_lock:
            self._cache.clear()
            self._cache_bytes = 0
            self._cache_revision = revision

    def __len__(self) -> int:
        with self._transaction() as connection:
            return row_count(connection, "paged_vector_rows")

    @property
    def storage_bytes(self) -> int:
        """Logical packed-level and update-counter bytes, without SQL overhead."""

        return len(self) * (self.row_bytes + 8)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        with self._transaction() as connection:
            return connection.execute(
                "SELECT 1 FROM paged_vector_rows WHERE vector_id=?",
                (key,),
            ).fetchone() is not None

    def __iter__(self) -> Iterator[str]:
        snapshot = self.current_snapshot()
        descriptor = self._decode_snapshot(snapshot)
        cursor = 0
        while True:
            with self._transaction() as connection:
                if (
                    self.revision_in_connection(connection) != descriptor["revision"]
                    or self.store_id_in_connection(connection) != descriptor["storeId"]
                ):
                    raise ValueError("paged vector iteration generation drift")
                rows = connection.execute(
                    "SELECT sequence,vector_id FROM paged_vector_rows "
                    "WHERE sequence>? AND sequence<=? ORDER BY sequence LIMIT 256",
                    (cursor, descriptor["highWaterSequence"]),
                ).fetchall()
            if not rows:
                return
            for sequence, identifier in rows:
                cursor = int(sequence)
                yield str(identifier)

    def _read_row(self, identifier: str) -> Tuple[int, str, bytes, int]:
        with self._transaction() as connection:
            size = connection.execute(
                "SELECT sequence,LENGTH(packed) FROM paged_vector_rows "
                "WHERE vector_id=?",
                (identifier,),
            ).fetchone()
            if size is None:
                raise KeyError(identifier)
            self._reserve_memory(
                _READ_OVERHEAD_BYTES + 4 * int(size[1]) + self.dimensions,
                "paged vector row read",
            )
            row = connection.execute(
                "SELECT sequence,vector_id,packed,updates,row_sha256 "
                "FROM paged_vector_rows WHERE sequence=?",
                (size[0],),
            ).fetchone()
            if row is None:
                raise ValueError("paged vector row changed during read")
            return self._decode_row(row)

    def packed_row(self, key: str) -> bytes:
        return self._read_row(_identifier(key))[2]

    def update_count(self, key: str) -> int:
        return self._read_row(_identifier(key))[3]

    def levels(self, key: str) -> torch.Tensor:
        identifier = _identifier(key)
        with self._transaction() as connection:
            revision = self.revision_in_connection(connection)
            with self._cache_lock:
                self._cache_clear_if_revision_changed(revision)
                cached = self._cache.get(identifier)
                if cached is not None:
                    self._cache.move_to_end(identifier)
                    return cached.clone()
            size = connection.execute(
                "SELECT sequence,LENGTH(packed) FROM paged_vector_rows "
                "WHERE vector_id=?",
                (identifier,),
            ).fetchone()
            if size is None:
                raise KeyError(identifier)
            self._reserve_memory(
                _READ_OVERHEAD_BYTES + 4 * int(size[1]) + self.dimensions,
                "paged vector decode",
            )
            row = connection.execute(
                "SELECT sequence,vector_id,packed,updates,row_sha256 "
                "FROM paged_vector_rows WHERE sequence=?",
                (size[0],),
            ).fetchone()
            if row is None:
                raise ValueError("paged vector row changed during decode")
            _sequence, _identifier_value, packed, _updates = self._decode_row(row)
            levels = _decode_exact_row(packed, self.dimensions)
            with self._cache_lock:
                self._cache_clear_if_revision_changed(revision)
                self._cache_put(identifier, levels)
            return levels.clone()

    def __getitem__(self, key: str) -> torch.Tensor:
        return _activity(self.levels(key))

    @staticmethod
    def write_packed_in_connection(
        connection: sqlite3.Connection,
        identifier: str,
        packed: bytes,
        *,
        dimensions: int,
        updates: int = 0,
        reserve_disk: Optional[Callable[[int, str], Any]] = None,
        preserve_counter_if_same: bool = False,
    ) -> Tuple[bool, int]:
        """Shared-DB atomic primitive used by metadata admission as well."""

        identifier = _identifier(identifier)
        updates = _counter(updates)
        _validate_packed_row(packed, dimensions)
        existing = connection.execute(
            "SELECT sequence,vector_id,packed,updates,row_sha256 "
            "FROM paged_vector_rows WHERE vector_id=?",
            (identifier,),
        ).fetchone()
        if existing is not None:
            _decoded_row(existing, dimensions)
            if preserve_counter_if_same and existing[2] == packed:
                return False, PagedPackedVectors.revision_in_connection(connection)
            if existing[2] == packed and existing[3] == updates:
                return False, PagedPackedVectors.revision_in_connection(connection)
        if reserve_disk is not None:
            PagedPackedVectors._reserve(
                reserve_disk,
                _WRITE_OVERHEAD_BYTES + 4 * len(packed) + 2 * len(identifier),
                "paged vector upsert",
            )
        digest = _row_sha(packed, updates)
        if existing is None:
            connection.execute(
                "INSERT INTO paged_vector_rows "
                "(vector_id,packed,updates,row_sha256,updated_at) "
                "VALUES (?,?,?,?,?)",
                (identifier, packed, updates, digest, time.time()),
            )
        else:
            connection.execute(
                "UPDATE paged_vector_rows SET packed=?,updates=?,"
                "row_sha256=?,updated_at=? WHERE vector_id=?",
                (packed, updates, digest, time.time(), identifier),
            )
        return True, PagedPackedVectors._bump_revision(connection)

    def _write_packed_in_transaction(
        self,
        connection: sqlite3.Connection,
        identifier: str,
        packed: bytes,
        *,
        updates: int = 0,
    ) -> Tuple[bool, int]:
        return self.write_packed_in_connection(
            connection, identifier, packed,
            dimensions=self.dimensions,
            updates=updates,
            reserve_disk=self._disk_reserve,
        )

    def set_packed(self, key: str, packed: bytes) -> None:
        identifier = _identifier(key)
        if not isinstance(packed, bytes):
            raise ValueError("paged vector packed rows must be bytes")
        _validate_packed_row(packed, self.dimensions)
        self._reserve_memory(
            _READ_OVERHEAD_BYTES + 4 * len(packed) + self.dimensions,
            "paged vector write buffer",
        )
        with self._transaction(write=True) as connection:
            _changed, revision = self._write_packed_in_transaction(
                connection, identifier, packed
            )
        self.invalidate(identifier, revision)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        levels = quantize_vsa_vector(
            value, self.dimensions, zero_deadband=self.zero_deadband
        )
        self.set_packed(key, _pack_exact_row(levels, self.dimensions))

    def __delitem__(self, key: str) -> None:
        identifier = _identifier(key)
        with self._transaction(write=True) as connection:
            existing = connection.execute(
                "SELECT sequence,vector_id,packed,updates,row_sha256 "
                "FROM paged_vector_rows WHERE vector_id=?",
                (identifier,),
            ).fetchone()
            if existing is None:
                raise KeyError(identifier)
            self._decode_row(existing)
            self._reserve_disk(_WRITE_OVERHEAD_BYTES, "paged vector delete")
            connection.execute(
                "DELETE FROM paged_vector_rows WHERE vector_id=?", (identifier,)
            )
            revision = self._bump_revision(connection)
        self.invalidate(identifier, revision)

    def adapt(self, key: str, target: torch.Tensor, rate: float) -> torch.Tensor:
        identifier = _identifier(key)
        alpha = _valid_rate(rate)
        desired = quantize_vsa_vector(
            target, self.dimensions, zero_deadband=self.zero_deadband
        )
        with self._transaction(write=True) as connection:
            levels, revision = self._adapt_in_transaction(
                connection, identifier, desired, alpha
            )
        self.invalidate(identifier, revision)
        self._cache_put(identifier, levels)
        return _activity(levels)

    def _adapt_in_transaction(
        self,
        connection: sqlite3.Connection,
        identifier: str,
        desired: torch.Tensor,
        alpha: float,
    ) -> Tuple[torch.Tensor, int]:
        existing = connection.execute(
            "SELECT sequence,vector_id,packed,updates,row_sha256 "
            "FROM paged_vector_rows WHERE vector_id=?",
            (identifier,),
        ).fetchone()
        if existing is None:
            raise KeyError(identifier)
        _sequence, _identifier_value, packed, updates = self._decode_row(existing)
        previous = _decode_exact_row(packed, self.dimensions)
        if alpha == 0.0 or torch.equal(previous, desired):
            return previous, self.revision_in_connection(connection)
        if updates >= _MAX_UPDATES:
            raise OverflowError("paged vector update counter is exhausted")
        digest = hashlib.sha256(
            json.dumps(
                [self.seed, identifier, updates],
                ensure_ascii=False, separators=(",", ":"),
            ).encode("utf-8")
        ).digest()
        state = int.from_bytes(digest[:8], "little")
        changed: list[int] = []
        for old, goal in zip(previous.tolist(), desired.tolist()):
            state, draw = _splitmix64(state)
            expectation = old + alpha * (goal - old)
            lower = math.floor(expectation)
            changed.append(lower + int(draw < expectation - lower))
        levels = torch.tensor(changed, dtype=torch.int8)
        next_packed = _pack_exact_row(levels, self.dimensions)
        _changed, revision = self._write_packed_in_transaction(
            connection, identifier, next_packed, updates=updates + 1
        )
        return levels, revision

    def batch(
        self, *, max_rows: int = 256, max_payload_bytes: int = 8 * 1024 * 1024
    ) -> "PagedVectorBatch":
        """Open one bounded WAL transaction for multiple exact mutations.

        The batch never truncates input; exceeding either operational window
        aborts and rolls back. Callers checkpoint their source after the
        context exits successfully. Standalone set/adapt remain one fsync per
        mutation and should not be used for large-corpus throughput.
        """

        return PagedVectorBatch(
            self, max_rows=max_rows, max_payload_bytes=max_payload_bytes
        )

    def decode_rows(self, keys: Sequence[str]) -> torch.Tensor:
        if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
            raise ValueError("paged vector decode request must be a sequence")
        if len(keys) > MAX_DECODE_ROWS:
            raise ValueError("paged vector decode exceeds bounded window")
        if not keys:
            return torch.empty((0, self.dimensions), dtype=torch.float32)
        return torch.stack([self[key] for key in keys])

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for identifier in self:
            yield identifier, self[identifier]

    def values(self) -> Iterator[torch.Tensor]:
        for identifier in self:
            yield self[identifier]

    def export_state(
        self, *, prefix: str = "", keys: Optional[Sequence[str]] = None
    ) -> Tuple[dict[str, Any], dict[str, torch.Tensor]]:
        """Export one bounded shard in PackedTernaryVectors' exact format.

        Explicit ``keys`` is required above ``MAX_EXPORT_ROWS``; whole-map
        materialization is never a fallback for a large paged substrate.
        """

        if not isinstance(prefix, str):
            raise ValueError("packed vector tensor prefix must be a string")
        if keys is not None:
            if isinstance(keys, (str, bytes)) or not isinstance(keys, Sequence):
                raise ValueError("packed vector export keys must be a sequence")
            ids = list(keys)
            if (
                len(ids) > MAX_EXPORT_ROWS
                or any(not isinstance(identifier, str) or not identifier
                       for identifier in ids)
                or len(set(ids)) != len(ids)
            ):
                raise ValueError("packed vector export keys are invalid or unbounded")
        else:
            ids = None
        with self._transaction() as connection:
            if ids is None:
                count = int(connection.execute(
                    "SELECT COUNT(*) FROM paged_vector_rows"
                ).fetchone()[0])
                if count > MAX_EXPORT_ROWS:
                    raise ValueError("paged vector full export exceeds bounded shard window")
                ids = [str(row[0]) for row in connection.execute(
                    "SELECT vector_id FROM paged_vector_rows ORDER BY sequence"
                ).fetchall()]
            self._reserve_memory(
                _READ_OVERHEAD_BYTES
                + 4 * len(ids) * (self.row_bytes + 8 + 96),
                "paged vector shard export",
            )
            rows = bytearray(len(ids) * self.row_bytes)
            counters = bytearray(len(ids) * 8)
            for index, identifier in enumerate(ids):
                value = connection.execute(
                    "SELECT sequence,vector_id,packed,updates,row_sha256 "
                    "FROM paged_vector_rows WHERE vector_id=?",
                    (identifier,),
                ).fetchone()
                if value is None:
                    raise ValueError("packed vector export key is missing")
                _sequence, _id, packed, updates = self._decode_row(value)
                rows[index * self.row_bytes : (index + 1) * self.row_bytes] = packed
                struct.pack_into("<Q", counters, index * 8, updates)
        packed_tensor = (
            torch.frombuffer(rows, dtype=torch.uint8).reshape(len(ids), self.row_bytes)
            if rows else torch.empty((0, self.row_bytes), dtype=torch.uint8)
        )
        counter_tensor = (
            torch.frombuffer(counters, dtype=torch.uint8).reshape(len(ids), 8)
            if counters else torch.empty((0, 8), dtype=torch.uint8)
        )
        ids_bytes = json.dumps(
            ids, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        metadata = {
            "format": SHARD_FORMAT,
            "formatVersion": SHARD_FORMAT_VERSION,
            "dimensions": self.dimensions,
            "seed": self.seed,
            "zeroDeadband": self.zero_deadband,
            "rowCount": len(ids),
            "ids": ids,
            "idsSha256": _sha256(ids_bytes),
            "packedSha256": _sha256(rows),
            "countersSha256": _sha256(counters),
        }
        return metadata, {
            prefix + "packed_rows": packed_tensor,
            prefix + "update_counters_le": counter_tensor,
        }

    def import_state(
        self,
        metadata: Mapping[str, Any],
        tensors: Mapping[str, torch.Tensor],
        *,
        prefix: str = "",
    ) -> int:
        """Append one verified bounded shard to a fresh/rebuilding cache.

        Repeat in deterministic shard order. Duplicate IDs fail closed; this
        is not a last-wins replay. A separate recovery coordinator must first
        choose a fresh cache database and verify the authoritative v3 shard
        generation. This method never promotes SQLite to checkpoint authority.
        """

        shard = PackedTernaryVectors.from_state(
            metadata, tensors, prefix=prefix
        )
        if (
            len(shard) > MAX_EXPORT_ROWS
            or shard.dimensions != self.dimensions
            or shard.seed != self.seed
            or shard.zero_deadband != self.zero_deadband
        ):
            raise ValueError("paged vector shard is too large or incompatible")
        if not shard:
            return 0
        estimated = _WRITE_OVERHEAD_BYTES + len(shard) * (
            4 * self.row_bytes + 8 + 128
        )
        self._reserve_disk(estimated, "paged vector shard import")
        self._reserve_memory(
            _READ_OVERHEAD_BYTES + len(shard) * (
                self.row_bytes + 8 + 128
            ),
            "paged vector shard import buffer",
        )
        with self._transaction(write=True) as connection:
            for identifier in shard:
                if connection.execute(
                    "SELECT 1 FROM paged_vector_rows WHERE vector_id=?",
                    (identifier,),
                ).fetchone() is not None:
                    raise ValueError("paged vector shard contains a duplicate ID")
                self.write_packed_in_connection(
                    connection,
                    identifier,
                    shard.packed_row(identifier),
                    dimensions=self.dimensions,
                    updates=shard.update_count(identifier),
                    reserve_disk=self._disk_reserve,
                )
            revision = self.revision_in_connection(connection)
        self.invalidate_all(revision)
        return len(shard)

    def current_snapshot(self) -> str:
        with self._transaction() as connection:
            count = row_count(connection, "paged_vector_rows")
            high_water = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM paged_vector_rows"
            ).fetchone()[0]
            revision = self.revision_in_connection(connection)
            store_id = self.store_id_in_connection(connection)
        return _token({
            "format": _SNAPSHOT_FORMAT,
            "formatVersion": FORMAT_VERSION,
            "storeId": store_id,
            "dimensions": self.dimensions,
            "revision": revision,
            "rowCount": int(count),
            "highWaterSequence": int(high_water),
        })

    def _decode_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        value = _untoken(snapshot_id, "paged vector snapshot")
        if (
            set(value) != {
                "format", "formatVersion", "storeId", "dimensions",
                "revision", "rowCount", "highWaterSequence",
            }
            or value.get("format") != _SNAPSHOT_FORMAT
            or value.get("formatVersion") != FORMAT_VERSION
            or not isinstance(value.get("storeId"), str)
            or not value["storeId"]
            or value.get("dimensions") != self.dimensions
            or any(
                isinstance(value.get(field), bool)
                or not isinstance(value.get(field), int)
                or value[field] < 0
                for field in ("revision", "rowCount", "highWaterSequence")
            )
        ):
            raise ValueError("paged vector snapshot is invalid")
        return value

    @staticmethod
    def _page_cursor(snapshot_id: str, after: int, seen: int) -> str:
        return _token({
            "format": _CURSOR_FORMAT,
            "formatVersion": FORMAT_VERSION,
            "snapshotSha256": _sha256(snapshot_id.encode("ascii")),
            "afterSequence": after,
            "rowsSeen": seen,
        })

    @staticmethod
    def _decode_page_cursor(
        cursor: str, snapshot_id: str, high_water: int, count: int
    ) -> Tuple[int, int]:
        value = _untoken(cursor, "paged vector cursor")
        if (
            set(value) != {
                "format", "formatVersion", "snapshotSha256",
                "afterSequence", "rowsSeen",
            }
            or value.get("format") != _CURSOR_FORMAT
            or value.get("formatVersion") != FORMAT_VERSION
            or value.get("snapshotSha256") != _sha256(snapshot_id.encode("ascii"))
            or any(
                isinstance(value.get(field), bool)
                or not isinstance(value.get(field), int)
                or not 0 <= value[field] <= maximum
                for field, maximum in (
                    ("afterSequence", high_water),
                    ("rowsSeen", count),
                )
            )
        ):
            raise ValueError("paged vector cursor is invalid")
        return value["afterSequence"], value["rowsSeen"]

    def page_rows(
        self, snapshot_id: str, cursor: Optional[str], page_size: int
    ) -> PackedVectorPage:
        if (
            isinstance(page_size, bool) or not isinstance(page_size, int)
            or not 1 <= page_size <= MAX_PAGE_ROWS
        ):
            raise ValueError("paged vector page size is invalid")
        snapshot = self._decode_snapshot(snapshot_id)
        high_water = snapshot["highWaterSequence"]
        after, seen = (
            self._decode_page_cursor(
                cursor, snapshot_id, high_water, snapshot["rowCount"]
            ) if cursor is not None else (0, 0)
        )
        with self._transaction() as connection:
            if (
                self.store_id_in_connection(connection) != snapshot["storeId"]
                or self.revision_in_connection(connection) != snapshot["revision"]
            ):
                raise ValueError("paged vector snapshot generation drift")
            sizes = connection.execute(
                "SELECT sequence,LENGTH(packed) FROM paged_vector_rows "
                "WHERE sequence>? AND sequence<=? ORDER BY sequence LIMIT ?",
                (after, high_water, page_size + 1),
            ).fetchall()
            has_more = len(sizes) > page_size
            selected = sizes[:page_size]
            self._reserve_memory(
                _READ_OVERHEAD_BYTES * max(1, len(selected))
                + 4 * sum(int(size) for _, size in selected),
                "paged vector page",
            )
            if selected:
                last = int(selected[-1][0])
                raw = connection.execute(
                    "SELECT sequence,vector_id,packed,updates,row_sha256 "
                    "FROM paged_vector_rows WHERE sequence>? AND sequence<=? "
                    "ORDER BY sequence",
                    (after, last),
                ).fetchall()
                if [row[0] for row in raw] != [row[0] for row in selected]:
                    raise ValueError("paged vector page changed during read")
            else:
                last = after
                raw = []
            rows = tuple(
                PackedVectorRow(sequence, identifier, packed)
                for sequence, identifier, packed, _updates
                in (self._decode_row(row) for row in raw)
            )
            next_seen = seen + len(rows)
            if (
                next_seen > snapshot["rowCount"]
                or (not has_more and next_seen != snapshot["rowCount"])
            ):
                raise ValueError("paged vector snapshot coverage mismatch")
            return PackedVectorPage(
                rows=rows,
                next_cursor=self._page_cursor(snapshot_id, last, next_seen),
                has_more=has_more,
                snapshot_id=snapshot_id,
            )

    def status(self) -> dict[str, Any]:
        with self._transaction() as connection:
            count = row_count(connection, "paged_vector_rows")
            high_water = connection.execute(
                "SELECT COALESCE(MAX(sequence),0) FROM paged_vector_rows"
            ).fetchone()[0]
            store_id = self.store_id_in_connection(connection)
            revision = self.revision_in_connection(connection)
            committed_generation, committed_revision = (
                self._committed_binding_in_connection(connection)
            )
            shares_assembly_index = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='assembly_records'"
            ).fetchone() is not None
        bytes_used = sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                Path(str(self.path) + "-wal"),
                Path(str(self.path) + "-shm"),
            ) if candidate.is_file()
        )
        return {
            "format": FORMAT,
            "formatVersion": FORMAT_VERSION,
            "storeId": store_id,
            "dimensions": self.dimensions,
            "seed": self.seed,
            "zeroDeadband": self.zero_deadband,
            "rowCount": int(count),
            "highWaterSequence": int(high_water),
            "revision": revision,
            "committedGenerationSha256": committed_generation,
            "committedRevision": committed_revision,
            "dirtySinceCommit": (
                committed_generation is None or committed_revision != revision
            ),
            "packedRowBytes": self.row_bytes,
            "logicalPackedBytes": int(count) * (self.row_bytes + 8),
            "cacheBytes": self._cache_bytes,
            "cacheCapacityBytes": self.cache_capacity_bytes,
            "storageBytes": bytes_used,
            "storageBytesIncludeAssemblyIndex": shares_assembly_index,
            "storage": "disk",
        }


class PagedVectorBatch:
    """One bounded multi-row SQLite transaction for low-write-amplification.

    It keeps no pending row dictionary: every operation mutates only the WAL
    transaction, which is committed once on successful context exit. Any
    exception or reserve/window refusal rolls the *whole* batch back.
    """

    def __init__(
        self,
        store: PagedPackedVectors,
        *,
        max_rows: int,
        max_payload_bytes: int,
    ) -> None:
        if (
            isinstance(max_rows, bool) or not isinstance(max_rows, int)
            or not 1 <= max_rows <= MAX_PAGE_ROWS
        ):
            raise ValueError("paged vector batch row window is invalid")
        if (
            isinstance(max_payload_bytes, bool)
            or not isinstance(max_payload_bytes, int)
            or not 1 <= max_payload_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("paged vector batch byte window is invalid")
        self.store = store
        self.max_rows = max_rows
        self.max_payload_bytes = max_payload_bytes
        self.rows_attempted = 0
        self.payload_bytes = 0
        self._connection: Optional[sqlite3.Connection] = None
        self._revision: Optional[int] = None
        self._aborted = False

    def __enter__(self) -> "PagedVectorBatch":
        if self._connection is not None:
            raise RuntimeError("paged vector batch is already open")
        self.store._reserve_disk(_WRITE_OVERHEAD_BYTES, "paged vector batch begin")
        connection = self.store._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            self._revision = self.store.revision_in_connection(connection)
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return self

    def _connection_or_raise(self) -> sqlite3.Connection:
        if self._connection is None or self._aborted:
            raise RuntimeError("paged vector batch is not active")
        return self._connection

    def _charge(self, identifier: str, packed_bytes: int) -> None:
        next_rows = self.rows_attempted + 1
        next_bytes = self.payload_bytes + 4 * packed_bytes + 2 * len(identifier) + 128
        if next_rows > self.max_rows or next_bytes > self.max_payload_bytes:
            self._aborted = True
            raise PagedVectorResourcePause(
                "paged vector batch window reached; retry in a new transaction"
            )
        self.rows_attempted = next_rows
        self.payload_bytes = next_bytes

    def set_packed(self, key: str, packed: bytes) -> bool:
        connection = self._connection_or_raise()
        try:
            identifier = _identifier(key)
            if not isinstance(packed, bytes):
                raise ValueError("paged vector packed rows must be bytes")
            _validate_packed_row(packed, self.store.dimensions)
            self._charge(identifier, len(packed))
            self.store._reserve_memory(
                _READ_OVERHEAD_BYTES + 4 * len(packed) + self.store.dimensions,
                "paged vector batch row buffer",
            )
            changed, self._revision = self.store._write_packed_in_transaction(
                connection, identifier, packed
            )
            return changed
        except BaseException:
            self._aborted = True
            raise

    def set(self, key: str, value: torch.Tensor) -> bool:
        self._connection_or_raise()
        try:
            levels = quantize_vsa_vector(
                value, self.store.dimensions,
                zero_deadband=self.store.zero_deadband,
            )
            return self.set_packed(
                key, _pack_exact_row(levels, self.store.dimensions)
            )
        except BaseException:
            self._aborted = True
            raise

    def adapt(self, key: str, target: torch.Tensor, rate: float) -> torch.Tensor:
        connection = self._connection_or_raise()
        try:
            identifier = _identifier(key)
            alpha = _valid_rate(rate)
            desired = quantize_vsa_vector(
                target, self.store.dimensions,
                zero_deadband=self.store.zero_deadband,
            )
            self._charge(identifier, self.store.row_bytes)
            self.store._reserve_memory(
                _READ_OVERHEAD_BYTES + 5 * self.store.row_bytes
                + self.store.dimensions * 2,
                "paged vector batch adaptation buffer",
            )
            levels, self._revision = self.store._adapt_in_transaction(
                connection, identifier, desired, alpha
            )
            return _activity(levels)
        except BaseException:
            self._aborted = True
            raise

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        connection = self._connection
        if connection is None:
            raise RuntimeError("paged vector batch was not opened")
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
        if committed:
            self.store.invalidate_all(self._revision)
        elif self._aborted and error_type is None:
            raise PagedVectorResourcePause("paged vector batch aborted")
        return False
