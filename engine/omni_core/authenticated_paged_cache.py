"""Process-authenticated, disk-backed proofs for derived paged caches.

Only a currently verified process session owns the authentication key. No key
is written to disk and no proof from a prior process is accepted as authority.
Cold recovery rebuilds from actual committed blob hashes. Proof population is
disk-backed, not a fixed-entry Python LRU; SQLite windows and each optional
write are resource guarded. A refused/invalid proof falls back to real hashes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Iterator, Optional


_ACTIVE_SESSION: ContextVar[Optional["AuthenticatedCacheSession"]] = ContextVar(
    "omni_authenticated_blob_proof_session", default=None
)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class AuthenticatedCacheSession:
    """One small key/owner handle; corpus-sized proof rows stay in SQLite."""

    def __init__(self, owner: Any, *, disk_reserve: Optional[Callable[[int, str], Any]] = None) -> None:
        self.owner = owner
        self.disk_reserve = disk_reserve
        self._secret = os.urandom(32)
        with owner._transaction() as connection:
            row = connection.execute(
                "SELECT value FROM index_metadata WHERE key='store_id'"
            ).fetchone()
        if row is None or not isinstance(row[0], str) or not row[0]:
            raise ValueError("authenticated cache has no paged store identity")
        self.store_id = row[0]
        self.proof_hits = 0
        self.proof_misses = 0
        self.proof_writes = 0
        self.proof_write_pauses = 0
        self._proof_schema_ready = False
        # This is a per-connection page window, never an entry/cardinality cap.
        self.sqlite_window_bytes = self._memory_window()

    def _memory_window(self) -> int:
        budget = 256 * 1024
        policy = getattr(getattr(self.owner, "_disk_reserve", None), "__self__", None)
        status = getattr(policy, "status", None)
        if callable(status):
            readings = status()
            for key in ("availableMemoryBytes", "systemRamBudgetBytes"):
                value = readings.get(key) if isinstance(readings, dict) else None
                if type(value) is int and value > 0:
                    budget = min(budget, max(4096, value // 128))
        return max(4096, budget)

    @property
    def path(self) -> Path:
        # The same owner is retargeted after private cache-directory publication.
        return Path(self.owner.path)

    def reserve_disk(self, size: int, operation: str) -> None:
        callback = self.disk_reserve
        if callback is not None:
            if callback(max(1, int(size)), operation) is False:
                raise RuntimeError("%s paused at disk reserve" % operation)
        else:
            self.owner._reserve_disk(size, operation)

    @contextmanager
    def transaction(self, *, write: bool = False) -> Iterator[Any]:
        self.owner._reserve_memory(self.sqlite_window_bytes + 4096, "authenticated SQLite page window")
        with self.owner._transaction(write=write) as connection:
            connection.execute("PRAGMA cache_size=-%d" % max(4, self.sqlite_window_bytes // 1024))
            connection.execute("PRAGMA mmap_size=0")
            yield connection

    def sign(self, domain: str, value: Any) -> str:
        return hmac.new(self._secret,
                        domain.encode("ascii") + b"\0" + self.store_id.encode("utf-8")
                        + b"\0" + canonical(value), hashlib.sha256).hexdigest()

    def authentic(self, domain: str, value: Any, observed: Any) -> bool:
        return isinstance(observed, str) and hmac.compare_digest(self.sign(domain, value), observed)

    def _ensure_proofs(self) -> None:
        if self._proof_schema_ready:
            return
        self.reserve_disk(65536, "authenticated file proof schema")
        with self.transaction(write=True) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS authenticated_blob_proofs ("
                "path TEXT PRIMARY KEY,identity_json BLOB NOT NULL,sha256 TEXT NOT NULL,"
                "authentication TEXT NOT NULL) WITHOUT ROWID"
            )
        self._proof_schema_ready = True

    def lookup_blob(self, path: Path, identity: tuple[int, ...]) -> Optional[str]:
        try:
            path = Path(path).resolve()
            self._ensure_proofs()
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT identity_json,sha256,authentication FROM authenticated_blob_proofs "
                    "WHERE path=?", (str(path),),
                ).fetchone()
            if row is not None and (
                isinstance(row[0], bytes) and row[0] == canonical(identity)
                and isinstance(row[1], str) and len(row[1]) == 64
                and all(char in "0123456789abcdef" for char in row[1])
                and self.authentic("immutable-file-proof-v1",
                                   [str(path), list(identity), row[1]], row[2])
            ):
                self.proof_hits += 1
                return row[1]
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
            # Optional proof storage never certifies a failed or unknown row.
            self.proof_write_pauses += 1
        self.proof_misses += 1
        return None

    def remember_blob(self, path: Path, checksum: str, identity: tuple[int, ...]) -> bool:
        try:
            path = Path(path).resolve()
            self._ensure_proofs()
            encoded = canonical(identity)
            self.reserve_disk(4096 + 4 * (len(encoded) + len(str(path))),
                              "authenticated immutable file proof")
            authentication = self.sign("immutable-file-proof-v1", [str(path), list(identity), checksum])
            with self.transaction(write=True) as connection:
                connection.execute(
                    "INSERT INTO authenticated_blob_proofs(path,identity_json,sha256,authentication) "
                    "VALUES(?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                    "identity_json=excluded.identity_json,sha256=excluded.sha256,"
                    "authentication=excluded.authentication",
                    (str(path), encoded, checksum, authentication),
                )
            self.proof_writes += 1
            return True
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
            self.proof_write_pauses += 1
            return False

    @contextmanager
    def blob_scope(self) -> Iterator["AuthenticatedCacheSession"]:
        token = _ACTIVE_SESSION.set(self)
        try:
            yield self
        finally:
            _ACTIVE_SESSION.reset(token)

    def prune_missing_blob_proofs(self, store_root: Path) -> None:
        """Reclaim only proofs for already removed immutable files, not blobs."""

        try:
            self._ensure_proofs()
            store = Path(store_root).resolve()
            after = ""
            while True:
                with self.transaction() as connection:
                    row = connection.execute(
                        "SELECT path FROM authenticated_blob_proofs WHERE path>? ORDER BY path LIMIT 1",
                        (after,),
                    ).fetchone()
                if row is None:
                    return
                after = row[0]
                path = Path(after)
                try:
                    relative = path.relative_to(store)
                except ValueError:
                    continue
                if relative.parts and relative.parts[0] in {"blobs", "generations"} and not path.exists():
                    with self.transaction(write=True) as connection:
                        connection.execute("DELETE FROM authenticated_blob_proofs WHERE path=?", (after,))
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
            self.proof_write_pauses += 1

    def forget_blob(self, path: Path) -> None:
        """Forget one proof after the existing authoritative GC removed a file."""

        try:
            self._ensure_proofs()
            with self.transaction(write=True) as connection:
                connection.execute("DELETE FROM authenticated_blob_proofs WHERE path=?",
                                   (str(Path(path).resolve()),))
        except (OSError, RuntimeError, ValueError, sqlite3.DatabaseError):
            self.proof_write_pauses += 1


def cache_session(owner: Any, *, disk_reserve: Optional[Callable[[int, str], Any]] = None) -> AuthenticatedCacheSession:
    value = getattr(owner, "_authenticated_cache_session", None)
    if not isinstance(value, AuthenticatedCacheSession) or value.owner is not owner:
        value = AuthenticatedCacheSession(owner, disk_reserve=disk_reserve)
        owner._authenticated_cache_session = value
    elif disk_reserve is not None:
        value.disk_reserve = disk_reserve
    return value


def active_blob_session() -> Optional[AuthenticatedCacheSession]:
    return _ACTIVE_SESSION.get()
