"""Atomic cooperating-process RAM/spill leases with durable file identities.

No OS RSS isolation or page pinning. RAM escrow closes the sampled-usage race
between cooperating allocations. File usage survives process death until the
backing path is proved absent; dev/inode identities deduplicate hardlinks.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional
from itertools import chain


_PROCESS_SESSION = uuid.uuid4().hex


class SharedQuotaPause(RuntimeError):
    def __init__(self, reason, status):
        super().__init__(reason)
        self.status = status


class SharedAllocationLease:
    def __init__(self, ledger, token, kind):
        self.ledger, self.token, self.kind = ledger, token, kind
        self.allocated = False

    def mark_allocated(self, actual_bytes: Optional[int] = None):
        self.ledger.commit_ram(self.token, actual_bytes)
        self.allocated = True
        return self

    def commit(self, actual_allocated_bytes: Optional[int] = None, *, path: Optional[Path] = None, promised_bytes: int = 0, retain_open_backing: bool = False):
        if self.kind == "ram":
            return self.mark_allocated(actual_allocated_bytes)
        if path is None:
            raise ValueError("spill lease requires an owned backing path")
        self.ledger.commit_spill(self.token, path, actual_allocated_bytes, promised_bytes, retain_open_backing)
        self.allocated = True
        return self

    def release(self):
        self.ledger.release(self.token)

    def bind_path(self, path: Path):
        self.ledger.bind_path(self.token, path)
        return self

    def backing_closed(self):
        self.ledger.backing_closed(self.token)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        # Marked RAM remains escrowed until a fresh verified sample observes
        # its allocation (including delayed allocator release). Disk survives
        # while its backing file exists, regardless of Python object lifetime.
        if not self.allocated:
            self.release()
        return False


class SharedResourceLedger:
    def __init__(self, path: Path, *, pid: Optional[int] = None, session: str = _PROCESS_SESSION,
                 process_alive: Optional[Callable[[int], bool]] = None, now_ns: Callable[[], int] = time.time_ns):
        self.path = Path(path).absolute()
        self.pid, self.session = os.getpid() if pid is None else pid, session
        self.process_alive = process_alive or self._process_alive
        self.now_ns = now_ns
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("resource ledger cannot be a symlink")
        with self._transaction() as db:
            schema = """
                CREATE TABLE IF NOT EXISTS owners(owner TEXT PRIMARY KEY, pool INTEGER NOT NULL, ram INTEGER, pid INTEGER NOT NULL, session TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS leases(token TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL, bytes INTEGER NOT NULL,
                    state TEXT NOT NULL, pid INTEGER NOT NULL, session TEXT NOT NULL, committed INTEGER, identity TEXT, path TEXT);
                CREATE TABLE IF NOT EXISTS files(identity TEXT PRIMARY KEY, bytes INTEGER NOT NULL, path TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS policy(key TEXT PRIMARY KEY, bytes INTEGER NOT NULL);
            """
            for statement in schema.split(";"):
                if statement.strip():
                    db.execute(statement)
            if "held" not in {row[1] for row in db.execute("PRAGMA table_info(leases)")}:
                db.execute("ALTER TABLE leases ADD COLUMN held INTEGER NOT NULL DEFAULT 0")
            db.execute("CREATE INDEX IF NOT EXISTS lease_state_kind ON leases(kind,state,committed)")
            db.execute("CREATE INDEX IF NOT EXISTS lease_identity ON leases(identity)")
            db.execute("INSERT OR IGNORE INTO policy SELECT 'spill_used',COALESCE(SUM(bytes),0) FROM files")

    @staticmethod
    def _process_alive(pid):
        try:
            os.kill(pid, 0)  # Existence check only, never a termination signal.
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(str(self.path), timeout=5.0)
        try:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def register_owner(self, owner: str, configured_pool_bytes: int, ram_budget_bytes: Optional[int] = None, *, preserve_existing_pool: bool = False):
        if not owner or type(configured_pool_bytes) is not int or configured_pool_bytes < 0:
            raise ValueError("shared resource owner/pool is invalid")
        if ram_budget_bytes is not None and (type(ram_budget_bytes) is not int or ram_budget_bytes < 0):
            raise ValueError("shared RAM ceiling is invalid")
        with self._transaction() as db:
            previous = db.execute("SELECT pool,ram FROM owners WHERE owner=?", (owner,)).fetchone()
            if preserve_existing_pool and previous:
                configured_pool_bytes = previous[0]
                if ram_budget_bytes is None:
                    ram_budget_bytes = previous[1]
            # A newly registered process incarnation may retire old pending
            # leases at this reused PID, but never file-backed disk usage.
            db.execute("DELETE FROM leases WHERE kind='ram' AND pid=? AND session<>?", (self.pid, self.session))
            db.execute("INSERT OR REPLACE INTO owners VALUES(?,?,?,?,?)", (owner, configured_pool_bytes, ram_budget_bytes, self.pid, self.session))

    def remove_owner(self, owner: str):
        with self._transaction() as db:
            db.execute("DELETE FROM owners WHERE owner=?", (owner,))
            # Deregistration is not proof that an in-flight allocation or
            # backing stopped existing. Leases leave through explicit release
            # or dead-process/absent-file reconciliation, never registry edits.

    def set_ram_ceiling(self, byte_count: Optional[int]):
        """Trusted main-selected global epoch; never supplied by renderer data."""
        with self._transaction() as db:
            if byte_count is None:
                db.execute("DELETE FROM policy WHERE key='ram'")
            elif type(byte_count) is int and byte_count >= 0:
                db.execute("INSERT OR REPLACE INTO policy VALUES('ram',?)", (byte_count,))
            else:
                raise ValueError("shared RAM ceiling is invalid")

    def _reconcile(self, db, *, observed_ns: Optional[int] = None, verified: bool = False, file_limit: int = 0, owner: Optional[str] = None, after_identity: str = ""):
        if verified and observed_ns is not None:
            db.execute("DELETE FROM leases WHERE kind='ram' AND state='allocated' AND committed<=?", (observed_ns,))
        for (pid,) in db.execute("SELECT DISTINCT pid FROM leases WHERE kind='ram'").fetchall():
            if not self.process_alive(pid):
                db.execute("DELETE FROM leases WHERE kind='ram' AND pid=?", (pid,))
        for (pid,) in db.execute("SELECT DISTINCT pid FROM leases WHERE kind='spill' AND held=1").fetchall():
            if not self.process_alive(pid):
                db.execute("UPDATE leases SET held=0 WHERE kind='spill' AND pid=?", (pid,))
        # Missing paths alone are not proof for a hardlinked identity: inspect
        # every registered alias before retiring the single charged backing.
        query = "SELECT identity,bytes,path FROM files WHERE identity>?"
        parameters = [after_identity]
        if owner is not None:
            query += " AND identity IN (SELECT identity FROM leases WHERE owner=?)"; parameters.append(owner)
        query += " ORDER BY identity LIMIT ?"; parameters.append(max(0, int(file_limit)))
        continuation = after_identity
        for identity, _bytes, path in db.execute(query, parameters):
            continuation = identity
            aliases = chain((row[0] for row in db.execute("SELECT path FROM leases WHERE identity=?", (identity,))), (path,))
            present = next((candidate for candidate in aliases if candidate and Path(candidate).exists()), None)
            held = db.execute("SELECT 1 FROM leases WHERE identity=? AND held=1 LIMIT 1", (identity,)).fetchone()
            if present is None and not held:
                self._bump_spill(db, -_bytes)
                db.execute("DELETE FROM files WHERE identity=?", (identity,))
                db.execute("DELETE FROM leases WHERE identity=?", (identity,))
            elif present is not None and present != path:
                db.execute("UPDATE files SET path=? WHERE identity=?", (present, identity))
        for token, pid, path, promised in db.execute("SELECT token,pid,path,bytes FROM leases WHERE kind='spill' AND state='pending'").fetchall():
            if not self.process_alive(pid):
                if path and Path(path).exists():
                    self._adopt_pending_file(db, token, Path(path), promised)
                else:
                    db.execute("DELETE FROM leases WHERE token=?", (token,))
        return continuation

    def reconcile_owner(self, owner: Optional[str] = None, *, max_entries: int = 256, after_identity: str = ""):
        with self._transaction() as db:
            continuation = self._reconcile(db, owner=owner, file_limit=max_entries, after_identity=after_identity)
            return {**self._status(db), "nextIdentity": continuation if continuation != after_identity else None}

    @staticmethod
    def _adopt_pending_file(db, token, path, promised):
        stat = path.stat()
        identity = "%d:%d" % (stat.st_dev, stat.st_ino)
        allocated = max(stat.st_size, int(getattr(stat, "st_blocks", 0)) * 512, promised)
        prior = db.execute("SELECT bytes FROM files WHERE identity=?", (identity,)).fetchone()
        previous = prior[0] if prior else 0
        SharedResourceLedger._bump_spill(db, max(allocated, previous) - previous)
        db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?)", (identity, max(allocated, previous), str(path)))
        db.execute("UPDATE leases SET state='allocated',bytes=0,identity=? WHERE token=?", (identity, token))

    def _status(self, db):
        pool = int(db.execute("SELECT COALESCE(MAX(pool),0) FROM owners").fetchone()[0])
        disk = int(db.execute("SELECT bytes FROM policy WHERE key='spill_used'").fetchone()[0])
        pending_disk = int(db.execute("SELECT COALESCE(SUM(bytes),0) FROM leases WHERE kind='spill' AND state='pending'").fetchone()[0])
        ram = int(db.execute("SELECT COALESCE(SUM(bytes),0) FROM leases WHERE kind='ram'").fetchone()[0])
        root_ram = db.execute("SELECT bytes FROM policy WHERE key='ram'").fetchone()
        return {"largestConfiguredPoolBytes": pool, "physicalSpillUsageBytes": disk, "pendingSpillBytes": pending_disk,
            "ramEscrowBytes": ram, "globalRamCeilingBytes": root_ram[0] if root_ram else None,
            "spillQuotaOvercommitted": disk + pending_disk > pool,
            "atomicReservations": True, "hardRssIsolation": False,
            "spillUsageBasis": "physical-blocks-or-larger-mutable-sparse-promise",
            "fileIdentityBasis": "device-inode; unverified-COW-sharing-counted-conservatively"}

    def status(self, *, observed_ns: Optional[int] = None, verified: bool = False):
        with self._transaction() as db:
            self._reconcile(db, observed_ns=observed_ns, verified=verified)
            return self._status(db)

    def reserve(self, owner: str, kind: str, byte_count: int, *, ram_budget_bytes: Optional[int] = None,
                observed_ram_bytes: Optional[int] = None, observed_ns: Optional[int] = None, verified: bool = False):
        if kind not in {"ram", "spill"} or type(byte_count) is not int or byte_count < 0:
            raise ValueError("invalid shared resource reservation")
        token = uuid.uuid4().hex
        with self._transaction() as db:
            self._reconcile(db, observed_ns=observed_ns, verified=verified)
            status = self._status(db)
            if db.execute("SELECT 1 FROM owners WHERE owner=?", (owner,)).fetchone() is None:
                raise SharedQuotaPause("shared resource owner is not registered", status)
            if kind == "ram":
                if not verified or observed_ram_bytes is None or ram_budget_bytes is None:
                    raise SharedQuotaPause("shared RAM admission has no verified residency/ceiling", status)
                limit = min(ram_budget_bytes, status["globalRamCeilingBytes"]) if status["globalRamCeilingBytes"] is not None else ram_budget_bytes
                if observed_ram_bytes + status["ramEscrowBytes"] + byte_count > limit:
                    raise SharedQuotaPause("atomic shared RAM reservation exceeds selected ceiling", status)
            elif status["physicalSpillUsageBytes"] + status["pendingSpillBytes"] + byte_count > status["largestConfiguredPoolBytes"]:
                raise SharedQuotaPause("atomic shared spill reservation exceeds largest configured pool", status)
            db.execute("INSERT INTO leases(token,owner,kind,bytes,state,pid,session,committed,identity,path) VALUES(?,?,?,?,?,?,?,?,?,?)", (token, owner, kind, byte_count, "pending", self.pid, self.session, None, None, None))
        return SharedAllocationLease(self, token, kind)

    def commit_ram(self, token: str, actual_bytes: Optional[int]):
        with self._transaction() as db:
            row = db.execute("SELECT bytes FROM leases WHERE token=? AND kind='ram'", (token,)).fetchone()
            if row is None:
                return
            if actual_bytes is not None and (type(actual_bytes) is not int or not 0 <= actual_bytes <= row[0]):
                raise ValueError("RAM allocation cannot exceed its atomic reservation")
            db.execute("UPDATE leases SET state='allocated',bytes=?,committed=? WHERE token=?", (row[0] if actual_bytes is None else actual_bytes, self.now_ns(), token))

    def bind_path(self, token: str, path: Path):
        with self._transaction() as db:
            db.execute("UPDATE leases SET path=? WHERE token=? AND kind='spill' AND state='pending'", (str(Path(path).absolute()), token))

    def commit_spill(self, token: str, path: Path, actual_bytes: Optional[int], promised_bytes: int, retain_open_backing: bool = False):
        path = Path(path).absolute()
        stat = path.stat()
        if path.is_symlink() or not path.is_file():
            raise ValueError("spill accounting requires a regular owned backing")
        identity = "%d:%d" % (stat.st_dev, stat.st_ino)
        measured = int(stat.st_blocks) * 512 if hasattr(stat, "st_blocks") else stat.st_size
        allocated = max(measured, int(actual_bytes or 0))
        allocated = max(allocated, int(promised_bytes))
        with self._transaction() as db:
            row = db.execute("SELECT bytes FROM leases WHERE token=? AND kind='spill'", (token,)).fetchone()
            if row is None:
                raise ValueError("spill reservation is absent")
            previous = db.execute("SELECT bytes FROM files WHERE identity=?", (identity,)).fetchone()
            old = previous[0] if previous else 0
            status = self._status(db)
            total = status["physicalSpillUsageBytes"] + status["pendingSpillBytes"] - row[0] + max(old, allocated) - old
            if total > status["largestConfiguredPoolBytes"]:
                raise SharedQuotaPause("actual spill allocation exceeds atomic configured pool", status)
            self._bump_spill(db, max(old, allocated) - old)
            db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?)", (identity, max(old, allocated), str(path)))
            db.execute("UPDATE leases SET state='allocated',bytes=0,identity=?,path=?,held=? WHERE token=?", (identity, str(path), int(retain_open_backing), token))
            # An append journal can have millions of tile writes. Once the
            # SAME path/inode is proved, supersede only this process/session's
            # non-held aliases; preserve other paths, owners and mmap holders.
            db.execute("DELETE FROM leases WHERE kind='spill' AND state='allocated' AND identity=? AND path=? "
                "AND pid=? AND session=? AND owner=(SELECT owner FROM leases WHERE token=?) AND held=0 AND token<>?",
                (identity, str(path), self.pid, self.session, token, token))

    def backing_closed(self, token: str):
        with self._transaction() as db:
            db.execute("UPDATE leases SET held=0 WHERE token=?", (token,))
        self.release(token)

    @staticmethod
    def _bump_spill(db, delta):
        db.execute("UPDATE policy SET bytes=bytes+? WHERE key='spill_used'", (int(delta),))

    def release(self, token: str):
        with self._transaction() as db:
            row = db.execute("SELECT kind,state,path,identity FROM leases WHERE token=?", (token,)).fetchone()
            if row is None:
                return
            if row[1] == "pending":
                if row[0] == "spill" and row[2] and Path(row[2]).exists():
                    promised = db.execute("SELECT bytes FROM leases WHERE token=?", (token,)).fetchone()[0]
                    self._adopt_pending_file(db, token, Path(row[2]), promised)
                else:
                    db.execute("DELETE FROM leases WHERE token=?", (token,))
            elif row[0] == "spill" and row[2] and not Path(row[2]).exists():
                # Keep aliases for extant hardlinks; reconcile removes usage
                # only after every registered backing name has disappeared.
                aliases = chain((item[0] for item in db.execute("SELECT path FROM leases WHERE identity=?", (row[3],))),
                    (item[0] for item in db.execute("SELECT path FROM files WHERE identity=?", (row[3],))))
                held = db.execute("SELECT 1 FROM leases WHERE identity=? AND held=1 LIMIT 1", (row[3],)).fetchone()
                if not held and not any(path and Path(path).exists() for path in aliases):
                    count = db.execute("SELECT bytes FROM files WHERE identity=?", (row[3],)).fetchone()
                    if count:
                        self._bump_spill(db, -count[0])
                    db.execute("DELETE FROM files WHERE identity=?", (row[3],))
                    db.execute("DELETE FROM leases WHERE identity=?", (row[3],))
