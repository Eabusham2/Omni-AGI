"""Atomic cooperating-process RAM/spill leases with durable file identities.

No OS RSS isolation or page pinning. RAM escrow closes the sampled-usage race
between cooperating allocations. File usage survives process death until the
backing path is proved absent; dev/inode identities deduplicate hardlinks.
"""

from __future__ import annotations

import os
import stat as stat_module
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
        self._path = None
        self._physical = None
        self._promised = 0
        self._held = False

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
        self._path, self._promised, self._held = Path(path), promised_bytes, retain_open_backing
        value = self._path.stat()
        self._physical = int(value.st_blocks) * 512 if hasattr(value, "st_blocks") else value.st_size
        return self

    def refresh_physical(self):
        if self._path is None:
            return
        value = self._path.stat()
        physical = int(value.st_blocks) * 512 if hasattr(value, "st_blocks") else value.st_size
        if physical != self._physical:
            self.commit(path=self._path, promised_bytes=self._promised, retain_open_backing=self._held)

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
            for name in ("pool_id", "pool_group"):
                if name not in {row[1] for row in db.execute("PRAGMA table_info(leases)")}:
                    db.execute("ALTER TABLE leases ADD COLUMN %s TEXT" % name)
            file_columns = {row[1] for row in db.execute("PRAGMA table_info(files)")}
            if "pool_group" not in file_columns:
                db.execute("ALTER TABLE files ADD COLUMN pool_group TEXT")
            if "physical" not in file_columns:
                # Unknown historical sparse promises remain conservative until
                # the exact existing backing is observed again.
                db.execute("ALTER TABLE files ADD COLUMN physical INTEGER NOT NULL DEFAULT 0")
            db.execute("CREATE TABLE IF NOT EXISTS drive_pools(pool_id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, backing_path TEXT NOT NULL, device TEXT NOT NULL, inode TEXT NOT NULL, backing_device TEXT NOT NULL, backing_inode TEXT NOT NULL, physical_group TEXT NOT NULL, rate INTEGER NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS drive_access(owner TEXT NOT NULL,pool_id TEXT NOT NULL,PRIMARY KEY(owner,pool_id))")
            db.execute("CREATE TABLE IF NOT EXISTS drive_limits(owner TEXT NOT NULL,pool_id TEXT NOT NULL,ceiling INTEGER NOT NULL,signature TEXT NOT NULL,PRIMARY KEY(owner,pool_id))")
            db.execute("CREATE TABLE IF NOT EXISTS spill_volumes(device TEXT PRIMARY KEY,physical_group TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS file_pool_group ON files(pool_group)")
            db.execute("CREATE INDEX IF NOT EXISTS lease_pool_group ON leases(pool_group,state)")
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
            if present is None and not held and not self._drive_unavailable(db, identity=identity):
                self._bump_spill(db, -_bytes)
                db.execute("DELETE FROM files WHERE identity=?", (identity,))
                db.execute("DELETE FROM leases WHERE identity=?", (identity,))
            elif present is not None and present != path:
                db.execute("UPDATE files SET path=? WHERE identity=?", (present, identity))
        for token, pid, path, promised, pool_id in db.execute("SELECT token,pid,path,bytes,pool_id FROM leases WHERE kind='spill' AND state='pending'").fetchall():
            if not self.process_alive(pid):
                if self._drive_unavailable(db, pool_id=pool_id):
                    continue
                if path and Path(path).exists():
                    self._adopt_pending_file(db, token, Path(path), promised)
                elif not self._drive_unavailable(db, pool_id=pool_id):
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
        physical = int(stat.st_blocks) * 512 if hasattr(stat, "st_blocks") else stat.st_size
        allocated = max(stat.st_size, physical, promised)
        prior = db.execute("SELECT bytes,pool_group FROM files WHERE identity=?", (identity,)).fetchone()
        previous = prior[0] if prior else 0
        SharedResourceLedger._bump_spill(db, max(allocated, previous) - previous)
        pool = db.execute("SELECT pool_group FROM leases WHERE token=?", (token,)).fetchone()
        group = prior[1] if prior and prior[1] is not None else pool[0] if pool else None
        db.execute("INSERT OR REPLACE INTO files(identity,bytes,path,pool_group,physical) VALUES(?,?,?,?,?)",
                   (identity, max(allocated, previous), str(path), group, physical))
        db.execute("UPDATE leases SET state='allocated',bytes=0,identity=?,pool_group=? WHERE token=?", (identity, group, token))

    @staticmethod
    def _drive_unavailable(db, *, identity=None, pool_id=None):
        ids = {pool_id} if pool_id is not None else set()
        if identity is not None:
            ids.update(row[0] for row in db.execute("SELECT DISTINCT pool_id FROM leases WHERE identity=? AND pool_id IS NOT NULL", (identity,)))
        for selected in ids:
            grant = db.execute("SELECT path,device,inode,backing_path,backing_device,backing_inode FROM drive_pools WHERE pool_id=?", (selected,)).fetchone()
            if grant is None:
                return True
            try:
                for path, device, inode in (grant[:3], grant[3:]):
                    value = Path(path).lstat()
                    if (not stat_module.S_ISDIR(value.st_mode) or stat_module.S_ISLNK(value.st_mode)
                            or int(getattr(value, "st_file_attributes", 0)) & 0x400
                            or (str(value.st_dev), str(value.st_ino)) != (device, inode)):
                        return True
            except OSError:
                return True
        return False

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

    def authorized_drive_pools(self, owner: str):
        """Read main-issued local grants; config/imported strings are not grants."""
        with self._transaction() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT p.* FROM drive_pools p JOIN drive_access a USING(pool_id) WHERE a.owner=?", (owner,))]

    def drive_cleanup_candidates(self, owner, *, after_token="", limit=128):
        with self._transaction() as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute("SELECT l.token,l.pid,l.path,l.identity,l.pool_id FROM leases l JOIN drive_access a ON a.pool_id=l.pool_id WHERE a.owner=? AND l.kind='spill' AND l.path IS NOT NULL AND l.token>? ORDER BY l.token LIMIT ?",
                (owner, after_token, max(1, min(128, int(limit)))))]

    def drive_backing_has_live_owner(self, token):
        with self._transaction() as db:
            row = db.execute("SELECT pid,identity FROM leases WHERE token=?", (token,)).fetchone()
            if row is None:
                return True
            pids = {row[0]}
            if row[1] is not None:
                pids.update(value[0] for value in db.execute("SELECT pid FROM leases WHERE identity=?", (row[1],)))
            # A reused live PID conservatively keeps the file. Never infer
            # process death from an observation timeout or lease age.
            return any(self.process_alive(pid) for pid in pids)

    @staticmethod
    def _drive_status(db, group):
        ceiling = int(db.execute("SELECT COALESCE(MAX(l.ceiling),0) FROM drive_limits l JOIN drive_pools p USING(pool_id) JOIN owners o ON o.owner=l.owner WHERE p.physical_group=?", (group,)).fetchone()[0])
        used, promised = db.execute("SELECT COALESCE(SUM(bytes),0),COALESCE(SUM(MAX(0,bytes-physical)),0) FROM files f WHERE pool_group=? OR (pool_group IS NULL AND EXISTS(SELECT 1 FROM spill_volumes v WHERE v.physical_group=? AND v.device=substr(f.identity,1,instr(f.identity,':')-1)))", (group, group)).fetchone()
        # Unbound old-path pending writes have not proved their physical group
        # yet; reserve them conservatively until bind_path observes a directory.
        pending = db.execute("SELECT COALESCE(SUM(bytes),0) FROM leases WHERE (pool_group=? OR pool_group IS NULL) AND kind='spill' AND state='pending'", (group,)).fetchone()[0]
        return {"physicalGroup": group, "largestConfiguredPoolBytes": ceiling,
                "chargedBytes": int(used), "observedAllocatedBytes": int(used) - int(promised),
                "unmaterializedPromiseBytes": int(promised), "pendingBytes": int(pending)}

    def drive_usage(self, owner):
        with self._transaction() as db:
            groups = [row[0] for row in db.execute("SELECT DISTINCT p.physical_group FROM drive_pools p JOIN drive_access a USING(pool_id) WHERE a.owner=? ORDER BY p.physical_group", (owner,))]
            return [self._drive_status(db, group) for group in groups]

    def reserve_drive(self, owner, pool_id, byte_count, *, proposed_ceiling, selection_signature,
                      auto_increase, free_bytes, reserve_bytes):
        for value in (byte_count, proposed_ceiling, free_bytes, reserve_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("drive reservations require nonnegative integer bytes")
        token = uuid.uuid4().hex
        with self._transaction() as db:
            self._reconcile(db)
            row = db.execute("SELECT p.physical_group FROM drive_pools p JOIN drive_access a USING(pool_id) JOIN owners o ON o.owner=a.owner WHERE a.owner=? AND p.pool_id=?", (owner, pool_id)).fetchone()
            if row is None:
                raise SharedQuotaPause("selected drive has no local owner authorization", self._status(db))
            previous = db.execute("SELECT ceiling,signature FROM drive_limits WHERE owner=? AND pool_id=?", (owner, pool_id)).fetchone()
            ceiling = proposed_ceiling
            if previous and previous[1] == selection_signature:
                ceiling = max(previous[0], proposed_ceiling) if auto_increase else previous[0]
            db.execute("INSERT OR REPLACE INTO drive_limits VALUES(?,?,?,?)", (owner, pool_id, ceiling, selection_signature))
            group = row[0]
            drive = self._drive_status(db, group)
            global_status = self._status(db)
            if global_status["physicalSpillUsageBytes"] + global_status["pendingSpillBytes"] + byte_count > global_status["largestConfiguredPoolBytes"]:
                raise SharedQuotaPause("atomic shared spill reservation exceeds largest configured pool", global_status)
            if drive["chargedBytes"] + drive["pendingBytes"] + byte_count > drive["largestConfiguredPoolBytes"]:
                raise SharedQuotaPause("selected physical drive pool is full", drive)
            if free_bytes - drive["pendingBytes"] - drive["unmaterializedPromiseBytes"] - byte_count <= reserve_bytes:
                raise SharedQuotaPause("selected drive awaits its free-space reserve", drive)
            db.execute("INSERT INTO leases(token,owner,kind,bytes,state,pid,session,pool_id,pool_group) VALUES(?,?,'spill',?,'pending',?,?,?,?)",
                       (token, owner, byte_count, self.pid, self.session, pool_id, group))
        return SharedAllocationLease(self, token, "spill")

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
            try:
                device = str(Path(path).parent.stat().st_dev)
                group = db.execute("SELECT physical_group FROM spill_volumes WHERE device=?", (device,)).fetchone()
                if group:
                    db.execute("UPDATE leases SET pool_group=? WHERE token=? AND pool_group IS NULL", (group[0], token))
            except OSError:
                pass  # Still charged as unbound pending, never treated as zero.

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
            row = db.execute("SELECT bytes,pool_group FROM leases WHERE token=? AND kind='spill'", (token,)).fetchone()
            if row is None:
                raise ValueError("spill reservation is absent")
            previous = db.execute("SELECT bytes,pool_group FROM files WHERE identity=?", (identity,)).fetchone()
            old = previous[0] if previous else 0
            existing_group = previous[1] if previous else None
            group = row[1] if row[1] is not None else existing_group
            if existing_group is not None and row[1] is not None and existing_group != row[1]:
                raise ValueError("one backing identity cannot belong to different physical drive groups")
            status = self._status(db)
            total = status["physicalSpillUsageBytes"] + status["pendingSpillBytes"] - row[0] + max(old, allocated) - old
            if total > status["largestConfiguredPoolBytes"]:
                raise SharedQuotaPause("actual spill allocation exceeds atomic configured pool", status)
            if group is not None:
                drive = self._drive_status(db, group)
                pending_here = row[0] if row[1] in (None, group) else 0
                if drive["chargedBytes"] + drive["pendingBytes"] - pending_here + max(old, allocated) - old > drive["largestConfiguredPoolBytes"]:
                    raise SharedQuotaPause("actual spill allocation exceeds selected physical drive pool", drive)
            self._bump_spill(db, max(old, allocated) - old)
            db.execute("INSERT OR REPLACE INTO files(identity,bytes,path,pool_group,physical) VALUES(?,?,?,?,?)",
                       (identity, max(old, allocated), str(path), group, measured))
            db.execute("UPDATE leases SET state='allocated',bytes=0,identity=?,path=?,held=?,pool_group=? WHERE token=?", (identity, str(path), int(retain_open_backing), group, token))
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
            row = db.execute("SELECT kind,state,path,identity,pool_id FROM leases WHERE token=?", (token,)).fetchone()
            if row is None:
                return
            if row[1] == "pending":
                if self._drive_unavailable(db, pool_id=row[4]):
                    return
                if row[0] == "spill" and row[2] and Path(row[2]).exists():
                    promised = db.execute("SELECT bytes FROM leases WHERE token=?", (token,)).fetchone()[0]
                    self._adopt_pending_file(db, token, Path(row[2]), promised)
                elif not self._drive_unavailable(db, pool_id=row[4]):
                    db.execute("DELETE FROM leases WHERE token=?", (token,))
            elif row[0] == "spill" and row[2] and not Path(row[2]).exists():
                # Keep aliases for extant hardlinks; reconcile removes usage
                # only after every registered backing name has disappeared.
                aliases = chain((item[0] for item in db.execute("SELECT path FROM leases WHERE identity=?", (row[3],))),
                    (item[0] for item in db.execute("SELECT path FROM files WHERE identity=?", (row[3],))))
                held = db.execute("SELECT 1 FROM leases WHERE identity=? AND held=1 LIMIT 1", (row[3],)).fetchone()
                if not held and not any(path and Path(path).exists() for path in aliases) and not self._drive_unavailable(db, identity=row[3]):
                    count = db.execute("SELECT bytes FROM files WHERE identity=?", (row[3],)).fetchone()
                    if count:
                        self._bump_spill(db, -count[0])
                    db.execute("DELETE FROM files WHERE identity=?", (row[3],))
                    db.execute("DELETE FROM leases WHERE identity=?", (row[3],))
