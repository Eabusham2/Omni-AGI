"""Admitted operation-local SQLite keepers for native READ-ONLY traversals.

No corpus-sized mirror or long frozen transaction is introduced. Existing
per-operation transactions become read savepoints; generation/revision checks
still see later commits. Primary connected learning transactions take priority.
"""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import sqlite3
import time
from types import SimpleNamespace
import weakref

from .connected_transaction import _ConnectionLease


_READ = ContextVar('omni_native_paged_read_session', default=None)
_CACHE_BYTES = 256 * 1024
_CONNECTION_ENVELOPE_BYTES = _CACHE_BYTES + 64 * 1024


def _key(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('native read keeper cannot follow a database symbolic link')
    return str(path.resolve())


def _identity(path):
    info = Path(path).stat()
    return info.st_dev, info.st_ino


def keeper_eligible_stores(stores):
    """Automatic retention requires escrow or an explicitly unmanaged store.

    A boolean-only custom row guard cannot hold a cache RAM lease. Such CLI
    stores retain their original unkept reads rather than being forced to
    allocate an extra cache or fail a previously valid tiny row budget.
    """
    result = []
    for store in stores:
        if store is None or not hasattr(store, 'path'):
            continue
        bound = getattr(getattr(store, '_disk_reserve', None), '__self__', None)
        if callable(getattr(bound, 'reserve_ram', None)) or getattr(store, '_memory_reserve', None) is None:
            result.append(store)
    return result


class _ReadLease(_ConnectionLease):
    def __init__(self, owner, participant):
        self.owner = owner
        super().__init__(participant)

    def execute(self, statement, parameters=()):
        self.owner.check(self._participant.key)
        normalized = statement.strip().rstrip(';').upper()
        if normalized.startswith(('BEGIN IMMEDIATE', 'BEGIN EXCLUSIVE', 'PRAGMA WAL_CHECKPOINT')) \
                or normalized.startswith('PRAGMA QUERY_ONLY') and ('=' in normalized or '(' in normalized):
            raise RuntimeError('write/checkpoint control is forbidden in a native read-only traversal')
        return super().execute(statement, parameters)

    def executemany(self, statement, parameters):
        self.owner.check(self._participant.key)
        return super().executemany(statement, parameters)


class ReadSession:
    def __init__(self, stores, *, cancelled=None):
        self.owners = {}
        for store in stores:
            if store is None or not hasattr(store, 'path'):
                continue
            key = _key(store.path)
            bound = getattr(getattr(store, '_disk_reserve', None), '__self__', None)
            policy = bound if callable(getattr(bound, 'reserve_ram', None)) else None
            if key not in self.owners:
                self.owners[key] = (policy, getattr(store, '_memory_reserve', None))
        self.cancelled, self.connections, self.leases = cancelled, {}, []
        self.token, self.parent = None, None
        self.next_pressure, self.checks = 0., 0
        self.opened_connections = 0

    def __enter__(self):
        parent = _READ.get()
        if parent is not None:
            if not set(self.owners) <= set(parent.owners):
                raise ValueError('nested native read keeper addresses a different owner set')
            self.parent = parent
            return parent
        self.token = _READ.set(self)
        return self

    @contextmanager
    def activate(self):
        """Keep caches between generator reads without leaking caller scope."""
        parent = _READ.get()
        if parent is not None:
            if not set(self.owners) <= set(parent.owners):
                raise ValueError('nested native read keeper addresses a different owner set')
            yield parent
            return
        token = _READ.set(self)
        try:
            yield self
        finally:
            _READ.reset(token)

    def check(self, key):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError('native paged read traversal cancelled')
        item = self.connections.get(key)
        if item is not None and _identity(key) != item.identity:
            raise ValueError('native paged database was replaced during read traversal')
        self.checks += 1
        now = time.monotonic()
        if now >= self.next_pressure or self.checks >= 256:
            policy = self.owners[key][0]
            if policy is not None:
                policy.check_active_pressure('native paged read traversal')
            self.next_pressure, self.checks = now + .25, 0

    def borrow(self, key):
        if key not in self.owners:
            return None
        self.check(key)
        item = self.connections.get(key)
        return self._lease(item) if item is not None else None

    def enlist(self, key, connection):
        if key not in self.owners:
            return connection
        self.check(key)
        if key in self.connections:
            connection.close()
            return self._lease(self.connections[key])
        policy, reserve = self.owners[key]
        allocation = None
        try:
            if policy is not None:
                allocation = policy.reserve_ram(_CONNECTION_ENVELOPE_BYTES, 'native read keeper cache')
            elif reserve is not None and reserve(_CONNECTION_ENVELOPE_BYTES, 'native read keeper cache') is False:
                from .offload import NeuralStateResourcePause
                raise NeuralStateResourcePause('native read keeper cache awaits RAM admission',
                    {'paused': True, 'recoverable': True, 'requiredBytes': _CONNECTION_ENVELOPE_BYTES})
            connection.execute('PRAGMA cache_size=-256')
            connection.execute('PRAGMA mmap_size=0')
            connection.execute('PRAGMA query_only=ON')
            def read_authority(action, first, second, _database, _source):
                if action in {sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                        sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH}:
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_TRANSACTION and str(first).upper() == 'BEGIN':
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_PRAGMA and (str(first).lower() == 'wal_checkpoint'
                        or str(first).lower() in {'query_only', 'writable_schema', 'journal_mode', 'locking_mode',
                            'mmap_size', 'temp_store', 'synchronous', 'cache_size', 'busy_timeout'} and second is not None):
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            connection.set_authorizer(read_authority)
            item = SimpleNamespace(connection=connection, serial=0, abort_only=False,
                leases=weakref.WeakSet(), identity=_identity(key), key=key)
            self.connections[key] = item
            if allocation is not None:
                self.leases.append(allocation)
            self.opened_connections += 1
            return self._lease(item)
        except BaseException:
            connection.close()
            if allocation is not None:
                allocation.release()
            raise

    def _lease(self, item):
        return _ReadLease(self, item)

    def __exit__(self, *_error):
        if self.parent is not None:
            return False
        if self.token is not None:
            _READ.reset(self.token)
            self.token = None
        try:
            for item in self.connections.values():
                for lease in list(item.leases):
                    lease.close()
                if item.connection.in_transaction:
                    item.connection.rollback()
                item.connection.close()
            self.connections.clear()
        finally:
            for allocation in self.leases:
                allocation.release()
            self.leases.clear()
        return False


def borrow_read_connection(path):
    current = _READ.get()
    return current.borrow(_key(path)) if current is not None else None


def enlist_read_connection(path, connection):
    current = _READ.get()
    return current.enlist(_key(path), connection) if current is not None else connection
