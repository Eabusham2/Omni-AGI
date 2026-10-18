"""Same-owner copy-on-first-use state for a private prepared learner wave.

The architecture/module objects are borrowed, not constructed or deep-copied.
Compact native fields are copied only when a declared owner is entered; small
controls prefer admitted RAM, otherwise selected-drive mapped bytes. This is
owner-field COW, not a claim of filesystem block reflinks or a floating master.
No learned operation, database, mapping or allocation occurs on import.
"""
from __future__ import annotations

import copy
import hashlib
import sqlite3
import uuid
import weakref
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch

from .decoder_execution_native import _tensor_guard
from .native_core_paging import NativeCorePager, _Owner

_SQLITE = ContextVar("native_private_training_sqlite", default=None)
_READ = ContextVar("native_private_training_read_only_owner", default=False)


@contextmanager
def training_owner_read_only():
    token = _READ.set(True)
    try: yield
    finally: _READ.reset(token)


def borrow_training_connection(path):
    scope = _SQLITE.get()
    return None if scope is None else scope.connection(path)


class PrivateTrainingSQLite:
    """Lazy database-first-access copies; original files are read-only.

    Known paged-store paths are explicitly inventoried by the runtime owner.
    Source bytes are copied by SQLite's bounded backup, not serialize()/RAM,
    and every later SQL write is confined to the private database. No savepoint
    or rollback writes the original database/WAL. Physical block-COW is not
    assumed on filesystems that cannot provide it.
    """
    def __init__(self, paths, directory, policy, cancelled, *, read_only=False):
        if any(Path(path).is_symlink() for path in paths): raise ValueError("private learner cannot follow a graph database symlink")
        self.paths = {str(Path(path).resolve()): None for path in paths}
        self.directory, self.policy, self.cancelled = Path(directory), policy, cancelled
        self.read_only = bool(read_only)
        self.leases, self.connections, self.participants = [], [], {}
    def check(self):
        if self.cancelled(): raise InterruptedError("private learner database stopped")
    def connection(self, path):
        self.check(); key = str(Path(path).resolve())
        if key not in self.paths:
            raise RuntimeError("prepared learner touched an uninventoried graph database")
        if key in self.participants:
            from .connected_transaction import _ConnectionLease
            return _ConnectionLease(self.participants[key])
        target = self.paths[key]
        if target is None:
            before = Path(key).stat()
            source = sqlite3.connect(Path(key).as_uri()+"?mode=ro", uri=True, isolation_level=None)
            target = self.directory/(uuid.uuid4().hex+".sqlite3")
            lease = None
            try:
                page_size = int(source.execute("PRAGMA page_size").fetchone()[0])
                pages = int(source.execute("PRAGMA page_count").fetchone()[0])
                lease = self.policy.reserve_spill(pages*page_size+131072, "private learner graph database")
                lease.bind_path(target); self.directory.mkdir(parents=True, exist_ok=True)
                destination = sqlite3.connect(target, isolation_level=None)
                try:
                    with self.policy.reserve_ram(page_size*128+131072, "bounded private learner database copy"):
                        source.backup(destination, pages=64, progress=lambda *_args: self.check())
                    destination.execute("PRAGMA journal_mode=DELETE")
                    destination.execute("PRAGMA synchronous=FULL")
                finally: destination.close()
                after = Path(key).stat()
                if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise RuntimeError("original graph database changed during private backup")
                lease.commit(path=target, promised_bytes=pages*page_size, retain_open_backing=True)
                self.paths[key] = target; self.leases.append((lease, target))
            except BaseException:
                if target.exists(): target.unlink()
                if lease is not None: lease.release()
                raise
            finally: source.close()
        connection = sqlite3.connect(target, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA cache_size=-256")
        if self.read_only: connection.execute("PRAGMA query_only=ON")
        cache = self.policy.reserve_ram(327680, "bounded private learner graph connection cache")
        cache.mark_allocated(327680); self.leases.append((cache, None))
        self.connections.append(connection)
        from .connected_transaction import _Participant, _ConnectionLease
        if self.read_only:
            from types import SimpleNamespace
            connection.execute("BEGIN")
            self.participants[key] = SimpleNamespace(connection=connection, serial=0, abort_only=False, leases=weakref.WeakSet())
        else: self.participants[key] = _Participant(connection)
        return _ConnectionLease(self.participants[key])
    @contextmanager
    def activate(self):
        token = _SQLITE.set(self)
        try: yield self
        finally: _SQLITE.reset(token)
    def summary(self):
        result = []
        for key in sorted(self.paths):
            self.check()
            participant = self.participants.get(key)
            connection = participant.connection if participant is not None else sqlite3.connect(Path(key).as_uri()+"?mode=ro", uri=True)
            try:
                hasher = hashlib.sha256()
                for statement in connection.iterdump():
                    self.check()
                    with self.policy.reserve_ram(len(statement)*4+4096, "bounded private graph row equality proof"):
                        raw = statement.encode(); hasher.update(len(raw).to_bytes(8, "little")); hasher.update(raw)
                result.append([key, hasher.hexdigest()])
            finally:
                if participant is None: connection.close()
        return result
    def close(self):
        for connection in self.connections:
            try: connection.close()
            except sqlite3.ProgrammingError: pass
        self.connections.clear()
        for lease, target in self.leases:
            if target is None:
                lease.release(); continue
            target.unlink(missing_ok=True)
            for suffix in ("-wal", "-shm", "-journal"): Path(str(target)+suffix).unlink(missing_ok=True)
            lease.backing_closed()
        self.leases.clear()
        self.participants.clear()


class TrainingCowPager(NativeCorePager):
    """The same module objects with separate native bytes/epochs/diagnostics."""
    def __init__(self, sources, directory, policy, cancelled, *, cpu_hot_bytes, accelerator_hot_bytes):
        super().__init__(directory, cpu_hot_bytes=cpu_hot_bytes, accelerator_hot_bytes=accelerator_hot_bytes,
            reserve_disk=policy.require_disk, resource_policy=policy,
            reserve_admission=lambda count, device: policy.require_device(device, count, "private native learner owner"))
        self.sources, self.cancelled, self._materialized = sources, cancelled, set()
        for module in sources:
            self._owners[module] = _Owner(loaded=True, name=sources[module]["name"])
    def check(self):
        if self.cancelled(): raise InterruptedError("private native learner owner stopped")
    def materialize(self, module):
        self.check()
        if module in self._materialized or module not in self.sources: return
        source, owner = self.sources[module], self._owners[module]
        for name, value in source["buffers"].items():
            if not isinstance(value, torch.Tensor) or (name in module._non_persistent_buffers_set
                    and name not in getattr(module, "_native_cached_buffers", ())): continue
            if not value.is_contiguous():
                raise ValueError("private learner requires contiguous native owner fields")
            guard = _tensor_guard(value)
            target = self._map_tensor(owner, name, tuple(value.shape), value.dtype)
            with self.shared_resource_policy.reserve_ram(min(value.numel()*value.element_size(), self.chunk_bytes)*3+4096,
                    "bounded private native learner owner copy"):
                original, private = value.detach().reshape(-1), target.reshape(-1)
                block = max(1, self.chunk_bytes//value.element_size())
                for start in range(0, value.numel(), block):
                    self.check(); part = original[start:start+block].to("cpu")
                    private[start:start+part.numel()].copy_(part)
                    self._transfer_bytes += part.numel()*part.element_size()
                    from .native_core_paging import release_native_tensor_chunk
                    release_native_tensor_chunk(target, start*target.element_size(), part.numel()*part.element_size())
            if guard != _tensor_guard(value): raise RuntimeError("original native owner changed during its private copy")
            owner.cold[name] = target; owner.maps[name].dirty = True; module._buffers[name] = target
        owner.versions = {name: int(value._version) for name, value in owner.cold.items()}
        self._materialized.add(module); self._invalidate_validation(module)
    @contextmanager
    def use(self, module, device):
        if module in self.sources and module not in self._materialized and _READ.get():
            # Identity checks read current original bytes without instantiating
            # a writable owner or publishing an original epoch/observation.
            self.check(); prior = dict(module._buffers)
            try:
                module._buffers.update(self.sources[module]["buffers"]); yield
            finally: module._buffers.clear(); module._buffers.update(prior)
            return
        self.materialize(module)
        with super().use(module, device): yield
    def owns(self, module, value):
        if getattr(module, "_native_core_pager", None) is not self: return False
        if module not in self._owners: return False
        self.materialize(module)
        pointer = value.untyped_storage().data_ptr()
        return any(isinstance(item, torch.Tensor) and item.untyped_storage().data_ptr()==pointer for item in module._buffers.values())


class TrainingControlCow:
    """Alias-preserving private control storage, not learned floating weights."""
    def __init__(self, pager, policy, cancelled):
        self.pager, self.policy, self.cancelled = pager, policy, cancelled
        self.values, self.storages, self.leases, self.roots = {}, {}, [], []
    def tensor(self, value):
        if id(value) in self.values: return self.values[id(value)]
        storage = value.untyped_storage(); size = storage.nbytes()
        key = (str(value.device), storage.data_ptr(), size)
        if key not in self.storages:
            lease = None
            try:
                lease = self.policy.reserve_ram(size+min(size, self.pager.chunk_bytes)*2+4096, "private learner control storage")
                raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (size,), (1,))
                private = torch.empty(size, dtype=torch.uint8)
                for start in range(0, size, self.pager.chunk_bytes):
                    if self.cancelled(): raise InterruptedError("private learner control copy stopped")
                    private[start:start+self.pager.chunk_bytes].copy_(raw[start:start+self.pager.chunk_bytes].to("cpu"))
                lease.mark_allocated(size); self.leases.append(lease)
            except Exception:
                if lease is not None: lease.release()
                if self.cancelled(): raise InterruptedError("private learner control copy stopped")
                if not size: private = torch.empty(0, dtype=torch.uint8)
                else:
                    owner = _Owner(loaded=True, name="private-control")
                    raw = torch.empty(0, dtype=torch.uint8, device=value.device).set_(storage, 0, (size,), (1,))
                    private = self.pager._map_tensor(owner, "value", (size,), torch.uint8)
                    block = self.pager.chunk_bytes
                    with self.policy.reserve_ram(min(size, block)*3+4096, "private learner mapped control transfer"):
                        for start in range(0, size, block):
                            if self.cancelled(): raise InterruptedError("private learner control copy stopped")
                            private[start:start+block].copy_(raw[start:start+block].to("cpu"))
                    self.roots.append(owner)
            self.storages[key] = private
        raw = self.storages[key]
        target = raw.view(value.dtype).as_strided(tuple(value.shape), tuple(value.stride()), int(value.storage_offset()))
        if value.requires_grad: target = target.detach().requires_grad_(True)
        self.values[id(value)] = target
        return target
    def value(self, value):
        if id(value) in self.values: return self.values[id(value)]
        if isinstance(value, torch.Tensor): return self.tensor(value)
        if isinstance(value, torch.nn.Module): return value
        if isinstance(value, dict):
            lease = self.policy.reserve_ram(1024+len(value)*128, "private learner control mapping metadata")
            self.leases.append(lease); lease.mark_allocated(1024+len(value)*128)
            result = defaultdict(value.default_factory) if isinstance(value, defaultdict) else OrderedDict() if isinstance(value, OrderedDict) else {}
            self.values[id(value)] = result
            for key, item in value.items(): result[key] = self.value(item)
            return result
        if isinstance(value, (list, tuple, set, frozenset)):
            lease = self.policy.reserve_ram(512+len(value)*64, "private learner control sequence metadata")
            self.leases.append(lease); lease.mark_allocated(512+len(value)*64)
            items = [self.value(item) for item in value]
            result = tuple(items) if isinstance(value, tuple) else set(items) if isinstance(value, set) else frozenset(items) if isinstance(value, frozenset) else items
            self.values[id(value)] = result; return result
        if value is None or isinstance(value, (str, int, float, bool, bytes, Path, torch.device)): return value
        # Immutable configuration/identity records can be privately copied;
        # live stores/pagers/locks are explicitly composed by the caller.
        if hasattr(value, "__dataclass_fields__"):
            result = copy.copy(value); self.values[id(value)] = result
            for name, item in vars(value).items(): object.__setattr__(result, name, self.value(item))
            return result
        return value
    def close(self):
        self.values.clear(); self.storages.clear(); self.roots.clear()
        for lease in self.leases: lease.release()
        self.leases.clear()


class TrainingNamespace:
    """Temporarily bind private fields on the actual objects, not a new model.

    The original dictionaries and packed-owner records remain untouched. A
    new sparse page is registered only on the private ModuleDict/private pager.
    Exact original tensor tokens are checked after restoration, including on
    Stop or a failed trial. Cleanup closes private owners BEFORE rebinding the
    originals; a pager must never clear an original module's buffers on close.
    """
    def __init__(self, brain, modules, directory, policy, cancelled, *, mutable_controls=None):
        self.brain, self.modules = brain, tuple(dict.fromkeys(modules))
        self.mutable_controls = set(id(module) for module in self.modules) if mutable_controls is None else set(mutable_controls)
        self.originals, self.guards = [], []
        self.sources = source = {module: {"buffers": dict(module._buffers),
            "name": brain.core_pager._owners[module].name}
            for module in self.modules if module in brain.core_pager._owners}
        self.pager = TrainingCowPager(source, Path(directory)/"packed", policy, cancelled,
            cpu_hot_bytes=brain.core_pager.cpu_hot_bytes,
            accelerator_hot_bytes=brain.core_pager.accelerator_hot_bytes)
        self.controls = TrainingControlCow(self.pager, policy, cancelled)
        self.original_brain, self.entered = None, False
        self.cancelled, self.policy = cancelled, policy
    def __enter__(self):
        if self.entered: raise RuntimeError("learner namespace cannot be entered twice")
        self.entered = True
        try:
            for module in self.modules:
                if self.cancelled(): raise InterruptedError("private learner namespace stopped")
                original = module.__dict__
                self.originals.append((module, original))
                self.guards.extend((value, _tensor_guard(value)) for value in original.get("_buffers", {}).values() if isinstance(value, torch.Tensor))
                if any(isinstance(value, torch.Tensor) for value in original.get("_parameters", {}).values()):
                    raise ValueError("private prepared learner requires native packed parameters only")
                private = {}
                for key, value in original.items():
                    if key in {"_native_core_pager", "native_core_pager"}:
                        private[key] = self.pager if value is self.brain.core_pager else value
                    elif key == "_native_core_pager_ref": private[key] = weakref.ref(self.pager)
                    elif key in {"_modules", "_parameters"}: private[key] = dict(value)
                    elif key == "_buffers":
                        private[key] = {name: (item if id(module) not in self.mutable_controls or module in self.sources and (name not in module._non_persistent_buffers_set
                            or name in getattr(module, "_native_cached_buffers", ())) else self.controls.value(item))
                            for name, item in value.items()}
                    elif key in {"_native_transmission_observer", "_recall_graph_observer", "_online_transaction", "_online_pager_scope"}:
                        private[key] = None
                    else: private[key] = self.controls.value(value)
                module.__dict__ = private
            self.original_brain = self.brain.__dict__
            private = dict(self.original_brain)
            for name in ("counters", "slow_anchors", "slow_importance", "liquid_state", "resource_pause", "_allocator_recovery_pause"):
                if name in private: private[name] = self.controls.value(private[name])
            private["core_pager"] = self.pager
            private["_optimizer_offloaded"] = False
            # A native packed objective has no dense optimizer state to copy.
            # Keep the same stateless optimizer object; reject residual state.
            if self.brain._optimizer.state or self.brain._optimizer.param_groups:
                raise ValueError("private prepared learner has an unsupported dense optimizer")
            self.brain.__dict__ = private
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise
    def __exit__(self, kind, value, traceback):
        failure = None
        try:
            # Failed autograd graphs cannot keep private packed owners pinned
            # beyond this scope. The actual loss graph must be released by its
            # consumer before exit; no original residency is evicted here.
            self.pager.close()
        except BaseException as error: failure = error
        finally:
            if self.original_brain is not None: self.brain.__dict__ = self.original_brain
            for module, original in reversed(self.originals): module.__dict__ = original
            self.controls.close()
        changed = any(_tensor_guard(tensor) != guard for tensor, guard in self.guards)
        self.originals.clear(); self.guards.clear()
        if changed:
            from .operation_auto_policy import OperationAutoRestoreRequired
            self.brain._native_execution_restore_required = True
            raise OperationAutoRestoreRequired("private learner did not restore its exact original owners") from failure
        if failure is not None:
            # Source dictionaries/tensor guards are restored. Failure to
            # retire a PRIVATE backing is not authority to unload/reset the
            # user's model or to mark its learned owners as corrupted.
            raise RuntimeError("private learner backing cleanup did not complete") from failure
        return False


@contextmanager
def private_training_diagnostics(pager, controls):
    """A private write must target private bytes before any observer sees it.

    Do not append a before-image to the live user's diagnostic journal, or
    credit the original owner epochs for private learner mutations.
    """
    from .parameter_diagnostics import _OBSERVERS
    class Guard:
        def before_write(self, module, target, replacement, start):
            owned = pager.owns(module, target)
            if not owned:
                pointer = target.untyped_storage().data_ptr()
                owned = any(value.untyped_storage().data_ptr() == pointer for value in controls.storages.values())
            if not owned: raise RuntimeError("private learner attempted an original parameter write")
            return None
        def register_sparse_zero_owner(self, parent, key, block):
            if getattr(parent, "_native_core_pager_ref", lambda: None)() is not pager:
                raise RuntimeError("private learner attempted an original sparse registry write")
            return False
        def retire_sparse_zero_owner(self, parent, key, block):
            if getattr(parent, "_native_core_pager_ref", lambda: None)() is not pager:
                raise RuntimeError("private learner attempted an original sparse registry retirement")
    token = _OBSERVERS.set((Guard(),))
    try: yield
    finally: _OBSERVERS.reset(token)
