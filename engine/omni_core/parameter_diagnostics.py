"""Exact scoped core deltas, with no whole packed-weight diagnostic copy.

Only first-original changed bytes enter a private sparse journal. The final
actual codes determine net L2, including reversals and retry rollback. This is
not a cumulative flip counter and deliberately does not measure VSA/edge state.
"""
import array
import functools
import math
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch


BLOCK_BYTES = 4096
_OBSERVERS = ContextVar("core_parameter_diagnostic_observers", default=())
_CALL = ContextVar("core_parameter_diagnostic_call", default=None)


def diagnostic_call(method):
    """Close even when a chat/idle method yields early or raises before delta."""
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        journals = []
        token = _CALL.set(journals)
        try:
            return method(*args, **kwargs)
        finally:
            try:
                for journal in reversed(journals):
                    journal.close()
            finally:
                _CALL.reset(token)
    return wrapped


def active_diagnostic_scopes():
    return tuple(id(journal) for journal in _OBSERVERS.get())


@contextmanager
def packed_diagnostic_restore(module, target, replacement, *, start=0, captured_scopes=()):
    """A same-operation rollback needs no new diagnostic original bytes.

    Every real write after that rollback point was already journaled first.
    Reading the actual final codes gives the exact baseline delta, even under
    pressure where allocating a second journal would obstruct real rollback.
    """
    touched = []
    try:
        for journal in _OBSERVERS.get():
            if id(journal) in captured_scopes:
                item = journal.lookup(module, target)
            else:
                item = journal.before_write(module, target, replacement, int(start))
            if item is not None:
                touched.append((journal, item))
        yield
    finally:
        for journal, item in touched:
            journal.mark_version(item)


@contextmanager
def packed_diagnostic_write(module, target, replacement, *, start=0):
    """Record before a real write, then track its actual version even on error."""
    touched = []
    try:
        for journal in _OBSERVERS.get():
            descriptor = journal.before_write(module, target, replacement, int(start))
            if descriptor is not None:
                touched.append((journal, descriptor))
        yield
    finally:
        for journal, descriptor in touched:
            journal.mark_version(descriptor)


class ParameterDeltaJournal:
    scope = {"format": "core-module-net-delta-v1", "modules": ["decoder", "memory_bridge", "idea_adapter", "liquid"],
        "packedUnit": "signed-ternary-level", "floatingUnit": "native-residual-control-value",
        "substrateDeltaMeasured": False, "includesVsaVectors": False, "includesSparseEdges": False}

    def __init__(self, roots, *, directory, reserve_ram, reserve_disk, include_state=False, resource_policy=None):
        self.roots = tuple(roots)
        self.directory = Path(directory)
        self.reserve_ram, self.reserve_disk = reserve_ram, reserve_disk
        self.database = None
        self.private_directory = None
        self.closed = False
        self.include_state = bool(include_state)
        self.resource_policy = resource_policy
        self._disk_quota = None
        self._sidecar_quotas = []
        self.original = self.inventory()
        self.by_module = {(id(item["module"]), item["name"]): item for item in self.original.values() if item["kind"] == "packed"}
        controls = [item for item in self.original.values() if item["kind"] == "control"]
        metadata_bytes = 1024 + len(self.original) * 1024
        reserve_ram(metadata_bytes + sum(item["tensor"].numel() * item["tensor"].element_size() * 2 for item in controls))
        for item in controls:
            item["baseline"] = item["tensor"].detach().to(device="cpu", copy=True)
        for item in self.by_module.values():
            self.mark_version(item)
        _OBSERVERS.set((*_OBSERVERS.get(), self))
        call = _CALL.get()
        if call is not None:
            call.append(self)

    def inventory(self):
        result, seen = {}, set()
        for prefix, root in self.roots:
            for name, parameter in (() if self.include_state else root.named_parameters()):
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                result[prefix + ".parameter." + name] = {"kind": "control", "tensor": parameter, "module": root, "name": name}
            for path, module in root.named_modules():
                getter = getattr(module, "authoritative_packed_tensors", None)
                if not callable(getter) and not self.include_state:
                    continue
                learned = {id(value) for value in getter()} if callable(getter) else set()
                for name, value in module.named_buffers(recurse=False):
                    if (id(value) not in learned and not (self.include_state and value.dtype == torch.uint8)) or id(value) in seen:
                        continue
                    if value.dtype != torch.uint8 or not value.is_contiguous():
                        raise ValueError("core diagnostic packed owner must expose contiguous uint8 codes")
                    seen.add(id(value))
                    key = prefix + "." + path + "." + name
                    result[key] = {"kind": "packed", "key": key, "module": module, "name": name,
                        "shape": tuple(value.shape), "elements": value.numel()}
        return result

    @staticmethod
    def current(item):
        value = item["module"]._buffers.get(item["name"])
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or not value.is_contiguous():
            raise RuntimeError("core diagnostic registered packed target disappeared or changed representation")
        return value

    def mark_version(self, item):
        value = self.current(item)
        item["observed"] = (id(value), int(value._version))

    def _open(self):
        if self.database is not None:
            return
        self.reserve_ram(131072)
        self.reserve_disk(131072, "sparse core diagnostic journal metadata")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.private_directory = Path(tempfile.mkdtemp(prefix="core-delta-", dir=str(self.directory)))
        path = self.private_directory / "changed.sqlite"
        reserve = getattr(self.resource_policy, "reserve_spill", None)
        self._disk_quota = reserve(131072, "core diagnostic journal metadata") if callable(reserve) else None
        if self._disk_quota is not None: self._disk_quota.bind_path(path)
        self.database = sqlite3.connect(str(path))
        self.database.execute("PRAGMA cache_size=-64")
        self.database.execute("PRAGMA journal_mode=DELETE")
        self.database.execute("CREATE TABLE original (owner TEXT, block INTEGER, positions BLOB, codes BLOB, PRIMARY KEY(owner,block)) WITHOUT ROWID")
        self.database.commit()
        if self._disk_quota is not None: self._disk_quota.commit(path=path)

    def before_write(self, module, target, replacement, start):
        if self.closed:
            raise RuntimeError("core diagnostic journal is closed")
        item = self.lookup(module, target)
        if item is None:
            return None # modalities, row resistance, and new owners are not this metric
        current = self.current(item).reshape(-1)
        after = replacement.detach().reshape(-1)
        if target.dtype != torch.uint8 or replacement.dtype != torch.uint8 or start < 0 or start + after.numel() > current.numel():
            raise ValueError("core diagnostic write range is invalid")
        position = 0
        while position < after.numel():
            absolute = start + position
            block, offset = divmod(absolute, BLOCK_BYTES)
            count = min(BLOCK_BYTES - offset, after.numel() - position)
            self.reserve_ram(count * 192 + 131072)
            before = current[absolute:absolute + count].detach().to(device="cpu")
            proposed = after[position:position + count].to(device="cpu")
            changed = torch.nonzero(before.ne(proposed), as_tuple=False).flatten().tolist()
            if changed:
                self._open()
                row = self.database.execute("SELECT positions,codes FROM original WHERE owner=? AND block=?", (item["key"], block)).fetchone()
                saved = {}
                if row is not None:
                    indices = array.array("H")
                    indices.frombytes(row[0])
                    saved.update(zip(indices, row[1]))
                for index in changed:
                    saved.setdefault(offset + index, int(before[index]))
                indices = sorted(saved)
                positions = array.array("H", indices).tobytes()
                codes = bytes(saved[index] for index in indices)
                self.reserve_disk(16384 + (len(positions) + len(codes)) * 3, "first-original changed core bytes")
                reserve = getattr(self.resource_policy, "reserve_spill", None)
                growth = reserve(65536 + (len(positions) + len(codes)) * 3, "first-original changed bytes") if callable(reserve) else None
                sidecar = reserve(65536 + (len(positions) + len(codes)) * 3, "diagnostic SQLite transaction") if callable(reserve) else None
                path = self.private_directory / "changed.sqlite"
                if growth is not None: growth.bind_path(path)
                if sidecar is not None:
                    sidecar.bind_path(self.private_directory / "changed.sqlite-journal")
                try:
                    self.database.execute("INSERT OR REPLACE INTO original VALUES (?,?,?,?)", (item["key"], block, positions, codes))
                    self.database.commit()
                    if growth is not None:
                        growth.commit(path=path)
                        self._disk_quota = growth
                finally:
                    if sidecar is not None:
                        sidecar.release()
                        if (self.private_directory / "changed.sqlite-journal").exists():
                            self._sidecar_quotas.append(sidecar)
                    if growth is not None: growth.release()
            position += count
            from .native_core_paging import release_native_tensor_chunk
            release_native_tensor_chunk(self.current(item), absolute, count)
        return item

    def lookup(self, module, target):
        candidates = (entry for (identity, _name), entry in self.by_module.items() if module is None or identity == id(module))
        for entry in candidates:
            value = self.current(entry)
            if value is target or (value.device == target.device and value.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()):
                return entry
        return None

    def _rows(self, key):
        if self.database is None:
            return iter(())
        return self.database.execute("SELECT block,positions,codes FROM original WHERE owner=? ORDER BY block", (key,))

    @staticmethod
    def _packed_square(now, prior):
        total = 0
        for shift in (0, 2, 4, 6):
            delta = ((now.to(torch.int16) >> shift) & 3) - ((prior.to(torch.int16) >> shift) & 3)
            total += int(delta.to(torch.int64).square().sum().item())
        return total

    def delta_norm(self):
        if self.closed:
            raise RuntimeError("core diagnostic journal already closed")
        current = self.inventory()
        total = 0.0
        for key, item in current.items():
            original = self.original.get(key)
            if item["kind"] == "control":
                value = item["tensor"].detach().reshape(-1)
                baseline = original.get("baseline") if original else None
                baseline = baseline.reshape(-1) if baseline is not None else None
                for start in range(0, value.numel(), BLOCK_BYTES):
                    count = min(BLOCK_BYTES, value.numel() - start)
                    self.reserve_ram(count * 32 + 131072)
                    block = value[start:start + count].to(device="cpu", dtype=torch.float64)
                    prior = baseline[start:start + count].double() if baseline is not None and baseline.numel() == value.numel() else torch.zeros_like(block)
                    total += float((block - prior).square().sum().item())
                continue
            value = self.current(item).reshape(-1)
            if original is None:
                for start in range(0, value.numel(), BLOCK_BYTES):
                    self.reserve_ram(min(BLOCK_BYTES, value.numel() - start) * 48 + 131072)
                    block = value[start:start + BLOCK_BYTES].detach().to(device="cpu")
                    total += self._packed_square(block, torch.full_like(block, 0x55))
                continue
            if original["module"] is not item["module"] or original["shape"] != tuple(self.current(item).shape):
                raise RuntimeError("core diagnostic owner topology changed; exact delta requires an explicit source migration boundary")
            observed = original["observed"]
            if observed[0] == id(self.current(item)) and observed[1] != int(self.current(item)._version):
                raise RuntimeError("core diagnostic observed a packed write outside its before-write journal")
            if observed[0] != id(self.current(item)) and getattr(item["module"], "_native_core_pager", None) is None:
                raise RuntimeError("core diagnostic observed an unregistered packed target replacement")
            for block, encoded_positions, codes in self._rows(key):
                self.reserve_ram(BLOCK_BYTES * 192 + 131072)
                indices = array.array("H")
                indices.frombytes(encoded_positions)
                offsets = torch.tensor(indices, dtype=torch.int64)
                selected = value[block * BLOCK_BYTES:block * BLOCK_BYTES + BLOCK_BYTES].index_select(0, offsets.to(value.device)).detach().cpu()
                total += self._packed_square(selected, torch.tensor(list(codes), dtype=torch.uint8))
                from .native_core_paging import release_native_tensor_chunk
                release_native_tensor_chunk(self.current(item), block * BLOCK_BYTES, min(BLOCK_BYTES, value.numel() - block * BLOCK_BYTES))
        missing = set(self.original) - set(current)
        if missing:
            raise RuntimeError("core diagnostic owner disappeared; exact delta is unavailable")
        return math.sqrt(total)

    def restore(self):
        """Restore only actual first-original byte changes, resolving live names."""
        if not self.include_state: raise RuntimeError("diagnostic-only journal is not a rollback owner")
        for item in self.original.values():
            if item["kind"] != "packed": continue
            target = self.current(item)
            if tuple(target.shape) != item["shape"]: raise RuntimeError("rollback packed topology did not restore")
            flat = target.reshape(-1)
            for block, positions, codes in self._rows(item["key"]):
                self.reserve_ram(BLOCK_BYTES * 192 + 131072)
                offsets = array.array("H")
                offsets.frombytes(positions)
                indices = torch.tensor(offsets, dtype=torch.int64, device=target.device) + block * BLOCK_BYTES
                values = torch.tensor(list(codes), dtype=torch.uint8, device=target.device)
                # All scoped originals already exist; restoration adds no new
                # byte history in this rollback or its enclosing diagnostics.
                touched = []
                for observer in _OBSERVERS.get():
                    descriptor = observer.lookup(item["module"], target)
                    if descriptor is not None: touched.append((observer, descriptor))
                flat.index_copy_(0, indices, values)
                for observer, descriptor in touched: observer.mark_version(descriptor)
                from .native_core_paging import release_native_tensor_chunk
                release_native_tensor_chunk(target, block * BLOCK_BYTES, min(BLOCK_BYTES, target.numel() - block * BLOCK_BYTES))

    def close(self):
        if self.closed:
            return
        self.closed = True
        _OBSERVERS.set(tuple(journal for journal in _OBSERVERS.get() if journal is not self))
        if self.database is not None:
            self.database.close()
            self.database = None
        if self.private_directory is not None:
            for name in ("changed.sqlite", "changed.sqlite-journal"):
                (self.private_directory / name).unlink(missing_ok=True)
            self.private_directory.rmdir()
        if self._disk_quota is not None: self._disk_quota.release()
        for quota in self._sidecar_quotas: quota.release()
        self.original.clear()
        self.by_module.clear()
