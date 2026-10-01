"""RAM-first, exact mutable router storage and bounded before-image rollback.

Only private session files are writable. Registered checkpoint tensor names,
shapes and dtypes stay ordinary tensors; the bounded loader/writer can stream
them without adopting an immutable checkpoint as mutable backing.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import math
import mmap
import os
import struct
import tempfile
import threading
import uuid
import weakref
from contextlib import contextmanager, nullcontext
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch
from .bounded_tensor_io import TRANSFER_BYTES
from .offload import NeuralStateResourcePause

MATRIX_FIELDS = ("_packed_weights", "eligibility_accumulator", "stability", "uses")
_CONSTRUCTION = contextvars.ContextVar("omni_router_state_construction", default=None)
_MAPPED: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def current_router_state_construction():
    return _CONSTRUCTION.get()


def release_router_tensor_chunk(value: torch.Tensor, start: int, count: int) -> None:
    if value.device.type == "cpu":
        pager = _MAPPED.get(value.untyped_storage().data_ptr())
        if pager is not None:
            pager.release_chunk(value, start, count)


def allocated_bytes(path: Path, logical: int) -> int:
    try:
        return max(0, int(path.stat().st_blocks) * 512)
    except (OSError, AttributeError):
        return int(logical)


def promised_bytes(logical: int) -> int:
    # Mutable sparse files need filesystem block rounding even before their
    # first nonzero write. A tiny mapped field is not physically a one-byte
    # promise merely because ftruncate initially reports zero allocated blocks.
    return math.ceil(logical / mmap.ALLOCATIONGRANULARITY) * mmap.ALLOCATIONGRANULARITY


def tile_ranges(rows: int, columns: int, budget_bytes: int) -> Iterator[tuple[int, int, int, int]]:
    # Includes controls, decoded codes, elementwise temporaries and old bytes.
    elements = (int(budget_bytes) - 1024) // 96
    if elements < 4:
        raise NeuralStateResourcePause("one exact router tile exceeds its admitted scratch", {"requiredBytes": 1408})
    if rows <= 0 or columns <= 0:
        return
    row_block = min(64, rows, max(1, elements // min(columns, 64)))
    column_block = max(4, (elements // row_block // 4) * 4)
    for r0 in range(0, rows, row_block):
        for c0 in range(0, columns, column_block):
            yield r0, min(rows, r0 + row_block), c0, min(columns, c0 + column_block)


def unpack_tile(packed: torch.Tensor, r0: int, r1: int, c0: int, c1: int) -> torch.Tensor:
    if c0 % 4:
        raise ValueError("router packed tile must start at a byte boundary")
    block = packed[r0:r1, c0 // 4:(c1 + 3) // 4].detach().to(device="cpu", copy=True)
    lanes = torch.stack(tuple((block >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
    if bool(lanes.eq(3).any()):
        raise ValueError("router tile contains reserved ternary codes")
    return (lanes.reshape(r1 - r0, -1)[:, :c1 - c0].to(torch.int8) - 1).contiguous()


def pack_tile(levels: torch.Tensor) -> torch.Tensor:
    if levels.dtype != torch.int8 or levels.ndim != 2 or not bool(((levels >= -1) & (levels <= 1)).all()):
        raise ValueError("router tile levels are not exact ternary int8")
    rows, columns = levels.shape
    codes = torch.ones((rows, ((columns + 3) // 4) * 4), dtype=torch.uint8)
    codes[:, :columns].copy_((levels.to(device="cpu", dtype=torch.int16) + 1).to(torch.uint8))
    lanes = codes.reshape(rows, -1, 4)
    return (lanes[:, :, 0] | (lanes[:, :, 1] << 2) | (lanes[:, :, 2] << 4) | (lanes[:, :, 3] << 6)).contiguous()


def iter_row_major_ternary_chunks(owner):
    """Virtual decoded-int8 matrix bytes, with no whole matrix allocation.

    Consume each yielded block before continuing; a caller retaining every
    block is deliberately not a bounded-memory use of this iterator. A full
    row block is used only when it fits. Otherwise each row is traversed left
    to right before the next row, unlike a two-dimensional compute-tile walk.
    """
    with owner._operation():
        owner._validate_packed()
        rows, columns = owner.post_neurons, owner.pre_neurons
        elements = (owner._tile_budget() - 1024) // 16
        if elements < 4:
            raise NeuralStateResourcePause("decoded router checksum exceeds its admitted tile", {"requiredBytes": 1088})
        row_block = max(1, elements // columns)
        column_block = columns if columns <= elements else max(4, elements // 4 * 4)
        packed = owner._packed_weights
        for r0 in range(0, rows, row_block):
            r1 = min(rows, r0 + row_block)
            for c0 in range(0, columns, column_block):
                c1 = min(columns, c0 + column_block)
                owner._check()
                try:
                    with owner._ram((r1-r0)*(c1-c0)*16 + 1024, "historical decoded router checksum tile"):
                        yield unpack_tile(packed, r0, r1, c0, c1).reshape(-1)
                finally:
                    if c0 == 0 and c1 == columns:
                        release_router_tensor_chunk(packed, r0 * packed.shape[1], (r1-r0) * packed.shape[1])
                    else:
                        for row in range(r0, r1):
                            release_router_tensor_chunk(packed, row * packed.shape[1] + c0//4, (c1+3)//4 - c0//4)


def update_historical_tensor_checksum(digest, tensor, owner):
    """Append persistence.tensor_checksum's exact dtype/shape/C-order format."""
    value = tensor.detach()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    elements = max(1, (owner._tile_budget() - 1024) // (value.element_size() * 3))

    def blocks(part):
        if part.numel() <= elements:
            yield part.contiguous().reshape(-1)
        elif part.is_contiguous():
            flat = part.reshape(-1)
            for start in range(0, flat.numel(), elements):
                yield flat[start:start+elements]
        else:
            tail = math.prod(part.shape[1:])
            if tail <= elements:
                rows = max(1, elements // max(1, tail))
                for start in range(0, part.shape[0], rows):
                    yield part[start:start+rows].contiguous().reshape(-1)
            else:
                for row in part:
                    yield from blocks(row)

    position = 0
    iterator = iter(blocks(value))
    while True:
        owner._check()
        source = None
        try:
            # A strided source may compact itself while advancing the block
            # iterator. Admit BEFORE next(), not after that allocation.
            with owner._ram(min(elements, value.numel()) * value.element_size() * 3 + 1024, "historical router control checksum tile"):
                try:
                    source = next(iterator)
                except StopIteration:
                    return
                block = source.to(device="cpu").contiguous()
                if block.numel() and block.stride(-1) != 1:
                    compact = torch.empty((block.numel(),), dtype=block.dtype, device="cpu")
                    compact.copy_(block)
                    block = compact
                digest.update(memoryview(block.view(torch.uint8).numpy()))
        finally:
            if source is not None and value.is_contiguous():
                release_router_tensor_chunk(value, position * value.element_size(), source.numel() * value.element_size())
        position += source.numel()


@dataclass
class _Backing:
    owner: Any
    name: str
    logical: int
    path: Optional[Path] = None
    mapping: Optional[mmap.mmap] = None
    lease: Any = None
    version: int = 0
    loaded: bool = True
    pointer: int = 0


class RouterStatePager:
    """Caller supplies residual budgets shared with core/activity, not extra RAM."""
    def __init__(self, directory: Path, *, hot_bytes: int, tile_bytes: int,
                 journal_ram_bytes: int, resource_policy=None, budget_provider=None,
                 cancelled: Optional[Callable[[], bool]] = None):
        parent = Path(directory)
        parent.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="router-session-", dir=str(parent)))
        self.hot_bytes, self.tile_bytes = max(0, int(hot_bytes)), max(0, int(tile_bytes))
        self.journal_ram_bytes = max(0, int(journal_ram_bytes))
        self.policy, self.budget_provider, self.cancelled = resource_policy, budget_provider, cancelled
        self._entries: dict[tuple[int, str], _Backing] = {}
        self._mapped_entries: dict[int, _Backing] = {}
        self._heap_owned_bytes = 0
        self._lock = threading.RLock()
        self._active = 0
        self._closed = False
        self.mutation_revision = 0
        self._cleanup: dict[str, Any] = {}
        self._hot_ranges = OrderedDict()
        self._hot_page_bytes = 0
        self._mapping_handles = []
        self._journals = weakref.WeakSet()

    @property
    def resource_policy(self):
        """Same writable policy binding as other native/activity pagers."""
        return self.policy

    @resource_policy.setter
    def resource_policy(self, value):
        self.policy = value

    def check(self):
        if self._closed:
            raise RuntimeError("router state pager is closed")
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("router operation cancelled at an exact tile boundary")

    def ram(self, amount: int, operation: str):
        reserve = getattr(self.policy, "reserve_ram", None)
        if callable(reserve):
            return reserve(int(amount), operation)
        if self.policy is not None:
            status = self.policy.status(estimated_ram_bytes=int(amount))
            if status.get("memoryPressure"):
                raise NeuralStateResourcePause(operation + " paused before allocation", status)
        return nullcontext()

    def spill(self, amount: int, operation: str, path: Path):
        reserve = getattr(self.policy, "reserve_spill", None)
        if callable(reserve):
            lease = reserve(int(amount), operation)
            lease.bind_path(path)
            return lease
        if self.policy is not None:
            self.policy.require_disk(int(amount), operation)
        return None

    @staticmethod
    def _mark(lease, amount):
        if hasattr(lease, "mark_allocated"):
            lease.mark_allocated(int(amount))

    def _register_entry(self, entry):
        key = (id(entry.owner), entry.name)
        prior = self._entries.get(key)
        if prior is not None and prior.path is None:
            self._heap_owned_bytes -= prior.logical
        self._entries[key] = entry
        if entry.path is None:
            self._heap_owned_bytes += entry.logical

    def _remove_entry(self, key):
        entry = self._entries.pop(key, None)
        if entry is not None and entry.path is None:
            self._heap_owned_bytes -= entry.logical
        return entry

    @contextmanager
    def construction(self, *, loading: bool = False):
        prior = set(self._entries)
        token = _CONSTRUCTION.set((self, bool(loading)))
        try:
            yield self
        except BaseException:
            for key in tuple(set(self._entries) - prior):
                entry = self._remove_entry(key)
                entry.owner._buffers[entry.name] = None
                self._retire(entry)
            raise
        finally:
            _CONSTRUCTION.reset(token)

    @contextmanager
    def operation(self, owner=None, *, validating=False):
        with self._lock:
            self.check()
            if owner is not None and getattr(owner, "_router_failed_journal", None) is not None:
                raise RuntimeError("router rollback was incomplete; reload its committed checkpoint")
            if owner is not None and not validating and any(not entry.loaded for entry in self._entries.values() if entry.owner is owner):
                raise RuntimeError("router used before its bounded checkpoint load completed")
            if not self._active and self.budget_provider is not None:
                budget = self.budget_provider(self.status())
                self.hot_bytes = max(0, int(budget["hotBytes"]))
                self.tile_bytes = max(0, int(budget["tileBytes"]))
                self.journal_ram_bytes = max(0, int(budget["journalRamBytes"]))
                self.cool_to_budget()
                self.warm_to_budget()
            self._active += 1
            try:
                yield self
            finally:
                self._active -= 1
                if not self._active:
                    self._observe_mutations()

    def allocate(self, owner, name: str, shape, dtype, *, fill, loading=False):
        self.check()
        if (id(owner), name) in self._entries:
            raise RuntimeError("router buffer is already registered with this pager")
        count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        entry = _Backing(owner, name, count, loaded=not loading)
        self._register_entry(entry)
        try:
            value = None
            # The pending entry is already included in heap_bytes(). Do not
            # count its destination twice and spill a fitting hot matrix.
            if self.heap_bytes() <= self.hot_bytes:
                try:
                    with self.ram(count, "router resident " + name) as lease:
                        value = torch.empty(shape, dtype=dtype, device="cpu")
                        if not loading:
                            value.fill_(fill)
                        self._mark(lease, count)
                except NeuralStateResourcePause:
                    value = None
            if value is None:
                value = self._map(entry, shape, dtype)
                if not loading and fill != 0:
                    flat = value.reshape(-1)
                    step = max(1, min(TRANSFER_BYTES, self.tile_bytes // 8) // value.element_size())
                    for start in range(0, flat.numel(), step):
                        self.check()
                        flat[start:start + step].fill_(fill)
                        self.release_chunk(value, start * value.element_size(), min(step, flat.numel() - start) * value.element_size())
                # Sparse newly truncated pages already contain zero bytes.
            entry.version = int(value._version)
            return value
        except BaseException:
            self._remove_entry((id(owner), name))
            self._retire(entry)
            raise

    def _map(self, entry, shape, dtype):
        path = self.directory / (uuid.uuid4().hex + ".router-page")
        allocation_upper_bound = promised_bytes(entry.logical) + 65536
        lease = self.spill(allocation_upper_bound, "mutable router/control backing", path)
        descriptor = None
        mapping = None
        was_current_heap = False
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.ftruncate(descriptor, entry.logical)
            mapping = mmap.mmap(descriptor, entry.logical, access=mmap.ACCESS_WRITE)
            value = torch.frombuffer(mapping, dtype=dtype).reshape(shape)
            was_current_heap = self._entries.get((id(entry.owner), entry.name)) is entry and entry.path is None
            if was_current_heap:
                self._heap_owned_bytes -= entry.logical
            entry.path, entry.mapping, entry.lease = path, mapping, lease
            entry.pointer = value.untyped_storage().data_ptr()
            self._mapped_entries[entry.pointer] = entry
            _MAPPED[value.untyped_storage().data_ptr()] = self
            self._mapping_handles.append((weakref.ref(mapping), entry.logical, path))
            if lease is not None:
                lease.commit(allocated_bytes(path, entry.logical), path=path, promised_bytes=promised_bytes(entry.logical), retain_open_backing=True)
                if hasattr(lease, "backing_closed"):
                    weakref.finalize(mapping, lease.backing_closed)
            return value
        except BaseException:
            if entry.pointer:
                _MAPPED.pop(entry.pointer, None)
                self._mapped_entries.pop(entry.pointer, None)
            if was_current_heap:
                self._heap_owned_bytes += entry.logical
            entry.path = entry.mapping = entry.lease = None
            entry.pointer = 0
            # Do not force-close a tensor-exported map on a failed commit.
            # Keep its inode charged until the last temporary/export dies.
            if mapping is not None:
                def release_failed_backing():
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        self._cleanup.setdefault("osHeldFiles", []).append(str(path))
                    if lease is not None:
                        lease.release()
                weakref.finalize(mapping, release_failed_backing)
            else:
                path.unlink(missing_ok=True)
                if lease is not None:
                    lease.release()
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def heap_bytes(self):
        return self._heap_owned_bytes

    def release_chunk(self, value, start: int, count: int):
        pointer = value.untyped_storage().data_ptr()
        entry = self._mapped_entries.get(pointer)
        if entry is None:
            return
        granularity = mmap.ALLOCATIONGRANULARITY
        begin = max(0, start // granularity * granularity)
        end = min(entry.logical, math.ceil((start + count) / granularity) * granularity)
        if end <= begin:
            return
        entry.mapping.flush(begin, end - begin)
        for position in range(begin, end, granularity):
            length = min(granularity, entry.logical - position)
            key = (pointer, position)
            if key in self._hot_ranges:
                self._hot_ranges.move_to_end(key)
            else:
                self._hot_ranges[key] = (entry, length)
                self._hot_page_bytes += length + 128
        self._evict_hot(max(0, self.hot_bytes - self.heap_bytes()))

    def _evict_hot(self, budget):
        while self._hot_ranges and self._hot_page_bytes > budget:
            (_, position), (entry, length) = self._hot_ranges.popitem(last=False)
            self._hot_page_bytes -= length + 128
            if hasattr(entry.mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                entry.mapping.madvise(mmap.MADV_DONTNEED, position, length)

    def release_tile(self, owner, r0, r1, c0, c1):
        for name in MATRIX_FIELDS:
            value = getattr(owner, name)
            a, b = (c0 // 4, (c1 + 3) // 4) if name == "_packed_weights" else (c0, c1)
            for row in range(r0, r1):
                self.release_chunk(value, (row * value.shape[1] + a) * value.element_size(), (b - a) * value.element_size())

    def finish_owner_load(self, owner, *, flush=True):
        for name in MATRIX_FIELDS:
            entry = self._entries.get((id(owner), name))
            if entry is not None:
                entry.loaded = True
                entry.version = int(getattr(owner, entry.name)._version)
        if flush:
            self.flush()

    def _observe_mutations(self):
        for entry in self._entries.values():
            value = getattr(entry.owner, entry.name, None)
            if value is None:
                continue
            version = int(value._version)
            if version != entry.version:
                self.mutation_revision += 1
                entry.version = version

    def flush(self):
        self._observe_mutations()
        for entry in self._entries.values():
            if entry.mapping is not None:
                entry.mapping.flush()
                if entry.lease is not None:
                    entry.lease.commit(allocated_bytes(entry.path, entry.logical), path=entry.path, promised_bytes=promised_bytes(entry.logical), retain_open_backing=True)

    def cool_to_budget(self):
        if self._active:
            return
        self._observe_mutations()
        for entry in tuple(self._entries.values()):
            if self.heap_bytes() <= self.hot_bytes:
                break
            if entry.path is not None or not entry.loaded:
                continue
            source = getattr(entry.owner, entry.name)
            pending = _Backing(entry.owner, entry.name, entry.logical, loaded=entry.loaded)
            target = self._map(pending, tuple(source.shape), source.dtype)
            self._register_entry(pending)
            try:
                step = max(1, min(TRANSFER_BYTES, self.tile_bytes // 8) // source.element_size())
                for start in range(0, source.numel(), step):
                    self.check()
                    target.reshape(-1)[start:start + step].copy_(source.reshape(-1)[start:start + step])
                    self.release_chunk(target, start * source.element_size(), min(step, source.numel() - start) * source.element_size())
                    # The old heap destination is still live until publication.
                    # Do not retain an additional mmap hot working set here.
                    self._evict_hot(0)
            except BaseException:
                self._register_entry(entry)
                self._retire(pending)
                raise
            entry.owner._buffers[entry.name] = target
            pending.version = int(target._version)
        self._evict_hot(max(0, self.hot_bytes - self.heap_bytes()))

    def warm_to_budget(self):
        """Recover fitting hot controls at a quiescent, admitted boundary."""
        if self._active:
            return
        self._observe_mutations()
        for entry in tuple(self._entries.values()):
            if entry.path is None or not entry.loaded or self.heap_bytes() + entry.logical > self.hot_bytes:
                continue
            self._evict_hot(0)
            source = getattr(entry.owner, entry.name)
            try:
                with self.ram(entry.logical, "recover hot router " + entry.name) as lease:
                    target = torch.empty_like(source, device="cpu")
                    step = max(1, min(TRANSFER_BYTES, self.tile_bytes // 8) // source.element_size())
                    for start in range(0, source.numel(), step):
                        self.check()
                        target.reshape(-1)[start:start + step].copy_(source.reshape(-1)[start:start + step])
                        self.release_chunk(source, start * source.element_size(), min(step, source.numel() - start) * source.element_size())
                        self._evict_hot(0)
                    self._mark(lease, entry.logical)
            except NeuralStateResourcePause:
                continue
            replacement = _Backing(entry.owner, entry.name, entry.logical, loaded=True,
                                   version=int(target._version))
            entry.owner._buffers[entry.name] = target
            self._register_entry(replacement)
            self._retire(entry)

    def release_owner(self, owner):
        """Retire an exact replaced population only after all jobs quiesce."""
        if self._active:
            raise RuntimeError("router owner retirement requires all operations to finish")
        for name in MATRIX_FIELDS:
            entry = self._remove_entry((id(owner), name))
            if entry is not None:
                owner._buffers[entry.name] = None
                self._retire(entry)

    def discard_unpublished_owner(self, owner):
        """Unwind only a newly staged sparse block under the operation lock.

        Existing owners may not be retired during an update. A sparse update
        marks its newly allocated block unpublished until its transaction
        commits; failed admission can therefore retire it without touching a
        previously learned connection or another operation's tensor exports.
        """
        with self._lock:
            if not getattr(owner, "_router_unpublished", False):
                raise RuntimeError("only an unpublished router block may be discarded during an operation")
            for name in MATRIX_FIELDS:
                entry = self._remove_entry((id(owner), name))
                if entry is not None:
                    owner._buffers[name] = None
                    self._retire(entry)

    def status(self):
        self._mapping_handles = [handle for handle in self._mapping_handles if handle[0]() is not None]
        return {"mode": "ram-first-tiled-router-mutable-controls", "cpuHeapBytes": self.heap_bytes(),
                "registeredFieldOwners": len(self._entries) // len(MATRIX_FIELDS),
                "registeredStateBytes": sum(entry.logical for entry in self._entries.values()),
                "mappedLogicalBytes": sum(entry.logical for entry in self._entries.values() if entry.path),
                "mappedAllocatedBytes": sum(allocated_bytes(entry.path, entry.logical) for entry in self._entries.values() if entry.path),
                "mappedPromisedBytes": sum(promised_bytes(entry.logical) for entry in self._entries.values() if entry.path),
                "tileBytes": self.tile_bytes, "journalRamBytes": self.journal_ram_bytes,
                "mappedHotRetainedBytesEstimate": self._hot_page_bytes,
                "mappedHotResidencyIsOsHint": True,
                "mappedBytesStillReferencedAfterClose": sum(size for ref, size, _ in self._mapping_handles if self._closed and ref() is not None),
                "hotTargetBytes": self.hot_bytes, "activeOperations": self._active,
                "rollbackJournalBytes": sum(journal.bytes_written for journal in self._journals),
                "rollbackFailureRetained": any(journal.rollback_failed for journal in self._journals),
                "checkpointWritableMapped": False, "closed": self._closed, "cleanup": dict(self._cleanup)}

    def _retire(self, entry):
        if entry.pointer:
            _MAPPED.pop(entry.pointer, None)
            self._mapped_entries.pop(entry.pointer, None)
        if entry.pointer:
            for position in range(0, entry.logical, mmap.ALLOCATIONGRANULARITY):
                cached = self._hot_ranges.pop((entry.pointer, position), None)
                if cached is not None:
                    self._hot_page_bytes -= cached[1] + 128
        if entry.path is not None:
            try:
                entry.path.unlink(missing_ok=True)
                if entry.lease is not None:
                    entry.lease.release()
            except OSError:
                self._cleanup.setdefault("osHeldFiles", []).append(str(entry.path))
        # Tensor exports keep their mmap alive. Never force-close it.

    def close(self):
        if self._active:
            raise RuntimeError("router cleanup requires all operations to finish")
        self.flush()
        for journal in tuple(self._journals):
            journal.discard_after_quiescence()
        for entry in tuple(self._entries.values()):
            entry.owner._buffers[entry.name] = None
            self._retire(entry)
        self._entries.clear()
        self._mapped_entries.clear()
        self._heap_owned_bytes = 0
        self._hot_ranges.clear(); self._hot_page_bytes = 0
        self._closed = True
        try:
            self.directory.rmdir()
        except OSError:
            self._cleanup["sessionDirectoryRetained"] = str(self.directory)
        return self.status()


class RouterMutationJournal:
    """Bounded old-byte blocks, RAM-first and sparse dirty-tile disk fallback."""
    def __init__(self, owner, pager: Optional[RouterStatePager], *, ram_bytes: int):
        from .parameter_diagnostics import active_diagnostic_scopes
        self._diagnostic_scopes = active_diagnostic_scopes()
        self.owner, self.pager, self.ram_limit = owner, pager, max(0, int(ram_bytes))
        self.records = []
        self.ram_used = 0
        self.path = None
        self.file = None
        self.last_lease = None
        self.failed_lease = None
        self.bytes_written = 0
        self.rollback_failed = False
        if pager is not None:
            pager._journals.add(self)

    def _resolve(self, name):
        # Sparse blocks use their registered, stable module path. Resolve the
        # current buffer on replay instead of retaining an old heap/mmap alias.
        if "." in name:
            path, field = name.rsplit(".", 1)
            return getattr(self.owner.get_submodule(path), field)
        return getattr(self.owner, name)

    def _capture(self, name, r0, r1, c0=None, c1=None):
        source = self._resolve(name)
        view = source.reshape(1) if source.ndim == 0 else source[r0:r1] if c0 is None else source[r0:r1, c0:c1]
        amount = view.numel() * view.element_size() + 256
        if self.pager is not None and amount > self.pager.tile_bytes + 256:
            raise NeuralStateResourcePause("router before-image requires a bounded tile", {"requiredBytes": amount})
        if self.pager is not None:
            lease = self.pager.ram(amount, "router rollback old bytes")
        else:
            lease = nullcontext()
        with lease as claim:
            before = view.detach().to(device="cpu", copy=True).contiguous()
            record = (name, r0, r1, c0, c1, before)
            if self.path is None and self.ram_used + amount <= self.ram_limit:
                self.records.append(record); self.ram_used += amount
                RouterStatePager._mark(claim, before.numel() * before.element_size())
            else:
                if self.pager is None:
                    raise NeuralStateResourcePause("router rollback exceeds bounded RAM without designated spill", {"requiredBytes": amount})
                self._ensure_file()
                for pending in self.records:
                    self._append(pending)
                self.records.clear(); self.ram_used = 0
                self._append(record)

    def capture(self, name, r0=0, r1=1, c0=None, c1=None):
        return self._capture(name, r0, r1, c0, c1)

    def _ensure_file(self):
        if self.path is None:
            self.path = self.pager.directory / (uuid.uuid4().hex + ".rollback")

    def _append(self, record):
        name, r0, r1, c0, c1, value = record
        header = json.dumps([name, r0, r1, c0, c1, list(value.shape), str(value.dtype)], separators=(",", ":")).encode()
        payload = memoryview(value.reshape(-1).view(torch.uint8).numpy())
        body_size = 4 + len(header) + len(payload) + 32 + 8
        old_size = self.file.seek(0, os.SEEK_END) if self.file is not None else 0
        claim = self.pager.spill(body_size + 65536, "router dirty-tile rollback", self.path)
        try:
            if self.file is None:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                self.file = os.fdopen(descriptor, "w+b", buffering=0)
            digest = hashlib.sha256(header); digest.update(payload)
            for block in (struct.pack("<I", len(header)), header, payload,
                          digest.digest(), struct.pack("<Q", body_size)):
                remaining = memoryview(block)
                while remaining:
                    written = self.file.write(remaining)
                    if written is None or written <= 0:
                        raise OSError("router rollback old-byte write was incomplete")
                    remaining = remaining[written:]
            if claim is not None:
                claim.commit(allocated_bytes(self.path, old_size + body_size), path=self.path)
                self.last_lease = claim
            self.bytes_written += body_size
        except BaseException:
            if self.file is not None:
                self.file.truncate(old_size)
            if claim is not None:
                self.failed_lease = claim
            raise

    def _restore(self, record):
        name, r0, r1, c0, c1, value = record
        target = self._resolve(name)
        view = target.reshape(1) if target.ndim == 0 else target[r0:r1] if c0 is None else target[r0:r1, c0:c1]
        if view.shape != value.shape or view.dtype != value.dtype:
            raise RuntimeError("router rollback target geometry changed")
        if target.dtype == torch.uint8:
            from .parameter_diagnostics import packed_diagnostic_restore
            module = self.owner.get_submodule(name.rsplit(".", 1)[0]) if "." in name else self.owner
            if target.ndim == 2 and c0 is not None:
                for row in range(r0, r1):
                    piece = value[row-r0].reshape(-1)
                    with packed_diagnostic_restore(module, target, piece,
                            start=row*target.shape[1]+c0, captured_scopes=self._diagnostic_scopes):
                        target[row, c0:c1].copy_(piece.to(target.device))
            else:
                start = 0 if target.ndim == 0 else r0 * math.prod(target.shape[1:])
                with packed_diagnostic_restore(module, target, value.reshape(-1),
                        start=start, captured_scopes=self._diagnostic_scopes):
                    view.copy_(value.to(device=view.device))
        else:
            view.copy_(value.to(device=view.device))

    @torch.no_grad()
    def rollback(self):
        try:
            for record in reversed(self.records):
                self._restore(record)
            if self.file is not None:
                end = self.file.seek(0, os.SEEK_END)
                dtypes = {str(value): value for value in (torch.uint8, torch.int16, torch.float32, torch.int64)}
                while end:
                    self.file.seek(end - 8); size = struct.unpack("<Q", self.file.read(8))[0]
                    start = end - size
                    maximum = (self.pager.tile_bytes if self.pager is not None else self.ram_limit) + 4096
                    if start < 0 or size < 44 or size > maximum:
                        raise ValueError("router rollback record is truncated")
                    self.file.seek(start); header_size = struct.unpack("<I", self.file.read(4))[0]
                    if header_size > 4096 or header_size > size - 44:
                        raise ValueError("router rollback header is invalid")
                    header = self.file.read(header_size); descriptor = json.loads(header)
                    payload_size = size - 4 - header_size - 32 - 8
                    payload = bytearray(self.file.read(payload_size)); digest = self.file.read(32)
                    proof = hashlib.sha256(header); proof.update(payload)
                    if len(payload) != payload_size or proof.digest() != digest:
                        raise ValueError("router rollback old-byte checksum mismatch")
                    name, r0, r1, c0, c1, shape, dtype = descriptor
                    value = torch.frombuffer(payload, dtype=dtypes[dtype]).reshape(shape)
                    self._restore((name, r0, r1, c0, c1, value)); end = start
        except BaseException:
            self.rollback_failed = True
            raise

    def close(self):
        if self.rollback_failed:
            self.owner._router_failed_journal = self
            return
        self.records.clear()
        if self.file is not None:
            self.file.close(); self.file = None
        if self.path is not None and not self.rollback_failed:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                if self.pager is not None:
                    self.pager._cleanup.setdefault("osHeldFiles", []).append(str(self.path))
            if self.failed_lease is not None:
                self.failed_lease.release()
            if self.last_lease is not None:
                self.last_lease.release()
        self.last_lease = self.failed_lease = None
        self.bytes_written = 0

    def discard_after_quiescence(self):
        if self.pager is None or self.pager._active:
            raise RuntimeError("failed router journal disposal requires shutdown quiescence")
        if self.rollback_failed:
            self.pager._cleanup["failedRollbackStateDiscardedDuringClose"] = True
        self.rollback_failed = False
        self.close()
        if getattr(self.owner, "_router_failed_journal", None) is self:
            del self.owner._router_failed_journal
