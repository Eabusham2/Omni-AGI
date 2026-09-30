"""RAM-first mutable residency for the *native packed model*, not replay.

Cold bytes live in private session-owned writable maps. Immutable checkpoint
blobs are never mapped writable. Accelerator working copies have one active
mutable owner and write back before eviction. No FP32 weight master exists.
CPU mmap reclamation is an OS hint, not a promised hard RSS/attention bound.
"""

from __future__ import annotations

import contextvars
import copy
import functools
import math
import mmap
import os
import threading
import tempfile
import time
import uuid
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch

from .bounded_tensor_io import TRANSFER_BYTES, BoundedTensorFile, atomic_save_tensors_bounded
from .packed_collective_hooks import packed_row_owner


_CONSTRUCTION = contextvars.ContextVar("omni_native_core_construction", default=None)
_MAPPED_PAGERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def shared_native_residency_budget(readings, *, baseline_bytes: int, current_core_heap_bytes: int = 0):
    """Partition one measured RAM envelope; never change neural dimensions."""
    selected = max(0, int(readings.get("systemRamBudgetBytes") or 0))
    resident = readings.get("admissionResidentMemoryBytes", readings.get("processMemoryBytes"))
    worker = readings.get("processMemoryBytes")
    other_managed = max(0, int(resident) - int(worker)) if resident is not None and worker is not None else 0
    baseline = min(selected, max(0, int(baseline_bytes)) + other_managed)
    residual = max(0, selected - baseline)
    core = residual * 55 // 100
    activity = residual * 30 // 100
    training = residual - core - activity
    live = int(readings.get("availableSafeRamBytes") or 0)
    if readings.get("availableMemoryBytes") is not None:
        live = int(readings["availableMemoryBytes"]) - int(readings.get("ramReserveBytes") or 0)
    if resident is not None:
        live = min(live, selected - int(resident))
    if readings.get("ramAdmissionVerified") is False:
        live = 0
    return {
        "selectedEnvelopeBytes": selected,
        "measuredBaselineBytes": baseline,
        "otherManagedProcessResidentBytes": other_managed,
        "residualEnvelopeBytes": residual,
        "corePartitionBytes": core,
        "workingActivityPartitionBytes": activity,
        "trainingTransferPartitionBytes": training,
        "liveCoreHotBytes": max(0, min(core, live + max(0, int(current_core_heap_bytes)))),
        "memoryPressure": bool(readings.get("memoryPressure")),
        "capacityPersistsAcrossPressure": True,
    }


@dataclass
class _Map:
    path: Path
    mapping: mmap.mmap
    byte_count: int
    dirty: bool = False


@dataclass
class _Owner:
    cold: dict[str, torch.Tensor] = field(default_factory=dict)
    maps: dict[str, _Map] = field(default_factory=dict)
    versions: dict[str, int] = field(default_factory=dict)
    dirty_names: set[str] = field(default_factory=set)
    device: Optional[torch.device] = None
    pins: int = 0
    heat: int = 0
    name: str = "<unbound>"
    loaded: bool = False


def current_native_core_pager() -> Optional["NativeCorePager"]:
    value = _CONSTRUCTION.get()
    return None if value is None else value[0]


def release_native_tensor_chunk(value: torch.Tensor, start: int, count: int) -> None:
    """Release mapped pages after streaming I/O, including detached state views."""
    if value.device.type != "cpu":
        return
    pointer = value.untyped_storage().data_ptr()
    pager = _MAPPED_PAGERS.get(pointer)
    if pager is not None:
        pager.release_chunk(value, start, count)


def paged_projection_call(method):
    """Admit the packed owner for any tensor-driven projection entry point."""
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        activity = next((value for value in args if isinstance(value, torch.Tensor)), None)
        if activity is None:
            activity = next((value for value in kwargs.values() if isinstance(value, torch.Tensor)), None)
        device = activity.device if activity is not None else self._native_compute_device
        with self._packed_residency_scope(device), packed_row_owner(self):
            result = method(self, *args, **kwargs)
            pager = getattr(self, "_native_core_pager", None)
            if method.__name__ == "forward" and pager is not None:
                pager.observe_output(self, activity, result)
            return result
    return wrapped


class NativePackedModuleMixin:
    """Projection allocation/movement hooks, inert outside native construction."""

    def _init_native_paging(self) -> None:
        context = _CONSTRUCTION.get()
        self._native_core_pager = None if context is None else context[0]
        self._native_loading_checkpoint = bool(context is not None and context[1])
        self._native_compute_device = torch.device("cpu")

    def _native_buffer(self, name: str, shape: tuple[int, ...], *, fill: Optional[int] = None) -> torch.Tensor:
        pager = self._native_core_pager
        if pager is None:
            value = torch.empty(shape, dtype=torch.uint8)
            if fill is not None:
                value.fill_(fill)
            return value
        return pager.allocate(self, name, shape, fill=fill)

    def _finish_native_paging(self) -> None:
        if self._native_core_pager is not None:
            self._native_core_pager.attach(self, loaded=not self._native_loading_checkpoint)

    def _packed_residency_scope(self, device: torch.device):
        pager = getattr(self, "_native_core_pager", None)
        return nullcontext() if pager is None else pager.use(self, device)

    def __deepcopy__(self, memo):
        """Isolated same-weight region snapshot; never copy a live pager/lock.

        Snapshot buffers are CPU packed bytes and independent leaf controls.
        The caller may move that private region to its admitted job device.
        This is not initialization or a float-weight master.
        """
        if getattr(self, "_online_transaction", None) is not None:
            raise RuntimeError("cannot snapshot an active packed online step")
        pager = getattr(self, "_native_core_pager", None)
        if pager is not None:
            pager.flush()
            total = sum(value.numel() * value.element_size() for value in self._buffers.values()
                        if isinstance(value, torch.Tensor))
            if pager.reserve_admission is not None:
                pager.reserve_admission(total, torch.device("cpu"))
        result = object.__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name == "_native_core_pager":
                result.__dict__[name] = None
            elif name == "_native_compute_device":
                result.__dict__[name] = torch.device("cpu")
            elif name == "_buffers":
                buffers = {}
                for key, tensor in value.items():
                    if tensor is None:
                        buffers[key] = None
                        continue
                    cloned = memo.get(id(tensor))
                    if cloned is None:
                        cloned = torch.empty_like(tensor, device="cpu", requires_grad=False)
                        if pager is not None:
                            pager._copy_bounded(tensor.detach(), cloned)
                        else:
                            cloned.copy_(tensor.detach())
                        cloned.requires_grad_(tensor.requires_grad)
                        memo[id(tensor)] = cloned
                    buffers[key] = cloned
                result.__dict__[name] = buffers
            else:
                result.__dict__[name] = copy.deepcopy(value, memo)
        for name in ("_packed_validated_version", "_bias_validated_version", "_scale_validated_version",
                     "_row_stability_validated_version", "_bias_row_stability_validated_version"):
            if hasattr(result, name):
                setattr(result, name, -1)
        result._validated_device = None
        return result

    def _apply(self, fn):
        pager = getattr(self, "_native_core_pager", None)
        if pager is None:
            return super()._apply(fn)
        # nn.Module.to must not eager-load the entire cold core onto a device.
        # Keep persistent owners cold; move derived trigger/index state only.
        pager.evict(self)
        saved = {name: value for name, value in self._buffers.items()
                 if name not in self._non_persistent_buffers_set}
        try:
            for name in saved:
                self._buffers[name] = None
            result = super()._apply(fn)
        finally:
            self._buffers.update(saved)
        trigger = getattr(self, "_autograd_trigger", None)
        if isinstance(trigger, torch.Tensor):
            self._native_compute_device = trigger.device
        return result


class NativeCorePager:
    """Access-ranked packed projection admission with bounded cold transfers.

    This bounds model **weight** residency. Saved activations, attention KV,
    online derivative scratch and arbitrary non-projection control registries
    have separate owners and are explicitly excluded from this contract.
    """

    def __init__(
        self,
        directory: Path,
        *,
        cpu_hot_bytes: int = 64 * 1024 * 1024,
        accelerator_hot_bytes: int = 64 * 1024 * 1024,
        reserve_disk: Optional[Callable[[int, str], Any]] = None,
        reserve_admission: Optional[Callable[[int, torch.device], Any]] = None,
        resource_pause: Optional[Callable[[str, dict[str, Any]], BaseException]] = None,
        budget_provider: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        resource_policy: Optional[Any] = None,
        chunk_bytes: int = TRANSFER_BYTES,
    ):
        self.directory = Path(directory)
        self.cpu_hot_bytes = max(0, int(cpu_hot_bytes))
        self.accelerator_hot_bytes = max(0, int(accelerator_hot_bytes))
        self.reserve_disk = reserve_disk
        self.reserve_admission = reserve_admission
        self.resource_pause = resource_pause
        self.budget_provider = budget_provider
        self.shared_resource_policy = resource_policy
        self._quota_leases: dict[Path, Any] = {}
        self.chunk_bytes = max(8, int(chunk_bytes))
        self._owners: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._map_by_pointer: dict[int, Any] = {}
        self._lock = threading.RLock()
        self._clock = 0
        self._revision = 0
        self._peak_transfer_bytes = 0
        self._page_ins = 0
        self._page_outs = 0
        self._writeback_bytes = 0
        self._spill_bytes = 0
        self._advice_calls = 0
        self._trace: list[dict[str, Any]] = []
        self._last_budget_refresh = 0.0
        self._budget_status: dict[str, Any] = {}
        self._owned_files: set[Path] = set()
        self._mapping_handles: list[tuple[Any, int, Path]] = []
        self._closed = False
        self._close_status: dict[str, Any] = {}
        self._observation_requests: dict[str, tuple[int, int]] = {}
        self._observations: dict[str, dict[str, Any]] = {}
        if self.budget_provider is not None:
            self.refresh_budget(force=True)

    @contextmanager
    def construction(self, *, from_checkpoint: bool = False) -> Iterator[None]:
        prior_owners = set(self._owners)
        prior_files = set(self._owned_files)
        token = _CONSTRUCTION.set((self, bool(from_checkpoint)))
        try:
            yield
        except BaseException:
            # Partial owners may still have exported tensor references. Unlink
            # only this session's files; never force-close an accessible map.
            try:
                for module in list(self._owners):
                    if module in prior_owners:
                        continue
                    owner = self._owners[module]
                    if owner.pins:
                        raise RuntimeError("failed construction still has active packed jobs")
                    for name in owner.cold:
                        module._buffers[name] = None
                    owner.cold.clear()
                    owner.maps.clear()
                    del self._owners[module]
                unlinked = sum(int(self._unlink_owned_page(path))
                               for path in tuple(self._owned_files - prior_files))
                self._close_status["failedConstructionCleanup"] = {
                    "filesUnlinked": unlinked, "checkpointDeleted": False,
                    "mappingForceClosed": False,
                    "osHeldFiles": len(self._owned_files - prior_files),
                }
            except Exception as cleanup_error:
                self._close_status["failedConstructionCleanup"] = {"cleanupError": str(cleanup_error)}
            raise
        finally:
            _CONSTRUCTION.reset(token)

    def _event(self, kind: str, owner: _Owner, byte_count: int) -> None:
        self._trace.append({"event": kind, "owner": owner.name, "bytes": int(byte_count)})
        del self._trace[:-64]

    def _pause(self, message: str) -> None:
        if self.resource_pause is not None:
            raise self.resource_pause(message, self.status())
        raise RuntimeError(message)

    def _heap_bytes(self) -> int:
        return sum(value.numel() * value.element_size()
                   for owner in self._owners.values()
                   for name, value in owner.cold.items() if name not in owner.maps)

    def _accelerator_bytes(self) -> int:
        return sum(self._owner_bytes(owner) for owner in self._owners.values()
                   if owner.device is not None)

    @staticmethod
    def _owner_bytes(owner: _Owner) -> int:
        return sum(value.numel() * value.element_size() for value in owner.cold.values())

    @staticmethod
    def _page_allocated_bytes(page: _Map) -> tuple[int, str]:
        try:
            stored = page.path.stat()
            blocks = getattr(stored, "st_blocks", None)
            if isinstance(blocks, int):
                return max(0, blocks * 512), "posix-st_blocks-times-512"
        except OSError:
            pass
        return page.byte_count, "conservative-logical-bytes-unmeasured"

    def _map_tensor(self, owner: _Owner, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        byte_count = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if byte_count == 0:
            return torch.empty(shape, dtype=dtype)
        # Account for sparse-file promises, not just currently allocated
        # blocks. This is conservative while a load is still filling maps.
        promised = sum(max(0, page.byte_count - self._page_allocated_bytes(page)[0])
                       for entry in self._owners.values() for page in entry.maps.values())
        if self.reserve_disk is not None:
            self.reserve_disk(byte_count + promised, "native packed model spill")
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (uuid.uuid4().hex + ".packed-page")
        reserve = getattr(self.shared_resource_policy, "reserve_spill", None)
        quota = reserve(((byte_count + 65535) // 65536) * 65536, "native packed model spill") if callable(reserve) else None
        if quota is not None:
            quota.bind_path(path)
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except BaseException:
            if quota is not None:
                quota.release()
            raise
        try:
            if hasattr(os, "posix_fallocate"):
                os.posix_fallocate(descriptor, 0, byte_count)
            else:
                os.ftruncate(descriptor, byte_count)
            mapping = mmap.mmap(descriptor, byte_count, access=mmap.ACCESS_WRITE)
        except BaseException:
            os.close(descriptor)
            path.unlink(missing_ok=True)
            if quota is not None:
                quota.release()
            raise
        os.close(descriptor)
        if quota is not None:
            try:
                quota.commit(path=path, promised_bytes=byte_count, retain_open_backing=True)
            except BaseException:
                mapping.close()
                path.unlink(missing_ok=True)
                quota.release()
                raise
            self._quota_leases[path] = quota
            weakref.finalize(mapping, quota.backing_closed)
        self._owned_files.add(path)
        owner.maps[name] = _Map(path, mapping, byte_count)
        self._mapping_handles.append((weakref.ref(mapping), byte_count, path))
        value = torch.frombuffer(mapping, dtype=dtype).reshape(shape)
        pointer = value.untyped_storage().data_ptr()
        self._map_by_pointer[pointer] = weakref.ref(owner.maps[name])
        _MAPPED_PAGERS[pointer] = self
        return value

    def allocate(self, module: torch.nn.Module, name: str, shape: tuple[int, ...], *, fill: Optional[int] = None) -> torch.Tensor:
        self.refresh_budget()
        with self._lock:
            if self._closed:
                raise RuntimeError("native packed model pager is closed")
            owner = self._owners.setdefault(module, _Owner())
            byte_count = math.prod(shape)
            if self._heap_bytes() + byte_count <= self.cpu_hot_bytes:
                reserve_ram = getattr(self.shared_resource_policy, "reserve_ram", None)
                lease = reserve_ram(byte_count, "native packed hot owner") if callable(reserve_ram) else nullcontext()
                with lease as held:
                    value = torch.empty(shape, dtype=torch.uint8)
                    if fill is not None:
                        value.fill_(fill)
                    if held is not None:
                        held.mark_allocated(byte_count)
            else:
                value = self._map_tensor(owner, name, shape, torch.uint8)
                self._spill_bytes += byte_count
            owner.cold[name] = value
            if fill is not None and name in owner.maps:
                value.fill_(fill)
                owner.maps[name].dirty = True
            return value

    def attach(self, module: torch.nn.Module, *, loaded: bool) -> None:
        with self._lock:
            owner = self._owners.setdefault(module, _Owner())
            owner.cold = {name: value for name, value in module._buffers.items()
                          if isinstance(value, torch.Tensor)
                          and name not in module._non_persistent_buffers_set}
            owner.versions = {name: int(value._version) for name, value in owner.cold.items()}
            owner.loaded = bool(loaded)
            self._clock += 1
            owner.heat = self._clock
            # Initialization touches cold bytes once; promptly make them
            # reclaimable instead of accumulating a full physical model.
            for page in owner.maps.values():
                page.dirty = bool(loaded)
                self._release_page(page)

    def bind_names(self, roots) -> None:
        for prefix, root in roots:
            for name, module in root.named_modules():
                owner = self._owners.get(module)
                if owner is not None:
                    owner.name = prefix + name

    def observation(self, module_name: str, *, enabled: bool, start: int = 0, count: int = 64) -> dict[str, Any]:
        """Retarget one bounded viewer viewport; never represent all rows as seen."""
        with self._lock:
            if start < 0 or count < 1:
                raise ValueError("invalid observed activation viewport")
            # Transfer envelope is resource-derived and can be retargeted to
            # any row; it is not a model/neuron cardinality restriction.
            limit = max(1, min(count, max(1, self.chunk_bytes // 32)))
            if enabled:
                if self._observation_requests.get(module_name) != (int(start), int(limit)):
                    self._observations.clear()
                self._observation_requests = {module_name: (int(start), int(limit))}
            else:
                self._observation_requests.clear()
            value = self._observations.get(module_name)
            return {"available": True, "enabled": enabled, "module": module_name,
                    "requestedStart": start, "requestedCount": count, "admittedCount": limit,
                    "observation": value, "observed": value is not None,
                    "evidence": "actual-forward-output-viewport-not-confidence-or-all-neuron-firing"}

    def observe_output(self, module, inputs, output) -> None:
        owner = self._owners.get(module)
        if owner is None or owner.name not in self._observation_requests or not isinstance(output, torch.Tensor):
            return
        start, count = self._observation_requests[owner.name]
        if output.numel() == 0:
            return
        axis_size = int(output.shape[1]) if getattr(module, "dimensions", 0) in (1, 2, 3) and hasattr(module, "out_channels") else int(output.shape[-1])
        if start >= axis_size:
            return
        if getattr(module, "dimensions", 0) in (1, 2, 3) and hasattr(module, "out_channels"):
            stop = min(int(output.shape[1]), start + count)
            selected = output.detach()[0, start:stop].reshape(stop - start, -1)[:, -1]
            axis, input_row = "output-channel", None
        else:
            stop = min(int(output.shape[-1]), start + count)
            selected = output.detach().reshape(-1, int(output.shape[-1]))[-1, start:stop]
            axis = "embedding-column" if hasattr(module, "num_embeddings") else "output-row"
            input_row = int(inputs.detach().reshape(-1)[-1].item()) if hasattr(module, "num_embeddings") and isinstance(inputs, torch.Tensor) else None
        if stop <= start:
            return
        values = selected.to(device="cpu", dtype=torch.float32).tolist()
        self._observations[owner.name] = {"observedAtUnix": time.time(), "axis": axis,
            "start": start, "end": stop, "values": [value if math.isfinite(value) else None for value in values],
            "embeddingRowObserved": input_row, "sample": "last-forward-token-or-spatial-position",
            "fullPopulationObserved": False, "firingClassification": None}

    def finish_load(self) -> None:
        with self._lock:
            for module, owner in self._owners.items():
                validate = getattr(module, "_validate_packed", None)
                if callable(validate):
                    validate()
                owner.loaded = True
                module._native_loading_checkpoint = False
                owner.versions = {name: int(value._version) for name, value in owner.cold.items()}
                for page in owner.maps.values():
                    page.dirty = True
                    self._release_page(page)
            self.refresh_budget(force=True)

    def _invalidate_validation(self, module) -> None:
        for name in ("_packed_validated_version", "_bias_validated_version", "_scale_validated_version",
                     "_row_stability_validated_version", "_bias_row_stability_validated_version"):
            if hasattr(module, name):
                setattr(module, name, -1)
        if hasattr(module, "_validated_device"):
            module._validated_device = None

    def _observe_mutations(self, module, owner: _Owner) -> None:
        changed = False
        for name in owner.cold:
            value = module._buffers.get(name)
            if not isinstance(value, torch.Tensor):
                continue
            version = int(value._version)
            if owner.versions.get(name) != version:
                changed = True
                owner.versions[name] = version
                if owner.device is None and name in owner.maps:
                    owner.maps[name].dirty = True
                elif owner.device is not None:
                    owner.dirty_names.add(name)
        if changed:
            self._revision += 1

    @property
    def mutation_revision(self) -> int:
        with self._lock:
            for module, owner in self._owners.items():
                self._observe_mutations(module, owner)
            return self._revision

    @contextmanager
    def use(self, module: torch.nn.Module, device: torch.device) -> Iterator[None]:
        device = torch.device(device)
        self.refresh_budget()
        with self._lock:
            if self._closed:
                raise RuntimeError("native packed model pager is closed")
            owner = self._owners[module]
            if not owner.loaded:
                raise RuntimeError("native packed owner used before its checkpoint load completed")
            if owner.pins and owner.device != (None if device.type == "cpu" else device):
                raise RuntimeError("cannot change a pinned packed owner's device")
            self._observe_mutations(module, owner)
            self._clock += 1
            owner.heat = self._clock
            if device.type == "cpu":
                if owner.device is not None:
                    self.evict(module)
            elif owner.device != device:
                self.evict(module)
                required = self._owner_bytes(owner)
                if required > self.accelerator_hot_bytes:
                    self._pause("one packed projection exceeds the accelerator residency budget; use CPU or a larger safe budget")
                for other, cold in sorted(list(self._owners.items()), key=lambda pair: pair[1].heat):
                    if self._accelerator_bytes() + required <= self.accelerator_hot_bytes:
                        break
                    if other is not module and cold.device is not None and not cold.pins:
                        self.evict(other)
                if self._accelerator_bytes() + required > self.accelerator_hot_bytes:
                    self._pause("pinned packed projections exhaust the accelerator residency budget")
                if self.reserve_admission is not None:
                    self.reserve_admission(required, device)
                admitted: dict[str, torch.Tensor] = {}
                try:
                    reserve_ram = getattr(self.shared_resource_policy, "reserve_ram", None)
                    reservation = reserve_ram(required, "native unified-memory packed admission") if callable(reserve_ram) and device.type != "cuda" else nullcontext()
                    with reservation as held:
                        for name, value in owner.cold.items():
                            admitted[name] = torch.empty_like(value, device=device)
                            self._copy_bounded(value, admitted[name])
                        if held is not None:
                            held.mark_allocated(required)
                except BaseException:
                    admitted.clear()
                    raise
                module._buffers.update(admitted)
                owner.device = device
                owner.versions = {name: int(value._version) for name, value in admitted.items()}
                owner.dirty_names.clear()
                self._invalidate_validation(module)
                self._page_ins += 1
                self._event("accelerator-admit", owner, required)
            owner.pins += 1
        try:
            yield
        finally:
            with self._lock:
                owner.pins -= 1
                self._observe_mutations(module, owner)
                # Hot device owners are retained under budget. Writeback is
                # on eviction/save, not on every fitting forward invocation.
                if owner.device is None:
                    self.cool_to_budget()

    @torch.no_grad()
    def _copy_bounded(self, source: torch.Tensor, target: torch.Tensor) -> int:
        source_flat, target_flat = source.reshape(-1), target.reshape(-1)
        elements = max(1, self.chunk_bytes // source.element_size())
        copied = 0
        for start in range(0, source.numel(), elements):
            end = min(source.numel(), start + elements)
            target_flat[start:end].copy_(source_flat[start:end], non_blocking=False)
            count = (end - start) * source.element_size()
            copied += count
            self._peak_transfer_bytes = max(self._peak_transfer_bytes, count)
        return copied

    def evict(self, module: torch.nn.Module) -> None:
        with self._lock:
            owner = self._owners.get(module)
            if owner is None or owner.device is None:
                return
            if owner.pins:
                raise RuntimeError("cannot evict a pinned packed projection")
            for name, target in owner.cold.items():
                source = module._buffers[name]
                if name in owner.dirty_names or int(source._version) != owner.versions.get(name):
                    copied = self._copy_bounded(source, target)
                    self._writeback_bytes += copied
                    if name in owner.maps:
                        owner.maps[name].dirty = True
                    self._revision += 1
            # Blocking copies completed before the mutable owner changes.
            module._buffers.update(owner.cold)
            owner.device = None
            owner.dirty_names.clear()
            owner.versions = {name: int(value._version) for name, value in owner.cold.items()}
            self._invalidate_validation(module)
            self._page_outs += 1
            self._event("accelerator-evict", owner, self._owner_bytes(owner))

    def _release_page(self, page: _Map) -> None:
        if page.dirty:
            page.mapping.flush()
            page.dirty = False
        if hasattr(page.mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
            page.mapping.madvise(mmap.MADV_DONTNEED)
            self._advice_calls += 1

    def release_chunk(self, value: torch.Tensor, start: int, count: int) -> None:
        """Make completed byte ranges reclaimable during a long load/save."""
        with self._lock:
            reference = self._map_by_pointer.get(value.untyped_storage().data_ptr())
            page = None if reference is None else reference()
            if page is None or count <= 0:
                return
            alignment = max(mmap.PAGESIZE, mmap.ALLOCATIONGRANULARITY)
            begin = max(0, int(start) // alignment * alignment)
            end = min(page.byte_count, int(start) + int(count))
            if end <= begin:
                return
            page.mapping.flush(begin, end - begin)
            if hasattr(page.mapping, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
                page.mapping.madvise(mmap.MADV_DONTNEED, begin, end - begin)
                self._advice_calls += 1

    def cool_to_budget(self, cpu_hot_bytes: Optional[int] = None) -> None:
        """Spill cold heap owners on pressure; never drop/update learned bytes."""
        with self._lock:
            if cpu_hot_bytes is not None:
                self.cpu_hot_bytes = max(0, int(cpu_hot_bytes))
            for module, owner in sorted(list(self._owners.items()), key=lambda pair: pair[1].heat):
                if owner.pins or not owner.loaded:
                    continue
                if self._heap_bytes() > self.cpu_hot_bytes:
                    if owner.device is not None:
                        self.evict(module)
                    for name, source in list(owner.cold.items()):
                        if name in owner.maps or source.numel() == 0:
                            continue
                        # Do not churn tiny controls onto disk. The few bytes
                        # stay explicit in heap accounting above the target.
                        if source.numel() * source.element_size() < 1024:
                            continue
                        target = self._map_tensor(owner, name, tuple(source.shape), source.dtype)
                        self._copy_bounded(source, target)
                        owner.cold[name] = target
                        owner.maps[name].dirty = True
                        self._spill_bytes += target.numel() * target.element_size()
                        if owner.device is None:
                            module._buffers[name] = target
                            owner.versions[name] = int(target._version)
                            self._invalidate_validation(module)
                        self._event("cpu-cold-spill", owner, target.numel() * target.element_size())
                for page in owner.maps.values():
                    self._release_page(page)

    def refresh_budget(self, *, force: bool = False) -> None:
        """Refresh live partitions, spill cold bytes and recover safe hot RAM."""
        if self.budget_provider is None or self._closed:
            return
        now = time.monotonic()
        if not force and now - self._last_budget_refresh < 1.0:
            return
        with self._lock:
            self._last_budget_refresh = now
            budget = dict(self.budget_provider(self.status()))
            self._budget_status = budget
            self.cpu_hot_bytes = max(0, int(budget["liveCoreHotBytes"]))
            self.accelerator_hot_bytes = max(0, int(budget.get("acceleratorCoreHotBytes", self.accelerator_hot_bytes)))
            for module, owner in sorted(list(self._owners.items()), key=lambda pair: pair[1].heat):
                if self._accelerator_bytes() <= self.accelerator_hot_bytes:
                    break
                if owner.device is not None and not owner.pins:
                    self.evict(module)
            self.cool_to_budget()
            self.promote_hot_to_budget()

    def _unlink_owned_page(self, path: Path) -> bool:
        if path not in self._owned_files or path.parent.resolve() != self.directory.resolve():
            raise ValueError("refusing cleanup of a non-session native spill file")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Windows may retain a mapped file. No force-close: exported
            # tensors remain valid, and residue is surfaced at close/status.
            return False
        self._owned_files.discard(path)
        quota = self._quota_leases.pop(path, None)
        if quota is not None:
            quota.release()
        return True

    def promote_hot_to_budget(self) -> None:
        """Restore fitting access-hot packed bytes to heap without a master."""
        with self._lock:
            heap = self._heap_bytes()
            growth = self.cpu_hot_bytes - heap
            total = heap + sum(page.byte_count for owner in self._owners.values() for page in owner.maps.values())
            # Small availability jitter must not repeatedly rewrite cold
            # weights. Recover a whole fitting core immediately; otherwise
            # require meaningful headroom before moving cold bytes back.
            if total > self.cpu_hot_bytes and growth < max(self.chunk_bytes, self.cpu_hot_bytes // 10):
                return
            for module, owner in sorted(list(self._owners.items()), key=lambda pair: -pair[1].heat):
                if owner.pins or owner.device is not None or not owner.loaded:
                    continue
                for name, page in list(owner.maps.items()):
                    source = owner.cold[name]
                    byte_count = source.numel() * source.element_size()
                    if self._heap_bytes() + byte_count > self.cpu_hot_bytes:
                        continue
                    if self.reserve_admission is not None:
                        self.reserve_admission(byte_count, torch.device("cpu"))
                    reserve_ram = getattr(self.shared_resource_policy, "reserve_ram", None)
                    reservation = reserve_ram(byte_count, "native cold-to-hot packed promotion") if callable(reserve_ram) else nullcontext()
                    with reservation as held:
                        target = torch.empty_like(source, device="cpu")
                        self._copy_bounded(source, target)
                        if held is not None:
                            held.mark_allocated(byte_count)
                    owner.cold[name] = target
                    module._buffers[name] = target
                    del owner.maps[name]
                    owner.versions[name] = int(target._version)
                    self._invalidate_validation(module)
                    self._unlink_owned_page(page.path)
                    self._event("cpu-hot-promote", owner, byte_count)

    def flush(self) -> None:
        """Quiesce device copies before checksum/snapshot/save/export."""
        with self._lock:
            if any(owner.pins for owner in self._owners.values()):
                raise RuntimeError("cannot checkpoint a pinned packed projection")
            for module, owner in list(self._owners.items()):
                self._observe_mutations(module, owner)
                self.evict(module)
                for page in owner.maps.values():
                    self._release_page(page)

    def status(self) -> dict[str, Any]:
        with self._lock:
            mapped = sum(page.byte_count for owner in self._owners.values() for page in owner.maps.values())
            allocations = [self._page_allocated_bytes(page) for owner in self._owners.values() for page in owner.maps.values()]
            return {
                "mode": "ram-first-native-packed-projection-paging",
                "weightAuthority": "mutable-packed-bytes-with-immutable-checkpoint-generations",
                "checkpointWritableMapped": False,
                "fp32WeightMaster": False,
                "cpuHeapBytes": self._heap_bytes(),
                "cpuMappedLogicalBytes": mapped,
                "cpuMappedAllocatedBytes": sum(count for count, _basis in allocations),
                "cpuMappedAllocationBasis": "posix-st_blocks-times-512" if all(basis == "posix-st_blocks-times-512" for _count, basis in allocations) else "mixed-or-conservative-logical-bytes",
                "cpuHotTargetBytes": self.cpu_hot_bytes,
                "cpuReclamation": "os-madvise-hint" if hasattr(mmap.mmap, "madvise") else "os-file-backed-reclamation",
                "hardCpuRssBound": False,
                "acceleratorAdmittedBytes": self._accelerator_bytes(),
                "acceleratorPackedBudgetBytes": self.accelerator_hot_bytes,
                "acceleratorPageIns": self._page_ins,
                "acceleratorPageOuts": self._page_outs,
                "acceleratorWritebackBytes": self._writeback_bytes,
                "cpuSpillBytes": self._spill_bytes,
                "reclaimAdviceCalls": self._advice_calls,
                "peakTransferScratchBytes": self._peak_transfer_bytes,
                "registeredPackedOwners": len(self._owners),
                "maximumPackedOwnerBytes": max((self._owner_bytes(owner) for owner in self._owners.values()), default=0),
                "totalPackedOwnerBytes": sum(self._owner_bytes(owner) for owner in self._owners.values()),
                "pinnedOwners": sum(owner.pins > 0 for owner in self._owners.values()),
                "attentionKvIncluded": False,
                "savedAutogradActivationsIncluded": False,
                "trace": list(self._trace),
                "sharedBudget": dict(self._budget_status),
                "closed": self._closed,
                "cleanup": dict(self._close_status),
                "sessionOwnedFiles": len(self._owned_files),
            }

    def close(self) -> dict[str, Any]:
        """After job cleanup, retire exactly these scratch files, not checkpoints.

        Do not call mmap.close(): exported Tensor views own mappings until their
        references disappear. POSIX unlink preserves those views; OS-held files
        on other systems remain reported and can be cleaned after process exit.
        """
        with self._lock:
            if self._closed:
                return dict(self._close_status)
            if any(owner.pins for owner in self._owners.values()):
                raise RuntimeError("native spill cleanup requires all packed jobs to finish")
            self.flush()
            # Drop only the pager's packed owners after all scoped jobs ended.
            # Exported tensor/storage views retain their own raw mmap objects.
            for module, owner in list(self._owners.items()):
                for name in owner.cold:
                    module._buffers[name] = None
                owner.cold.clear()
                owner.maps.clear()
            self._owners.clear()
            unlinked = 0
            for path in tuple(self._owned_files):
                unlinked += int(self._unlink_owned_page(path))
            residue = []
            for path in self._owned_files:
                try:
                    residue.append({"path": str(path), "bytes": path.stat().st_size})
                except OSError:
                    residue.append({"path": str(path), "bytes": None})
            removed_directory = False
            if self.directory.is_dir() and not self.directory.is_symlink():
                try:
                    self.directory.rmdir()
                    removed_directory = True
                except OSError:
                    pass
            self._closed = True
            self.budget_provider = None
            self.reserve_admission = None
            self.resource_pause = None
            self.reserve_disk = None
            self._close_status = {
                "closed": True, "filesUnlinked": unlinked,
                "mappingForceClosed": False, "checkpointDeleted": False,
                "directoryRemoved": removed_directory,
                "osHeldResidue": residue,
                "osHeldResidueBytes": sum(int(item["bytes"] or 0) for item in residue),
                "mappedBytesRetainedByTensorViews": sum(
                    byte_count for reference, byte_count, _ in self._mapping_handles
                    if reference() is not None
                ),
            }
            return dict(self._close_status)


class BoundedPackedRollback:
    """Private disk rollback for packed codes/resistance, without heap clones.

    Restore resolves registered buffer names after paging has changed tensor
    objects. Snapshot writes are whole packed-state checkpoints, not a claimed
    dirty-row journal; their I/O cost is reported by ``byte_count``.
    """

    def __init__(self, directory: Path, path: Path, entries, byte_count: int):
        self.directory = directory
        self.path = path
        self.entries = entries
        self.byte_count = int(byte_count)
        from .parameter_diagnostics import active_diagnostic_scopes
        self._diagnostic_scopes = active_diagnostic_scopes()

    @classmethod
    def capture(cls, roots, *, directory: Path, reserve_disk=None):
        roots = tuple(roots)
        pagers = {getattr(module, "_native_core_pager", None)
                  for root in roots for module in root.modules()}
        for pager in pagers - {None}:
            pager.flush()
        entries = []
        tensors = {}
        seen = set()
        for root in roots:
            for module in root.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                getter = getattr(module, "authoritative_packed_tensors", None)
                if not callable(getter):
                    continue
                identities = {id(value) for value in getter()}
                names = []
                for name, value in module.named_buffers(recurse=False):
                    if id(value) in identities or name in {"_row_stability", "_bias_row_stability"}:
                        if value.dtype != torch.uint8:
                            raise ValueError("packed rollback contains non-uint8 state")
                        key = "%d.%s" % (len(entries), name)
                        names.append((name, key))
                        tensors[key] = value
                if len(identities) > len(names):
                    raise ValueError("packed rollback owner is not a registered buffer")
                entries.append((module, names, int(getattr(module, "_pending_stability_events", 0))))
        byte_count = sum(value.numel() for value in tensors.values()) + 4096 + len(tensors) * 256
        if reserve_disk is not None:
            reserve_disk(byte_count, "native packed rollback snapshot")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        private = Path(tempfile.mkdtemp(prefix="packed-rollback-", dir=str(directory)))
        path = private / "packed.safetensors"
        try:
            atomic_save_tensors_bounded(path, tensors, metadata={"format": "omni-packed-rollback"})
        except BaseException:
            path.unlink(missing_ok=True)
            private.rmdir()
            raise
        return cls(private, path, entries, byte_count)

    def restore(self) -> None:
        pagers = {getattr(module, "_native_core_pager", None) for module, _, _ in self.entries}
        for pager in pagers - {None}:
            pager.flush()
        reader = BoundedTensorFile(self.path)
        for module, names, pending in self.entries:
            for name, key in names:
                target = module._buffers.get(name)
                if not isinstance(target, torch.Tensor):
                    raise RuntimeError("packed rollback target disappeared")
                spec = reader.specs[key]
                if target.dtype != spec.dtype or tuple(target.shape) != spec.shape or not target.is_contiguous():
                    raise RuntimeError("packed rollback target changed shape, dtype or contiguity")
                from .parameter_diagnostics import packed_diagnostic_restore
                flat = target.reshape(-1)
                offset = 0
                for source in reader.chunks(key):
                    with packed_diagnostic_restore(module, target, source, start=offset, captured_scopes=self._diagnostic_scopes):
                        flat[offset:offset + source.numel()].copy_(source)
                    release_native_tensor_chunk(target, offset, source.numel())
                    offset += source.numel()
            if hasattr(module, "_pending_stability_events"):
                module._pending_stability_events = pending
            validate = getattr(module, "_validate_packed", None)
            if callable(validate):
                validate()
        for pager in pagers - {None}:
            pager.flush()

    def close(self) -> None:
        self.path.unlink(missing_ok=True)
        if self.directory.exists():
            self.directory.rmdir()
