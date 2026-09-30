"""Optimizer adapter for packed-authoritative online synapses.

Packed projections update their ternary codes during backward and have no
floating weight Parameter or dense Adam moments. The adapter still accepts
floating Parameters if a separate non-native objective supplies them; a
native packed-only brain uses the no-op optimizer facade.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional
from pathlib import Path
from contextvars import ContextVar
import functools

import torch
from torch import nn
from .parameter_diagnostics import packed_diagnostic_restore, active_diagnostic_scopes


_BATCH_SNAPSHOTS = ContextVar("logical_packed_batch_snapshots", default=None)


def packed_snapshot_lifetime(method):
    """Keep rollback across all retries; close only at logical-batch exit."""
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        snapshots = []
        token = _BATCH_SNAPSHOTS.set(snapshots)
        try:
            return method(*args, **kwargs)
        finally:
            try:
                for snapshot in reversed(snapshots):
                    snapshot.close()
            finally:
                _BATCH_SNAPSHOTS.reset(token)
    return wrapped


class PackedOnlyOptimizer:
    """No-op optimizer API when backward itself updated all learned weights."""

    def __init__(self) -> None:
        self.state: dict[Any, Any] = {}
        self.param_groups: list[Any] = []

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        del set_to_none

    def step(self, closure: Optional[Callable[[], Any]] = None) -> Any:
        return closure() if closure is not None else None

    def state_dict(self) -> dict[str, Any]:
        return {"state": {}, "param_groups": []}

    def load_state_dict(self, value: dict[str, Any]) -> None:
        if value.get("state") or value.get("param_groups"):
            raise ValueError("packed-only objective has no dense optimizer state")


class PackedMutationSnapshot:
    """CPU-byte rollback point for one logical packed-gradient batch.

    Snapshots contain only authoritative uint8 codes, uint8 row resistance,
    and tiny event counters. No unpacked or floating weight mirror is made.
    The caller checks a host-memory reserve before any copy is allocated. A
    future dirty-row journal could reduce this full packed-byte copy, but is
    not assumed by the current retry contract.
    """

    def __init__(self, entries: list[tuple[nn.Module, list[tuple[str, torch.Tensor]], int]], byte_count: int, *, disk_snapshot=None, disk_ram_reserve=None, disk_io_ram_bytes=0):
        self._entries = entries
        self.byte_count = byte_count
        self._disk_snapshot = disk_snapshot
        self._disk_ram_reserve = disk_ram_reserve
        self._disk_io_ram_bytes = int(disk_io_ram_bytes)
        self._diagnostic_scopes = active_diagnostic_scopes()
        self.storage_mode = "private-disk-packed-snapshot" if disk_snapshot is not None else "reserved-cpu-packed-bytes"
        active = _BATCH_SNAPSHOTS.get()
        if active is not None:
            active.append(self)

    @classmethod
    def capture(
        cls,
        roots: Iterable[nn.Module],
        *,
        reserve: Optional[Callable[[int], None]] = None,
        disk_directory: Optional[Path] = None,
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
    ) -> "PackedMutationSnapshot":
        roots = tuple(roots)
        pagers = {getattr(module, "_native_core_pager", None)
                  for root in roots for module in root.modules()} - {None}
        for pager in pagers:
            pager.flush()
        owners: list[tuple[nn.Module, tuple[tuple[str, torch.Tensor], ...]]] = []
        seen: set[int] = set()
        byte_count = 131072 # conservative metadata, RNG/control and transfer headroom
        for root in roots:
            for module in root.modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                packed_tensors = getattr(module, "authoritative_packed_tensors", None)
                if not callable(packed_tensors):
                    continue
                tensors = list(packed_tensors())
                for name in ("_row_stability", "_bias_row_stability"):
                    buffer = getattr(module, name, None)
                    if isinstance(buffer, torch.Tensor):
                        tensors.append(buffer)
                if not tensors or any(
                    not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8
                    for tensor in tensors
                ):
                    raise ValueError("packed rollback owner has non-uint8 state")
                byte_count += sum(int(tensor.numel()) for tensor in tensors) + 1024
                identities = {id(tensor) for tensor in tensors}
                names = tuple((name, value) for name, value in module.named_buffers(recurse=False)
                              if id(value) in identities)
                if len(names) != len(identities):
                    raise ValueError("packed rollback must resolve registered buffer names")
                owners.append((module, names))
        if reserve is not None:
            try:
                reserve(byte_count)
            except Exception:
                if disk_directory is None:
                    raise
                from .native_core_paging import BoundedPackedRollback
                io_ram_bytes = min(byte_count, 8 * 1024 * 1024) + len(owners) * 2048 + 131072
                reserve(io_ram_bytes)
                disk_snapshot = BoundedPackedRollback.capture(
                    roots, directory=disk_directory, reserve_disk=disk_reserve,
                )
                return cls([], disk_snapshot.byte_count, disk_snapshot=disk_snapshot,
                    disk_ram_reserve=reserve, disk_io_ram_bytes=io_ram_bytes)
        entries = []
        with torch.no_grad():
            for module, tensors in owners:
                entries.append((
                    module,
                    [
                        (name, tensor.detach().to(device="cpu", copy=True))
                        for name, tensor in tensors
                    ],
                    int(getattr(module, "_pending_stability_events", 0)),
                ))
        return cls(entries, byte_count)

    def restore(self) -> None:
        """Restore exactly the pre-batch packed state; safe for repeated retries."""

        if self._disk_snapshot is not None:
            if self._disk_ram_reserve is not None:
                self._disk_ram_reserve(self._disk_io_ram_bytes)
            self._disk_snapshot.restore()
            return
        pagers = {getattr(module, "_native_core_pager", None) for module, _, _ in self._entries} - {None}
        for pager in pagers:
            pager.flush()
        with torch.no_grad():
            for module, tensors, pending_events in self._entries:
                for name, original in tensors:
                    target = module._buffers.get(name)
                    if not isinstance(target, torch.Tensor):
                        raise RuntimeError("packed rollback registered target disappeared")
                    if target.dtype != original.dtype or target.shape != original.shape:
                        raise RuntimeError("packed rollback target changed shape or dtype")
                    with packed_diagnostic_restore(module, target, original, captured_scopes=self._diagnostic_scopes):
                        target.copy_(original)
                if hasattr(module, "_pending_stability_events"):
                    module._pending_stability_events = pending_events
                validate = getattr(module, "_validate_packed", None)
                if callable(validate):
                    validate()

    def close(self) -> None:
        if self._disk_snapshot is not None:
            self._disk_snapshot.close()
            self._disk_snapshot = None
        self._entries.clear()
        self._disk_ram_reserve = None


def adamw_for_remaining_parameters(
    parameters: Iterable[nn.Parameter | dict[str, Any]],
    *,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer | PackedOnlyOptimizer:
    """Never allocate Adam moments for packed ternary synapses."""

    floating = list(parameters)
    has_parameters = any(
        bool(group.get("params")) if isinstance(group, dict) else True
        for group in floating
    )
    if not has_parameters:
        return PackedOnlyOptimizer()
    return torch.optim.AdamW(floating, lr=lr, weight_decay=weight_decay)
