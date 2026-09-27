"""Optimizer adapter for packed-authoritative online synapses.

Packed projections update their ternary codes during backward and have no
floating weight Parameter or dense Adam moments. The adapter still accepts
floating Parameters if a separate non-native objective supplies them; a
native packed-only brain uses the no-op optimizer facade.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import torch
from torch import nn


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

    def __init__(self, entries: list[tuple[nn.Module, list[tuple[torch.Tensor, torch.Tensor]], int]], byte_count: int):
        self._entries = entries
        self.byte_count = byte_count

    @classmethod
    def capture(
        cls,
        roots: Iterable[nn.Module],
        *,
        reserve: Optional[Callable[[int], None]] = None,
    ) -> "PackedMutationSnapshot":
        owners: list[tuple[nn.Module, tuple[torch.Tensor, ...]]] = []
        seen: set[int] = set()
        byte_count = 0
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
                byte_count += sum(int(tensor.numel()) for tensor in tensors) + 256
                owners.append((module, tuple(tensors)))
        if reserve is not None:
            reserve(byte_count)
        entries = []
        with torch.no_grad():
            for module, tensors in owners:
                entries.append((
                    module,
                    [
                        (tensor, tensor.detach().to(device="cpu", copy=True))
                        for tensor in tensors
                    ],
                    int(getattr(module, "_pending_stability_events", 0)),
                ))
        return cls(entries, byte_count)

    def restore(self) -> None:
        """Restore exactly the pre-batch packed state; safe for repeated retries."""

        with torch.no_grad():
            for module, tensors, pending_events in self._entries:
                for target, original in tensors:
                    if target.dtype != original.dtype or target.shape != original.shape:
                        raise RuntimeError("packed rollback target changed shape or dtype")
                    target.copy_(original)
                if hasattr(module, "_pending_stability_events"):
                    module._pending_stability_events = pending_events
                validate = getattr(module, "_validate_packed", None)
                if callable(validate):
                    validate()


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
