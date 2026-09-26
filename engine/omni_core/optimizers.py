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
