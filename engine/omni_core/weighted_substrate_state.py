"""Compact integer gain companions for native memory connections.

Graph eligibility, plasticity, uses and stability already have authoritative
owners. This module adds magnitude only; it is not a second floating weight.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch


def synapse_strength(record: Mapping) -> int:
    count = record.get("parallel_count", 0)
    if type(count) is not int or count < 0:
        raise ValueError("memory parallel connection count is invalid")
    value = record.get("connection_strength", 1)
    if type(value) is not int or not 1 <= value <= 16:
        raise ValueError("memory connection strength must be an integer from 1 to 16")
    return value


def synapse_contribution(record: Mapping) -> int:
    sign = record.get("effective_weight", 0)
    if isinstance(sign, bool) or not isinstance(sign, (int, float)) or sign not in (-1, 0, 1):
        raise ValueError("memory connection sign must be exact ternary")
    return int(sign) * synapse_strength(record)


def synapse_inspection_fields(record: Mapping) -> dict:
    """Inspector facts derived from the actual contact, not synthetic links."""
    strength = synapse_strength(record)
    base = "%s>%s:%s" % (record.get("source_id", ""), record.get("target_id", ""), record.get("kind", "associates"))
    identifier = str(record.get("id", ""))
    prefix = base + "#contact="
    suffix = identifier[len(prefix):] if identifier.startswith(prefix) else ""
    ordinal = int(suffix) if suffix.isdecimal() and str(int(suffix)) == suffix and int(suffix) > 0 else 0
    return {"connectionStrength": strength, "effectiveContribution": synapse_contribution(record),
            "isRepeatedContact": ordinal > 0, "parallelRootId": base if ordinal else identifier,
            "parallelOrdinal": ordinal, "additionalContactCount": record.get("parallel_count", 0)}


def validate_contribution(value) -> int:
    if type(value) is not int or not -16 <= value <= 16:
        raise ValueError("memory connection contribution must be a signed integer of magnitude at most 16")
    return value


def pack_strengths(values) -> bytes:
    packed = bytearray()
    for index, value in enumerate(values):
        if type(value) is not int or not 1 <= value <= 16:
            raise ValueError("memory connection strength must be an integer from 1 to 16")
        if index % 2 == 0:
            packed.append(value - 1)
        else:
            packed[-1] |= (value - 1) << 4
    return bytes(packed)


def strength_at(packed: bytes, index: int) -> int:
    return ((packed[index // 2] >> (4 * (index % 2))) & 15) + 1


def validate_strengths(packed: bytes, count: int) -> bytes:
    if not isinstance(packed, bytes) or len(packed) != (count + 1) // 2:
        raise ValueError("memory strength companion length is invalid")
    if count % 2 and packed[-1] >> 4:
        raise ValueError("memory strength companion padding is invalid")
    return packed


def unpack_strengths(value: torch.Tensor | None, count: int) -> torch.Tensor:
    if value is None:
        return torch.ones(count, dtype=torch.uint8)
    if value.dtype != torch.uint8 or tuple(value.shape) != ((count + 1) // 2,):
        raise ValueError("packed memory strength tensor is invalid")
    raw = validate_strengths(bytes(value.tolist()), count)
    return torch.tensor([strength_at(raw, index) for index in range(count)], dtype=torch.uint8)


class WeightedForwardSigns(bytes):
    """A bounded derived sign page with its compact gain companion."""

    def __new__(cls, signs: bytes, strengths: bytes, count: int, positions=None):
        result = super().__new__(cls, signs)
        result.strengths = validate_strengths(strengths, count)
        result.positions = positions
        return result


def forward_contribution(signs: bytes, index: int) -> int:
    code = (signs[index // 4] >> (2 * (index % 4))) & 3
    if code == 3:
        raise ValueError("memory forward sign contains a reserved code")
    strengths = getattr(signs, "strengths", None)
    positions = getattr(signs, "positions", None)
    return (code - 1) * (1 if strengths is None else strength_at(strengths, index if positions is None else positions[index]))
