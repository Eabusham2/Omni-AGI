"""Discrete strength/resistance and repeated coordinates for a native VSA row.

Only packed bytes are durable. Decoding creates one short-lived activity row;
there is no floating-point learned master or text/fact answering store.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct

import torch

MAGIC = b"\xffOV2"  # Invalid as an old canonical ternary prefix.
HEADER = struct.Struct("<4sQ")
CONTACT = struct.Struct("<QBB")  # coordinate, sign-with-padding, strength/resistance


def validate_contacts(records: bytes, dimensions: int) -> bytes:
    if not isinstance(records, bytes) or len(records) % CONTACT.size:
        raise ValueError("packed vector repeated-contact length is invalid")
    for coordinate, sign, _state in CONTACT.iter_unpack(records):
        if coordinate >= dimensions or sign >> 2 != 0x15 or sign & 3 == 3:
            raise ValueError("packed vector repeated contact is invalid")
    return records


def payload_parts(payload: bytes, dimensions: int):
    from .packed_vsa_vectors import _validate_packed_row
    width = (dimensions + 3) // 4
    if len(payload) == width:
        _validate_packed_row(payload, dimensions)
        return payload, bytes(dimensions), b""
    if len(payload) < HEADER.size + width + dimensions:
        raise ValueError("weighted vector payload width is incomplete")
    magic, count = HEADER.unpack_from(payload)
    if magic != MAGIC or len(payload) != HEADER.size + width + dimensions + count * CONTACT.size:
        raise ValueError("weighted vector payload shape is invalid")
    signs = payload[HEADER.size:HEADER.size + width]
    state = payload[HEADER.size + width:HEADER.size + width + dimensions]
    records = payload[HEADER.size + width + dimensions:]
    _validate_packed_row(signs, dimensions)
    validate_contacts(records, dimensions)
    return signs, state, records


def make_payload(signs: bytes, state: bytes, records: bytes, dimensions: int) -> bytes:
    from .packed_vsa_vectors import _validate_packed_row
    _validate_packed_row(signs, dimensions)
    if not isinstance(state, bytes) or len(state) != dimensions:
        raise ValueError("weighted vector connection-state width is invalid")
    validate_contacts(records, dimensions)
    return HEADER.pack(MAGIC, len(records) // CONTACT.size) + signs + state + records


def iter_weighted_levels(payload: bytes, dimensions: int):
    """Read discrete contributions directly; no expanded learned row needed."""
    signs, state, records = payload_parts(payload, dimensions)
    repeats = {}
    for coordinate, sign, value in CONTACT.iter_unpack(records):
        repeats[coordinate] = repeats.get(coordinate, 0) + ((sign & 3) - 1) * ((value & 15) + 1)
    for coordinate in range(dimensions):
        sign = ((signs[coordinate // 4] >> (2 * (coordinate % 4))) & 3) - 1
        yield sign * ((state[coordinate] & 15) + 1) + repeats.get(coordinate, 0)


def weighted_levels(payload: bytes, dimensions: int) -> torch.Tensor:
    return torch.tensor(list(iter_weighted_levels(payload, dimensions)), dtype=torch.int64)


def activity(payload: bytes, dimensions: int) -> torch.Tensor:
    values = weighted_levels(payload, dimensions).to(torch.float32)
    norm = values.norm()
    return values / norm if float(norm) else values


def adapt_payload(payload: bytes, target, rate: float, dimensions: int, *, seed: int,
                  identifier: str, updates: int, zero_deadband: float):
    """Generalize the existing seeded, stochastic blend to signed integer gains.

    Equal-magnitude ternary targets retain their original unit target levels.
    Relative continuous target amplitudes can train unequal gains. Per-contact
    resistance attenuates changes; overflow creates explicit discrete contacts.
    """
    from .packed_vsa_vectors import _decode_exact_row, _pack_exact_row, quantize_vsa_vector, _splitmix64
    signs, state_bytes, record_bytes = payload_parts(payload, dimensions)
    desired_signs = quantize_vsa_vector(target, dimensions, zero_deadband=zero_deadband)
    raw = torch.as_tensor(target).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    masked = raw * desired_signs.ne(0).to(torch.float64)
    selected = masked[desired_signs.ne(0)].abs()
    if not selected.numel():
        goals = torch.zeros(dimensions, dtype=torch.float64)
    elif bool(selected.eq(selected[0]).all()):
        goals = desired_signs.to(torch.float64)
    else:
        goals = masked * (math.sqrt(selected.numel()) / float(masked.norm()))
    previous = weighted_levels(payload, dimensions)
    if rate == 0.0 or bool((previous.to(torch.float64) - goals).abs().le(1e-12).all()):
        return payload, False
    base = _decode_exact_row(signs, dimensions).tolist()
    states = bytearray(state_bytes)
    records = [list(record) for record in CONTACT.iter_unpack(record_bytes)]
    by_coordinate = {}
    for index, record in enumerate(records):
        by_coordinate.setdefault(record[0], []).append(index)
    digest = hashlib.sha256(json.dumps([seed, identifier, updates], ensure_ascii=False, separators=(",", ":")).encode()).digest()
    random = int.from_bytes(digest[:8], "little")
    for coordinate, (old, goal) in enumerate(zip(previous.tolist(), goals.tolist())):
        contacts = [-1, *by_coordinate.get(coordinate, ())]
        resistance = min((states[coordinate] >> 4) if index == -1 else records[index][2] >> 4 for index in contacts)
        expectation = old + float(rate) / (1.0 + resistance / 16.0) * (goal - old)
        random, draw = _splitmix64(random)
        lower = math.floor(expectation)
        delta = lower + int(draw < expectation - lower) - old
        if not delta:
            continue
        contacts.sort(key=lambda index: ((states[coordinate] >> 4) if index == -1 else records[index][2] >> 4, index))
        for index in contacts:
            code = states[coordinate] if index == -1 else records[index][2]
            sign = base[coordinate] if index == -1 else (records[index][1] & 3) - 1
            current = sign * ((code & 15) + 1)
            updated = max(-16, min(16, current + delta))
            used = updated - current
            if not used:
                continue
            new_sign = 1 if updated > 0 else -1 if updated < 0 else 0
            new_state = max(1, abs(updated)) - 1 | min(15, (code >> 4) + 1) << 4
            if index == -1:
                base[coordinate] = new_sign
                states[coordinate] = new_state
            else:
                records[index][1:] = [0x54 | (new_sign + 1), new_state]
            delta -= used
            if delta == 0:
                break
        while delta:
            amount = min(16, abs(delta))
            sign = 1 if delta > 0 else -1
            records.append([coordinate, 0x54 | (sign + 1), amount - 1])
            delta -= sign * amount
    packed = _pack_exact_row(torch.tensor(base, dtype=torch.int8), dimensions)
    repeats = b"".join(CONTACT.pack(*record) for record in records)
    return make_payload(packed, bytes(states), repeats, dimensions), True
