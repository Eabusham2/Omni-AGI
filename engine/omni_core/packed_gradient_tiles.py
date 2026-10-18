"""Two-axis transient derivatives over authoritative packed connections.

These are learning tiles, never resident floating weights. Column boundaries
are byte-aligned; every write is journaled against the real owner and offset.
The caller advances shared row resistance once after all its column tiles.
"""
from __future__ import annotations

import torch

from .native_core_paging import release_native_tensor_chunk
from .packed_connections import CONNECTION_STRENGTH_MAX, connection_resistance, encode_connection_state
from .parameter_diagnostics import packed_diagnostic_write
from .training_randomness import learning_rand


def release_gradient_tile(packed, connections, row_start, rows, column_start, columns):
    for row in range(row_start, row_start + rows):
        first, last = column_start // 4, (column_start + columns + 3) // 4
        release_native_tensor_chunk(packed, row * packed.shape[1] + first, last - first)
        if connections is not None:
            release_native_tensor_chunk(connections, row * connections.shape[1] + column_start, columns)


@torch.no_grad()
def finish_row_resistance(owner, stability, row_start, grew, strength):
    from .model import PACKED_STABILITY_MAX
    if stability is None or strength <= 0 or not bool(grew.any()):
        return
    rows = stability[row_start:row_start + grew.numel()]
    replacement = torch.where(grew.to(rows.device), (rows.to(torch.int16) + 1).clamp_max(PACKED_STABILITY_MAX).to(torch.uint8), rows)
    with packed_diagnostic_write(owner, stability, replacement, start=row_start):
        rows.copy_(replacement)
    release_native_tensor_chunk(stability, row_start, rows.numel())


@torch.no_grad()
def apply_packed_gradient_tile(owner, packed, width, row_start, column_start, gradient,
                               rate, scale, *, generator=None, row_stability=None,
                               stability_strength=0., connection_state=None):
    """Same discrete step/probabilities as the row updater, for a code slice.

    Return (changed contacts, rows with base movement). Repeated contacts use
    their own resistance, not the shared base-row resistance, as before.
    Streaming changes RNG draw grouping, not the probability law. No sample,
    tile, code-count, or rank normalization is introduced here.
    """
    from .model import pack_ternary_weight, unpack_ternary_weight_rows

    if (gradient.ndim != 2 or column_start % 4 or column_start < 0 or row_start < 0
            or row_start + gradient.shape[0] > packed.shape[0]
            or column_start + gradient.shape[1] > width
            or (column_start + gradient.shape[1] != width and gradient.shape[1] % 4)
            or gradient.device != packed.device or not bool(torch.isfinite(gradient).all())):
        raise ValueError("packed gradient tile geometry or values are invalid")
    rows, columns = gradient.shape
    if connection_state is None:
        connection_state = owner.connection_state_for(packed)
    state = connection_state[row_start:row_start + rows, column_start:column_start + columns]
    if state.dtype != torch.uint8 or state.shape != gradient.shape or state.device != packed.device:
        raise ValueError("packed gradient tile connection state is invalid")
    first, last = column_start // 4, (column_start + columns + 3) // 4
    signs = packed[row_start:row_start + rows, first:last]
    try:
        levels = unpack_ternary_weight_rows(signs, columns, validate_reserved=False)
        signed = levels.to(torch.int16) * ((state & 15).to(torch.int16) + 1)
        strength = gradient.abs() * (float(rate) / float(scale))
        strength = strength / (1. + connection_resistance(state).float() * max(float(stability_strength), 1. / 16.))
        if row_stability is not None and stability_strength > 0:
            strength = strength / (1. + float(stability_strength) * row_stability[row_start:row_start + rows].float()[:, None])
        probability = -torch.expm1(-strength)
        draws = learning_rand(probability.shape, dtype=probability.dtype, device=probability.device, generator=generator)
        changed = 0
        bias = packed is getattr(owner, "_packed_forward_bias", None)
        # The native iterator captures original page-slot boundaries. New
        # saturated contacts are not taught twice in this logical update.
        for key, row, column, packed_sign, connection in owner.iter_parallel_connection_records(
                bias=bias, start=row_start, end=row_start + rows):
            if not column_start <= column < column_start + columns:
                continue
            derivative = gradient[row - row_start, column - column_start]
            resistance = connection >> 4
            chance = -torch.expm1(-derivative.abs() * (float(rate) / float(scale)) /
                (1. + resistance * max(float(stability_strength), 1. / 16.)))
            if not bool(learning_rand((), device=gradient.device, generator=generator) < chance):
                continue
            direction = 1 if float(derivative) < 0 else -1 if float(derivative) > 0 else 0
            if not direction:
                continue
            updated = ((packed_sign & 3) - 1) * ((connection & 15) + 1) + direction
            if abs(updated) > CONNECTION_STRENGTH_MAX:
                owner.add_parallel_connection_(row, column, sign=direction, bias=bias)
            else:
                sign = 0x54 | ((1 if updated > 0 else -1 if updated < 0 else 0) + 1)
                owner.parallel_connections.update(key, sign, max(1, abs(updated)) - 1 | (min(15, resistance + 1) << 4))
            changed += 1
        saturated = (draws < probability) & (((gradient < 0) & (signed == CONNECTION_STRENGTH_MAX)) |
                                            ((gradient > 0) & (signed == -CONNECTION_STRENGTH_MAX)))
        for row, column in torch.nonzero(saturated, as_tuple=False).tolist():
            owner.add_parallel_connection_(row_start + row, column_start + column,
                sign=1 if float(gradient[row, column]) < 0 else -1, bias=bias)
            changed += 1
        movement = torch.where((draws < probability) & (gradient < 0) & (signed < CONNECTION_STRENGTH_MAX), 1,
            torch.where((draws < probability) & (gradient > 0) & (signed > -CONNECTION_STRENGTH_MAX), -1, 0)).to(torch.int8)
        grew = movement.ne(0).any(dim=1)
        base_changes = int(movement.ne(0).sum())
        if base_changes:
            updated = signed + movement
            replacement = pack_ternary_weight(updated.sign().to(torch.int8))
            resistance = torch.where(movement.ne(0), (connection_resistance(state).to(torch.int16) + 1).clamp_max(15),
                                     connection_resistance(state).to(torch.int16))
            replacement_state = torch.where(movement.ne(0),
                encode_connection_state(updated.abs().clamp_min(1), resistance), state)
            for row in range(rows):
                with packed_diagnostic_write(owner, packed, replacement[row], start=(row_start + row) * packed.shape[1] + first):
                    packed[row_start + row, first:last].copy_(replacement[row])
                with packed_diagnostic_write(owner, connection_state, replacement_state[row], start=(row_start + row) * width + column_start):
                    state[row].copy_(replacement_state[row])
        return changed + base_changes, grew
    finally:
        release_gradient_tile(packed, connection_state, row_start, rows, column_start, columns)
