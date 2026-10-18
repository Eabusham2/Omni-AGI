"""Bounded transient VSA superposition; no learned floating weight storage.

Short sequences retain Torch's existing stack/mean reduction. Long sequences
use the same rows and positions in bounded blocks, with compensated float32
activity accumulation. Floating reduction order may differ on the long path;
parity is measured numerically, never described as bit-identical inference.
"""
from __future__ import annotations

from collections.abc import Iterable

import torch

_BLOCK_BYTES = 64 * 1024
_BLOCK_ROWS = 32


def activity_block_rows(dimensions: int) -> int:
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("activity dimensions must be a positive integer")
    return max(1, min(_BLOCK_ROWS, _BLOCK_BYTES // (dimensions * 4)))


def activity_reduction_scratch_bytes(dimensions: int, rows: int) -> int:
    """Conservative reduction tensor/list bytes, excluding row-store decoding.

    The caller separately admits token/label storage, packed-row decoding and
    repeated-contact metadata. No term scales as rows * dimensions once a
    sequence exceeds one block.
    """
    block = min(max(0, int(rows)), activity_block_rows(dimensions))
    return (2 * block + 10) * dimensions * 4 + block * 64 + 4096


def streamed_activity_mean(vectors: Iterable[torch.Tensor], dimensions: int) -> torch.Tensor:
    from .working_attention_paging import current_working_attention_pager

    block_rows = activity_block_rows(dimensions)
    pager = current_working_attention_pager()
    iterator = iter(vectors)
    pending: list[torch.Tensor] = []
    total = correction = None
    count = 0
    while True:
        if pager is not None:
            pager.check_cancelled()
        try:
            row = next(iterator)
        except StopIteration:
            break
        if row.ndim != 1 or row.numel() != dimensions or row.dtype != torch.float32:
            raise ValueError("VSA activity needs one float32 row of the declared width")
        if len(pending) == block_rows:
            summed = torch.stack(pending).sum(dim=0)
            pending.clear()
            if total is None:
                total = summed
                correction = torch.zeros_like(total)
            else:
                increment = summed - correction
                updated = total + increment
                correction = (updated - total) - increment
                total = updated
        pending.append(row)
        count += 1
    if not count:
        raise ValueError("cannot average an empty VSA activity sequence")
    if total is None:
        # Preserve the exact ordinary short-sequence operation, including its
        # dtype and reduction implementation.
        return torch.stack(pending).mean(dim=0)
    if pending:
        increment = torch.stack(pending).sum(dim=0) - correction
        total = total + increment
    return total / count
