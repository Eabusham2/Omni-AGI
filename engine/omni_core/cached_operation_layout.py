"""Actual paging/tile-transition range for fresh retained-cache inference.

This is a structural admission class, not measured evidence for unseen lengths
or values. Pure metadata only: no Torch/model/cache I/O on import or use.
"""
from __future__ import annotations

import hashlib
import json
import math


def _integer(value, *, positive=True):
    if type(value) is not int or not (1 if positive else 0) <= value <= (1 << 53) - 1:
        raise ValueError("cached operation geometry needs exact positive safe integers")
    return value


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def attention_range(length, configured_key_tokens, maximum_safe_key_tokens, capacity):
    """The real q=1 halving regime, then the real fixed-key-tile count band.

    `_tile_shape` halves the requested/min(length) key axis until it fits. The
    warm range below that request preserves its halving regime, NOT a claim of
    identical final partial widths/counts. Above it, actual fixed tile-count
    transitions bound the range. Per-frontier byte admission remains required.
    """
    length, configured, safe, capacity = map(_integer, (length, configured_key_tokens, maximum_safe_key_tokens, capacity))
    if length > capacity or safe > configured: raise ValueError("cached attention transition geometry is inconsistent")
    maximum = min(configured, safe)
    if length < configured:
        halves, current = 0, length
        while current > maximum: current //= 2; halves += 1
        lower = 1 if halves == 0 else (maximum + 1) * (1 << (halves - 1))
        upper = maximum if halves == 0 else (maximum + 1) * (1 << halves) - 1
        return max(1, lower), min(configured - 1, upper, capacity), {"mode": "min-key-halving-regime", "halvings": halves,
            "requestedKeyTokens": configured, "maximumSafeKeyTokens": safe}
    effective = configured
    halves = 0
    while effective > maximum: effective //= 2; halves += 1
    tiles = (length + effective - 1) // effective
    return max(configured, (tiles - 1) * effective + 1), min(tiles * effective, capacity), {
        "mode": "fixed-key-tile-count", "halvings": halves, "effectiveKeyTokens": effective, "tiles": tiles,
        "requestedKeyTokens": configured, "maximumSafeKeyTokens": safe}


def cached_operation_range(length, capacity, rows):
    """Intersect actual next-append and key-tile transition intervals.

    rows are actual retained K/V sequence descriptors plus one attention-tile
    description per owner. A full old tail starts one new-page operation; later
    appends share the actual coalescing range. Unknown/changed capacity, missing
    pages, invalid ends and opaque layouts fail closed rather than guessing a
    generic context bucket. No token/cache bytes enter the reusable range SHA.
    """
    length, capacity = _integer(length), _integer(capacity)
    if length < 2 or length > capacity or not isinstance(rows, list) or not rows:
        raise ValueError("complete real retained attention layout is required")
    lower, upper, identities = 2, capacity, []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"owner", "axis", "shape", "dtype", "tokenBytes", "pageTokens",
                "tileBudgetBytes", "previousPageTokens", "ends", "configuredKeyTokens", "maximumSafeKeyTokens"}:
            raise ValueError("cached page range row has an opaque or incomplete layout")
        if not isinstance(row["owner"], str) or not row["owner"] or not isinstance(row["dtype"], str):
            raise ValueError("cached page range owner/dtype is missing")
        shape, ends, axis = row["shape"], row["ends"], row["axis"]
        if (not isinstance(shape, list) or not shape or type(axis) is not int or not 0 <= axis < len(shape)
                or any(type(value) is not int or value < 1 for value in shape) or shape[axis] != length - 1
                or not isinstance(ends, list) or not ends or ends[-1] != length - 1
                or any(type(end) is not int or end < 1 or index and end <= ends[index - 1] for index, end in enumerate(ends))):
            raise ValueError("retained page ends disagree with actual cache length")
        token_bytes = _integer(row["tokenBytes"])
        page_tokens, budget = _integer(row["pageTokens"]), _integer(row["tileBudgetBytes"])
        actual_capacity = min(page_tokens, max(1, budget // (4 * token_bytes)))
        if row["previousPageTokens"] != actual_capacity:
            raise ValueError("KV allocation policy changed; require an ordinary valid frontier first")
        start = ends[-2] if len(ends) > 1 else 0
        tail = ends[-1] - start
        if tail < actual_capacity:
            page_lower, page_upper, page_mode = start + 2, start + actual_capacity, "coalesce-existing-tail"
        else:
            page_lower = page_upper = length
            page_mode = "allocate-new-page"
        tile_lower, tile_upper, tile_class = attention_range(length, row["configuredKeyTokens"], row["maximumSafeKeyTokens"], capacity)
        lower, upper = max(lower, page_lower, tile_lower), min(upper, page_upper, tile_upper)
        fixed_shape = list(shape); fixed_shape[axis] = None
        identities.append({"owner": row["owner"], "axis": axis, "shapeWithoutSequence": fixed_shape, "dtype": row["dtype"],
            "tokenBytes": token_bytes, "pageCapacity": actual_capacity, "tileBudgetBytes": budget, "pageMode": page_mode,
            "prefixEnds": ends[:-1] if page_mode == "coalesce-existing-tail" else ends,
            "tileClass": tile_class})
    if not lower <= length <= upper: raise ValueError("actual paging/tile transition intersection is empty")
    layout = _sha({"minimumTokens": lower, "maximumTokens": upper, "queryTokens": 1, "owners": identities})
    return {"minimumTokens": lower, "maximumTokens": upper, "queryTokens": 1,
            "basis": "actual-pager-and-attention-tile-transitions", "layoutSha256": layout}


def maximum_safe_key_tokens(configured, *, budget, cost):
    """Read the existing exact attention byte formula, never time/extrapolate it."""
    configured, budget = _integer(configured), _integer(budget)
    if cost(1) > budget: raise RuntimeError("one exact cached attention cell does not fit current admission")
    low, high = 1, configured
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle) <= budget: low = middle
        else: high = middle - 1
    return low


def sampling_inference_class(sampling):
    """Bind the ACTUAL current law/mask, not a changing unused token budget.

    Qualification still binds the full actual boundary/budget/RNG separately.
    UTF-8 budget values >=4 have the same mask at a complete boundary; exact
    allowed IDs, not a guessed clipping bucket, establish that transition.
    """
    from .utf8_text_boundary import Utf8TextBoundary
    if not isinstance(sampling, dict) or set(sampling) != {"temperature", "top_k", "noise", "text_boundaries", "remaining_tokens"}:
        raise ValueError("complete native sampling policy is required")
    if any(type(sampling[name]) not in {int, float} or not math.isfinite(sampling[name]) for name in ("temperature", "noise")):
        raise ValueError("native sampling controls must be finite")
    if type(sampling["top_k"]) is not int or sampling["top_k"] < 0 or type(sampling["remaining_tokens"]) is not int or sampling["remaining_tokens"] < 1:
        raise ValueError("native sampling count controls are invalid")
    boundaries = sampling["text_boundaries"]
    if boundaries is not None and (not isinstance(boundaries, list) or any(type(value) is not Utf8TextBoundary for value in boundaries)):
        raise ValueError("opaque sampling boundary cannot enter reusable Auto")
    masks = None if boundaries is None else [value.allowed_token_ids(sampling["remaining_tokens"]) for value in boundaries]
    return _sha({"temperature": sampling["temperature"], "topK": sampling["top_k"], "noise": sampling["noise"], "exactAllowedMasks": masks})
