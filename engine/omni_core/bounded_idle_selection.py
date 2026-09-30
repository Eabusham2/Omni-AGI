"""Exact workspace selection with no corpus-sized assembly list or sort.

The caller's hardware-derived workspace capacity remains the attention policy,
not a storage/learning/addressability limit. Every learned assembly is visited;
only the current active workspace is buffered. Ties preserve insertion order.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable, Iterable, Mapping
from typing import Any


def select_idle_workspace(
    assemblies: Iterable[Mapping[str, Any]], *, capacity: int,
    score: Callable[[Mapping[str, Any]], float],
    eligible: Callable[[Mapping[str, Any]], bool],
    reserve: Callable[[int], Any] | None = None,
) -> list[Mapping[str, Any]]:
    if type(capacity) is not int or capacity < 1:
        raise ValueError("idle workspace capacity must be positive")
    heap: list[tuple[float, int, Mapping[str, Any]]] = []
    for ordinal, record in enumerate(assemblies):
        if not eligible(record):
            continue
        entry = (float(score(record)), -ordinal, record)
        if len(heap) < capacity:
            if reserve is not None and reserve(4096) is False:
                raise RuntimeError("idle workspace selection paused at resource reserve")
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    return [record for _score, _ordinal, record in sorted(heap, key=lambda value: value[:2], reverse=True)]
