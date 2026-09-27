"""Exact, bounded two-pass similarity over packed ternary vector pages.

The page provider owns an immutable vector generation and deterministic row
order. This helper never keeps the corpus or all positive matches in memory:
pass one finds the global positive best/count, and pass two streams every row
at or above the same adaptive activation floor used by substrate recall. It
does not use an ANN candidate cutoff or claim a top-k approximation is exact.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional, Protocol, Sequence, Tuple

import torch


MAX_PAGE_ROWS = 4096  # Physical I/O window, never a neural cardinality limit.
_MIN_NORM = 1e-8


class PagedVectorScoringCancelled(RuntimeError):
    """The caller cancelled a two-pass packed-vector scan."""


class PagedVectorScoringPaused(RuntimeError):
    """A reserve callback refused the next bounded page."""


@dataclass(frozen=True)
class PackedVectorRow:
    sequence: int
    assembly_id: str
    packed: bytes


@dataclass(frozen=True)
class PackedVectorPage:
    rows: Tuple[PackedVectorRow, ...]
    next_cursor: Optional[str]
    has_more: bool
    snapshot_id: str


class PackedVectorPageProvider(Protocol):
    """Immutable-generation page interface for a future disk-paged store.

    ``page_rows`` must honor ``snapshot_id`` across both passes, return each
    live assembly exactly once in strictly increasing sequence order, and
    bound returned rows by ``page_size``. A mutable-only provider cannot make
    two-pass results exact and must instead expose a frozen generation.
    """

    dimensions: int

    def current_snapshot(self) -> str: ...

    def page_rows(
        self, snapshot_id: str, cursor: Optional[str], page_size: int
    ) -> PackedVectorPage: ...


@dataclass(frozen=True)
class SimilaritySummary:
    snapshot_id: str
    dimensions: int
    rows_seen: int
    positive_count: int
    best_score: float
    workspace_slots: int
    adaptive_floor: float
    content_sha256: str


@dataclass(frozen=True)
class SimilarityMatch:
    sequence: int
    assembly_id: str
    score: float
    packed: bytes


def _cue_values(cue: torch.Tensor | Sequence[float], dimensions: int) -> Tuple[float, ...]:
    if isinstance(cue, torch.Tensor):
        if cue.is_complex() or cue.dtype == torch.bool or cue.numel() != dimensions:
            raise ValueError("similarity cue must be a real vector of provider dimensions")
        values = tuple(float(item) for item in cue.detach().cpu().reshape(-1).tolist())
    else:
        if isinstance(cue, (str, bytes)) or not isinstance(cue, Sequence):
            raise ValueError("similarity cue must be a numeric sequence")
        if len(cue) != dimensions or any(isinstance(item, bool) for item in cue):
            raise ValueError("similarity cue must match provider dimensions")
        try:
            values = tuple(float(item) for item in cue)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("similarity cue must be numeric") from error
    if not all(math.isfinite(item) for item in values):
        raise ValueError("similarity cue must contain only finite values")
    return values


def _packed_cosine(
    cue: Tuple[float, ...], cue_norm: float, packed: bytes, dimensions: int
) -> float:
    """Cosine against exact {-1,0,+1} codes without decoding a float row."""

    if not isinstance(packed, bytes) or len(packed) != (dimensions + 3) // 4:
        raise ValueError("packed similarity row width is invalid")
    terms = []
    nonzero = 0
    for index in range(dimensions):
        code = (packed[index // 4] >> ((index % 4) * 2)) & 3
        if code == 3:
            raise ValueError("packed similarity row has a reserved ternary code")
        if code != 1:
            nonzero += 1
            terms.append(cue[index] if code == 2 else -cue[index])
    for index in range(dimensions, len(packed) * 4):
        if ((packed[index // 4] >> ((index % 4) * 2)) & 3) != 1:
            raise ValueError("packed similarity row has non-canonical padding")
    if nonzero == 0:
        return 0.0
    score = math.fsum(terms) / (max(cue_norm, _MIN_NORM) * math.sqrt(nonzero))
    return max(-1.0, min(1.0, score))


def _update_digest(digest: Any, row: PackedVectorRow) -> None:
    identifier = row.assembly_id.encode("utf-8")
    digest.update(struct.pack(">Q", row.sequence))
    digest.update(struct.pack(">I", len(identifier)))
    digest.update(identifier)
    digest.update(row.packed)


def _check_cancelled(cancelled: Optional[Callable[[], bool]]) -> None:
    if cancelled is not None and cancelled():
        raise PagedVectorScoringCancelled("packed-vector similarity scan cancelled")


def _reserve_page(
    reserve_memory: Optional[Callable[[int, str], Any]],
    page_size: int,
    row_bytes: int,
    stage: str,
) -> None:
    # Reserve before asking the provider to materialize a page. The provider
    # should additionally preflight its own exact disk/decode overhead.
    estimate = 64 * 1024 + page_size * (4 * row_bytes + 512)
    if reserve_memory is not None and reserve_memory(estimate, stage) is False:
        raise PagedVectorScoringPaused("%s paused at memory reserve" % stage)


def _iter_scored_rows(
    provider: PackedVectorPageProvider,
    *,
    snapshot_id: str,
    cue: Tuple[float, ...],
    cue_norm: float,
    page_size: int,
    cancelled: Optional[Callable[[], bool]],
    reserve_memory: Optional[Callable[[int, str], Any]],
    digest: Any,
    stage: str,
) -> Iterator[Tuple[PackedVectorRow, float]]:
    cursor: Optional[str] = None
    last_sequence = 0
    dimensions = len(cue)
    while True:
        _check_cancelled(cancelled)
        _reserve_page(reserve_memory, page_size, (dimensions + 3) // 4, stage)
        page = provider.page_rows(snapshot_id, cursor, page_size)
        if (
            not isinstance(page, PackedVectorPage)
            or page.snapshot_id != snapshot_id
            or not isinstance(page.rows, tuple)
            or not isinstance(page.has_more, bool)
            or len(page.rows) > page_size
            or (page.has_more and not page.rows)
            or (page.has_more and (
                not isinstance(page.next_cursor, str)
                or not page.next_cursor
                or page.next_cursor == cursor
            ))
        ):
            raise ValueError("packed-vector provider returned an invalid page")
        for row in page.rows:
            _check_cancelled(cancelled)
            if (
                not isinstance(row, PackedVectorRow)
                or isinstance(row.sequence, bool)
                or not isinstance(row.sequence, int)
                or row.sequence <= last_sequence
                or not isinstance(row.assembly_id, str)
                or not row.assembly_id
            ):
                raise ValueError("packed-vector provider row order or identity is invalid")
            score = _packed_cosine(cue, cue_norm, row.packed, dimensions)
            last_sequence = row.sequence
            _update_digest(digest, row)
            yield row, score
        if not page.has_more:
            return
        cursor = page.next_cursor


class PreparedPagedSimilarity:
    """A pass-one result whose matches are streamed by a verified second pass."""

    def __init__(
        self,
        provider: PackedVectorPageProvider,
        cue: Tuple[float, ...],
        cue_norm: float,
        summary: SimilaritySummary,
        page_size: int,
        cancelled: Optional[Callable[[], bool]],
        reserve_memory: Optional[Callable[[int, str], Any]],
    ) -> None:
        self.provider = provider
        self._cue = cue
        self._cue_norm = cue_norm
        self.summary = summary
        self.page_size = page_size
        self.cancelled = cancelled
        self.reserve_memory = reserve_memory

    def iter_matches(self) -> Iterator[SimilarityMatch]:
        """Yield every eligible assembly in provider sequence order.

        Fully exhaust the iterator before publishing downstream effects: its
        final corpus digest check detects a provider that failed to keep the
        supposedly immutable vector generation stable between passes.
        """

        digest = hashlib.sha256()
        rows_seen = 0
        positive_count = 0
        best_score = 0.0
        for row, score in _iter_scored_rows(
            self.provider,
            snapshot_id=self.summary.snapshot_id,
            cue=self._cue,
            cue_norm=self._cue_norm,
            page_size=self.page_size,
            cancelled=self.cancelled,
            reserve_memory=self.reserve_memory,
            digest=digest,
            stage="packed-vector similarity pass 2",
        ):
            rows_seen += 1
            if score > 0.0:
                positive_count += 1
                best_score = max(best_score, score)
                if score >= self.summary.adaptive_floor:
                    yield SimilarityMatch(
                        sequence=row.sequence,
                        assembly_id=row.assembly_id,
                        score=score,
                        packed=row.packed,
                    )
        if (
            rows_seen != self.summary.rows_seen
            or positive_count != self.summary.positive_count
            or best_score != self.summary.best_score
            or digest.hexdigest() != self.summary.content_sha256
        ):
            raise ValueError("packed-vector snapshot changed between similarity passes")


def prepare_exact_paged_similarity(
    provider: PackedVectorPageProvider,
    cue: torch.Tensor | Sequence[float],
    *,
    page_size: int = 128,
    workspace_slots: Optional[int] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    reserve_memory: Optional[Callable[[int, str], Any]] = None,
) -> PreparedPagedSimilarity:
    """Measure global best/count, then prepare an exact streamed floor scan.

    A caller must use ``iter_matches`` to perform pass two. No matching row is
    silently truncated, including when the number of matches exceeds working
    slots. A large active set may still need its own paged downstream state.
    """

    dimensions = getattr(provider, "dimensions", None)
    if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions < 1:
        raise ValueError("packed-vector provider dimensions are invalid")
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or not 1 <= page_size <= MAX_PAGE_ROWS
    ):
        raise ValueError("packed-vector page size is invalid")
    if (
        workspace_slots is not None
        and (isinstance(workspace_slots, bool) or not isinstance(workspace_slots, int))
    ):
        raise ValueError("working-memory slots must be an integer")
    values = _cue_values(cue, dimensions)
    cue_norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(cue_norm):
        raise ValueError("similarity cue norm is not finite")
    snapshot_id = provider.current_snapshot()
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("packed-vector provider snapshot identity is invalid")
    digest = hashlib.sha256()
    rows_seen = 0
    positive_count = 0
    best_score = 0.0
    for _row, score in _iter_scored_rows(
        provider,
        snapshot_id=snapshot_id,
        cue=values,
        cue_norm=cue_norm,
        page_size=page_size,
        cancelled=cancelled,
        reserve_memory=reserve_memory,
        digest=digest,
        stage="packed-vector similarity pass 1",
    ):
        rows_seen += 1
        if score > 0.0:
            positive_count += 1
            best_score = max(best_score, score)
    slots = max(
        1,
        int(workspace_slots or max(8, math.sqrt(positive_count))),
    )
    adaptive_floor = max(0.01, best_score / (2.0 + math.log2(slots + 1.0)))
    summary = SimilaritySummary(
        snapshot_id=snapshot_id,
        dimensions=dimensions,
        rows_seen=rows_seen,
        positive_count=positive_count,
        best_score=best_score,
        workspace_slots=slots,
        adaptive_floor=adaptive_floor,
        content_sha256=digest.hexdigest(),
    )
    return PreparedPagedSimilarity(
        provider,
        values,
        cue_norm,
        summary,
        page_size,
        cancelled,
        reserve_memory,
    )
