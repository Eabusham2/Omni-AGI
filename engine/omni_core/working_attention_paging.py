"""Exact tiled attention and a bounded RAM-first, spill-backed activity pool.

These are ephemeral neural activations, never model weights or a saved-answer
lookup. Files are private raw tensor pages with checksums; no pickle, mmap of a
whole context, or unbounded accelerator KV tensor is used. Context capacity and
the size of a compute tile are deliberately independent.
"""

from __future__ import annotations

import bisect
import contextvars
import hashlib
import math
import os
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.autograd.function import once_differentiable

from .offload import NeuralStateResourcePause


_ACTIVE_PAGER: contextvars.ContextVar[Optional["WorkingAttentionPager"]] = (
    contextvars.ContextVar("omni_working_attention_pager", default=None)
)


class WorkingAttentionCancelled(RuntimeError):
    """A cooperative cancellation at a query/key/page boundary."""


def current_working_attention_pager() -> Optional["WorkingAttentionPager"]:
    return _ACTIVE_PAGER.get()


def kv_context_bytes(
    tokens: int, batch: int, layers: int, width: int, element_bytes: int = 4
) -> int:
    """Exact raw K+V bytes, independent of query/key tile sizes."""
    return 2 * int(tokens) * int(batch) * int(layers) * int(width) * int(element_bytes)


def attention_tile_bytes(
    batch: int,
    heads: int,
    head_dim: int,
    query_tokens: int,
    key_tokens: int,
    element_bytes: int = 4,
    *,
    training: bool = False,
) -> int:
    """Conservative simultaneous tile working set, not a quadratic context.

    Float32/64 online-softmax accumulators, scores, probabilities, causal mask,
    and Q/K/V tiles are included. Backward needs extra probability/gradient
    tiles. Whole returned training gradients/outputs are admitted separately.
    """
    q, k, d = int(query_tokens), int(key_tokens), int(head_dim)
    scalar = max(4, int(element_bytes))
    vectors = (8 * q + 4 * k) * d * scalar
    scores = q * k * (7 * scalar + 1)
    statistics = 8 * q * scalar
    if training:
        vectors += (4 * q + 4 * k) * d * scalar
        scores += 3 * q * k * scalar
    return int(batch) * int(heads) * (vectors + scores + statistics)


@dataclass
class _ActivityPage:
    identifier: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    size: int
    tensor: Optional[torch.Tensor]
    path: Optional[Path] = None
    digest: Optional[str] = None
    verified_stat: Optional[Tuple[int, int, int, int]] = None
    allocated_size: int = 0


class WorkingAttentionPager:
    """One shared hot pool and designated spill allowance across all layers.

    ``device_tile_budget_bytes`` is a compute reservation, separate from the
    resident activity pool. A caller's existing resource policy is consulted
    before admissions; configured capacity is never reduced under pressure.
    The optional policy is intentionally duck typed for small, pure fixtures.
    """

    def __init__(
        self,
        directory: Path,
        *,
        resident_budget_bytes: int,
        scratch_budget_bytes: int,
        device_tile_budget_bytes: int,
        resource_policy: Optional[Any] = None,
        page_tokens: int = 256,
        compute_device: Optional[torch.device] = None,
    ):
        self.directory = Path(directory).resolve()
        self.resident_budget_bytes = max(0, int(resident_budget_bytes))
        self.scratch_budget_bytes = max(0, int(scratch_budget_bytes))
        self.device_tile_budget_bytes = max(1, int(device_tile_budget_bytes))
        self.page_tokens = max(1, int(page_tokens))
        self.resource_policy = resource_policy
        self.compute_device = torch.device("cpu") if compute_device is None else compute_device
        self._pages: Dict[str, _ActivityPage] = {}
        self._hot: OrderedDict[str, None] = OrderedDict()
        self._resident_bytes = 0
        self._spill_bytes = 0
        self._allocated_spill_bytes = 0
        probe = self.directory
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            self._allocation_unit = max(4096, int(os.statvfs(probe).f_frsize))
        except (AttributeError, OSError):
            # Without a measured filesystem unit, a conservative 64-KiB
            # cluster reservation cannot promote logical bytes to physical
            # capacity. Runtime disk-free policy remains the final gate.
            self._allocation_unit = 64 * 1024
        self._cancelled: Optional[Callable[[], bool]] = None
        self._policy_snapshot: Dict[str, Any] = {}
        self._policy_at = 0.0
        self._writes_since_policy = 0
        self._closed = False
        self._metrics: Dict[str, int] = {
            "peakResidentBytes": 0,
            "peakSpillBytes": 0,
            "bytesWritten": 0,
            "bytesRead": 0,
            "pageReads": 0,
            "pageWrites": 0,
            "largestPageBytes": 0,
            "largestReadBytes": 0,
            "peakHostTransferBytes": 0,
            "largestKeyTileTokens": 0,
            "largestQueryTileTokens": 0,
            "largestScoreElements": 0,
            "peakComputeTileBytes": 0,
            "largestPrefillBlockTokens": 0,
            "processedPrefillTokens": 0,
            "savedActivationBytes": 0,
            "cancellationChecks": 0,
        }

    def status(self) -> Dict[str, Any]:
        return {
            "mode": "exact-query-key-tiled-ram-first-activity-spill",
            "residentBudgetBytes": self.resident_budget_bytes,
            "scratchBudgetBytes": self.scratch_budget_bytes,
            "deviceTileBudgetBytes": self.device_tile_budget_bytes,
            "residentBudgetExcludesBoundedTransfers": True,
            "residentBytes": self._resident_bytes,
            "spillBytes": self._allocated_spill_bytes,
            "logicalSpillBytes": self._spill_bytes,
            "filesystemAllocationUnitBytes": self._allocation_unit,
            "livePages": len(self._pages),
            "pageMetadataEstimatedBytes": len(self._pages) * 2048,
            "pageMetadataEstimatePerTensorPageBytes": 2048,
            "spillDirectory": str(self.directory),
            "contextPagedToStorage": self._allocated_spill_bytes > 0,
            "exactCausalAttention": True,
            "trainingBackward": "exact-first-order-recomputed-tile-gradients",
            "trainingFullOutputs": "resource-admitted-bounded-windows",
            **self._metrics,
        }

    def _pause(self, message: str, **extra: Any) -> None:
        raise NeuralStateResourcePause(message, {**self.status(), **extra, "paused": True})

    def check_cancelled(self) -> None:
        self._metrics["cancellationChecks"] += 1
        if self._closed:
            raise RuntimeError("working attention pager is closed")
        if self._cancelled is not None and self._cancelled():
            raise WorkingAttentionCancelled("working attention cancelled at a tile boundary")

    @contextmanager
    def activate(
        self, cancelled: Optional[Callable[[], bool]] = None
    ) -> Iterator["WorkingAttentionPager"]:
        previous = self._cancelled
        if cancelled is not None:
            self._cancelled = cancelled
        token = _ACTIVE_PAGER.set(self)
        try:
            self.check_cancelled()
            yield self
        finally:
            _ACTIVE_PAGER.reset(token)
            self._cancelled = previous

    def _resource_status(self, estimated_ram_bytes: int = 0) -> Dict[str, Any]:
        if self.resource_policy is None:
            return {}
        # Disk reads/writes and thousands of tiny tiles cannot each spawn an
        # OS probe. Refresh the live policy at bounded intervals; track writes
        # since that measurement rather than reusing an unchanged free count.
        now = time.monotonic()
        if now - self._policy_at >= 0.25 or not self._policy_snapshot:
            self._policy_snapshot = dict(
                self.resource_policy.status(estimated_ram_bytes=int(estimated_ram_bytes))
            )
            self._policy_at = now
            self._writes_since_policy = 0
        status = dict(self._policy_snapshot)
        status["estimatedRamBytes"] = int(estimated_ram_bytes)
        available = status.get("availableMemoryBytes")
        reserve = int(status.get("ramReserveBytes", 0))
        resident = status.get("admissionResidentMemoryBytes", status.get("processMemoryBytes", 0))
        process = int(resident or 0)
        budget = status.get("systemRamBudgetBytes")
        status["memoryPressure"] = bool(
            status.get("ramAdmissionVerified") is False
            or
            (available is not None and int(available) - estimated_ram_bytes <= reserve)
            or (budget is not None and process + estimated_ram_bytes > int(budget))
        )
        return status

    def admit_compute(
        self, estimated_bytes: int, operation: str = "attention tile", *, device: Optional[torch.device] = None
    ) -> None:
        self.check_cancelled()
        needed = max(0, int(estimated_bytes))
        if needed > self.device_tile_budget_bytes:
            self._pause(
                operation + " exceeds its live compute reservation",
                estimatedComputeBytes=needed,
            )
        status = self._resource_status(needed)
        if status.get("memoryPressure"):
            # The hot pool is expendable. Spill before making a live resource
            # pause, but never reduce requested context or discard KV pages.
            self._evict_to(0)
            self._policy_at = 0.0
            status = self._resource_status(needed)
            if status.get("memoryPressure"):
                self._pause(operation + " paused at the live RAM reserve", resource=status)
        free = status.get("acceleratorFreeMemoryBytes")
        target = self.compute_device if device is None else device
        if target.type != "cpu" and free is not None and needed > int(free):
            self._pause(operation + " paused at live accelerator headroom", resource=status)
        self._metrics["peakComputeTileBytes"] = max(
            self._metrics["peakComputeTileBytes"], needed
        )

    def _admit_write(self, size: int) -> None:
        allocated = math.ceil(int(size) / self._allocation_unit) * self._allocation_unit
        if self._allocated_spill_bytes + allocated > self.scratch_budget_bytes:
            self._pause(
                "working activity spill pool is full; context was not truncated",
                requiredAdditionalSpillBytes=allocated,
            )
        status = self._resource_status()
        free = status.get("diskFreeBytes")
        reserve = int(status.get("diskReserveBytes", 0))
        if free is not None and int(free) - self._writes_since_policy - allocated <= reserve:
            self._pause("working activity spill paused at disk reserve", resource=status)

    def _spill(self, page: _ActivityPage) -> None:
        if page.path is None:
            self.check_cancelled()
            self._admit_write(page.size)
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (page.identifier + ".activity")
            assert page.tensor is not None
            # Each payload is one admitted page, never the complete context.
            payload = page.tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            try:
                with path.open("xb") as handle:
                    handle.write(payload)
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            stat = path.stat()
            blocks = getattr(stat, "st_blocks", None)
            allocated = (
                max(page.size, int(blocks) * 512)
                if blocks is not None
                else math.ceil(page.size / self._allocation_unit) * self._allocation_unit
            )
            if self._allocated_spill_bytes + allocated > self.scratch_budget_bytes:
                path.unlink(missing_ok=True)
                self._pause("physical activity-page allocation exceeds the spill pool; context was not truncated")
            page.path = path
            page.digest = hashlib.sha256(payload).hexdigest()
            page.allocated_size = allocated
            self._spill_bytes += page.size
            self._allocated_spill_bytes += allocated
            self._writes_since_policy += allocated
            self._metrics["bytesWritten"] += page.size
            self._metrics["pageWrites"] += 1
            self._metrics["peakSpillBytes"] = max(
                self._metrics["peakSpillBytes"], self._allocated_spill_bytes
            )
        if page.tensor is not None:
            self._resident_bytes -= page.size
            page.tensor = None
        self._hot.pop(page.identifier, None)

    def _evict_to(self, target: int) -> None:
        target = max(0, int(target))
        while self._resident_bytes > target and self._hot:
            identifier = next(iter(self._hot))
            self._spill(self._pages[identifier])

    def save_page(self, tensor: torch.Tensor) -> str:
        self.check_cancelled()
        size = tensor.numel() * tensor.element_size()
        # Reserve before copying an accelerator tensor to host. The temporary
        # copy is bounded by page size even when the hot pool is zero bytes.
        if size * 4 > self.device_tile_budget_bytes:
            self._pause("activity page exceeds bounded transfer reservation", pageBytes=size)
        self._evict_to(max(0, self.resident_budget_bytes - size))
        status = self._resource_status(size)
        if status.get("memoryPressure"):
            self._evict_to(0)
            self._policy_at = 0.0
            if self._resource_status(size).get("memoryPressure"):
                self._pause("activity page transfer paused at live RAM reserve")
        page = _ActivityPage(
            uuid.uuid4().hex,
            tuple(int(value) for value in tensor.shape),
            tensor.dtype,
            int(size),
            tensor.detach().to(device="cpu", copy=True).contiguous(),
        )
        self._pages[page.identifier] = page
        self._hot[page.identifier] = None
        self._resident_bytes += page.size
        self._metrics["largestPageBytes"] = max(self._metrics["largestPageBytes"], size)
        self._metrics["peakHostTransferBytes"] = max(self._metrics["peakHostTransferBytes"], size)
        try:
            self._evict_to(self.resident_budget_bytes)
        except BaseException:
            self.release_page(page.identifier)
            raise
        self._metrics["peakResidentBytes"] = max(
            self._metrics["peakResidentBytes"], self._resident_bytes
        )
        return page.identifier

    def _verified_file(self, page: _ActivityPage) -> Path:
        assert page.path is not None
        stat = page.path.stat()
        signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if stat.st_size != page.size:
            raise ValueError("working activity page checksum or size mismatch")
        if signature != page.verified_stat:
            digest = hashlib.sha256()
            # Validate cold pages without allocating a whole host page. Once
            # verified, exact inode/size/mtime/ctime changes trigger validation
            # again. These private ephemeral pages are never adopted on load.
            block_bytes = max(1, min(256 * 1024, self.device_tile_budget_bytes // 8))
            with page.path.open("rb") as handle:
                while True:
                    payload = handle.read(block_bytes)
                    if not payload:
                        break
                    digest.update(payload)
                    self._metrics["bytesRead"] += len(payload)
                    self._metrics["largestReadBytes"] = max(self._metrics["largestReadBytes"], len(payload))
                    self.check_cancelled()
            if digest.hexdigest() != page.digest:
                raise ValueError("working activity page checksum or size mismatch")
            page.verified_stat = signature
        return page.path

    def read_page(self, identifier: str) -> torch.Tensor:
        self.check_cancelled()
        page = self._pages[identifier]
        if page.tensor is not None:
            self._hot.move_to_end(identifier)
            return page.tensor
        payload = bytearray(self._verified_file(page).read_bytes())
        self._metrics["pageReads"] += 1
        self._metrics["bytesRead"] += page.size
        self._metrics["largestReadBytes"] = max(self._metrics["largestReadBytes"], page.size)
        # Cold scans do not evict recently appended hot KV just to cache every
        # old key on the way through. Returned page lifetime is one key tile.
        return torch.frombuffer(payload, dtype=page.dtype).reshape(page.shape)

    def read_page_range(self, identifier: str, start: int, end: int) -> torch.Tensor:
        """Read only a contiguous token range from a time-major activity page."""
        self.check_cancelled()
        page = self._pages[identifier]
        if start < 0 or end <= start or end > page.shape[0]:
            raise ValueError("activity subpage range is invalid")
        if page.tensor is not None:
            self._hot.move_to_end(identifier)
            return page.tensor[start:end]
        row_bytes = page.size // page.shape[0]
        count = (end - start) * row_bytes
        with self._verified_file(page).open("rb") as handle:
            handle.seek(start * row_bytes)
            payload = bytearray(handle.read(count))
        if len(payload) != count:
            raise ValueError("activity page became shorter during a bounded read")
        self._metrics["pageReads"] += 1
        self._metrics["bytesRead"] += count
        self._metrics["largestReadBytes"] = max(self._metrics["largestReadBytes"], count)
        return torch.frombuffer(payload, dtype=page.dtype).reshape(end - start, *page.shape[1:])

    def release_page(self, identifier: str) -> None:
        page = self._pages.pop(identifier, None)
        if page is None:
            return
        self._hot.pop(identifier, None)
        if page.tensor is not None:
            self._resident_bytes -= page.size
        if page.path is not None:
            page.path.unlink(missing_ok=True)
            self._spill_bytes -= page.size
            self._allocated_spill_bytes -= page.allocated_size
            # Resource free-space samples include existing files. Signed
            # allocation delta therefore credits a released/replaced page,
            # not every rewrite as if it accumulated another permanent file.
            self._writes_since_policy -= page.allocated_size

    def close(self) -> None:
        # Remove only exact page paths created by this object, never recurse
        # through a caller's designated storage directory.
        for identifier in tuple(self._pages):
            self.release_page(identifier)
        self._closed = True

    def sequence(self, tensor: torch.Tensor, axis: int = -2) -> "PagedTensorSequence":
        sequence = PagedTensorSequence(self, tuple(tensor.shape), tensor.dtype, axis)
        try:
            sequence.append(tensor)
        except BaseException:
            sequence.close()
            raise
        return sequence

    def save_activation(self, tensor: torch.Tensor) -> "SavedActivityTensor":
        saved = SavedActivityTensor(self, tensor)
        self._metrics["savedActivationBytes"] += tensor.numel() * tensor.element_size()
        return saved

    @contextmanager
    def saved_activation_hooks(self) -> Iterator[None]:
        def pack(tensor: torch.Tensor) -> Any:
            # Packed learned synapses remain the core pager's responsibility.
            # A detached small tensor avoids hook reference cycles.
            if not (tensor.is_floating_point() or tensor.is_complex()) or tensor.numel() < 64:
                return tensor.detach()
            return self.save_activation(tensor)

        def unpack(value: Any) -> torch.Tensor:
            return value.restore() if isinstance(value, SavedActivityTensor) else value

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def note_prefill(self, tokens: int) -> None:
        self._metrics["largestPrefillBlockTokens"] = max(
            self._metrics["largestPrefillBlockTokens"], int(tokens)
        )
        self._metrics["processedPrefillTokens"] += int(tokens)


class PagedTensorSequence:
    """Append-only exact tensor pages along one sequence axis."""

    def __init__(
        self, pager: WorkingAttentionPager, shape: Tuple[int, ...], dtype: torch.dtype, axis: int = -2
    ):
        self.pager = pager
        self.axis = int(axis) % len(shape)
        self._shape = list(int(value) for value in shape)
        self._shape[self.axis] = 0
        self.dtype = dtype
        self.pages: List[str] = []
        self.ends: List[int] = []
        self._closed = False
        self._page_tokens: Optional[int] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._shape)

    @property
    def length(self) -> int:
        return self._shape[self.axis]

    def append(self, tensor: torch.Tensor) -> None:
        if self._closed:
            raise RuntimeError("cannot append a released activity sequence")
        if tensor.dtype != self.dtype or tensor.ndim != len(self._shape) or any(
            int(size) != self._shape[index]
            for index, size in enumerate(tensor.shape) if index != self.axis
        ):
            raise ValueError("activity sequence shape or dtype changed")
        per_token = math.prod(
            int(size) for index, size in enumerate(tensor.shape) if index != self.axis
        ) * tensor.element_size()
        page_tokens = min(
            self.pager.page_tokens,
            max(1, self.pager.device_tile_budget_bytes // max(1, 4 * per_token)),
        )
        self._page_tokens = page_tokens
        start = 0
        if self.pages and tensor.shape[self.axis]:
            previous_start = 0 if len(self.ends) == 1 else self.ends[-2]
            previous_tokens = self.ends[-1] - previous_start
            if previous_tokens < page_tokens:
                count = min(page_tokens - previous_tokens, int(tensor.shape[self.axis]))
                size = (previous_tokens + count) * per_token
                self.pager.admit_compute(size * 4, "bounded KV/activity tail coalescing", device=torch.device("cpu"))
                old_identifier = self.pages[-1]
                old = self.pager.read_page(old_identifier)
                incoming = tensor.narrow(self.axis, 0, count).movedim(self.axis, 0).detach().to(device="cpu", copy=True).contiguous()
                merged = torch.cat((old, incoming), dim=0)
                del old, incoming
                # KV is invocation-local scratch. After a resource/cancel
                # failure the decoder closes the entire cache and retries its
                # source prefix; no saved experience/gradient source is lost.
                # Release before replacement so a full spill pool needs only
                # the allocation *delta*, not a duplicate trailing page.
                self.pager.release_page(old_identifier)
                try:
                    replacement = self.pager.save_page(merged)
                except BaseException:
                    self.close()
                    raise
                self.pages[-1] = replacement
                self._shape[self.axis] += count
                self.ends[-1] = self.length
                start = count
        for start in range(start, tensor.shape[self.axis], page_tokens):
            count = min(page_tokens, tensor.shape[self.axis] - start)
            # Time-major bytes let a cold key tile seek only its actual token
            # range rather than loading a complete multi-head backing page.
            identifier = self.pager.save_page(tensor.narrow(self.axis, start, count).movedim(self.axis, 0))
            self.pages.append(identifier)
            self._shape[self.axis] += int(count)
            self.ends.append(self.length)

    def read(self, start: int, end: int, device: torch.device) -> torch.Tensor:
        if start < 0 or end <= start or end > self.length:
            raise ValueError("activity page read range is invalid")
        pieces: List[torch.Tensor] = []
        index = bisect.bisect_right(self.ends, int(start))
        position = int(start)
        while position < end:
            page_start = 0 if index == 0 else self.ends[index - 1]
            stop = min(end, self.ends[index])
            tensor = self.pager.read_page_range(self.pages[index], position - page_start, stop - page_start)
            pieces.append(tensor.movedim(0, self.axis))
            position = stop
            index += 1
        values = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=self.axis)
        return values.to(device=device)

    def close(self) -> None:
        if self._closed:
            return
        for identifier in self.pages:
            self.pager.release_page(identifier)
        self.pages.clear()
        self.ends.clear()
        self._closed = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class SavedActivityTensor:
    """Lossless, page-bounded saved activation owned by an autograd graph."""

    def __init__(self, pager: WorkingAttentionPager, tensor: torch.Tensor):
        self.pager = pager
        self.shape = tuple(tensor.shape)
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.numel = tensor.numel()
        self.pages: List[str] = []
        # A noncontiguous reshape can allocate a complete accelerator copy.
        # Recursively slice in logical row-major order *before* flattening.
        max_elements = max(1, pager.device_tile_budget_bytes // (4 * tensor.element_size()))

        def bounded_chunks(value: torch.Tensor) -> Iterator[torch.Tensor]:
            if value.numel() <= max_elements:
                yield value.contiguous().reshape(-1)
                return
            per_row = math.prod(value.shape[1:])
            if per_row <= max_elements:
                rows = max(1, max_elements // max(1, per_row))
                for start in range(0, value.shape[0], rows):
                    yield value[start:start + rows].contiguous().reshape(-1)
            else:
                for index in range(value.shape[0]):
                    yield from bounded_chunks(value[index])

        try:
            for chunk in bounded_chunks(tensor):
                self.pages.append(pager.save_page(chunk))
        except BaseException:
            self.close()
            raise

    def restore(self, device: Optional[torch.device] = None) -> torch.Tensor:
        target = self.device if device is None else device
        size = self.numel * torch.empty((), dtype=self.dtype).element_size()
        self.pager.admit_compute(size, "bounded saved-activation restore", device=target)
        result = torch.empty(self.numel, dtype=self.dtype, device=target)
        offset = 0
        for identifier in self.pages:
            page = self.pager.read_page(identifier).reshape(-1)
            result[offset:offset + page.numel()].copy_(page.to(device=target))
            offset += page.numel()
        return result.reshape(self.shape)

    def close(self) -> None:
        for identifier in self.pages:
            self.pager.release_page(identifier)
        self.pages.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def freeze_runtime_tensor(tensor: torch.Tensor) -> Union[torch.Tensor, SavedActivityTensor]:
    pager = current_working_attention_pager()
    return tensor if pager is None else pager.save_activation(tensor)


def materialize_runtime_tensor(value: Any, device: torch.device) -> torch.Tensor:
    return value.restore(device) if isinstance(value, SavedActivityTensor) else value.to(device=device)


TensorSource = Union[torch.Tensor, PagedTensorSequence]


def _read(source: TensorSource, start: int, end: int, device: torch.device) -> torch.Tensor:
    return source.read(start, end, device) if isinstance(source, PagedTensorSequence) else source[..., start:end, :]


def _tile_shape(
    query: torch.Tensor, q_tokens: int, k_tokens: int, training: bool, pager: Optional[WorkingAttentionPager]
) -> Tuple[int, int]:
    q, k = max(1, int(q_tokens)), max(1, int(k_tokens))
    if pager is not None:
        while attention_tile_bytes(
            query.shape[0], query.shape[1], query.shape[-1], q, k, query.element_size(), training=training
        ) > pager.device_tile_budget_bytes:
            if q == k == 1:
                pager._pause("one exact attention cell exceeds live tile headroom")
            if q >= k and q > 1:
                q = max(1, q // 2)
            else:
                k = max(1, k // 2)
    return q, k


def _dropout_mask(
    shape: Tuple[int, ...], device: torch.device, seed: int, q_start: int, k_start: int, p: float, dtype: torch.dtype
) -> Union[float, torch.Tensor]:
    if p == 0.0:
        return 1.0
    generator = torch.Generator(device=device)
    generator.manual_seed((int(seed) + 1000003 * q_start + 1000033 * k_start) % (2 ** 63 - 1))
    return (torch.rand(shape, generator=generator, device=device) >= p).to(dtype) / (1.0 - p)


def _attention_forward(
    query: torch.Tensor,
    key: TensorSource,
    value: TensorSource,
    offset: int,
    query_tokens: int,
    key_tokens: int,
    dropout: float,
    seed: int,
    pager: Optional[WorkingAttentionPager],
    causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, heads, sequence, dimensions = query.shape
    compute_dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    scale = 1.0 / math.sqrt(float(dimensions))
    output = torch.empty(query.shape, dtype=compute_dtype, device=query.device)
    log_norm = torch.empty((batch, heads, sequence, 1), dtype=compute_dtype, device=query.device)
    for q_start in range(0, sequence, query_tokens):
        q_end = min(sequence, q_start + query_tokens)
        q = query[..., q_start:q_end, :].to(compute_dtype)
        maximum = torch.full((*q.shape[:-1], 1), -torch.inf, dtype=compute_dtype, device=query.device)
        mass = torch.zeros_like(maximum)
        numerator = torch.zeros_like(q)
        q_positions = torch.arange(offset + q_start, offset + q_end, device=query.device)
        key_limit = min(key.shape[-2], offset + q_end) if causal else key.shape[-2]
        for k_start in range(0, key_limit, key_tokens):
            k_end = min(key_limit, k_start + key_tokens)
            if pager is not None:
                estimate = attention_tile_bytes(batch, heads, dimensions, q_end - q_start, k_end - k_start, query.element_size())
                pager.admit_compute(estimate, device=query.device)
                pager._metrics["largestQueryTileTokens"] = max(pager._metrics["largestQueryTileTokens"], q_end - q_start)
                pager._metrics["largestKeyTileTokens"] = max(pager._metrics["largestKeyTileTokens"], k_end - k_start)
                pager._metrics["largestScoreElements"] = max(pager._metrics["largestScoreElements"], batch * heads * (q_end - q_start) * (k_end - k_start))
            k = _read(key, k_start, k_end, query.device).to(compute_dtype)
            v = _read(value, k_start, k_end, query.device).to(compute_dtype)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            k_positions = torch.arange(k_start, k_end, device=query.device)
            if causal:
                scores.masked_fill_(k_positions[None, :] > q_positions[:, None], -torch.inf)
            new_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            correction = (maximum - new_maximum).exp()
            weights = (scores - new_maximum).exp()
            mass = correction * mass + weights.sum(dim=-1, keepdim=True)
            drop = _dropout_mask(tuple(weights.shape), query.device, seed, q_start, k_start, dropout, compute_dtype)
            numerator = correction * numerator + torch.matmul(weights * drop, v)
            maximum = new_maximum
        output[..., q_start:q_end, :] = numerator / mass
        log_norm[..., q_start:q_end, :] = maximum + mass.log()
    return output, log_norm


class _ExactTiledCausalAttention(torch.autograd.Function):
    """Exact first-order gradients without retaining N-by-N probabilities."""

    @staticmethod
    def forward(ctx, query, key, value, offset, query_tokens, key_tokens, dropout, seed, pager, causal):
        output, log_norm = _attention_forward(query, key, value, offset, query_tokens, key_tokens, dropout, seed, pager, causal)
        ctx.settings = (offset, query_tokens, key_tokens, dropout, seed, pager, causal)
        ctx.device, ctx.dtype = query.device, query.dtype
        ctx.shapes = (tuple(query.shape), tuple(key.shape), tuple(value.shape))
        if pager is None:
            ctx.save_for_backward(query, key, value, output, log_norm)
            ctx.sources = None
        else:
            sources: List[PagedTensorSequence] = []
            try:
                for tensor in (query, key, value, output, log_norm):
                    sources.append(pager.sequence(tensor))
            except BaseException:
                for source in sources:
                    source.close()
                raise
            ctx.sources = tuple(sources)
        return output.to(query.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        offset, q_tokens, k_tokens, dropout, seed, pager, causal = ctx.settings
        query, key, value, output, log_norm = ctx.saved_tensors if ctx.sources is None else ctx.sources
        compute_dtype = torch.float64 if ctx.dtype == torch.float64 else torch.float32
        scale = 1.0 / math.sqrt(float(ctx.shapes[0][-1]))
        if pager is not None:
            total = sum(math.prod(shape) for shape in ctx.shapes) * max(4, grad_output.element_size())
            pager.admit_compute(total, "bounded training QKV gradients", device=ctx.device)
        d_query = torch.zeros(ctx.shapes[0], dtype=compute_dtype, device=ctx.device)
        d_key = torch.zeros(ctx.shapes[1], dtype=compute_dtype, device=ctx.device)
        d_value = torch.zeros(ctx.shapes[2], dtype=compute_dtype, device=ctx.device)
        batch, heads, sequence, dimensions = ctx.shapes[0]
        for q_start in range(0, sequence, q_tokens):
            q_end = min(sequence, q_start + q_tokens)
            q = _read(query, q_start, q_end, ctx.device).to(compute_dtype)
            do = grad_output[..., q_start:q_end, :].to(compute_dtype)
            o = _read(output, q_start, q_end, ctx.device).to(compute_dtype)
            norm = _read(log_norm, q_start, q_end, ctx.device)
            delta = (do * o).sum(dim=-1, keepdim=True)
            q_positions = torch.arange(offset + q_start, offset + q_end, device=ctx.device)
            key_limit = min(ctx.shapes[1][-2], offset + q_end) if causal else ctx.shapes[1][-2]
            for k_start in range(0, key_limit, k_tokens):
                k_end = min(key_limit, k_start + k_tokens)
                if pager is not None:
                    pager.admit_compute(attention_tile_bytes(batch, heads, dimensions, q_end - q_start, k_end - k_start, grad_output.element_size(), training=True), "attention backward tile", device=ctx.device)
                k = _read(key, k_start, k_end, ctx.device).to(compute_dtype)
                v = _read(value, k_start, k_end, ctx.device).to(compute_dtype)
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                k_positions = torch.arange(k_start, k_end, device=ctx.device)
                if causal:
                    scores.masked_fill_(k_positions[None, :] > q_positions[:, None], -torch.inf)
                probabilities = (scores - norm).exp()
                drop = _dropout_mask(tuple(probabilities.shape), ctx.device, seed, q_start, k_start, dropout, compute_dtype)
                dp = torch.matmul(do, v.transpose(-2, -1)) * drop
                ds = probabilities * (dp - delta)
                d_query[..., q_start:q_end, :].add_(torch.matmul(ds, k) * scale)
                d_key[..., k_start:k_end, :].add_(torch.matmul(ds.transpose(-2, -1), q) * scale)
                d_value[..., k_start:k_end, :].add_(torch.matmul((probabilities * drop).transpose(-2, -1), do))
        return d_query.to(ctx.dtype), d_key.to(ctx.dtype), d_value.to(ctx.dtype), None, None, None, None, None, None, None


def exact_causal_attention(
    query: torch.Tensor,
    key: TensorSource,
    value: TensorSource,
    *,
    position_offset: int = 0,
    query_chunk_tokens: int = 256,
    key_chunk_tokens: int = 256,
    dropout: float = 0.0,
    training: bool = False,
    pager: Optional[WorkingAttentionPager] = None,
    causal: bool = True,
) -> torch.Tensor:
    """Online-softmax exact attention with bounded query *and* key tiles.

    Cold K/V are transferred one tile at a time. Training uses recomputed exact
    derivatives and deterministic dropout masks, not detached approximate
    attention. Higher-order gradients are explicitly unsupported.
    """
    pager = pager if pager is not None else current_working_attention_pager()
    if pager is None and isinstance(key, PagedTensorSequence):
        pager = key.pager
    if query.ndim != 4 or len(key.shape) != 4 or key.shape != value.shape:
        raise ValueError("attention needs matching [batch, heads, tokens, width] K/V")
    if tuple(query.shape[:2]) != tuple(key.shape[:2]) or query.shape[-1] != key.shape[-1]:
        raise ValueError("attention Q/K/V batch, heads or width differ")
    offset = int(position_offset)
    if offset < 0 or (causal and key.shape[-2] < offset + query.shape[-2]) or query.shape[-2] < 1 or key.shape[-2] < 1:
        raise ValueError("causal attention positions are outside the supplied keys")
    if not 0.0 <= float(dropout) < 1.0:
        raise ValueError("attention dropout must be in [0, 1)")
    use_grad = torch.is_grad_enabled() and any(
        isinstance(tensor, torch.Tensor) and tensor.requires_grad for tensor in (query, key, value)
    )
    if use_grad and (not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor)):
        raise RuntimeError("append-only inference KV pages cannot enter a training graph")
    q_tokens, k_tokens = _tile_shape(query, query_chunk_tokens, key_chunk_tokens, use_grad, pager)
    p = float(dropout) if training else 0.0
    seed = int(torch.randint(0, 2 ** 31 - 1, (), device=query.device).item()) if p else 0
    if use_grad:
        return _ExactTiledCausalAttention.apply(query, key, value, offset, q_tokens, k_tokens, p, seed, pager, causal)
    return _attention_forward(query, key, value, offset, q_tokens, k_tokens, p, seed, pager, causal)[0].to(query.dtype)
