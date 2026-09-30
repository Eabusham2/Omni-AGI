"""Bounded exact-trit sources and format-compatible packing/verification.

Only transient int8 pages are decoded. No floating shadow or full sparse-edge
tensor is formed, and chunk/row boundaries do not change global trit order or
canonical final padding. Neural ownership remains with the source stores.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch


PAGE_PACKED_BYTES = 64 * 1024
PAGE_TRITS = PAGE_PACKED_BYTES * 4


@dataclass(frozen=True)
class TernaryPackedSource:
    shape: tuple[int, ...]
    pages: Callable[[], Iterable[torch.Tensor]]

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or any(type(value) is not int or value < 0 for value in self.shape):
            raise ValueError("streamed ternary shape must contain nonnegative integer dimensions")
        if not callable(self.pages):
            raise ValueError("streamed ternary source requires a replayable page factory")

    @property
    def dtype(self):
        return torch.int8

    def numel(self) -> int:
        return math.prod(self.shape)


def tensor_digest(shape) -> "hashlib._Hash":
    from .ternary_packing import _canonical_json
    digest = hashlib.sha256()
    digest.update(_canonical_json({"dtype": "int8", "shape": list(shape)}))
    digest.update(b"\0")
    return digest


def source_chunks(source: TernaryPackedSource, *, logical_digest=None) -> Iterator[bytes]:
    from .ternary_packing import _exact_ternary, encode_ternary_2bit, TernaryPackingError
    carry = torch.empty(0, dtype=torch.int8)
    seen = 0
    for page in source.pages():
        if not isinstance(page, torch.Tensor) or page.numel() > PAGE_TRITS:
            raise TernaryPackingError("streamed ternary source exceeds its bounded page")
        exact = _exact_ternary(page, name="streamed ternary page").reshape(-1)
        seen += int(exact.numel())
        if seen > source.numel():
            raise TernaryPackingError("streamed ternary source exceeds its declared shape")
        if logical_digest is not None:
            logical_digest.update(exact.numpy().tobytes())
        if carry.numel():
            exact = torch.cat((carry, exact))
        complete = (int(exact.numel()) // 4) * 4
        if complete:
            yield encode_ternary_2bit(exact[:complete])
        carry = exact[complete:].clone()
    if seen != source.numel():
        raise TernaryPackingError("streamed ternary source has incomplete coverage")
    if carry.numel():
        yield encode_ternary_2bit(carry)


def packed_rows_source(packed: torch.Tensor, shape: tuple[int, ...]) -> TernaryPackedSource:
    """Strip only verified internal row padding, retaining every logical trit."""

    count = math.prod(shape)
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("streamed authoritative packed source must be a byte matrix")
    rows = int(packed.shape[0])
    width = count // rows if rows else math.prod(shape[1:]) if len(shape) > 1 else 0
    if rows and count % rows or int(packed.shape[1]) != (width + 3) // 4:
        raise ValueError("streamed authoritative packed shape is inconsistent")
    version = int(packed._version)
    def pages():
        from .ternary_packing import decode_ternary_2bit
        for row in range(rows):
            consumed = 0
            for start in range(0, int(packed.shape[1]), PAGE_PACKED_BYTES):
                if int(packed._version) != version:
                    raise ValueError("packed source changed during streamed export")
                block = packed[row, start:start + PAGE_PACKED_BYTES].detach().cpu().contiguous()
                payload = block.numpy().tobytes()
                trits = min(len(payload) * 4, width - consumed)
                yield decode_ternary_2bit(payload, (trits,))
                consumed += trits
        if int(packed._version) != version:
            raise ValueError("packed source changed during streamed export")
    return TernaryPackedSource(tuple(shape), pages)


def dense_source(values: torch.Tensor) -> TernaryPackedSource:
    """Compatibility dense owners are read in bounded slices, never copied whole."""

    if not isinstance(values, torch.Tensor):
        raise ValueError("streamed dense owner must be a tensor")
    version = int(values._version)
    def pages():
        def strided_pages(view):
            if view.ndim < 2:
                flat = view.reshape(-1)
                for start in range(0, flat.numel(), PAGE_TRITS):
                    yield flat[start:start + PAGE_TRITS]
            else:
                for index in range(int(view.shape[0])):
                    yield from strided_pages(view[index])
        if values.is_contiguous():
            flat = values.detach().reshape(-1)
            iterator = (flat[start:start + PAGE_TRITS] for start in range(0, flat.numel(), PAGE_TRITS))
        else:
            iterator = strided_pages(values.detach())
        for page in iterator:
            if int(values._version) != version:
                raise ValueError("dense owner changed during streamed packing")
            yield page
        if int(values._version) != version:
            raise ValueError("dense owner changed during streamed packing")
    return TernaryPackedSource(tuple(values.shape), pages)


def verify_file(path: Path, shape: tuple[int, ...]) -> tuple[str, str]:
    """Check complete packed bytes, reserved codes and exact logical digest."""

    from .ternary_packing import decode_ternary_2bit, TernaryIntegrityError
    count = math.prod(shape)
    expected = (count + 3) // 4
    before = path.stat()
    if before.st_size != expected:
        raise TernaryIntegrityError("streamed ternary shard length mismatch")
    packed_digest = hashlib.sha256()
    logical = tensor_digest(shape)
    consumed = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(PAGE_PACKED_BYTES)
            if not block:
                break
            packed_digest.update(block)
            trits = min(len(block) * 4, count - consumed)
            decoded = decode_ternary_2bit(block, (trits,))
            logical.update(decoded.numpy().tobytes())
            consumed += trits
    after = path.stat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if consumed != count or identity(before) != identity(after):
        raise TernaryIntegrityError("streamed ternary shard changed or has incomplete coverage")
    return packed_digest.hexdigest(), logical.hexdigest()


def source_hashes(source: TernaryPackedSource) -> tuple[str, str]:
    logical = tensor_digest(source.shape)
    packed = hashlib.sha256()
    for block in source_chunks(source, logical_digest=logical):
        packed.update(block)
    return packed.hexdigest(), logical.hexdigest()
