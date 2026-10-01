"""Bounded, format-compatible safe-tensor checkpoint I/O.

Only a bounded byte chunk is copied at once, including for one tensor larger
than the transfer budget. The safe-tensor reader validates the header before
exposing raw data offsets. No checkpoint file is ever opened writable.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from safetensors import safe_open


TRANSFER_BYTES = 4 * 1024 * 1024
_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
}
_DTYPE_CODES = {value: key for key, value in _DTYPES.items()}


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    byte_count: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


class BoundedTensorFile:
    """Verified shape inventory and byte-chunk reads from one immutable file."""

    def __init__(self, path: Path, *, chunk_bytes: int = TRANSFER_BYTES):
        if sys.byteorder != "little":
            raise RuntimeError("bounded safe-tensor I/O requires a little-endian host")
        self.path = Path(path)
        self.chunk_bytes = max(8, int(chunk_bytes))
        self.peak_transfer_bytes = 0
        # Let the installed primary reader reject malformed dtype, shape,
        # overlapping offsets, trailing bytes and invalid header layouts.
        with safe_open(str(self.path), framework="pt", device="cpu") as source:
            names = set(source.keys())
        with self.path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated safe-tensor header")
            length = struct.unpack("<Q", prefix)[0]
            if length > 100_000_000:
                raise ValueError("safe-tensor header exceeds the format limit")
            encoded = handle.read(length)
            if len(encoded) != length:
                raise ValueError("truncated safe-tensor header")
            header = json.loads(encoded)
        self.specs: dict[str, TensorSpec] = {}
        for name in sorted(names):
            value = header[name]
            dtype = _DTYPES.get(value["dtype"])
            if dtype is None:
                raise ValueError("unsupported bounded checkpoint dtype: %s" % value["dtype"])
            start, end = value["data_offsets"]
            shape = tuple(int(part) for part in value["shape"])
            self.specs[name] = TensorSpec(shape, dtype, 8 + length + start, end - start)
        self.metadata = dict(header.get("__metadata__", {}))

    @torch.no_grad()
    def copy_into(
        self,
        name: str,
        target: torch.Tensor,
        *,
        on_chunk: Optional[Callable[[str, int, int], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ) -> None:
        spec = self.specs[name]
        if target.dtype != spec.dtype or tuple(target.shape) != spec.shape:
            raise ValueError("checkpoint shape/dtype mismatch for %s" % name)
        if not target.is_contiguous() or target.layout != torch.strided:
            raise ValueError("bounded checkpoint target must be contiguous")
        item_bytes = target.element_size()
        step = max(item_bytes, self.chunk_bytes // item_bytes * item_bytes)
        flat = target.reshape(-1)
        scratch = bytearray(min(step, spec.byte_count))
        with self.path.open("rb") as handle:
            handle.seek(spec.offset)
            for start in range(0, spec.byte_count, step):
                if cancelled is not None and cancelled():
                    raise InterruptedError("checkpoint load cancelled at a bounded byte boundary")
                count = min(step, spec.byte_count - start)
                view = memoryview(scratch)[:count]
                if handle.readinto(view) != count:
                    raise ValueError("checkpoint tensor was truncated during load")
                source = torch.frombuffer(view, dtype=spec.dtype)
                flat[start // item_bytes : (start + count) // item_bytes].copy_(source)
                self.peak_transfer_bytes = max(self.peak_transfer_bytes, count)
                # Imported lazily to keep the tensor wire-format layer usable
                # without a pager and to avoid a module dependency cycle.
                from .native_core_paging import release_native_tensor_chunk
                release_native_tensor_chunk(target, start, count)
                from .router_state_paging import release_router_tensor_chunk
                release_router_tensor_chunk(target, start, count)
                if on_chunk is not None:
                    on_chunk(name, start, count)

    def tensor(self, name: str) -> torch.Tensor:
        """Deliberate full *single*-tensor access for small control recovery."""
        spec = self.specs[name]
        value = torch.empty(spec.shape, dtype=spec.dtype)
        self.copy_into(name, value)
        return value

    def chunks(self, name: str):
        """Yield temporary bounded CPU views; consume before requesting next."""
        spec = self.specs[name]
        item_bytes = torch.empty((), dtype=spec.dtype).element_size()
        step = max(item_bytes, self.chunk_bytes // item_bytes * item_bytes)
        scratch = bytearray(min(step, spec.byte_count))
        with self.path.open("rb") as handle:
            handle.seek(spec.offset)
            for start in range(0, spec.byte_count, step):
                count = min(step, spec.byte_count - start)
                view = memoryview(scratch)[:count]
                if handle.readinto(view) != count:
                    raise ValueError("checkpoint tensor was truncated during scan")
                self.peak_transfer_bytes = max(self.peak_transfer_bytes, count)
                yield torch.frombuffer(view, dtype=spec.dtype)


class LazyTensorMapping(Mapping[str, torch.Tensor]):
    """Recovery inventory: reads one requested tensor, never retains a core map."""

    def __init__(self, path: Path, *, max_tensor_bytes: int = TRANSFER_BYTES):
        self.reader = BoundedTensorFile(path)
        self.max_tensor_bytes = max(0, int(max_tensor_bytes))

    def __getitem__(self, name: str) -> torch.Tensor:
        if self.reader.specs[name].byte_count > self.max_tensor_bytes:
            raise ValueError("recovery control tensor exceeds bounded scratch allowance")
        return self.reader.tensor(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.reader.specs)

    def __len__(self) -> int:
        return len(self.reader.specs)


class AdmittedTensorMapping(Mapping[str, torch.Tensor]):
    """Selected final resident state, read lazily without retaining a source map.

    Large learned/module tensors must instead use ``load_module_bounded`` into
    their existing owners. This view is for activity/cache destinations that
    the caller adopts directly. Admission precedes allocation, transfer scratch
    stays chunk-bounded, and this mapping never caches the returned tensor.
    """

    def __init__(
        self,
        reader: BoundedTensorFile,
        *,
        names: Optional[Iterable[str]] = None,
        reserve: Optional[Callable[[int, str], None]] = None,
        cancelled: Optional[Callable[[], bool]] = None,
    ):
        self.reader = reader
        self._names = frozenset(reader.specs if names is None else names)
        if not self._names.issubset(reader.specs):
            raise ValueError("admitted checkpoint view contains an unknown tensor")
        self.reserve = reserve
        self.cancelled = cancelled
        self.destination_bytes_read = 0

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._names:
            raise KeyError(name)
        spec = self.reader.specs[name]
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("resident checkpoint state load cancelled before allocation")
        if self.reserve is not None:
            self.reserve(spec.byte_count + min(spec.byte_count, self.reader.chunk_bytes), name)
        target = torch.empty(spec.shape, dtype=spec.dtype, device="cpu")
        self.reader.copy_into(name, target, cancelled=self.cancelled)
        self.destination_bytes_read += spec.byte_count
        return target

    def __contains__(self, name: object) -> bool:
        return name in self._names

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._names))

    def __len__(self) -> int:
        return len(self._names)


def atomic_save_tensors_bounded(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Optional[Mapping[str, str]] = None,
    *,
    chunk_bytes: int = TRANSFER_BYTES,
    on_chunk: Optional[Callable[[torch.Tensor, int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    spill_lease: Any = None,
) -> None:
    """Write the standard safe-tensor wire format without a whole CPU clone.

    The caller serializes neural mutation with checkpointing. Shared tensor
    storage is rejected, as in safetensors.save_file. A cancelled/error write
    removes only its private temporary; the previously committed file survives.
    """
    if sys.byteorder != "little":
        raise RuntimeError("bounded safe-tensor I/O requires a little-endian host")
    path = Path(path)
    names = sorted(tensors)
    header: dict[str, Any] = {}
    spans: list[tuple[torch.device, int, int]] = []
    offset = 0
    for name in names:
        value = tensors[name]
        if not isinstance(name, str) or name == "__metadata__":
            raise ValueError("invalid checkpoint tensor name")
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError("checkpoint values must be dense tensors")
        if not value.is_contiguous() or value.dtype not in _DTYPE_CODES:
            raise ValueError("unsupported/noncontiguous bounded checkpoint tensor")
        count = value.numel() * value.element_size()
        if count:
            start = value.data_ptr()
            end = start + count
            if any(device == value.device and start < stop and first < end
                   for device, first, stop in spans):
                raise ValueError("checkpoint tensors share storage")
            spans.append((value.device, start, end))
        header[name] = {
            "dtype": _DTYPE_CODES[value.dtype], "shape": list(value.shape),
            "data_offsets": [offset, offset + count],
        }
        offset += count
    if metadata:
        if any(not isinstance(key, str) or not isinstance(value, str)
               for key, value in metadata.items()):
            raise ValueError("safe-tensor metadata must contain strings")
        header["__metadata__"] = dict(metadata)
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    if spill_lease is not None:
        spill_lease.bind_path(Path(temporary))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            with torch.no_grad():
                for name in names:
                    value = tensors[name].detach()
                    flat = value.reshape(-1)
                    elements = max(1, int(chunk_bytes) // value.element_size())
                    for start in range(0, value.numel(), elements):
                        if cancelled is not None and cancelled():
                            raise InterruptedError("checkpoint save cancelled at a bounded byte boundary")
                        end = min(value.numel(), start + elements)
                        block = flat[start:end].to(device="cpu").contiguous()
                        handle.write(memoryview(block.view(torch.uint8).numpy()))
                        from .native_core_paging import release_native_tensor_chunk
                        release_native_tensor_chunk(value, start * value.element_size(), (end - start) * value.element_size())
                        from .router_state_paging import release_router_tensor_chunk
                        release_router_tensor_chunk(value, start * value.element_size(), (end - start) * value.element_size())
                        if on_chunk is not None:
                            on_chunk(value, start * value.element_size(), (end - start) * value.element_size())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
        if spill_lease is not None:
            spill_lease.bind_path(path)
            spill_lease.commit(path=path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@torch.no_grad()
def load_module_bounded(
    module: torch.nn.Module,
    reader: BoundedTensorFile,
    prefix: str,
    *,
    overrides: Optional[Mapping[str, torch.Tensor]] = None,
    on_chunk: Optional[Callable[[str, int, int], None]] = None,
    cancelled: Optional[Callable[[], bool]] = None,
    excluded_prefixes: tuple[str, ...] = (),
) -> None:
    """Strict native state loading into already-sized owners in byte chunks."""
    defaults: dict[str, torch.Tensor] = {}
    if any(not value or not value.endswith(".") for value in excluded_prefixes):
        raise ValueError("bounded load exclusions must name complete child prefixes")
    def excluded(name):
        return any(name.startswith(value) for value in excluded_prefixes)
    for name, child in module.named_modules():
        child_path = name + "." if name else ""
        if excluded(child_path):
            continue
        prepare = getattr(child, "prepare_bounded_state_load", None)
        if callable(prepare):
            prepare(reader.specs, prefix + child_path)
        optional = getattr(child, "bounded_optional_state_defaults", None)
        if callable(optional):
            provided = optional(reader.specs, prefix + child_path)
            if not isinstance(provided, Mapping):
                raise ValueError("bounded optional state defaults must be an explicit mapping")
            for key, value in provided.items():
                if not isinstance(key, str) or not key or not isinstance(value, torch.Tensor):
                    raise ValueError("invalid bounded optional state default")
                relative_name = child_path + key
                if relative_name in defaults or prefix + relative_name in reader.specs:
                    raise ValueError("bounded optional state defaults overlap saved state")
                defaults[relative_name] = value
    state = {name: value for name, value in module.state_dict(keep_vars=True).items() if not excluded(name)}
    actual = {name[len(prefix):] for name in reader.specs if name.startswith(prefix) and not excluded(name[len(prefix):])}
    expected = set(state)
    if (actual | set(defaults)) != expected:
        raise ValueError("%s checkpoint mismatch (missing=%s, unexpected=%s)" % (
            prefix.rstrip("."), sorted(expected - actual - set(defaults)), sorted((actual | set(defaults)) - expected)))
    # Verify every destination before the first learned/control byte changes.
    for name, target in state.items():
        spec = reader.specs.get(prefix + name)
        source = defaults.get(name)
        dtype, shape = (spec.dtype, spec.shape) if spec is not None else (source.dtype, tuple(source.shape))
        if target.dtype != dtype or tuple(target.shape) != shape:
            raise ValueError("checkpoint shape/dtype mismatch for %s" % (prefix + name))
    for name, target in state.items():
        full_name = prefix + name
        replacement = (overrides or {}).get(full_name)
        if cancelled is not None and cancelled():
            raise InterruptedError("checkpoint load cancelled at a bounded state boundary")
        if name in defaults:
            if replacement is not None:
                raise ValueError("recovered checkpoint override overlaps optional default")
            target.copy_(defaults[name])
        elif replacement is None:
            reader.copy_into(full_name, target, on_chunk=on_chunk, cancelled=cancelled)
        else:
            spec = reader.specs[full_name]
            if replacement.dtype != spec.dtype or tuple(replacement.shape) != spec.shape:
                raise ValueError("invalid recovered checkpoint override")
            target.copy_(replacement)
    for name, child in module.named_modules():
        if excluded(name + "."):
            continue
        validate_bounded = getattr(child, "validate_bounded_state_load", None)
        if callable(validate_bounded):
            validate_bounded()
        validate = getattr(child, "_validate_packed", None)
        if callable(validate):
            validate()
