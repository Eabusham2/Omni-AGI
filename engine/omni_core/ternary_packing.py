"""Safe, deterministic packed-ternary synapse shards.

The native OmniCortex path stores eligible learned weights as packed ternary
codes and mutates them directly. This module writes their exact
``{-1, 0, +1}`` values in a
small, non-executable format:

* ``manifest.json`` is canonical JSON and describes every eligible tensor.
* ``ternary-*.bin`` files contain four 2-bit values per byte.
* ``manifest.sha256`` authenticates the exact manifest bytes.

Integration API
---------------

``export_module_ternary_shards`` is the normal integration point.  Pass the
authoritative model roots, any dynamically allocated synapse tensors, and the
complete set of names expected by the architecture audit::

    manifest = export_module_ternary_shards(
        engine_path / "packed-ternary",
        {
            "decoder": brain.decoder,
            "router": brain.router,
            "modalities": brain.modalities,
        },
        dynamic_synapses={
            "substrate.dynamic_synapses": effective_synapse_tensor,
        },
        expected_names=architecture_ternary_names,
    )

The exporter discovers packed-adaptive projections and STDP synapses. A
custom module may declare ``ternary_eligible = True`` only to make omission
fail closed; it must be converted to a supported exact-ternary projection
before export.  Supplying ``expected_names`` is how the caller makes an
architecture contract explicit: missing *or unexpected* tensors abort export.

``verify_ternary_shards`` must be called before installing or loading a bundle.
It validates the canonical manifest checksum, coverage contract, shard sizes
and hashes, reserved codes, padding, shapes, and decoded tensor hashes.  It
returns only CPU ``torch.int8`` tensors and never deserializes Python objects.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .model import (
    PACKED_AUTHORITATIVE_PROJECTION_TYPES,
    TERNARY_PROJECTION_TYPES,
)
from .spiking import STDPSynapses
from .streamed_ternary import TernaryPackedSource, packed_rows_source, dense_source, source_chunks, tensor_digest, verify_file
from .packed_connections import CONNECTION_STATE_VERSION
from .native_core_paging import release_native_tensor_chunk


FORMAT_NAME = "omni-packed-ternary"
FORMAT_VERSION = 2
MANIFEST_NAME = "manifest.json"
MANIFEST_CHECKSUM_NAME = "manifest.sha256"

_CODE_TO_VALUE = (-1, 0, 1)
_ZERO_CODE = 1
_RESERVED_CODE = 3
_ALLOWED_KINDS = frozenset({"projection", "dynamic-synapse"})
_PARALLEL_RECORD = struct.Struct("<QQQBB")


class TernaryPackingError(ValueError):
    """Base error for an invalid packed-ternary operation."""


class TernaryCoverageError(TernaryPackingError):
    """The exported tensors do not match the architecture contract."""


class TernaryIntegrityError(TernaryPackingError):
    """A stored manifest, shard, or decoded tensor failed verification."""


@dataclass(frozen=True)
class ParallelConnectionSource:
    count: int
    records: Callable[[], Iterable[Tuple[Any, ...]]]


@dataclass(frozen=True)
class ConnectionStateSource:
    """Replayable bounded strength/resistance pages in the sign stream's order.

    Database/paged graph owners provide a generation-bound page factory; they
    need not gather every connection's byte into one resident matrix.
    """

    shape: Tuple[int, int]
    pages: Callable[[], Iterable[torch.Tensor]]

    def __post_init__(self):
        if (not isinstance(self.shape, tuple) or len(self.shape) != 2
                or any(type(size) is not int or size < 0 for size in self.shape)
                or not callable(self.pages)):
            raise TernaryPackingError("connection source requires a nonnegative matrix shape and page factory")

    def numel(self) -> int:
        return math.prod(self.shape)


def _connection_state_chunks(state):
    if isinstance(state, ConnectionStateSource):
        seen = 0
        for page in state.pages():
            if not isinstance(page, torch.Tensor) or page.dtype != torch.uint8 or page.numel() > 1 << 20:
                raise TernaryPackingError("connection source page must be bounded uint8 state")
            seen += page.numel()
            if seen > state.numel():
                raise TernaryPackingError("connection source exceeds its declared shape")
            yield page.detach().cpu().contiguous().numpy().tobytes()
        if seen != state.numel():
            raise TernaryPackingError("connection source has incomplete coverage")
        return
    if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or not state.is_contiguous():
        raise TernaryPackingError("connection state must be a contiguous uint8 matrix")
    version = int(state._version)
    flat = state.reshape(-1)
    for start in range(0, flat.numel(), 1 << 20):
        if int(state._version) != version:
            raise TernaryPackingError("connection state changed during streamed export")
        block = flat[start:start + (1 << 20)].detach().cpu().numpy().tobytes()
        yield block
        release_native_tensor_chunk(state, start, len(block))
    if int(state._version) != version:
        raise TernaryPackingError("connection state changed during streamed export")


@dataclass(frozen=True)
class TernaryTensorSpec:
    """One exact-ternary tensor and its post-projection floating scale."""

    name: str
    values: Union[torch.Tensor, TernaryPackedSource]
    scale: float = 1.0
    kind: str = "projection"
    source_dtype: Optional[str] = None
    connection_state: Optional[Union[torch.Tensor, ConnectionStateSource]] = None
    parallel_connections: Union[Tuple[Tuple[Any, ...], ...], ParallelConnectionSource] = ()
    packed_row_layout: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class VerifiedTernaryBundle:
    """Verified manifest and decoded CPU int8 inference tensors."""

    manifest: Dict[str, Any]
    tensors: Dict[str, torch.Tensor]
    connection_states: Dict[str, torch.Tensor] = field(default_factory=dict)
    parallel_connections: Dict[str, Tuple[Tuple[int, ...], ...]] = field(default_factory=dict)


DynamicSynapse = Union[TernaryTensorSpec, TernaryPackedSource, torch.Tensor, nn.Module]
ModuleRoots = Union[nn.Module, Mapping[str, nn.Module]]


def _parallel_source(module, *, bias=False):
    pool = module.parallel_connections
    revision = pool._revision
    count = pool.count(bias=bias)
    def records():
        if pool._revision != revision:
            raise TernaryPackingError("parallel connections changed during streamed export")
        for record in module.iter_parallel_connection_records(bias=bias):
            if pool._revision != revision:
                raise TernaryPackingError("parallel connections changed during streamed export")
            yield record
        if pool._revision != revision:
            raise TernaryPackingError("parallel connections changed during streamed export")
    return ParallelConnectionSource(count, records)


def connection_matrix_shape(shape, kind, layout=None):
    """Map saved logical geometry to the native packed row ordering."""
    if layout is None:
        return (int(shape[0]), math.prod(shape[1:])) if len(shape) >= 2 else (1, math.prod(shape))
    if (not isinstance(layout, dict) or set(layout) != {"kind", "groups"}
            or layout["kind"] != "transpose-convolution" or kind != "projection"
            or len(shape) not in (3, 4, 5) or type(layout["groups"]) is not int
            or layout["groups"] < 1 or shape[0] < 1 or shape[1] < 1
            or any(value < 1 for value in shape[2:]) or shape[0] % layout["groups"]):
        raise TernaryIntegrityError("packed row layout is invalid")
    return (int(shape[1]) * layout["groups"] * math.prod(shape[2:]), int(shape[0]) // layout["groups"])


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TernaryPackingError("manifest metadata is not canonical JSON") from error
    return encoded.encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _normalize_shape(shape: Sequence[int]) -> Tuple[int, ...]:
    normalized = []
    for dimension in shape:
        if isinstance(dimension, bool):
            raise TernaryPackingError("tensor shapes must contain integers")
        try:
            value = int(dimension)
        except (TypeError, ValueError) as error:
            raise TernaryPackingError("tensor shapes must contain integers") from error
        if value != dimension or value < 0:
            raise TernaryPackingError("tensor shape dimensions must be non-negative")
        normalized.append(value)
    return tuple(normalized)


def _exact_ternary(values: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TernaryPackingError("%s is not a tensor" % name)
    if values.is_complex():
        raise TernaryPackingError("%s has a complex dtype" % name)
    contiguous = values.detach().cpu().contiguous()
    if contiguous.is_floating_point() and not bool(torch.isfinite(contiguous).all()):
        raise TernaryPackingError("%s contains a non-finite value" % name)
    allowed = (contiguous == -1) | (contiguous == 0) | (contiguous == 1)
    if not bool(allowed.all()):
        raise TernaryPackingError(
            "%s is not exact ternary; only -1, 0, and +1 may be packed" % name
        )
    return contiguous.to(torch.int8)


def _tensor_sha256(values: torch.Tensor) -> str:
    tensor = _exact_ternary(values, name="decoded tensor")
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "dtype": "int8",
                "shape": [int(value) for value in tensor.shape],
            }
        )
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def encode_ternary_2bit(values: torch.Tensor) -> bytes:
    """Pack exact ternary values deterministically, four values per byte.

    Codes are ``00 = -1``, ``01 = 0``, ``10 = +1``, and ``11 = reserved``.
    Values occupy a byte from least-significant to most-significant pair.
    Unused pairs in the final byte are encoded as zero, which the decoder also
    verifies.
    """

    tensor = _exact_ternary(values, name="ternary tensor").reshape(-1)
    count = int(tensor.numel())
    if count == 0:
        return b""
    codes = tensor.to(torch.int16) + 1
    padding = (-count) % 4
    if padding:
        codes = torch.cat(
            (codes, torch.full((padding,), _ZERO_CODE, dtype=torch.int16))
        )
    groups = codes.reshape(-1, 4)
    packed = (
        groups[:, 0]
        | (groups[:, 1] << 2)
        | (groups[:, 2] << 4)
        | (groups[:, 3] << 6)
    ).to(torch.uint8)
    return packed.numpy().tobytes(order="C")


def decode_ternary_2bit(payload: bytes, shape: Sequence[int]) -> torch.Tensor:
    """Decode a packed tensor and reject reserved codes or non-zero padding."""

    normalized_shape = _normalize_shape(shape)
    count = math.prod(normalized_shape)
    expected_bytes = (count + 3) // 4
    if len(payload) != expected_bytes:
        raise TernaryIntegrityError(
            "packed tensor length is %d bytes; expected %d"
            % (len(payload), expected_bytes)
        )
    if not payload:
        return torch.empty(normalized_shape, dtype=torch.int8)

    packed = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    codes = torch.stack(
        tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)),
        dim=1,
    ).reshape(-1)
    active_codes = codes[:count]
    if bool((active_codes == _RESERVED_CODE).any()):
        raise TernaryIntegrityError("packed tensor contains the reserved 2-bit code")
    padding_codes = codes[count:]
    if padding_codes.numel() and not bool((padding_codes == _ZERO_CODE).all()):
        raise TernaryIntegrityError("packed tensor has non-canonical padding")
    return (active_codes.to(torch.int8) - 1).reshape(normalized_shape)


def _qualified_name(root_name: str, module_name: str, field: str) -> str:
    components = [component for component in (root_name, module_name, field) if component]
    return ".".join(components)


def _module_spec(root_name: str, module_name: str, module: nn.Module) -> TernaryTensorSpec:
    supported_projection = isinstance(module, TERNARY_PROJECTION_TYPES)
    supported_synapse = isinstance(module, STDPSynapses)
    claimed_eligible = bool(getattr(module, "ternary_eligible", False)) or hasattr(
        module, "ternary"
    )
    if not (supported_projection or supported_synapse):
        if claimed_eligible:
            name = _qualified_name(root_name, module_name, "weight")
            raise TernaryCoverageError(
                "%s claims ternary eligibility but uses an unsupported/dense "
                "projection; convert it to a mandatory ternary module" % name
            )
        raise TernaryCoverageError("module is not an eligible ternary tensor source")
    if getattr(module, "ternary", False) is not True:
        name = _qualified_name(root_name, module_name, "weight")
        raise TernaryCoverageError("%s is eligible but not marked ternary" % name)
    effective_weight = getattr(module, "effective_weight", None)
    if not callable(effective_weight):
        name = _qualified_name(root_name, module_name, "weight")
        raise TernaryCoverageError("%s has no effective ternary weight" % name)

    if supported_synapse:
        source = module._packed_weights
        return TernaryTensorSpec(
            name=_qualified_name(root_name, module_name, "weights"),
            values=packed_rows_source(source, (int(module.post_neurons), int(module.pre_neurons))),
            scale=1.0,
            kind="dynamic-synapse",
            source_dtype="packed-2bit",
        )

    if not isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES):
        raise TernaryCoverageError("projection lacks authoritative packed storage")
    return TernaryTensorSpec(
        name=_qualified_name(root_name, module_name, "weight"),
        values=packed_rows_source(module._packed_forward_weight, tuple(module.ternary_weight_shape)),
        scale=float(module._packed_forward_scale.item()),
        kind="projection",
        source_dtype="packed-2bit",
        connection_state=module._connection_state,
        parallel_connections=_parallel_source(module),
        packed_row_layout={"kind": "transpose-convolution", "groups": module.groups}
            if getattr(module, "transposed", False) else None,
    )


def _root_items(roots: ModuleRoots) -> Iterable[Tuple[str, nn.Module]]:
    if isinstance(roots, nn.Module):
        return (("model", roots),)
    items = []
    for name, module in roots.items():
        if not isinstance(name, str) or not name or "\0" in name:
            raise TernaryCoverageError("module root names must be non-empty strings")
        if not isinstance(module, nn.Module):
            raise TernaryCoverageError("%s is not a torch module root" % name)
        items.append((name, module))
    return tuple(sorted(items))


def collect_module_ternary_tensors(
    roots: ModuleRoots,
    *,
    dynamic_synapses: Optional[Mapping[str, DynamicSynapse]] = None,
) -> Tuple[TernaryTensorSpec, ...]:
    """Discover all built-in eligible projections and explicit synapse tensors.

    Dense modules are not guessed from their names.  An architecture that
    considers a dense module eligible must either mark it ``ternary_eligible``
    (which raises immediately) or include its fully qualified weight name in
    ``expected_names`` when exporting (which raises a coverage mismatch).
    """

    collected: Dict[str, TernaryTensorSpec] = {}
    for root_name, root in _root_items(roots):
        for module_name, module in root.named_modules():
            supported = isinstance(
                module, TERNARY_PROJECTION_TYPES
            ) or isinstance(module, STDPSynapses)
            claimed = bool(getattr(module, "ternary_eligible", False)) or hasattr(
                module, "ternary"
            )
            if not (supported or claimed):
                continue
            if isinstance(module, STDPSynapses) and hasattr(module, "_sparse_state"):
                if getattr(module, "ternary", False) is not True:
                    raise TernaryCoverageError("sparse router is not marked ternary")
                module._validate_packed()
                for key, block in module.blocks.items():
                    name = _qualified_name(root_name, module_name, "blocks." + key + ".weights")
                    if name in collected:
                        raise TernaryCoverageError("duplicate eligible tensor name: %s" % name)
                    collected[name] = TernaryTensorSpec(
                        name=name, values=packed_rows_source(block._packed_weights, (block.rows, block.columns)),
                        scale=1.0, kind="dynamic-synapse", source_dtype="packed-2bit",
                        connection_state=getattr(block, "_connection_state", None))
                continue
            spec = _module_spec(root_name, module_name, module)
            if spec.name in collected:
                raise TernaryCoverageError(
                    "duplicate eligible tensor name: %s" % spec.name
                )
            collected[spec.name] = spec
            if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES) and module.has_bias:
                bias_name = _qualified_name(root_name, module_name, "bias")
                if bias_name in collected:
                    raise TernaryCoverageError(
                        "duplicate eligible tensor name: %s" % bias_name
                    )
                packed_bias = module._packed_forward_bias
                if packed_bias is None:
                    raise TernaryCoverageError("%s has no packed bias" % bias_name)
                collected[bias_name] = TernaryTensorSpec(
                    name=bias_name,
                    values=packed_rows_source(packed_bias, tuple(module.ternary_bias_shape)),
                    scale=float(module._packed_forward_scale.item()),
                    kind="projection",
                    source_dtype="packed-2bit",
                    connection_state=module._bias_connection_state,
                    parallel_connections=_parallel_source(module, bias=True),
                )

    for name, source in sorted((dynamic_synapses or {}).items()):
        if not isinstance(name, str) or not name or "\0" in name:
            raise TernaryCoverageError(
                "dynamic synapse names must be non-empty strings"
            )
        if isinstance(source, TernaryTensorSpec):
            if source.name != name:
                raise TernaryCoverageError(
                    "dynamic synapse mapping key does not match its tensor spec"
                )
            spec = source
        elif isinstance(source, TernaryPackedSource):
            spec = TernaryTensorSpec(name=name, values=source, kind="dynamic-synapse", source_dtype="packed-2bit")
        elif isinstance(source, torch.Tensor):
            spec = TernaryTensorSpec(
                name=name,
                values=source,
                scale=1.0,
                kind="dynamic-synapse",
                source_dtype=str(source.dtype).replace("torch.", ""),
            )
        elif isinstance(source, nn.Module):
            spec = _module_spec("", name.rsplit(".", 1)[0], source)
            spec = TernaryTensorSpec(
                name=name,
                values=spec.values,
                scale=spec.scale,
                kind="dynamic-synapse",
                source_dtype=spec.source_dtype,
                connection_state=spec.connection_state,
                parallel_connections=spec.parallel_connections,
                packed_row_layout=spec.packed_row_layout,
            )
        else:
            raise TernaryCoverageError(
                "%s is not a tensor, tensor spec, or ternary module" % name
            )
        if spec.name in collected:
            raise TernaryCoverageError(
                "duplicate eligible tensor name: %s" % spec.name
            )
        collected[spec.name] = spec

    return tuple(collected[name] for name in sorted(collected))


def inspect_module_ternary_layout(
    roots: ModuleRoots,
    *,
    dynamic_synapses: Mapping[str, Union[TernaryTensorSpec, torch.Tensor, TernaryPackedSource]],
) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Inspect eligible names and shapes without evaluating accelerator weights.

    A cold checkpoint load only needs to establish that its authenticated pack
    still describes the current architecture. Computing every effective MPS
    projection just to obtain names causes a synchronous GPU scalar read per
    module even when the saved parameter checksum has not changed.
    """

    layout: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for root_name, root in _root_items(roots):
        for module_name, module in root.named_modules():
            supported_projection = isinstance(module, TERNARY_PROJECTION_TYPES)
            supported_synapse = isinstance(module, STDPSynapses)
            claimed = bool(getattr(module, "ternary_eligible", False)) or hasattr(
                module, "ternary"
            )
            if not (supported_projection or supported_synapse):
                if claimed:
                    raise TernaryCoverageError(
                        "%s claims ternary eligibility but uses an unsupported/dense "
                        "projection"
                        % _qualified_name(root_name, module_name, "weight")
                    )
                continue
            if supported_synapse and hasattr(module, "_sparse_state"):
                if getattr(module, "ternary", False) is not True:
                    raise TernaryCoverageError("sparse router is not marked ternary")
                for key, block in module.blocks.items():
                    name = _qualified_name(root_name, module_name, "blocks." + key + ".weights")
                    if name in layout:
                        raise TernaryCoverageError("%s has a duplicate weight" % name)
                    layout[name] = ((block.rows, block.columns), "dynamic-synapse")
                continue
            field = "weights" if supported_synapse else "weight"
            name = _qualified_name(root_name, module_name, field)
            if getattr(module, "ternary", False) is not True:
                raise TernaryCoverageError("%s is eligible but not marked ternary" % name)
            if not callable(getattr(module, "effective_weight", None)):
                raise TernaryCoverageError("%s has no effective ternary weight" % name)
            if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES):
                source_shape = module.ternary_weight_shape
            else:
                source = getattr(module, field)
                if not isinstance(source, torch.Tensor):
                    raise TernaryCoverageError("%s has an invalid weight" % name)
                source_shape = tuple(int(value) for value in source.shape)
            if name in layout:
                raise TernaryCoverageError("%s has a duplicate weight" % name)
            layout[name] = (
                source_shape,
                "dynamic-synapse" if supported_synapse else "projection",
            )
            if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES) and module.has_bias:
                bias_name = _qualified_name(root_name, module_name, "bias")
                if bias_name in layout:
                    raise TernaryCoverageError("%s has a duplicate bias" % bias_name)
                layout[bias_name] = (module.ternary_bias_shape, "projection")
    for name, source in dynamic_synapses.items():
        if not isinstance(name, str) or not name or "\0" in name:
            raise TernaryCoverageError("dynamic synapse names must be non-empty strings")
        kind = "dynamic-synapse"
        if isinstance(source, TernaryTensorSpec):
            if source.name != name or source.kind not in _ALLOWED_KINDS:
                raise TernaryCoverageError("dynamic source specification disagrees with its identity")
            kind, source = source.kind, source.values
        if not isinstance(source, (torch.Tensor, TernaryPackedSource)) or name in layout:
            raise TernaryCoverageError("%s is not a unique dynamic tensor" % name)
        layout[name] = (tuple(int(value) for value in source.shape), kind)
    return dict(sorted(layout.items()))


def _normalize_expected_names(
    names: Optional[Iterable[str]],
) -> Optional[Tuple[str, ...]]:
    if names is None:
        return None
    normalized = []
    for name in names:
        if not isinstance(name, str) or not name or "\0" in name:
            raise TernaryCoverageError(
                "expected tensor names must be non-empty strings"
            )
        normalized.append(name)
    if len(set(normalized)) != len(normalized):
        raise TernaryCoverageError("expected tensor names contain duplicates")
    return tuple(sorted(normalized))


def _assert_coverage(
    actual_names: Iterable[str],
    expected_names: Optional[Iterable[str]],
) -> Tuple[str, ...]:
    actual = tuple(sorted(actual_names))
    expected = _normalize_expected_names(expected_names)
    if expected is None:
        return actual
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise TernaryCoverageError(
            "ternary coverage mismatch; missing=%s unexpected=%s"
            % (missing, unexpected)
        )
    return expected


def export_ternary_shards(
    destination: Path,
    tensors: Iterable[TernaryTensorSpec],
    *,
    expected_names: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write tensor specs and return the verified, canonical manifest.

    Files use generation/content-derived names and the manifest is replaced
    last.  An interrupted export therefore cannot make a new partial
    generation authoritative.
    """

    destination = Path(destination)
    specs: Dict[str, TernaryTensorSpec] = {}
    for spec in tensors:
        if not isinstance(spec, TernaryTensorSpec):
            raise TernaryPackingError("all exported values must be tensor specs")
        if not spec.name or "\0" in spec.name:
            raise TernaryPackingError("tensor names must be non-empty strings")
        if spec.name in specs:
            raise TernaryCoverageError("duplicate tensor name: %s" % spec.name)
        if spec.kind not in _ALLOWED_KINDS:
            raise TernaryPackingError(
                "%s has unsupported tensor kind %s" % (spec.name, spec.kind)
            )
        if not math.isfinite(float(spec.scale)) or float(spec.scale) <= 0.0:
            raise TernaryPackingError("%s has an invalid scale" % spec.name)
        specs[spec.name] = spec

    contract = _assert_coverage(specs, expected_names)
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    shard_entries = []
    for index, name in enumerate(sorted(specs)):
        spec = specs[name]
        source = spec.values if isinstance(spec.values, TernaryPackedSource) else dense_source(spec.values)
        logical_digest = tensor_digest(source.shape)
        packed_digest = hashlib.sha256()
        descriptor, temporary_name = tempfile.mkstemp(prefix=".ternary-stream-", dir=destination)
        byte_length = 0
        try:
            with os.fdopen(descriptor, "wb") as handle:
                for block in source_chunks(source, logical_digest=logical_digest):
                    handle.write(block)
                    packed_digest.update(block)
                    byte_length += len(block)
                handle.flush()
                os.fsync(handle.fileno())
            packed_hash = packed_digest.hexdigest()
            filename = "ternary-%05d-%s.bin" % (index, packed_hash[:16])
            os.replace(temporary_name, destination / filename)
        finally:
            Path(temporary_name).unlink(missing_ok=True)
        entry = {
            "name": name,
            "kind": spec.kind,
            "shape": [int(value) for value in source.shape],
            "dtype": "int8",
            "sourceDtype": spec.source_dtype
            or str(spec.values.dtype).replace("torch.", ""),
            "scale": float(spec.scale),
            "numel": source.numel(),
            "shard": filename,
            "byteOffset": 0,
            "byteLength": byte_length,
            "packedSha256": packed_hash,
            "tensorSha256": logical_digest.hexdigest(),
        }
        expected_connection_shape = connection_matrix_shape(source.shape, spec.kind, spec.packed_row_layout)
        if spec.packed_row_layout is not None:
            entry["packedRowLayout"] = dict(spec.packed_row_layout)
        entries.append(entry)
        shard_entries.append(
            {
                "file": filename,
                "byteLength": byte_length,
                "sha256": packed_hash,
            }
        )
        if spec.connection_state is None:
            entry["connectionState"] = {"encoding": "unit-strength", "strength": 1}
        else:
            state = spec.connection_state
            if not isinstance(state, (torch.Tensor, ConnectionStateSource)) or tuple(state.shape) != expected_connection_shape:
                raise TernaryPackingError("%s connection state is not a matching uint8 matrix" % name)
            descriptor, temporary_name = tempfile.mkstemp(prefix=".connection-stream-", dir=destination)
            digest = hashlib.sha256()
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    for block in _connection_state_chunks(state):
                        handle.write(block)
                        digest.update(block)
                    handle.flush()
                    os.fsync(handle.fileno())
                connection_hash = digest.hexdigest()
                connection_file = "connections-%05d-%s.bin" % (index, connection_hash[:16])
                os.replace(temporary_name, destination / connection_file)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
            entry["connectionState"] = {
                "formatVersion": CONNECTION_STATE_VERSION,
                "encoding": "uint4-strength-minus-one-high4-resistance",
                "shape": list(state.shape), "shard": connection_file,
                "byteLength": state.numel(), "sha256": connection_hash,
            }
            shard_entries.append({"file": connection_file, "byteLength": state.numel(), "sha256": connection_hash})
        parallel_count = spec.parallel_connections.count if isinstance(spec.parallel_connections, ParallelConnectionSource) else len(spec.parallel_connections)
        if parallel_count:
            if spec.connection_state is None:
                raise TernaryPackingError("parallel contacts require explicit connection state")
            if isinstance(spec.parallel_connections, ParallelConnectionSource):
                rows = ((int(row), int(column), int(key.rsplit("_n", 1)[1]), int(sign), int(state))
                    for key, row, column, sign, state in spec.parallel_connections.records())
            else:
                rows = iter(sorted((int(row), int(column), int(key.rsplit("_n", 1)[1]), int(sign), int(state))
                    for key, row, column, sign, state in spec.parallel_connections))
            descriptor, temporary_name = tempfile.mkstemp(prefix=".parallel-stream-", dir=destination)
            digest = hashlib.sha256()
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    previous = None
                    written = 0
                    for record in rows:
                        row, column, ordinal, sign, state_value = record
                        if (row < 0 or column < 0 or ordinal < 0 or row >= spec.connection_state.shape[0]
                                or column >= spec.connection_state.shape[1] or sign >> 2 != 0x15 or sign & 3 == 3
                                or not 0 <= state_value <= 255 or previous is not None and record[:3] <= previous):
                            raise TernaryPackingError("parallel contact has an invalid endpoint, identity or code")
                        payload = _PARALLEL_RECORD.pack(*record)
                        handle.write(payload)
                        digest.update(payload)
                        written += 1
                        previous = record[:3]
                    if written != parallel_count:
                        raise TernaryPackingError("parallel contact stream count changed during export")
                    handle.flush()
                    os.fsync(handle.fileno())
                parallel_hash = digest.hexdigest()
                parallel_file = "parallel-%05d-%s.bin" % (index, parallel_hash[:16])
                os.replace(temporary_name, destination / parallel_file)
            finally:
                Path(temporary_name).unlink(missing_ok=True)
            entry["parallelConnections"] = {"encoding": "u64-row-column-ordinal-u8-sign-state-le",
                "count": parallel_count, "byteLength": parallel_count * _PARALLEL_RECORD.size,
                "shard": parallel_file, "sha256": parallel_hash}
            shard_entries.append({"file": parallel_file, "byteLength": parallel_count * _PARALLEL_RECORD.size, "sha256": parallel_hash})

    manifest_without_hash: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "formatVersion": FORMAT_VERSION,
        "architecture": "OmniCortex",
        "encoding": {
            "bitsPerValue": 2,
            "byteOrder": "four-values-lsb-first",
            "codes": {"-1": 0, "0": 1, "+1": 2},
            "reservedCode": _RESERVED_CODE,
            "paddingValue": 0,
        },
        "coverage": {
            "eligibleTensorCount": len(contract),
            "eligibleTensorNames": list(contract),
            "complete": True,
        },
        "tensors": entries,
        "shards": shard_entries,
        "metadata": dict(metadata or {}),
    }
    content_hash = _sha256(_canonical_json(manifest_without_hash))
    manifest = dict(manifest_without_hash)
    manifest["contentSha256"] = content_hash
    manifest_bytes = _canonical_json(manifest)
    _atomic_write(destination / MANIFEST_NAME, manifest_bytes)
    _atomic_write(
        destination / MANIFEST_CHECKSUM_NAME,
        (_sha256(manifest_bytes) + "\n").encode("ascii"),
    )
    # Verify what reached disk instead of trusting the in-memory tensors.
    return verify_ternary_shards(
        destination, expected_names=contract, retain_names=()
    ).manifest


def export_module_ternary_shards(
    destination: Path,
    roots: ModuleRoots,
    *,
    dynamic_synapses: Optional[Mapping[str, DynamicSynapse]] = None,
    expected_names: Optional[Iterable[str]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Discover, strictly cover, export, and verify an OmniCortex generation."""

    tensors = collect_module_ternary_tensors(
        roots, dynamic_synapses=dynamic_synapses
    )
    return export_ternary_shards(
        destination,
        tensors,
        expected_names=expected_names,
        metadata=metadata,
    )


def _manifest_entry_list(value: Any, field: str) -> Sequence[Mapping[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TernaryIntegrityError("manifest %s must be a list of objects" % field)
    return value


def _safe_shard_path(destination: Path, filename: Any) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise TernaryIntegrityError("manifest contains an unsafe shard path")
    path = destination / filename
    if path.is_symlink() or not path.is_file():
        raise TernaryIntegrityError("missing or unsafe ternary shard: %s" % filename)
    if path.resolve().parent != destination.resolve():
        raise TernaryIntegrityError("ternary shard escapes its bundle directory")
    return path


def verify_ternary_shards(
    destination: Path,
    *,
    expected_names: Optional[Iterable[str]] = None,
    retain_names: Optional[Iterable[str]] = None,
) -> VerifiedTernaryBundle:
    """Verify every shard; optionally retain only selected decoded tensors.

    The default retains all tensors for existing callers. An empty selection
    still decodes and checks every shard, but releases each decoded tensor
    before reading the next one. This bounds cold-load verification memory
    without relaxing packed, decoded, manifest, or coverage checks.
    """

    destination = Path(destination)
    manifest_path = destination / MANIFEST_NAME
    checksum_path = destination / MANIFEST_CHECKSUM_NAME
    try:
        manifest_bytes = manifest_path.read_bytes()
        checksum_text = checksum_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise TernaryIntegrityError("packed ternary manifest is incomplete") from error
    if len(checksum_text) != 64 or any(
        character not in "0123456789abcdef" for character in checksum_text
    ):
        raise TernaryIntegrityError("manifest checksum is malformed")
    if _sha256(manifest_bytes) != checksum_text:
        raise TernaryIntegrityError("manifest checksum mismatch")
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TernaryIntegrityError("manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest, dict):
        raise TernaryIntegrityError("manifest root must be an object")
    if _canonical_json(manifest) != manifest_bytes:
        raise TernaryIntegrityError("manifest is not in canonical form")
    if (
        manifest.get("format") != FORMAT_NAME
        or manifest.get("formatVersion") not in (1, FORMAT_VERSION)
        or manifest.get("architecture") != "OmniCortex"
    ):
        raise TernaryIntegrityError("unsupported packed ternary format")
    encoding = manifest.get("encoding")
    expected_encoding = {
        "bitsPerValue": 2,
        "byteOrder": "four-values-lsb-first",
        "codes": {"-1": 0, "0": 1, "+1": 2},
        "reservedCode": _RESERVED_CODE,
        "paddingValue": 0,
    }
    if encoding != expected_encoding:
        raise TernaryIntegrityError("unsupported or ambiguous ternary encoding")
    claimed_content_hash = manifest.get("contentSha256")
    content = dict(manifest)
    content.pop("contentSha256", None)
    if (
        not isinstance(claimed_content_hash, str)
        or _sha256(_canonical_json(content)) != claimed_content_hash
    ):
        raise TernaryIntegrityError("manifest content checksum mismatch")

    tensor_entries = _manifest_entry_list(manifest.get("tensors"), "tensors")
    shard_entries = _manifest_entry_list(manifest.get("shards"), "shards")
    shard_table: Dict[str, Mapping[str, Any]] = {}
    for shard in shard_entries:
        filename = shard.get("file")
        if not isinstance(filename, str) or filename in shard_table:
            raise TernaryIntegrityError("manifest contains duplicate shard entries")
        shard_table[filename] = shard

    retained = None
    if retain_names is not None:
        if isinstance(retain_names, (str, bytes)):
            raise ValueError("retained tensor names must be an iterable of strings")
        requested = tuple(retain_names)
        if any(
            not isinstance(name, str) or not name or "\0" in name
            for name in requested
        ):
            raise ValueError("retained tensor names must be non-empty strings")
        retained = set(requested)
    names = []
    seen_names = set()
    tensors: Dict[str, torch.Tensor] = {}
    connection_states: Dict[str, torch.Tensor] = {}
    parallel_connections: Dict[str, Tuple[Tuple[int, ...], ...]] = {}
    used_shards = set()
    for entry in tensor_entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise TernaryIntegrityError("manifest contains duplicate tensor names")
        seen_names.add(name)
        if entry.get("kind") not in _ALLOWED_KINDS or entry.get("dtype") != "int8":
            raise TernaryIntegrityError("%s has unsupported tensor metadata" % name)
        shape_value = entry.get("shape")
        if not isinstance(shape_value, list):
            raise TernaryIntegrityError("%s has no valid tensor shape" % name)
        try:
            shape = _normalize_shape(shape_value)
        except TernaryPackingError as error:
            raise TernaryIntegrityError("%s has an invalid shape" % name) from error
        count = math.prod(shape)
        expected_connection_shape = connection_matrix_shape(shape, entry.get("kind"), entry.get("packedRowLayout"))
        if entry.get("numel") != count:
            raise TernaryIntegrityError("%s has inconsistent element count" % name)
        scale = entry.get("scale")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise TernaryIntegrityError("%s has an invalid scale" % name)
        filename = entry.get("shard")
        if filename not in shard_table or filename in used_shards:
            raise TernaryIntegrityError("%s has a missing or reused shard" % name)
        if entry.get("byteOffset") != 0:
            raise TernaryIntegrityError("%s has an unsupported shard offset" % name)
        path = _safe_shard_path(destination, filename)
        shard = shard_table[filename]
        expected_length = (count + 3) // 4
        if (
            entry.get("byteLength") != expected_length
            or shard.get("byteLength") != expected_length
            or path.stat().st_size != expected_length
        ):
            raise TernaryIntegrityError("%s shard length mismatch" % name)
        payload_hash, decoded_hash = verify_file(path, shape)
        if (
            entry.get("packedSha256") != payload_hash
            or shard.get("sha256") != payload_hash
        ):
            raise TernaryIntegrityError("%s shard checksum mismatch" % name)
        if entry.get("tensorSha256") != decoded_hash:
            raise TernaryIntegrityError("%s decoded tensor checksum mismatch" % name)
        if retained is None or name in retained:
            # Explicit legacy retention materializes a requested tensor.
            # Production validation uses retain_names=(), and compares exact
            # streamed hashes without a complete decoded sparse/core copy.
            tensors[name] = decode_ternary_2bit(path.read_bytes(), shape)
        names.append(name)
        used_shards.add(filename)
        connection = entry.get("connectionState")
        if connection is None and manifest.get("formatVersion") == 1:
            if entry.get("parallelConnections") is not None:
                raise TernaryIntegrityError("old native sign-only shards cannot contain parallel contacts")
            continue
        if connection == {"encoding": "unit-strength", "strength": 1}:
            if entry.get("parallelConnections") is not None:
                raise TernaryIntegrityError("parallel contacts require explicit connection state")
            continue
        if not isinstance(connection, dict) or connection.get("formatVersion") != CONNECTION_STATE_VERSION or connection.get("encoding") != "uint4-strength-minus-one-high4-resistance":
            raise TernaryIntegrityError("%s has an invalid connection-state encoding" % name)
        connection_shape = connection.get("shape")
        if (not isinstance(connection_shape, list) or len(connection_shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in connection_shape)
                or tuple(connection_shape) != expected_connection_shape):
            raise TernaryIntegrityError("%s connection-state shape mismatch" % name)
        connection_filename = connection.get("shard")
        if connection_filename not in shard_table or connection_filename in used_shards:
            raise TernaryIntegrityError("%s has a missing or reused connection shard" % name)
        connection_path = _safe_shard_path(destination, connection_filename)
        connection_shard = shard_table[connection_filename]
        if connection.get("byteLength") != count or connection_shard.get("byteLength") != count or connection_path.stat().st_size != count:
            raise TernaryIntegrityError("%s connection-state length mismatch" % name)
        digest = hashlib.sha256()
        retained_bytes = bytearray() if retained is None or name in retained else None
        before = connection_path.stat()
        total = 0
        with connection_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
                total += len(block)
                if retained_bytes is not None:
                    retained_bytes.extend(block)
        after = connection_path.stat()
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if total != count or identity(before) != identity(after):
            raise TernaryIntegrityError("%s connection-state shard changed during verification" % name)
        if connection.get("sha256") != digest.hexdigest() or connection_shard.get("sha256") != digest.hexdigest():
            raise TernaryIntegrityError("%s connection-state checksum mismatch" % name)
        used_shards.add(connection_filename)
        if retained_bytes is not None:
            connection_states[name] = (torch.frombuffer(retained_bytes, dtype=torch.uint8).reshape(connection_shape)
                if count else torch.empty(connection_shape, dtype=torch.uint8))
        parallel = entry.get("parallelConnections")
        if parallel is not None:
            if (not isinstance(parallel, dict) or parallel.get("encoding") != "u64-row-column-ordinal-u8-sign-state-le"
                    or isinstance(parallel.get("count"), bool) or not isinstance(parallel.get("count"), int) or parallel["count"] < 1):
                raise TernaryIntegrityError("parallel contact encoding/count is invalid")
            parallel_filename = parallel.get("shard")
            if parallel_filename not in shard_table or parallel_filename in used_shards:
                raise TernaryIntegrityError("parallel contact shard is missing or reused")
            parallel_path = _safe_shard_path(destination, parallel_filename)
            length = parallel["count"] * _PARALLEL_RECORD.size
            if parallel.get("byteLength") != length or shard_table[parallel_filename].get("byteLength") != length or parallel_path.stat().st_size != length:
                raise TernaryIntegrityError("parallel contact shard length mismatch")
            digest, records, previous = hashlib.sha256(), [], None
            before = parallel_path.stat()
            total = 0
            with parallel_path.open("rb") as handle:
                for payload in iter(lambda: handle.read(_PARALLEL_RECORD.size), b""):
                    if len(payload) != _PARALLEL_RECORD.size:
                        raise TernaryIntegrityError("parallel contact shard record is incomplete")
                    digest.update(payload)
                    total += len(payload)
                    record = _PARALLEL_RECORD.unpack(payload)
                    row, column, ordinal, sign, _state = record
                    if (row >= connection_shape[0] or column >= connection_shape[1] or sign >> 2 != 0x15
                            or sign & 3 == 3 or previous is not None and record[:3] <= previous):
                        raise TernaryIntegrityError("parallel contact endpoint/identity/code is invalid")
                    previous = record[:3]
                    if retained is None or name in retained:
                        records.append(record)
            if total != length or identity(before) != identity(parallel_path.stat()):
                raise TernaryIntegrityError("parallel contact shard changed during verification")
            if parallel.get("sha256") != digest.hexdigest() or shard_table[parallel_filename].get("sha256") != digest.hexdigest():
                raise TernaryIntegrityError("parallel contact checksum mismatch")
            used_shards.add(parallel_filename)
            if retained is None or name in retained:
                parallel_connections[name] = tuple(records)

    if used_shards != set(shard_table):
        raise TernaryIntegrityError("manifest contains unreferenced ternary shards")
    if names != sorted(names):
        raise TernaryIntegrityError("tensor manifest is not deterministically ordered")
    coverage = manifest.get("coverage")
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        raise TernaryIntegrityError("manifest does not assert complete coverage")
    covered_names = coverage.get("eligibleTensorNames")
    if covered_names != names or coverage.get("eligibleTensorCount") != len(names):
        raise TernaryIntegrityError("manifest coverage does not match its tensors")
    try:
        _assert_coverage(names, expected_names)
    except TernaryCoverageError as error:
        raise TernaryIntegrityError(str(error)) from error
    return VerifiedTernaryBundle(manifest=manifest, tensors=tensors, connection_states=connection_states,
                                parallel_connections=parallel_connections)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "TernaryCoverageError",
    "TernaryIntegrityError",
    "TernaryPackingError",
    "TernaryTensorSpec",
    "VerifiedTernaryBundle",
    "collect_module_ternary_tensors",
    "decode_ternary_2bit",
    "encode_ternary_2bit",
    "export_module_ternary_shards",
    "export_ternary_shards",
    "verify_ternary_shards",
]
