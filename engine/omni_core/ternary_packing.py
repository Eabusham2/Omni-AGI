"""Safe, deterministic packed-ternary inference shards.

The stable OmniCortex checkpoint keeps floating master weights for learning,
but deployment should not need to infer which values were used by a ternary
forward pass.  This module writes the *effective* ``{-1, 0, +1}`` tensors in a
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

The exporter discovers every built-in BitLinear/BitConv and STDP module.  A
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .model import TERNARY_PROJECTION_TYPES
from .spiking import STDPSynapses


FORMAT_NAME = "omni-packed-ternary"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MANIFEST_CHECKSUM_NAME = "manifest.sha256"

_CODE_TO_VALUE = (-1, 0, 1)
_ZERO_CODE = 1
_RESERVED_CODE = 3
_ALLOWED_KINDS = frozenset({"projection", "dynamic-synapse"})


class TernaryPackingError(ValueError):
    """Base error for an invalid packed-ternary operation."""


class TernaryCoverageError(TernaryPackingError):
    """The exported tensors do not match the architecture contract."""


class TernaryIntegrityError(TernaryPackingError):
    """A stored manifest, shard, or decoded tensor failed verification."""


@dataclass(frozen=True)
class TernaryTensorSpec:
    """One exact-ternary tensor and its post-projection floating scale."""

    name: str
    values: torch.Tensor
    scale: float = 1.0
    kind: str = "projection"
    source_dtype: Optional[str] = None


@dataclass(frozen=True)
class VerifiedTernaryBundle:
    """Verified manifest and decoded CPU int8 inference tensors."""

    manifest: Dict[str, Any]
    tensors: Dict[str, torch.Tensor]


DynamicSynapse = Union[TernaryTensorSpec, torch.Tensor, nn.Module]
ModuleRoots = Union[nn.Module, Mapping[str, nn.Module]]


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
        source = getattr(module, "weights")
        return TernaryTensorSpec(
            name=_qualified_name(root_name, module_name, "weights"),
            values=effective_weight(),
            scale=1.0,
            kind="dynamic-synapse",
            source_dtype=str(source.dtype).replace("torch.", ""),
        )

    source = getattr(module, "weight")
    scale = float(source.detach().abs().mean().clamp_min(1e-6).item())
    return TernaryTensorSpec(
        name=_qualified_name(root_name, module_name, "weight"),
        values=effective_weight(),
        scale=scale,
        kind="projection",
        source_dtype=str(source.dtype).replace("torch.", ""),
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
            spec = _module_spec(root_name, module_name, module)
            if spec.name in collected:
                raise TernaryCoverageError(
                    "duplicate eligible tensor name: %s" % spec.name
                )
            collected[spec.name] = spec

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
        exact = _exact_ternary(spec.values, name=name)
        packed = encode_ternary_2bit(exact)
        packed_hash = _sha256(packed)
        filename = "ternary-%05d-%s.bin" % (index, packed_hash[:16])
        _atomic_write(destination / filename, packed)
        entry = {
            "name": name,
            "kind": spec.kind,
            "shape": [int(value) for value in exact.shape],
            "dtype": "int8",
            "sourceDtype": spec.source_dtype
            or str(spec.values.dtype).replace("torch.", ""),
            "scale": float(spec.scale),
            "numel": int(exact.numel()),
            "shard": filename,
            "byteOffset": 0,
            "byteLength": len(packed),
            "packedSha256": packed_hash,
            "tensorSha256": _tensor_sha256(exact),
        }
        entries.append(entry)
        shard_entries.append(
            {
                "file": filename,
                "byteLength": len(packed),
                "sha256": packed_hash,
            }
        )

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
        destination, expected_names=contract
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
) -> VerifiedTernaryBundle:
    """Verify and decode a packed generation without executing stored code."""

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
        or manifest.get("formatVersion") != FORMAT_VERSION
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

    names = []
    tensors: Dict[str, torch.Tensor] = {}
    used_shards = set()
    for entry in tensor_entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in tensors:
            raise TernaryIntegrityError("manifest contains duplicate tensor names")
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
        payload = path.read_bytes()
        shard = shard_table[filename]
        payload_hash = _sha256(payload)
        expected_length = (count + 3) // 4
        if (
            entry.get("byteLength") != expected_length
            or shard.get("byteLength") != expected_length
            or len(payload) != expected_length
        ):
            raise TernaryIntegrityError("%s shard length mismatch" % name)
        if (
            entry.get("packedSha256") != payload_hash
            or shard.get("sha256") != payload_hash
        ):
            raise TernaryIntegrityError("%s shard checksum mismatch" % name)
        decoded = decode_ternary_2bit(payload, shape)
        if entry.get("tensorSha256") != _tensor_sha256(decoded):
            raise TernaryIntegrityError("%s decoded tensor checksum mismatch" % name)
        tensors[name] = decoded
        names.append(name)
        used_shards.add(filename)

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
    return VerifiedTernaryBundle(manifest=manifest, tensors=tensors)


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
