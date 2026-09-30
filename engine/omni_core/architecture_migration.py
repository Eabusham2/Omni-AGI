"""Bounded, state-preserving compatible native architecture operations.

No model constructor, averaging, floating weight master, or quality claim.
Width/head geometry is deliberately not migrated by these operations.
"""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from .bounded_tensor_io import BoundedTensorFile, TRANSFER_BYTES
from .native_architecture import native_architecture_sha256, native_core_inventory, validate_native_architecture


_OPERATIONS = {
    "grow-experts": ("addExperts",),
    "grow-depth": ("addLayers",),
    "grow-router": ("addNeurons",),
    "grow-regions": ("addRegions", "neuronsPerRegion"),
}


def normalize_architecture_change(value: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if not value:
        return None
    mutation = value.get("mutation")
    if mutation not in _OPERATIONS:
        raise ValueError("unsupported architecture mutation; width/head geometry is not blindly padded")
    fields = _OPERATIONS[mutation]
    if set(value) - {"mutation", *fields}:
        raise ValueError("unsupported architecture mutation fields")
    result: dict[str, Any] = {"mutation": mutation}
    for field in fields:
        amount = value.get(field, 1 if field != "neuronsPerRegion" else None)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1 or amount > (1 << 53) - 1:
            raise ValueError("architecture %s must be a positive safe integer" % field)
        result[field] = amount
    result["compatibilityBoundary"] = (
        "Existing learned trits, controls and activity preserved at insertion; "
        "new residual outputs start at zero; width, head geometry, normalization "
        "and RoPE remain unchanged. Future isolated learning is evaluated separately."
    )
    return result


@torch.no_grad()
def copy_tensor_prefix(source: torch.Tensor, target: torch.Tensor, *, chunk_bytes: int = TRANSFER_BYTES) -> int:
    """Copy exact one/two-dimensional prefixes without a whole-device clone."""
    if source.dtype != target.dtype or source.ndim != target.ndim or source.ndim > 2 or any(a > b for a, b in zip(source.shape, target.shape)):
        raise ValueError("unsupported exact prefix migration shape/dtype")
    if source.ndim == 0:
        target.copy_(source.to(target.device))
        return source.element_size()
    item = source.element_size()
    elements = max(1, int(chunk_bytes) // item)
    peak = 0
    if source.ndim == 1:
        for start in range(0, source.numel(), elements):
            end = min(source.numel(), start + elements)
            target[start:end].copy_(source[start:end].to(target.device))
            peak = max(peak, (end - start) * item)
    else:
        columns = int(source.shape[1])
        if columns <= elements:
            rows = max(1, elements // max(1, columns))
            for start in range(0, source.shape[0], rows):
                end = min(source.shape[0], start + rows)
                target[start:end, :columns].copy_(source[start:end].to(target.device))
                peak = max(peak, (end - start) * columns * item)
        else:
            for row in range(source.shape[0]):
                for start in range(0, columns, elements):
                    end = min(columns, start + elements)
                    target[row, start:end].copy_(source[row, start:end].to(target.device))
                    peak = max(peak, (end - start) * item)
    return peak


@torch.no_grad()
def copy_packed_prefix(source: torch.Tensor, target: torch.Tensor, old_width: int, new_width: int, *, chunk_bytes: int = TRANSFER_BYTES) -> int:
    """Preserve canonical two-bit old trits while adding exact ternary zeros."""
    if source.dtype != torch.uint8 or target.dtype != torch.uint8 or source.ndim != 2 or target.ndim != 2 or old_width > new_width or source.shape[1] != (old_width + 3) // 4 or target.shape[1] != (new_width + 3) // 4:
        raise ValueError("packed prefix migration layout is invalid")
    if source.shape[0] > target.shape[0]:
        raise ValueError("packed prefix migration cannot remove learned rows")
    source_flat = source.reshape(-1)
    validation_step = max(1, int(chunk_bytes) // 8)
    for start in range(0, source_flat.numel(), validation_step):
        block = source_flat[start:start + validation_step]
        if any(bool((((block >> shift) & 3) == 3).any()) for shift in (0, 2, 4, 6)):
            raise ValueError("old packed trits contain reserved codes")
    if old_width % 4:
        for start in range(0, source.shape[0], validation_step):
            final_bytes = source[start:start + validation_step, -1]
            if any(bool((((final_bytes >> (2 * lane)) & 3) != 1).any()) for lane in range(old_width % 4, 4)):
                raise ValueError("old packed trits contain nonzero padding codes")
    # Fill and copy are themselves bounded. No dense FP representation exists.
    flat = target.reshape(-1)
    for start in range(0, flat.numel(), max(1, int(chunk_bytes))):
        flat[start:start + int(chunk_bytes)].fill_(0x55)
    return copy_tensor_prefix(source, target, chunk_bytes=chunk_bytes)


def reseal_native_descriptor(config: Any, mutation: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    descriptor = getattr(config, "native_architecture", None)
    if descriptor is None:
        return None
    previous = validate_native_architecture(descriptor)
    validate_compatible_architecture_lineage(previous)
    result = copy.deepcopy(previous)
    shape = result["shape"]
    shape["layers"] = int(config.n_layers)
    shape["routerNeurons"] = int(config.router_neurons)
    result["inventory"] = native_core_inventory(shape)
    old_lineage = previous.get("evolutionLineage", {})
    history = list(old_lineage.get("mutations", ()))
    normalized = normalize_architecture_change({key: value for key, value in mutation.items() if key != "compatibilityBoundary"})
    history.append({key: value for key, value in normalized.items() if key != "compatibilityBoundary"})
    result["evolutionLineage"] = {
        "format": "omni-compatible-architecture-lineage", "formatVersion": 1,
        "rootArchitectureSha256": old_lineage.get("rootArchitectureSha256", previous["sha256"]),
        "parentArchitectureSha256": previous["sha256"],
        "rootShape": copy.deepcopy(old_lineage.get("rootShape", previous["shape"])),
        "mutations": history,
        "normalizationAndHeadGeometryChanged": False,
        "qualityVerified": False,
    }
    result["sha256"] = native_architecture_sha256(result)
    return validate_native_architecture(result)


def validate_compatible_architecture_lineage(descriptor: Mapping[str, Any], origin_descriptor: Optional[Mapping[str, Any]] = None) -> None:
    """Bind every shape step to the unchanged Build descriptor, not fake training.

    An immutable origin descriptor, when available to the central loader, is
    checked against the reconstructed root. No origin receipt is rewritten.
    """
    current = validate_native_architecture(descriptor)
    lineage = current.get("evolutionLineage")
    if lineage is None:
        if origin_descriptor is not None and validate_native_architecture(origin_descriptor)["sha256"] != current["sha256"]:
            raise ValueError("native architecture changed without compatible lineage")
        return
    fields = {"format", "formatVersion", "rootArchitectureSha256", "parentArchitectureSha256", "rootShape", "mutations", "normalizationAndHeadGeometryChanged", "qualityVerified"}
    if (not isinstance(lineage, Mapping) or set(lineage) != fields
        or lineage["format"] != "omni-compatible-architecture-lineage" or lineage["formatVersion"] != 1
        or lineage["normalizationAndHeadGeometryChanged"] is not False or lineage["qualityVerified"] is not False
        or not isinstance(lineage["mutations"], list) or not lineage["mutations"]):
        raise ValueError("native compatible architecture lineage is invalid")
    root = copy.deepcopy(current)
    root.pop("evolutionLineage")
    root["shape"] = copy.deepcopy(lineage["rootShape"])
    if (not isinstance(root["shape"], Mapping) or set(root["shape"]) != set(current["shape"])
        or any(type(value) is not int or not 1 <= value <= (1 << 53) - 1
            for key, value in root["shape"].items() if key != "liquidMode")):
        raise ValueError("native compatible architecture root shape is invalid")
    root["inventory"] = native_core_inventory(root["shape"])
    root["sha256"] = native_architecture_sha256(root)
    root = validate_native_architecture(root)
    if root["sha256"] != lineage["rootArchitectureSha256"]:
        raise ValueError("native compatible architecture root hash is invalid")
    if origin_descriptor is not None and validate_native_architecture(origin_descriptor)["sha256"] != root["sha256"]:
        raise ValueError("native compatible architecture lineage does not bind immutable origin")
    previous, history = root, []
    for payload in lineage["mutations"]:
        if not isinstance(payload, Mapping):
            raise ValueError("native compatible architecture mutation is invalid")
        mutation = normalize_architecture_change(payload)
        canonical = {key: value for key, value in mutation.items() if key != "compatibilityBoundary"}
        if dict(payload) != canonical:
            raise ValueError("native compatible architecture mutation is not canonical")
        history.append(canonical)
        step = copy.deepcopy(root)
        step["shape"] = copy.deepcopy(previous["shape"])
        if canonical["mutation"] == "grow-depth":
            step["shape"]["layers"] += canonical["addLayers"]
        elif canonical["mutation"] == "grow-router":
            step["shape"]["routerNeurons"] += canonical["addNeurons"]
        elif canonical["mutation"] == "grow-regions":
            step["shape"]["routerNeurons"] += canonical["addRegions"] * canonical["neuronsPerRegion"]
        step["inventory"] = native_core_inventory(step["shape"])
        step["evolutionLineage"] = {
            "format": lineage["format"], "formatVersion": 1,
            "rootArchitectureSha256": root["sha256"], "parentArchitectureSha256": previous["sha256"],
            "rootShape": copy.deepcopy(root["shape"]), "mutations": list(history),
            "normalizationAndHeadGeometryChanged": False, "qualityVerified": False,
        }
        step["sha256"] = native_architecture_sha256(step)
        previous = validate_native_architecture(step)
    if previous["sha256"] != current["sha256"]:
        raise ValueError("native compatible architecture shape/parent chain is invalid")


def assert_architecture_quiescent(brain: Any) -> None:
    for owner in (brain, getattr(brain, "trainer", None), getattr(brain, "_distributed_trainer", None)):
        controller = getattr(owner, "_packed_collective_controller", None)
        if controller is not None and getattr(controller, "step_id", None) is not None:
            raise RuntimeError("architecture mutation requires a quiescent packed collective checkpoint")
        if owner is not None and any(getattr(owner, key, None) is not None for key in ("_active_distributed_record_window", "_distributed_record_window_cursor")):
            raise RuntimeError("architecture mutation cannot cross an active distributed record window")
    checkpoints = getattr(brain, "ingestion_checkpoints", {})
    for checkpoint in checkpoints.values() if isinstance(checkpoints, Mapping) else ():
        if isinstance(checkpoint, Mapping) and any(checkpoint.get(key) for key in ("activeRecordWindowSha256", "activeRecordWindowCursor", "windowOffset")):
            raise RuntimeError("architecture mutation cannot cross an active source-record window cursor")
    if checkpoints:
        # A paused between-record cursor is still bound to the exact packed
        # parameter checksum. Changing shape would invalidate its resumable
        # joint generation. Do not silently drop or rewrite that evidence.
        raise RuntimeError("architecture mutation requires completing the outstanding source-ingestion checkpoint")
    seal = getattr(brain, "distributed_training_seal", None)
    if seal is not None:
        from .distributed_seal import validate_distributed_training_seal
        seal = validate_distributed_training_seal(seal)
        if any("recordWindow" in cursor or cursor["epoch"] < seal["epochsRequested"] for cursor in seal["rankCursors"]):
            raise RuntimeError("architecture mutation cannot invalidate unfinished distributed native cursor topology")


def file_tensor_inventory(path: Path) -> dict[str, dict[str, Any]]:
    reader = BoundedTensorFile(path)
    return {name: {"shape": list(spec.shape), "dtype": str(spec.dtype), "bytes": spec.byte_count} for name, spec in reader.specs.items()}


def isolated_checkpoint_resident_bytes(engine_path: Path) -> int:
    """Conservative plain control-state load lifetime, excluding cold uint8.

    Native packed owners are independently admitted by the core pager. Loading
    an isolated candidate still duplicates nonweight router/activity controls;
    two lifetimes plus fixed transfer scratch are charged before construction.
    """
    nonweight = sum(
        spec.byte_count for filename in ("core.safetensors", "plasticity.safetensors")
        for spec in BoundedTensorFile(engine_path / filename).specs.values()
        if spec.dtype != torch.uint8
    )
    return 2 * nonweight + 2 * TRANSFER_BYTES


def tensor_file_checksums(path: Path) -> dict[str, str]:
    reader = BoundedTensorFile(path)
    result = {}
    for name in reader.specs:
        digest = hashlib.sha256()
        for chunk in reader.chunks(name):
            digest.update(chunk.reshape(-1).view(torch.uint8).numpy().tobytes())
        result[name] = digest.hexdigest()
    return result


def growth_dimensions(config: Any, mutation: Mapping[str, Any]) -> tuple[int, int]:
    layers, neurons = int(config.n_layers), int(config.router_neurons)
    if mutation["mutation"] == "grow-depth":
        layers += int(mutation["addLayers"])
    elif mutation["mutation"] == "grow-router":
        neurons += int(mutation["addNeurons"])
    elif mutation["mutation"] == "grow-regions":
        neurons += int(mutation["addRegions"]) * int(mutation["neuronsPerRegion"])
    return layers, neurons


def verify_preserved_tensor_prefixes(before: Path, after: Path, *, chunk_bytes: int = TRANSFER_BYTES) -> dict[str, Any]:
    """Byte-proof existing learned/control prefixes before isolated training."""
    old, new = BoundedTensorFile(before, chunk_bytes=chunk_bytes), BoundedTensorFile(after, chunk_bytes=chunk_bytes)
    checked, peak = 0, 0
    digest = hashlib.sha256()
    with before.open("rb") as left, after.open("rb") as right:
        for name, spec in old.specs.items():
            target = new.specs.get(name)
            if target is None or target.dtype != spec.dtype or len(target.shape) != len(spec.shape) or any(a > b for a, b in zip(spec.shape, target.shape)):
                raise ValueError("architecture insertion removed or changed an old tensor: " + name)
            digest.update(name.encode())
            if spec.shape == target.shape or len(spec.shape) < 2:
                rows, row_bytes, target_row_bytes = 1, spec.byte_count, target.byte_count
            elif len(spec.shape) == 2:
                item = torch.empty((), dtype=spec.dtype).element_size()
                rows, row_bytes, target_row_bytes = spec.shape[0], spec.shape[1] * item, target.shape[1] * item
            else:
                raise ValueError("expanded higher-rank tensors require an explicit geometry migration")
            for row in range(rows):
                for offset in range(0, row_bytes, max(1, int(chunk_bytes))):
                    count = min(int(chunk_bytes), row_bytes - offset)
                    left.seek(spec.offset + row * row_bytes + offset)
                    right.seek(target.offset + row * target_row_bytes + offset)
                    a, b = left.read(count), right.read(count)
                    if len(a) != count or a != b:
                        raise ValueError("architecture insertion changed learned/control bytes: " + name)
                    digest.update(a)
                    peak = max(peak, count)
            checked += 1
    return {"verified": True, "oldTensorsChecked": checked, "peakTransferBytes": peak, "preservedPrefixSha256": digest.hexdigest(), "verifiedBeforeTraining": True}


@contextmanager
def preserve_runtime_rng(device: torch.device):
    cpu = torch.get_rng_state()
    accelerator = None
    if device.type == "cuda":
        accelerator = torch.cuda.get_rng_state(device)
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        accelerator = torch.mps.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if device.type == "cuda" and accelerator is not None:
            torch.cuda.set_rng_state(accelerator, device)
        elif device.type == "mps" and accelerator is not None:
            torch.mps.set_rng_state(accelerator)
