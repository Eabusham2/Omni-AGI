"""Bounded, state-preserving compatible native architecture operations.

No model constructor, averaging, floating weight master, or quality claim.
Width/head geometry is an explicit isolated candidate operation; copied
values are exact but changed geometry is never called function-preserving.
"""

from __future__ import annotations

import copy
import hashlib
import math
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
    "resize-width": ("dModel", "feedForward", "nHeads"),
    "repartition-heads": ("nHeads",),
}


def normalize_architecture_change(value: Optional[Mapping[str, Any]]) -> Optional[dict[str, Any]]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("architecture change must be a typed mapping")
    if not value:
        return None
    mutation = value.get("mutation")
    if mutation not in _OPERATIONS:
        raise ValueError("unsupported typed architecture mutation")
    fields = _OPERATIONS[mutation]
    if set(value) - {"mutation", *fields}:
        raise ValueError("unsupported architecture mutation fields")
    result: dict[str, Any] = {"mutation": mutation}
    for field in fields:
        if mutation == "resize-width" and field in {"feedForward", "nHeads"} and field not in value:
            continue
        amount = value.get(field, 1 if field != "neuronsPerRegion" else None)
        if mutation in {"resize-width", "repartition-heads"} and field not in value:
            amount = None
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1 or amount > (1 << 53) - 1:
            raise ValueError("architecture %s must be a positive safe integer" % field)
        result[field] = amount
    result["compatibilityBoundary"] = ("Isolated geometry candidate only: copied logical trits/controls are exact; "
        "new coordinates are explicitly initialized; normalization, head partitioning or RoPE may change the function. "
        "Training and evaluation must pass before any live activation. No improvement is asserted."
        if mutation in {"resize-width", "repartition-heads"} else
        "Existing learned trits, controls and activity preserved at insertion; new residual outputs start at zero; "
        "width, head geometry, normalization and RoPE remain unchanged. Future isolated learning is evaluated separately.")
    return result


def architecture_mutation_policy(mutation):
    if not isinstance(mutation, Mapping):
        raise ValueError("architecture policy requires a typed operation")
    change = normalize_architecture_change({key: value for key, value in mutation.items() if key != "compatibilityBoundary"})
    if change is None:
        raise ValueError("architecture policy requires an explicit operation")
    geometry = change["mutation"] in {"resize-width", "repartition-heads"}
    return {"candidateOnly": True, "geometryChanges": geometry,
        "functionPreservingAtInsertion": not geometry, "requiresTrainingEvaluation": True,
        "improvementVerified": False, "externalBackbone": False, "floatingProjectionMaster": False}


def geometry_candidate_config(parent, mutation):
    """Pure config proposal; never mutate the live parent or run a model."""
    if not isinstance(parent, Mapping) or not isinstance(mutation, Mapping):
        raise ValueError("geometry candidate requires typed parent config and mutation")
    change = normalize_architecture_change({key: value for key, value in mutation.items() if key != "compatibilityBoundary"})
    if change is None or change["mutation"] not in {"resize-width", "repartition-heads"}:
        raise ValueError("geometry config requires an explicit width/head candidate operation")
    for key in ("d_model", "d_ff", "n_heads", "idea_dim"):
        if type(parent.get(key)) is not int or not 1 <= parent[key] <= (1 << 53) - 1:
            raise ValueError("parent geometry needs positive safe integer " + key)
    if parent["d_model"] % parent["n_heads"] or (parent["d_model"] // parent["n_heads"]) % 2:
        raise ValueError("parent geometry has invalid width/head dimensions")
    result = copy.deepcopy(dict(parent))
    old_width = int(parent["d_model"])
    if change["mutation"] == "resize-width":
        result["d_model"] = change["dModel"]
        result["d_ff"] = change.get("feedForward", parent["d_ff"])
        # Native descriptors inventory one shared d-wide idea space. Legacy
        # independent idea dimensions stay explicit rather than guessed.
        if parent.get("native_architecture") is not None or parent.get("idea_dim") == old_width:
            result["idea_dim"] = result["d_model"]
    result["n_heads"] = change.get("nHeads", parent["n_heads"])
    d, heads = int(result["d_model"]), int(result["n_heads"])
    if d % heads or (d // heads) % 2 or int(result["d_ff"]) < 1:
        raise ValueError("geometry candidate needs divisible width and an even RoPE head dimension")
    if all(result.get(key) == parent.get(key) for key in ("d_model", "d_ff", "n_heads", "idea_dim")):
        raise ValueError("geometry candidate is a no-op")
    return result


def geometry_axis_segments(name, old_shape, new_shape, old_geometry, new_geometry, *, role="weight"):
    """Semantic logical axes, including segmented gates/concatenations.

    Shapes are logical, not ceil(width/4) packed byte shapes. Unknown changed
    tensors require a caller declaration rather than a guessed reshape.
    """
    if (not isinstance(name, str) or not name or len(old_shape) != 2 or len(new_shape) != 2
        or any(type(size) is not int or size < 1 for size in (*old_shape, *new_shape))):
        raise ValueError("geometry axis migration needs logical matrices")
    def prefix(old, new): return [[0, 0, min(old, new)]]
    def groups(old, new, count):
        return [[index * old, index * new, min(old, new)] for index in range(count)]
    rows, cols = prefix(old_shape[0], new_shape[0]), prefix(old_shape[1], new_shape[1])
    old_d, new_d = old_geometry["d_model"], new_geometry["d_model"]
    if role == "bias":
        if old_shape[0] != 1 or new_shape[0] != 1:
            raise ValueError("packed bias is one logical vector row")
        if name.endswith(".attention.qkv"):
            if old_shape[1] != 3 * old_d or new_shape[1] != 3 * new_d:
                raise ValueError("Q/K/V bias coordinates do not match the declared width")
            cols = groups(old_d, new_d, 3)
        elif name.endswith(".up") and (".feed_forward." in name or ".network." in name):
            if old_shape[1] % 2 or new_shape[1] % 2:
                raise ValueError("packed SwiGLU bias has incomplete gate/value geometry")
            cols = groups(old_shape[1] // 2, new_shape[1] // 2, 2)
    elif role != "weight":
        raise ValueError("unknown packed geometry role")
    elif name.endswith(".attention.qkv"):
        if list(old_shape) != [3 * old_d, old_d] or list(new_shape) != [3 * new_d, new_d]:
            raise ValueError("Q/K/V coordinates do not match the declared width")
        rows = groups(old_d, new_d, 3)
    elif name.endswith(".up") and (".feed_forward." in name or ".network." in name):
        before = old_shape[0] // 2
        after = new_shape[0] // 2
        if old_shape[0] % 2 or new_shape[0] % 2:
            raise ValueError("SwiGLU gate/value migration needs complete paired rows")
        rows = groups(before, after, 2)
    elif name.endswith(".action_argument_head.condition"):
        fixed = old_shape[1] - old_d
        if fixed != new_shape[1] - new_d or fixed < 1:
            raise ValueError("argument schema feature axis must remain unchanged")
        cols = [[0, 0, min(old_d, new_d)], [old_d, new_d, fixed]]
    elif name.endswith(".action_argument_head.transition"):
        if old_shape[1] != 2 * old_d or new_shape[1] != 2 * new_d:
            raise ValueError("argument recurrent state/token axes are invalid")
        cols = groups(old_d, new_d, 2)
    elif name.startswith("liquid.cell.") and old_shape[1] == 2 * old_geometry["idea_dim"]:
        if new_shape[1] != 2 * new_geometry["idea_dim"]:
            raise ValueError("liquid input/state axes are invalid")
        cols = groups(old_geometry["idea_dim"], new_geometry["idea_dim"], 2)
    return {"rowSegments": rows, "columnSegments": cols,
        "oldLogicalShape": list(old_shape), "newLogicalShape": list(new_shape),
        "functionPreserved": False, "mappingPolicy": "exact-semantic-coordinate-overlap-v1"}


def _validate_axis_segments(segments, old_size, new_size):
    if not isinstance(segments, (tuple, list)) or not segments:
        raise ValueError("geometry copy requires declared axis segments")
    clean, used_source, used_target = [], [], []
    for segment in segments:
        if not isinstance(segment, (tuple, list)) or len(segment) != 3 or any(type(value) is not int or value < 0 for value in segment):
            raise ValueError("geometry axis segment is invalid")
        old, new, count = segment
        if count < 1 or old + count > old_size or new + count > new_size:
            raise ValueError("geometry axis segment exceeds its declared shape")
        if any(old < end and old + count > start for start, end in used_source) or any(new < end and new + count > start for start, end in used_target):
            raise ValueError("geometry axis segments alias or repeat learned coordinates")
        used_source.append((old, old + count)); used_target.append((new, new + count))
        clean.append([old, new, count])
    return clean


def _coordinate_bytes(seed, name, start, count):
    """Counter-hashed initialization invariant to caller transfer chunking."""
    key = hashlib.sha256((str(seed) + "\0" + name).encode("utf-8")).digest()
    result = bytearray()
    for block in range(start // 32, (start + count + 31) // 32):
        value = hashlib.sha256(key + block.to_bytes(8, "little")).digest()
        left, right = max(start, block * 32) - block * 32, min(start + count, (block + 1) * 32) - block * 32
        result.extend(value[left:right])
    return result


def _reject_shared_storage(source, target):
    if (isinstance(source, torch.Tensor) and source.device == target.device
        and source.numel() and target.numel()
        and source.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()):
        raise ValueError("isolated geometry source and candidate alias the same storage")


def _release_geometry_chunk(value, start, count):
    if isinstance(value, torch.Tensor) and value.device.type == "cpu":
        from .native_core_paging import release_native_tensor_chunk
        release_native_tensor_chunk(value, start + value.storage_offset() * value.element_size(), count)


@torch.no_grad()
def copy_packed_geometry(source, target, old_width, new_width, *, row_segments=None, column_segments=None,
    seed, tensor_name, source_name=None, chunk_bytes=TRANSFER_BYTES, cancelled=None):
    """Copy exact logical trits into an isolated geometry candidate.

    ``source`` may be an immutable BoundedTensorFile (with source_name), so
    even one huge source tensor need not be loaded as a whole. Only int trit
    tiles exist; no FP projection master or averaging is used.
    """
    _reject_shared_storage(source, target)
    if type(old_width) is not int or type(new_width) is not int:
        raise ValueError("geometry logical widths must be positive integers")
    chunk_bytes = max(32, int(chunk_bytes))
    if isinstance(source, BoundedTensorFile):
        spec = source.specs[source_name]
        shape, dtype = spec.shape, spec.dtype
        def read(row, column, count):
            with source.path.open("rb") as handle:
                handle.seek(spec.offset + row * shape[1] + column)
                raw = handle.read(count)
            if len(raw) != count: raise ValueError("geometry source tensor ended before its declared range")
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    else:
        shape, dtype = tuple(source.shape), source.dtype
        def read(row, column, count): return source[row, column:column + count].detach().cpu()
    if (dtype != torch.uint8 or target.dtype != torch.uint8 or len(shape) != 2 or target.ndim != 2
        or isinstance(source, torch.Tensor) and not source.is_contiguous()
        or not target.is_contiguous() or old_width < 1 or new_width < 1
        or shape[1] != (old_width + 3) // 4 or target.shape[1] != (new_width + 3) // 4):
        raise ValueError("geometry packed layout is invalid")
    rows = _validate_axis_segments(row_segments or [[0, 0, min(shape[0], target.shape[0])]], shape[0], target.shape[0])
    columns = _validate_axis_segments(column_segments or [[0, 0, min(old_width, new_width)]], old_width, new_width)
    # Validate all source codes, including coordinates dropped by an explicit
    # narrowing candidate. Parent corruption must not be silently hidden.
    for row in range(shape[0]):
        for start in range(0, shape[1], max(1, chunk_bytes // 8)):
            if cancelled is not None and cancelled(): raise InterruptedError("geometry parent validation cancelled")
            block = read(row, start, min(chunk_bytes // 8, shape[1] - start))
            if any(bool((((block >> shift) & 3) == 3).any()) for shift in (0, 2, 4, 6)):
                raise ValueError("geometry source contains reserved packed codes")
            _release_geometry_chunk(source, row * shape[1] + start, block.numel())
        if old_width % 4:
            last = read(row, shape[1] - 1, 1)
            if any(int((last[0] >> (2 * lane)) & 3) != 1 for lane in range(old_width % 4, 4)):
                raise ValueError("geometry source has noncanonical padding")
            _release_geometry_chunk(source, row * shape[1] + shape[1] - 1, 1)
    row_bytes = int(target.shape[1])
    initialization_step = max(1, chunk_bytes // 16)
    for row in range(target.shape[0]):
        for column in range(0, row_bytes, initialization_step):
            if cancelled is not None and cancelled(): raise InterruptedError("geometry candidate initialization cancelled")
            count = min(initialization_step, row_bytes - column)
            start = row * row_bytes + column
            random = torch.frombuffer(_coordinate_bytes(seed, tensor_name, start, count), dtype=torch.uint8).to(target.device)
            packed = torch.zeros_like(random)
            for lane in range(4): packed |= ((random >> (lane * 2)) % 3) << (lane * 2)
            if new_width % 4 and column + count == row_bytes:
                mask = (1 << (2 * (new_width % 4))) - 1
                packed[-1] = (packed[-1] & mask) | (0x55 & (0xFF ^ mask))
            target[row, column:column + count].copy_(packed)
            _release_geometry_chunk(target, start, count)
    digest, copied = hashlib.sha256(), 0
    tile = max(1, chunk_bytes // 64)
    for old_row, new_row, row_count in rows:
        for index in range(row_count):
            for old_column, new_column, width in columns:
                for start in range(0, width, tile):
                    if cancelled is not None and cancelled(): raise InterruptedError("geometry candidate copy cancelled")
                    count = min(tile, width - start)
                    a, b = old_column + start, new_column + start
                    raw = read(old_row + index, a // 4, (a % 4 + count + 3) // 4).to(target.device)
                    offsets = torch.arange(count, dtype=torch.long, device=target.device)
                    codes = ((raw[(offsets + a % 4) // 4] >> (2 * ((offsets + a % 4) % 4))) & 3).to(torch.uint8)
                    for lane in range(4):
                        positions = offsets[(offsets + b) % 4 == lane]
                        if not positions.numel(): continue
                        indexes = (positions + b) // 4
                        old_bytes = target[new_row + index, indexes]
                        replacement = (old_bytes & (0xFF ^ (3 << (2 * lane)))) | (codes[positions] << (2 * lane))
                        target[new_row + index, indexes] = replacement.to(torch.uint8)
                    actual = (target[new_row + index, (offsets + b) // 4] >> (2 * ((offsets + b) % 4))) & 3
                    if not torch.equal(actual, codes):
                        raise ValueError("geometry candidate failed exact copied-trit verification")
                    # Digest verifies the semantic copied coordinates, not
                    # byte padding/reshape accident or a float approximation.
                    digest.update(codes.detach().cpu().numpy().tobytes())
                    copied += count
                    _release_geometry_chunk(source, (old_row + index) * shape[1] + a // 4, raw.numel())
                    _release_geometry_chunk(target, (new_row + index) * row_bytes + b // 4, (b % 4 + count + 3) // 4)
    return {"copiedLogicalTrits": copied, "initializedLogicalTrits": int(target.shape[0]) * new_width - copied,
        "unmappedParentLogicalTrits": int(shape[0]) * old_width - copied,
        "copiedCoordinatesSha256": digest.hexdigest(), "peakTransferBytes": chunk_bytes,
        "initialization": "coordinate-sha256-ternary-v1", "functionPreserved": False,
        "rowSegments": rows, "columnSegments": columns,
        "oldLogicalShape": [int(shape[0]), old_width], "newLogicalShape": [int(target.shape[0]), new_width]}


@torch.no_grad()
def copy_control_geometry(source, target, *, axis_segments=None, chunk_bytes=TRANSFER_BYTES, cancelled=None):
    """Explicit exact control/activity overlap; newly created controls are zero.

    Floating controls are permitted, floating projection weights are not a
    migration representation. Caller declares these as nonweight controls.
    """
    _reject_shared_storage(source, target)
    if (source.dtype != target.dtype or source.ndim != target.ndim or source.ndim > 2
        or not source.is_contiguous() or not target.is_contiguous()):
        raise ValueError("geometry controls require matching scalar/vector/matrix dtype")
    budget = max(source.element_size(), int(chunk_bytes))
    mappings = axis_segments or [[[0, 0, min(old, new)]] for old, new in zip(source.shape, target.shape)]
    if len(mappings) != source.ndim:
        raise ValueError("control coordinate axes do not match tensor rank")
    mappings = [_validate_axis_segments(mapping, old, new) for mapping, old, new in zip(mappings, source.shape, target.shape)]
    flat = target.reshape(-1)
    for start in range(0, flat.numel(), max(1, budget // target.element_size())):
        if cancelled is not None and cancelled(): raise InterruptedError("geometry control initialization cancelled")
        block = flat[start:start + max(1, budget // target.element_size())]
        block.zero_()
        _release_geometry_chunk(target, start * target.element_size(), block.numel() * target.element_size())
    if source.ndim == 0:
        target.copy_(source.to(target.device))
        digest = hashlib.sha256(source.detach().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
        return {"copiedElements": 1, "initializedElements": 0, "droppedElements": 0, "peakTransferBytes": source.element_size(),
            "oldShape": [], "newShape": [], "axisSegments": [], "copiedCoordinatesSha256": digest}
    copied, peak, digest = 0, 0, hashlib.sha256()
    rows = [[0, 0, 1]] if source.ndim == 1 else mappings[0]
    columns = mappings[0] if source.ndim == 1 else mappings[1]
    step = max(1, budget // source.element_size())
    for old_row, new_row, count_rows in rows:
        for row in range(count_rows):
            a = source if source.ndim == 1 else source[old_row + row]
            b = target if target.ndim == 1 else target[new_row + row]
            for old_column, new_column, count in columns:
                for start in range(0, count, step):
                    if cancelled is not None and cancelled(): raise InterruptedError("geometry control copy cancelled")
                    size = min(step, count - start)
                    chunk = a[old_column + start:old_column + start + size].to(target.device)
                    b[new_column + start:new_column + start + size].copy_(chunk)
                    if not torch.equal(b[new_column + start:new_column + start + size], chunk):
                        raise ValueError("control migration failed exact coordinate verification")
                    digest.update(chunk.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                    copied += size; peak = max(peak, size * source.element_size())
                    source_start = old_column + start if source.ndim == 1 else (old_row + row) * source.shape[1] + old_column + start
                    target_start = new_column + start if target.ndim == 1 else (new_row + row) * target.shape[1] + new_column + start
                    _release_geometry_chunk(source, source_start * source.element_size(), size * source.element_size())
                    _release_geometry_chunk(target, target_start * target.element_size(), size * target.element_size())
    return {"copiedElements": copied, "initializedElements": target.numel() - copied,
        "droppedElements": source.numel() - copied, "peakTransferBytes": peak,
        "oldShape": list(source.shape), "newShape": list(target.shape), "axisSegments": mappings,
        "copiedCoordinatesSha256": digest.hexdigest()}


def geometry_candidate_manifest(*, candidate_id, parent_metadata_sha256, parent_architecture_sha256,
    root_architecture_sha256, candidate_architecture_sha256, mutation, old_geometry, new_geometry, tensor_proofs,
    owner_inventory):
    """Source-free migration evidence, never an approval or quality receipt."""
    from .text_spool import bounded_json_sha256
    change = normalize_architecture_change({key: value for key, value in mutation.items() if key != "compatibilityBoundary"})
    body = {"format": "omni-isolated-geometry-migration", "formatVersion": 1,
        "candidateId": candidate_id, "parentMetadataSha256": parent_metadata_sha256,
        "parentArchitectureSha256": parent_architecture_sha256, "rootArchitectureSha256": root_architecture_sha256,
        "candidateArchitectureSha256": candidate_architecture_sha256,
        "mutation": {key: value for key, value in change.items() if key != "compatibilityBoundary"},
        "oldGeometry": copy.deepcopy(dict(old_geometry)), "newGeometry": copy.deepcopy(dict(new_geometry)),
        "tensorProofs": copy.deepcopy(dict(tensor_proofs)), "ownerInventory": copy.deepcopy(dict(owner_inventory)),
        "policy": architecture_mutation_policy(change),
        "stage": "migrated-not-trained-not-evaluated", "activationAllowed": False}
    return validate_geometry_candidate_manifest({**body, "contentSha256": bounded_json_sha256(body)})


def validate_geometry_candidate_manifest(value, *, parent_metadata_sha256=None, root_architecture_sha256=None):
    from .text_spool import bounded_json_sha256
    fields = {"format", "formatVersion", "candidateId", "parentMetadataSha256", "parentArchitectureSha256",
        "rootArchitectureSha256", "candidateArchitectureSha256", "mutation", "oldGeometry", "newGeometry",
        "tensorProofs", "ownerInventory", "policy", "stage", "activationAllowed", "contentSha256"}
    if not isinstance(value, Mapping) or set(value) != fields or value["format"] != "omni-isolated-geometry-migration" or value["formatVersion"] != 1:
        raise ValueError("geometry candidate manifest schema is invalid")
    if not isinstance(value["candidateId"], str) or not value["candidateId"] or value["stage"] != "migrated-not-trained-not-evaluated" or value["activationAllowed"] is not False:
        raise ValueError("migration evidence cannot activate or certify a candidate")
    for name in ("parentMetadataSha256", "parentArchitectureSha256", "rootArchitectureSha256", "candidateArchitectureSha256", "contentSha256"):
        if not isinstance(value[name], str) or len(value[name]) != 64 or any(char not in "0123456789abcdef" for char in value[name]):
            raise ValueError("geometry candidate identity hash is invalid")
    for geometry in (value["oldGeometry"], value["newGeometry"]):
        if not isinstance(geometry, Mapping) or set(geometry) != {"d_model", "d_ff", "n_heads", "idea_dim"}:
            raise ValueError("geometry candidate needs an explicit width/FF/head/idea geometry inventory")
        if any(type(size) is not int or not 1 <= size <= (1 << 53) - 1 for size in geometry.values()):
            raise ValueError("geometry candidate dimensions must be positive safe integers")
    normalized = normalize_architecture_change(value["mutation"])
    if normalized is None or value["mutation"] != {key: item for key, item in normalized.items() if key != "compatibilityBoundary"}:
        raise ValueError("geometry candidate mutation must use the exact canonical typed fields")
    proposed = geometry_candidate_config(value["oldGeometry"], value["mutation"])
    if any(proposed.get(key) != value["newGeometry"].get(key) for key in ("d_model", "d_ff", "n_heads", "idea_dim")):
        raise ValueError("geometry candidate target does not match its declared mutation")
    if value["policy"] != architecture_mutation_policy(value["mutation"]) or value["policy"]["geometryChanges"] is not True:
        raise ValueError("geometry candidate has an invalid preservation/activation policy")
    if not isinstance(value["tensorProofs"], Mapping) or not value["tensorProofs"]:
        raise ValueError("geometry candidate lacks exact tensor migration proofs")
    if not isinstance(value["ownerInventory"], Mapping) or set(value["ownerInventory"]) != set(value["tensorProofs"]):
        raise ValueError("geometry candidate proofs do not declare every migrated owner")
    for name, proof in value["tensorProofs"].items():
        if not isinstance(name, str) or not name or not isinstance(proof, Mapping):
            raise ValueError("geometry tensor proof identity is invalid")
        if "copiedLogicalTrits" in proof:
            fields = ("copiedLogicalTrits", "initializedLogicalTrits", "unmappedParentLogicalTrits", "peakTransferBytes")
            if any(type(proof.get(key)) is not int or proof[key] < 0 for key in fields) or proof.get("functionPreserved") is not False:
                raise ValueError("geometry packed proof counters/preservation claim are invalid")
            digest = proof.get("copiedCoordinatesSha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("geometry copied-coordinate proof hash is invalid")
            before, after = proof.get("oldLogicalShape"), proof.get("newLogicalShape")
            if any(not isinstance(shape, list) or len(shape) != 2 or any(type(size) is not int or size < 1 for size in shape) for shape in (before, after)):
                raise ValueError("geometry packed proof has invalid logical shapes")
            rows = _validate_axis_segments(proof.get("rowSegments"), before[0], after[0])
            columns = _validate_axis_segments(proof.get("columnSegments"), before[1], after[1])
            copied = sum(segment[2] for segment in rows) * sum(segment[2] for segment in columns)
            if (proof["copiedLogicalTrits"] != copied or proof["initializedLogicalTrits"] != math.prod(after) - copied
                or proof["unmappedParentLogicalTrits"] != math.prod(before) - copied
                or proof.get("initialization") != "coordinate-sha256-ternary-v1"):
                raise ValueError("geometry packed proof does not account for every copied/new/unmapped coordinate")
        elif "copiedElements" in proof:
            if any(type(proof.get(key)) is not int or proof[key] < 0 for key in ("copiedElements", "initializedElements", "droppedElements", "peakTransferBytes")):
                raise ValueError("geometry control proof is invalid")
            before, after, axes = proof.get("oldShape"), proof.get("newShape"), proof.get("axisSegments")
            if (any(not isinstance(shape, list) or len(shape) > 2 or any(type(size) is not int or size < 1 for size in shape) for shape in (before, after))
                or len(before) != len(after) or not isinstance(axes, list) or len(axes) != len(before)):
                raise ValueError("geometry control proof has invalid declared axes")
            copied = math.prod(sum(segment[2] for segment in _validate_axis_segments(axis, old, new)) for axis, old, new in zip(axes, before, after))
            if (proof["copiedElements"] != copied or proof["initializedElements"] != math.prod(after) - copied
                or proof["droppedElements"] != math.prod(before) - copied):
                raise ValueError("geometry control proof does not account for every coordinate")
            digest = proof.get("copiedCoordinatesSha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("geometry copied-control proof hash is invalid")
        else:
            raise ValueError("geometry candidate tensor proof has an unsupported representation")
        owner = value["ownerInventory"][name]
        if (not isinstance(owner, Mapping) or set(owner) != {"kind", "owner", "role", "oldShape", "newShape", "dtype"}
            or not isinstance(owner["owner"], str) or not owner["owner"]):
            raise ValueError("geometry candidate owner inventory schema is invalid")
        if "copiedLogicalTrits" in proof:
            old_shape = [proof["oldLogicalShape"][0], (proof["oldLogicalShape"][1] + 3) // 4]
            new_shape = [proof["newLogicalShape"][0], (proof["newLogicalShape"][1] + 3) // 4]
            if owner["kind"] != "packed" or owner["role"] not in {"weight", "bias"} or owner["dtype"] != "torch.uint8":
                raise ValueError("geometry packed owner representation is invalid")
        else:
            old_shape, new_shape = proof["oldShape"], proof["newShape"]
            if owner["kind"] != "control" or owner["role"] != "control" or not isinstance(owner["dtype"], str) or not owner["dtype"].startswith("torch."):
                raise ValueError("geometry control owner representation is invalid")
        if owner["oldShape"] != old_shape or owner["newShape"] != new_shape:
            raise ValueError("geometry owner physical shapes do not match their logical proof")
    if parent_metadata_sha256 is not None and value["parentMetadataSha256"] != parent_metadata_sha256 or root_architecture_sha256 is not None and value["rootArchitectureSha256"] != root_architecture_sha256:
        raise ValueError("geometry candidate does not bind its exact parent/origin")
    body = {key: item for key, item in value.items() if key != "contentSha256"}
    if bounded_json_sha256(body) != value["contentSha256"]:
        raise ValueError("geometry candidate manifest checksum is invalid")
    return copy.deepcopy(dict(value))


def verify_geometry_checkpoint_migration(before, after, manifest, *, chunk_bytes=TRANSFER_BYTES):
    """Prove every declared coordinate against the pre-training checkpoint.

    Every undeclared checkpoint tensor must remain byte-identical, including
    controls outside module state. Added/removed tensors and undeclared shape
    changes fail closed. Verification never loads a whole packed owner.
    """
    manifest = validate_geometry_candidate_manifest(manifest)
    before, after = Path(before), Path(after)
    budget = max(32, int(chunk_bytes))
    pairs = {}
    for filename, prefix in (("core.safetensors", "core:"), ("plasticity.safetensors", "plasticity:")):
        old = BoundedTensorFile(before / filename, chunk_bytes=budget)
        new = BoundedTensorFile(after / filename, chunk_bytes=budget)
        if set(old.specs) != set(new.specs):
            raise ValueError("geometry migration added or removed undeclared tensor owners")
        pairs.update({prefix + key: (old, new, key) for key in old.specs})
    if set(manifest["ownerInventory"]) - set(pairs):
        raise ValueError("geometry migration inventory refers to an unavailable checkpoint owner")
    required_packed = {name for name in pairs if name.endswith(("._packed_forward_weight", "._packed_forward_bias"))}
    if any(manifest["ownerInventory"].get(name, {}).get("kind") != "packed" for name in required_packed):
        raise ValueError("geometry migration lacks a logical-coordinate proof for every native packed projection owner")
    verified, unchanged, peak = 0, 0, 0
    for name, (old_reader, new_reader, key) in pairs.items():
        old, new = old_reader.specs[key], new_reader.specs[key]
        owner = manifest["ownerInventory"].get(name)
        if owner is None:
            if old.shape != new.shape or old.dtype != new.dtype:
                raise ValueError("geometry migration changed an undeclared tensor owner: " + name)
            for left, right in zip(old_reader.chunks(key), new_reader.chunks(key)):
                if not torch.equal(left.reshape(-1).view(torch.uint8), right.reshape(-1).view(torch.uint8)):
                    raise ValueError("geometry migration changed undeclared owner bytes: " + name)
                peak = max(peak, left.numel() * left.element_size())
            unchanged += 1
            continue
        if (list(old.shape) != owner["oldShape"] or list(new.shape) != owner["newShape"]
            or str(old.dtype) != owner["dtype"] or new.dtype != old.dtype):
            raise ValueError("geometry checkpoint owner does not match declared shape/dtype: " + name)
        proof = manifest["tensorProofs"][name]
        digest = hashlib.sha256()
        with old_reader.path.open("rb") as left, new_reader.path.open("rb") as right:
            def raw(handle, spec, offset, count):
                nonlocal peak
                handle.seek(spec.offset + offset)
                block = handle.read(count)
                if len(block) != count: raise ValueError("geometry checkpoint ended before its declared tensor range")
                peak = max(peak, count)
                return block
            if owner["kind"] == "packed":
                old_width, new_width = proof["oldLogicalShape"][1], proof["newLogicalShape"][1]
                def codes(handle, spec, row, column, count):
                    payload = raw(handle, spec, row * spec.shape[1] + column // 4, (column % 4 + count + 3) // 4)
                    block = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                    offsets = torch.arange(count, dtype=torch.long) + column % 4
                    return ((block[offsets // 4] >> (2 * (offsets % 4))) & 3).to(torch.uint8)
                for reader, spec, handle, width in ((old_reader, old, left, old_width), (new_reader, new, right, new_width)):
                    for block in reader.chunks(key):
                        if any(bool((((block >> shift) & 3) == 3).any()) for shift in (0, 2, 4, 6)):
                            raise ValueError("geometry checkpoint contains reserved packed codes")
                    if width % 4:
                        for row in range(spec.shape[0]):
                            padding = raw(handle, spec, row * spec.shape[1] + spec.shape[1] - 1, 1)[0]
                            if any(((padding >> (2 * lane)) & 3) != 1 for lane in range(width % 4, 4)):
                                raise ValueError("geometry checkpoint has noncanonical packed padding")
                for old_row, new_row, rows in proof["rowSegments"]:
                    for row in range(rows):
                        for old_column, new_column, columns in proof["columnSegments"]:
                            for start in range(0, columns, max(1, budget // 64)):
                                count = min(max(1, budget // 64), columns - start)
                                a = codes(left, old, old_row + row, old_column + start, count)
                                b = codes(right, new, new_row + row, new_column + start, count)
                                if not torch.equal(a, b):
                                    raise ValueError("geometry checkpoint changed copied logical trits: " + name)
                                digest.update(a.numpy().tobytes())
            else:
                item = torch.empty((), dtype=old.dtype).element_size()
                if not old.shape:
                    a, b = raw(left, old, 0, item), raw(right, new, 0, item)
                    if a != b: raise ValueError("geometry checkpoint changed a copied scalar control: " + name)
                    digest.update(a)
                else:
                    axes = proof["axisSegments"]
                    rows = [[0, 0, 1]] if len(old.shape) == 1 else axes[0]
                    columns = axes[0] if len(old.shape) == 1 else axes[1]
                    for old_row, new_row, row_count in rows:
                        for row in range(row_count):
                            for old_column, new_column, count in columns:
                                for start in range(0, count, max(1, budget // item)):
                                    elements = min(max(1, budget // item), count - start)
                                    old_start = old_column + start if len(old.shape) == 1 else (old_row + row) * old.shape[1] + old_column + start
                                    new_start = new_column + start if len(new.shape) == 1 else (new_row + row) * new.shape[1] + new_column + start
                                    a = raw(left, old, old_start * item, elements * item)
                                    b = raw(right, new, new_start * item, elements * item)
                                    if a != b: raise ValueError("geometry checkpoint changed copied control coordinates: " + name)
                                    digest.update(a)
            if digest.hexdigest() != proof["copiedCoordinatesSha256"]:
                raise ValueError("geometry checkpoint copied-coordinate hash does not match its proof: " + name)
        verified += 1
    return {"verified": True, "migratedOwnersChecked": verified, "unchangedOwnersChecked": unchanged,
        "peakTransferBytes": peak, "verifiedBeforeTraining": True, "functionPreserved": False,
        "migrationManifestSha256": manifest["contentSha256"]}


def geometry_owner_plan(name, old_logical_shape, new_logical_shape, old_geometry, new_geometry, *, role="weight"):
    """Cheap metadata contract for the central candidate loader/migrator."""
    result = geometry_axis_segments(name, old_logical_shape, new_logical_shape, old_geometry, new_geometry, role=role)
    result["rowSegments"] = _validate_axis_segments(result["rowSegments"], old_logical_shape[0], new_logical_shape[0])
    result["columnSegments"] = _validate_axis_segments(result["columnSegments"], old_logical_shape[1], new_logical_shape[1])
    result["role"] = role
    result["oldPackedShape"] = [old_logical_shape[0], (old_logical_shape[1] + 3) // 4]
    result["newPackedShape"] = [new_logical_shape[0], (new_logical_shape[1] + 3) // 4]
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
    if mutation["mutation"] in {"resize-width", "repartition-heads"}:
        shape["dModel"] = int(config.d_model)
        shape["feedForward"] = int(config.d_ff)
        shape["nHeads"] = int(config.n_heads)
    result["inventory"] = native_core_inventory(shape)
    old_lineage = previous.get("evolutionLineage", {})
    history = list(old_lineage.get("mutations", ()))
    normalized = normalize_architecture_change({key: value for key, value in mutation.items() if key != "compatibilityBoundary"})
    history.append({key: value for key, value in normalized.items() if key != "compatibilityBoundary"})
    geometry_history = any(item["mutation"] in {"resize-width", "repartition-heads"} for item in history)
    result["evolutionLineage"] = {
        "format": "omni-native-candidate-architecture-lineage" if geometry_history else "omni-compatible-architecture-lineage",
        "formatVersion": 2 if geometry_history else 1,
        "rootArchitectureSha256": old_lineage.get("rootArchitectureSha256", previous["sha256"]),
        "parentArchitectureSha256": previous["sha256"],
        "rootShape": copy.deepcopy(old_lineage.get("rootShape", previous["shape"])),
        "mutations": history,
        "normalizationAndHeadGeometryChanged": geometry_history,
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
        or (lineage["format"], lineage["formatVersion"]) not in {("omni-compatible-architecture-lineage", 1), ("omni-native-candidate-architecture-lineage", 2)}
        or type(lineage["normalizationAndHeadGeometryChanged"]) is not bool or lineage["qualityVerified"] is not False
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
        elif canonical["mutation"] == "resize-width":
            step["shape"]["dModel"] = canonical["dModel"]
            step["shape"]["feedForward"] = canonical.get("feedForward", step["shape"]["feedForward"])
            step["shape"]["nHeads"] = canonical.get("nHeads", step["shape"]["nHeads"])
        elif canonical["mutation"] == "repartition-heads":
            step["shape"]["nHeads"] = canonical["nHeads"]
        step["inventory"] = native_core_inventory(step["shape"])
        geometry_history = any(item["mutation"] in {"resize-width", "repartition-heads"} for item in history)
        step["evolutionLineage"] = {
            "format": "omni-native-candidate-architecture-lineage" if geometry_history else "omni-compatible-architecture-lineage",
            "formatVersion": 2 if geometry_history else 1,
            "rootArchitectureSha256": root["sha256"], "parentArchitectureSha256": previous["sha256"],
            "rootShape": copy.deepcopy(root["shape"]), "mutations": list(history),
            "normalizationAndHeadGeometryChanged": geometry_history, "qualityVerified": False,
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
