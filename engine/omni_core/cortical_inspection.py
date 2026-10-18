"""Load-free bounded inspection of committed packed native cortical weights.

No brain/module construction, decoder execution or sparse-substrate index.
Tensor-axis links are computational relationships, not semantic explanations.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from .bounded_tensor_io import BoundedTensorFile


PAGE_ITEMS = 256
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _json(path: Path):
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("cortical metadata exceeds bounded inspection envelope")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("invalid cortical metadata")
    return value


def _safe(root: Path, relative: str):
    if not relative or "\\" in relative or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("unsafe cortical role path")
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def _sources(engine: Path, brain_id: str):
    metadata = _json(engine / "brain.json")
    if str(metadata.get("brain_id")) != brain_id:
        raise ValueError("cortical brain identity mismatch")
    pointer = metadata.get("mutable_state")
    if not isinstance(pointer, Mapping) or not _SHA.fullmatch(str(pointer.get("activeGeneration", ""))):
        raise ValueError("cortical inspection requires a committed native generation")
    generation_id = str(pointer["activeGeneration"])
    relative = str(pointer.get("generationManifest", ""))
    if relative != "generations/%s/manifest.json" % generation_id:
        raise ValueError("invalid cortical generation pointer")
    path = _safe(engine / "state", relative)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("cortical generation metadata exceeds inspection envelope")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != pointer.get("generationManifestSha256"):
        raise ValueError("cortical generation manifest checksum mismatch")
    generation = json.loads(encoded)
    if hashlib.sha256(_canonical({key: value for key, value in generation.items() if key != "contentSha256"})).hexdigest() != generation_id:
        raise ValueError("cortical generation content mismatch")
    if generation.get("contentSha256") != generation_id or pointer.get("contentSha256") != generation_id:
        raise ValueError("cortical generation identity mismatch")
    readers = {}
    for role in ("core", "plasticity"):
        spec = generation.get("roles", {}).get(role, {})
        digest = str(spec.get("sha256", ""))
        if not _SHA.fullmatch(digest) or spec.get("path") != "blobs/%s.safetensors" % digest:
            raise ValueError("invalid cortical role descriptor")
        blob = _safe(engine / "state", spec["path"])
        if blob.stat().st_size != int(spec.get("bytes", -1)):
            raise ValueError("cortical role size changed")
        readers[role] = BoundedTensorFile(blob)
    shapes = {}
    packed_path = engine / "packed-ternary" / "manifest.json"
    checksum_path = engine / "packed-ternary" / "manifest.sha256"
    if packed_path.is_file() and checksum_path.is_file():
        if packed_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("packed layout metadata exceeds inspection envelope")
        encoded = packed_path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() == checksum_path.read_text().strip():
            manifest = json.loads(encoded)
            for entry in manifest.get("tensors", []):
                name, shape = entry.get("name"), entry.get("shape")
                if isinstance(name, str) and isinstance(shape, list) and all(isinstance(size, int) and size > 0 for size in shape):
                    shapes[name] = shape
    revision = hashlib.sha256(_canonical({
        "generation": generation_id, "layout": shapes,
        "tokenContext": metadata.get("recent_token_context", []),
        "roles": {role: [reader.path.stat().st_ino, reader.path.stat().st_mtime_ns, reader.path.stat().st_size]
                  for role, reader in readers.items()},
    })).hexdigest()
    return metadata, revision, readers, shapes


def _inventory(readers, shapes):
    result = []
    for role, reader in readers.items():
        for name, spec in reader.specs.items():
            if name.endswith("._packed_forward_weight"):
                module, field, logical_key = name.rsplit(".", 1)[0], "weight", name.rsplit(".", 1)[0] + ".weight"
            elif name.endswith("._packed_forward_bias"):
                module, field, logical_key = name.rsplit(".", 1)[0], "bias", name.rsplit(".", 1)[0] + ".bias"
            elif name.endswith("._packed_weights"):
                module, field, logical_key = name.rsplit(".", 1)[0], "weight", name.rsplit(".", 1)[0] + ".weights"
            elif name.endswith("._packed_sign") and ".parallel_connections.pages." in name:
                module, field = name.rsplit(".", 1)[0], "weight"
                from .packed_connections import PooledParallelConnections, CONTACT_PAGE_CAPACITY
                role_key, bucket, page_id = PooledParallelConnections.parse_page_key(module.rsplit(".", 1)[1])
                parent_owner = module.split('.parallel_connections.pages.', 1)[0]
                parent_state = reader.specs.get(parent_owner + ('._bias_connection_state' if role_key == 'b' else '._connection_state'))
                state = reader.specs.get(module + "._connection_state")
                registry = reader.specs.get(module + "._record_meta")
                if (tuple(spec.shape) != (1, CONTACT_PAGE_CAPACITY // 4) or spec.dtype != torch.uint8
                        or state is None or state.dtype != torch.uint8 or tuple(state.shape) != (1, CONTACT_PAGE_CAPACITY)
                        or registry is None or registry.dtype != torch.uint8 or tuple(registry.shape) != (CONTACT_PAGE_CAPACITY, 10)
                        or parent_state is None or parent_state.dtype != torch.uint8 or len(parent_state.shape) != 2):
                    raise ValueError("cortical repeated-contact page is incomplete")
                result.append({"id": name, "module": module, "field": "bias" if role_key == "b" else "weight",
                    "role": role, "group": module.split(".")[0], "rows": 1, "columns": CONTACT_PAGE_CAPACITY,
                    "packedColumns": CONTACT_PAGE_CAPACITY // 4, "packedBytes": spec.byte_count,
                    "connectionStateBytes": state.byte_count, "structuralBytes": registry.byte_count,
                    "connectionStateKey": module + "._connection_state", "registryKey": module + "._record_meta",
                    "logicalShape": None, "logicalParameters": None, "layout": "repeated-contact-slots",
                    "storageKind": "repeated-contact-page", "contactRowBucket": bucket, "contactPageId": page_id,
                    "activityOwner": parent_owner,
                    "parentRows": parent_state.shape[0], "parentColumns": parent_state.shape[1],
                    "shapeEvidence": "exact-contact-page-live-entries-paged-on-selection",
                    "activationObserved": False, "activation": None})
                continue
            else:
                continue
            if spec.dtype != torch.uint8 or len(spec.shape) != 2:
                raise ValueError("cortical synapse storage is not a packed uint8 matrix")
            rows, packed_columns = spec.shape
            state_name = module + ("._bias_connection_state" if field == "bias" else "._connection_state")
            state = reader.specs.get(state_name)
            logical = shapes.get(logical_key)
            width = None
            layout = "row-packed"
            if logical:
                if field == "bias" and len(logical) == 1 and rows == 1:
                    width = logical[0]
                elif len(logical) == 2 and logical[0] == rows:
                    width = logical[1]
                elif len(logical) > 2 and logical[0] == rows:
                    width = math.prod(logical[1:]); layout = "convolution-output-row"
                elif len(logical) > 2 and logical[1] * math.prod(logical[2:]) == rows:
                    width = logical[0]; layout = "transposed-convolution-kernel-row"
                if width is not None and (width + 3) // 4 != packed_columns:
                    width = None
            if state is not None:
                if state.dtype != torch.uint8 or len(state.shape) != 2 or state.shape[0] != rows or (state.shape[1] + 3) // 4 != packed_columns:
                    raise ValueError("cortical connection state geometry differs from signs")
                if width is not None and width != state.shape[1]:
                    raise ValueError("cortical declared layout differs from connection state")
                width = state.shape[1]
            parts = module.split(".")
            group = ".".join(parts[:3]) if len(parts) > 2 and parts[1] in {"blocks", "experts"} else parts[0]
            result.append({"id": name, "module": module, "field": field, "role": role, "group": group,
                           "rows": rows, "columns": width, "packedColumns": packed_columns,
                           "packedBytes": spec.byte_count, "logicalShape": logical if width is not None else None,
                           "logicalParameters": rows * width if width is not None else None,
                           "connectionStateBytes": state.byte_count if state is not None else 0, "structuralBytes": 0,
                           "connectionStateKey": state_name if state is not None else None, "storageKind": "base-projection",
                           "activityOwner": module,
                           "layout": layout, "shapeEvidence": ("native-connection-state-shape" if state is not None else "checksum-bound-export-layout-matches-storage") if width is not None else "logical-width-unavailable",
                           "activationObserved": False, "activation": None})
    return sorted(result, key=lambda item: item["id"])


def _read_range(reader, name, first, count):
    spec = reader.specs[name]
    if first < 0 or count < 0 or first + count > spec.byte_count:
        raise ValueError("cortical selected byte range is invalid")
    with reader.path.open("rb") as handle:
        handle.seek(spec.offset + first)
        value = handle.read(count)
    if len(value) != count:
        raise ValueError("cortical selected byte range was truncated")
    return value


def _contacts(reader, selected, row, first_column, end_column):
    first, last = first_column // 4, (end_column + 3) // 4
    packed = _read_range(reader, selected['id'], row * selected['packedColumns'] + first, last - first)
    state_key = selected.get('connectionStateKey')
    state = (_read_range(reader, state_key, row * selected['columns'] + first_column, end_column - first_column)
             if state_key else bytes(end_column - first_column))
    repeated = selected.get('storageKind') == 'repeated-contact-page'
    registry = (_read_range(reader, selected['registryKey'], first_column * 10, (end_column - first_column) * 10)
                if repeated else None)
    for column in range(first_column, end_column):
        code = (packed[column // 4 - first] >> (2 * (column % 4))) & 3
        if code == 3:
            raise ValueError("cortical selected synapse contains reserved ternary code")
        value = state[column - first_column]
        actual_row, actual_column = row, column
        metadata = {}
        if repeated:
            start = (column - first_column) * 10
            row_offset, present = registry[start], registry[start + 1]
            if present not in (0, 1, 2):
                raise ValueError("cortical repeated-contact presence flag is invalid")
            if present != 1:
                continue
            if row_offset >= 32:
                raise ValueError("cortical repeated-contact endpoint is invalid")
            actual_row = selected['contactRowBucket'] + row_offset
            actual_column = int.from_bytes(registry[start + 2:start + 10], 'little')
            if actual_row >= selected['parentRows'] or actual_column >= selected['parentColumns']:
                raise ValueError("cortical repeated-contact endpoint exceeds its native owner")
            metadata = {'slot': column, 'ordinal': selected['contactPageId'] * 1024 + column}
        yield {"row": actual_row, "column": actual_column, "value": code - 1,
               "strength": (value & 15) + 1, "resistance": value >> 4,
               "effectiveContribution": (code - 1) * ((value & 15) + 1),
               "isRepeatedContact": repeated, **metadata,
               "activationObserved": False, "activation": None}


def _count(value, name, default=0):
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("invalid cortical %s" % name)
    return value


def query_committed_cortex(engine: Path, brain_id: str, query: Mapping[str, Any]):
    metadata, revision, readers, shapes = _sources(Path(engine), brain_id)
    modules = _inventory(readers, shapes)
    entity = str(query.get("entity", "modules"))
    if entity not in {"modules", "rows", "elements", "links", "boundaries"}:
        raise ValueError("invalid cortical entity")
    page_size = max(1, min(PAGE_ITEMS, _count(query.get("pageSize"), "page size", 64)))
    offset = _count(query.get("offset"), "offset")
    selected_id = str(query.get("moduleId", ""))
    group, search = str(query.get("group", "")), str(query.get("search", "")).casefold()
    if any(len(value) > 512 for value in (selected_id, group, search)):
        raise ValueError("cortical filter exceeds inspection envelope")
    scope = [item for item in modules if (not group or item["group"] == group) and (not search or search in item["id"].casefold())]
    selected = next((item for item in modules if item["id"] == selected_id), None)
    row = _count(query.get("row"), "row")
    binding = {"revision": revision, "entity": entity, "moduleId": selected_id, "row": row, "group": group, "search": search}
    cursor = query.get("cursor")
    if cursor:
        try:
            decoded = json.loads(base64.urlsafe_b64decode(str(cursor)))
        except (ValueError, TypeError):
            raise ValueError("invalid cortical cursor")
        if not isinstance(decoded, Mapping) or any(decoded.get(key) != value for key, value in binding.items()):
            raise ValueError("cortical generation/filter cursor changed")
        offset = _count(decoded.get("offset"), "cursor offset")
    records = []
    scanned_to = None
    total = len(scope)
    if entity == "modules":
        records = scope[offset:offset + page_size]
    elif entity in {"rows", "elements"}:
        if selected is None:
            raise ValueError("cortical module is missing")
        if entity == "rows":
            total = selected["rows"]
            records = [{"row": index, "columns": selected["columns"], "activationObserved": False, "activation": None}
                       for index in range(offset, min(total, offset + page_size))]
        else:
            if row >= selected["rows"] or selected["columns"] is None:
                raise ValueError("row out of range or exact logical width unavailable")
            total = selected["columns"]
            end = min(total, offset + page_size)
            if offset > total:
                raise ValueError("cortical column offset is invalid")
            records = list(_contacts(readers[selected["role"]], selected, row, offset, end))
            scanned_to = end
    elif entity == "links":
        if selected is None:
            raise ValueError("cortical module is missing")
        if row >= selected["rows"]:
            raise ValueError("cortical output row is out of range")
        total = selected["columns"] or 0
        end = min(total, offset + page_size)
        if offset > total:
            raise ValueError("cortical column offset is invalid")
        bias = selected["field"] == "bias"
        records = [{"sourceAxis": "constant-one" if bias else "tensor-column",
                    "sourceIndex": 0 if bias else contact["column"],
                    "targetAxis": "output-row" if bias else "tensor-row",
                    "targetIndex": contact["column"] if bias else contact["row"],
                    "moduleId": selected_id, "relationship": "packed-bias-addition" if bias else "packed-weight-coordinate",
                    **{key: value for key, value in contact.items() if key not in {'row', 'column'}},
                    "semanticExplanationVerified": False}
                   for contact in _contacts(readers[selected['role']], selected, row, offset, end)]
        scanned_to = end
    else:
        tokens = metadata.get("recent_token_context", [])
        tokens = tokens if isinstance(tokens, list) else []
        total = len(tokens)
        end = min(total, offset + page_size)
        if any(not isinstance(token, int) or isinstance(token, bool) or not 0 <= token <= 260
               for token in tokens[offset:end]):
            raise ValueError("cortical selected token boundary is invalid")
        special = {0: "padding", 1: "start", 2: "end", 259: "human-boundary", 260: "brain-boundary"}
        records = [{"position": index, "tokenId": token, "kind": special.get(token, "utf8-byte"),
                    "byteValue": token - 3 if 3 <= token <= 258 else None,
                    "embeddingRow": token, "languageHeadRow": token,
                    "observedInput": True, "firingObserved": False, "activation": None}
                   for index, token in enumerate(tokens[offset:end], offset)]
        scanned_to = end
    next_offset = scanned_to if scanned_to is not None else offset + len(records)
    has_more = next_offset < total
    next_cursor = base64.urlsafe_b64encode(_canonical({**binding, "offset": next_offset})).decode() if has_more else None
    base_modules = [item for item in modules if item["storageKind"] == "base-projection"]
    repeated_modules = [item for item in modules if item["storageKind"] == "repeated-contact-page"]
    owners = {item["activityOwner"] for item in modules}
    return {"brainId": brain_id, "revision": revision, "source": "committed-packed-native-cortex",
            "entity": entity, "total": total, "offset": offset, "returned": len(records),
            "hasMore": has_more, "nextCursor": next_cursor, "records": records,
            "selectedModule": selected,
            "groups": sorted({item["group"] for item in modules}),
            "moduleCount": len(modules), "packedBytes": sum(item["packedBytes"] for item in modules),
            "logicalParameters": sum(item["logicalParameters"] or 0 for item in modules),
            "logicalInventoryComplete": all(item["logicalParameters"] is not None for item in modules),
            "connectionStateBytes": sum(item['connectionStateBytes'] for item in modules),
            "structuralBytes": sum(item['structuralBytes'] for item in modules),
            "inventory": {"baseLogicalParameters": sum(item["logicalParameters"] or 0 for item in base_modules),
                          "baseLogicalInventoryComplete": all(item["logicalParameters"] is not None for item in base_modules),
                          "repeatedContactPages": len(repeated_modules),
                          "allocatedRepeatedContactSlots": sum(item["columns"] for item in repeated_modules),
                          "repeatedLiveContacts": None},
            "scannedThrough": next_offset,
            "activity": {"observed": False, "reason": "committed-weights-do-not-record-live-forward-firing"},
            "integrity": {"generationManifestVerified": True, "selectedTernaryCodesValidated": entity in {"elements", "links"},
                          "wholeRolePayloadScanned": False},
            "relationships": {"kind": "computational-dependencies-not-semantic-explanation",
                              "ideaInputs": [name for name in ("memory_bridge", "idea_adapter", "decoder.memory_projection")
                                             if any(owner == name or owner.startswith(name + ".") for owner in owners)],
                              "tokenInputs": [name for name in ("decoder.embedding", "decoder.language_head")
                                              if name in owners]}}
