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
            else:
                continue
            if spec.dtype != torch.uint8 or len(spec.shape) != 2:
                raise ValueError("cortical synapse storage is not a packed uint8 matrix")
            rows, packed_columns = spec.shape
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
            parts = module.split(".")
            group = ".".join(parts[:3]) if len(parts) > 2 and parts[1] in {"blocks", "experts"} else parts[0]
            result.append({"id": name, "module": module, "field": field, "role": role, "group": group,
                           "rows": rows, "columns": width, "packedColumns": packed_columns,
                           "packedBytes": spec.byte_count, "logicalShape": logical if width is not None else None,
                           "logicalParameters": math.prod(logical) if width is not None else None,
                           "layout": layout, "shapeEvidence": "checksum-bound-export-layout-matches-storage" if width is not None else "logical-width-unavailable",
                           "activationObserved": False, "activation": None})
    return sorted(result, key=lambda item: item["id"])


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
        if any(decoded.get(key) != value for key, value in binding.items()):
            raise ValueError("cortical generation/filter cursor changed")
        offset = _count(decoded.get("offset"), "cursor offset")
    records = []
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
            reader = readers[selected["role"]]
            spec = reader.specs[selected_id]
            first, last = offset // 4, (end + 3) // 4
            with reader.path.open("rb") as handle:
                handle.seek(spec.offset + row * selected["packedColumns"] + first)
                packed = handle.read(last - first)
            if len(packed) != last - first:
                raise ValueError("cortical selected byte range was truncated")
            for column in range(offset, end):
                code = (packed[column // 4 - first] >> (2 * (column % 4))) & 3
                if code == 3:
                    raise ValueError("cortical selected synapse contains reserved ternary code")
                records.append({"row": row, "column": column, "value": code - 1,
                                "activationObserved": False, "activation": None})
    elif entity == "links":
        if selected is None:
            raise ValueError("cortical module is missing")
        if row >= selected["rows"]:
            raise ValueError("cortical output row is out of range")
        total = selected["columns"] or 0
        records = [{"sourceAxis": "input-column", "sourceIndex": index,
                    "targetAxis": "output-row", "targetIndex": row,
                    "moduleId": selected_id, "relationship": "packed-projection-computational-dependency",
                    "semanticExplanationVerified": False}
                   for index in range(offset, min(total, offset + page_size))]
    else:
        tokens = metadata.get("recent_token_context", [])
        tokens = tokens if isinstance(tokens, list) else []
        total = len(tokens)
        special = {0: "padding", 1: "start", 2: "end", 259: "human-boundary", 260: "brain-boundary"}
        records = [{"position": index, "tokenId": token, "kind": special.get(token, "utf8-byte"),
                    "byteValue": token - 3 if 3 <= token <= 258 else None,
                    "embeddingRow": token, "languageHeadRow": token,
                    "observedInput": True, "firingObserved": False, "activation": None}
                   for index, token in enumerate(tokens[offset:offset + page_size], offset)
                   if isinstance(token, int) and not isinstance(token, bool) and 0 <= token <= 260]
    next_offset = offset + len(records)
    has_more = next_offset < total
    next_cursor = base64.urlsafe_b64encode(_canonical({**binding, "offset": next_offset})).decode() if has_more else None
    return {"brainId": brain_id, "revision": revision, "source": "committed-packed-native-cortex",
            "entity": entity, "total": total, "offset": offset, "returned": len(records),
            "hasMore": has_more, "nextCursor": next_cursor, "records": records,
            "selectedModule": selected,
            "groups": sorted({item["group"] for item in modules}),
            "moduleCount": len(modules), "packedBytes": sum(item["packedBytes"] for item in modules),
            "logicalParameters": sum(item["logicalParameters"] or 0 for item in modules),
            "logicalInventoryComplete": all(item["columns"] is not None for item in modules),
            "activity": {"observed": False, "reason": "committed-weights-do-not-record-live-forward-firing"},
            "integrity": {"generationManifestVerified": True, "selectedTernaryCodesValidated": entity == "elements",
                          "wholeRolePayloadScanned": False},
            "relationships": {"kind": "computational-dependencies-not-semantic-explanation",
                              "ideaInputs": ["memory_bridge", "idea_adapter", "decoder.memory_projection"],
                              "tokenInputs": ["decoder.embedding", "decoder.language_head"]}}
