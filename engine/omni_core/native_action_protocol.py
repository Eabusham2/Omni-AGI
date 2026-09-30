"""Structural-only action schemas and invocation-local emission identity.

No tool descriptions, behavioral prompts, lexical route rules, stored answers,
or model/provider calls live here. The native cortex supplies every choice and
argument value; this module validates structure and prevents duplicate effects.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional


_TYPES = {"object", "array", "string", "number", "integer", "boolean", "null", "unknown"}
_ANNOTATIONS = {"title", "description", "examples", "default", "$comment", "$schema", "$id", "deprecated", "readOnly", "writeOnly", "format"}
_NUMERIC = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}
_COUNTS = {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"}


def structural_schema(raw: Any, *, max_depth: int = 32) -> Optional[Dict[str, Any]]:
    """Preserve recursive typed structure and discard annotation prose.

    Local references are resolved without network access. Unsupported asserting
    keywords stay an explicit rejection, never silently weakened validation.
    This bounded subset is not advertised as full JSON Schema conformance.
    """
    if not isinstance(raw, Mapping):
        return None
    root = raw

    def clean(value: Any, depth: int, references: tuple[str, ...] = ()) -> Dict[str, Any]:
        if depth > max_depth or not isinstance(value, Mapping):
            return {"type": "unknown", "unsupported": ["invalid-or-deep-schema"]}
        result: Dict[str, Any] = {}
        reference = value.get("$ref")
        if reference is not None:
            if not isinstance(reference, str) or not reference.startswith("#/") or reference in references:
                return {"type": "unknown", "unsupported": ["nonlocal-or-recursive-ref"]}
            target: Any = root
            for part in reference[2:].split("/"):
                target = target.get(part.replace("~1", "/").replace("~0", "~")) if isinstance(target, Mapping) else None
            result = clean(target, depth + 1, (*references, reference))
        for key, item in value.items():
            if key in _ANNOTATIONS or key in {"$ref", "$defs", "definitions"}:
                continue
            if key == "type":
                types = item if isinstance(item, (list, tuple)) else [item]
                if not types or any(kind not in _TYPES for kind in types):
                    result[key] = "unknown"
                else:
                    result[key] = list(types) if isinstance(item, (list, tuple)) else item
            elif key == "properties" and isinstance(item, Mapping):
                result[key] = {str(name): clean(spec, depth + 1, references) for name, spec in sorted(item.items())}
            elif key == "required" and isinstance(item, (list, tuple)) and all(isinstance(name, str) for name in item):
                result[key] = sorted(set(item))
            elif key in {"items", "additionalProperties"} and isinstance(item, (Mapping, bool)):
                result[key] = clean(item, depth + 1, references) if isinstance(item, Mapping) else item
            elif key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(item, (list, tuple)):
                result[key] = [clean(spec, depth + 1, references) for spec in item]
            elif key == "not" and isinstance(item, Mapping):
                result[key] = clean(item, depth + 1, references)
            elif key in {"enum", "const"}:
                try:
                    result[key] = json.loads(json.dumps(item, allow_nan=False))
                except (TypeError, ValueError):
                    result.setdefault("unsupported", []).append(key)
            elif key in _NUMERIC and isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)):
                result[key] = item
            elif key in _COUNTS and isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                result[key] = item
            elif key == "uniqueItems" and isinstance(item, bool):
                result[key] = item
            else:
                result.setdefault("unsupported", []).append(str(key))
        return result

    return clean(raw, 0)


def validate_structural_value(value: Any, schema: Mapping[str, Any], *, depth: int = 0) -> bool:
    if depth > 32 or schema.get("unsupported"):
        return False
    try:
        finite_number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False
    primitive = {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, (list, tuple)),
        "string": isinstance(value, str),
        "number": finite_number,
        "integer": finite_number and float(value).is_integer(),
        "boolean": isinstance(value, bool),
        "null": value is None,
        "unknown": False,
    }
    types = schema.get("type")
    if types is not None and not any(primitive.get(kind, False) for kind in (types if isinstance(types, list) else [types])):
        return False
    canonical = lambda item: json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
    try:
        encoded = canonical(value)
        if "enum" in schema and not any(encoded == canonical(item) for item in schema["enum"]):
            return False
        if "const" in schema and encoded != canonical(schema["const"]):
            return False
    except (TypeError, ValueError):
        return False
    for keyword, predicate in (
        ("allOf", lambda matches: all(matches)),
        ("anyOf", lambda matches: any(matches)),
        ("oneOf", lambda matches: sum(matches) == 1),
    ):
        if keyword in schema and not predicate([validate_structural_value(value, item, depth=depth + 1) for item in schema[keyword]]):
            return False
    if "not" in schema and validate_structural_value(value, schema["not"], depth=depth + 1):
        return False
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        if any(name not in value for name in schema.get("required", ())):
            return False
        # Native action channels intentionally keep unspecified extra fields
        # closed; schemas may explicitly opt into additional typed properties.
        additional = schema.get("additionalProperties", "properties" not in schema)
        for name, item in value.items():
            if name in properties:
                if not validate_structural_value(item, properties[name], depth=depth + 1):
                    return False
            elif additional is False:
                return False
            elif isinstance(additional, Mapping) and not validate_structural_value(item, additional, depth=depth + 1):
                return False
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get("maxProperties", math.inf):
            return False
    if isinstance(value, (list, tuple)):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            return False
        prefix = schema.get("prefixItems", ())
        for index, item in enumerate(value):
            spec = prefix[index] if index < len(prefix) else schema.get("items", True)
            if spec is False or (isinstance(spec, Mapping) and not validate_structural_value(item, spec, depth=depth + 1)):
                return False
        if schema.get("uniqueItems") and len({canonical(item) for item in value}) != len(value):
            return False
    if isinstance(value, str) and (len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf)):
        return False
    if primitive["number"]:
        for key, invalid in (
            ("minimum", lambda boundary: value < boundary),
            ("maximum", lambda boundary: value > boundary),
            ("exclusiveMinimum", lambda boundary: value <= boundary),
            ("exclusiveMaximum", lambda boundary: value >= boundary),
        ):
            if key in schema and invalid(schema[key]):
                return False
        divisor = schema.get("multipleOf")
        if divisor is not None and (divisor <= 0 or not math.isclose(float(value) / divisor, round(float(value) / divisor), abs_tol=1e-9)):
            return False
    return True


def action_fingerprint(action: Mapping[str, Any]) -> str:
    arguments = dict(action.get("arguments", {}))
    if not str(action.get("toolId", "")).startswith("mcp."):
        for name in ("assemblyIds", "conceptIds", "organic", "completedInTurn", "ponderTrace", "selectionPhase", "selectionStep"):
            arguments.pop(name, None)
    payload = [action.get("kind"), action.get("toolId"), action.get("action"), arguments]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class NativeActionEmissionLedger:
    """Per-turn exactly-once identities; no cross-turn answer/action replay."""

    def __init__(self, turn_id: str):
        self.turn_id = str(turn_id)
        self.actions: Dict[str, Dict[str, Any]] = {}

    def register(self, action: Mapping[str, Any], *, step: int, phase: str) -> Optional[Dict[str, Any]]:
        fingerprint = action_fingerprint(action)
        if action.get("kind") == "ponder":
            # Repeated external effects remain exactly-once, but an internal
            # Ponder choice at a new real prefix is a new computation. Keep
            # replay of that exact decision idempotent, not the whole turn.
            fingerprint = hashlib.sha256(json.dumps(
                [fingerprint, int(step), str(phase)], separators=(",", ":")
            ).encode()).hexdigest()
        if fingerprint in self.actions:
            return None
        result = dict(action)
        result.update({
            "actionId": hashlib.sha256((self.turn_id + ":" + fingerprint).encode()).hexdigest()[:32],
            "selectionStep": int(step),
            "selectionPhase": str(phase),
        })
        self.actions[fingerprint] = result
        return result


def observed_argument_schema(value: Any) -> Dict[str, Any]:
    """Infer only host-observed field/type structure, never argument values."""
    if isinstance(value, Mapping):
        return {"type": "object", "properties": {str(key): observed_argument_schema(item) for key, item in value.items()}, "required": sorted(value), "additionalProperties": False}
    if isinstance(value, (list, tuple)):
        unique = {json.dumps(observed_argument_schema(item), sort_keys=True): observed_argument_schema(item) for item in value}
        items = True if not unique else next(iter(unique.values())) if len(unique) == 1 else {"anyOf": list(unique.values())}
        return {"type": "array", "items": items}
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    return {"type": "unknown"}
