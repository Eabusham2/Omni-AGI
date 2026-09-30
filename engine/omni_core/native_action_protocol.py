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


from .schema_validation import structural_schema_document, validate_schema_value


def structural_schema(raw: Any, *, max_depth: Optional[int] = None) -> Optional[Dict[str, Any]]:
    # No default product depth cap; references remain structural, not expanded.
    del max_depth
    return structural_schema_document(raw)


def validate_structural_value(value: Any, schema: Mapping[str, Any], *, depth: int = 0) -> bool:
    del depth
    return validate_schema_value(value, schema)


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
