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
from .ponder_episode import NativePonderEpisode


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
        for name in ("assemblyIds", "conceptIds", "assemblyIdView", "conceptIdView", "sourceTurnId", "organic", "handledInTurn", "completedInTurn", "ponderTrace", "ponderSelfReport", "_ponderSelfReportCandidate", "selectionPhase", "selectionStep"):
            arguments.pop(name, None)
    payload = [action.get("kind"), action.get("toolId"), action.get("action"), arguments]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class NativeActionEmissionLedger:
    """Per-turn exactly-once identities; no cross-turn answer/action replay."""

    def __init__(self, turn_id: str):
        self.turn_id = str(turn_id)
        self.actions: Dict[str, Dict[str, Any]] = {}
        self.effect_fingerprints = set()
        self.active_ponder_action_id = None
        self.stop_selected = False

    def register(self, action: Mapping[str, Any], *, step: int, phase: str,
                 parent_ponder_action_id=None) -> Optional[Dict[str, Any]]:
        if parent_ponder_action_id is not None:
            if (parent_ponder_action_id != self.active_ponder_action_id or action.get("kind") == "ponder"
                    or not any(value.get("actionId") == parent_ponder_action_id and value.get("kind") == "ponder"
                               for value in self.actions.values())):
                raise ValueError("selected action has no active Ponder invocation")
        fingerprint = action_fingerprint(action)
        if action.get("kind") == "ponder":
            # Repeated external effects remain exactly-once, but an internal
            # Ponder choice at a new real prefix is a new computation. Keep
            # replay of that exact decision idempotent, not the whole turn.
            fingerprint = hashlib.sha256(json.dumps(
                [fingerprint, int(step), str(phase)], separators=(",", ":")
            ).encode()).hexdigest()
        if fingerprint in self.actions or action.get("kind") != "ponder" and fingerprint in self.effect_fingerprints:
            return None
        result = dict(action)
        result.pop("parentPonderActionId", None)
        result.update({
            "actionId": hashlib.sha256((self.turn_id + ":" + fingerprint).encode()).hexdigest()[:32],
            "selectionStep": int(step),
            "selectionPhase": str(phase),
        })
        if parent_ponder_action_id is not None:
            result["parentPonderActionId"] = parent_ponder_action_id
        self.actions[fingerprint] = result
        if action.get("kind") == "stop": self.stop_selected = True
        if action.get("kind") != "ponder": self.effect_fingerprints.add(fingerprint)
        return result

    def ponder_episode(self, action, emit=None):
        return NativePonderEpisode(self, action, emit)

    def suspension_state(self):
        if self.active_ponder_action_id is not None:
            raise ValueError("active Ponder invocation is not a token suspension frontier")
        return {"turnId": self.turn_id, "actions": dict(self.actions)}

    @classmethod
    def from_suspension_state(cls, value, turn_id):
        if not isinstance(value, dict) or set(value) != {"turnId", "actions"} or value["turnId"] != turn_id or not isinstance(value["actions"], dict):
            raise ValueError("invalid native action continuation owner")
        result = cls(turn_id)
        for fingerprint, action in value["actions"].items():
            if not isinstance(fingerprint, str) or len(fingerprint) != 64 or not isinstance(action, dict):
                raise ValueError("invalid native action continuation identity")
            expected = hashlib.sha256((turn_id + ":" + fingerprint).encode()).hexdigest()[:32]
            if action.get("actionId") != expected:
                raise ValueError("native action continuation lost its original effect identity")
            result.actions[fingerprint] = dict(action)
            if action.get("kind") == "stop": result.stop_selected = True
            if action.get("kind") != "ponder": result.effect_fingerprints.add(action_fingerprint(action))
        for action in result.actions.values():
            parent = action.get("parentPonderActionId")
            if parent is not None and (action.get("kind") == "ponder" or not any(
                    value.get("actionId") == parent and value.get("kind") == "ponder" for value in result.actions.values())):
                raise ValueError("native action continuation lost its Ponder parent")
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
