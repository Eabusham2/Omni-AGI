"""Reviewed JSON Schema validation; no model, prompt, or implicit URL fetch."""
import math
from collections.abc import Mapping
from functools import lru_cache
import json

from jsonschema import FormatChecker, validators
from jsonschema.exceptions import SchemaError
from jsonschema_specifications import REGISTRY as SPECIFICATIONS
from referencing import Registry
from referencing.exceptions import Unresolvable

from .offload import NeuralStateResourcePause

_PROSE = {"title", "description", "examples", "default", "$comment", "deprecated", "readOnly", "writeOnly"}
_MAPS = {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}
_LISTS = {"allOf", "anyOf", "oneOf", "prefixItems"}
_SCHEMAS = {"items", "additionalItems", "additionalProperties", "unevaluatedItems",
            "unevaluatedProperties", "contains", "propertyNames", "not", "if", "then", "else"}
_HOST_SAFE_INTEGER = (1 << 53) - 1


def _host_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, int):
        # The JS host cannot carry a larger JSON integer exactly.
        return -_HOST_SAFE_INTEGER <= value <= _HOST_SAFE_INTEGER
    return math.isfinite(value)


def _host_integer(value):
    return _host_number(value) and abs(value) <= _HOST_SAFE_INTEGER and (
        isinstance(value, int) or value.is_integer()
    )


def structural_schema_document(raw):
    """Preserve references/assertions while stripping schema annotation prose.

    References are not expanded, so recursive schemas stay finite. Instance
    literals (const/enum) and property names are never mistaken for annotations.
    The existing closed-native-object rule remains only for simple undeclared
    object schemas; explicit schema dialects keep their standard semantics.
    """
    if isinstance(raw, bool): return {} if raw else {"not": {}}
    if not isinstance(raw, Mapping): return None
    result = {}
    standard = "$schema" in raw
    tasks = [(raw, result, "schema")]
    while tasks:
        source, target, context = tasks.pop()
        entries = source.items() if isinstance(source, Mapping) else enumerate(source)
        for key, value in entries:
            if context == "schema" and key in _PROSE: continue
            child_context = "value"
            if context == "schema-map": child_context = "schema"
            elif context == "schema-list": child_context = "schema"
            elif context == "schema":
                if key in _MAPS: child_context = "schema-map"
                elif key in _LISTS or (key == "items" and isinstance(value, list)): child_context = "schema-list"
                elif key in _SCHEMAS: child_context = "schema"
                elif key == "dependencies": child_context = "schema-map"
            if isinstance(value, Mapping):
                child = {}
                if isinstance(target, list): target.append(child)
                else: target[key] = child
                tasks.append((value, child, child_context))
            elif isinstance(value, (list, tuple)):
                child = []
                if isinstance(target, list): target.append(child)
                else: target[key] = child
                tasks.append((value, child, child_context))
            else:
                if isinstance(target, list): target.append(value)
                else: target[key] = value
        if context == "schema" and not standard and "properties" in source and \
                not any(name in source for name in ("additionalProperties", "patternProperties", "unevaluatedProperties",
                    "allOf", "anyOf", "oneOf", "$ref", "$dynamicRef")):
            target["additionalProperties"] = False
    return result


@lru_cache(maxsize=128)
def _validator(serialized):
    schema = json.loads(serialized)
    kind = validators.validator_for(schema, default=None if "$schema" in schema else validators.Draft202012Validator)
    if kind is None: raise SchemaError("The declared JSON Schema dialect is unavailable.")
    kind.check_schema(schema)
    checker = kind.TYPE_CHECKER.redefine("object", lambda _check, value: isinstance(value, Mapping)) \
        .redefine("array", lambda _check, value: isinstance(value, (list, tuple))) \
        .redefine("number", lambda _check, value: _host_number(value)) \
        .redefine("integer", lambda _check, value: _host_integer(value))
    checked = validators.extend(kind, type_checker=checker)
    # No retrieve callback: unprovided remote schemas cannot trigger network,
    # file access, credentials, or another hidden tool invocation.
    registry = SPECIFICATIONS.combine(Registry())
    return checked(schema, registry=registry, format_checker=FormatChecker())


def validate_schema_value(value, schema):
    if not isinstance(schema, Mapping) or schema.get("unsupported"): return False
    try:
        serialized = json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return _validator(serialized).is_valid(value)
    except (SchemaError, Unresolvable, ValueError, TypeError, OverflowError):
        return False
    except RecursionError as error:
        # Physical interpreter stack, not a product-defined depth or silent
        # weakening of validation. The owning operation can recover/reschedule.
        raise NeuralStateResourcePause("schema validation needs an admitted deeper interpreter stack",
            {"paused": True, "recoverable": True, "stage": "tool-schema-validation", "validationTruncated": False}) from error
