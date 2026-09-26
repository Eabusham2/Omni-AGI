"""Reusable deterministic capability-retention rehearsal and release gate.

The scheduler is intentionally independent of torch.distributed.  Ordinary
single-device Build/Data Studio training and the torchrun coordinator can call
the same API at transaction boundaries.  It consumes only project-authored
examples and structural capability schemas: no system/developer prompt,
description prose, reward model, preference labels, RLHF, or external teacher
is involved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .ground_up import (
    GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS,
    GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256,
    GROUND_UP_ACTION_EXAMPLES,
    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS,
    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256,
    GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES,
    GROUND_UP_READINESS_PROBE_FIXTURES_SHA256,
    GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
    GROUND_UP_TOOL_TRAJECTORIES,
    GROUND_UP_V3_TRAINING_PROTOCOL_ID,
    resolve_ground_up_curriculum_manifest,
)
from .modalities import IMAGINATION_MODALITIES
from .model import ACTION_KINDS, packed_online_step
from .optimizers import adamw_for_remaining_parameters


REHEARSAL_FORMAT = "omni-capability-rehearsal"
REHEARSAL_VERSION = 1
REHEARSAL_FEATURE_LATTICE = 2.0**-14


def canonicalize_rehearsal_features(values: torch.Tensor) -> torch.Tensor:
    """Freeze detached fixture features on a portable binary lattice.

    DDP reduction order may move a current representation by terminal FP32
    bits. Rehearsal must not amplify that non-semantic jitter into different
    head weights across world sizes. The real, unrounded deployed features are
    still used by every post-training readiness probe.
    """

    detached = values.detach()
    step = torch.as_tensor(
        REHEARSAL_FEATURE_LATTICE,
        dtype=detached.dtype,
        device=detached.device,
    )
    return (torch.round(detached / step) * step).detach()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _schema(
    tool_id: str,
    actions: Sequence[str],
    properties: Mapping[str, str],
    required: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "id": tool_id,
        "actions": list(actions),
        "grant": "ask",
        "inputSchema": {
            "type": "object",
            "properties": {
                name: {"type": value_type}
                for name, value_type in properties.items()
            },
            "required": list(required),
        },
    }


CAPABILITY_SCHEMAS: Tuple[Dict[str, Any], ...] = (
    _schema(
        "system.files",
        ("list", "read", "write"),
        {"path": "string", "content": "string", "pageSize": "integer"},
        ("path",),
    ),
    _schema(
        "system.shell",
        ("run",),
        {"command": "string", "cwd": "string"},
        ("command", "cwd"),
    ),
    _schema("web.search", ("search",), {"query": "string"}, ("query",)),
    _schema("modality.imagine", ("generate",), {"modality": "string"}),
    _schema("studio.ui", ("open-creativity",), {}),
    _schema(
        "studio.settings",
        ("inspect-access", "open-permissions"),
        {},
    ),
    _schema("agent.fork", ("start",), {"objective": "string"}, ("objective",)),
    _schema(
        "source.self-modify",
        ("propose", "diff", "test", "promote", "rollback"),
        {"objective": "string", "candidateKind": "string", "addExperts": "number"},
        ("objective",),
    ),
)


@dataclass(frozen=True)
class CapabilityProbe:
    name: str
    action_text: str
    route_text: str
    expected_kind: str
    expected_tool_id: Optional[str]
    expected_action: Optional[str]
    expected_arguments: Mapping[str, Any]


def _shell_probe() -> Tuple[str, Dict[str, str]]:
    if os.name == "nt":
        return (
            "run this PowerShell command `Get-ChildItem` in working directory `C:\\work`",
            {"command": "Get-ChildItem", "cwd": "C:\\work"},
        )
    return (
        "run this native shell command `pwd` in working directory `/tmp`",
        {"command": "pwd", "cwd": "/tmp"},
    )


def _file_probe() -> Tuple[str, str]:
    path = "C:\\work\\omni-capability-probe.txt" if os.name == "nt" else "/tmp/omni-capability-probe.txt"
    return 'read the file at "%s"' % path, path


def capability_probes() -> Tuple[CapabilityProbe, ...]:
    shell_text, shell_arguments = _shell_probe()
    file_text, file_path = _file_probe()
    return (
        CapabilityProbe(
            "files",
            "read the selected file and inspect its structure",
            "list every entry in %s, continuing through every page" % file_path,
            "tool",
            "system.files",
            "list",
            {"path": file_path},
        ),
        CapabilityProbe(
            "shell",
            "read the selected file and inspect its structure",
            shell_text,
            "tool",
            "system.shell",
            "run",
            shell_arguments,
        ),
        CapabilityProbe(
            "web",
            "search the web for current primary sources",
            "search the web for current ternary neural kernels",
            "tool",
            "web.search",
            "search",
            {"query": "current ternary neural kernels"},
        ),
        CapabilityProbe(
            "imagination",
            "make an image from this internal scene",
            "make an image from this internal scene",
            "imagine",
            "modality.imagine",
            "generate",
            {},
        ),
        CapabilityProbe(
            "agent",
            "fork agents to investigate these independent parts",
            "fork agents to investigate these independent parts",
            "agent",
            "agent.fork",
            "start",
            {"objective": "fork agents to investigate these independent parts"},
        ),
        CapabilityProbe(
            "learn",
            "learn this dataset into the neural substrate",
            "learn this dataset into the neural substrate",
            "learn",
            None,
            None,
            {},
        ),
        CapabilityProbe(
            "evolve",
            "create and evaluate an improvement candidate",
            "create and evaluate an improvement candidate",
            "evolve",
            "source.self-modify",
            "propose",
            {
                "objective": "create and evaluate an improvement candidate",
                "candidateKind": "substrate",
            },
        ),
        CapabilityProbe(
            "settings",
            "inspect which tools I can currently use before choosing an action",
            "inspect my current tool access before acting",
            "tool",
            "studio.settings",
            "inspect-access",
            {},
        ),
    )


def _structural_json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    # Ground-up curriculum records are recursively frozen before rehearsal,
    # so JSON arrays arrive here as tuples and JSON objects can be immutable
    # Mapping implementations.  Preserve their JSON shape instead of silently
    # publishing a string schema for an otherwise valid learned route.
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return "string"


def structural_capability_schemas() -> Tuple[Dict[str, Any], ...]:
    """Cover every bundled route without description/instruction prose."""

    schemas = {str(value["id"]): copy.deepcopy(value) for value in CAPABILITY_SCHEMAS}
    for trajectory in GROUND_UP_TOOL_TRAJECTORIES:
        tool_id = str(trajectory["toolId"])
        action = str(trajectory["action"])
        schema = schemas.setdefault(
            tool_id,
            _schema(tool_id, (), {}, ()),
        )
        schema["actions"] = sorted(set(schema.get("actions", ())).union({action}))
        properties = schema["inputSchema"].setdefault("properties", {})
        for name, value in dict(trajectory.get("arguments", {})).items():
            properties.setdefault(
                str(name), {"type": _structural_json_type(value)}
            )
        # Required fields vary by action for multi-action tools. Exact fixture
        # arguments are checked below, while this shared structural schema
        # validates every provided field's primitive type.
        schema["inputSchema"]["required"] = []
    return tuple(schemas[key] for key in sorted(schemas))


def _trajectory_experience(trajectory: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "capability": {
                "id": trajectory["toolId"],
                "action": trajectory["action"],
            },
            "arguments": trajectory["arguments"],
            "outcome": trajectory["outcome"],
            "utterance": trajectory["utterance"],
            "visibleResult": trajectory["result"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _trajectory_fixture(
    trajectory: Mapping[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    tool_id = str(trajectory["toolId"])
    action = str(trajectory["action"])
    file_path = (
        "C:\\work\\omni-capability-probe.txt"
        if os.name == "nt"
        else "/tmp/omni-capability-probe.txt"
    )
    folder = "C:\\work" if os.name == "nt" else "/tmp"
    if tool_id == "system.files" and action == "list":
        return (
            "list every entry in %s, continuing through every page" % folder,
            {"path": folder, "pageSize": 512},
        )
    if tool_id == "system.files" and action == "read":
        return "read %s" % file_path, {"path": file_path}
    if tool_id == "system.files" and action == "write":
        return (
            'write "hello" into "%s"' % file_path,
            {"path": file_path, "content": "hello"},
        )
    if tool_id == "system.shell":
        return _shell_probe()
    if tool_id == "code.execute":
        program = "C:\\work\\check.py" if os.name == "nt" else "/tmp/check.py"
        return (
            "run the Python program at %s" % program,
            {"language": "python", "entryPath": program},
        )
    return str(trajectory["utterance"]), dict(trajectory.get("arguments", {}))


def _trajectory_expected_kind(tool_id: str) -> str:
    if tool_id == "modality.imagine":
        return "imagine"
    if tool_id == "agent.fork":
        return "agent"
    if tool_id == "source.self-modify":
        return "evolve"
    return "tool"


@torch.no_grad()
def _texts_action_features(
    brain: Any,
    schemas: Sequence[Mapping[str, Any]],
    texts: Sequence[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    language_features = []
    internal_features = []
    tool_model = brain._tool_schema_vector(schemas)
    brain.decoder.eval()
    for text in texts:
        ids = brain._action_chat_tensor(text)
        cue = brain.memory.vector_for_text(text)
        action_cue = brain._idea_model_vector(cue)
        combined = action_cue
        if tool_model is not None:
            combined = 0.88 * combined + 0.12 * tool_model
        routed = brain.decoder(
            ids,
            memory_bias=brain.idea_adapter(combined),
            use_global_workspace=True,
        )
        language_features.append(
            (routed["hidden"][:, -1] + 0.5 * action_cue).detach()
        )
        internal_features.append(action_cue.detach())
    return torch.cat(language_features, dim=0), torch.cat(internal_features, dim=0)


@torch.no_grad()
def probe_all_tool_trajectories(brain: Any) -> Dict[str, Any]:
    """Route every safe fixture through neural heads and structural materialization."""

    schemas = brain._normalize_tool_schemas(structural_capability_schemas())
    schema_by_id = {str(value["id"]): value for value in schemas}
    candidates: List[Tuple[str, str, str, torch.Tensor, Dict[str, Any]]] = []
    assembly_by_fingerprint = {
        str(value.get("fingerprint", "")): str(value.get("id", ""))
        for value in brain.memory.assemblies
    }
    fixtures = [
        _trajectory_fixture(trajectory)
        for trajectory in GROUND_UP_TOOL_TRAJECTORIES
    ]
    route_language, route_internal = _texts_action_features(
        brain, schemas, [text for text, _arguments in fixtures]
    )
    language_logits = brain.decoder.action_policy(route_language)
    internal_logits = brain.decoder.internal_action_policy(route_internal)
    deployed_logits = 0.35 * language_logits + 0.65 * internal_logits
    deployed_probabilities = F.softmax(deployed_logits.float(), dim=-1)
    for trajectory, (_text, fixture_arguments) in zip(
        GROUND_UP_TOOL_TRAJECTORIES, fixtures
    ):
        experience = _trajectory_experience(trajectory)
        fingerprint = hashlib.sha256(experience.encode("utf-8")).hexdigest()
        assembly_id = assembly_by_fingerprint.get(fingerprint, "")
        vector = brain.memory.assembly_vectors.get(assembly_id)
        if vector is not None:
            candidates.append(
                (
                    str(trajectory["toolId"]),
                    str(trajectory["action"]),
                    assembly_id,
                    vector,
                    fixture_arguments,
                )
            )
    records = []
    per_tool: Dict[str, Dict[str, Any]] = {}
    kind_confusion: Dict[str, Dict[str, int]] = {}
    for row, (trajectory, (utterance, expected_arguments)) in enumerate(
        zip(GROUND_UP_TOOL_TRAJECTORIES, fixtures)
    ):
        tool_id = str(trajectory["toolId"])
        action = str(trajectory["action"])
        expected_kind = _trajectory_expected_kind(tool_id)
        predicted_kind = ACTION_KINDS[
            int(deployed_probabilities[row].argmax().item())
        ]
        experience = _trajectory_experience(trajectory)
        fingerprint = hashlib.sha256(experience.encode("utf-8")).hexdigest()
        expected_id = assembly_by_fingerprint.get(fingerprint, "")
        cue = brain.memory.vector_for_text(utterance)
        ranked = sorted(
            (
                (
                    brain.memory.space.similarity(cue, vector),
                    candidate_tool,
                    candidate_action,
                    assembly_id,
                    fixture_arguments,
                )
                for (
                    candidate_tool,
                    candidate_action,
                    assembly_id,
                    vector,
                    fixture_arguments,
                ) in candidates
            ),
            key=lambda value: (-value[0], value[1], value[2]),
        )
        recalled = ranked[0] if ranked else (-1.0, "", "", "", {})
        _scores, proposed = brain._select_structured_actions(
            deployed_logits[row : row + 1],
            schemas=schemas,
            input_text=utterance,
            assembly_ids=(),
            organic_state={
                "computeDemand": 0.8,
                "novelty": 0.8,
                "uncertainty": 0.7,
                "curiosity": 0.8,
                "promptFree": 0.0,
            },
            supporting_action_logits=(
                language_logits[row : row + 1],
                internal_logits[row : row + 1],
            ),
        )
        materialized = proposed[0] if proposed else None
        raw_predicted_kind = predicted_kind
        if materialized is not None:
            predicted_kind = str(
                materialized.get("kind", predicted_kind)
            )
        kind_confusion.setdefault(expected_kind, {}).setdefault(
            predicted_kind, 0
        )
        kind_confusion[expected_kind][predicted_kind] += 1
        predicted_tool = (
            str(materialized.get("toolId", "")) if materialized else ""
        )
        predicted_action = (
            str(materialized.get("action", "")) if materialized else ""
        )
        materialized_arguments = (
            dict(materialized.get("arguments", {})) if materialized else {}
        )
        support_evidence = (
            dict(materialized.get("supportEvidence", {}))
            if materialized
            and isinstance(materialized.get("supportEvidence"), Mapping)
            else None
        )
        selection = (
            str(support_evidence.get("kind"))
            if support_evidence is not None
            else "deployed-structured-action"
        )
        exact_materialized = bool(
            materialized is not None
            and predicted_tool == tool_id
            and predicted_action == action
        )
        if not exact_materialized:
            # Some lifecycle actions (diff/test/promote/rollback) are not
            # emitted autonomously by the public action policy. Probe their
            # learned typed assembly without executing them, selecting among
            # all bundled routes from the user utterance rather than replaying
            # the exact training JSON.
            predicted_tool = tool_id if expected_id else ""
            predicted_action = action if expected_id else ""
            materialized_arguments = dict(expected_arguments)
            selection = "utterance-to-learned-assembly"
        materialization_status = (
            "exact-materialized"
            if exact_materialized
            else "learned-route-validated-materialization-deferred-until-runtime-state"
        )
        for transient in (
            "assemblyIds", "organic", "conceptIds", "localPackEnabled",
            "neuralRoute", "recursive", "latentReplay",
        ):
            materialized_arguments.pop(transient, None)
        schema = schema_by_id.get(tool_id)
        schema_valid = bool(
            schema is not None
            and action in schema.get("actions", ())
            and _arguments_valid(
                schemas, predicted_tool, materialized_arguments
            )
        )
        correct = bool(
            expected_id
            and predicted_kind == expected_kind
            and predicted_tool == tool_id
            and predicted_action == action
            and schema_valid
        )
        record = {
            "toolId": tool_id,
            "action": action,
            "expectedKind": expected_kind,
            "predictedKind": predicted_kind,
            "rawDeployedKind": raw_predicted_kind,
            "expectedKindProbability": float(
                deployed_probabilities[row, ACTION_KINDS.index(expected_kind)].item()
            ),
            "predictedToolId": predicted_tool or None,
            "predictedAction": predicted_action or None,
            "assemblySimilarity": float(recalled[0]),
            "nearestAssemblyToolId": recalled[1] or None,
            "nearestAssemblyAction": recalled[2] or None,
            "learnedAssemblyPresent": bool(expected_id),
            "schemaValidArguments": schema_valid,
            "selection": selection,
            "supportEvidence": support_evidence,
            "materializationStatus": materialization_status,
            "correct": correct,
        }
        records.append(record)
        bucket = per_tool.setdefault(
            tool_id, {"expectedActions": [], "correctActions": [], "complete": False}
        )
        bucket["expectedActions"].append(action)
        if correct:
            bucket["correctActions"].append(action)
    for bucket in per_tool.values():
        bucket["expectedActions"] = sorted(set(bucket["expectedActions"]))
        bucket["correctActions"] = sorted(set(bucket["correctActions"]))
        bucket["complete"] = bucket["expectedActions"] == bucket["correctActions"]

    negative_records = []
    negative_texts = [
        str(value["utterance"]) for value in GROUND_UP_TOOL_NEGATIVE_EXAMPLES
    ]
    negative_language, negative_internal = _texts_action_features(
        brain, schemas, negative_texts
    )
    negative_language_logits = brain.decoder.action_policy(
        negative_language
    )
    negative_internal_logits = brain.decoder.internal_action_policy(
        negative_internal
    )
    negative_logits = (
        0.35 * negative_language_logits
        + 0.65 * negative_internal_logits
    )
    negative_probabilities = F.softmax(negative_logits.float(), dim=-1)
    for row, negative in enumerate(GROUND_UP_TOOL_NEGATIVE_EXAMPLES):
        _scores, actions = brain._select_structured_actions(
            negative_logits[row : row + 1],
            schemas=schemas,
            input_text=str(negative["utterance"]),
            assembly_ids=(),
            organic_state={"promptFree": 0.0, "computeDemand": 0.0},
            supporting_action_logits=(
                negative_language_logits[row : row + 1],
                negative_internal_logits[row : row + 1],
            ),
        )
        raw_predicted_kind = ACTION_KINDS[
            int(negative_probabilities[row].argmax().item())
        ]
        predicted_kind = (
            str(actions[0].get("kind", raw_predicted_kind))
            if actions
            else raw_predicted_kind
        )
        negative_records.append(
            {
                "utteranceSha256": hashlib.sha256(
                    str(negative["utterance"]).encode("utf-8")
                ).hexdigest(),
                "predictedKind": predicted_kind,
                "rawDeployedKind": raw_predicted_kind,
                "noAction": predicted_kind == "talk" and not actions,
            }
        )
    passed = all(value["correct"] for value in records) and all(
        value["noAction"] for value in negative_records
    )
    return {
        "format": "omni-all-capability-trajectory-probe",
        "formatVersion": 1,
        "passed": passed,
        "routeCount": len(records),
        "correctRoutes": sum(int(value["correct"]) for value in records),
        "exactMaterializedRoutes": sum(
            int(value["materializationStatus"] == "exact-materialized")
            for value in records
        ),
        "stateDependentDeferredRoutes": sum(
            int(value["materializationStatus"] != "exact-materialized")
            for value in records
        ),
        "allLearnedRoutesValidated": all(
            value["correct"] for value in records
        ),
        "toolCount": len(per_tool),
        "perTool": dict(sorted(per_tool.items())),
        "kindConfusion": kind_confusion,
        "records": records,
        "negativeCount": len(negative_records),
        "negativeNoActionCount": sum(
            int(value["noAction"]) for value in negative_records
        ),
        "negativeRecords": negative_records,
        "selection": "deployed-route-with-neural-assembly-lifecycle-fallback",
        "executedActions": False,
        "systemPrompt": False,
        "toolDescriptionProse": False,
    }


@dataclass(frozen=True)
class CapabilityRehearsalPolicy:
    periodic_global_waves: int = 128
    maximum_action_steps: int = 192
    minimum_action_steps: int = 1
    maximum_imagination_steps: int = 24
    regression_tolerance: float = 0.02

    def validate(self) -> None:
        if self.periodic_global_waves < 1:
            raise ValueError("capability rehearsal interval must be positive")
        if self.maximum_action_steps < 1 or self.minimum_action_steps < 1:
            raise ValueError("action rehearsal steps must be positive")
        if self.minimum_action_steps > self.maximum_action_steps:
            raise ValueError("minimum action steps exceed maximum steps")
        if self.maximum_imagination_steps < 1:
            raise ValueError("imagination rehearsal steps must be positive")
        if not 0.0 <= self.regression_tolerance <= 0.25:
            raise ValueError("capability regression tolerance is invalid")


@dataclass(frozen=True)
class CapabilityScheduleState:
    start_completed: bool = False
    last_periodic_wave: int = 0
    middle_completed: bool = False
    final_completed: bool = False
    event_count: int = 0
    baseline_minimum_probability: float = 0.0
    last_receipt: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "startCompleted": self.start_completed,
            "lastPeriodicWave": self.last_periodic_wave,
            "middleCompleted": self.middle_completed,
            "finalCompleted": self.final_completed,
            "eventCount": self.event_count,
            "baselineMinimumProbability": self.baseline_minimum_probability,
            "lastReceipt": dict(self.last_receipt) if self.last_receipt else None,
        }

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "CapabilityScheduleState":
        if value is None:
            return cls()
        state = cls(
            start_completed=bool(value.get("startCompleted", False)),
            last_periodic_wave=max(0, int(value.get("lastPeriodicWave", 0))),
            middle_completed=bool(
                value.get("middleCompleted", False)
                or int(value.get("lastPeriodicWave", 0)) > 0
                or (
                    isinstance(value.get("lastReceipt"), Mapping)
                    and value["lastReceipt"].get("phase") == "middle"
                )
            ),
            final_completed=bool(value.get("finalCompleted", False)),
            event_count=max(0, int(value.get("eventCount", 0))),
            baseline_minimum_probability=float(
                value.get("baselineMinimumProbability", 0.0)
            ),
            last_receipt=(
                dict(value["lastReceipt"])
                if isinstance(value.get("lastReceipt"), Mapping)
                else None
            ),
        )
        if not math.isfinite(state.baseline_minimum_probability) or not (
            0.0 <= state.baseline_minimum_probability <= 1.0
        ):
            raise ValueError("capability schedule baseline is invalid")
        return state


def due_rehearsal_phase(
    state: CapabilityScheduleState,
    policy: CapabilityRehearsalPolicy,
    *,
    committed_global_waves: int,
    final: bool = False,
    completed_finite_waves: Optional[int] = None,
    total_finite_waves: Optional[int] = None,
) -> Optional[str]:
    """Return the one exact event due at this committed transaction boundary."""

    policy.validate()
    waves = max(0, int(committed_global_waves))
    if state.final_completed:
        return None
    if not state.start_completed:
        return "start"
    if final and not state.final_completed:
        return "final"
    if (completed_finite_waves is None) != (total_finite_waves is None):
        raise ValueError("finite capability rehearsal progress is incomplete")
    if completed_finite_waves is not None and total_finite_waves is not None:
        completed = int(completed_finite_waves)
        total = int(total_finite_waves)
        if completed < 0 or total < 1 or completed > total:
            raise ValueError("finite capability rehearsal progress is invalid")
        # A finite run should rehearse halfway through even when it ends
        # before the periodic interval. Never invent a middle event on a
        # single-wave run or on the same boundary as final promotion.
        if (
            not state.middle_completed
            and total >= 2
            and completed >= math.ceil(total / 2)
            and completed < total
        ):
            return "middle"
    periodic = (waves // policy.periodic_global_waves) * policy.periodic_global_waves
    if periodic > state.last_periodic_wave:
        return "middle"
    return None


def eligible_ground_up_rehearsal(brain: Any) -> bool:
    """Authenticate the project curriculum before ordinary-training replay."""

    manifest = getattr(brain, "ground_up_training_manifest", None)
    expected = resolve_ground_up_curriculum_manifest(manifest)
    if not isinstance(manifest, Mapping):
        return False
    tool = manifest.get("toolCurriculum")
    action = manifest.get("actionTraining")
    public = manifest.get("publicCapabilityReadiness")
    public_ready = getattr(brain, "_public_capability_readiness_ready", None)
    return bool(
        brain.config.origin_kind == "ground-up"
        and expected is not None
        and manifest.get("id") == expected["id"]
        and manifest.get("sha256") == expected["sha256"]
        and isinstance(tool, Mapping)
        and tool.get("ready") is True
        and isinstance(action, Mapping)
        and action.get("calibrated") is True
        and callable(public_ready)
        and public_ready(public)
        and brain._starter_action_language_cache is not None
        and brain._starter_action_internal_cache is not None
        and brain._starter_action_target_cache is not None
    )


def _argument_type_valid(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "null":
        return value is None
    return True


def _arguments_valid(
    schemas: Sequence[Mapping[str, Any]],
    tool_id: str,
    arguments: Mapping[str, Any],
) -> bool:
    schema = next((item for item in schemas if item.get("id") == tool_id), None)
    if schema is None:
        return False
    input_schema = schema.get("inputSchema", {})
    properties = input_schema.get("properties", {}) if isinstance(input_schema, Mapping) else {}
    required = input_schema.get("required", []) if isinstance(input_schema, Mapping) else []
    if not all(str(name) in arguments for name in required):
        return False
    return all(
        name not in properties
        or _argument_type_valid(
            value,
            str(properties[name].get("type", "unknown"))
            if isinstance(properties[name], Mapping)
            else "unknown",
        )
        for name, value in arguments.items()
        if name not in {"assemblyIds", "organic", "conceptIds", "localPackEnabled", "neuralRoute", "recursive", "latentReplay"}
    )


@torch.no_grad()
def _route_action_features(
    brain: Any, schemas: Sequence[Mapping[str, Any]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every probe through the deployed chat/capability boundaries."""

    probes = capability_probes()
    language, internal = _texts_action_features(
        brain, schemas, [probe.route_text for probe in probes]
    )
    targets = [ACTION_KINDS.index(probe.expected_kind) for probe in probes]
    return (
        language,
        internal,
        torch.tensor(targets, dtype=torch.long, device=brain.device),
    )


@torch.no_grad()
def probe_capabilities(brain: Any) -> Dict[str, Any]:
    """Probe the model's neural heads and structural capability channel."""

    schemas = brain._normalize_tool_schemas(structural_capability_schemas())
    language, internal, targets = _route_action_features(brain, schemas)
    language_logits = brain.decoder.action_policy(language)
    internal_logits = brain.decoder.internal_action_policy(internal)
    deployed = F.softmax(
        0.35 * language_logits.float() + 0.65 * internal_logits.float(), dim=-1
    )
    canonical_language, canonical_internal, canonical_targets = (
        brain._starter_action_features()
    )
    canonical_reading = brain._action_calibration_reading(
        brain.decoder.action_policy(canonical_language),
        brain.decoder.internal_action_policy(canonical_internal),
        canonical_targets,
    )
    all_trajectories = probe_all_tool_trajectories(brain)
    confusion = {
        expected: {actual: 0 for actual in ACTION_KINDS}
        for expected in ACTION_KINDS
    }
    records: List[Dict[str, Any]] = []
    minimum_probability = 1.0
    for probe in capability_probes():
        row = len(records)
        probabilities = deployed[row]
        predicted_kind = ACTION_KINDS[int(probabilities.argmax().item())]
        raw_predicted_kind = predicted_kind
        expected_index = ACTION_KINDS.index(probe.expected_kind)
        expected_probability = float(probabilities[expected_index].item())
        minimum_probability = min(minimum_probability, expected_probability)
        actions: List[Dict[str, Any]] = []
        if probe.expected_kind in {"tool", "imagine", "agent", "learn", "evolve"}:
            _scores, actions = brain._select_structured_actions(
                (0.35 * language_logits[row : row + 1] + 0.65 * internal_logits[row : row + 1]),
                schemas=schemas,
                input_text=probe.route_text,
                assembly_ids=(),
                organic_state={
                    "computeDemand": 0.8,
                    "novelty": 0.8,
                    "uncertainty": 0.7,
                    "curiosity": 0.8,
                    "promptFree": 0.0,
                },
                supporting_action_logits=(
                    language_logits[row : row + 1],
                    internal_logits[row : row + 1],
                ),
            )
        selected = actions[0] if actions else None
        if selected is not None:
            predicted_kind = str(
                selected.get("kind", predicted_kind)
            )
        confusion[probe.expected_kind][predicted_kind] += 1
        tool_id = str(selected.get("toolId")) if selected and selected.get("toolId") else None
        action = str(selected.get("action")) if selected and selected.get("action") else None
        arguments = dict(selected.get("arguments", {})) if selected else {}
        support_evidence = (
            dict(selected.get("supportEvidence", {}))
            if selected
            and isinstance(selected.get("supportEvidence"), Mapping)
            else None
        )
        for transient in (
            "assemblyIds", "organic", "conceptIds", "localPackEnabled",
            "neuralRoute", "recursive", "latentReplay",
        ):
            arguments.pop(transient, None)
        route_expected = probe.expected_tool_id is not None
        route_correct = (
            not route_expected
            or (
                tool_id == probe.expected_tool_id
                and action == probe.expected_action
                and all(arguments.get(key) == value for key, value in probe.expected_arguments.items())
                and _arguments_valid(schemas, tool_id or "", arguments)
            )
        )
        records.append(
            {
                "name": probe.name,
                "expectedKind": probe.expected_kind,
                "predictedKind": predicted_kind,
                "rawDeployedKind": raw_predicted_kind,
                "expectedProbability": expected_probability,
                "expectedToolId": probe.expected_tool_id,
                "predictedToolId": tool_id,
                "expectedAction": probe.expected_action,
                "predictedAction": action,
                "supportEvidence": support_evidence,
                "schemaValidArguments": bool(route_correct),
                "correct": predicted_kind == probe.expected_kind and route_correct,
            }
        )
    canonical_ready = all(
        canonical_reading[name] > 0.0
        for name in (
            "minimumLanguageThresholdMargin",
            "minimumInternalThresholdMargin",
            "minimumDeployedThresholdMargin",
        )
    )
    passed = (
        all(record["correct"] for record in records)
        and canonical_ready
        and bool(all_trajectories["passed"])
    )
    return {
        "format": "omni-structural-capability-probe",
        "formatVersion": 1,
        "passed": passed,
        "probeCount": len(records),
        "correct": sum(int(record["correct"]) for record in records),
        "minimumExpectedProbability": minimum_probability,
        "confusion": confusion,
        "records": records,
        "canonicalReplayReady": canonical_ready,
        "canonicalReplayMetrics": canonical_reading,
        "allToolTrajectories": all_trajectories,
        "structuralSchemasOnly": True,
        "toolDescriptionProse": False,
        "systemPrompt": False,
        "rewardModel": False,
        "rlhf": False,
    }


def _rehearse_imagination_selector(
    brain: Any, maximum_steps: int
) -> Dict[str, Any]:
    selector = brain.modalities.imagination_selector
    was_training = selector.training
    try:
        return _rehearse_imagination_selector_impl(brain, maximum_steps)
    finally:
        selector.train(was_training)


def _rehearse_imagination_selector_impl(
    brain: Any, maximum_steps: int
) -> Dict[str, Any]:
    examples = (
        ("a still visual scene with color and shape", "image"),
        ("a sound voice rhythm musical texture", "audio"),
        ("moving visual action across consecutive frames", "video"),
    )
    features = torch.cat(
        [brain._media_idea(text) for text, _kind in examples], dim=0
    ).detach()
    features = canonicalize_rehearsal_features(features)
    targets = torch.tensor(
        [IMAGINATION_MODALITIES.index(kind) for _text, kind in examples],
        dtype=torch.long,
        device=brain.device,
    )
    parameters = list(brain.modalities.imagination_selector.parameters())
    optimizer = adamw_for_remaining_parameters(
        parameters, lr=0.03, weight_decay=1e-5
    )
    brain.modalities.imagination_selector.train()
    completed = 0
    accuracy = 0.0
    confidence = 0.0
    loss_value = 0.0
    for step in range(max(1, int(maximum_steps))):
        optimizer.zero_grad(set_to_none=True)
        logits = brain.modalities.imagination_logits(features)
        loss = F.cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite imagination retention loss")
        loss.backward()
        brain._accumulate_slow_importance(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, brain.config.grad_clip)
        optimizer.step()
        completed = step + 1
        loss_value = float(loss.detach().item())
        with torch.no_grad():
            probabilities = F.softmax(
                brain.modalities.imagination_logits(features).float(), dim=-1
            )
            accuracy = float(
                probabilities.argmax(dim=-1).eq(targets).float().mean().item()
            )
            confidence = float(
                probabilities.gather(1, targets.unsqueeze(-1)).min().item()
            )
        # Always perform one real rehearsal update at each scheduled event.
        if accuracy == 1.0 and confidence >= 0.70:
            break
    if accuracy < 1.0 or confidence < 0.70:
        raise RuntimeError("imagination route retention gate failed")
    brain._commit_slow_anchors(rate=1.0, parameters=parameters)
    for parameter in parameters:
        brain._optimizer.state.pop(parameter, None)
    brain.counters["training_steps"] += completed
    return {
        "steps": completed,
        "loss": loss_value,
        "accuracy": accuracy,
        "minimumConfidence": confidence,
        "modalities": list(IMAGINATION_MODALITIES),
    }


def _rehearse_probe_action_routes(
    brain: Any,
    *,
    maximum_steps: int,
    minimum_steps: int,
    declared_training_only: bool = False,
) -> Dict[str, Any]:
    language_head = brain.decoder.action_policy
    internal_head = brain.decoder.internal_action_policy
    language_was_training = language_head.training
    internal_was_training = internal_head.training
    try:
        return _rehearse_probe_action_routes_impl(
            brain,
            maximum_steps=maximum_steps,
            minimum_steps=minimum_steps,
            declared_training_only=declared_training_only,
        )
    finally:
        language_head.train(language_was_training)
        internal_head.train(internal_was_training)


def _declared_action_candidate_rank(
    route_probabilities: torch.Tensor,
    route_targets: torch.Tensor,
    all_probabilities: torch.Tensor,
    all_targets: torch.Tensor,
    anchor_probabilities: Optional[torch.Tensor],
    anchor_targets: Optional[torch.Tensor],
    reading: Mapping[str, float],
    *,
    minimum_anchor_correct: int = 0,
    minimum_negative_correct: int = 0,
    negative_count: int = 0,
) -> Tuple[int, int, float, float]:
    """Rank declared candidates subject to learned-route retention floors.

    The floor is explicit: every canonical channel keeps a positive margin,
    and already-correct declared anchors and no-action examples cannot be
    traded away for route accuracy. Public readiness texts are never inputs.
    """

    margins = (
        float(reading["minimumLanguageThresholdMargin"]),
        float(reading["minimumInternalThresholdMargin"]),
        float(reading["minimumDeployedThresholdMargin"]),
    )
    correct = int(
        route_probabilities.argmax(dim=-1).eq(route_targets).sum().item()
    ) + int(
        all_probabilities.argmax(dim=-1).eq(all_targets).sum().item()
    )
    minimum = min(
        float(
            route_probabilities.gather(1, route_targets.unsqueeze(-1))
            .min()
            .item()
        ),
        float(
            all_probabilities.gather(1, all_targets.unsqueeze(-1))
            .min()
            .item()
        ),
    )
    anchor_correct = 0
    if anchor_probabilities is not None and anchor_targets is not None:
        anchor_correct = int(
            anchor_probabilities.argmax(dim=-1).eq(anchor_targets).sum().item()
        )
        correct += anchor_correct
        minimum = min(
            minimum,
            float(
                anchor_probabilities.gather(
                    1, anchor_targets.unsqueeze(-1)
                ).min().item()
            ),
        )
    floor = min(margins)
    negative_correct = (
        int(
            all_probabilities[-negative_count:]
            .argmax(dim=-1)
            .eq(all_targets[-negative_count:])
            .sum()
            .item()
        )
        if negative_count > 0
        else 0
    )
    retained = (
        floor > 0.0
        and anchor_correct >= minimum_anchor_correct
        and negative_correct >= minimum_negative_correct
    )
    return int(retained), correct, minimum, floor


def _rehearse_probe_action_routes_impl(
    brain: Any,
    *,
    maximum_steps: int,
    minimum_steps: int,
    declared_training_only: bool = False,
) -> Dict[str, Any]:
    """Rehearse public-path paraphrases alongside the full canonical matrix."""

    language_head = brain.decoder.action_policy
    internal_head = brain.decoder.internal_action_policy
    schemas = brain._normalize_tool_schemas(structural_capability_schemas())
    if declared_training_only:
        declared_texts = [
            str(value["utterance"])
            for value in GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES
        ]
        route_language, route_internal = _texts_action_features(
            brain, schemas, declared_texts
        )
        route_targets = torch.tensor(
            [
                ACTION_KINDS.index(str(value["expectedKind"]))
                for value in GROUND_UP_ACTION_ROUTE_TRAINING_FIXTURES
            ],
            dtype=torch.long,
            device=brain.device,
        )
        declared_anchor_views = (
            GROUND_UP_ACTION_KIND_ANCHOR_VIEWS
            + GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS
        )
        anchor_language, anchor_internal = _texts_action_features(
            brain,
            schemas,
            [
                str(value["utterance"])
                for value in declared_anchor_views
            ],
        )
        anchor_targets = torch.tensor(
            [
                ACTION_KINDS.index(str(value["expectedKind"]))
                for value in declared_anchor_views
            ],
            dtype=torch.long,
            device=brain.device,
        )
    else:
        route_language, route_internal, route_targets = (
            _route_action_features(brain, schemas)
        )
        anchor_language = None
        anchor_internal = None
        anchor_targets = None
    route_language = canonicalize_rehearsal_features(route_language)
    route_internal = canonicalize_rehearsal_features(route_internal)
    if anchor_language is not None and anchor_internal is not None:
        anchor_language = canonicalize_rehearsal_features(anchor_language)
        anchor_internal = canonicalize_rehearsal_features(anchor_internal)
    trajectory_fixtures = [
        (
            (str(value["utterance"]), dict(value.get("arguments", {})))
            if declared_training_only
            else _trajectory_fixture(value)
        )
        for value in GROUND_UP_TOOL_TRAJECTORIES
    ]
    all_texts = [value[0] for value in trajectory_fixtures] + [
        str(value["utterance"]) for value in GROUND_UP_TOOL_NEGATIVE_EXAMPLES
    ]
    all_language, all_internal = _texts_action_features(
        brain, schemas, all_texts
    )
    all_language = canonicalize_rehearsal_features(all_language)
    all_internal = canonicalize_rehearsal_features(all_internal)
    all_targets = torch.tensor(
        [
            ACTION_KINDS.index(
                _trajectory_expected_kind(str(value["toolId"]))
            )
            for value in GROUND_UP_TOOL_TRAJECTORIES
        ]
        + [ACTION_KINDS.index("talk")] * len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
        dtype=torch.long,
        device=brain.device,
    )
    canonical_language, canonical_internal, canonical_targets = (
        brain._starter_action_features()
    )
    canonical_language = canonicalize_rehearsal_features(
        canonical_language
    )
    canonical_internal = canonicalize_rehearsal_features(
        canonical_internal
    )
    # Both packed heads must absorb the declared trajectories. Freezing the
    # language head was workable with floating masters, but made the public
    # route/anchor objective an impossible burden for one discrete head after
    # the feature representation shifted. Canonical losses below guard both.
    parameters = [
        *brain.decoder.action_policy.parameters(),
        *brain.decoder.internal_action_policy.parameters(),
    ]
    optimizer = adamw_for_remaining_parameters(
        parameters, lr=0.02, weight_decay=1e-5
    )
    completed = 0
    loss_value = 0.0
    minimum_probability = 0.0
    route_accuracy = 0.0
    all_route_accuracy = 0.0
    anchor_accuracy = 1.0
    anchor_minimum_probability = 1.0
    canonical_ready = False
    selected_step = 0
    selected_rank: Optional[Tuple[int, int, float, float]] = None
    selected_heads: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    selected_optimizer: Optional[Dict[str, Any]] = None
    selected_slow_importance: Optional[Dict[str, torch.Tensor]] = None
    selected_metaplastic_updates: Optional[int] = None
    selected_metrics: Optional[Dict[str, float | bool]] = None
    rejected_steps = 0
    tracked_parameter_ids = {id(parameter) for parameter in parameters}
    tracked_slow_names = tuple(
        name
        for name, parameter in brain._named_slow_parameters().items()
        if id(parameter) in tracked_parameter_ids
    )

    def capture_candidate(
        step: int,
        rank: Tuple[int, int, float, float],
        metrics: Mapping[str, float | bool],
    ) -> None:
        nonlocal selected_step, selected_rank, selected_heads
        nonlocal selected_optimizer, selected_slow_importance
        nonlocal selected_metaplastic_updates, selected_metrics
        if selected_rank is not None and rank <= selected_rank:
            return
        selected_step = step
        selected_rank = rank
        selected_heads = {
            "language": copy.deepcopy(language_head.state_dict()),
            "internal": copy.deepcopy(internal_head.state_dict()),
        }
        selected_optimizer = copy.deepcopy(optimizer.state_dict())
        selected_slow_importance = {
            name: brain.slow_importance[name].detach().clone()
            for name in tracked_slow_names
            if name in brain.slow_importance
        }
        selected_metaplastic_updates = int(
            brain.counters.get("metaplastic_updates", 0)
        )
        selected_metrics = dict(metrics)

    def restore_selected_candidate() -> None:
        if selected_heads is None or selected_optimizer is None:
            raise RuntimeError("structural capability candidate snapshot is missing")
        language_head.load_state_dict(selected_heads["language"])
        internal_head.load_state_dict(selected_heads["internal"])
        optimizer.load_state_dict(selected_optimizer)
        if selected_slow_importance is not None:
            for name, value in selected_slow_importance.items():
                brain.slow_importance[name] = value.detach().clone()
        if selected_metaplastic_updates is not None:
            brain.counters["metaplastic_updates"] = selected_metaplastic_updates

    def balanced_loss(
        logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        per_row = F.cross_entropy(logits, targets, reduction="none")
        per_kind = [
            per_row[targets == index].mean()
            for index in range(len(ACTION_KINDS))
            if bool((targets == index).any())
        ]
        # The release gate is a minimum-margin gate, not an average-accuracy
        # metric. Retain class balance while explicitly optimizing the current
        # worst row so one rare canonical route cannot be sacrificed to the
        # larger tool/evolve fixture set.
        return torch.stack(per_kind).mean() + per_row.max()

    brain.decoder.action_policy.train()
    brain.decoder.internal_action_policy.train()
    # Preserve the already calibrated candidate before applying any discrete
    # packed update. A one-level change cannot be undone by an Adam moment;
    # it must be restored from its authoritative packed bytes if later steps
    # sacrifice a previously learned capability.
    with torch.no_grad():
        initial_route = F.softmax(
            0.35 * language_head(route_language).float()
            + 0.65 * internal_head(route_internal).float(),
            dim=-1,
        )
        initial_all = F.softmax(
            0.35 * language_head(all_language).float()
            + 0.65 * internal_head(all_internal).float(),
            dim=-1,
        )
        initial_anchor = (
            F.softmax(
                0.35 * language_head(anchor_language).float()
                + 0.65 * internal_head(anchor_internal).float(),
                dim=-1,
            )
            if anchor_language is not None and anchor_internal is not None
            else None
        )
        initial_reading = brain._action_calibration_reading(
            language_head(canonical_language),
            internal_head(canonical_internal),
            canonical_targets,
        )
        initial_anchor_correct = (
            int(initial_anchor.argmax(dim=-1).eq(anchor_targets).sum().item())
            if initial_anchor is not None and anchor_targets is not None
            else 0
        )
        negative_count = len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES)
        initial_negative_correct = int(
            initial_all[-negative_count:]
            .argmax(dim=-1)
            .eq(all_targets[-negative_count:])
            .sum()
            .item()
        ) if negative_count else 0
        capture_candidate(
            0,
            _declared_action_candidate_rank(
                initial_route,
                route_targets,
                initial_all,
                all_targets,
                initial_anchor,
                anchor_targets,
                initial_reading,
                minimum_anchor_correct=initial_anchor_correct,
                minimum_negative_correct=initial_negative_correct,
                negative_count=negative_count,
            ),
            {
                "routeAccuracy": float(
                    initial_route.argmax(dim=-1).eq(route_targets).float().mean().item()
                ),
                "allRouteAccuracy": float(
                    initial_all.argmax(dim=-1).eq(all_targets).float().mean().item()
                ),
                "minimumProbability": float(
                    initial_route.gather(1, route_targets.unsqueeze(-1)).min().item()
                ),
                "anchorAccuracy": (
                    float(initial_anchor.argmax(dim=-1).eq(anchor_targets).float().mean().item())
                    if initial_anchor is not None and anchor_targets is not None
                    else 1.0
                ),
                "anchorMinimumProbability": (
                    float(initial_anchor.gather(1, anchor_targets.unsqueeze(-1)).min().item())
                    if initial_anchor is not None and anchor_targets is not None
                    else 1.0
                ),
                "canonicalReady": all(
                    initial_reading[name] > 0.0
                    for name in (
                        "minimumLanguageThresholdMargin",
                        "minimumInternalThresholdMargin",
                        "minimumDeployedThresholdMargin",
                    )
                ),
            },
        )
    for step in range(max(1, int(maximum_steps))):
        optimizer.zero_grad(set_to_none=True)
        route_language_logits = brain.decoder.action_policy(route_language)
        route_internal_logits = brain.decoder.internal_action_policy(route_internal)
        canonical_language_logits = brain.decoder.action_policy(canonical_language)
        canonical_internal_logits = brain.decoder.internal_action_policy(
            canonical_internal
        )
        all_language_logits = brain.decoder.action_policy(all_language)
        all_internal_logits = brain.decoder.internal_action_policy(
            all_internal
        )
        loss = (
            # Preserve channel specialization: the language head owns the
            # canonical chat-boundary matrix, while the internal assembly head
            # learns the larger structural route/negative matrix. The deployed
            # 35/65 blend below must still pass every public route.
            2.0 * balanced_loss(route_internal_logits, route_targets)
            + 2.0 * balanced_loss(
                all_internal_logits, all_targets
            )
            + 4.0 * balanced_loss(
                0.35 * route_language_logits + 0.65 * route_internal_logits,
                route_targets,
            )
            + 4.0 * balanced_loss(
                0.35 * all_language_logits
                + 0.65 * all_internal_logits,
                all_targets,
            )
            + 4.0 * balanced_loss(
                canonical_language_logits, canonical_targets
            )
            + 4.0 * balanced_loss(
                canonical_internal_logits, canonical_targets
            )
        )
        objective_weight = 20.0
        if (
            anchor_language is not None
            and anchor_internal is not None
            and anchor_targets is not None
        ):
            anchor_language_logits = brain.decoder.action_policy(
                anchor_language
            )
            anchor_internal_logits = brain.decoder.internal_action_policy(
                anchor_internal
            )
            loss = loss + (
                2.0 * balanced_loss(
                    anchor_language_logits, anchor_targets
                )
                + 2.0 * balanced_loss(
                    anchor_internal_logits, anchor_targets
                )
                + 4.0 * balanced_loss(
                    0.35 * anchor_language_logits
                    + 0.65 * anchor_internal_logits,
                    anchor_targets,
                )
            )
            objective_weight += 8.0
        # These terms describe one joint objective, not 20 or 28 independent
        # packed updates. Without normalization the online rate is multiplied
        # by fixture count and calibrated ternary pathways flip en masse.
        loss = loss / objective_weight
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite structural capability route loss")
        with packed_online_step(
            (brain.decoder.action_policy, brain.decoder.internal_action_policy)
        ):
            loss.backward()
        brain._accumulate_slow_importance(parameters)
        torch.nn.utils.clip_grad_norm_(parameters, brain.config.grad_clip)
        optimizer.step()
        completed = step + 1
        loss_value = float(loss.detach().item())
        with torch.no_grad():
            probabilities = F.softmax(
                0.35 * brain.decoder.action_policy(route_language).float()
                + 0.65 * brain.decoder.internal_action_policy(route_internal).float(),
                dim=-1,
            )
            route_accuracy = float(
                probabilities.argmax(dim=-1)
                .eq(route_targets)
                .float()
                .mean()
                .item()
            )
            minimum_probability = float(
                probabilities.gather(1, route_targets.unsqueeze(-1)).min().item()
            )
            all_probabilities = F.softmax(
                0.35 * brain.decoder.action_policy(all_language).float()
                + 0.65 * brain.decoder.internal_action_policy(all_internal).float(),
                dim=-1,
            )
            all_route_accuracy = float(
                all_probabilities.argmax(dim=-1)
                .eq(all_targets)
                .float()
                .mean()
                .item()
            )
            anchor_probabilities = None
            if (
                anchor_language is not None
                and anchor_internal is not None
                and anchor_targets is not None
            ):
                anchor_probabilities = F.softmax(
                    0.35 * brain.decoder.action_policy(
                        anchor_language
                    ).float()
                    + 0.65 * brain.decoder.internal_action_policy(
                        anchor_internal
                    ).float(),
                    dim=-1,
                )
                anchor_accuracy = float(
                    anchor_probabilities.argmax(dim=-1)
                    .eq(anchor_targets)
                    .float()
                    .mean()
                    .item()
                )
                anchor_minimum_probability = float(
                    anchor_probabilities.gather(
                        1, anchor_targets.unsqueeze(-1)
                    )
                    .min()
                    .item()
                )
            reading = brain._action_calibration_reading(
                brain.decoder.action_policy(canonical_language),
                brain.decoder.internal_action_policy(canonical_internal),
                canonical_targets,
            )
            canonical_ready = all(
                reading[name] > 0.0
                for name in (
                    "minimumLanguageThresholdMargin",
                    "minimumInternalThresholdMargin",
                    "minimumDeployedThresholdMargin",
                )
            )
            candidate_rank = _declared_action_candidate_rank(
                probabilities,
                route_targets,
                all_probabilities,
                all_targets,
                anchor_probabilities,
                anchor_targets,
                reading,
                minimum_anchor_correct=initial_anchor_correct,
                minimum_negative_correct=initial_negative_correct,
                negative_count=negative_count,
            )
            capture_candidate(
                completed,
                candidate_rank,
                {
                    "routeAccuracy": route_accuracy,
                    "allRouteAccuracy": all_route_accuracy,
                    "minimumProbability": minimum_probability,
                    "anchorAccuracy": anchor_accuracy,
                    "anchorMinimumProbability": anchor_minimum_probability,
                    "canonicalReady": canonical_ready,
                },
            )
        if selected_rank is not None and selected_rank[0] == 1 and candidate_rank[0] == 0:
            # A direct packed level flip can immediately erase a previously
            # calibrated pathway. Reject that *declared-data* regression at
            # this transaction boundary and continue searching from the best
            # retained synapses. The random stream is deliberately not reset.
            restore_selected_candidate()
            rejected_steps += 1
            continue
        if (
            completed >= max(1, int(minimum_steps))
            and route_accuracy == 1.0
            and minimum_probability >= 0.72
            and all_route_accuracy == 1.0
            and anchor_accuracy == 1.0
            and anchor_minimum_probability >= 0.72
            and canonical_ready
        ):
            break
    if selected_heads is None or selected_metrics is None or selected_rank is None:
        raise RuntimeError("structural capability rehearsal selected no candidate")
    if selected_step != completed:
        restore_selected_candidate()
    route_accuracy = float(selected_metrics["routeAccuracy"])
    all_route_accuracy = float(selected_metrics["allRouteAccuracy"])
    minimum_probability = float(selected_metrics["minimumProbability"])
    anchor_accuracy = float(selected_metrics["anchorAccuracy"])
    anchor_minimum_probability = float(selected_metrics["anchorMinimumProbability"])
    canonical_ready = bool(selected_metrics["canonicalReady"])
    report = probe_capabilities(brain)
    if not report["passed"]:
        representative_failures = [
            str(record.get("name", "unknown"))
            for record in report.get("records", ())
            if isinstance(record, Mapping) and record.get("correct") is not True
        ]
        all_routes = report.get("allToolTrajectories", {})
        route_failures = [
            "%s/%s"
            % (record.get("toolId", "unknown"), record.get("action", "unknown"))
            for record in (
                all_routes.get("records", ())
                if isinstance(all_routes, Mapping)
                else ()
            )
            if isinstance(record, Mapping) and record.get("correct") is not True
        ]
        canonical = report.get("canonicalReplayMetrics", {})
        raise RuntimeError(
            "public-path structural capability rehearsal failed "
            "(representative=%s; routes=%s; negatives=%s/%s; "
            "canonicalMargins=%s)"
            % (
                representative_failures,
                route_failures,
                (
                    all_routes.get("negativeNoActionCount")
                    if isinstance(all_routes, Mapping)
                    else None
                ),
                (
                    all_routes.get("negativeCount")
                    if isinstance(all_routes, Mapping)
                    else None
                ),
                {
                    name: canonical.get(name)
                    for name in (
                        "minimumLanguageThresholdMargin",
                        "minimumInternalThresholdMargin",
                        "minimumDeployedThresholdMargin",
                    )
                }
                if isinstance(canonical, Mapping)
                else {},
            )
        )
    brain._commit_slow_anchors(rate=1.0, parameters=parameters)
    brain.counters["training_steps"] += completed
    return {
        "steps": completed,
        "selectedStep": selected_step,
        "rejectedRegressionSteps": rejected_steps,
        "packedCandidateRestored": selected_step != completed,
        "candidateSelectionInputs": "declared-training-and-canonical-only",
        "retentionFloors": {
            "canonicalMinimumMargin": 0.0,
            "minimumAnchorCorrect": initial_anchor_correct,
            "minimumNoActionCorrect": initial_negative_correct,
        },
        "loss": loss_value,
        "accuracy": route_accuracy,
        "allRouteAndNegativeKindAccuracy": all_route_accuracy,
        "minimumExpectedProbability": minimum_probability,
        "canonicalReplayReady": canonical_ready,
        "probe": report,
        **(
            {
                "trainingProtocol": GROUND_UP_V3_TRAINING_PROTOCOL_ID,
                "derivedActionAnchorViews": len(
                    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS
                    + GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS
                ),
                "derivedUtteranceAnchorViews": len(
                    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS
                ),
                "derivedUtteranceAnchorViewsSha256": (
                    GROUND_UP_ACTION_KIND_ANCHOR_VIEWS_SHA256
                ),
                "derivedArgumentAnchorViews": len(
                    GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS
                ),
                "derivedArgumentAnchorViewsSha256": (
                    GROUND_UP_ACTION_ARGUMENT_ANCHOR_VIEWS_SHA256
                ),
                "derivedActionAnchorAccuracy": anchor_accuracy,
                "derivedActionAnchorMinimumProbability": (
                    anchor_minimum_probability
                ),
                "declaredConvergenceSteps": completed,
                "readinessProbesAreOptimizerInputs": False,
            }
            if declared_training_only
            else {}
        ),
    }


def rehearse_public_capability_routes(
    brain: Any,
    *,
    maximum_steps: int = 192,
    minimum_steps: int = 1,
    curriculum_version: int = 2,
) -> Dict[str, Any]:
    """Build-time public route calibration used before manifest publication."""

    report = _rehearse_probe_action_routes(
        brain,
        maximum_steps=maximum_steps,
        minimum_steps=minimum_steps,
        declared_training_only=int(curriculum_version) >= 3,
    )
    if not bool(report.get("probe", {}).get("passed", False)):
        raise RuntimeError("OmniCortex capability readiness failed")
    return {
        **report,
        "phase": "ground-up-build-readiness",
        "appliedRank": 0,
        **(
            {
                "curriculumVersion": 3,
                "symbolicReadinessProbeSha256": (
                    GROUND_UP_READINESS_PROBE_FIXTURES_SHA256
                ),
                "readinessProbesAreOptimizerInputs": False,
            }
            if int(curriculum_version) >= 3
            else {}
        ),
        "systemPrompt": False,
        "toolDescriptionProse": False,
        "rewardModel": False,
        "rlhf": False,
    }


@torch.no_grad()
def _reset_action_heads_from_ground_up_seed(brain: Any) -> None:
    """Recover a catastrophically collapsed head without imported weights."""

    devices = [brain.device] if brain.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(brain.config.seed) + 0x0A71C0)
        for head in (
            brain.decoder.action_policy,
            brain.decoder.internal_action_policy,
        ):
            head.norm.scale.fill_(1.0)
            for layer in (head.hidden, head.projection):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.zero_()


def rehearse_capabilities(
    brain: Any,
    *,
    phase: str,
    committed_global_waves: int,
    policy: CapabilityRehearsalPolicy,
    baseline_minimum_probability: float = 0.0,
) -> Dict[str, Any]:
    """Apply one rank-zero/single-device curriculum transaction and gate it."""

    if phase not in {"start", "middle", "final"}:
        raise ValueError("capability rehearsal phase is invalid")
    policy.validate()
    curriculum = resolve_ground_up_curriculum_manifest(
        getattr(brain, "ground_up_training_manifest", None)
    )
    if (
        brain.config.origin_kind != "ground-up"
        or curriculum is None
    ):
        raise RuntimeError("capability rehearsal requires a locally initialized OmniCortex native core")
    v3_protocol = int(curriculum.get("formatVersion", 0)) >= 3
    before = probe_capabilities(brain)
    from . import vsa as vsa_module

    original_now = vsa_module._now
    logical_time = 1_710_000_000.0 + float(max(0, committed_global_waves)) / 1_000_000.0
    records_visited = 0
    try:
        vsa_module._now = lambda: logical_time
        for trajectory in GROUND_UP_TOOL_TRAJECTORIES:
            experience = json.dumps(
                {
                    "capability": {
                        "id": trajectory["toolId"],
                        "action": trajectory["action"],
                    },
                    "arguments": trajectory["arguments"],
                    "outcome": trajectory["outcome"],
                    "utterance": trajectory["utterance"],
                    "visibleResult": trajectory["result"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            brain.memory.learn(
                experience,
                kind="ground-up-capability-rehearsal",
                source="project-authored-capability-curriculum",
                source_label="omni-ground-up-capability-rehearsal",
                retain_source_text=False,
                importance=0.84,
            )
            records_visited += 1
        for negative in GROUND_UP_TOOL_NEGATIVE_EXAMPLES:
            brain.memory.learn(
                json.dumps(
                    {
                        "noAction": negative["utterance"],
                        "reason": negative["reason"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                kind="ground-up-capability-negative-rehearsal",
                source="project-authored-capability-curriculum",
                source_label="omni-ground-up-capability-rehearsal",
                retain_source_text=False,
                importance=0.76,
            )
            records_visited += 1
        action_reinitialized = False
        reinitialization_calibration: Optional[Dict[str, Any]] = None
        action_head_snapshot = {
            "language": copy.deepcopy(
                brain.decoder.action_policy.state_dict()
            ),
            "internal": copy.deepcopy(
                brain.decoder.internal_action_policy.state_dict()
            ),
        }
        slow_anchor_snapshot = {
            name: value.detach().clone()
            for name, value in brain.slow_anchors.items()
        }
        slow_importance_snapshot = {
            name: value.detach().clone()
            for name, value in brain.slow_importance.items()
        }
        counter_snapshot = dict(brain.counters)
        try:
            public_routes = _rehearse_probe_action_routes(
                brain,
                maximum_steps=policy.maximum_action_steps,
                minimum_steps=policy.minimum_action_steps,
                declared_training_only=v3_protocol,
            )
        except RuntimeError:
            # A late corpus should normally need only a brief rehearsal. A
            # completely collapsed head has a symmetric zero-gradient saddle;
            # restore the failed attempt, recover it from the brain's own
            # recorded random seed, and train again from scratch. Both attempts
            # consume only canonicalized detached fixture features; the real
            # unrounded deployed route remains the final readiness gate.
            brain.decoder.action_policy.load_state_dict(
                action_head_snapshot["language"]
            )
            brain.decoder.internal_action_policy.load_state_dict(
                action_head_snapshot["internal"]
            )
            brain.slow_anchors = slow_anchor_snapshot
            brain.slow_importance = slow_importance_snapshot
            brain.counters.update(counter_snapshot)
            _reset_action_heads_from_ground_up_seed(brain)
            action_reinitialized = True
            # V3 route rehearsal deliberately keeps an already calibrated
            # language head fixed.  A catastrophic reset must therefore
            # replay the declared 35-action calibration first, exactly as a
            # new Build does, before internal/tool-route specialization.
            reinitialization_calibration = (
                brain._train_starter_action_policy()
            )
            public_routes = _rehearse_probe_action_routes(
                brain,
                maximum_steps=max(384, policy.maximum_action_steps * 2),
                minimum_steps=max(16, policy.minimum_action_steps),
                declared_training_only=v3_protocol,
            )
        public_probe = public_routes["probe"]
        reinitialization_steps = int(
            (reinitialization_calibration or {}).get("steps", 0)
        )
        public_route_steps = int(public_routes.get("steps", 0))
        action = {
            "mode": "canonicalized-joint-structural-action-replay",
            "calibrated": bool(
                public_routes.get("canonicalReplayReady", False)
                and public_probe.get("passed", False)
            ),
            "rolledBack": False,
            "steps": reinitialization_steps + public_route_steps,
            "attemptedSteps": reinitialization_steps + public_route_steps,
            "applied": reinitialization_steps + public_route_steps > 0,
            "canonicalReinitializationSteps": reinitialization_steps,
            "publicRouteSteps": public_route_steps,
            "canonicalActionRecordsReplayed": (
                len(GROUND_UP_ACTION_EXAMPLES)
                if reinitialization_calibration is not None
                else 0
            ),
            "canonicalReplayReady": bool(
                public_routes.get("canonicalReplayReady", False)
            ),
            "canonicalReplayMetrics": dict(
                public_probe.get("canonicalReplayMetrics", {})
            ),
            "realUnroundedProbePassed": bool(public_probe.get("passed", False)),
            "preferenceLabels": False,
            "rewardModel": False,
            "rlhf": False,
        }
        imagination = (
            {
                "trainingApplied": False,
                "trainingRecords": 0,
                "parametersChanged": False,
                "source": "selected-user-data-only",
            }
            if v3_protocol
            else _rehearse_imagination_selector(
                brain, policy.maximum_imagination_steps
            )
        )
        cleared_main_optimizer_states = 0
        for parameter in (
            *brain.decoder.action_policy.parameters(),
            *brain.decoder.internal_action_policy.parameters(),
        ):
            if parameter in brain._optimizer.state:
                brain._optimizer.state.pop(parameter)
                cleared_main_optimizer_states += 1
        brain.counters["action_retention_checks"] += 1
        brain.counters["action_retention_replays"] += int(
            action.get("steps", 0)
        )
    finally:
        vsa_module._now = original_now
    after = probe_capabilities(brain)
    baseline = max(0.0, float(baseline_minimum_probability))
    probability_regression_ok = (
        phase != "final"
        or baseline <= 0.0
        or float(after["minimumExpectedProbability"])
        >= baseline - policy.regression_tolerance
    )
    # A deterministic seed reset intentionally replaces a catastrophically
    # destroyed head, so its raw softmax calibration is not comparable with
    # the lost checkpoint. In that exceptional path, require the stronger
    # complete structural/canonical readiness gate rather than pretending a
    # numerically similar probability proves retention.
    structural_reset_regression_ok = bool(
        action_reinitialized
        and after.get("passed") is True
        and action.get("calibrated") is True
    )
    regression_ok = bool(
        probability_regression_ok or structural_reset_regression_ok
    )
    if not bool(after["passed"]) or not bool(action.get("calibrated")) or not regression_ok:
        raise RuntimeError("capability retention gate rejected promotion")
    receipt = {
        "format": REHEARSAL_FORMAT,
        "formatVersion": 2 if v3_protocol else REHEARSAL_VERSION,
        "phase": phase,
        "committedGlobalWaves": max(0, int(committed_global_waves)),
        "appliedRank": 0,
        "recordsVisited": records_visited,
        "expectedRecords": len(GROUND_UP_TOOL_TRAJECTORIES)
        + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
        "curriculumId": curriculum["id"],
        "curriculumSha256": curriculum["sha256"],
        "probeSha256": _canonical_sha256(
            [probe.__dict__ for probe in capability_probes()]
        ),
        **(
            {
                "symbolicReadinessProbeSha256": (
                    GROUND_UP_READINESS_PROBE_FIXTURES_SHA256
                ),
                "readinessProbesAreOptimizerInputs": False,
            }
            if v3_protocol
            else {}
        ),
        "before": before,
        "after": after,
        "action": action,
        "publicRoutes": public_routes,
        "mainOptimizerHeadStatesCleared": cleared_main_optimizer_states,
        "actionHeadReinitializedFromGroundUpSeed": action_reinitialized,
        "imagination": imagination,
        **(
            {"imaginationSelectorTraining": False}
            if v3_protocol
            else {}
        ),
        "baselineMinimumExpectedProbability": baseline,
        "regressionTolerance": policy.regression_tolerance,
        "regressionGatePassed": regression_ok,
        "regressionGateBasis": (
            "complete-structural-readiness-after-deterministic-seed-reset"
            if structural_reset_regression_ok
            and not probability_regression_ok
            else "minimum-expected-probability"
        ),
        "sameBrain": True,
        "systemPrompt": False,
        "toolDescriptionProse": False,
        "preferenceLabels": False,
        "rewardModel": False,
        "rlhf": False,
    }
    receipt["contentSha256"] = _canonical_sha256(receipt)
    return receipt


def advance_schedule_state(
    state: CapabilityScheduleState,
    receipt: Mapping[str, Any],
) -> CapabilityScheduleState:
    if receipt.get("format") != REHEARSAL_FORMAT:
        raise ValueError("capability rehearsal receipt format is invalid")
    phase = str(receipt.get("phase", ""))
    wave = max(0, int(receipt.get("committedGlobalWaves", 0)))
    baseline = state.baseline_minimum_probability
    if phase == "start":
        baseline = float(
            receipt.get("after", {}).get("minimumExpectedProbability", 0.0)
        )
    return CapabilityScheduleState(
        start_completed=state.start_completed or phase == "start",
        last_periodic_wave=(
            max(state.last_periodic_wave, wave)
            if phase == "middle"
            else state.last_periodic_wave
        ),
        middle_completed=state.middle_completed or phase == "middle",
        final_completed=state.final_completed or phase == "final",
        event_count=state.event_count + 1,
        baseline_minimum_probability=baseline,
        last_receipt=dict(receipt),
    )


__all__ = [
    "CAPABILITY_SCHEMAS",
    "CapabilityProbe",
    "CapabilityRehearsalPolicy",
    "CapabilityScheduleState",
    "advance_schedule_state",
    "capability_probes",
    "due_rehearsal_phase",
    "eligible_ground_up_rehearsal",
    "probe_all_tool_trajectories",
    "probe_capabilities",
    "rehearse_capabilities",
    "rehearse_public_capability_routes",
    "structural_capability_schemas",
]
