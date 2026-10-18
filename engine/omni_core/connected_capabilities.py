"""Inherited declared-trajectory retention, without a new initial receipt.

This is learning on the existing curriculum, not a readiness quiz, external
action execution, a generated-answer lookup, or evidence of broad competence.
Only the joined action/route/argument owners participate in the transaction.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import contextmanager
from functools import wraps
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .ground_up import (GROUND_UP_ACTION_EXAMPLES, GROUND_UP_TOOL_TRAJECTORIES,
                        GROUND_UP_TOOL_NEGATIVE_EXAMPLES, resolve_ground_up_curriculum_manifest)
from .persistence import tensor_checksum


@contextmanager
def preserved_feature_activity(brain):
    """Keep curriculum feature reads out of the live attention/activity state."""
    from .connected_regions import NativeRegion

    modes = [(module, module.training) for module in brain.decoder.modules()]
    controls = {}
    for owner in (brain, *(module for module in brain.decoder.modules() if isinstance(module, NativeRegion))):
        values = [getattr(owner, "liquid_state", None)]
        population = getattr(getattr(owner, "router", None), "population", None)
        values.extend(getattr(population, name, None) for name in ("membrane", "spike_count"))
        for value in values:
            if isinstance(value, torch.Tensor):
                controls.setdefault(id(value), value)
    reserve = getattr(getattr(brain, "resource_policy", None), "reserve_ram", None)
    lease = reserve(sum(value.numel() * value.element_size() for value in controls.values()) + 8192,
                    "training feature activity preservation") if callable(reserve) else None
    try:
        snapshots = [(value, value.detach().cpu().clone()) for value in controls.values()]
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    missing = object()
    memories = {id(brain.memory): brain.memory}
    memories.update((id(memory), memory) for _name, memory in brain._connected_regions.iter_memories())
    audits = [(memory, "_last_recall_audit", getattr(memory, "_last_recall_audit", missing)) for memory in memories.values()]
    activity = [(module, "last_region_activity", getattr(module, "last_region_activity", missing)) for module in brain.decoder.modules()
                if hasattr(module, "last_region_activity")]
    # Feature generation does not call workspace admission or lifecycle
    # mutation. Retain these exact existing owners if a helper replaces one;
    # do not copy a potentially disk-backed entire user context into RAM.
    workspace = {name: getattr(brain, name) for name in ("working_memory", "workspace_items",
        "recent_token_context", "recent_token_activity", "current_context", "memory_lifecycle") if hasattr(brain, name)}
    try:
        yield
    finally:
        with torch.no_grad():
            for value, saved in snapshots:
                value.copy_(saved.to(value))
        for owner, name, old in [*audits, *activity]:
            if old is missing:
                if hasattr(owner, name):
                    delattr(owner, name)
            else:
                setattr(owner, name, old)
        for name, owner in workspace.items():
            setattr(brain, name, owner)
        for module, mode in modes:
            module.training = mode
        if lease is not None:
            lease.release()


def preserve_connected_feature_activity(function):
    @wraps(function)
    def wrapped(brain, *args, **kwargs):
        if getattr(brain, "_connected_regions", None) is None:
            return function(brain, *args, **kwargs)
        with preserved_feature_activity(brain):
            return function(brain, *args, **kwargs)
    return wrapped


def _samples(brain):
    from .capability_rehearsal import structural_capability_schemas

    language, internal, targets = brain._starter_action_features()
    route_head = brain.decoder.tool_route_head
    route_states, route_targets, arguments = [], [], []
    schemas = {row["id"]: row for row in brain._normalize_tool_schemas(structural_capability_schemas())}
    for row in GROUND_UP_TOOL_TRAJECTORIES:
        index = route_head.route_index(row["toolId"], row["action"])
        if index is None:
            raise ValueError("inherited curriculum lost a registered route")
        state = brain._idea_model_vector(brain.memory.vector_for_text(row["utterance"])).detach().reshape(1, -1)
        route_states.append(state)
        route_targets.append(index + 1)
        schema = schemas.get(row["toolId"])
        argument_head = brain.decoder.action_argument_head
        features = argument_head.schema_features(row["toolId"], row["action"], schema) if schema else None
        encoded = json.dumps(row["arguments"], ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
        if features is not None and len(encoded) <= int(brain.config.max_seq_len):
            arguments.append((state, features[None].to(brain.device), row["arguments"]))
    for row in GROUND_UP_TOOL_NEGATIVE_EXAMPLES:
        route_states.append(brain._idea_model_vector(brain.memory.vector_for_text(row["utterance"])).detach().reshape(1, -1))
        route_targets.append(0)
    return (language, internal, targets, torch.cat(route_states, 0),
            torch.tensor(route_targets, dtype=torch.long, device=brain.device), arguments)


@torch.no_grad()
def _measure(brain, samples):
    language, internal, targets, routes, route_targets, arguments = samples
    language_logits = brain.decoder.action_policy(language)
    internal_logits = brain.decoder.internal_action_policy(internal)
    logits = 0.35 * language_logits + 0.65 * internal_logits
    route_logits = brain.decoder.tool_route_head.forward_internal(routes)
    argument_losses = [float(brain.decoder.action_argument_head.supervised_loss(*row)) for row in arguments]
    result = {
        "actionLoss": float(F.cross_entropy(language_logits, targets) + F.cross_entropy(internal_logits, targets)),
        "routeLoss": float(F.cross_entropy(route_logits, route_targets)),
        "argumentLoss": sum(argument_losses) / max(1, len(argument_losses)),
        "correctActions": int((logits.argmax(-1) == targets).sum()),
        "correctRoutes": int((route_logits.argmax(-1) == route_targets).sum()),
        "minimumExpectedProbability": float(logits.softmax(-1).gather(1, targets[:, None]).min()),
        "actionRecords": int(targets.numel()), "routeRecords": int(route_targets.numel()),
        "argumentRecords": len(arguments),
    }
    if not all(math.isfinite(float(value)) for value in result.values()):
        raise RuntimeError("non-finite inherited capability measurements")
    return result


@torch.no_grad()
def _retention_rows(brain, samples):
    """Keep a stronger row from hiding another row's regression in a mean."""
    language, internal, targets, routes, route_targets, arguments = samples
    action_logits = 0.35 * brain.decoder.action_policy(language) + 0.65 * brain.decoder.internal_action_policy(internal)
    route_logits = brain.decoder.tool_route_head.forward_internal(routes)
    return {
        "actionCorrect": action_logits.argmax(-1).eq(targets).tolist(),
        "routeCorrect": route_logits.argmax(-1).eq(route_targets).tolist(),
        "actionTargetProbabilities": action_logits.softmax(-1).gather(1, targets[:, None]).flatten().tolist(),
        "routeTargetProbabilities": route_logits.softmax(-1).gather(1, route_targets[:, None]).flatten().tolist(),
        "argumentLosses": [float(brain.decoder.action_argument_head.supervised_loss(*row)) for row in arguments],
    }


def _rows_retained(before, after, tolerance):
    return (all(len(before[key]) == len(after[key]) and
                all(not old or new for old, new in zip(before[key], after[key]))
                for key in ("actionCorrect", "routeCorrect"))
        and all(len(before[key]) == len(after[key]) and
                all(math.isfinite(new) and new + tolerance >= old for old, new in zip(before[key], after[key]))
                for key in ("actionTargetProbabilities", "routeTargetProbabilities"))
        and len(before["argumentLosses"]) == len(after["argumentLosses"])
        and all(math.isfinite(new) and new <= old + max(1e-7, old * tolerance)
                for old, new in zip(before["argumentLosses"], after["argumentLosses"])))


def rehearse_inherited_capabilities(brain, *, phase, committed_global_waves, policy):
    """Replay real joined owners; reject a regression without blocking chat.

    Origin validation occurs at load against unchanged private snapshots.
    The current root intentionally never acquires a ground-up birth manifest.
    """
    from .capability_rehearsal import REHEARSAL_FORMAT, eligible_ground_up_rehearsal

    if not eligible_ground_up_rehearsal(brain):
        raise RuntimeError("connected rehearsal has no authenticated inherited curriculum")
    inherited = brain._inherited_capability_curriculum
    curriculum = resolve_ground_up_curriculum_manifest(inherited["curriculum"])
    if curriculum is None:
        raise RuntimeError("inherited curriculum is no longer supported")
    names = ("action_policy", "internal_action_policy", "tool_route_head", "action_argument_head")
    heads = {name: getattr(brain.decoder, name) for name in names}
    if any(tuple(head.parameters()) for head in heads.values()):
        raise RuntimeError("connected capability replay cannot introduce floating learned head parameters")
    from .capability_rollback import CapabilityRollbackSnapshot
    snapshot = CapabilityRollbackSnapshot(brain, heads, label="connected capability retention transaction")
    restored = False

    def restore():
        nonlocal restored
        snapshot.restore()
        restored = True

    try:
        for head in heads.values():
            head.eval()
        samples = _samples(brain)
        before = _measure(brain, samples)
        before_rows = _retention_rows(brain, samples)
        before_checksum = snapshot.before_checksum
        # A bounded pass over declared data, not optimization until a quiz
        # becomes perfect. Low-capacity brains remain truthfully primitive.
        budget = min(32, int(policy.maximum_action_steps))
        action = brain._calibrate_starter_action_policy(max_steps=budget,
            minimum_steps=min(budget, int(policy.minimum_action_steps)), strict=False,
            _feature_batch=samples[:3])
        protected = []
        if int(heads["tool_route_head"].experience_updates) > 0:
            protected.append("tool_route_head")
            route = {"steps": 0, "applied": False, "reason": "confirmed-experience-owner-protected"}
        else:
            route = brain._train_tool_route_head(GROUND_UP_TOOL_TRAJECTORIES,
                negative_examples=GROUND_UP_TOOL_NEGATIVE_EXAMPLES, maximum_steps=budget)
        if (int(heads["action_argument_head"].grounded_steps) > 0
                or bool((heads["action_argument_head"].grounded_route_updates > 0).any())):
            protected.append("action_argument_head")
            argument = {"steps": 0, "applied": False, "reason": "confirmed-experience-owner-protected"}
        else:
            argument = brain._train_action_argument_head(GROUND_UP_TOOL_TRAJECTORIES, grounded=False)
        for head in heads.values():
            head.eval()
        candidate = _measure(brain, samples)
        candidate_rows = _retention_rows(brain, samples)
        candidate_checksum = tensor_checksum(value for head in heads.values() for value in head.state_dict().values())
        retained = (_rows_retained(before_rows, candidate_rows, policy.regression_tolerance)
                    and all(candidate[key] <= before[key] + max(1e-7, before[key] * policy.regression_tolerance)
                        for key in ("actionLoss", "routeLoss", "argumentLoss"))
                    and candidate["correctActions"] >= before["correctActions"]
                    and candidate["correctRoutes"] >= before["correctRoutes"])
        if not retained or action.get("rolledBack"):
            restore()
        after = before if restored else candidate
        steps = int(action.get("steps", 0)) + int(route.get("steps", 0)) + int(argument.get("steps", 0))
        receipt = {
            "format": REHEARSAL_FORMAT, "formatVersion": 3, "phase": phase,
            "committedGlobalWaves": int(committed_global_waves), "appliedRank": 0,
            "curriculumId": curriculum["id"], "curriculumSha256": curriculum["sha256"],
            "inheritedOrigins": copy.deepcopy(inherited["origins"]),
            "recordsVisited": len(GROUND_UP_ACTION_EXAMPLES) + len(GROUND_UP_TOOL_TRAJECTORIES) + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
            "before": before, "candidate": candidate, "after": after,
            "retentionRowsBefore": before_rows, "retentionRowsCandidate": candidate_rows,
            "ownerChecksumBefore": before_checksum, "ownerChecksumCandidate": candidate_checksum,
            "ownerChecksumAfter": before_checksum if restored else candidate_checksum,
            "action": {**action, "steps": 0 if restored else int(action.get("steps", 0)),
                       "applied": not restored and steps > 0, "rolledBack": restored},
            "route": route, "argument": argument,
            "protectedOwners": protected,
            "parameterSteps": 0 if restored else steps, "candidateAccepted": not restored,
            "regressionGatePassed": True, "regressionGateBasis": "declared-trajectories-or-exact-owner-rollback",
            "regressionTolerance": policy.regression_tolerance,
            "initialCurriculumRerun": False, "readinessProbeExecuted": False,
            "capabilityOutcomeVerified": False, "executedActions": False, "sameBrain": True,
            "systemPrompt": False, "toolDescriptionProse": False,
            "preferenceLabels": False, "rewardModel": False, "rlhf": False,
        }
        receipt["contentSha256"] = hashlib.sha256(json.dumps(receipt, sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        return receipt
    except BaseException:
        restore()
        raise
    finally:
        try:
            snapshot.restore_modes()
        finally:
            snapshot.close()
