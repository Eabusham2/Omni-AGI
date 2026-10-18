"""Versioned declared tool teaching, separate from immutable origin training.

This is native supervised action/route/JSON learning, not an observed successful
host action, a readiness quiz, a reward loop, or an answer lookup.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from contextlib import nullcontext
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from .capability_rollback import CapabilityRollbackSnapshot
from .capability_rehearsal import _texts_action_features
from .connected_capabilities import preserve_connected_feature_activity
from .model import ACTION_KINDS, packed_online_step
from .persistence import tensor_checksum
from .native_action_protocol import validate_structural_value


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_pack(brain, value):
    if not isinstance(value, Mapping):
        raise ValueError("supplementary capability pack must be an object")
    body = {key: item for key, item in value.items() if key != "sha256"}
    if (body.get("format") != "omni-declared-capability-pack" or body.get("version") != 1
            or not isinstance(body.get("id"), str) or not body["id"].strip()
            or body.get("sourceKind") != "project-authored-declared-examples"
            or body.get("executedHostActions") is not False
            or body.get("preferenceLabels") is not False or body.get("rewardModel") is not False
            or canonical_sha(body) != value.get("sha256")):
        raise ValueError("supplementary capability pack declaration failed integrity validation")
    rows = body.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("supplementary capability teaching needs declared records")
    schemas = brain._normalize_tool_schemas(body.get("schemas"))
    by_id = {row["id"]: row for row in schemas}
    for row in rows:
        if (not isinstance(row, Mapping) or set(row) != {"utterance", "kind", "toolId", "action", "arguments"}
                or not isinstance(row["utterance"], str) or not row["utterance"].strip()
                or row["kind"] not in ACTION_KINDS or not isinstance(row["arguments"], Mapping)):
            raise ValueError("invalid declared capability trajectory")
        if row["toolId"] is None and row["action"] is None and row["kind"] == "learn":
            if row["arguments"]:
                raise ValueError("an admitted-experience action cannot grant another source")
            continue
        schema = by_id.get(row["toolId"])
        if schema is None or row["action"] not in schema["actions"]:
            raise ValueError("declared capability route has no structural schema")
        typed = brain._tool_action_input_schema(schema, row["action"])
        if typed is None or not validate_structural_value(row["arguments"], typed):
            raise ValueError("declared capability arguments fail their actual schema")
    return copy.deepcopy(body), schemas


def _learned_checksum(heads):
    values, seen = [], set()
    for head in heads.values():
        for module in head.modules():
            getter = getattr(module, "authoritative_packed_tensors", None)
            if callable(getter):
                for value in getter():
                    if id(value) not in seen:
                        seen.add(id(value)); values.append(value)
    return tensor_checksum(values, chunk_bytes=65536)


@preserve_connected_feature_activity
def _features(brain, schemas, rows):
    return _texts_action_features(brain, schemas, [row["utterance"] for row in rows])


@torch.no_grad()
def _measure(heads, language, internal, labels, routes, arguments):
    actions = .35 * heads["action_policy"](language) + .65 * heads["internal_action_policy"](internal)
    action_losses = F.cross_entropy(actions, labels, reduction="none").tolist()
    route_losses = [float(F.cross_entropy(heads["tool_route_head"].forward_internal(internal[index:index+1]),
        torch.tensor([target], device=internal.device))) for index, target in routes]
    argument_losses = [float(heads["action_argument_head"].supervised_loss(internal[index:index+1], features, args))
        for index, features, args in arguments]
    if not all(math.isfinite(value) for value in [*action_losses, *route_losses, *argument_losses]):
        raise RuntimeError("non-finite supplementary capability measurement")
    return {"actionLosses": action_losses, "routeLosses": route_losses, "argumentLosses": argument_losses}


@torch.no_grad()
def _anchors(heads, width, device, previous_routes):
    """Stream the full numeric basis; do not allocate a width-squared identity."""
    action_rows, route_rows = [], []
    for start in range(0, width, 32):
        end = min(width, start + 32)
        values = torch.zeros((end-start, width), device=device)
        values[torch.arange(end-start, device=device), torch.arange(start, end, device=device)] = 1
        for signs in (values, -values):
            action_rows.append((.35 * heads["action_policy"](signs) + .65 * heads["internal_action_policy"](signs)).softmax(-1).cpu())
            route_rows.append(heads["tool_route_head"].forward_internal(signs, candidate_features=previous_routes).softmax(-1).cpu())
    return torch.cat(action_rows), torch.cat(route_rows)


def _retained(before, after, tolerance):
    return all(len(before[key]) == len(after[key]) and all(new <= old + max(1e-7, abs(old) * tolerance)
        for old, new in zip(before[key], after[key])) for key in before)


def teach_supplementary_capabilities(brain: Any, pack, *, steps=2, cancel_check=None, commit=None):
    body, schemas = validate_pack(brain, pack)
    if type(steps) is not int or steps < 1:
        raise ValueError("declared teaching needs a positive update budget")
    receipts = getattr(brain, "supplementary_capability_receipts", {})
    prior = receipts.get(pack["sha256"])
    if prior is not None:
        if prior.get("packId") != body["id"]:
            raise ValueError("saved supplementary teaching identity changed")
        return {**copy.deepcopy(prior), "duplicate": True}
    names = ("action_policy", "internal_action_policy", "tool_route_head", "action_argument_head")
    heads = {name: getattr(brain.decoder, name) for name in names}
    if any(tuple(head.parameters()) for head in heads.values()):
        raise RuntimeError("supplementary native teaching rejects floating learned Parameters")
    snapshot = CapabilityRollbackSnapshot(brain, heads, label="supplementary declared capability transaction")
    # Route registration replaces small structural buffers. Keep their original
    # owners so a declined candidate restores topology as well as learned bytes.
    structural = [(module, name, value) for head in heads.values() for module in head.modules()
        for name, value in module.named_buffers(recurse=False)
        if name in {"route_keys", "route_features", "grounded_route_updates"}]
    original_receipts = copy.deepcopy(receipts)
    restored = False
    control_lease = None
    def restore():
        nonlocal restored
        for module, name, value in structural:
            module._buffers[name] = value
        snapshot.restore()
        brain.supplementary_capability_receipts = original_receipts
        restored = True
    try:
        rows = body["records"]
        reserve = getattr(getattr(brain, "resource_policy", None), "reserve_ram", None)
        width = int(brain.config.d_model)
        # Full before/after anchor outputs are control evidence, not weight
        # shadows. Refuse before allocation if their actual size cannot fit.
        route_count = int(heads["tool_route_head"].route_features.shape[0])
        control_bytes = 2 * width * (len(ACTION_KINDS) + route_count + 1) * 4 * 3 + len(rows) * width * 32 + 8192
        control_lease = reserve(control_bytes, "supplementary capability features and retention controls") if callable(reserve) else None
        for head in heads.values(): head.eval()
        language, internal = _features(brain, schemas, rows)
        labels = torch.tensor([ACTION_KINDS.index(row["kind"]) for row in rows], dtype=torch.long, device=brain.device)
        old_features = heads["tool_route_head"].route_features.detach().clone()
        before_anchors = _anchors(heads, width, brain.device, old_features)
        learned_before = _learned_checksum(heads)
        by_id = {row["id"]: row for row in schemas}
        routes, arguments = [], []
        for index, row in enumerate(rows):
            if row["toolId"] is None: continue
            target = heads["tool_route_head"].register_route(row["toolId"], row["action"]) + 1
            heads["action_argument_head"].register_route(row["toolId"], row["action"])
            features = heads["action_argument_head"].schema_features(row["toolId"], row["action"], by_id[row["toolId"]])
            routes.append((index, target))
            arguments.append((index, features[None].to(brain.device), row["arguments"]))
        before = _measure(heads, language, internal, labels, routes, arguments)
        protected = []
        if int(heads["tool_route_head"].experience_updates) > 0: protected.append("tool_route_head")
        argument_head = heads["action_argument_head"]
        if int(argument_head.grounded_steps) > 0 or bool((argument_head.grounded_route_updates > 0).any()): protected.append("action_argument_head")
        active = {name: head for name, head in heads.items() if name not in protected}
        unique = {id(module): module for head in active.values() for module in head.modules()}
        scratch = sum(int(getattr(module, "logical_ternary_parameter_count", 0)) for module in unique.values())
        pager_method = getattr(brain.decoder, "_working_pager", None)
        pager = pager_method() if callable(pager_method) else None
        for step in range(steps):
            if cancel_check is not None and cancel_check():
                raise RuntimeError("supplementary teaching paused at a packed update boundary")
            boundary = getattr(brain, "_check_neural_learning_boundary", None)
            if callable(boundary): boundary("supplementary declared capability learning")
            for name, head in heads.items(): head.train(name not in protected)
            if pager is not None: pager.admit_compute(scratch * 5 + control_bytes, "supplementary packed gradients", device=brain.device)
            with pager.activate() if pager is not None else nullcontext():
                terms = [F.cross_entropy(heads["action_policy"](language), labels),
                    F.cross_entropy(heads["internal_action_policy"](internal), labels)]
                if routes and "tool_route_head" not in protected:
                    terms.append(torch.stack([F.cross_entropy(heads["tool_route_head"].forward_internal(internal[index:index+1]),
                        torch.tensor([target], device=brain.device)) for index, target in routes]).mean())
                if arguments and "action_argument_head" not in protected:
                    index, features, args = arguments[step % len(arguments)]
                    terms.append(argument_head.supervised_loss(internal[index:index+1], features, args))
                loss = torch.stack(terms).mean()
                if not bool(torch.isfinite(loss)): raise RuntimeError("non-finite supplementary learning loss")
                with packed_online_step(tuple(active.values()), max_scratch_synapses=max(1, scratch)):
                    loss.backward()
        for head in heads.values(): head.eval()
        candidate = _measure(heads, language, internal, labels, routes, arguments)
        after_anchors = _anchors(heads, width, brain.device, old_features)
        anchor_retained = all(bool(torch.isfinite(new).all()) and bool((old.argmax(-1) == new.argmax(-1)).all()) and
            bool((new.gather(1, old.argmax(-1)[:, None]) + .02 >= old.max(-1).values[:, None]).all()) for old, new in zip(before_anchors, after_anchors))
        learned_candidate = _learned_checksum(heads)
        accepted = _retained(before, candidate, .02) and anchor_retained and learned_candidate != learned_before
        if not accepted: restore()
        else:
            with torch.no_grad():
                if "tool_route_head" not in protected: heads["tool_route_head"].internal_training_steps.add_(steps)
                if "action_argument_head" not in protected: argument_head.training_steps.add_(steps if arguments else 0)
            brain.counters["training_steps"] = int(brain.counters.get("training_steps", 0)) + steps
        receipt = {"format": "omni-supplementary-capability-receipt", "version": 1, "packId": body["id"],
            "packSha256": pack["sha256"], "recordsVisited": len(rows), "attemptedSteps": steps,
            "parameterSteps": steps if accepted else 0, "candidateAccepted": accepted,
            "protectedOwners": protected, "before": before, "candidate": candidate, "after": candidate if accepted else before,
            "anchorRecords": 2 * width, "anchorRetentionPassed": anchor_retained,
            "retentionScope": "complete numeric basis and declared pack rows; not all prior experiences",
            "learnedChecksumBefore": learned_before, "learnedChecksumCandidate": learned_candidate,
            "learnedChecksumAfter": _learned_checksum(heads), "rollbackVerified": not accepted and _learned_checksum(heads) == learned_before,
            "ownerChecksumBefore": snapshot.before_checksum,
            "ownerChecksumAfter": tensor_checksum((value for head in heads.values() for value in head.state_dict().values()), chunk_bytes=65536),
            "initialCurriculumRerun": False, "readinessQuizExecuted": False, "executedHostActions": False,
            "groundingEnabledByDeclaration": False, "sourceKind": body["sourceKind"], "sameBrain": True,
            "preferenceLabels": False, "rewardModel": False, "hiddenPrompt": False}
        receipt["sha256"] = canonical_sha(receipt)
        brain.supplementary_capability_receipts = {**original_receipts, pack["sha256"]: receipt}
        if commit is not None: commit(receipt)
        return {**receipt, "duplicate": False}
    except BaseException:
        restore()
        raise
    finally:
        try: snapshot.restore_modes()
        finally:
            snapshot.close()
            if control_lease is not None: control_lease.release()


def validated_capability_receipts(value):
    if value is None: return {}
    if not isinstance(value, dict): raise ValueError("supplementary capability receipts must be an object")
    for key, receipt in value.items():
        if (not isinstance(receipt, Mapping) or receipt.get("format") != "omni-supplementary-capability-receipt"
                or receipt.get("version") != 1 or key != receipt.get("packSha256")
                or len(key) != 64 or any(char not in "0123456789abcdef" for char in key)
                or canonical_sha({name: item for name, item in receipt.items() if name != "sha256"}) != receipt.get("sha256")
                or receipt.get("executedHostActions") is not False or receipt.get("groundingEnabledByDeclaration") is not False):
            raise ValueError("supplementary capability receipt failed validation")
    return copy.deepcopy(value)
