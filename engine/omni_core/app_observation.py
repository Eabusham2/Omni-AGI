"""Exactly-once ordinary learning of reviewed connected-App observations.

The host owns immutable review/audit blobs and strict queue claims. Native
state keeps only one bounded committed frontier, atomically beside its neural
generation. This is not a chat turn, tool-result inbox or source-text lookup.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone


EMPTY_FRONTIER = "0" * 64
HEAD_FORMAT = "omni-native-app-observation-head"
FRONTIER_FORMAT = "omni-native-app-observation-frontier"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_HASH = re.compile(r"[a-f0-9]{64}\Z")
_MAX_INTEGER = (1 << 53) - 1


def _text(value, *, identifier=False, nonempty=False):
    if not isinstance(value, str) or "\0" in value:
        raise ValueError("App observation text or ownership is invalid")
    value.encode("utf-8", errors="strict")
    if identifier and not _ID.fullmatch(value) or nonempty and not value:
        raise ValueError("App observation text or ownership is invalid")
    return value


def _hash(value):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError("App observation SHA-256 is invalid")
    return value


def _integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= _MAX_INTEGER:
        raise ValueError("App observation count is invalid")
    return value


def _instant(value):
    _text(value, nonempty=True)
    if len(value) > 64:
        raise ValueError("App observation timestamp is invalid")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("App observation timestamp requires an explicit timezone")
    return value


def raw_sha256(raw):
    return hashlib.sha256(_text(raw).encode("utf-8")).hexdigest()


def _json_object(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("App observation JSON contains duplicate fields")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("App observation JSON contains a non-finite number")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("App observation JSON must be an object")
    return result


def validate_frontier_head(value):
    if value is None:
        return None
    keys = {"format", "version", "streamId", "ordinal", "recordSha256", "dataSha256", "previousSha256",
            "baseFrontierSha256", "disposition", "state", "committedAt", "learning"}
    if (not isinstance(value, dict) or set(value) - {"content"} != keys or value["format"] != HEAD_FORMAT
            or type(value["version"]) is not int or value["version"] != 1):
        raise ValueError("Native App observation frontier format is invalid")
    _text(value["streamId"], identifier=True)
    _integer(value["ordinal"], 1)
    for name in ("recordSha256", "dataSha256", "previousSha256", "baseFrontierSha256"):
        _hash(value[name])
    _instant(value["committedAt"])
    observed = value["disposition"] == "observe" and value["state"] == "observed"
    superseded = value["disposition"] == "superseded" and value["state"] == "advanced-without-observation"
    if not (observed or superseded):
        raise ValueError("Native App observation disposition is invalid")
    learning = value["learning"]
    if superseded:
        if learning is not None:
            raise ValueError("Superseded App data cannot claim neural learning")
    else:
        if (not isinstance(learning, dict) or set(learning) != {"kind", "source", "sourceLabel", "steps",
                "experienceCountBefore", "experienceCountAfter", "assemblyId", "afterimageId", "textSha256", "textBytes"}
                or learning["kind"] != "experience" or learning["source"] != "mcp-app"
                or type(learning["steps"]) is not int or learning["steps"] != 0):
            raise ValueError("Native App observation learning receipt is invalid")
        _text(learning["sourceLabel"], nonempty=True)
        if len(learning["sourceLabel"]) > 512:
            raise ValueError("Native App observation source label is invalid")
        _text(learning["assemblyId"], identifier=True)
        _text(learning["afterimageId"])
        if learning["afterimageId"]:
            _text(learning["afterimageId"], identifier=True)
        _hash(learning["textSha256"])
        _integer(learning["textBytes"], 0 if "content" in value else 1)
        before, after = (_integer(learning[name]) for name in ("experienceCountBefore", "experienceCountAfter"))
        if after != before + 1:
            raise ValueError("Native App observation must account for exactly one ordinary experience")
    if "content" in value:
        from .app_observation_audit import validate_content
        if not observed:
            raise ValueError("Superseded App metadata cannot claim sensory content admission")
        content = validate_content(value["content"], record_sha=value["recordSha256"], data_sha=value["dataSha256"])
        if learning["textBytes"] == 0 and not content["summary"]["nativeSensory"]:
            raise ValueError("Empty non-sensory App content cannot claim a learned experience")
    return copy.deepcopy(value)


def frontier_sha256(head):
    if head is None:
        return EMPTY_FRONTIER
    head = validate_frontier_head(head)
    learning = head["learning"]
    values = [head[name] for name in ("format", "version", "streamId", "ordinal", "recordSha256", "dataSha256",
                                    "previousSha256", "baseFrontierSha256", "disposition", "state", "committedAt")]
    values.append(None if learning is None else [learning[name] for name in ("kind", "source", "sourceLabel", "steps",
        "experienceCountBefore", "experienceCountAfter", "assemblyId", "afterimageId", "textSha256", "textBytes")])
    if "content" in head:
        from .app_observation_audit import content_hash_values
        values.append(content_hash_values(head["content"]))
    return raw_sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False))


def native_app_observation_frontier(brain):
    head = validate_frontier_head(getattr(brain, "app_observation_frontier", None))
    if head is not None and head.get("content"):
        from .app_observation_audit import verify_audit_file
        verify_audit_file(brain.engine_path, head["content"]["audit"], policy=brain.resource_policy)
    return {"format": FRONTIER_FORMAT, "version": 1, "brainId": brain.brain_id,
            "frontierSha256": frontier_sha256(head), "head": head}


def _content_text(block):
    if not isinstance(block, dict):
        raise ValueError("App observation content must be a typed block")
    common = {"annotations", "_meta"}
    if block.get("type") == "text" and set(block) <= common | {"type", "text"}:
        return _text(block.get("text"))
    if block.get("type") == "resource" and set(block) <= common | {"type", "resource"}:
        resource = block.get("resource")
        if (not isinstance(resource, dict) or set(resource) - {"uri", "mimeType", "_meta", "text"}
                or "text" not in resource):
            raise ValueError("Native App observation supports inline text resources, not binary or fetched resources")
        _text(resource.get("uri"), nonempty=True)
        if "mimeType" in resource:
            _text(resource["mimeType"], nonempty=True)
        return _text(resource["text"])
    raise ValueError("Native App observation currently supports text/structured data only; media and resource links were not learned")


def _task_text(data, *, observe):
    if set(data) != {"method", "params"} or data["method"] not in {"ui/message", "ui/update-model-context"}:
        raise ValueError("Unsupported App observation task method or fields")
    params = data["params"]
    allowed = {"role", "content", "_meta"} if data["method"] == "ui/message" else {"content", "structuredContent", "_meta"}
    if not isinstance(params, dict) or set(params) - allowed or "_meta" in params and not isinstance(params["_meta"], dict):
        raise ValueError("App observation cannot provide owner, role policy or hidden instructions")
    if data["method"] == "ui/message":
        if params.get("role") != "user" or "content" not in params:
            raise ValueError("App message must be attributed user task data")
        blocks = params["content"] if isinstance(params["content"], list) else [params["content"]]
        if not blocks:
            raise ValueError("App message has no content")
    else:
        blocks = params.get("content", [])
        if not isinstance(blocks, list) or "content" not in params and "structuredContent" not in params:
            raise ValueError("App context update has no typed task data")
        if "structuredContent" in params and not isinstance(params["structuredContent"], dict):
            raise ValueError("App structured content must be an object")
    # A superseded, reviewed record advances only audit metadata. No sensory
    # decoding or textual representation of discarded media is manufactured.
    if not observe:
        return None
    from .app_observation_content import needs_content_path
    if needs_content_path(data):
        return None
    parts = [_content_text(block) for block in blocks]
    if "structuredContent" in params:
        parts.append(json.dumps(params["structuredContent"], ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    text = "\n".join(parts)
    if data["method"] == "ui/message":
        # Match the existing native chat boundary for an explicitly reviewed
        # follow-up; no prompt prefix or source-label prose is inserted.
        text = text.replace("\0", "").strip()
    if not text.strip():
        raise ValueError("Empty App observation cannot execute neural learning")
    return text


@dataclass(frozen=True)
class ApprovedAppObservation:
    record: dict
    record_sha256: str
    data_sha256: str
    expected_frontier_sha256: str
    disposition: str
    text: str | None
    task_data: dict
    data_bytes: int
    extended: bool
    previous_audit_archived_sha256: str | None


def parse_app_observation(params, brain_id):
    required = {"brainId", "recordJson", "recordSha256", "dataJson", "dataSha256", "expectedFrontierSha256", "disposition"}
    if not isinstance(params, dict) or set(params) - {"storagePath", "previousAuditArchivedSha256"} != required or params.get("brainId") != brain_id:
        raise ValueError("App observation has a different native owner or unsupported wire fields")
    for key in ("recordSha256", "dataSha256", "expectedFrontierSha256"):
        _hash(params[key])
    if (raw_sha256(params["recordJson"]) != params["recordSha256"]
            or raw_sha256(params["dataJson"]) != params["dataSha256"]):
        raise ValueError("App observation reviewed bytes do not match their exact hashes")
    record, data = _json_object(params["recordJson"]), _json_object(params["dataJson"])
    required_record = {"format", "version", "id", "artifactId", "captureSha256", "source", "observedAt", "reviewId",
                       "method", "content", "streamId", "ordinal", "previousSha256", "untrusted"}
    if (set(record) != required_record or record["format"] != "omni-mcp-app-observation"
            or type(record["version"]) is not int or record["version"] != 1 or record["untrusted"] is not True):
        raise ValueError("App observation immutable record format is invalid")
    for key in ("id", "artifactId", "reviewId", "streamId"):
        _text(record[key], identifier=True)
    for key in ("captureSha256", "previousSha256"):
        _hash(record[key])
    _integer(record["ordinal"], 1)
    _instant(record["observedAt"])
    source = record["source"]
    if not isinstance(source, dict) or set(source) != {"brainId", "serverId", "serverLabel", "endpoint"} or source["brainId"] != brain_id:
        raise ValueError("App observation source belongs to a different brain")
    _text(source["serverId"], identifier=True)
    _text(source["serverLabel"], nonempty=True)
    _text(source["endpoint"], nonempty=True)
    content = record["content"]
    if (not isinstance(content, dict) or set(content) != {"sha256", "bytes", "mimeType"}
            or content["sha256"] != params["dataSha256"] or content["mimeType"] != "application/json"
            or _integer(content["bytes"], 1) != len(params["dataJson"].encode("utf-8"))
            or data.get("method") != record["method"]):
        raise ValueError("App observation content reference or method differs from reviewed data")
    if params["disposition"] not in {"observe", "superseded"}:
        raise ValueError("App observation queue disposition is invalid")
    text = _task_text(data, observe=params["disposition"] == "observe")
    from .app_observation_content import needs_content_path
    archived = _hash(params["previousAuditArchivedSha256"]) if "previousAuditArchivedSha256" in params else None
    return ApprovedAppObservation(record, params["recordSha256"], params["dataSha256"],
        params["expectedFrontierSha256"], params["disposition"], text, data, content["bytes"],
        params["disposition"] == "observe" and needs_content_path(data), archived)


class AppObservationCommitError(RuntimeError):
    """This warm owner must be released; retry only after explicit Load."""
    def __init__(self, message, *, committed=False):
        super().__init__(message)
        self.committed = committed


def _durable_head_is(brain, candidate):
    from .execution_checkpoint import ExecutionCheckpointStore
    metadata = ExecutionCheckpointStore(brain.engine_path, brain.brain_id, policy=brain.resource_policy)._read(brain.engine_path / "brain.json")
    return (metadata.get("brain_id") == brain.brain_id
            and validate_frontier_head(metadata.get("app_observation_frontier")) == candidate)


def admit_native_app_observation(brain, params, *, cancelled=lambda: False):
    if getattr(brain, "_connected_recovery_required", False):
        raise AppObservationCommitError("Explicit Load of the last atomic checkpoint is required before App admission")
    from .connected_execution import require_joint_mutation_allowed
    require_joint_mutation_allowed(brain, operation="admit_app_observation")
    raw_count = sum(len(params.get(key, "")) for key in ("recordJson", "dataJson") if isinstance(params.get(key), str))
    reserve = getattr(brain.resource_policy, "reserve_ram", None)
    # Reserve decoded JSON, exact UTF-8 hashing and joined learning text before
    # parsing. This scales with complete input; it is not a product size cap.
    with reserve(raw_count * 16 + 65536, "reviewed App observation parsing") if callable(reserve) else nullcontext():
        approved = parse_app_observation(params, brain.brain_id)
        previous = validate_frontier_head(getattr(brain, "app_observation_frontier", None))
        prior_audit = (previous or {}).get("content", {}).get("audit")
        if prior_audit is not None:
            from .app_observation_audit import verify_audit_file
            verify_audit_file(brain.engine_path, prior_audit, policy=brain.resource_policy)
        digest = frontier_sha256(previous)
        record = approved.record
        if previous is not None and previous["streamId"] == record["streamId"] and previous["ordinal"] == record["ordinal"]:
            if (previous["recordSha256"] != approved.record_sha256 or previous["dataSha256"] != approved.data_sha256
                    or previous["disposition"] != approved.disposition):
                raise ValueError("App observation retry conflicts with the committed current head")
            return {**native_app_observation_frontier(brain), "duplicate": True}
        if approved.expected_frontier_sha256 != digest:
            raise ValueError("App observation frontier changed; read the current native head before retry")
        if prior_audit is not None and approved.previous_audit_archived_sha256 != prior_audit["sha256"]:
            raise ValueError("The applied current App audit must be durably archived by the host before any next claim")
        if prior_audit is None and approved.previous_audit_archived_sha256 is not None:
            raise ValueError("App archival proof has no matching prior native audit")
        same_stream = previous is not None and previous["streamId"] == record["streamId"]
        if same_stream:
            if record["ordinal"] != previous["ordinal"] + 1 or record["previousSha256"] != previous["recordSha256"]:
                raise ValueError("App observation requires strict FIFO and its exact source predecessor")
        elif record["ordinal"] != 1 or record["previousSha256"] != EMPTY_FRONTIER:
            raise ValueError("A new App observation stream must start at its reviewed first record")
        candidate = {"format": HEAD_FORMAT, "version": 1, "streamId": record["streamId"], "ordinal": record["ordinal"],
            "recordSha256": approved.record_sha256, "dataSha256": approved.data_sha256, "previousSha256": record["previousSha256"],
            "baseFrontierSha256": digest,
            "disposition": approved.disposition, "state": "observed" if approved.disposition == "observe" else "advanced-without-observation",
            "committedAt": datetime.now(timezone.utc).isoformat(), "learning": None}
        mutated = committed = False
        spool = None
        result = None
        try:
            with ExitStack() as scope:
                text_projection, vector, transaction = approved.text, None, None
                if approved.extended:
                    from .app_observation_content import encode_app_content
                    vector, text_projection, content, spool = scope.enter_context(encode_app_content(brain, approved, cancelled=cancelled))
                    candidate["content"] = content
                    if getattr(brain, "_connected_regions", None) is not None:
                        from .connected_transaction import ConnectedExperienceTransaction
                        transaction = scope.enter_context(ConnectedExperienceTransaction(brain))
                try:
                    if cancelled():
                        raise InterruptedError("App observation stopped before ordinary neural mutation")
                    before_checksum = brain.parameter_checksum() if approved.disposition == "observe" and getattr(brain, "ingestion_checkpoints", {}) else None
                    if approved.disposition == "observe":
                        before_count = _integer(brain.counters["experiences"])
                        label = "reviewed-mcp-app:" + record["source"]["serverId"] + "/" + record["artifactId"]
                        mutated = True
                        if approved.extended:
                            learned = brain._admit_sensory_embedding(vector, kind="mcp-app-experience", source_name=label,
                                fingerprint="mcp-app:" + approved.record_sha256, substrate_space=True, experience_source="mcp-app")
                            if transaction is not None:
                                activity = brain._idea_model_vector(learned["vector"]).detach()
                                brain.decoder.observe_experience(activity, event_key="mcp-app:" + approved.record_sha256, importance=0.72)
                            settling, assembly = learned.get("memorySettling"), learned.get("assemblyId")
                        else:
                            learned = brain.learn_experience(text_projection, kind="experience", source="mcp-app", source_label=label, steps=0)
                            settling, assembly = learned.get("memory_settling"), learned.get("assembly_id", learned.get("idea_id"))
                        candidate["learning"] = {"kind": "experience", "source": "mcp-app", "sourceLabel": label, "steps": 0,
                            "experienceCountBefore": before_count, "experienceCountAfter": _integer(brain.counters["experiences"]),
                            "assemblyId": assembly, "afterimageId": str(settling.get("afterimageId", "")) if isinstance(settling, dict) else "",
                            "textSha256": raw_sha256(text_projection), "textBytes": len(text_projection.encode("utf-8"))}
                    candidate["committedAt"] = datetime.now(timezone.utc).isoformat()
                    candidate = validate_frontier_head(candidate)
                    brain.app_observation_frontier = candidate
                    mutated = True
                    if transaction is not None:
                        transaction.prepare_publication()
                    # Once publication might begin, an uncertain I/O outcome
                    # must retain the replayable audit rather than delete a
                    # possibly committed head's only ACK detail artifact.
                    if spool is not None:
                        spool.committed = True
                    if before_checksum is not None:
                        brain._save_observed_neural_boundary(before_checksum, record["id"])
                    else:
                        brain.save()
                    committed = True
                    result = {**native_app_observation_frontier(brain), "duplicate": False}
                except BaseException as error:
                    known = False
                    if mutated:
                        brain._connected_recovery_required = True
                        try:
                            known = _durable_head_is(brain, candidate)
                        except Exception:
                            known = None
                        if spool is not None:
                            spool.committed = known is not False
                        if known is not True:
                            brain.app_observation_frontier = previous
                    if known is True:
                        committed = True
                        result = {**native_app_observation_frontier(brain), "duplicate": False, "reloadRequired": True}
                    elif mutated:
                        raise AppObservationCommitError("App publication could not be confirmed; explicitly load the last atomic checkpoint before resolving this claim",
                            committed=False if known is False else None) from error
                    else:
                        raise
            if prior_audit is not None:
                from .app_observation_audit import prune_archived_previous_audit
                prune_archived_previous_audit(brain, previous, candidate, approved.previous_audit_archived_sha256)
        except BaseException as error:
            if committed:
                brain._connected_recovery_required = True
                try:
                    return {**native_app_observation_frontier(brain), "duplicate": False, "reloadRequired": True}
                except Exception as audit_error:
                    raise AppObservationCommitError("Neural publication committed, but its current ACK audit is unavailable; restore that audit before resolving the claim",
                        committed=True) from audit_error
            raise
        return result
