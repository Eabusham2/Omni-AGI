"""Main-only source proof for a reviewed App's ordinary task follow-up.

Canonical decoder user role is not a claim of human authorship. Source and
original review lineage remain typed metadata, never hidden prompt prose.
"""
from __future__ import annotations

import copy
import re
from contextlib import nullcontext

from .app_observation import (_hash, _integer, _json_object, _text, frontier_sha256,
                              parse_app_observation, raw_sha256, validate_frontier_head)

_SOURCE_IDS = ("sourceBrainId", "serverId", "artifactId", "originalCallBrainId", "reviewId", "observationId", "streamId", "toolId")
_SOURCE_HASHES = ("captureSha256", "recordSha256", "dataSha256", "nativeFrontierSha256", "inputSha256")
_SOURCE_TEXT = ("serverLabel", "endpoint", "resourceUri")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")


def validate_app_input_provenance(value, *, input_sha256=None):
    if (not isinstance(value, dict) or set(value) != {"kind", "ordinal", *_SOURCE_IDS, *_SOURCE_HASHES, *_SOURCE_TEXT}
            or value["kind"] != "mcp-app"):
        raise ValueError("App chat source attribution is invalid")
    _integer(value["ordinal"], 1)
    for name in _SOURCE_IDS:
        _text(value[name], nonempty=True)
        if not _ID.fullmatch(value[name]):
            raise ValueError("App chat source identity is invalid")
    for name in _SOURCE_HASHES:
        _hash(value[name])
    for name in _SOURCE_TEXT:
        _text(value[name], nonempty=True)
        if "\r" in value[name] or "\n" in value[name]:
            raise ValueError("App chat source labels cannot contain control lines")
    if input_sha256 is not None and value["inputSha256"] != input_sha256:
        raise ValueError("App chat source input hash differs")
    return copy.deepcopy(value)


def validate_native_app_chat_input(brain, proof, clean, *, prior_provenance=None):
    keys = {"recordJson", "recordSha256", "dataJson", "dataSha256", "captureJson", "captureSha256", "nativeFrontierSha256", "provenance"}
    if not isinstance(proof, dict) or set(proof) != keys:
        raise ValueError("App chat requires its exact internal source proof")
    for name in ("recordJson", "dataJson", "captureJson"):
        _text(proof[name], nonempty=True)
    reserve = getattr(brain.resource_policy, "reserve_ram", None)
    size = sum(len(proof[name]) for name in ("recordJson", "dataJson", "captureJson")) * 16 + 65536
    with reserve(size, "reviewed App chat source proof") if callable(reserve) else nullcontext():
        previous = validate_app_input_provenance(prior_provenance) if prior_provenance is not None else None
        owner = previous["sourceBrainId"] if previous is not None else brain.brain_id
        approved = parse_app_observation({"brainId": owner, "recordJson": proof["recordJson"], "recordSha256": proof["recordSha256"],
            "dataJson": proof["dataJson"], "dataSha256": proof["dataSha256"],
            "expectedFrontierSha256": proof["nativeFrontierSha256"], "disposition": "observe"}, owner)
        record = approved.record
        data = _json_object(proof["dataJson"])
        if record["method"] != "ui/message":
            raise ValueError("Only a reviewed App message may request an ordinary chat follow-up")
        blocks = data["params"]["content"]
        blocks = blocks if isinstance(blocks, list) else [blocks]
        if any(block.get("type") != "text" for block in blocks):
            raise ValueError("App chat follow-up currently supports text blocks only; no media or resource was dropped")
        if approved.text != clean or raw_sha256(clean) != raw_sha256(approved.text):
            raise ValueError("App chat input differs from the exact reviewed plain task")
        if proof["captureSha256"] != record["captureSha256"] or raw_sha256(proof["captureJson"]) != record["captureSha256"]:
            raise ValueError("App chat lost its exact original completed-call capture")
        capture = _json_object(proof["captureJson"])
        if (set(capture) != {"format", "version", "artifactId", "source", "observedAt", "resourceUri", "tool", "input", "result", "untrusted"}
                or capture.get("format") != "omni-mcp-app-capture" or type(capture.get("version")) is not int
                or capture["version"] != 1 or capture.get("untrusted") is not True or capture.get("artifactId") != record["artifactId"]):
            raise ValueError("App chat capture lineage is invalid")
        source, original = record["source"], capture.get("source")
        if (not isinstance(original, dict) or set(original) != {"brainId", "serverId", "serverLabel", "endpoint"}
                or original["serverId"] != source["serverId"] or original["endpoint"] != source["endpoint"]
                or not isinstance(capture.get("tool"), dict)):
            raise ValueError("App chat original call differs from the reviewed current source")
        provenance = validate_app_input_provenance({"kind": "mcp-app", "sourceBrainId": source["brainId"],
            "serverId": source["serverId"], "serverLabel": source["serverLabel"], "endpoint": source["endpoint"],
            "resourceUri": capture["resourceUri"], "artifactId": record["artifactId"], "captureSha256": record["captureSha256"],
            "originalCallBrainId": original["brainId"], "toolId": capture["tool"].get("id"),
            "reviewId": record["reviewId"], "observationId": record["id"], "streamId": record["streamId"], "ordinal": record["ordinal"],
            "recordSha256": approved.record_sha256, "dataSha256": approved.data_sha256,
            "nativeFrontierSha256": _hash(proof["nativeFrontierSha256"]), "inputSha256": raw_sha256(clean)})
        if validate_app_input_provenance(proof["provenance"]) != provenance:
            raise ValueError("App chat asserted attribution differs from its source bytes")
        if previous is not None:
            if previous != provenance:
                raise ValueError("App chat retry differs from the already accepted source")
        else:
            head = validate_frontier_head(getattr(brain, "app_observation_frontier", None))
            if (head is None or head["state"] != "observed" or frontier_sha256(head) != provenance["nativeFrontierSha256"]
                    or head["recordSha256"] != approved.record_sha256 or head["dataSha256"] != approved.data_sha256
                    or head["streamId"] != record["streamId"] or head["ordinal"] != record["ordinal"]
                    or head["learning"]["textSha256"] != provenance["inputSha256"]):
                raise ValueError("App chat follow-up has no matching current native learned admission")
        return provenance


def admitted_app_chat_experience(brain, clean, cue, recalled):
    """Read current neural conditioning without admitting the episode twice."""
    head = validate_frontier_head(getattr(brain, "app_observation_frontier", None))
    if head is None or head["state"] != "observed" or head["learning"]["textSha256"] != raw_sha256(clean):
        raise ValueError("App chat experience is not the current native admitted input")
    experience = brain._preview_chat_experience(clean, cue, recalled)
    experience["idea_id"] = experience["assembly_id"] = head["learning"]["assemblyId"]
    experience["memory_settling"] = {"afterimageId": head["learning"]["afterimageId"], "previouslyAdmittedAppObservation": True}
    return experience
