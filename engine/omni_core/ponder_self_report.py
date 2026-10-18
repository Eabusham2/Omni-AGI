"""Optional model-emitted Ponder self-reports with no prompt or canned prose."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

import torch


ROUTE_ID = "brain.ponder"
ROUTE_ACTION = "self-report"
PRIVATE_CANDIDATE = "_ponderSelfReportCandidate"
PUBLIC_REPORT = "ponderSelfReport"
SCHEMA: Dict[str, Any] = {
    "id": ROUTE_ID,
    "actions": [ROUTE_ACTION],
    "grant": "auto",
    "actionInputSchemas": {
        ROUTE_ACTION: {
            "type": "object",
            "properties": {
                "emit": {"type": "boolean"},
                "text": {"type": "string"},
            },
            "required": ["emit"],
            "additionalProperties": False,
        }
    },
}
FRAME_BYTES = 16 * 1024


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def report_identity(action_id: str, phase: str) -> str:
    return _sha256(
        ("omni-ponder-self-report-v1\0%s\0%s" % (
            action_id, phase
        )).encode("utf-8")
    )


@torch.no_grad()
def decode_ponder_self_report(
    decoder: Any,
    neural_state: torch.Tensor,
    *,
    phase: str,
    max_output_bytes: int,
    cancelled: Optional[Callable[[], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Decode an optional report from learned weights and current neural state.

    The existing typed-argument head performs whole inference before any text
    is published. A blank/untrained/uncertain head remains silent.
    """

    if phase not in {"pre-speech", "mid-generation"}:
        raise ValueError("Ponder self-report phase is invalid")
    head = getattr(decoder, "action_argument_head", None)
    if head is None or int(getattr(head, "grounded_steps", torch.zeros(())).item()) < 1:
        return None
    features = head.schema_features(ROUTE_ID, ROUTE_ACTION, SCHEMA)
    if features is None or not isinstance(neural_state, torch.Tensor) or neural_state.ndim != 2:
        return None
    evidence = head.decode(
        neural_state,
        features[None].to(neural_state.device),
        tool_id=ROUTE_ID,
        action=ROUTE_ACTION,
        max_output_bytes=max(1, int(max_output_bytes)),
        minimum_mean_probability=0.90,
        input_schema=SCHEMA["actionInputSchemas"][ROUTE_ACTION],
        # Generalization is permitted only after the same head has learned
        # actual grounded argument values. No declared self-report target or
        # canned text is installed here.
        allow_declared_training=True,
        cancelled=cancelled,
    )
    arguments = evidence.get("arguments")
    if not isinstance(arguments, Mapping) or set(arguments) - {"emit", "text"}:
        return None
    if arguments.get("emit") is not True:
        return None
    text = arguments.get("text")
    if not isinstance(text, str) or "\x00" in text:
        return None
    text = text.strip()
    if not text:
        return None
    encoded = text.encode("utf-8")
    return {
        "format": "omni-ponder-self-report-candidate",
        "version": 1,
        "phase": phase,
        "text": text,
        "textSha256": _sha256(encoded),
        "bytes": len(encoded),
        "modelEmitted": True,
        "selfReportNotChainOfThought": True,
        "hiddenPromptUsed": False,
        "inferenceCompletedBeforePublication": True,
        "generalizationVerified": False,
    }


def _candidate(value: Any) -> Dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {
            "format", "version", "phase", "text", "textSha256", "bytes",
            "modelEmitted", "selfReportNotChainOfThought", "hiddenPromptUsed",
            "inferenceCompletedBeforePublication", "generalizationVerified",
        }
        or value.get("format") != "omni-ponder-self-report-candidate"
        or value.get("version") != 1
        or value.get("phase") not in {"pre-speech", "mid-generation"}
        or not isinstance(value.get("text"), str)
        or not value["text"]
        or "\x00" in value["text"]
        or value.get("textSha256") != _sha256(value["text"].encode("utf-8"))
        or value.get("bytes") != len(value["text"].encode("utf-8"))
        or value.get("modelEmitted") is not True
        or value.get("selfReportNotChainOfThought") is not True
        or value.get("hiddenPromptUsed") is not False
        or value.get("inferenceCompletedBeforePublication") is not True
        or value.get("generalizationVerified") is not False
    ):
        raise ValueError("Ponder self-report candidate is invalid")
    return dict(value)


def complete_report(candidate: Mapping[str, Any], action_id: str) -> Dict[str, Any]:
    value = _candidate(candidate)
    if not isinstance(action_id, str) or len(action_id) != 32 or any(
        character not in "0123456789abcdef" for character in action_id.lower()
    ):
        raise ValueError("Ponder self-report action identity is invalid")
    phase, text_sha256 = str(value["phase"]), str(value["textSha256"])
    return {
        "format": "omni-ponder-self-report",
        "version": 1,
        "id": report_identity(action_id.lower(), phase),
        "actionId": action_id.lower(),
        "phase": phase,
        "text": value["text"],
        "state": "complete",
        "emittedBytes": int(value["bytes"]),
        "textSha256": text_sha256,
        "modelEmitted": True,
        "selfReportNotChainOfThought": True,
        "hiddenPromptUsed": False,
        "inferenceCompletedBeforePublication": True,
    }


def publication_frames(report: Mapping[str, Any]) -> Iterator[Dict[str, Any]]:
    text = str(report["text"])
    chunks = []
    current = ""
    current_bytes = 0
    for character in text:
        size = len(character.encode("utf-8"))
        if current and current_bytes + size > FRAME_BYTES:
            chunks.append(current)
            current, current_bytes = "", 0
        current += character
        current_bytes += size
    if current:
        chunks.append(current)
    offset = 0
    for index, delta in enumerate(chunks):
        size = len(delta.encode("utf-8"))
        final = index == len(chunks) - 1
        yield {
            "format": "omni-ponder-self-report-delta",
            "version": 1,
            "reportId": report["id"],
            "actionId": report["actionId"],
            "phase": report["phase"],
            "chunkIndex": index,
            "byteOffset": offset,
            "delta": delta,
            "publicationComplete": final,
            **({"totalBytes": report["emittedBytes"], "textSha256": report["textSha256"]} if final else {}),
            "modelEmitted": True,
            "selfReportNotChainOfThought": True,
            "hiddenPromptUsed": False,
            "inferenceCompletedBeforePublication": True,
        }
        offset += size


def validate_publication_frame(value: Any) -> Dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) - {
            "format", "version", "reportId", "actionId", "phase", "chunkIndex",
            "byteOffset", "delta", "publicationComplete", "totalBytes", "textSha256",
            "modelEmitted", "selfReportNotChainOfThought", "hiddenPromptUsed",
            "inferenceCompletedBeforePublication",
        }
        or value.get("format") != "omni-ponder-self-report-delta"
        or value.get("version") != 1
        or not isinstance(value.get("reportId"), str)
        or len(value["reportId"]) != 64
        or any(character not in "0123456789abcdef" for character in value["reportId"])
        or not isinstance(value.get("actionId"), str)
        or len(value["actionId"]) != 32
        or any(character not in "0123456789abcdef" for character in value["actionId"])
        or value.get("phase") not in {"pre-speech", "mid-generation"}
        or not isinstance(value.get("chunkIndex"), int)
        or value["chunkIndex"] < 0
        or not isinstance(value.get("byteOffset"), int)
        or value["byteOffset"] < 0
        or not isinstance(value.get("delta"), str)
        or not value["delta"]
        or "\x00" in value["delta"]
        or len(value["delta"].encode("utf-8")) > 64 * 1024
        or not isinstance(value.get("publicationComplete"), bool)
        or value.get("modelEmitted") is not True
        or value.get("selfReportNotChainOfThought") is not True
        or value.get("hiddenPromptUsed") is not False
        or value.get("inferenceCompletedBeforePublication") is not True
    ):
        raise ValueError("Ponder self-report publication frame is invalid")
    complete = value["publicationComplete"]
    if complete:
        if (
            not isinstance(value.get("totalBytes"), int)
            or value["totalBytes"] < 1
            or not isinstance(value.get("textSha256"), str)
            or len(value["textSha256"]) != 64
            or any(character not in "0123456789abcdef" for character in value["textSha256"])
        ):
            raise ValueError("Completed Ponder self-report frame lacks its exact digest")
    elif "totalBytes" in value or "textSha256" in value:
        raise ValueError("Incomplete Ponder self-report frame claims completion")
    return dict(value)


class PonderPublicationTracker:
    """Validate ordered bytes without retaining a second full text copy."""

    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = {}

    def accept(self, value: Any, published_actions: Any) -> Dict[str, Any]:
        frame = validate_publication_frame(value)
        action_id, report_id = frame["actionId"], frame["reportId"]
        if not any(
            isinstance(action, Mapping)
            and action.get("actionId") == action_id
            and action.get("kind") == "ponder"
            for action in published_actions
        ):
            raise ValueError("Ponder self-report has no preceding visible Ponder action")
        current = self._states.get(action_id)
        if current is None:
            if frame["chunkIndex"] != 0 or frame["byteOffset"] != 0:
                raise ValueError("Ponder self-report publication did not start at its first byte")
            current = {
                "reportId": report_id, "phase": frame["phase"], "nextIndex": 0,
                "bytes": 0, "hash": hashlib.sha256(), "complete": False,
            }
            self._states[action_id] = current
        if (
            current["complete"]
            or current["reportId"] != report_id
            or current["phase"] != frame["phase"]
            or frame["chunkIndex"] != current["nextIndex"]
            or frame["byteOffset"] != current["bytes"]
        ):
            raise ValueError("Ponder self-report publication order or identity changed")
        encoded = frame["delta"].encode("utf-8")
        current["hash"].update(encoded)
        current["bytes"] += len(encoded)
        current["nextIndex"] += 1
        if frame["publicationComplete"]:
            digest = current["hash"].hexdigest()
            if (
                frame["totalBytes"] != current["bytes"]
                or frame["textSha256"] != digest
                or report_id != report_identity(action_id, frame["phase"])
            ):
                raise ValueError("Completed Ponder self-report digest differs from its published bytes")
            current["complete"] = True
        return frame


def emit_registered_action(emit: Callable[[str, Dict[str, Any]], None], action: Dict[str, Any]) -> None:
    """Publish action first, then the already-decoded text in bounded frames."""

    arguments = action.get("arguments")
    candidate = arguments.get(PRIVATE_CANDIDATE) if isinstance(arguments, dict) else None
    if candidate is None:
        emit("action", {"actionId": action["actionId"], "action": action})
        return
    report = complete_report(_candidate(candidate), str(action.get("actionId", "")))
    visible = copy.deepcopy(action)
    visible_arguments = visible.setdefault("arguments", {})
    visible_arguments.pop(PRIVATE_CANDIDATE, None)
    visible_arguments[PUBLIC_REPORT] = {
        "format": report["format"], "version": report["version"],
        "id": report["id"], "actionId": report["actionId"],
        "phase": report["phase"], "state": "publishing",
        "modelEmitted": True, "selfReportNotChainOfThought": True,
        "hiddenPromptUsed": False, "inferenceCompletedBeforePublication": True,
    }
    emit("action", {"actionId": action["actionId"], "action": visible})
    for frame in publication_frames(report):
        emit("ponder-self-report", frame)
    arguments.pop(PRIVATE_CANDIDATE, None)
    arguments[PUBLIC_REPORT] = report


def finalize_registered_action(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    arguments = action.get("arguments")
    if not isinstance(arguments, dict):
        return None
    existing = arguments.get(PUBLIC_REPORT)
    if isinstance(existing, Mapping) and existing.get("state") == "complete":
        return dict(existing)
    candidate = arguments.pop(PRIVATE_CANDIDATE, None)
    if candidate is None:
        return None
    report = complete_report(_candidate(candidate), str(action.get("actionId", "")))
    arguments[PUBLIC_REPORT] = report
    return report
