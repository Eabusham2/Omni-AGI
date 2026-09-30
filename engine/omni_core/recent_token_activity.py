"""Temporary token residency derived from existing neural afterimage activity.

This is span/cycle metadata, not a learned memory or an answer store. Tokens
remain in the caller's single recent-context list. Cooling never reconstructs
tokens from history, neural IDs, or retired spans.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


def token_hash(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        ",".join(str(int(value)) for value in tokens).encode("ascii")
    ).hexdigest()


def _unit(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return max(0.0, min(1.0, number)) if math.isfinite(number) else 0.0


def _raw_activity(item: Mapping[str, Any], cycle: int) -> float:
    """Use the lifecycle's live signals and dormancy, not text/turn grammar."""

    signals = item.get("signals", {})
    if not isinstance(signals, Mapping):
        signals = {}
    reuse = _unit(signals.get("reuse"))
    recurrence = _unit(signals.get("recurrence"))
    stability = _unit(signals.get("stability"))
    unfinished = _unit(item.get("unfinishedScore"))
    inactive = max(0, cycle - int(item.get("lastActiveCycle", cycle)))
    # Same dormancy shape as OrganicMemoryLifecycle's afterimage rescoring.
    # Stable/reused activity cools more slowly, but stability alone cannot
    # permanently pin raw words. Unfinished/current/reused spans are protected
    # separately below while the rest of their activity continues to change.
    scale = 6.0 + 12.0 * reuse + 14.0 * stability + 8.0 * unfinished
    drive = (
        0.25 * _unit(item.get("strength"))
        + 0.25 * _unit(item.get("activityScore"))
        + 0.20 * _unit(item.get("salience"))
        + 0.15 * reuse
        + 0.15 * recurrence
    )
    return drive * math.exp(-inactive / scale) * (
        1.0 - 0.5 * _unit(signals.get("interference"))
    )


@dataclass
class RecentTokenActivity:
    """Ordered token spans linked to exact transient episode IDs, never text."""

    SCHEMA = "recent-token-activity-1"
    spans: List[Dict[str, Any]] = field(default_factory=list)

    def _validate_alignment(self, tokens: Sequence[int]) -> None:
        if sum(int(span["tokenCount"]) for span in self.spans) != len(tokens):
            raise ValueError("recent token activity does not align with token context")

    @classmethod
    def from_state(
        cls,
        metadata: Any,
        tokens: Sequence[int],
        *,
        human_id: Optional[int] = None,
    ) -> "RecentTokenActivity":
        if metadata is None:
            # Legacy tokens have no proved episode binding. Retain them
            # conservatively under the existing capacity rule and report them
            # as untracked, never guess a match from an assembly ID or text.
            starts = [0] if tokens else []
            if human_id is not None:
                starts.extend(
                    index
                    for index, token in enumerate(tokens)
                    if index > 0 and token == human_id
                )
            ends = starts[1:] + [len(tokens)]
            return cls(
                [
                    {"tokenCount": end - start, "afterimageId": ""}
                    for start, end in zip(starts, ends)
                ]
            )
        if not isinstance(metadata, Mapping) or metadata.get("schema") != cls.SCHEMA:
            raise ValueError("recent token activity schema is invalid")
        if metadata.get("tokenHash") != token_hash(tokens):
            raise ValueError("recent token activity token binding changed")
        raw_spans = metadata.get("spans")
        if not isinstance(raw_spans, list):
            raise ValueError("recent token activity spans are invalid")
        spans = []
        for raw in raw_spans:
            if not isinstance(raw, Mapping):
                raise ValueError("recent token activity span is invalid")
            count, identifier = raw.get("tokenCount"), raw.get("afterimageId")
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
                or not isinstance(identifier, str)
            ):
                raise ValueError("recent token activity span binding is invalid")
            spans.append({"tokenCount": count, "afterimageId": identifier})
        result = cls(spans)
        result._validate_alignment(tokens)
        return result

    def metadata(self, tokens: Sequence[int]) -> Dict[str, Any]:
        self._validate_alignment(tokens)
        return {
            "schema": self.SCHEMA,
            "tokenHash": token_hash(tokens),
            "spans": [dict(span) for span in self.spans],
            "rawTextStored": False,
            "rawTokenIdsStored": False,
        }

    def summary(self) -> Dict[str, Any]:
        tracked = sum(
            int(span["tokenCount"]) for span in self.spans if span["afterimageId"]
        )
        untracked = sum(
            int(span["tokenCount"]) for span in self.spans if not span["afterimageId"]
        )
        return {
            "trackedTokens": tracked,
            "untrackedTokens": untracked,
            "spanCount": len(self.spans),
            "dynamicCooling": True,
            "legacyBindingGuessed": False,
            "evictedRawTextReinjected": False,
        }

    def append(
        self, tokens: List[int], turn: Sequence[int], afterimage_id: str
    ) -> List[int]:
        self._validate_alignment(tokens)
        if not turn:
            return tokens
        self.spans.append({"tokenCount": len(turn), "afterimageId": str(afterimage_id)})
        return tokens + list(turn)

    def cool(
        self, tokens: List[int], afterimages: Sequence[Mapping[str, Any]], cycle: int
    ) -> List[int]:
        self._validate_alignment(tokens)
        by_id = {
            str(item.get("id", "")): item for item in afterimages if item.get("id")
        }
        kept_spans, kept_ranges = [], []
        offset = 0
        for index, span in enumerate(self.spans):
            end = offset + int(span["tokenCount"])
            identifier = str(span["afterimageId"])
            item = by_id.get(identifier)
            # No arbitrary turn-count/time cutoff. Preserve the current turn,
            # proved unfinished activity and exact episode reuse. Missing
            # legacy bindings are explicit, not fabricated neural evidence.
            protected = index == len(self.spans) - 1 or not identifier
            if item is not None:
                protected = protected or (
                    _unit(item.get("unfinishedScore")) >= 0.35
                    or int(item.get("lastActiveCycle", -1)) >= cycle
                )
            activity = _raw_activity(item, cycle) if item is not None else 0.0
            if protected or activity >= 0.035:
                kept_spans.append(span)
                kept_ranges.append((offset, end))
            offset = end
        if len(kept_spans) == len(self.spans):
            return tokens
        self.spans = kept_spans
        return [token for start, end in kept_ranges for token in tokens[start:end]]

    def fit_capacity(
        self,
        tokens: List[int],
        capacity: int,
        *,
        human_id: int,
        brain_id: int,
        eos_id: int,
    ) -> List[int]:
        """Keep completed role boundaries when the physical budget shrinks."""

        self._validate_alignment(tokens)
        capacity = max(3, int(capacity))
        if len(tokens) <= capacity:
            return tokens
        offset = 0
        while len(tokens) - offset > capacity and len(self.spans) > 1:
            offset += int(self.spans.pop(0)["tokenCount"])
        retained = tokens[offset:]
        if len(retained) > capacity:
            # A single newly reduced current turn may exceed the new budget.
            # Retain both roles as _bounded_completed_turn_tokens does; this
            # capacity fallback neither interprets prose nor stores it anew.
            try:
                boundary = retained.index(brain_id)
            except ValueError:
                if retained and retained[0] == human_id and retained[-1] == eos_id:
                    # A completed no-reply turn has only observed human input,
                    # not a fabricated BRAIN target. Preserve its real roles.
                    retained = [human_id] + retained[1:-1][-(capacity - 2):] + [eos_id]
                else:
                    retained = retained[-capacity:]
            else:
                human = (
                    retained[1:boundary]
                    if retained[0] == human_id
                    else retained[:boundary]
                )
                brain = (
                    retained[boundary + 1 : -1]
                    if retained[-1] == eos_id
                    else retained[boundary + 1 :]
                )
                budget = capacity - 3
                brain_budget = min(len(brain), budget // 2)
                human_budget = min(len(human), budget - brain_budget)
                brain_budget += min(
                    len(brain) - brain_budget, budget - human_budget - brain_budget
                )
                retained = [human_id] + (human[-human_budget:] if human_budget else [])
                retained += (
                    [brain_id]
                    + (brain[-brain_budget:] if brain_budget else [])
                    + [eos_id]
                )
            if self.spans:
                self.spans[-1]["tokenCount"] = len(retained)
        self._validate_alignment(retained)
        return retained
