"""Bounded literal token waves with source-bound resumable record cursors.

No neural modules are imported. A stream owns one live DatasetRecord lease;
advancing its source iterator before ``complete`` is a scheduling error.
"""

from dataclasses import dataclass
import hashlib

from .text_spool import TextPayload, leased_dialogue_windows


CURSOR_FIELDS = {"format", "formatVersion", "recordId", "ordinal", "contentSha256",
    "phase", "tokenCursor", "completedWindows", "pairIndex", "sequenceTokens"}


def validate_record_window_cursor(value):
    if not isinstance(value, dict) or set(value) not in (CURSOR_FIELDS, CURSOR_FIELDS | {"lastWavePlan"}):
        raise ValueError("distributed record-window cursor schema is invalid")
    if value["format"] != "omni-record-token-windows" or value["formatVersion"] != 1 or value["phase"] not in {"text", "targets", "complete"}:
        raise ValueError("distributed record-window cursor policy is invalid")
    for name in ("ordinal", "tokenCursor", "completedWindows", "pairIndex", "sequenceTokens"):
        if type(value[name]) is not int or not 0 <= value[name] <= (1 << 63) - 1:
            raise ValueError("distributed record-window cursor counter is invalid")
    if value["sequenceTokens"] < 2:
        raise ValueError("distributed record-window cursor has no causal context")
    for name in ("recordId", "contentSha256"):
        if not isinstance(value[name], str) or len(value[name]) != 64 or any(char not in "0123456789abcdef" for char in value[name]):
            raise ValueError("distributed record-window cursor identity is invalid")
    if value["phase"] == "complete" and value["tokenCursor"]:
        raise ValueError("complete record-window cursor retains a target")
    clean = dict(value)
    if "lastWavePlan" in value:
        plan = value["lastWavePlan"]
        if not isinstance(plan, dict) or set(plan) != {"physicalBatchWindows", "waveWindowTarget", "recordGroup", "adaptAtCommittedBoundary"}:
            raise ValueError("record-window physical wave audit is invalid")
        if any(type(plan[key]) is not int or plan[key] < 1 for key in ("physicalBatchWindows", "waveWindowTarget")) or plan["adaptAtCommittedBoundary"] is not True:
            raise ValueError("record-window physical wave policy is invalid")
        group = plan["recordGroup"]
        if not isinstance(group, list) or len(group) != 2 or any(type(item) is not int or item < 0 for item in group) or not group[0] <= value["ordinal"] < group[1]:
            raise ValueError("record-window group does not contain its exact source ordinal")
        clean["lastWavePlan"] = {**plan, "recordGroup": list(group)}
    return clean


@dataclass(frozen=True)
class RecordTokenWindow:
    ids: list
    text: str
    human: object
    window_index: int
    phase: str


class RecordWindowStream:
    def __init__(self, record, entry, tokenizer, sequence_tokens, cursor=None):
        self.record, self.entry, self.tokenizer = record, entry, tokenizer
        payload = getattr(record, "text_payload", None)
        self.payload = payload if payload is not None else TextPayload(text=record.text,
            bytes=len(record.text.encode("utf-8")), sha256=hashlib.sha256(record.text.encode("utf-8")).hexdigest())
        self.dialogue = getattr(self.payload, "dialogue", None)
        self.small_pairs = list(record.provenance.get("dialoguePairs", ())) if self.dialogue is None else ()
        self.pair_count = self.dialogue.pair_count if self.dialogue is not None else len(self.small_pairs)
        self.cursor = validate_record_window_cursor(cursor) if cursor is not None else {
            "format": "omni-record-token-windows", "formatVersion": 1,
            "recordId": entry.record_id, "ordinal": entry.ordinal, "contentSha256": entry.content_sha256,
            "phase": "text" if entry.kind == "text" and self.payload.bytes else "complete",
            "tokenCursor": 0, "completedWindows": 0, "pairIndex": 0,
            "sequenceTokens": int(sequence_tokens),
        }
        if (self.cursor["recordId"] != entry.record_id or self.cursor["ordinal"] != entry.ordinal
            or self.cursor["contentSha256"] != entry.content_sha256
            or self.cursor["sequenceTokens"] != sequence_tokens):
            raise ValueError("distributed record lease does not match its frozen token-window cursor")
        self.iterator = None
        self.active_response = None
        self.active_human = None
        self._advance_phase()
        self._validate_window_count()

    @property
    def complete(self):
        return self.cursor["phase"] == "complete"

    def _pair(self, ordinal):
        if self.dialogue is not None:
            return self.dialogue.pair(ordinal)
        pair = self.small_pairs[ordinal]
        def payload(text):
            if not isinstance(text, str) or not text:
                raise ValueError("distributed typed target is invalid")
            encoded = text.encode("utf-8")
            return TextPayload(text=text, bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
        return payload(pair["human"]), payload(pair["brain"])

    def _advance_phase(self):
        phase = self.cursor["phase"]
        if phase == "text" and self.cursor["tokenCursor"] == self.payload.bytes + 1:
            self.cursor.update(phase="targets" if self.pair_count else "complete", tokenCursor=0)
            self.iterator = None
        if self.cursor["phase"] == "targets":
            if self.cursor["pairIndex"] > self.pair_count:
                raise ValueError("distributed target cursor exceeds its pair index")
            while self.cursor["pairIndex"] < self.pair_count:
                self.active_human, self.active_response = self._pair(self.cursor["pairIndex"])
                if self.cursor["tokenCursor"] > self.active_response.bytes + 1:
                    raise ValueError("distributed target cursor exceeds literal response bytes")
                if self.cursor["tokenCursor"] != self.active_response.bytes + 1:
                    break
                self.cursor["pairIndex"] += 1
                self.cursor["tokenCursor"] = 0
                self.iterator = None
            if self.cursor["pairIndex"] == self.pair_count:
                self.cursor.update(phase="complete", tokenCursor=0)
        elif self.cursor["phase"] == "text" and self.cursor["tokenCursor"] > self.payload.bytes + 1:
            raise ValueError("distributed text cursor exceeds literal source bytes")

    def _validate_window_count(self):
        stride = self.cursor["sequenceTokens"] - 1
        phase = self.cursor["phase"]
        if self.entry.kind != "text":
            expected = 0
        elif phase == "text":
            expected = (self.cursor["tokenCursor"] + stride - 1) // stride
        else:
            expected = (self.payload.bytes + stride) // stride
            before = self.cursor["pairIndex"]
            if self.dialogue is not None:
                expected += int(self.dialogue.index.execute(
                    "SELECT COALESCE(SUM((response_bytes+?) / ?),0) FROM pairs WHERE ordinal < ?",
                    (stride, stride, before)).fetchone()[0])
            else:
                expected += sum((len(pair["brain"].encode("utf-8")) + stride) // stride for pair in self.small_pairs[:before])
            if phase == "targets":
                expected += (self.cursor["tokenCursor"] + stride - 1) // stride
        if expected != self.cursor["completedWindows"]:
            raise ValueError("distributed literal-window count differs from its exact byte/pair cursor")

    def next_batch(self, window_budget):
        if type(window_budget) is not int or window_budget < 1:
            raise ValueError("distributed physical window budget is invalid")
        windows = []
        while len(windows) < window_budget:
            self._advance_phase()
            if self.complete:
                break
            phase = self.cursor["phase"]
            payload = self.payload if phase == "text" else self.active_response
            if self.iterator is None:
                self.iterator = iter(leased_dialogue_windows(payload, self.tokenizer, self.cursor["sequenceTokens"],
                    start_target=self.cursor["tokenCursor"],
                    context_id=self.tokenizer.bos_id if phase == "text" else self.tokenizer.brain_id))
            try:
                ids, position = next(self.iterator)
            except StopIteration:
                self.iterator = None
                self._advance_phase()
                continue
            literal = bytes(value - self.tokenizer.byte_offset for value in ids[1:] if value >= self.tokenizer.byte_offset).decode("utf-8", errors="replace")
            windows.append(RecordTokenWindow(ids, literal, None if phase == "text" else self.active_human,
                self.cursor["completedWindows"], phase))
            self.cursor["tokenCursor"] = position
            self.cursor["completedWindows"] += 1
        self._advance_phase()
        return windows

    def state(self):
        return validate_record_window_cursor(self.cursor)

    def close(self):
        if self.iterator is not None:
            self.iterator.close()
            self.iterator = None
