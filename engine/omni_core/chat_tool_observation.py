"""Owned live external evidence inbox; no inference, prompt or neural writes."""
import hashlib
import json
import re
import threading
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime

from .offload import NeuralStateResourcePause


FORMAT = "omni-chat-tool-observation-v1"
FIELDS = {"format", "brainId", "turnId", "neuralActionId", "actionEventId", "executionId",
          "toolId", "action", "completedAt", "payloadJson", "payloadSha256", "observationId"}


class _RetainedObservation(dict):
    """The decoded RPC lease follows evidence until its native read ends."""
    def __init__(self, value, allocation=None):
        super().__init__(value)
        self.allocation = allocation

    def release(self):
        allocation, self.allocation = self.allocation, None
        if allocation is not None: allocation.release()


def _tool(value):
    return {"windows.files": "system.files", "windows.powershell": "system.shell"}.get(value, value)


class ChatToolObservationInbox:
    def __init__(self, brain_id, turn_id, admit, *, cancelled=lambda: False, wait_seconds=30.0,
                 reserve_parse=None, service_pending=None):
        self.brain_id, self.turn_id, self.admit = brain_id, turn_id, admit
        self.reserve_parse = reserve_parse
        self.expected, self.seen, self.pending = {}, set(), deque()
        self.receipts = {}
        self.awaiting = set()
        self.closed = False
        self.lock = threading.RLock()
        self.ready = threading.Condition(self.lock)
        self.cancelled, self.wait_seconds = cancelled, max(0.0, float(wait_seconds))
        self.service_pending = service_pending

    def register(self, action_id, action):
        if action.get("kind") != "tool": return
        with self.lock:
            if not self.closed:
                self.expected[action_id] = (_tool(action.get("toolId")), action.get("action"))
                self.awaiting.add(action_id)

    def offer(self, value, allocation=None):
        if not isinstance(value, dict) or set(value) != FIELDS:
            raise ValueError("invalid live tool observation structure")
        if any(not isinstance(value[key], str) or not value[key] or "\x00" in value[key]
               for key in FIELDS): raise ValueError("invalid live observation identity")
        if value["format"] != FORMAT or value["brainId"] != self.brain_id or value["turnId"] != self.turn_id:
            raise ValueError("live tool observation has another brain/turn owner")
        result = {"brainId": self.brain_id, "turnId": self.turn_id,
                  "observationId": value["observationId"], "accepted": False}
        with self.lock:
            if self.closed: return {**result, "reason": "turn-output-closed"}
            if self.expected.get(value["neuralActionId"]) != (_tool(value["toolId"]), value["action"]):
                raise ValueError("tool observation does not match an emitted native action")
            if any(not re.fullmatch(r"[a-f0-9]{64}", value[key]) for key in ("payloadSha256", "observationId")) or \
                    not re.fullmatch(r"[a-f0-9]{32}", value["neuralActionId"]):
                raise ValueError("invalid live tool observation proof")
            datetime.fromisoformat(value["completedAt"].replace("Z", "+00:00"))
            if not self.admit(131072 + len(value["payloadJson"]) * 128):
                return {**result, "reason": "observation-resource-pressure"}
            digest = hashlib.sha256()
            for offset in range(0, len(value["payloadJson"]), 16_384):
                digest.update(value["payloadJson"][offset:offset + 16_384].encode("utf-8"))
            if digest.hexdigest() != value["payloadSha256"]: raise ValueError("tool observation payload checksum failed")
            fields = [FORMAT, self.brain_id, self.turn_id, value["neuralActionId"], value["actionEventId"],
                      value["executionId"], value["toolId"], value["action"], value["completedAt"], value["payloadSha256"]]
            if hashlib.sha256("\x00".join(fields).encode()).hexdigest() != value["observationId"]:
                raise ValueError("tool observation identity checksum failed")
            if value["observationId"] in self.seen: return {**result, "accepted": True, "duplicate": True}
            try:
                parse_lease = (self.reserve_parse(131072 + len(value["payloadJson"]) * 128,
                                                  "live tool observation JSON validation")
                               if self.reserve_parse is not None else nullcontext())
            except NeuralStateResourcePause:
                return {**result, "reason": "observation-resource-pressure"}
            with parse_lease:
                def invalid_constant(_value): raise ValueError("nonfinite tool observation JSON")
                payload = json.loads(value["payloadJson"], parse_constant=invalid_constant)
                if not isinstance(payload, dict) or type(payload.get("outputPresent")) is not bool:
                    raise ValueError("tool observation has no exact actual-output wrapper")
                ordinary = {"outputPresent", "output"} if payload["outputPresent"] else {"outputPresent"}
                keys = set(payload)
                if keys != ordinary:
                    failed = ordinary | {"executionState", "dispatchStarted"}
                    if "executionError" in payload: failed.add("executionError")
                    if (keys != failed or payload.get("executionState") != "failed" or payload.get("dispatchStarted") is not True
                        or "executionError" in payload and not isinstance(payload["executionError"], str)):
                        raise ValueError("tool observation has no exact actual failed-dispatch wrapper")
                del payload
            self.pending.append(_RetainedObservation(value, allocation))
            self.seen.add(value["observationId"])
            self.receipts[value["observationId"]] = {key: value[key] for key in FIELDS if key != "payloadJson"}
            self.awaiting.discard(value["neuralActionId"])
            self.ready.notify_all()
            # This acknowledges inbox ownership, not neural use or learning.
            return {**result, "accepted": True}

    def drain(self):
        with self.lock:
            if self.closed: return []
            result = list(self.pending)
            self.pending.clear()
            return result

    def suspension_state(self):
        """Same-turn inbox ownership, not an external handle or new effect."""
        with self.lock:
            if self.closed:
                raise ValueError("cannot suspend a closed observation inbox")
            return {"format": "omni-chat-observation-frontier", "version": 2,
                "brainId": self.brain_id, "turnId": self.turn_id,
                "expected": {key: list(value) for key, value in self.expected.items()},
                "seen": sorted(self.seen), "awaiting": sorted(self.awaiting),
                "pending": [dict(value) for value in self.pending], "receipts": dict(self.receipts), "waitSeconds": self.wait_seconds}

    @classmethod
    def from_suspension_state(cls, value, brain_id, turn_id, admit, *, cancelled=lambda: False, reserve_parse=None):
        fields = {"format", "version", "brainId", "turnId", "expected", "seen", "awaiting", "pending", "receipts", "waitSeconds"}
        if (not isinstance(value, dict) or set(value) != fields or value.get("format") != "omni-chat-observation-frontier"
                or value.get("version") != 2 or value.get("brainId") != brain_id or value.get("turnId") != turn_id
                or not isinstance(value.get("expected"), dict) or not isinstance(value.get("seen"), list)
                or not isinstance(value.get("awaiting"), list) or not isinstance(value.get("pending"), list)
                or not isinstance(value.get("receipts"), dict) or set(value["receipts"]) != set(value["seen"])):
            raise ValueError("invalid observation suspension ownership")
        wait = value["waitSeconds"]
        if not isinstance(wait, (int, float)) or isinstance(wait, bool) or not 0 <= wait <= 3600:
            raise ValueError("invalid suspended observation wait")
        result = cls(brain_id, turn_id, admit, cancelled=cancelled, wait_seconds=wait, reserve_parse=reserve_parse)
        for key, route in value["expected"].items():
            if (not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{32}", key) or not isinstance(route, list)
                    or len(route) != 2 or any(not isinstance(item, str) or not item or "\0" in item for item in route)):
                raise ValueError("invalid suspended native action binding")
            result.expected[key] = tuple(route)
        if (any(not isinstance(item, str) or not re.fullmatch(r"[a-f0-9]{64}", item) for item in value["seen"])
                or len(set(value["seen"])) != len(value["seen"])
                or len(set(value["awaiting"])) != len(value["awaiting"])
                or not set(value["awaiting"]).issubset(result.expected)):
            raise ValueError("invalid suspended observation identity set")
        prior = set(value["seen"])
        for identifier, receipt in value["receipts"].items():
            if (not isinstance(receipt, dict) or set(receipt) != FIELDS - {"payloadJson"}
                    or any(not isinstance(item, str) or not item or "\0" in item for item in receipt.values())
                    or receipt["brainId"] != brain_id or receipt["turnId"] != turn_id or receipt["observationId"] != identifier
                    or receipt["format"] != FORMAT or not re.fullmatch(r"[a-f0-9]{64}", receipt["payloadSha256"])
                    or result.expected.get(receipt["neuralActionId"]) != (_tool(receipt["toolId"]), receipt["action"])):
                raise ValueError("suspended consumed-observation receipt lost its original effect identity")
            parts = [receipt[key] for key in ("format", "brainId", "turnId", "neuralActionId", "actionEventId",
                "executionId", "toolId", "action", "completedAt", "payloadSha256")]
            if hashlib.sha256("\0".join(parts).encode()).hexdigest() != identifier:
                raise ValueError("suspended consumed-observation receipt checksum failed")
        for item in value["pending"]:
            if not isinstance(item, dict) or item.get("observationId") not in prior:
                raise ValueError("suspended pending observation lacks prior admission")
            receipt = result.offer(item)
            if not receipt.get("accepted") or receipt.get("duplicate"):
                raise ValueError("suspended pending observations are duplicated or not admitted")
        if set(value["awaiting"]) & {item["neuralActionId"] for item in result.pending}:
            raise ValueError("an already received suspended action is still marked awaiting")
        result.seen = prior
        result.receipts = dict(value["receipts"])
        result.awaiting = set(value["awaiting"])
        return result

    def close(self):
        with self.lock:
            self.closed = True
            while self.pending: self.pending.popleft().release()
            self.ready.notify_all()

    def wait_pending(self):
        """Natural EOS can await a real requested tool, never a forced reply.

        Explicit Stop/Steer bypass this method. Main's permission-wait setting
        bounds the ownership; expiry does not manufacture an observation.
        """
        deadline = time.monotonic() + self.wait_seconds
        while True:
            # A nested MCP sampling request must run on this writer while its
            # tool awaits the result. Never hold the inbox/control-reader lock
            # over native computation or the producer cannot deliver evidence.
            if callable(self.service_pending) and not self.cancelled():
                computation_started = time.monotonic()
                try:
                    self.service_pending()
                finally:
                    # This interval was owned native computation, not an
                    # external-result wait. Keep the configured wait budget
                    # available to the tool after its nested sample finishes.
                    deadline += max(0.0, time.monotonic() - computation_started)
            with self.ready:
                if not self.awaiting or self.pending or self.closed or self.cancelled():
                    return self.drain() if not self.cancelled() else []
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self.drain() if not self.cancelled() else []
                self.ready.wait(min(remaining, 0.05))

    __call__ = drain
