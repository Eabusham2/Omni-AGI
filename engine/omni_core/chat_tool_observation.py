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
                 reserve_parse=None):
        self.brain_id, self.turn_id, self.admit = brain_id, turn_id, admit
        self.reserve_parse = reserve_parse
        self.expected, self.seen, self.pending = {}, set(), deque()
        self.awaiting = set()
        self.closed = False
        self.lock = threading.RLock()
        self.ready = threading.Condition(self.lock)
        self.cancelled, self.wait_seconds = cancelled, max(0.0, float(wait_seconds))

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
                if not isinstance(payload, dict) or type(payload.get("outputPresent")) is not bool or \
                        set(payload) != ({"outputPresent", "output"} if payload["outputPresent"] else {"outputPresent"}):
                    raise ValueError("tool observation has no exact actual-output wrapper")
                del payload
            self.pending.append(_RetainedObservation(value, allocation))
            self.seen.add(value["observationId"])
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
        with self.ready:
            while self.awaiting and not self.pending and not self.closed and not self.cancelled():
                remaining = deadline - time.monotonic()
                if remaining <= 0: break
                self.ready.wait(min(remaining, 0.05))
            return self.drain() if not self.cancelled() else []

    __call__ = drain
