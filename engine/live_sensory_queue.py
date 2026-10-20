"""Request-owned capture handoff. No neural imports, work, or early use receipt.

The stdin reader may retain an already-admitted immutable request here. Only
the owning generation's ordinary world boundary takes it. The request's RAM
lease follows it until integration or rejection; control/shutdown reserves are
never borrowed for capture payloads.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import deque
from dataclasses import dataclass


LIVE_SENSORY_CONTROL = "enqueue_live_observation"
LIVE_OPERATIONS = frozenset({"start_observation", "observe_packet", "resolve_observation_control",
                              "stop_observation", "cancel_observation"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def capture_permission_hash(permission):
    fields = ["omni-live-permission-v1", permission.get("source"), permission.get("scope"),
              permission.get("grantedAt"), permission.get("deviceIdHash", "")]
    return hashlib.sha256(json.dumps(fields, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class SensoryRequest:
    request_id: object
    operation: str
    session_id: str
    payload_json: str
    allocation: object
    retained_bytes: int
    terminal: bool = False

    def payload(self):
        return json.loads(self.payload_json)

    def release(self):
        allocation, self.allocation = self.allocation, None
        self.payload_json = ""
        if allocation is not None:
            allocation.release()


class LiveSensoryInbox:
    def __init__(self, brain_id, turn_id, request_id, send, current):
        self.owner = (brain_id, turn_id, request_id)
        self.send, self.current = send, current
        self.lock = threading.RLock()
        self.pending = deque()
        self.session_bytes = {}
        self.retired_sessions = set()
        self.binding_allocations = []
        self.closed = False

    def has_pending(self):
        with self.lock:
            return not self.closed and bool(self.pending) and self.current()

    def retain_binding(self, allocation):
        """Charge accumulated provenance while search/Ponder has no tokens."""
        with self.lock:
            if self.closed:
                allocation.release()
                raise ValueError("live capture turn retired before retaining its binding")
            self.binding_allocations.append(allocation)

    def offer(self, request, allocation, session=None):
        """Take the retained lease only on success. This sends no RPC ACK."""
        if allocation is None:
            raise ValueError("capture control requires its ordinary admitted RAM allocation")
        params = request.get("params")
        if not isinstance(params, dict) or set(params) != {"brainId", "streamId", "operation", "payload"}:
            raise ValueError("invalid live capture handoff")
        if (params["brainId"], params["streamId"]) != self.owner[:2]:
            raise ValueError("live capture has another brain/turn owner")
        operation, payload = params["operation"], params["payload"]
        if operation not in LIVE_OPERATIONS or not isinstance(payload, dict) or payload.get("brainId") != self.owner[0]:
            raise ValueError("invalid live capture operation owner")
        session_id = payload.get("sessionId")
        if not isinstance(session_id, str) or not _IDENTIFIER.fullmatch(session_id):
            raise ValueError("invalid capture session identity")
        # The immutable JSON is metadata/bytes only, never a tensor or callback.
        immutable = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
        with self.lock:
            if self.closed or not self.current():
                raise ValueError("live capture originating turn has retired")
            if session_id in self.retired_sessions and not (operation == "resolve_observation_control" and
                    isinstance(payload.get("control"), dict) and payload["control"].get("state") == "cancelled"):
                raise ValueError("live capture session has retired")
            retained = len(immutable) * 4 + 1024
            if operation == "observe_packet":
                if not isinstance(session, dict) or session.get("brainId") != self.owner[0]:
                    raise ValueError("live capture has no registered session")
                if payload.get("captureRevision") != session.get("captureRevision") or payload.get("permissionSha256") != session.get("permissionSha256"):
                    raise ValueError("live capture permission or capture revision changed")
                encoded = payload.get("dataBase64")
                if not isinstance(encoded, str) or len(encoded) > ((session["maxPacketBytes"] + 2) // 3) * 4:
                    raise ValueError("live capture exceeds its admitted byte envelope")
                if not re.fullmatch(r"[a-f0-9]{64}", str(payload.get("packetSha256", ""))):
                    raise ValueError("live capture has no byte checksum")
                current_bytes = self.session_bytes.get(session_id, 0)
                # This is the existing negotiated transport budget, not a
                # model/learning cap. The retained RPC lease also remains held.
                maximum = ((((session["maxPacketBytes"] + 2) // 3) * 4) * 4 + 65536) * session["maxInFlight"]
                if current_bytes + retained > maximum:
                    raise ValueError("live capture awaits its current transport RAM budget")
            item = SensoryRequest(request.get("id"), operation, session_id, immutable, allocation, retained)
            self.pending.append(item)
            self.session_bytes[session_id] = self.session_bytes.get(session_id, 0) + retained
        return item

    def take(self):
        with self.lock:
            if self.closed or not self.current() or not self.pending:
                return None
            return self.pending.popleft()

    def finish(self, item, *, result=None, error=None):
        with self.lock:
            if item.terminal:
                return
            item.terminal = True
            remaining = max(0, self.session_bytes.get(item.session_id, 0) - item.retained_bytes)
            if remaining: self.session_bytes[item.session_id] = remaining
            else: self.session_bytes.pop(item.session_id, None)
        try:
            self.send({"jsonrpc": "2.0", "id": item.request_id,
                       **({"error": error} if error is not None else {"result": result})})
        finally:
            item.release()

    def retire_session(self, session_id, reason="capture-session-retired"):
        with self.lock:
            self.retired_sessions.add(session_id)
            rejected = [item for item in self.pending if item.session_id == session_id]
            self.pending = deque(item for item in self.pending if item.session_id != session_id)
        failure = None
        for item in rejected:
            try:
                self.finish(item, error={"code": -32800, "message": reason})
            except Exception as error:
                failure = failure or error
        if failure is not None:
            raise failure

    def close(self, reason="capture-turn-retired"):
        with self.lock:
            self.closed = True
            rejected, self.pending = list(self.pending), deque()
            allocations, self.binding_allocations = self.binding_allocations, []
        failure = None
        try:
            for item in rejected:
                try:
                    self.finish(item, error={"code": -32800, "message": reason})
                except Exception as error:
                    failure = failure or error
        finally:
            for allocation in allocations:
                allocation.release()
        if failure is not None:
            raise failure
