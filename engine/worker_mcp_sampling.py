"""Metadata-only MCP controls; inference runs on the existing neural writer.

The stdin/control reader never executes a neural operation. A live chat services
its exact turn-owned requests at cooperative boundaries, including its external
evidence wait. Idle requests use the ordinary serial dispatcher but never chat,
autoload, experience learning, or a copied foundation model.
"""
from __future__ import annotations

import threading
from contextlib import nullcontext
from dataclasses import dataclass, field

from omni_core.mcp_sampling import (NativeSamplingCancelled, parse_approved_sampling, sample_native_mcp)
from codec_gateway import CodecRuntimeCancelled


MCP_SAMPLING_CONTROLS = frozenset({"mcp_sampling_enqueue", "mcp_sampling_status",
                                  "mcp_sampling_cancel", "mcp_sampling_consume"})


@dataclass
class SamplingRecord:
    brain_id: str
    request_id: str
    input_sha256: str
    turn_id: str
    request: object
    allocation: object = None
    input_lease: object = None
    state: str = "queued"
    cancel: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None
    error: dict | None = None
    result_lease: object = None

    def release_input(self):
        self.request = None
        allocation, self.allocation = self.allocation, None
        if allocation is not None:
            allocation.release()
        lease, self.input_lease = self.input_lease, None
        if lease is not None:
            lease.release()

    def close(self):
        self.release_input()
        lease, self.result_lease = self.result_lease, None
        if lease is not None:
            lease.release()
        self.result = None


class WorkerMcpSampling:
    def initialize_native_mcp_sampling(self):
        self._mcp_sampling_lock = threading.RLock()
        self._mcp_sampling_records = {}

    def _sampling_status(self, record):
        return {"brainId": record.brain_id, "requestId": record.request_id, "inputSha256": record.input_sha256,
            "state": record.state, **({"result": record.result} if record.result is not None else {}),
            **({"error": record.error} if record.error is not None else {})}

    def _sampling_owned_record(self, params):
        if (not isinstance(params, dict) or set(params) != {"brainId", "requestId", "inputSha256"}
                or any(not isinstance(params[name], str) for name in params)):
            raise ValueError("native sampling control ownership is invalid")
        record = self._mcp_sampling_records.get((params["brainId"], params["requestId"]))
        if record is None or record.input_sha256 != params["inputSha256"]:
            raise ValueError("native sampling control has no matching reviewed request")
        return record

    def _sampling_interrupted(self, record):
        if record.cancel.is_set() or not self.running or self.brains.get(record.brain_id) is None:
            return True
        if not record.turn_id:
            return False
        with self._steering_lock:
            session = self._chat_steering.get((record.brain_id, record.turn_id))
            return bool(session is None or not session.generation_open or session.requested.is_set()
                        or session.suspension_requested.is_set() or session.suspension_freezing
                        or self._cooperative_cancel.is_set())

    def _cancel_sampling_record(self, record, reason):
        record.cancel.set()
        if record.state == "queued":
            record.state = "cancelled"
            record.error = {"code": -32800, "message": reason, "reason": reason}
            record.release_input()

    def native_mcp_sampling_control(self, method, params, admitted_request=None):
        if method == "mcp_sampling_enqueue":
            if not isinstance(params, dict) or not isinstance(params.get("brainId"), str) or not isinstance(params.get("approvedInputJson"), str):
                raise ValueError("native sampling transport ownership is invalid")
            # Merely checking residency is not authority to construct/load a
            # brain. The renderer/server can never call this protocol directly.
            brain = self.brains.get(params["brainId"])
            if brain is None:
                raise ValueError("Explicit Load is required before MCP sampling")
            from stdio_request_admission import decoded_request_bound, retain_request_allocation
            input_lease = brain.resource_policy.reserve_ram(
                decoded_request_bound(params["approvedInputJson"].encode("utf-8")), "MCP reviewed input parsing and ownership")
            try:
                approved = parse_approved_sampling(params)
                if approved.prompt_tokens > int(brain.config.max_seq_len):
                    raise ValueError("native sampling input exceeds selected context; no truncation was performed")
                with self._steering_lock:
                    live = [session for session in self._chat_steering.values() if session.claimed]
                    if approved.turn_id:
                        session = self._chat_steering.get((approved.brain_id, approved.turn_id))
                        if (session is None or not session.claimed or not session.generation_open
                                or session.requested.is_set() or session.suspension_requested.is_set()
                                or session.suspension_freezing or self._cooperative_cancel.is_set()):
                            raise ValueError("native sampling has no supported originating-turn boundary")
                    elif live:
                        raise ValueError("busy chat sampling requires its trusted originating turn")
                with self._mcp_sampling_lock:
                    prior = self._mcp_sampling_records.get((approved.brain_id, approved.request_id))
                    if prior is not None:
                        if prior.input_sha256 != approved.input_sha256:
                            raise ValueError("native sampling request ID was reused with different reviewed input")
                        return {**self._sampling_status(prior), "accepted": True, "duplicate": True,
                                "boundary": "turn" if prior.turn_id else "serial"}
                    allocation = retain_request_allocation(admitted_request) if admitted_request is not None else None
                    record = SamplingRecord(approved.brain_id, approved.request_id, approved.input_sha256,
                                            approved.turn_id, approved, allocation, input_lease)
                    input_lease = None
                    self._mcp_sampling_records[(approved.brain_id, approved.request_id)] = record
                    return {**self._sampling_status(record), "accepted": True, "boundary": "turn" if record.turn_id else "serial"}
            finally:
                if input_lease is not None:
                    input_lease.release()
        with self._mcp_sampling_lock:
            record = self._sampling_owned_record(params)
            if method == "mcp_sampling_status":
                return self._sampling_status(record)
            if method == "mcp_sampling_cancel":
                self._cancel_sampling_record(record, "request-cancelled")
                return self._sampling_status(record)
            if method == "mcp_sampling_consume":
                if record.state not in {"complete", "cancelled", "failed"}:
                    raise ValueError("native sampling cannot consume an active request")
                result = {**self._sampling_status(record), "consumed": True}
                del self._mcp_sampling_records[(record.brain_id, record.request_id)]
                record.close()
                return result
        raise ValueError("unsupported native sampling control")

    def _execute_native_mcp_sample(self, record, brain):
        with self._mcp_sampling_lock:
            if record.state != "queued":
                return self._sampling_status(record)
            if self._sampling_interrupted(record):
                self._cancel_sampling_record(record, "originating-turn-interrupted")
                return self._sampling_status(record)
            record.state = "running"
        result, error, result_lease = None, None, None
        try:
            gateway = getattr(self, "_codec_gateway", None)
            owner = gateway.current_owner() if gateway is not None else None
            # Keep the real outer RPC's codec correlation while using this
            # exact private request's cancellation flags. A codec challenge
            # must never borrow the main turn's control-tick cancellation.
            scope = gateway.scope(owner, lambda: self._sampling_interrupted(record)) if owner is not None else nullcontext()
            with scope:
                result = sample_native_mcp(brain, record.request, cancelled=lambda: self._sampling_interrupted(record))
            # The result survives after the private token/cache lease closes,
            # until main has received it and explicitly consumes the receipt.
            size = 8192 + len(result["content"]["text"].encode("utf-8")) * 8
            result_lease = brain.resource_policy.reserve_ram(size, "MCP reviewed native result awaiting delivery")
            if self._sampling_interrupted(record):
                raise NativeSamplingCancelled("originating-turn-interrupted")
        except (NativeSamplingCancelled, CodecRuntimeCancelled) as interrupted:
            error = {"code": -32800, "message": str(interrupted), "reason": "owner-interrupted"}
        except Exception as failure:
            error = {"code": -32000, "message": str(failure), "reason": "native-sampling-failed"}
        finally:
            with self._mcp_sampling_lock:
                record.release_input()
                if error is not None:
                    record.state = "cancelled" if error["code"] == -32800 else "failed"
                    record.error = error
                    if result_lease is not None:
                        result_lease.release()
                else:
                    record.state, record.result, record.result_lease = "complete", result, result_lease
        return self._sampling_status(record)

    def service_native_mcp_sampling(self, brain_id, turn_id):
        """Called only by the existing writer, never by the control reader."""
        brain = self.brains.get(brain_id)
        if brain is None:
            return
        with self._mcp_sampling_lock:
            records = [record for record in self._mcp_sampling_records.values()
                       if record.brain_id == brain_id and record.turn_id == turn_id and record.state == "queued"]
        for record in records:
            self._execute_native_mcp_sample(record, brain)

    def mcp_sample(self, params, _request_id=None):
        with self._mcp_sampling_lock:
            record = self._sampling_owned_record(params)
            if record.turn_id:
                raise ValueError("live turn sampling cannot re-enter the normal neural writer")
        brain = self.brains.get(record.brain_id)
        if brain is None:
            with self._mcp_sampling_lock:
                self._cancel_sampling_record(record, "instance-unloaded")
                return self._sampling_status(record)
        return self._execute_native_mcp_sample(record, brain)

    def close_native_mcp_turn(self, brain_id, turn_id):
        with self._mcp_sampling_lock:
            for record in self._mcp_sampling_records.values():
                if record.brain_id == brain_id and record.turn_id == turn_id and record.state in {"queued", "running"}:
                    self._cancel_sampling_record(record, "originating-turn-ended")

    def cancel_native_mcp_for_brain(self, brain_id, reason="instance-unloaded"):
        with self._mcp_sampling_lock:
            for record in self._mcp_sampling_records.values():
                if record.brain_id == brain_id:
                    if record.state in {"queued", "running"}:
                        self._cancel_sampling_record(record, reason)
                    # Terminal receipts retain only actual result/error data,
                    # not a borrowed brain, request input or bound callback.
                    record.release_input()

    def shutdown_native_mcp_sampling(self):
        with self._mcp_sampling_lock:
            for record in self._mcp_sampling_records.values():
                record.cancel.set()
                record.close()
            self._mcp_sampling_records.clear()
