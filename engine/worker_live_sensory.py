"""Capture controls queue metadata; only the real reply owner encodes packets."""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from live_sensory_queue import LIVE_SENSORY_CONTROL, LiveSensoryInbox, capture_permission_hash


class LiveSensoryIntegrationFailed(RuntimeError):
    """An actual owner must recover; an unknown partial effect is not retryable."""


class WorkerLiveSensory:
    def initialize_live_sensory(self):
        self._live_sensory_lock = threading.RLock()
        self._live_sensory = {}

    def _live_sensory_current(self, key, session, request_id):
        with self._active_request_lock, self._steering_lock:
            return (self._active_request is not None and self._active_request[1] == request_id
                and self._active_request[0] in {"chat", "resume_chat", "resume_inherited_chat"}
                and self._chat_steering.get(key) is session and session.claimed and session.generation_open
                and not session.requested.is_set() and not session.suspension_requested.is_set()
                and not session.suspension_freezing and not self._cooperative_cancel.is_set())

    @staticmethod
    def _live_session_binding(session):
        return {"brainId": session.brain_id, "captureRevision": session.capture_revision,
            "permissionSha256": capture_permission_hash(session.permission),
            "maxPacketBytes": session.max_packet_bytes, "maxInFlight": session.max_in_flight}

    def open_live_sensory(self, key, request_id):
        with self._steering_lock:
            session = self._chat_steering.get(key)
            if session is None or session.request_id != str(request_id or ""):
                raise ValueError("capture inbox has no exact native request owner")
        inbox = LiveSensoryInbox(key[0], key[1], str(request_id or ""), self._send,
            lambda: self._live_sensory_current(key, session, str(request_id or "")))
        with self._live_sensory_lock:
            if key in self._live_sensory:
                raise ValueError("capture inbox already belongs to this turn")
            self._live_sensory[key] = inbox
        return inbox

    def close_live_sensory(self, key, inbox):
        with self._live_sensory_lock:
            if self._live_sensory.get(key) is inbox:
                self._live_sensory.pop(key, None)
        inbox.close()

    def live_sensory_control(self, request):
        """Reader lane: owned metadata/bytes only, no brain or neural access."""
        from stdio_request_admission import retain_request_allocation
        params = request.get("params", {})
        key = (params.get("brainId"), params.get("streamId"))
        with self._live_sensory_lock:
            inbox = self._live_sensory.get(key)
        if inbox is None or not inbox.current():
            raise ValueError("live sensory request does not own an active reply")
        payload = params.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("live sensory request has no capture payload")
        operation, session_id = params.get("operation"), payload.get("sessionId")
        if operation in {"stop_observation", "cancel_observation"}:
            with self._observation_lock:
                session = self._observation_sessions.get(session_id)
                if session is None or session.brain_id != key[0]:
                    raise ValueError("capture retirement has another session owner")
                session.retired.set()
            # Retire before Main waits for any outstanding packet. No pending
            # capture bytes are carried into another turn or resume checkpoint.
            inbox.retire_session(session_id)
            if payload.get("retireOnly") is True:
                result = {**self._observation_session_payload(session), "state": "stopping"}
            else:
                result = self._finish_observation(payload, cancelled=operation == "cancel_observation")
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": result}
        with self._observation_lock:
            session = self._observation_sessions.get(session_id)
            binding = self._live_session_binding(session) if session is not None and not session.retired.is_set() else None
        allocation = retain_request_allocation(request)
        try:
            inbox.offer(request, allocation, binding)
        except BaseException:
            if allocation is not None: allocation.release()
            raise
        # Deliberately deferred. Only an actual integration/failure below sends
        # this RPC response. Queue admission is not a consumed observation.
        return None

    def drain_live_sensory(self, inbox, brain, authority_brain, current, step, episode_id):
        from omni_core.whole_path_scheduler import world_effect_barrier
        bindings, refined = [], current
        with world_effect_barrier():
            while True:
                item = inbox.take()
                if item is None: break
                integration_started = False
                binding_allocation = None
                try:
                    payload = item.payload()
                    if not isinstance(payload.get("storagePath"), str) or Path(payload["storagePath"]).resolve() != authority_brain.storage_path.resolve():
                        raise ValueError("live capture storage has another native owner")
                    if not inbox.current():
                        raise ValueError("live capture turn retired before integration")
                    if item.operation == "start_observation":
                        result = self.start_observation(payload, None)
                    elif item.operation == "resolve_observation_control":
                        integration_started = True  # records owned operational evidence
                        result = self.resolve_observation_control(payload, None, persist=False)
                    else:
                        with self._observation_lock:
                            session = self._observation_sessions.get(item.session_id)
                            if session is None or session.retired.is_set():
                                raise ValueError("capture session retired before integration")
                            binding = self._live_session_binding(session)
                            if payload.get("captureRevision") != binding["captureRevision"] or payload.get("permissionSha256") != binding["permissionSha256"]:
                                raise ValueError("capture revision or permission changed before integration")
                        modality, mime, data, settings, sequence, timestamp = self._decode_observation_packet(session, payload)
                        if payload.get("videoRuntime") is not None:
                            self.configure_video_runtime(payload["videoRuntime"], None)
                        if hashlib.sha256(data).hexdigest() != payload.get("packetSha256"):
                            raise ValueError("live capture byte checksum changed")
                        if not inbox.current() or session.retired.is_set():
                            raise ValueError("capture retired before neural admission")
                        # Pending proof can grow while a real Ponder/search
                        # runs. Charge its metadata before any neural effects;
                        # the ordinary packet allocation is still held too.
                        binding_allocation = brain.resource_policy.reserve_ram(
                            8192 + 8 * sum(len(str(value)) for value in (*inbox.owner, item.session_id, episode_id)),
                            "live sensory binding provenance")
                        integration_started = True
                        observation, refined = brain.condition_live_packet(refined,
                            modality=modality, mime_type=mime, payload=data, session_id=item.session_id,
                            sequence=sequence, timestamp_ms=timestamp, retention=session.retention,
                            settings=settings, permission_source=session.permission["source"],
                            episode_assembly_id=episode_id,
                            cancelled=lambda: not inbox.current() or session.retired.is_set())
                        if not inbox.current() or session.retired.is_set():
                            raise RuntimeError("capture retired during native integration")
                        # Updating the metadata after the actual conditioning is
                        # the linearized consume point, never enqueue or decode.
                        with self._observation_lock:
                            session.last_sequence, session.last_timestamp_ms = sequence, timestamp
                            session.packets_accepted += 1; session.bytes_accepted += len(data)
                        source = ["omni-live-sensory-v1", inbox.owner[0], inbox.owner[1], item.session_id,
                            str(sequence), str(payload["captureRevision"]), observation["packetSha256"]]
                        identifier = hashlib.sha256("\0".join(source).encode()).hexdigest()
                        bindings.append({"observationId": identifier, "observationKind": "live-sensory",
                            "sessionId": item.session_id, "sequence": sequence,
                            "captureRevision": payload["captureRevision"], "packetSha256": observation["packetSha256"],
                            "episodeAssemblyId": observation["assemblyId"], "generationStep": int(step)})
                        inbox.retain_binding(binding_allocation)
                        binding_allocation = None
                        result = {"session": self._observation_session_payload(session), "observation": observation,
                            "actions": [], "sensoryBinding": bindings[-1]}
                    inbox.finish(item, result=result)
                except Exception as error:
                    if integration_started and brain is not authority_brain:
                        # Fence before callbacks/finally blocks can replace
                        # the raised exception: these regional owners are
                        # borrowed from the actual connected root.
                        authority_brain._connected_recovery_required = True
                    delivery_error = None
                    try:
                        inbox.finish(item, error={"code": getattr(error, "code", -32000), "message": str(error),
                            "data": {"packetConsumed": None if integration_started else False,
                                "effectDisposition": "unknown" if integration_started else "not-started",
                                "automaticRetryAllowed": False, "workerRestarted": False}})
                    except Exception as failure:
                        delivery_error = failure
                    finally:
                        if binding_allocation is not None:
                            binding_allocation.release()
                    if integration_started:
                        # Settling may already have touched packed state. Do
                        # not continue and later save it as an unconsumed
                        # packet, or claim an in-memory snapshot rolled it all
                        # back. The existing failed-chat checkpoint recovery
                        # owns this actual turn. Borrowed inherited owners
                        # instead require the explicit shared-root fence.
                        try:
                            inbox.close("capture-integration-effects-unknown")
                        finally:
                            raise LiveSensoryIntegrationFailed(str(error)) from error
                    if delivery_error is not None:
                        raise delivery_error
        return {"memoryBias": refined, "observationBindings": bindings} if bindings else {}
