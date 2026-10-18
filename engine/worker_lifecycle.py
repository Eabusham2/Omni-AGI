"""Explicit no-load observations and exact supported native turn suspension.

Control-reader methods only inspect metadata or set flags. All learned-state
writes and releases stay on the worker's existing serial neural dispatcher.
"""
from __future__ import annotations

import copy
import re
from types import SimpleNamespace

from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.chat_tool_observation import ChatToolObservationInbox


class RootOwnedObservationInbox:
    """Root permission receipts enter only their exact borrowed native inbox."""
    def __init__(self, inbox, execution):
        self.inbox, self.execution = inbox, execution

    def offer(self, value, allocation=None):
        inward = self.execution.inward_observation(value)
        receipt = self.inbox.offer(inward, allocation)
        return {**receipt, "brainId": self.execution.root.brain_id, "turnId": self.execution.root_turn_id,
            "observationId": value["observationId"]}

    def close(self):
        self.inbox.close()


class WorkerLifecycle:
    def reviewed_execution_runtime_config(self, brain, raw, *, store=None):
        """Resource edits may not change a preserved turn's learned policy."""
        if not isinstance(raw, dict):
            raise ValueError("reviewed runtime settings must be an object")
        store = store or ExecutionCheckpointStore(brain.engine_path, brain.brain_id, policy=brain.resource_policy)
        current = store.current()
        from omni_core.connected_execution import connected_execution_status
        inherited = connected_execution_status(brain)
        suspended = current is not None and current["status"] == "suspended"
        minimum = max(int(brain.current_context.get("tokenCount", 0)), inherited["minimumContextTokens"],
            int(current["minimumContextTokens"]) if suspended else 0)
        requested = raw.get("contextWindowTokens", brain.config.max_seq_len)
        if type(requested) is not int or requested < minimum:
            raise ValueError("runtime context cannot be lower than a retained native execution frontier")
        if suspended or inherited["rootMutationHeld"]:
            from omni_core.runtime_load_config import REVIEWED_RUNTIME_KEYS
            return {key: value for key, value in raw.items() if key in REVIEWED_RUNTIME_KEYS}
        return raw

    def _execution_store(self, params, *, brain=None):
        identifier = self._brain_id(params)
        storage = self._storage(params, identifier)
        if brain is not None and brain.storage_path != storage:
            raise ValueError("native continuation storage owner differs")
        return ExecutionCheckpointStore(storage / "engine", identifier,
            policy=getattr(brain, "resource_policy", None) or self._stdio_resource_policy())

    def instance_status(self, params, request_id=None):
        del request_id
        identifier = self._brain_id(params)
        brain = self.brains.get(identifier)
        store = self._execution_store(params, brain=brain)
        current = store.current()
        suspended = current if current is not None and current["status"] == "suspended" else None
        minimum = max(0, int(brain.current_context.get("tokenCount", 0))) if brain is not None else 0
        if suspended is not None:
            minimum = max(minimum, suspended["minimumContextTokens"])
        status = {"brainId": identifier, "learnedRuntimeResident": brain is not None,
            "minimumContextTokens": minimum, "reconnect": []}
        from omni_core.connected_execution import connected_execution_status
        if brain is not None:
            inherited = connected_execution_status(brain)
        else:
            metadata = store._read(store.engine / "brain.json")
            inherited = connected_execution_status(SimpleNamespace(connected_executions=metadata.get("connected_executions")))
            recent = metadata.get("recent_token_context", [])
            context_count = (metadata.get("current_context") or {}).get("tokenCount", 0)
            if not isinstance(recent, list) or type(context_count) is not int or context_count < 0:
                raise ValueError("saved working-context occupancy is invalid")
            minimum = max(minimum, len(recent), context_count)
            status["minimumContextTokens"] = minimum
        if inherited["entries"]:
            status["inheritedExecutions"] = inherited
            minimum = max(minimum, inherited["minimumContextTokens"])
            status["minimumContextTokens"] = minimum
            pending_entry = next((entry for entry in inherited["entries"] if entry["status"] in {"queued", "suspended"}), None)
            if pending_entry is not None:
                from omni_core.connected_execution import resolve_execution_engine, validated_connected_executions
                metadata = store._read(store.engine / "brain.json")
                owned = next(entry for entry in validated_connected_executions(metadata.get("connected_executions"))["entries"] if entry["id"] == pending_entry["id"])
                source = resolve_execution_engine(store.engine, owned, resumed=bool(owned.get("resumeEnginePath")))
                source_store = ExecutionCheckpointStore(source, owned["parentId"], policy=store.policy)
                manifest = source_store.current(owned.get("resumeCheckpointId", owned["sourceCheckpointId"]))
                try:
                    source_store.require_supported_frontier(manifest)
                    status["resumeCompatibility"] = {"compatible": True}
                except (ValueError, RuntimeError, OSError) as error:
                    status["resumeCompatibility"] = {"compatible": False, "reason": str(error)}
        if suspended is not None:
            status.update(resume=copy.deepcopy(suspended["frontier"]), presentation=copy.deepcopy(suspended["presentation"]))
            try:
                store.require_supported_frontier(suspended)
                status["resumeCompatibility"] = {"compatible": True}
            except (ValueError, RuntimeError, OSError) as error:
                status["resumeCompatibility"] = {"compatible": False, "reason": str(error)}
        with self._steering_lock:
            sessions = [session for (brain_id, _), session in self._chat_steering.items() if brain_id == identifier]
            if sessions:
                minimum = max(minimum, *(session.minimum_context_tokens for session in sessions))
                status["minimumContextTokens"] = minimum
            if any(session.suspension_requested.is_set() for session in sessions):
                status["waitingFor"] = {"kind": "native-boundary", "label": "Waiting for the next supported native text token checkpoint"}
        return status

    def inherited_execution_presentation(self, params, request_id=None):
        del request_id
        from omni_core.connected_execution import _digest, resolve_execution_engine, validated_connected_executions
        identifier = self._brain_id(params)
        root_store = self._execution_store(params, brain=self.brains.get(identifier))
        metadata = root_store._read(root_store.engine / "brain.json")
        entries = validated_connected_executions(metadata.get("connected_executions"))["entries"]
        entry = next((value for value in entries if value["id"] == params.get("executionId")), None)
        if entry is None:
            raise ValueError("The inherited turn is not owned by this instance")
        source = resolve_execution_engine(root_store.engine, entry, resumed=bool(entry.get("resumeEnginePath")))
        store = ExecutionCheckpointStore(source, entry["parentId"], policy=root_store.policy)
        manifest = store.current(entry.get("resumeCheckpointId", entry["sourceCheckpointId"]))
        if (_digest(manifest) != entry.get("resumeManifestSha256", entry["sourceManifestSha256"])
                or manifest["bindingId"] != entry["sourceBindingId"]):
            raise ValueError("The inherited published frontier lost its exact source binding")
        return {"brainId": identifier, "executionId": entry["id"], "parentId": entry["parentId"],
            "originalTurnId": entry["originalTurnId"], "rootHumanMessageId": entry["rootHumanMessageId"],
            "frontier": {**copy.deepcopy(manifest["frontier"]), "turnId": entry["rootTurnId"]},
            "presentation": copy.deepcopy(manifest["presentation"])}

    def request_chat_suspension(self, params):
        identifier, turn_id = self._brain_id(params), params.get("streamId")
        if not isinstance(turn_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", turn_id):
            raise ValueError("invalid native suspension turn identity")
        with self._steering_lock:
            session = self._chat_steering.get((identifier, turn_id))
            if session is None or not session.generation_open:
                return {"requested": False}
            session.suspension_requested.set()
        return {"requested": True, "waitingFor": {"kind": "native-boundary",
            "label": "Waiting for a supported native text token boundary; Unload is not Stop"}}

    def suspend_instance(self, params, request_id=None):
        del request_id
        identifier = self._brain_id(params)
        brain = self.brains.get(identifier)
        store = self._execution_store(params, brain=brain)
        current = store.current(params.get("expectedCheckpointId"))
        with self._steering_lock:
            if any(owner == identifier for owner, _ in self._chat_steering):
                return {**self.instance_status(params), "unloaded": False,
                    "waitingFor": {"kind": "native-boundary", "label": "The active native turn has not released its checkpoint boundary"}}
        with self._inline_lock:
            if any(record.brain_id == identifier for record in self._inline_generations.values()):
                return {**self.instance_status(params), "unloaded": False,
                    "waitingFor": {"kind": "external-effect", "label": "An owned imagination artifact must finish or be claimed before unloading"}}
        with self._observation_lock:
            if any(session.brain_id == identifier for session in self._observation_sessions.values()):
                return {**self.instance_status(params), "unloaded": False,
                    "waitingFor": {"kind": "external-effect", "label": "Disconnect the live sensory session before unloading; reconnect explicitly after Load"}}
        pending = params.get("pendingObservations", [])
        if not isinstance(pending, list):
            raise ValueError("pending suspended observations must be a list")
        if current is not None and current["status"] == "suspended" and pending:
            worker = self._merge_suspended_observations(current["worker"], pending, brain)
            current = store.update_worker(current["checkpointId"], worker)
        elif pending:
            raise ValueError("pending observations have no owned suspended turn")
        inherited_pending = params.get("pendingInheritedObservations", [])
        if not isinstance(inherited_pending, list):
            raise ValueError("inherited suspended observations must be a list")
        if inherited_pending:
            if brain is None:
                raise ValueError("late inherited effects must be durably staged before native unload acknowledgement")
            from omni_core.connected_execution import stage_inherited_observations
            for entry in inherited_pending:
                if not isinstance(entry, dict) or set(entry) != {"executionId", "observations"} or not isinstance(entry["observations"], list):
                    raise ValueError("inherited suspended evidence has invalid ownership")
                stage_inherited_observations(brain, entry["executionId"], entry["observations"])
        if brain is not None:
            if current is None or current["status"] != "suspended":
                clock = brain._owned_activity_clock()
                previously_paused = clock.paused
                clock.pause()
                try:
                    brain.save()
                except BaseException:
                    if not previously_paused:
                        clock.resume()
                    raise
            cancel_samples = getattr(self, "cancel_native_mcp_for_brain", None)
            if callable(cancel_samples): cancel_samples(identifier, "instance-unloaded")
            brain.close()
            self.brains.pop(identifier, None)
        store.mark_unloaded()
        self._release_owner_lease(identifier)
        output_leases = getattr(self, "_spontaneous_output_leases", None)
        temporary_media = output_leases.discard_brain(identifier) if output_leases is not None else {"complete": True, "preservedUnverifiedJobIds": []}
        return {**self.instance_status(params), "unloaded": True, "temporaryMediaCleanup": temporary_media}

    @staticmethod
    def _merge_suspended_observations(worker_state, pending, brain):
        policy = getattr(brain, "resource_policy", None)
        reserve = getattr(policy, "reserve_ram", None)
        inbox = ChatToolObservationInbox.from_suspension_state(worker_state["inbox"],
            worker_state["brainId"], worker_state["turnId"],
            lambda size: policy is None or not policy.status(estimated_ram_bytes=size).get("memoryPressure"), reserve_parse=reserve)
        try:
            for observation in pending:
                if not inbox.offer(observation).get("accepted"):
                    raise ValueError("suspended observation could not be admitted; original checkpoint retained")
            return {**worker_state, "inbox": inbox.suspension_state()}
        finally:
            inbox.close()

    def resume_chat(self, params, request_id=None):
        from worker import RpcFault
        if any(key in params for key in ("input", "text", "message")):
            raise RpcFault(-32602, "Resume uses the owned checkpoint, never a new input-bearing chat")
        identifier = self._brain_id(params)
        brain = self.brains.get(identifier)
        if brain is None:
            raise RpcFault(-32004, "Explicit Load is required before continuing a saved turn")
        store = self._execution_store(params, brain=brain)
        checkpoint_id = params.get("checkpointId")
        if not isinstance(checkpoint_id, str) or not re.fullmatch(r"[a-f0-9]{32}", checkpoint_id):
            raise RpcFault(-32602, "Resume requires its exact native checkpoint identity")
        saved = store.current(checkpoint_id)
        turn_id = saved["frontier"]["turnId"]
        if params.get("streamId", turn_id) != turn_id:
            raise RpcFault(-32602, "Resume stream differs from the saved native turn")
        completed = next((receipt for receipt in brain.completed_chat_turns if receipt.get("turnId") == turn_id
            and receipt.get("inputSha256") == saved["frontier"]["inputSha256"]), None)
        if completed is not None:
            result = brain._completed_chat_result(completed)
            store.finish(saved["checkpointId"], turn_id=turn_id)
            return result
        if saved["status"] != "suspended":
            raise RpcFault(-32602, "The saved native turn is not suspended")
        pending = params.get("pendingObservations", [])
        if not isinstance(pending, list):
            raise RpcFault(-32602, "pendingObservations must be a list")
        if pending:
            saved = store.update_worker(saved["checkpointId"], self._merge_suspended_observations(saved["worker"], pending, brain))
        manifest, payload, reader = store.restore(brain, saved["checkpointId"])
        from worker import ChatSteeringState
        key = (identifier, turn_id)
        session = ChatSteeringState(identifier, turn_id, request_id=str(request_id or ""), claimed=True)
        with self._steering_lock:
            reserved = self._chat_steering.get(key)
            if reserved is not None:
                if reserved.claimed or reserved.request_id != str(request_id or ""):
                    reader.close(); payload["context"]["before_parameters"].close()
                    raise RpcFault(-32602, "The retained turn already has an active worker owner")
                session = reserved
                session.claimed = True
            self._chat_steering[key] = session
        try:
            result = self._chat({**params, "streamId": turn_id}, request_id, session.requested.is_set,
                resume_payload=payload, resume_manifest=manifest)
            if not result.get("suspended"):
                store.finish(manifest["checkpointId"], turn_id=turn_id)
            return result
        finally:
            reader.close()
            payload["context"]["before_parameters"].close()
            close_samples = getattr(self, "close_native_mcp_turn", None)
            if callable(close_samples): close_samples(identifier, turn_id)
            with self._steering_lock:
                if session.observation_inbox is not None:
                    session.observation_inbox.close()
                self._chat_steering.pop(key, None)

    def resume_inherited_chat(self, params, request_id=None):
        """Continue an adopted frontier via its registered regional owners."""
        from worker import ChatSteeringState, RpcFault
        from omni_core.connected_execution import ConnectedExecutionQueue
        if any(key in params for key in ("input", "text", "message")):
            raise RpcFault(-32602, "Inherited resume never admits another input")
        root = self._get(params)
        execution_id = params.get("executionId")
        if not isinstance(execution_id, str) or not re.fullmatch(r"[a-f0-9]{64}", execution_id):
            raise RpcFault(-32602, "Inherited resume requires its owned execution identity")
        with ConnectedExecutionQueue(root).prepare(execution_id) as execution:
            if not hasattr(execution, "payload"):
                return execution.run()
            turn_id = execution.root_turn_id
            if params.get("streamId", turn_id) != turn_id:
                raise RpcFault(-32602, "Inherited resume stream has another root turn owner")
            key = (root.brain_id, turn_id)
            session = ChatSteeringState(root.brain_id, turn_id, request_id=str(request_id or ""), claimed=True)
            with self._steering_lock:
                reserved = self._chat_steering.get(key)
                if reserved is not None:
                    if reserved.claimed or reserved.request_id != str(request_id or ""):
                        raise RpcFault(-32602, "The inherited frontier already has an active writer")
                    session = reserved; session.claimed = True
                self._chat_steering[key] = session
            try:
                return self._chat({**params, "streamId": turn_id}, request_id, session.requested.is_set,
                    resume_payload=execution.payload, resume_manifest=execution.manifest, execution_session=execution)
            finally:
                close_samples = getattr(self, "close_native_mcp_turn", None)
                if callable(close_samples): close_samples(root.brain_id, turn_id)
                with self._steering_lock:
                    if session.observation_inbox is not None: session.observation_inbox.close()
                    self._chat_steering.pop(key, None)

    def stop_inherited_chat(self, params, request_id=None):
        del request_id
        from omni_core.connected_execution import ConnectedExecutionQueue, stop_unloaded_connected_execution
        identifier = self._brain_id(params)
        root = self.brains.get(identifier)
        with self._steering_lock:
            if any(owner == identifier for owner, _ in self._chat_steering):
                raise ValueError("An active inherited turn must reach its live Stop boundary first")
        if root is not None:
            return ConnectedExecutionQueue(root).stop(params.get("executionId"))
        storage = self._storage(params, identifier)
        _lease, acquired = self._owner_lease(identifier, storage)
        try:
            return stop_unloaded_connected_execution(storage, identifier, params.get("executionId"), self._stdio_resource_policy())
        finally:
            if acquired: self._release_owner_lease(identifier)

    def stop_suspended_chat(self, params, request_id=None):
        del request_id
        identifier = self._brain_id(params)
        _lease, acquired = self._owner_lease(identifier, self._storage(params, identifier))
        try:
            return self._stop_suspended_chat_owned(params)
        finally:
            if acquired: self._release_owner_lease(identifier)

    def _stop_suspended_chat_owned(self, params):
        store = self._execution_store(params, brain=self.brains.get(self._brain_id(params)))
        checkpoint_id = params.get("checkpointId")
        if not isinstance(checkpoint_id, str) or not re.fullmatch(r"[a-f0-9]{32}", checkpoint_id):
            raise ValueError("Stop requires its exact suspended native checkpoint identity")
        value = store.current(checkpoint_id)
        turn_id = params.get("turnId")
        with self._steering_lock:
            if (self._brain_id(params), turn_id) in self._chat_steering:
                raise ValueError("Stop on an active turn must use its live generation control")
        store.finish(value["checkpointId"], turn_id=turn_id, stopped=True)
        return {"stopped": True, "brainId": self._brain_id(params), "turnId": turn_id,
            "checkpointId": value["checkpointId"], "inputReceiptRetained": True}
