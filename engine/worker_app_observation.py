"""Loaded-only serial writer boundary for reviewed App experience admission."""
from omni_core.app_observation import (AppObservationCommitError, admit_native_app_observation,
                                       native_app_observation_frontier)


class WorkerAppObservation:
    def app_observation_frontier(self, params, request_id=None):
        del request_id
        if not isinstance(params, dict) or set(params) != {"brainId", "storagePath"}:
            raise ValueError("App frontier query requires its exact loaded storage owner")
        return native_app_observation_frontier(self._get(params))

    def _release_failed_app_observation_owner(self, brain, *, committed, reason="app-observation-checkpoint-boundary"):
        identifier = brain.brain_id
        warnings = []
        try:
            self.cancel_native_mcp_for_brain(identifier, "Native transaction requires explicit saved-state reload")
        except Exception as error:
            warnings.append(type(error).__name__)
        self.brains.pop(identifier, None)
        try:
            brain.close()
        except Exception as error:
            warnings.append(type(error).__name__)
        try:
            self._release_owner_lease(identifier)
        except Exception as error:
            warnings.append(type(error).__name__)
        release = {"format": "omni-native-owner-release", "version": 1, "brainId": identifier,
            "reason": reason, "committed": committed,
            "reloadRequired": True, "nativeOwnerReleased": True,
            **({"cleanupWarnings": warnings} if warnings else {})}
        # The typed result/error remains authority if notification delivery
        # fails. Do not replace an already durable ACK with telemetry failure.
        try:
            self.notify("native-owner-released", brain_id=identifier, data=release)
        except Exception:
            pass
        return release

    def admit_app_observation(self, params, request_id=None):
        request_cancelled = self._request_cancel_check(request_id)
        cancelled = lambda: not self.running or request_cancelled()
        # This method is ordinary serialized dispatch only, never reader-side
        # control inference or a reentry behind an inline tool's own writer.
        brain = self._get(params)
        with self._steering_lock:
            if any(owner == brain.brain_id for owner, _ in self._chat_steering):
                raise RuntimeError("App observation waits for the active native turn to release its writer")
        current = self._execution_store(params, brain=brain).current()
        if current is not None and current["status"] == "suspended":
            raise RuntimeError("App observation waits for the retained native turn; no decoder frontier was changed")
        with self._inline_lock:
            if any(record.brain_id == brain.brain_id for record in self._inline_generations.values()):
                raise RuntimeError("App observation waits for the owned native imagination boundary")
        try:
            result = admit_native_app_observation(brain, params, cancelled=cancelled)
        except AppObservationCommitError as error:
            from worker import RpcFault
            release = self._release_failed_app_observation_owner(brain, committed=error.committed)
            raise RpcFault(-32031, str(error), {**release, "nativeAppObservation": True, "learningApplied": error.committed}) from error
        except InterruptedError as error:
            if not cancelled():
                raise
            from worker import RpcFault
            raise RpcFault(-32800, str(error), {"brainId": brain.brain_id, "nativeAppObservation": True,
                "cancelled": True, "safeBoundary": True, "learningApplied": False, "nativeOwnerReleased": False}) from error
        if result.get("reloadRequired"):
            self._release_failed_app_observation_owner(brain, committed=True)
            result["nativeOwnerReleased"] = True
        return result
