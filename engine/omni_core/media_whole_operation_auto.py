"""Post-delivery qualification of real prepared media, never cached output.

The first requested artifact and previews run once on the ordinary route. Only
after its response may the existing serial worker visit one private complete
sample at a time. New user work retires that pool using flags, without touching
the user's generation, context, learned state, or worker process.
"""
from __future__ import annotations

import copy
import threading
import uuid

from .whole_path_execution import digest, select_whole_path_candidate
from .whole_path_scheduler import select_complete_operation, qualification_allowed


class _MediaFlags:
    def __init__(self, controller, event): self.controller, self.event = controller, event
    def is_set(self): return self.event.is_set() or self.controller._cancel.is_set() or self.controller.ops.parent_cancelled()


class WholeMediaOperationPolicy:
    def retire_private_media(self, reason="new-real-work"):
        """Reader/control lane: flags and integers only, no tensor/file cleanup."""
        with self._lock:
            self._media_request_revision = getattr(self, "_media_request_revision", 0) + 1
            pending = getattr(self, "_media_pending", None)
            if pending is not None: pending["retiredReason"] = reason; pending["event"].set()

    def release_private_media(self):
        """Existing serial owner lane only."""
        with self._lock:
            pending, self._media_pending = getattr(self, "_media_pending", None), None
        if pending is not None: self.ops.media_operations.release_operation(pending["operation"])

    def consume_media(self, brain, prepared, producer, preview, directory, cancelled, started_at):
        ordinary = lambda: producer(prepared["idea"], preview, directory, cancelled, started_at)
        if not qualification_allowed(): return ordinary()
        media = getattr(self.ops, "media_operations", None)
        with self._lock:
            eligible = (not self._closed and not self._cancel.is_set() and not self._active
                and self._phase in {"awaiting-operation", "active"} and media is not None)
            review = copy.deepcopy(self._review) if eligible else None
            revision = getattr(self, "_media_request_revision", 0)
        if not eligible: return ordinary()
        kind = "native.media." + str(prepared.get("modality", "")) + "-artifact"
        if kind not in review["supportedWorkKinds"] or not media.eligible(brain, prepared): return ordinary()
        operation, handed, real_started, produced = None, False, False, None
        try:
            self.release_private_media()
            if self.ops.hardware_binding(self.owner_session) != review["binding"]: return ordinary()
            operation = media.capture_operation(prepared, producer, preview is not None, self.owner_session, review)
            identity, key = operation["identity"], operation["stratumKey"]
            if identity["workKind"] not in review["supportedWorkKinds"]: return ordinary()
            with self._lock:
                self._active, self._kind = True, identity["workKind"]
                plan = self._plans.get(key)
            if plan is not None and plan["nativeState"] != identity["nativeState"]:
                with self._lock: removed = self._plans.pop(key, None); self._revision += 1
                if removed is not None: removed["lease"].release()
                plan = None
            if plan is not None:
                candidate = plan["public"]["candidate"]
                if candidate == review["currentCandidate"]:
                    with self._lock: self._last = plan["public"]; self._reason = "tested-home-media-route-no-repeat-pool"
                    real_started = True
                    return ordinary()
                if media.auto_allowed(candidate):
                    def begin():
                        nonlocal real_started
                        # Once actual preview/artifact work starts it is never
                        # retried as a fallback, even if later output fails.
                        real_started = True
                    produced, actual = media.execute_live(candidate, operation, preview, directory, cancelled, started_at, begin)
                    media.check_output(actual, plan["descriptor"])
                    with self._lock:
                        self._last = plan["public"]; self._phase, self._reason = "active", "fresh-media-on-qualified-layout-inference"
                    self.ops.append_receipt("native-media-use-" + uuid.uuid4().hex,
                        {"phase": "reused-route", "operationId": self._operation, "workKind": identity["workKind"],
                         "planId": plan["public"]["id"], "workloadSha256": identity["workloadSha256"],
                         "reuseIsInferenceForUnseenInput": True, "previousArtifactWasReused": False,
                         "actualOutputContractSha256": actual.contract})
                    return produced
            # No candidate pool holds up the first real preview or result.
            real_started = True
            produced, baseline = media.execute_genuine(operation, preview, directory, cancelled, started_at)
            media.after_genuine(operation, baseline, self.owner_session)
            with self._lock:
                if (not self._closed and not self._cancel.is_set() and revision == getattr(self, "_media_request_revision", 0)):
                    self._media_pending = {"operation": operation, "baseline": baseline, "review": review,
                        "event": threading.Event(), "sample": -1, "candidate": 0, "issued": "0" * 64, "count": 0,
                        "job": "native-media-auto-" + uuid.uuid4().hex, "retiredReason": None}
                    handed = True
                    self._phase, self._reason = "active", "real-media-delivered-private-pool-awaits-idle-serial-boundary"
            return produced
        except BaseException as error:
            with self._lock:
                self._reason = "media-route-not-qualified:" + type(error).__name__
                if not self._cancel.is_set(): self._phase = "failed-fallback"
            if getattr(error, "native_owner_restore_required", False): raise
            # A post-result proof/evidence refusal cannot turn a completed
            # genuine artifact into an apparent generation failure.
            if produced is not None: return produced
            if real_started: raise
            return ordinary()
        finally:
            if operation is not None and not handed: media.release_operation(operation)
            with self._lock: self._active = self._committing = False
            if self._cancel.is_set(): self._release_all()

    def step_private_media(self):
        """One whole private workload, called only after the real RPC response."""
        with self._lock:
            pending = getattr(self, "_media_pending", None)
            if pending is None or self._active: return False
            retired = pending["event"].is_set() or self._closed or self._cancel.is_set()
        if retired:
            self.release_private_media(); return False
        media, operation, review = self.ops.media_operations, pending["operation"], pending["review"]
        flags = _MediaFlags(self, pending["event"])
        failure, trial, observer, receipt = None, None, None, None
        try:
            def current():
                if flags.is_set(): raise InterruptedError("private media pool retired for real user work")
                if (self.ops.hardware_binding(self.owner_session) != review["binding"]
                        or not media.unchanged(operation, self.owner_session)):
                    raise RuntimeError("prepared media source, codec, policy or learned owners changed")
            current()
            sample, index = pending["sample"], pending["candidate"]
            candidate = review["candidates"][index]
            with self._lock:
                self._active, self._phase = True, "qualifying"
                self._progress = {"candidate": index+1, "candidates": len(review["candidates"]),
                    "sample": max(0, sample+1), "samplesPerCandidate": self._samples,
                    "stage": "warming" if sample < 0 else "measuring"}
            observed = operation["identity"] if sample >= 0 else {**operation["identity"],
                "cache": {**operation["identity"]["cache"], "backendInitialization": "unknown"}}
            with media.observation_resources(candidate, operation["identity"]["budgetSha256"]) as resources:
                observer = self.recorder.begin(observed, self.primitive(candidate), synchronize=media.synchronize,
                    counters=media.counters, resources=resources)
                try:
                    trial = media.execute_private(candidate, operation, observer, flags)
                    if trial.contract != pending["baseline"].contract:
                        raise RuntimeError("complete media neural/preview/timeline/RNG/codec equality failed")
                    current()
                except BaseException as error: failure = error
                receipt = observer.finish(native_state_after=media.operation_state(operation, self.owner_session),
                    output_contract_sha256=pending["baseline"].contract if failure is None else None,
                    status="cancelled" if failure and flags.is_set() else "failed" if failure else "completed",
                    error=type(failure).__name__ if failure else None)
            receipt["exactCompleteMediaNeuralTimelineRngCodecEquality"] = failure is None
            receipt["actualOutputContractSha256"] = None if trial is None else trial.contract
            receipt["receiptSha256"] = digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
            self.ops.append_receipt(pending["job"], {"phase": "warmup" if sample < 0 else "measured",
                "operationId": self._operation, "observation": receipt})
            if failure is not None: raise failure
            if sample >= 0:
                pending["issued"], pending["count"] = digest([pending["issued"], receipt["receiptSha256"]]), pending["count"]+1
            pending["candidate"] += 1
            if pending["candidate"] == len(review["candidates"]):
                pending["candidate"], pending["sample"] = 0, sample+1
            if pending["sample"] >= self._samples:
                decision = select_complete_operation(self, self.ops.receipts(pending["job"]), selector=select_whole_path_candidate, identity=operation["identity"],
                    current_candidate=self.primitive(review["currentCandidate"]), minimum_samples=self._samples,
                    verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id)
                if self.ops.receipt_head(pending["job"]) != (pending["issued"], pending["count"]):
                    raise RuntimeError("complete media qualification receipt head changed")
                current(); selected = review["currentCandidate"]
                if decision["decision"] == "propose-at-quiescence":
                    proposed = next(value for value in review["candidates"] if value["planSha256"] == decision["proposedCandidate"]["planSha256"])
                    if media.auto_allowed(proposed): selected = proposed
                self._store_plan(operation["stratumKey"], operation, selected, None,
                    (pending["issued"], pending["count"]), descriptor=pending["baseline"].descriptor)
                self.ops.append_receipt(pending["job"], {"phase": "selection", "operationId": self._operation,
                    "comparison": decision, "firstRealMediaWasNotReplayedOrWithheld": True,
                    "futureRouteReuseIsLayoutInference": True})
                self.release_private_media()
                with self._lock: self._phase, self._reason = "active", "complete-private-media-pool-qualified-real-result-was-already-delivered"
                return False
            return True
        except BaseException as error:
            # Unsupported/nondeterministic codecs and pressure keep normal
            # generation working. Never forge a qualified home/sample receipt.
            with self._lock:
                self._phase = "cancelled" if self._cancel.is_set() else "failed-fallback"
                self._reason = "media-private-pool-retired:" + type(error).__name__
            try: self.ops.append_receipt(pending["job"], {"phase": "terminal-fallback", "operationId": self._operation,
                "failureType": type(error).__name__, "realMediaAlreadyDelivered": True, "noUserWorkReplayed": True})
            except Exception: pass
            self.release_private_media()
            if getattr(error, "native_owner_restore_required", False): raise
            return False
        finally:
            with self._lock: self._active = self._committing = False; self._progress = None
            if self._cancel.is_set(): self._release_all()
