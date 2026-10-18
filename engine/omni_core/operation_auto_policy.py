"""Explicit session-owned automatic pure-operation placement, not a model.

Qualified route reuse is an inference for a fresh input of the tested operator
and geometry, never reuse of old output bytes or proof of cached token identity.
Import is stdlib-only; opening an app does not benchmark anything.
"""
from __future__ import annotations

import base64
import copy
import functools
import json
import hashlib
import hmac
import re
import threading
import uuid
import weakref
from collections import OrderedDict

from .whole_path_execution import WholePathRecorder, digest, select_whole_path_candidate
from .operation_auto_registry import PERCEPTION_KINDS, CACHED_KINDS, MEDIA_KINDS, TRAINING_KINDS, PREFILL_KINDS, FORWARD_KINDS, PERCEPTION_STAGES, PRIVATE_OPERATION, coverage
from .cached_whole_operation_auto import CachedWholeOperationPolicy
from .media_whole_operation_auto import WholeMediaOperationPolicy
from .prefill_whole_operation_auto import FullPrefixOperationPolicy
from .forward_whole_operation_auto import FullForwardOperationPolicy
from .whole_path_scheduler import select_complete_operation, qualification_allowed, owned_operation_choice

WORK_KINDS = [*PERCEPTION_KINDS, *CACHED_KINDS, *MEDIA_KINDS, *TRAINING_KINDS, *PREFILL_KINDS, *FORWARD_KINDS]
STAGES = PERCEPTION_STAGES
UNSUPPORTED = ["general-retained-KV-whole-turn", "opaque-or-unfenced-connected-text-recall-recurrent", "private-migration-of-mutable-memory-and-STDP-owners",
    "graph-working-memory-migration-and-VRAM-context", "opaque-unpinned-or-nondeterministic-media-codec", "unfenced-cross-rank-or-complete-training-checkpoint-pipeline",
    "concurrent-multi-device-schedules", "direct-CUDA-XPU-NPU", "whole-brain-end-to-end-latency", "live-cache-subtree-migration"]
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,255}\Z")
_HUBS = weakref.WeakKeyDictionary()
_BYPASS = PRIVATE_OPERATION


class OperationAutoRestoreRequired(RuntimeError):
    """A failed exact restore cannot safely fall through into the live body."""
    native_owner_restore_required = True


def bind_operation_auto(hub, controller): _HUBS[hub] = weakref.ref(controller)


def measured_perception(method):
    @functools.wraps(method)
    def wrapped(hub, modality, tensor):
        reference = _HUBS.get(hub)
        controller = None if reference is None else reference()
        normal = lambda: method(hub, modality, tensor)
        return normal() if controller is None or not qualification_allowed() else controller.consume(hub, modality, tensor, normal)
    return wrapped


class _Flags:
    def __init__(self, event, parent): self.event, self.parent = event, parent
    def is_set(self): return self.event.is_set() or bool(self.parent is not None and self.parent())


class OperationAutoPolicy(CachedWholeOperationPolicy, WholeMediaOperationPolicy, FullPrefixOperationPolicy, FullForwardOperationPolicy):
    def __init__(self, owner_id, ops, *, owner_session=None, recorder=None):
        self.owner_id, self.ops = owner_id, ops
        self.owner_session = owner_session or uuid.uuid4().hex
        self.recorder = recorder or WholePathRecorder()
        self._lock, self._cancel = threading.RLock(), threading.Event()
        self._phase, self._reason = "idle", "not-armed"
        self._operation = self._queued_stop = self._review = self._progress = self._kind = None
        self._plans, self._last, self._revision = OrderedDict(), None, 0
        self._authority_revision, self._preparing = 0, None
        self._samples, self._closed, self._active, self._needs_release, self._committing = 3, False, False, False, False
        self._supported = []
        self._cursor_secret = uuid.uuid4().bytes + uuid.uuid4().bytes

    def status(self):
        with self._lock:
            return {"format": "omni-native-operation-auto-status", "version": 2, "ownerId": self.owner_id,
                "ownerSessionId": self.owner_session, "phase": self._phase, "operationId": self._operation,
                "reason": self._reason, "progress": copy.deepcopy(self._progress), "cancellationRequested": self._cancel.is_set(),
                "workKind": self._kind, "qualifiedPlanCount": 0 if self._cancel.is_set() else len(self._plans),
                "lastPlan": None if self._cancel.is_set() else copy.deepcopy(self._last), "supportedWorkKinds": list(self._supported),
                "unsupportedRoutes": list(UNSUPPORTED), "globalPlacementChanged": False, "calibrationOnAppOpen": False,
                "wholeBrainOrTrainingMeasured": False, "qualificationIsExact": True, "reusableRouteIsInferenceForUnseenInputs": True}

    def stop(self, *, operation_id=None):
        if operation_id is not None and (not isinstance(operation_id, str) or not _ID.fullmatch(operation_id)):
            raise ValueError("invalid exact operation Auto id")
        with self._lock:
            if operation_id is not None and operation_id != self._operation: self._queued_stop = operation_id
            else:
                self._authority_revision += 1; self._preparing = None; self._cancel.set(); self._needs_release = True
                self.retire_private_media("explicit-policy-stop")
                if not self._active: self._phase, self._reason = "cancelled", "explicit-policy-stop-original-work-unchanged"
            return self.status()

    def release_stopped(self, *, operation_id=None):
        with self._lock:
            if operation_id is not None and operation_id != self._operation: raise ValueError("stopped cleanup belongs to another Auto policy")
            if self._active or not self._cancel.is_set(): raise RuntimeError("Auto cleanup awaits its stopped serial boundary")
        self._release_all()
        return self.status()

    def _release_all(self):
        self.release_private_media()
        with self._lock:
            plans = list(self._plans.values()); self._plans.clear(); self._last = None; self._revision += 1
        for plan in plans: plan["lease"].release()
        self.ops.retire_observations()
        self._needs_release = False

    def prepare(self, *, topology=None, acknowledge_cuda=False):
        with self._lock:
            if self._closed or self._active: raise RuntimeError("operation Auto review requires its idle loaded owner")
            if self._phase in {"awaiting-operation", "active", "failed-fallback"} and not self._cancel.is_set():
                raise RuntimeError("Stop the current operation Auto policy before reviewing a replacement")
            self._authority_revision += 1; ticket = self._preparing = self._authority_revision
            self._review = self._operation = None; self._cancel.clear()
            self._phase, self._reason = "idle", "reviewing-current-native-operation-hardware"
        def current():
            with self._lock:
                if self._closed or self._preparing != ticket or self._authority_revision != ticket or self._cancel.is_set():
                    raise InterruptedError("operation Auto review was stopped or retired")
        try:
            self._release_all(); current(); self.ops.quiescent(); current()
            supported, reason, connected = self.ops.supported(); current()
            if not supported: raise RuntimeError(reason)
            supported_kinds = self.ops.work_kinds(); current()
            candidate, candidates = self.ops.candidates(acknowledge_cuda); current()
            binding = self.ops.hardware_binding(self.owner_session, topology=topology); current()
            review = {"format": "omni-native-operation-auto-review", "version": 2, "ownerId": self.owner_id,
                "binding": binding, "currentCandidate": candidate, "candidates": candidates, "minimumSamples": 3,
                "stageCoverageByWorkKind": coverage(supported_kinds), "supportedWorkKinds": supported_kinds, "connectedPerceptionSupported": connected}
            with self._lock:
                current(); self._review = review; self._supported = list(supported_kinds)
                self._phase, self._reason = "prepared", "reviewed-no-model-operation-performed"
                return {**copy.deepcopy(review), "status": self.status()}
        except BaseException as error:
            with self._lock:
                if self._preparing == ticket and not self._cancel.is_set():
                    self._phase, self._reason = "unsupported", "review-refused:" + type(error).__name__
            raise
        finally:
            with self._lock:
                if self._preparing == ticket: self._preparing = None

    def arm(self, *, operation_id, reviewed, samples=3):
        if not isinstance(operation_id, str) or not _ID.fullmatch(operation_id) or type(samples) is not int or samples < 3:
            raise ValueError("operation Auto needs its main id and at least three admitted samples")
        self.ops.quiescent()
        binding = self.ops.hardware_binding(self.owner_session)
        with self._lock:
            if self._phase != "prepared" or self._review is None or reviewed != self._review["binding"] or reviewed != binding:
                raise ValueError("operation Auto hardware/owner review is stale")
            self._operation, self._samples = operation_id, samples
            if self._queued_stop == operation_id:
                self._queued_stop = None; self._cancel.set(); self._needs_release = True
                self._phase, self._reason = "cancelled", "exact-queued-policy-arm-stopped"
            else: self._phase, self._reason = "awaiting-operation", "explicitly-armed-awaiting-real-native-operation"
        return self.status()

    @staticmethod
    def primitive(candidate): return {key: candidate[key] for key in ("planSha256", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged")}

    def plans(self, *, after=None, limit=32):
        if type(limit) is not int or not 1 <= limit <= 128: raise ValueError("Auto plan page limit must be 1..128")
        with self._lock:
            start = 0
            if after is not None:
                if not isinstance(after, str) or len(after) > 512: raise ValueError("invalid Auto plan cursor")
                try:
                    encoded, signature = after.split(".")
                    expected = hmac.new(self._cursor_secret, encoded.encode(), hashlib.sha256).hexdigest()
                    if not hmac.compare_digest(signature, expected): raise ValueError("Auto plan cursor authentication failed")
                    cursor = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
                except Exception as error: raise ValueError("invalid Auto plan cursor") from error
                if (not isinstance(cursor, dict) or set(cursor) != {"owner", "revision", "offset", "filter"} or cursor["owner"] != self.owner_session
                        or cursor["filter"] != "all-perception-plans"
                        or cursor["revision"] != self._revision or type(cursor["offset"]) is not int or cursor["offset"] < 0):
                    raise ValueError("Auto plan cursor belongs to a retired snapshot")
                start = cursor["offset"]
            total = 0 if self._cancel.is_set() else len(self._plans)
            if start > total: raise ValueError("Auto plan cursor is beyond the current snapshot")
            import itertools
            values = [] if self._cancel.is_set() else [copy.deepcopy(value["public"]) for value in itertools.islice(self._plans.values(), start, start + limit)]
            result = {"ownerId": self.owner_id, "ownerSessionId": self.owner_session, "revision": self._revision, "plans": values, "total": total}
            if start + len(values) < total:
                encoded = base64.urlsafe_b64encode(json.dumps({"owner": self.owner_session, "revision": self._revision,
                    "filter": "all-perception-plans", "offset": start + len(values)}, separators=(",", ":")).encode()).decode().rstrip("=")
                result["nextAfter"] = encoded + "." + hmac.new(self._cursor_secret, encoded.encode(), hashlib.sha256).hexdigest()
            return result

    def _store_plan(self, key, operation, candidate, reference, issued, *, descriptor=None):
        public = {"id": digest([self.owner_session, key, issued, uuid.uuid4().hex]), "workKind": operation["identity"]["workKind"], "geometrySha256": operation["identity"]["geometrySha256"],
            "qualifiedNativeStateSha256": operation["identity"]["nativeState"]["bindingSha256"], "candidate": copy.deepcopy(candidate),
            "qualifiedSamples": self._samples, "qualificationWorkloads": 1, "numericGate": "exact-embedding-equality",
            "reuseBasis": "same-validated-operator-geometry-current-owner-policy-inference", "unseenInputOrFrontierWasPreviouslyMeasured": False,
            "wholeBrainOrTrainingCovered": False}
        if "planContract" in operation: public.update(copy.deepcopy(operation["planContract"]))
        if descriptor is None: descriptor = self.ops.output_descriptor(reference)
        payload = {"public": public, "descriptor": descriptor, "nativeState": operation["identity"]["nativeState"], "issued": issued}
        size = len(json.dumps(payload, sort_keys=True, separators=(",", ":"))) * 4 + 2048
        while True:
            try: lease = self.ops.reserve_plan(size); break
            except Exception:
                with self._lock:
                    removed = self._plans.popitem(last=False)[1] if self._plans else None
                    if removed is not None: self._revision += 1
                if removed is None: raise RuntimeError("derived Auto route metadata cannot fit the current RAM envelope")
                removed["lease"].release()
        try: lease.mark_allocated(size)
        except BaseException:
            lease.release(); raise
        payload["lease"] = lease
        with self._lock:
            old = self._plans.pop(key, None); self._plans[key] = payload; self._last = public; self._revision += 1
        if old is not None: old["lease"].release()
        return payload

    def consume(self, hub, modality, tensor, normal):
        if not qualification_allowed(): return normal()
        with self._lock:
            if self._closed or self._cancel.is_set() or self._phase not in {"awaiting-operation", "active"} or self._active:
                eligible, review = False, None
            else: eligible, review = True, copy.deepcopy(self._review)
        if not eligible: return normal()
        # An unavailable safe boundary is not a reason to clear live context or
        # cancel the original chat/perception operation.
        if not self.ops.eligible(hub, modality, tensor): return normal()
        operation = reference = selected = None
        kept = {}; job = "native-operation-auto-" + uuid.uuid4().hex
        flags = _Flags(self._cancel, self.ops.parent_cancelled)
        issued, count, original_started = "0" * 64, 0, False
        try:
            if self.ops.hardware_binding(self.owner_session) != review["binding"]: raise RuntimeError("Auto hardware/registry/owner changed")
            with self._lock: self._active = True; self._phase = "qualifying"
            operation = self.ops.capture_operation(hub, modality, tensor, self.owner_session, review)
            identity, key = operation["identity"], operation["stratumKey"]
            with self._lock: self._kind = identity["workKind"]
            def current():
                if flags.is_set(): raise InterruptedError("private operation Auto stopped; original work preserved")
                if self.ops.hardware_binding(self.owner_session) != review["binding"] or not self.ops.unchanged(operation, self.owner_session):
                    raise RuntimeError("actual Auto operation owner/input/geometry changed")
            current()
            with self._lock: plan = self._plans.get(key)
            if (plan is not None and plan["nativeState"] == identity["nativeState"] and
                    (plan["public"]["candidate"] == review["currentCandidate"] or self.ops.auto_allowed(plan["public"]["candidate"]))):
                if plan["public"]["candidate"] == review["currentCandidate"]:
                    current()
                    with self._lock: self._last = plan["public"]; self._phase, self._reason = "active", "tested-ordinary-route-retained-no-repeated-qualification"
                    original_started = True
                    return normal()
                with self.ops.pure_scope(flags): selected = self.ops.execute(plan["public"]["candidate"], operation, None, flags)
                current(); self.ops.check_output(selected, plan["descriptor"])
                self.ops.append_receipt(job, {"phase": "reused-route", "operationId": self._operation, "workKind": identity["workKind"],
                    "planId": plan["public"]["id"], "workloadSha256": identity["workloadSha256"], "reuseIsInferenceForUnseenInput": True,
                    "previousOutputWasReused": False, "wholeBrainMeasured": False})
                current()
                with self._lock:
                    if self._closed or self._cancel.is_set(): raise InterruptedError("perception result authority was stopped before delivery")
                    self._last = plan["public"]; self._phase, self._reason = "active", "qualified-route-current-operator-inference-not-previous-output"
                    with owned_operation_choice(self, identity, plan["public"]["candidate"]):
                        return selected.take()
            if plan is not None:
                with self._lock: removed = self._plans.pop(key, None); self._revision += 1
                if removed is not None: removed["lease"].release()
            with self.ops.calibration_reservation(self._samples, len(review["candidates"])), self.ops.pure_scope(flags):
                for sample in range(-1, self._samples):
                    for index, candidate in enumerate(review["candidates"]):
                        current()
                        with self._lock: self._progress = {"candidate": index + 1, "candidates": len(review["candidates"]),
                            "sample": max(0, sample + 1), "samplesPerCandidate": self._samples, "stage": "warming" if sample < 0 else "measuring"}
                        observed = identity if sample >= 0 else {**identity, "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                        trial, failure, exact = None, None, False
                        with self.ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                            observer = self.recorder.begin(observed, self.primitive(candidate), synchronize=self.ops.synchronize,
                                counters=self.ops.counters, resources=resources)
                            try:
                                trial = self.ops.execute(candidate, operation, observer, flags)
                                if reference is None:
                                    if candidate != review["currentCandidate"]: raise RuntimeError("Auto baseline must execute first")
                                    reference = trial
                                exact = self.ops.exact_output(reference, trial); current()
                            except BaseException as error: failure = error
                            receipt = observer.finish(native_state_after=self.ops.operation_state(operation, self.owner_session),
                                output_contract_sha256=reference.contract if exact and not failure else None,
                                status="cancelled" if failure and flags.is_set() else "failed" if failure else "completed",
                                error=type(failure).__name__ if failure else None)
                        receipt["exactEmbeddingEquality"] = exact
                        receipt["receiptSha256"] = digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
                        self.ops.append_receipt(job, {"phase": "warmup" if sample < 0 else "measured", "operationId": self._operation, "observation": receipt})
                        if sample >= 0: issued, count = digest([issued, receipt["receiptSha256"]]), count + 1
                        if trial is not None and trial is not reference:
                            if exact and not failure and sample >= 0:
                                old = kept.get(candidate["planSha256"])
                                if old is not None: old.close()
                                kept[candidate["planSha256"]] = trial
                            else: trial.close()
                        if failure is not None: raise failure
                with self._lock: self._progress["stage"] = "selecting"
                decision = select_complete_operation(self, self.ops.receipts(job), selector=select_whole_path_candidate, identity=identity,
                    current_candidate=self.primitive(review["currentCandidate"]), minimum_samples=self._samples,
                    verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id)
                if self.ops.receipt_head(job) != (issued, count): raise RuntimeError("Auto qualification evidence custody differs")
                current()
                if decision["decision"] == "propose-at-quiescence":
                    candidate = next(value for value in review["candidates"] if value["planSha256"] == decision["proposedCandidate"]["planSha256"])
                    if self.ops.auto_allowed(candidate):
                        selected = kept.pop(candidate["planSha256"])
                        self._store_plan(key, operation, candidate, reference, (issued, count))
                self.ops.append_receipt(job, {"phase": "selection", "operationId": self._operation, "comparison": decision,
                    "exactEmbeddingEqualityRequired": True, "qualificationWorkloads": 1, "futureRouteReuseIsInference": True})
            current()
            if selected is None:
                selected, reference = reference, None
                # A checked ordinary route is also a useful conservative
                # decision. Do not re-run the same failed/inconclusive pool on
                # every subsequent frame of unchanged operator state/geometry.
                self._store_plan(key, operation, review["currentCandidate"], selected, (issued, count))
                candidate = review["currentCandidate"]
            with self._lock:
                if self._closed or self._cancel.is_set(): raise InterruptedError("perception result authority was stopped before delivery")
                self._phase, self._reason = "active", "actual-perception-qualified-or-original-baseline-retained"
                with owned_operation_choice(self, identity, candidate):
                    return selected.take()
        except Exception as error:
            if original_started: raise  # Never replay the actual public body.
            with self._lock:
                self._phase = "cancelled" if flags.is_set() else "failed-fallback"
                self._reason = type(error).__name__ + ":" + str(error)
            try: self.ops.append_receipt(job, {"phase": "terminal-fallback", "operationId": self._operation,
                "failureType": type(error).__name__, "originalPublicBodyReplayed": False})
            except Exception: pass
            if isinstance(error, OperationAutoRestoreRequired): raise
        finally:
            if selected is not None: selected.close()
            if reference is not None: reference.close()
            for value in kept.values(): value.close()
            if operation is not None: self.ops.release_operation(operation)
            with self._lock: self._active = False; self._progress = None
            if self._cancel.is_set():
                with self._lock: self._phase, self._reason = "cancelled", "policy-stopped-original-work-not-cancelled"
                self._release_all()
            elif self._phase == "failed-fallback": self._release_all()
        return normal()  # Never run under a private scope or metadata lock.

    def close(self):
        with self._lock:
            self._closed = True; self._authority_revision += 1; self._preparing = None; self._cancel.set()
            active = self._active
        if not active: self._release_all()
