"""Exact private qualification of one actual uncached full forward.

Import and binding do no model work. The real caller supplies its complete
input and ordinary fallback; no output from another input is reusable here.
"""
from __future__ import annotations

import copy
import functools
import uuid
import weakref

from .operation_auto_registry import FORWARD_KINDS, FORWARD_STAGES, PRIVATE_OPERATION
from .whole_path_execution import digest, select_whole_path_candidate
from .whole_path_scheduler import select_complete_operation, qualification_allowed, owned_operation_choice

FORWARD_NUMERIC_GATE = "exact-full-forward-output-workspace-action-activity-equality"
FORWARD_REUSE_BASIS = "same-prepared-full-forward-current-owner-policy-no-cross-input-inference"
_DECODERS = weakref.WeakKeyDictionary()


def bind_full_forward_operation(decoder, controller):
    _DECODERS[decoder] = weakref.ref(controller)


def measured_full_forward(method):
    @functools.wraps(method)
    def wrapped(decoder, input_ids, memory_bias=None, labels=None,
                use_global_workspace=None, attention_mask=None):
        reference = _DECODERS.get(decoder)
        controller = None if reference is None else reference()
        def normal():
            return method(decoder, input_ids, memory_bias, labels,
                          use_global_workspace, attention_mask)
        if controller is None or not qualification_allowed() or labels is not None:
            return normal()
        return controller.consume_full_forward(window=input_ids, cue=memory_bias,
            use_global_workspace=use_global_workspace, attention_mask=attention_mask,
            fallback=normal)
    return wrapped


class _Flags:
    def __init__(self, event, parent): self.event, self.parent = event, parent
    def is_set(self): return self.event.is_set() or bool(self.parent())


class FullForwardOperationPolicy:
    def consume_full_forward(self, *, window, cue, use_global_workspace,
                             attention_mask, fallback):
        if not qualification_allowed(): return fallback()
        with self._lock:
            eligible = (not self._closed and not self._cancel.is_set() and not self._active
                and self._phase in {"awaiting-operation", "active"}
                and getattr(self.ops, "forward_operations", None) is not None)
            review = copy.deepcopy(self._review) if eligible else None
        if not eligible: return fallback()
        ops = self.ops.forward_operations
        # A bound native callback cannot enlarge an older public review.
        if (not set(FORWARD_KINDS).intersection(review["supportedWorkKinds"])
                or not ops.valid_forward(window, cue, use_global_workspace, attention_mask)):
            return fallback()
        flags = _Flags(self._cancel, self.ops.parent_cancelled)
        operation = reference = chosen = plan = trial = None
        kept, issued, count, published = {}, "0" * 64, 0, False
        job = "native-full-forward-auto-" + uuid.uuid4().hex
        try:
            if self.ops.hardware_binding(self.owner_session) != review["binding"]:
                raise RuntimeError("full-forward Auto hardware changed")
            with self._lock: self._active, self._phase = True, "qualifying"
            operation = ops.capture_forward(window, cue, use_global_workspace,
                attention_mask, self.owner_session, review, stopped=flags)
            identity = operation["identity"]
            with self._lock: self._kind = identity["workKind"]
            if identity["workKind"] not in review["supportedWorkKinds"]:
                raise RuntimeError("uncached full forward was not in the exact reviewed registry")
            def current():
                ops.require_original_frontiers(operation, self.owner_session)
                if flags.is_set(): raise InterruptedError("private full-forward Auto stopped")
                if (self.ops.hardware_binding(self.owner_session) != review["binding"]
                        or not ops.forward_unchanged(operation, self.owner_session)):
                    raise RuntimeError("full-forward input, owner or resource policy changed")
            current()
            prepare_candidates = getattr(ops, "mixed_candidates", None)
            candidates = prepare_candidates(operation, review, current=current) if callable(prepare_candidates) else review["candidates"]
            mixed_pool = bool(getattr(candidates, "has_mixed", False))
            with ops.calibration_reservation(self._samples, len(candidates)), ops.pure_scope(flags):
                for sample in range(-1, self._samples):
                    for index, candidate in enumerate(candidates):
                        current()
                        with self._lock:
                            self._progress = {"candidate": index + 1, "candidates": len(candidates),
                                "sample": max(0, sample + 1), "samplesPerCandidate": self._samples,
                                "stage": "warming" if sample < 0 else "measuring"}
                        observed = identity if sample >= 0 else {**identity,
                            "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                        trial, failure, exact = None, None, False
                        with ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                            observer = self.recorder.begin(observed, self.primitive(candidate),
                                synchronize=ops.synchronize, counters=ops.counters, resources=resources)
                            try:
                                trial = ops.execute(candidate, operation, observer, flags)
                                if reference is None:
                                    if candidate != review["currentCandidate"]:
                                        raise RuntimeError("full-forward current-home reference must run first")
                                    reference = trial
                                exact = ops.exact_output(reference, trial)
                                if not exact:
                                    raise RuntimeError("full-forward outputs, workspace, actions or activity differ")
                                current()
                            except BaseException as error: failure = error
                            try:
                                receipt = observer.finish(native_state_after=ops.operation_state(operation, self.owner_session),
                                    output_contract_sha256=reference.contract if exact and failure is None else None,
                                    status="cancelled" if failure and flags.is_set() else "failed" if failure else "completed",
                                    error=type(failure).__name__ if failure else None)
                            except BaseException:
                                if trial is not None and trial is not reference: trial.close()
                                raise
                        receipt["exactFullForwardOutputWorkspaceActionActivityEquality"] = exact
                        receipt["actualOutputContractSha256"] = None if trial is None else trial.contract
                        receipt["receiptSha256"] = digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
                        self.ops.append_receipt(job, {"phase": "warmup" if sample < 0 else "measured",
                            "operationId": self._operation, "observation": receipt})
                        if sample >= 0: issued, count = digest([issued, receipt["receiptSha256"]]), count + 1
                        if trial is not None and trial is not reference:
                            if exact and failure is None and sample >= 0 and not mixed_pool:
                                old = kept.pop(candidate["planSha256"], None)
                                if old is not None: old.close()
                                kept[candidate["planSha256"]] = trial
                            else: trial.close()
                        if failure is not None: raise failure
                current()
                with self._lock: self._progress["stage"] = "selecting"
                decision = select_complete_operation(self, self.ops.receipts(job), selector=select_whole_path_candidate, identity=identity,
                    current_candidate=self.primitive(review["currentCandidate"]), minimum_samples=self._samples,
                    verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id)
                if self.ops.receipt_head(job) != (issued, count):
                    raise RuntimeError("full-forward qualification receipt custody changed")
                current()
                candidate = review["currentCandidate"]
                if decision["decision"] == "propose-at-quiescence":
                    proposed = next(row for row in candidates
                        if row["planSha256"] == decision["proposedCandidate"]["planSha256"])
                    if ops.auto_allowed(proposed): candidate = proposed
                if mixed_pool:
                    chosen = ops.execute(candidate, operation, None, flags)
                    if not ops.exact_output(reference, chosen): raise RuntimeError("selected mixed forward result lost its exact complete contract")
                    current()
                else: chosen = kept.pop(candidate["planSha256"])
                # A plan is an inspection receipt for this prepared input only.
                # There is deliberately no branch that reuses it on a new call.
                plan = self._store_plan(operation["stratumKey"], operation, candidate,
                    reference, (issued, count), descriptor=ops.output_descriptor(reference))
                self.ops.append_receipt(job, {"phase": "selection", "operationId": self._operation,
                    "comparison": decision, "exactFullForwardEqualityRequired": True, "crossInputReuse": False})
            current()
            with self._lock:
                if self._closed or self._cancel.is_set():
                    raise InterruptedError("full-forward authority retired before real delivery")
                # Publication and Stop have one lock boundary. Returning the
                # already computed result never invokes the ordinary body again.
                self._committing = True
                published = True
                with owned_operation_choice(self, identity, plan["public"]["candidate"]):
                    ops.publish_activity(chosen)
                    result = chosen.take()
                self._last, self._phase, self._reason = (plan["public"], "active",
                    "exact-current-uncached-full-forward-no-cross-input-inference")
                return result
        except BaseException as error:
            with self._lock:
                self._phase = "cancelled" if flags.is_set() else "failed-fallback"
                self._reason = type(error).__name__ + ":" + str(error)
            try:
                self.ops.append_receipt(job, {"phase": "terminal-fallback", "operationId": self._operation,
                    "failureType": type(error).__name__, "originalForwardReplayed": False})
            except Exception: pass
            if published or getattr(error, "native_owner_restore_required", False): raise
        finally:
            cleanup_error = None
            for value in (chosen, reference, trial, *kept.values()):
                if value is not None:
                    try: value.close()
                    except BaseException as error:
                        if cleanup_error is None: cleanup_error = error
            try:
                if operation is not None: ops.release_forward(operation)
            except BaseException as error:
                if cleanup_error is None: cleanup_error = error
            finally:
                with self._lock: self._active = self._committing = False; self._progress = None
            if self._cancel.is_set():
                with self._lock: self._phase, self._reason = "cancelled", "policy-stopped-original-forward-retained"
                self._release_all()
            elif self._phase == "failed-fallback": self._release_all()
            if cleanup_error is not None:
                with self._lock: self._phase, self._reason = "failed-fallback", "private-forward-release-failed-no-real-replay"
                raise cleanup_error
        return fallback()
