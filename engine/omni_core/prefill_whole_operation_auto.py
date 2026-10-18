"""One actual full-prefix/sampling boundary under the existing Auto owner.

No model, tensor or I/O work on import. A new private cache is owned by each
trial; neither an old cache nor any real callback is replayed or reset. Plans
record this exact prefix only, with no cross-prefix inference authority.
"""
from __future__ import annotations

import copy
import uuid
import weakref

from .operation_auto_registry import PRIVATE_OPERATION, PREFILL_KINDS, PREFILL_STAGES
from .whole_path_execution import digest, select_whole_path_candidate
from .whole_path_scheduler import select_complete_operation, qualification_allowed, owned_operation_choice

PREFILL_NUMERIC_GATE = "exact-prefix-cache-connected-activity-neural-sampling-equality"
PREFILL_REUSE_BASIS = "same-prepared-prefix-current-owner-policy-no-cross-prefix-inference"
_DECODERS = weakref.WeakKeyDictionary()


def bind_prefill_whole_operation(decoder, controller): _DECODERS[decoder] = weakref.ref(controller)


def full_prefix_sampling_operation(decoder, *, allowed, window, cue, cache, generator, sampling,
                                   flags_cancelled, before_sample, fallback, fallback_after_boundary):
    reference = _DECODERS.get(decoder)
    controller = None if reference is None else reference()
    if not allowed or controller is None or not qualification_allowed(): return fallback()
    return controller.consume_prefill(window=window, cue=cue, cache=cache, generator=generator,
        sampling=sampling, flags_cancelled=flags_cancelled, before_sample=before_sample,
        fallback=fallback, fallback_after_boundary=fallback_after_boundary)


class _Flags:
    def __init__(self, event, parent): self.event, self.parent = event, parent
    def is_set(self): return self.event.is_set() or bool(self.parent is not None and self.parent())


class FullPrefixOperationPolicy:
    def consume_prefill(self, *, window, cue, cache, generator, sampling, flags_cancelled,
                        before_sample, fallback, fallback_after_boundary):
        if not qualification_allowed(): return fallback()
        with self._lock:
            eligible = (not self._closed and not self._cancel.is_set() and not self._active
                and self._phase in {"awaiting-operation", "active"}
                and getattr(self.ops, "prefill_operations", None) is not None)
            review = copy.deepcopy(self._review) if eligible else None
        if not eligible: return fallback()
        ops = self.ops.prefill_operations
        if not ops.valid_prefix(window, cue, cache): return fallback()
        # The strict shared/main contract advertises these kinds explicitly.
        # Binding this callback cannot extend a previously reviewed registry.
        if not set(PREFILL_KINDS).intersection(review["supportedWorkKinds"]): return fallback()
        flags = _Flags(self._cancel, flags_cancelled)
        operation = reference = chosen = plan = None
        retained, issued, count, boundary_started, after_boundary = {}, "0" * 64, 0, False, False
        job = "native-full-prefix-auto-" + uuid.uuid4().hex
        try:
            if self.ops.hardware_binding(self.owner_session) != review["binding"]:
                raise RuntimeError("full-prefix Auto hardware changed")
            with self._lock: self._active, self._phase = True, "qualifying"
            operation = ops.capture_prefix(window, cue, cache, generator, sampling, self.owner_session, review, stopped=flags)
            identity = operation["identity"]
            with self._lock: self._kind = identity["workKind"]
            if identity["workKind"] not in review["supportedWorkKinds"]:
                raise RuntimeError("full prefix was not part of the actual reviewed native registry")
            def current():
                if flags.is_set(): raise InterruptedError("private full-prefix Auto stopped; original chat retained")
                if (self.ops.hardware_binding(self.owner_session) != review["binding"]
                        or not ops.prefix_unchanged(operation, self.owner_session)):
                    raise RuntimeError("complete prefix owner/cache/RNG/policy changed")
            current()
            prepare_candidates = getattr(ops, "mixed_candidates", None)
            candidates = prepare_candidates(operation, review, current=current) if callable(prepare_candidates) else review["candidates"]
            mixed_pool = bool(getattr(candidates, "has_mixed", False))
            with ops.calibration_reservation(self._samples, len(candidates)), ops.pure_scope(flags):
                for sample in range(-1, self._samples):
                    for index, candidate in enumerate(candidates):
                        current()
                        with self._lock: self._progress = {"candidate": index + 1, "candidates": len(candidates),
                            "sample": max(0, sample + 1), "samplesPerCandidate": self._samples,
                            "stage": "warming" if sample < 0 else "measuring"}
                        observed = identity if sample >= 0 else {**identity,
                            "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                        trial, failure, exact = None, None, False
                        with ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                            observer = self.recorder.begin(observed, self.primitive(candidate), synchronize=ops.synchronize,
                                counters=ops.counters, resources=resources)
                            try:
                                trial = ops.execute(candidate, operation, observer, flags)
                                if reference is None:
                                    if candidate != review["currentCandidate"]:
                                        raise RuntimeError("full-prefix home reference must execute first")
                                    reference = trial
                                exact = ops.exact_output(reference, trial)
                                if not exact: raise RuntimeError("complete prefix cache/activity/sampling is not exactly equal")
                                current()
                            except BaseException as error: failure = error
                            receipt = observer.finish(native_state_after=ops.operation_state(operation, self.owner_session),
                                output_contract_sha256=reference.contract if exact and failure is None else None,
                                status="cancelled" if failure and flags.is_set() else "failed" if failure else "completed",
                                error=type(failure).__name__ if failure else None)
                        receipt["exactCompletePrefixCacheActivitySamplingEquality"] = exact
                        receipt["actualOutputContractSha256"] = None if trial is None else trial.contract
                        receipt["receiptSha256"] = digest({name: value for name, value in receipt.items() if name != "receiptSha256"})
                        self.ops.append_receipt(job, {"phase": "warmup" if sample < 0 else "measured",
                            "operationId": self._operation, "observation": receipt})
                        if sample >= 0: issued, count = digest([issued, receipt["receiptSha256"]]), count + 1
                        if trial is not None and trial is not reference:
                            if exact and failure is None and sample >= 0 and not mixed_pool:
                                old = retained.pop(candidate["planSha256"], None)
                                if old is not None: old.close()
                                retained[candidate["planSha256"]] = trial
                            else: trial.close()
                        if failure is not None: raise failure
                current()
                with self._lock: self._progress["stage"] = "selecting"
                decision = select_complete_operation(self, self.ops.receipts(job), selector=select_whole_path_candidate, identity=identity,
                    current_candidate=self.primitive(review["currentCandidate"]), minimum_samples=self._samples,
                    verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id)
                if self.ops.receipt_head(job) != (issued, count):
                    raise RuntimeError("full-prefix qualification receipt custody changed")
                current()
                candidate = review["currentCandidate"]
                if decision["decision"] == "propose-at-quiescence":
                    proposed = next(row for row in candidates
                        if row["planSha256"] == decision["proposedCandidate"]["planSha256"])
                    if ops.auto_allowed(proposed): candidate = proposed
                if mixed_pool:
                    chosen = ops.execute(candidate, operation, None, flags)
                    if not ops.exact_output(reference, chosen): raise RuntimeError("selected mixed prefix result lost its exact complete contract")
                    current()
                else: chosen = retained.pop(candidate["planSha256"])
                plan = self._store_plan(operation["stratumKey"], operation, candidate, reference, (issued, count),
                    descriptor=ops.output_descriptor(reference))
                self.ops.append_receipt(job, {"phase": "selection", "operationId": self._operation,
                    "comparison": decision, "exactCompletePrefixCacheActivitySamplingRequired": True,
                    "crossPrefixReuse": False})
            current()
            with self._lock:
                if self._closed or self._cancel.is_set(): raise InterruptedError("full-prefix authority retired before real boundary")
                self._committing = True
            boundary_started = True
            with owned_operation_choice(self, identity, plan["public"]["candidate"]) as actual_choice:
                ops.publish_activity(chosen)
                if not before_sample(chosen.cache, chosen.neural_state):
                    if actual_choice is not None: actual_choice["aborted"] = True
                    if cache is not None: cache.close()
                    chosen.take_cache()
                    return None
                sample = ops.commit_sampling(operation, chosen)
                if sample is None and actual_choice is not None: actual_choice["aborted"] = True
            if sample is None:
                ops.restore_published_activity(operation)
                chosen.close(); chosen = None
                ops.rebind_original(operation, self.owner_session)
                with self._lock: self._phase, self._reason = "failed-fallback", "home-RNG-changed-real-boundary-not-replayed"
                after_boundary = True
            else:
                if cache is not None: cache.close()
                chosen.take_cache()
                with self._lock:
                    self._last, self._phase, self._reason = plan["public"], "active", "exact-current-full-prefix-consumed-no-cross-prefix-inference"
                return sample
        except BaseException as error:
            with self._lock:
                self._phase = "cancelled" if flags.is_set() else "failed-fallback"
                self._reason = type(error).__name__ + ":" + str(error)
            try: self.ops.append_receipt(job, {"phase": "terminal-fallback", "operationId": self._operation,
                "failureType": type(error).__name__, "originalChatReplayed": False})
            except Exception: pass
            if boundary_started or getattr(error, "native_owner_restore_required", False): raise
        finally:
            cleanup_error = None
            for value in (chosen, reference, *retained.values()):
                if value is not None:
                    try: value.close()
                    except BaseException as error:
                        if cleanup_error is None: cleanup_error = error
            try:
                if operation is not None:
                    try: ops.rebind_original(operation, self.owner_session)
                    finally: ops.release_prefix(operation)
            except BaseException as error:
                if cleanup_error is None: cleanup_error = error
            finally:
                with self._lock: self._active = self._committing = False; self._progress = None
            if cleanup_error is not None:
                with self._lock: self._phase, self._reason = "failed-fallback", "private-prefix-release-failed-no-real-replay"
            if self._cancel.is_set():
                with self._lock: self._phase, self._reason = "cancelled", "policy-stopped-original-chat-not-cancelled"
                self._release_all()
            elif self._phase == "failed-fallback": self._release_all()
            if cleanup_error is not None: raise cleanup_error
        return fallback_after_boundary() if after_boundary else fallback()
