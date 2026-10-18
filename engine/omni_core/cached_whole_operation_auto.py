"""Unified Auto's retained-step branch, no model/Torch/I/O on import.

Qualification proves one actual frontier. Later fresh frontiers in the actual
allocator/tile range use the route as explicitly labelled inference, never an
old cache/token/output or measured proof for unseen state. World effects stay
outside private trials and occur at the existing real boundary exactly once.
"""
from __future__ import annotations

import copy
import uuid
import weakref

from .operation_auto_registry import PRIVATE_OPERATION
from .whole_path_execution import digest, select_whole_path_candidate
from .whole_path_scheduler import select_complete_operation, qualification_allowed, owned_operation_choice

_DECODERS = weakref.WeakKeyDictionary()


class _OrdinaryFallback(Exception): pass


def bind_cached_whole_operation(decoder, controller): _DECODERS[decoder] = weakref.ref(controller)


def reusable_cached_sampling_operation(decoder, *, allowed, window, cue, cache, generator, sampling,
                                      flags_cancelled, before_sample, fallback, fallback_after_boundary):
    reference = _DECODERS.get(decoder)
    controller = None if reference is None else reference()
    if not allowed or cache is None or controller is None or not qualification_allowed(): return fallback()
    return controller.consume_cached(window=window, cue=cue, cache=cache, generator=generator, sampling=sampling,
        flags_cancelled=flags_cancelled, before_sample=before_sample, fallback=fallback, fallback_after_boundary=fallback_after_boundary)


class _Flags:
    def __init__(self, event, parent): self.event, self.parent = event, parent
    def is_set(self): return self.event.is_set() or bool(self.parent is not None and self.parent())


class CachedWholeOperationPolicy:
    def consume_cached(self, *, window, cue, cache, generator, sampling, flags_cancelled,
                       before_sample, fallback, fallback_after_boundary):
        if not qualification_allowed(): return fallback()
        with self._lock:
            eligible = (not self._closed and not self._cancel.is_set() and self._phase in {"awaiting-operation", "active"}
                        and not self._active and getattr(self.ops, "cached_operations", None) is not None)
            review = copy.deepcopy(self._review) if eligible else None
        if not eligible: return fallback()
        ops = self.ops.cached_operations
        if not ops.valid_frontier(window, cue, cache): return fallback()
        flags = _Flags(self._cancel, flags_cancelled)
        operation = reference = chosen = None
        retained, issued, count, boundary_started, after_boundary = {}, "0" * 64, 0, False, False
        job = "native-cached-whole-auto-" + uuid.uuid4().hex
        try:
            if self.ops.hardware_binding(self.owner_session) != review["binding"]: raise RuntimeError("complete cached Auto hardware changed")
            with self._lock: self._active = True; self._phase = "qualifying"
            operation = ops.capture_frontier(window, cue, cache, generator, sampling, self.owner_session, review, stopped=flags)
            identity, key = operation["identity"], operation["stratumKey"]
            with self._lock: self._kind = identity["workKind"]
            if identity["workKind"] not in review["supportedWorkKinds"]: raise RuntimeError("cached operation was not in the actual reviewed registry")
            def current():
                if flags.is_set(): raise InterruptedError("private complete cached Auto stopped; normal chat retained")
                if self.ops.hardware_binding(self.owner_session) != review["binding"] or not ops.frontier_unchanged(operation, self.owner_session):
                    raise RuntimeError("complete cached source/cache/RNG/learned frontier changed")
            current()
            prepare_candidates = getattr(ops, "mixed_candidates", None)
            candidates = prepare_candidates(operation, review, current=current) if callable(prepare_candidates) else review["candidates"]
            mixed_pool = bool(getattr(candidates, "has_mixed", False))
            with self._lock: plan = self._plans.get(key)
            if plan is not None and plan["nativeState"] == identity["nativeState"] and (
                    plan["public"]["candidate"] == review["currentCandidate"] or ops.auto_allowed(plan["public"]["candidate"])):
                if plan["public"]["candidate"] == review["currentCandidate"]:
                    current()
                    with self._lock: self._last = plan["public"]; self._phase, self._reason = "active", "tested-home-cached-route-retained-no-repeat-pool"
                    raise _OrdinaryFallback()
                with ops.pure_scope(flags): chosen = ops.execute(plan["public"]["candidate"], operation, None, flags)
                current(); ops.check_output(chosen, plan["descriptor"])
                self.ops.append_receipt(job, {"phase": "reused-route", "operationId": self._operation, "workKind": identity["workKind"],
                    "planId": plan["public"]["id"], "workloadSha256": identity["workloadSha256"],
                    "freshFrontierInference": True, "previousCacheTokenOutputReused": False, "contextRange": operation["contextRange"]})
            else:
                if plan is not None:
                    with self._lock: removed = self._plans.pop(key, None); self._revision += 1
                    if removed is not None: removed["lease"].release()
                with ops.calibration_reservation(self._samples, len(candidates)), ops.pure_scope(flags):
                    for sample in range(-1, self._samples):
                        for index, candidate in enumerate(candidates):
                            current()
                            with self._lock: self._progress = {"candidate": index + 1, "candidates": len(candidates),
                                "sample": max(0, sample + 1), "samplesPerCandidate": self._samples,
                                "stage": "warming" if sample < 0 else "measuring"}
                            observed = identity if sample >= 0 else {**identity, "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                            trial, failure, exact = None, None, False
                            with ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                                observer = self.recorder.begin(observed, self.primitive(candidate), synchronize=ops.synchronize,
                                    counters=ops.counters, resources=resources)
                                try:
                                    trial = ops.execute(candidate, operation, observer, flags)
                                    if reference is None:
                                        if candidate != review["currentCandidate"]: raise RuntimeError("complete cached baseline must execute first")
                                        reference = trial
                                    exact = ops.exact_output(reference, trial); current()
                                except BaseException as error: failure = error
                                receipt = observer.finish(native_state_after=ops.operation_state(operation, self.owner_session),
                                    output_contract_sha256=reference.contract if exact and failure is None else None,
                                    status="cancelled" if failure and flags.is_set() else "failed" if failure else "completed",
                                    error=type(failure).__name__ if failure else None)
                            receipt["exactCompleteCacheActivitySamplingEquality"] = exact
                            receipt["actualOutputContractSha256"] = None if trial is None else trial.contract
                            receipt["receiptSha256"] = digest({name: value for name, value in receipt.items() if name != "receiptSha256"})
                            self.ops.append_receipt(job, {"phase": "warmup" if sample < 0 else "measured", "operationId": self._operation, "observation": receipt})
                            if sample >= 0: issued, count = digest([issued, receipt["receiptSha256"]]), count + 1
                            if trial is not None and trial is not reference:
                                if exact and failure is None and sample >= 0 and not mixed_pool:
                                    old = retained.get(candidate["planSha256"])
                                    if old is not None: old.close()
                                    retained[candidate["planSha256"]] = trial
                                else: trial.close()
                            if failure is not None: raise failure
                    with self._lock: self._progress["stage"] = "selecting"
                    decision = select_complete_operation(self, self.ops.receipts(job), selector=select_whole_path_candidate, identity=identity,
                        current_candidate=self.primitive(review["currentCandidate"]), minimum_samples=self._samples,
                        verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id)
                    if self.ops.receipt_head(job) != (issued, count): raise RuntimeError("complete cached qualification receipt head changed")
                    current()
                    if decision["decision"] == "propose-at-quiescence":
                        candidate = next(value for value in candidates if value["planSha256"] == decision["proposedCandidate"]["planSha256"])
                        if ops.auto_allowed(candidate):
                            if mixed_pool:
                                chosen = ops.execute(candidate, operation, None, flags)
                                if not ops.exact_output(reference, chosen): raise RuntimeError("selected mixed cached result lost its exact complete contract")
                                current()
                            else: chosen = retained.pop(candidate["planSha256"])
                            plan = self._store_plan(key, operation, candidate, reference, (issued, count), descriptor=ops.output_descriptor(reference))
                    self.ops.append_receipt(job, {"phase": "selection", "operationId": self._operation, "comparison": decision,
                        "exactCompleteCacheActivitySamplingRequired": True, "futureFrontierReuseIsRangeInference": True,
                        "contextRange": operation["contextRange"]})
                current()
                if chosen is None:
                    self._store_plan(key, operation, review["currentCandidate"], reference, (issued, count), descriptor=ops.output_descriptor(reference))
                    with self._lock: self._phase, self._reason = "active", "no-exact-dominant-complete-cached-route-home-retained"
                    raise _OrdinaryFallback()
            current()
            with self._lock:
                if self._closed or self._cancel.is_set(): raise InterruptedError("complete cached authority stopped before its real boundary")
                self._committing = True
            # Linearized real delivery begins here. A later private Stop applies
            # to future policy work and cannot undo this actual world boundary.
            boundary_started = True
            with owned_operation_choice(self, identity, plan["public"]["candidate"]) as actual_choice:
                ops.publish_activity(chosen)
                if not before_sample(chosen.cache, chosen.neural_state):
                    if actual_choice is not None: actual_choice["aborted"] = True
                    cache.close(); chosen.take_cache()
                    return None
                sample = ops.commit_sampling(operation, chosen)
                if sample is None and actual_choice is not None: actual_choice["aborted"] = True
            if sample is None:
                ops.restore_published_activity(operation)
                chosen.close(); chosen = None
                ops.rebind_original(operation, self.owner_session)
                with self._lock: self._phase, self._reason = "failed-fallback", "home-RNG-changed-real-boundary-not-replayed"
                after_boundary = True
                raise _OrdinaryFallback()
            cache.close(); chosen.take_cache()
            with self._lock:
                self._last = plan["public"]
                self._phase, self._reason = "active", "fresh-complete-cached-frontier-consumed-on-qualified-range-inference"
            return sample
        except _OrdinaryFallback: pass
        except BaseException as error:
            with self._lock:
                self._phase = "cancelled" if flags.is_set() else "failed-fallback"
                self._reason = type(error).__name__ + ":" + str(error)
            try: self.ops.append_receipt(job, {"phase": "terminal-fallback", "operationId": self._operation,
                "failureType": type(error).__name__, "originalChatReplayed": False})
            except Exception: pass
            if boundary_started or getattr(error, "native_owner_restore_required", False): raise
        finally:
            if chosen is not None: chosen.close()
            if reference is not None: reference.close()
            for value in retained.values(): value.close()
            if operation is not None:
                ops.rebind_original(operation, self.owner_session); ops.release_frontier(operation)
            with self._lock: self._active = self._committing = False; self._progress = None
            if self._cancel.is_set():
                with self._lock: self._phase, self._reason = "cancelled", "policy-stopped-original-chat-not-cancelled"
                self._release_all()
            elif self._phase == "failed-fallback": self._release_all()
        return fallback_after_boundary() if after_boundary else fallback()
