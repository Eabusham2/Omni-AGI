"""Opt-in one-shot comparison at an actual retained-KV token frontier.

No Torch/model/I/O on import. Private trials use COW activity, current shared
weights, cloned home RNG and no world callbacks. Never a persistent/global plan.
"""
from __future__ import annotations

import copy
import re
import threading
import uuid
import weakref

from .whole_path_execution import WholePathRecorder, digest, select_whole_path_candidate
from .whole_path_scheduler import qualification_allowed

WORK_KIND = "native.decoder.cached-next-token-sampling"
STAGES = ["cache-cow-snapshot", "cached-workspace", "routed-decoder-blocks", "home-readout",
          "same-home-sampling", "retained-cache-return"]
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,255}\Z")
_CONSUMERS = weakref.WeakKeyDictionary()


def bind_cached_decoder(decoder, consumer):
    _CONSUMERS[decoder] = weakref.ref(consumer)


def cached_sampling_operation(decoder, *, allowed, window, cue, cache, generator, sampling,
                              flags_cancelled, before_sample, fallback, fallback_after_boundary):
    reference = _CONSUMERS.get(decoder)
    consumer = None if reference is None else reference()
    if not allowed or cache is None or consumer is None or not qualification_allowed():
        return fallback()
    return consumer.consume(window=window, cue=cue, cache=cache, generator=generator, sampling=sampling,
        flags_cancelled=flags_cancelled, before_sample=before_sample, fallback=fallback,
        fallback_after_boundary=fallback_after_boundary)


class _CombinedFlags:
    def __init__(self, event, parent): self.event, self.parent = event, parent
    def is_set(self): return self.event.is_set() or bool(self.parent is not None and self.parent())


class CachedDecoderExecution:
    def __init__(self, owner_id, ops, *, owner_session=None, recorder=None):
        self.owner_id, self.ops = owner_id, ops
        self.owner_session = owner_session or uuid.uuid4().hex
        self.recorder = recorder or WholePathRecorder()
        self._lock, self._cancel = threading.RLock(), threading.Event()
        self._phase, self._reason = "idle", "not-armed"
        self._operation = self._queued_stop = None
        self._review, self._progress, self._selected = None, None, None
        self._samples, self._committing = 3, False
        self._consumed = self._matched = self._closed = False
        self._authority_revision, self._prepare_ticket = 0, None

    def status(self):
        with self._lock:
            return {"format": "omni-cached-decoder-execution-status", "version": 1,
                "ownerId": self.owner_id, "ownerSessionId": self.owner_session,
                "phase": self._phase, "workKind": WORK_KIND, "operationId": self._operation,
                "reason": self._reason, "progress": copy.deepcopy(self._progress),
                "cancellationRequested": self._cancel.is_set(), "selectedCandidate": copy.deepcopy(self._selected),
                "actualOperationConsumed": self._consumed, "exactCacheAndSamplingMatched": self._matched,
                "globalPlacementChanged": False, "connectedDecoderSupported": False,
                "wholeBrainOrMediaMeasured": False, "reusableAcrossWorkloads": False}

    def stop(self, *, operation_id=None):
        if operation_id is not None and (not isinstance(operation_id, str) or not _ID.fullmatch(operation_id)):
            raise ValueError("invalid exact cached operationId")
        with self._lock:
            if operation_id is not None and operation_id != self._operation:
                self._queued_stop = operation_id
            elif self._phase == "completed":
                self._reason = "one-shot-already-completed-no-future-authority"
            elif not self._committing:
                self._authority_revision += 1
                self._prepare_ticket = None
                self._cancel.set()
                if self._phase != "calibrating":
                    self._review = None
                    self._phase, self._reason = "cancelled", "explicit-one-shot-cached-stop-chat-unchanged"
            return self.status()

    def prepare(self, *, topology=None, acknowledge_cuda=False):
        with self._lock:
            if self._closed: raise RuntimeError("cached execution owner is closed")
            if self._phase == "calibrating": raise RuntimeError("cached operation is already comparing")
            self._authority_revision += 1
            ticket = self._prepare_ticket = self._authority_revision
            self._review, self._operation = None, None
            self._cancel.clear()
            self._phase, self._reason = "idle", "reviewing-hardware-no-model-operation-performed"
            self._consumed = self._matched = False
            self._selected = None
        def current():
            with self._lock:
                if (self._closed or self._prepare_ticket != ticket or self._authority_revision != ticket
                        or self._cancel.is_set()):
                    raise InterruptedError("cached review preparation was stopped or retired")
        try:
            self.ops.quiescent(); current()
            supported, reason = self.ops.supported(); current()
            if not supported:
                with self._lock:
                    current()
                    self._phase, self._reason = "unsupported", reason
                raise RuntimeError(reason)
            candidate, candidates = self.ops.candidates(acknowledge_cuda); current()
            binding = self.ops.hardware_binding(self.owner_session, topology=topology); current()
            review = {"format": "omni-cached-decoder-execution-review", "version": 1,
                "ownerId": self.owner_id, "binding": binding,
                "workKind": WORK_KIND, "currentCandidate": candidate, "candidates": candidates,
                "minimumSamples": 3, "stageCoverage": list(STAGES)}
            with self._lock:
                current()  # No late callback can resurrect a stopped review.
                self._review = review
                self._phase, self._reason = "prepared", "hardware-reviewed-no-model-operation-performed"
                return {**copy.deepcopy(review), "status": self.status()}
        except BaseException as error:
            with self._lock:
                if (self._prepare_ticket == ticket and self._authority_revision == ticket and not self._closed
                        and not self._cancel.is_set() and self._phase != "unsupported"):
                    self._phase, self._reason = "failed-fallback", "metadata-review-failed:" + type(error).__name__
            raise
        finally:
            with self._lock:
                if self._prepare_ticket == ticket: self._prepare_ticket = None

    def arm(self, *, operation_id, reviewed, samples=3):
        if not isinstance(operation_id, str) or not _ID.fullmatch(operation_id) or type(samples) is not int or samples < 3:
            raise ValueError("cached arm requires its main operationId and at least three admitted samples")
        self.ops.quiescent()
        if self._review is None or reviewed != self._review["binding"] or reviewed != self.ops.hardware_binding(self.owner_session):
            raise ValueError("cached arm hardware/owner review is stale")
        with self._lock:
            if self._phase != "prepared": raise ValueError("cached arm requires one current prepared review")
            self._operation, self._samples = operation_id, samples
            self._cancel.clear()
            if self._queued_stop == operation_id:
                self._queued_stop = None
                self._cancel.set()
                self._phase, self._reason = "cancelled", "exact-queued-cached-arm-was-stopped"
            else:
                self._phase, self._reason = "awaiting-retained-frontier", "one-explicit-real-visible-chat-cache-frontier-no-quiz"
        return self.status()

    @staticmethod
    def primitive(candidate):
        return {key: candidate[key] for key in ("planSha256", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged")}

    def consume(self, *, window, cue, cache, generator, sampling, flags_cancelled,
                before_sample, fallback, fallback_after_boundary):
        if not qualification_allowed(): return fallback()
        with self._lock:
            eligible = self._phase == "awaiting-retained-frontier" and not self._cancel.is_set()
        if not eligible: return fallback()
        if not self.ops.valid_frontier(window, cue, cache): return fallback()
        private_stop = _CombinedFlags(self._cancel, flags_cancelled)
        frontier, reference, chosen = None, None, None
        boundary_started = False
        retained = {}
        operation, issued_head, issued_count = self._operation, "0" * 64, 0
        job = "cached-decoder-execution-" + uuid.uuid4().hex
        decision = None
        try:
            if self.ops.hardware_binding(self.owner_session) != self._review["binding"]:
                raise RuntimeError("cached hardware/owner/policy changed before real frontier")
            with self._lock:
                self._phase, self._reason = "calibrating", "explicit-one-shot-real-retained-cache-comparison"
            frontier = self.ops.capture_frontier(window, cue, cache, generator, sampling, self.owner_session, self._review)
            identity = frontier["identity"]
            def current():
                if private_stop.is_set(): raise InterruptedError("cached comparison stopped; normal chat preserved")
                if (self.ops.hardware_binding(self.owner_session) != self._review["binding"]
                        or not self.ops.frontier_unchanged(frontier, self.owner_session)):
                    raise RuntimeError("original learned/cache/input/RNG frontier changed")
            current()
            with self.ops.calibration_reservation(self._samples, len(self._review["candidates"])), self.ops.pure_scope(private_stop):
                for sample in range(-1, self._samples):
                    for index, candidate in enumerate(self._review["candidates"]):
                        current()
                        with self._lock:
                            self._progress = {"candidate": index + 1, "candidates": len(self._review["candidates"]),
                                "sample": max(0, sample + 1), "samplesPerCandidate": self._samples,
                                "stage": "warming" if sample < 0 else "measuring"}
                        observed = identity if sample >= 0 else {**identity,
                            "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                        trial, failure, exact = None, None, False
                        with self.ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                            observation = self.recorder.begin(observed, self.primitive(candidate),
                                synchronize=self.ops.synchronize, counters=self.ops.counters, resources=resources)
                            try:
                                trial = self.ops.execute_cached(candidate, frontier, observation, private_stop)
                                if reference is None:
                                    if candidate != self._review["currentCandidate"]: raise RuntimeError("current reference must execute first")
                                    reference = trial
                                exact = self.ops.exact_output(reference, trial)
                                current()
                            except BaseException as error:
                                failure = error
                            receipt = observation.finish(native_state_after=self.ops.native_state(self.owner_session),
                                output_contract_sha256=reference.contract if exact and failure is None else None,
                                status="cancelled" if failure and (isinstance(failure, InterruptedError) or private_stop.is_set()) else "failed" if failure else "completed",
                                error=type(failure).__name__ if failure else None)
                        receipt["exactCacheSamplingEquality"] = exact
                        receipt["actualOutputContractSha256"] = None if trial is None else trial.contract
                        receipt["receiptSha256"] = digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
                        self.ops.append_receipt(job, {"phase": "warmup" if sample < 0 else "measured",
                            "operationId": operation, "observation": receipt})
                        if sample >= 0:
                            issued_head, issued_count = digest([issued_head, receipt["receiptSha256"]]), issued_count + 1
                        if trial is not None and trial is not reference:
                            if exact and failure is None and sample >= 0:
                                previous = retained.get(candidate["planSha256"])
                                if previous is not None: previous.close()
                                retained[candidate["planSha256"]] = trial
                            else: trial.close()
                        if failure is not None: raise failure
                with self._lock: self._progress["stage"] = "selecting"
                decision = select_whole_path_candidate(self.ops.receipts(job), identity=identity,
                    current_candidate=self.primitive(self._review["currentCandidate"]),
                    verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id, minimum_samples=self._samples)
                if self.ops.receipt_head(job) != (issued_head, issued_count): raise RuntimeError("cached evidence custody head differs")
                current()
                if decision["decision"] == "propose-at-quiescence":
                    target = next(item for item in self._review["candidates"] if item["planSha256"] == decision["proposedCandidate"]["planSha256"])
                    if self.ops.auto_allowed(target):
                        chosen = retained.pop(target["planSha256"])
                        self._selected = copy.deepcopy(target)
                self.ops.append_receipt(job, {"phase": "selection", "operationId": operation,
                    "comparison": decision, "exactCacheSamplingRequired": True})
            # Private scopes have ended before the genuine world/cancel boundary.
            current()
            if chosen is None:
                self._reason = "no-exact-admitted-dominant-route-original-normal-step-retained"
            else:
                with self._lock:
                    if self._cancel.is_set(): raise InterruptedError("cached selection stopped before commit")
                    self._committing = True  # Stop cannot undo a checked completed operation.
                boundary_started = True
                if not before_sample(chosen.cache, chosen.neural_state):
                    cache.close()
                    chosen.take_cache()
                    self._phase, self._reason = "cancelled", "normal-chat-cancelled-before-sampling"
                    return None
                sample = self.ops.commit_sampling(frontier, chosen)
                if sample is None:
                    # The boundary already ran ONCE; never replay its world
                    # control callback. Original RNG was not changed by us.
                    chosen.close(); chosen = None
                    self.ops.rebind_original(frontier, self.owner_session)
                    self._phase, self._reason = "failed-fallback", "home-RNG-changed-boundary-not-replayed"
                    self._selected = None
                    return fallback_after_boundary()
                cache.close()
                chosen.take_cache()
                with self._lock:
                    self._phase, self._reason = "completed", "exact-one-shot-cached-route-consumed-no-global-placement"
                    self._consumed = self._matched = True
                return sample
        except BaseException as error:
            with self._lock:
                self._phase = "cancelled" if isinstance(error, InterruptedError) or private_stop.is_set() else "failed-fallback"
                self._reason = type(error).__name__ + ":" + str(error)
                self._selected = None
            try:
                self.ops.append_receipt(job, {"phase": "terminal-fallback", "operationId": operation,
                    "failureType": type(error).__name__, "originalChatReplayed": False})
            except Exception: pass
            if boundary_started:
                # This is the real caller's world/control boundary, not a
                # private calibration error. Its effects must never be
                # replayed by an ordinary-step fallback after it raised.
                raise
        finally:
            if chosen is not None: chosen.close()
            for trial in retained.values(): trial.close()
            if reference is not None: reference.close()
            if frontier is not None:
                self.ops.rebind_original(frontier, self.owner_session)
                self.ops.release_frontier(frontier)
            self.ops.retire_observation()
            with self._lock:
                self._progress = None
                self._committing = False
        # Actual normal fallback executes only once, outside private scope,
        # and sees the original cache/context and unadvanced generator.
        with self._lock:
            if self._phase == "calibrating": self._phase = "completed"
        return fallback()

    def close(self):
        with self._lock:
            self._closed = True
            self._authority_revision += 1
            self._prepare_ticket = None
            self._cancel.set()
            self._review = None
            self._phase, self._reason = "cancelled", "loaded-owner-retired"
        if self.ops is not None: self.ops.retire_observation()
