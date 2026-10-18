"""Process-local measured Auto for ONE exact, isolated decoder prefill.

No Torch/model/I/O on import. Native composition supplies the actual callbacks.
Flags-only status/Stop never inspect a tensor or cancel the supplying chat.
An imported receipt cannot restore a capture, proposal, or execution lease.
"""
from __future__ import annotations

import copy
import contextvars
import functools
import re
import threading
import uuid
import weakref

from .whole_path_execution import (WholePathRecorder, apply_whole_path_proposal,
    comparison_key, digest, select_whole_path_candidate)

WORK_KIND = "native.decoder.prefill-next-logits-isolated-cache"
STAGES = ["exact-placement-admission", "input-and-cue-transfer", "native-prefill",
          "output-home-transfer", "private-cache-release", "home-placement-restore"]
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_CONSUMERS = weakref.WeakKeyDictionary()
_REPLAY = contextvars.ContextVar("omni_decoder_measured_replay", default=False)


def bind_decoder_execution(decoder, consumer):
    existing = _CONSUMERS.get(decoder)
    if existing is not None and existing() not in (None, consumer):
        raise ValueError("decoder already has a different measured execution owner")
    _CONSUMERS[decoder] = weakref.ref(consumer)


def capture_decoder_prefill(method):
    @functools.wraps(method)
    def wrapped(decoder, window, memory_bias, cache):
        reference = _CONSUMERS.get(decoder)
        consumer = None if reference is None else reference()
        if consumer is None or _REPLAY.get() or cache is not None:
            return method(decoder, window, memory_bias, cache)
        return consumer.capture_call(window, memory_bias,
            lambda: method(decoder, window, memory_bias, cache))
    return wrapped


def measured_isolated_prefill(decoder, window, memory_bias, fallback):
    """Only the explicit no-retained-cache caller can consume this lease.

    A different input/state or absent lease retains its original full-forward
    path. Cached continuation, connected decoders and media never call this.
    """
    reference = _CONSUMERS.get(decoder)
    consumer = None if reference is None else reference()
    if consumer is None or _REPLAY.get():
        return fallback()
    if consumer.status()["phase"] == "awaiting-normal-input":
        return consumer.capture_call(window, memory_bias, fallback)
    return consumer.execute_if_matching(window, memory_bias, fallback)


class DecoderPathExecution:
    """Loaded-owner orchestration with injected, resource-admitted native ops.

    Required ops: supported, quiescent, retain_input, release_input,
    materialize_input, describe_input, binding, candidates, identity,
    pure_scope, reference_output, execute, compare, release_output,
    observation_resources, synchronize, counters, append_receipt,
    receipts(job_id), receipt_head(job_id), admit_candidate, auto_allowed.
    Durable receipts use the owner's existing EventLog; active authority does
    not. All neural callbacks run only on the serialized owner lane.
    """
    def __init__(self, owner_id, ops, *, recorder=None, owner_session=None):
        self.owner_id, self.ops = owner_id, ops
        self.owner_session = owner_session or uuid.uuid4().hex
        self.recorder = recorder or WholePathRecorder()
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._queued_stop = None
        self._phase, self._reason = "idle", "not-armed"
        self._operation_id = self._capture_id = None
        self._capture = self._review = self._proposal = self._lease = None
        self._progress = None
        self._measured = self._adopted = 0
        self._ack_cuda = False
        self._closed = False

    def status(self):
        # No ops calls here: this runs concurrently on the bytes-only lane.
        with self._lock:
            return {"format": "omni-decoder-execution-status", "version": 1,
                "ownerId": self.owner_id, "ownerSessionId": self.owner_session,
                "phase": self._phase, "operationId": self._operation_id,
                "captureId": self._capture_id, "reason": self._reason,
                "workKind": WORK_KIND, "progress": copy.deepcopy(self._progress),
                "proposal": copy.deepcopy(self._public_proposal()), "measuredOperations": self._measured,
                "adoptedOperationCalls": self._adopted, "cancellationRequested": self._cancel.is_set(),
                "retainedKvContinuationMeasured": False, "connectedDecoderSupported": False,
                "wholeBrainOrMediaMeasured": False, "globalPlacementChanged": False}

    def _public_proposal(self):
        return None if self._proposal is None else self._proposal["public"]

    def stop(self, *, operation_id=None):
        if operation_id is not None and (not isinstance(operation_id, str) or not _ID.fullmatch(operation_id)):
            raise ValueError("invalid exact decoder calibration operationId")
        with self._lock:
            if operation_id is not None and self._operation_id != operation_id:
                # One slot only; prevents a queued-abort race, cannot cancel a
                # later *different* operation. Main also cancels queued work.
                self._queued_stop = operation_id
                return self.status()
            self._cancel.set()
            self._lease = self._proposal = None
            if self._phase != "calibrating":
                self._phase, self._reason = "cancelled", "explicit-decoder-execution-stop"
            return self.status()

    def release_stopped(self, *, operation_id=None):
        """Serialized cleanup after flags-only ACK; never stops newer work."""
        with self._lock:
            if operation_id is not None and operation_id != self._operation_id:
                raise ValueError("stopped-input cleanup does not own the current exact operation")
            if not self._cancel.is_set() or self._phase == "calibrating":
                raise RuntimeError("stopped-input cleanup cannot retire active/new decoder authority")
        self._clear_input()
        self.ops.retire_observation()
        with self._lock:
            self._capture_id, self._review = None, None
        return self.status()

    def _clear_input(self):
        capture, self._capture = self._capture, None
        if capture is not None:
            self.ops.release_input(capture)

    def capture_next(self, *, acknowledge_cuda=False):
        if type(acknowledge_cuda) is not bool:
            raise ValueError("CUDA CPU-tile acknowledgment must be explicit boolean")
        if self._closed:
            raise RuntimeError("decoder execution owner is closed")
        supported, reason = self.ops.supported()
        if supported: self.ops.quiescent()
        self._clear_input()
        self.ops.retire_observation()
        with self._lock:
            self._cancel.clear()
            self._proposal = self._lease = self._review = None
            self._operation_id = None
            self._capture_id = uuid.uuid4().hex if supported else None
            self._ack_cuda = acknowledge_cuda
            self._phase = "awaiting-normal-input" if supported else "unsupported"
            self._reason = "awaiting-an-actual-normal-prefill-no-synthetic-input" if supported else reason
        return self.status()

    def capture_call(self, window, cue, body):
        with self._lock:
            armed = self._phase == "awaiting-normal-input" and not self._cancel.is_set()
            capture_id = self._capture_id
        retained = None
        if armed:
            try:
                retained = self.ops.retain_input(window, cue)
            except Exception as error:
                with self._lock:
                    self._phase = "failed"
                    self._reason = "normal-turn-preserved-capture-refused:" + type(error).__name__
        try:
            result = body()  # Exactly once, irrespective of capture/refusal.
        except BaseException:
            if retained is not None:
                self.ops.release_input(retained)
            raise
        with self._lock:
            accept = (retained is not None and self._phase == "awaiting-normal-input"
                      and self._capture_id == capture_id and not self._cancel.is_set())
            if accept:
                self._capture = retained
                self._phase, self._reason = "captured", "actual-normal-prefill-input-retained-awaiting-quiescence"
        if retained is not None and not accept:
            self.ops.release_input(retained)
        return result

    def prepare(self, *, topology=None):
        self.ops.quiescent()
        with self._lock:
            if self._phase not in {"captured", "prepared", "proposal", "applied"} or self._cancel.is_set():
                raise RuntimeError("an actual captured normal input is required before decoder review")
            capture, capture_id = self._capture, self._capture_id
        materialized = self.ops.materialize_input(capture)
        if materialized is not capture:
            self.ops.release_input(capture)
            self._capture = materialized
        description = self.ops.describe_input(materialized)
        binding = self.ops.binding(materialized, self.owner_session, capture_id, topology=topology)
        current, candidates = self.ops.candidates(self._ack_cuda)
        if not any(item == current for item in candidates):
            raise RuntimeError("current supported placement is absent from admitted candidates")
        review = {"format": "omni-decoder-execution-review", "version": 1, "ownerId": self.owner_id,
            "binding": binding, "workKind": WORK_KIND, "stageCoverage": list(STAGES),
            "currentCandidate": current, "candidates": candidates, "inputShape": description["inputShape"],
            "cueShape": description["cueShape"], "minimumSamples": 3}
        with self._lock:
            if self._cancel.is_set():
                raise InterruptedError("decoder review was stopped")
            self._review, self._proposal, self._lease = review, None, None
            self._operation_id = None
            self._phase, self._reason = "prepared", "exact-live-owner-input-and-resource-review"
        return {**copy.deepcopy(review), "status": self.status()}

    def _fresh_binding(self):
        return self.ops.binding(self._capture, self.owner_session, self._capture_id)

    def _check_current(self):
        if self._closed or self._cancel.is_set():
            raise InterruptedError("decoder calibration stopped without stopping its supplying chat")
        if self._review is None or self._fresh_binding() != self._review["binding"]:
            raise RuntimeError("decoder input, learned state, topology, geometry or resource policy changed")

    @staticmethod
    def _primitive(candidate):
        return {key: candidate[key] for key in ("planSha256", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged")}

    def calibrate(self, *, operation_id, reviewed, samples_per_candidate=3):
        if not isinstance(operation_id, str) or not _ID.fullmatch(operation_id):
            raise ValueError("calibration requires a main-seeded operationId")
        if type(samples_per_candidate) is not int or samples_per_candidate < 3:
            raise ValueError("calibration requires at least three real samples per candidate")
        self.ops.quiescent()
        with self._lock:
            if self._phase != "prepared" or self._review is None or reviewed != self._review["binding"]:
                raise ValueError("calibration does not match the current exact prepared review")
            self._operation_id = operation_id
            if self._queued_stop == operation_id:
                self._queued_stop = None
                self._cancel.set()
            self._phase, self._reason = "calibrating", "measuring-only-the-declared-isolated-prefill"
        reference, event_ids = None, []
        job_id = "decoder-execution-" + uuid.uuid4().hex
        issued_head, issued_count = "0" * 64, 0
        max_logits = max_state = 0.0
        comparison = None
        try:
            self._check_current()
            # Honest full-forward equivalence for this exact input is required
            # before any no-cache caller can consume a measured prefill lease.
            token = _REPLAY.set(True)
            try:
                with self.ops.calibration_reservation(samples_per_candidate, len(self._review["candidates"])), self.ops.pure_scope(self._cancel):
                    reference = self.ops.reference_output(self._capture, self._cancel)
                    contract = self.ops.output_contract(reference)
                    identity = self.ops.identity(self._review["binding"], self.owner_session, WORK_KIND, STAGES)
                    candidates = self._review["candidates"]
                    for sample in range(-1, samples_per_candidate):
                        for index, candidate in enumerate(candidates):
                            self._check_current()
                            self.ops.quiescent()
                            with self._lock:
                                self._progress = {"candidate": index + 1, "candidates": len(candidates),
                                    "sample": max(0, sample + 1), "samplesPerCandidate": samples_per_candidate,
                                    "stage": "warming" if sample < 0 else "measuring"}
                            output, observation, receipt, failure, checked = None, None, None, None, None
                            observed_identity = identity if sample >= 0 else {**identity,
                                "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                            with self.ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                                observation = self.recorder.begin(observed_identity, self._primitive(candidate),
                                    synchronize=self.ops.synchronize, counters=self.ops.counters, resources=resources)
                                try:
                                    output = self.ops.execute(candidate, self._capture, observation, self._cancel)
                                    checked = self.ops.compare(reference, output)
                                    max_logits = max(max_logits, checked["maximumAbsoluteLogitError"] or 0.0)
                                    max_state = max(max_state, checked["maximumAbsoluteStateError"] or 0.0)
                                    if not checked["equivalent"]:
                                        raise RuntimeError("actual native output is outside the unchanged float32 equivalence contract")
                                    self._check_current()
                                except BaseException as error:
                                    failure = error
                                finally:
                                    if output is not None:
                                        self.ops.release_output(output)
                                    receipt = observation.finish(native_state_after=self.ops.native_state(self.owner_session),
                                        output_contract_sha256=contract if failure is None else None,
                                        status="cancelled" if isinstance(failure, InterruptedError) else "failed" if failure else "completed",
                                        error=type(failure).__name__ if failure else None)
                                    # Preserve actual numerical evidence, including differing
                                    # raw output hashes. Contract SHA is the CHECKED tolerance
                                    # contract, never a claim of bit-identical CPU/Metal logits.
                                    receipt["outputEquivalence"] = checked
                                    receipt["receiptSha256"] = digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
                            # Warm-up is preserved but never included in selection.
                            event_ids.append(self.ops.append_receipt(job_id, {"operationId": operation_id,
                                "phase": "warmup" if sample < 0 else "measured", "observation": receipt}))
                            if sample >= 0:
                                issued_head = digest([issued_head, receipt["receiptSha256"]])
                                issued_count += 1
                                with self._lock: self._measured += 1
                            if failure is not None:
                                raise failure
                    with self._lock: self._progress["stage"] = "selecting"
                    comparison = select_whole_path_candidate(self.ops.receipts(job_id), identity=identity,
                        current_candidate=self._primitive(self._review["currentCandidate"]),
                        verify_receipt=lambda row: row["observerId"] == self.recorder.observer_id,
                        minimum_samples=samples_per_candidate)
                    if self.ops.receipt_head(job_id) != (issued_head, issued_count):
                        raise RuntimeError("durable observation custody differs from this live operation's issued evidence")
                    self._check_current()
                    if comparison["decision"] == "propose-at-quiescence":
                        selected = next(item for item in candidates if item["planSha256"] == comparison["proposedCandidate"]["planSha256"])
                        public = {"proposalId": uuid.uuid4().hex, "proposalSha256": comparison["proposalSha256"],
                            "workKind": WORK_KIND, "candidate": copy.deepcopy(selected),
                            "comparisonSha256": comparison_key(identity), "outputContractSha256": contract,
                            "baselineMeanNanoseconds": str(int(comparison["baselineEvidence"]["sumNanoseconds"]) // comparison["baselineEvidence"]["samples"]),
                            "candidateMeanNanoseconds": str(int(comparison["candidateEvidence"]["sumNanoseconds"]) // comparison["candidateEvidence"]["samples"]),
                            "minimumSamples": samples_per_candidate, "scope": "exact-isolated-prefill-operation"}
                        with self._lock:
                            if self._cancel.is_set(): raise InterruptedError("decoder proposal was stopped")
                            self._proposal = {"public": public, "comparison": comparison, "identity": identity,
                                "binding": copy.deepcopy(reviewed), "candidate": selected}
                    self.ops.append_receipt(job_id, {"operationId": operation_id, "phase": "selection", "comparison": comparison})
                    with self._lock:
                        if self._cancel.is_set(): raise InterruptedError("decoder selection was stopped")
                        self._phase = "proposal" if self._proposal else "prepared"
                        self._reason = comparison["reason"]
            finally:
                _REPLAY.reset(token)
        except BaseException as error:
            with self._lock:
                self._proposal = self._lease = None
                self._phase = "cancelled" if isinstance(error, InterruptedError) or self._cancel.is_set() else "failed"
                self._reason = str(error)
            try:
                self.ops.append_receipt(job_id, {"operationId": operation_id, "phase": "terminal-failure",
                    "failureType": type(error).__name__, "completedObservationCount": issued_count,
                    "cancellationRequested": self._cancel.is_set()})
            except Exception:
                pass  # A refused journal allocation cannot fabricate persistence.
            if self._cancel.is_set() and not isinstance(error, InterruptedError):
                raise InterruptedError("decoder calibration cancelled at its native safe boundary") from error
            raise
        finally:
            if reference is not None:
                self.ops.release_output(reference)
            if self._cancel.is_set():
                self._clear_input()  # Serialized completion, never control-thread Torch work.
                self.ops.retire_observation()
                with self._lock:
                    self._capture_id, self._review = None, None
            with self._lock: self._progress = None
        return {"format": "omni-decoder-execution-calibration", "version": 1,
            "ownerId": self.owner_id, "operationId": operation_id, "binding": copy.deepcopy(reviewed),
            "status": self.status(), "proposal": copy.deepcopy(self._public_proposal()), "receiptEventIds": event_ids,
            "outputEquivalence": {"checked": True, "relativeTolerance": self.ops.relative_tolerance,
                "absoluteTolerance": self.ops.absolute_tolerance, "noSamplingOrRngDraws": True,
                "maximumAbsoluteLogitError": max_logits, "maximumAbsoluteStateError": max_state}}

    def apply(self, *, proposal_id, proposal_sha256):
        self._check_current()
        value = self._proposal
        if value is None or value["public"]["proposalId"] != proposal_id or value["public"]["proposalSha256"] != proposal_sha256:
            raise ValueError("unknown or already-consumed live decoder proposal")
        def authorize(_proposal):
            if self._proposal is not value:
                return False
            self._proposal = None  # One-use even if subsequent admission fails.
            return True
        def adopt(_candidate):
            if not self.ops.auto_allowed(value["candidate"]):
                raise RuntimeError("current device policy does not allow this measured Auto route")
            with self._lock:
                if self._cancel.is_set(): raise InterruptedError("decoder adoption was stopped")
                self._lease = copy.deepcopy(value)
        def revalidate():
            self._check_current()
            current, _candidates = self.ops.candidates(self._ack_cuda)
            return (self.ops.identity(self._fresh_binding(), self.owner_session, WORK_KIND, STAGES), self._primitive(current))
        try:
            apply_whole_path_proposal(value["comparison"], identity=value["identity"],
                current_candidate=self._primitive(self._review["currentCandidate"]), authorize_proposal=authorize,
                revalidate=revalidate,
                quiescent=self.ops.quiescent, admit_candidate=lambda _value: self.ops.admit_candidate(value["candidate"]),
                apply_candidate=adopt)
        except BaseException as error:
            with self._lock:
                self._lease = self._proposal = None
                self._phase, self._reason = "failed", str(error)
            raise
        with self._lock:
            self._phase, self._reason = "applied", "exact-isolated-prefill-lease-only-global-placement-unchanged"
        return self.status()

    def execute_if_matching(self, window, cue, fallback):
        lease = self._lease
        if lease is None or self._cancel.is_set():
            return fallback()
        # Shape/version fast rejection precedes any materialization/hash.
        if not self.ops.input_may_match(self._capture, window, cue):
            return fallback()
        if (self._fresh_binding() != lease["binding"] or not self.ops.matches_input(self._capture, window, cue)
                or not self.ops.auto_allowed(lease["candidate"])):
            self._lease = None
            return fallback()
        self.ops.quiescent()
        token = _REPLAY.set(True)
        try:
            # Same actual declared operation and source admission path used in
            # calibration, never a global backend preference or retained cache.
            try:
                with self.ops.pure_scope(self._cancel):
                    output = self.ops.execute(lease["candidate"], self._capture, None, self._cancel)
            except InterruptedError:
                self._lease = None
                return fallback()  # Calibration Stop must not stop normal chat.
            if self._cancel.is_set():
                self.ops.release_output(output)
                self._lease = None
                return fallback()
            if self._fresh_binding() != lease["binding"]:
                self.ops.release_output(output)
                self._lease = None
                raise RuntimeError("exact isolated-prefill owner changed during consumption")
            with self._lock: self._adopted += 1
            return self.ops.consumer_result(output)
        finally:
            _REPLAY.reset(token)

    def close(self):
        self._cancel.set()
        self._clear_input()
        if self.ops is not None: self.ops.retire_observation()
        with self._lock:
            self._closed = True
            self._lease = self._proposal = self._review = None
            self._phase, self._reason = "cancelled", "loaded-owner-closed-no-authority-restored"
