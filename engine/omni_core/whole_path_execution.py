"""Whole declared native-operation observations and conservative proposals.

No model, Torch, hardware probe, I/O, callback installation or device adoption
on import. Runtime composition must supply actual stage/counter/owner-fence
observations and authenticate its own receipts; a checksum alone is not trust.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ASYNC = re.compile(r"mps|cuda:[0-9]+\Z")
COUNTERS = ("coreReadBytes", "coreWriteBytes", "coreTransferBytes", "corePageIns", "corePageOuts",
            "attentionReadBytes", "attentionWriteBytes", "activationTransferBytes")
COUNTER_BASIS = {"coreReadBytes": "logical-native-pager-file-or-mapped-page-visits-not-physical-disk-traffic",
    "coreWriteBytes": "managed-native-pager-file-writes", "coreTransferBytes": "materialized-packed-buffer-transfers",
    "corePageIns": "packed-owner-accelerator-admissions", "corePageOuts": "packed-owner-accelerator-evictions",
    "attentionReadBytes": "managed-attention-file-reads", "attentionWriteBytes": "managed-attention-file-writes",
    "activationTransferBytes": "materialized-activation-transfers"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _sha(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _integer(value):
    return type(value) is int and value >= 0


def _identity(value):
    required = {"ownerId", "ownerSessionId", "workKind", "stageCoverage", "workloadSha256", "geometrySha256",
                "samplingContractSha256", "policySha256", "topologySha256", "budgetSha256", "cache", "nativeState"}
    if (not isinstance(value, dict) or set(value) != required
            or any(not isinstance(value.get(key), str) or not value[key] for key in ("ownerId", "ownerSessionId", "workKind"))
            or any(not _sha(value.get(key)) for key in required if key.endswith("Sha256"))
            or not isinstance(value["stageCoverage"], list) or not value["stageCoverage"]
            or any(not isinstance(stage, str) or not stage for stage in value["stageCoverage"])
            or len(set(value["stageCoverage"])) != len(value["stageCoverage"])):
        raise ValueError("whole-path identity needs exact owner/work/stage/geometry/sampling/budget bindings")
    cache = value["cache"]
    if (not isinstance(cache, dict) or set(cache) != {"heat", "logicalStateSha256", "backendInitialization"}
            or cache["heat"] not in {"cold", "warm", "unknown"}
            or cache["backendInitialization"] not in {"cold", "warm", "unknown"}
            or not (_sha(cache["logicalStateSha256"]) or cache["logicalStateSha256"] is None)):
        raise ValueError("whole-path cache identity is incomplete")
    _state(value["nativeState"])
    return copy.deepcopy(value)


def _state(value):
    if (not isinstance(value, dict) or set(value) != {"kind", "bindingSha256", "ownerSessionId", "complete", "ownerVersions"}
            or value["kind"] not in {"committed-generation", "live-owner-epochs", "unproven"}
            or not (_sha(value["bindingSha256"]) or value["bindingSha256"] is None)
            or type(value["complete"]) is not bool or not isinstance(value["ownerSessionId"], str)
            or not isinstance(value["ownerVersions"], dict)
            or any(not isinstance(owner, str) or not owner or not _integer(version) for owner, version in value["ownerVersions"].items())):
        raise ValueError("whole-path native-state fence is invalid")
    return copy.deepcopy(value)


def _candidate(value):
    if (not isinstance(value, dict) or set(value) != {"planSha256", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged"}
            or not _sha(value["planSha256"]) or type(value["supported"]) is not bool
            or type(value["fallbackAcknowledged"]) is not bool or not isinstance(value["asyncDevices"], list)
            or any(not isinstance(device, str) or not _ASYNC.fullmatch(device) for device in value["asyncDevices"])
            or len(set(value["asyncDevices"])) != len(value["asyncDevices"])):
        raise ValueError("whole-path candidate needs an exact supported plan/device descriptor")
    mode, devices = value["backendMode"], value["asyncDevices"]
    # asyncDevices describes ALL actual completion participants, including a
    # GPU home-output transfer for CPU computation. It is not a claim that CPU
    # mode secretly computes on that GPU. Production binds the exact compute
    # target/home union to the checked plan descriptor and workload geometry.
    if (mode == "native-cpu" and value["fallbackAcknowledged"]
            or mode == "native-metal" and ("mps" not in devices or value["fallbackAcknowledged"])
            or mode == "explicit-cuda-cpu-tile" and (not any(device.startswith("cuda:") for device in devices) or not value["fallbackAcknowledged"])
            or mode == "native-mixed-sequential" and (not devices or value["fallbackAcknowledged"] != any(device.startswith("cuda:") for device in devices))
            or mode not in {"native-cpu", "native-metal", "explicit-cuda-cpu-tile", "native-mixed-sequential"}):
        raise ValueError("whole-path mode cannot invent a native CUDA/XPU/NPU backend or inherit fallback authority")
    return copy.deepcopy(value)


def _counter_snapshot(value):
    if not isinstance(value, dict) or set(value) != set(COUNTERS) or any(not (_integer(item) or item is None) for item in value.values()):
        raise ValueError("whole-path counters require their exact monotonic source fields; missing values are None")
    return dict(value)


def _resource_snapshot(value):
    if (not isinstance(value, dict) or set(value) != {"budgetSha256", "admissionComplete", "pressureObserved", "physicalGroups"}
            or not _sha(value["budgetSha256"]) or type(value["admissionComplete"]) is not bool
            or type(value["pressureObserved"]) is not bool or not isinstance(value["physicalGroups"], dict)):
        raise ValueError("whole-path resource observations need a bound actual admission envelope")
    for group, row in value["physicalGroups"].items():
        if (not isinstance(group, str) or not group or not isinstance(row, dict)
                or set(row) != {"ceilingBytes", "observedPeakBytes", "peakObservation"}
                or not _integer(row["ceilingBytes"]) or not (_integer(row["observedPeakBytes"]) or row["observedPeakBytes"] is None)
                or row["peakObservation"] not in {"operation-start", "sampled-full-operation", "unavailable"}):
            raise ValueError("whole-path resource peaks must identify their physical group and observed scope")
    return copy.deepcopy(value)


def comparison_key(identity):
    """Candidate/backend excluded; all actual-work/state/heat/budget facts kept."""
    return digest(_identity(identity))


class WholePathRecorder:
    def __init__(self, *, clock_ns=time.monotonic_ns, origin="runtime-source-callback"):
        if origin not in {"runtime-source-callback", "inert-fixture"}:
            raise ValueError("whole-path observation origin is invalid")
        self.clock_ns, self.origin = clock_ns, origin
        self.observer_id, self._sequence = uuid.uuid4().hex, 0

    def begin(self, identity, candidate, *, synchronize, counters, resources):
        identity, candidate = _identity(identity), _candidate(candidate)
        # Caller has already admitted this actual operation. A pre-start drain
        # excludes earlier asynchronous work; finish includes its own completion.
        drain_start = self.clock_ns()
        cold_backend = identity["cache"]["backendInitialization"] == "cold"
        # A genuinely proven cold backend has no prior native work to drain.
        # Synchronizing it here could initialize it outside the measured total.
        drained = [] if cold_backend else list(synchronize(tuple(candidate["asyncDevices"])))
        drain_end = self.clock_ns()
        start = self.clock_ns()
        self._sequence += 1
        return WholePathObservation(self, identity, candidate, start, drain_end - drain_start,
            drained, synchronize, counters, resources, _counter_snapshot(counters()), _resource_snapshot(resources()), self._sequence)


class WholePathObservation:
    def __init__(self, recorder, identity, candidate, start, drain_ns, drained, synchronize, counters, resources, before, resource_before, sequence):
        self.recorder, self.identity, self.candidate = recorder, identity, candidate
        self.start, self.drain_ns, self.drained = start, drain_ns, drained
        self.synchronize, self.counters, self.resources = synchronize, counters, resources
        self.before, self.resource_before = before, resource_before
        self.completed_stages, self.closed = [], False
        self.sequence = sequence

    def stage_completed(self, stage):
        if self.closed:
            raise RuntimeError("whole-path observation is already closed")
        self.completed_stages.append(stage)

    def finish(self, *, native_state_after, output_contract_sha256=None, status="completed", error=None):
        if self.closed:
            raise RuntimeError("whole-path observation cannot be completed twice")
        self.closed = True
        reasons = []
        if status not in {"completed", "cancelled", "failed"}:
            raise ValueError("whole-path terminal status is invalid")
        if status != "completed":
            reasons.append(status)
        sync_start = self.recorder.clock_ns()
        try:
            synchronized = list(self.synchronize(tuple(self.candidate["asyncDevices"])))
        except Exception as failure:
            synchronized = []
            reasons.append("completion-synchronization-failed")
            error = error or type(failure).__name__
        end = self.recorder.clock_ns()
        try:
            after = _counter_snapshot(self.counters())
        except Exception as failure:
            after = dict.fromkeys(COUNTERS)
            reasons.append("terminal-counter-capture-failed")
            error = error or type(failure).__name__
        try:
            resource_after = _resource_snapshot(self.resources())
        except Exception as failure:
            resource_after = {**self.resource_before, "admissionComplete": False, "physicalGroups": {}}
            reasons.append("terminal-resource-capture-failed")
            error = error or type(failure).__name__
        native_state_after = _state(native_state_after)
        if end <= self.start or sync_start < self.start or self.drain_ns < 0:
            reasons.append("invalid-monotonic-duration")
        required_devices = self.candidate["asyncDevices"]
        cold_backend = self.identity["cache"]["backendInitialization"] == "cold"
        if ((not cold_backend and sorted(self.drained) != sorted(required_devices))
                or cold_backend and self.drained or sorted(synchronized) != sorted(required_devices)):
            reasons.append("incomplete-device-synchronization")
        if self.completed_stages != self.identity["stageCoverage"]:
            reasons.append("incomplete-or-different-native-stage-coverage")
        before_state = self.identity["nativeState"]
        if (before_state["kind"] == "unproven" or not before_state["complete"] or not before_state["ownerVersions"]
                or not _sha(before_state["bindingSha256"]) or before_state["ownerSessionId"] != self.identity["ownerSessionId"]):
            reasons.append("native-owner-fence-unproven")
        if before_state != native_state_after:
            reasons.append("native-owner-fence-changed-during-operation")
        cache = self.identity["cache"]
        if cache["heat"] == "unknown" or cache["backendInitialization"] == "unknown" or not _sha(cache["logicalStateSha256"]):
            reasons.append("cache-or-initialization-state-unproven")
        if not self.candidate["supported"]:
            reasons.append("candidate-backend-unsupported")
        if not _sha(output_contract_sha256):
            reasons.append("sampling-output-contract-unverified")
        deltas = {}
        for key in COUNTERS:
            first, last = self.before[key], after[key]
            if first is None or last is None or last < first:
                deltas[key] = None
                reasons.append("incomplete-or-reset-paging-transfer-counters")
            else:
                deltas[key] = str(last - first)
        for reading in (self.resource_before, resource_after):
            if (reading["budgetSha256"] != self.identity["budgetSha256"] or not reading["admissionComplete"]
                    or reading["pressureObserved"] or not reading["physicalGroups"]):
                reasons.append("resource-envelope-not-admitted")
            for row in reading["physicalGroups"].values():
                if ((reading is resource_after and row["peakObservation"] != "sampled-full-operation")
                        or row["peakObservation"] == "unavailable" or row["observedPeakBytes"] is None or row["observedPeakBytes"] > row["ceilingBytes"]):
                    reasons.append("resource-peak-unavailable-or-over-budget")
        if ({group: row["ceilingBytes"] for group, row in self.resource_before["physicalGroups"].items()}
                != {group: row["ceilingBytes"] for group, row in resource_after["physicalGroups"].items()}):
            reasons.append("physical-budget-changed-during-operation")
        payload = {"format": "omni-declared-whole-path-observation", "version": 1, "id": uuid.uuid4().hex,
            "observerId": self.recorder.observer_id, "observationSequence": str(self.sequence),
            "origin": self.recorder.origin, "identity": self.identity, "comparisonKeySha256": comparison_key(self.identity),
            "candidate": self.candidate, "status": status, "error": error,
            "completedStages": list(self.completed_stages), "nativeStateAfter": native_state_after,
            "outputContractSha256": output_contract_sha256, "totalNanoseconds": str(max(0, end - self.start)),
            "completionSynchronizationNanoseconds": str(max(0, end - sync_start)),
            "priorWorkDrainExcludedNanoseconds": str(max(0, self.drain_ns)),
            "priorWorkDrainMode": "not-required-proven-cold-backend" if cold_backend else "completed-before-timed-operation",
            "drainedDevices": self.drained, "synchronizedDevices": synchronized,
            "counterDeltas": deltas, "resourceBefore": self.resource_before, "resourceAfter": resource_after,
            "counterBasis": dict(COUNTER_BASIS), "physicalDiskTrafficMeasured": False,
            "eligible": not reasons, "ineligibleReasons": sorted(set(reasons)),
            "scope": "entire-declared-native-stage-coverage-not-host-network-user-wait-or-undeclared-brain-stages"}
        return {**payload, "receiptSha256": digest(payload)}


def select_whole_path_candidate(receipts, *, identity, current_candidate, verify_receipt, minimum_samples=3):
    """Stream complete receipts; no unbounded latency list or device adoption.

    A required caller verifier binds source/local custody. Imported JSON plus a
    recomputable public checksum is not authenticated runtime evidence. Dominant
    observed ranges are a conservative finite-sample rule, not a confidence or
    quality guarantee. No matched evidence always retains the current fallback.
    """
    identity, current_candidate = _identity(identity), _candidate(current_candidate)
    if type(minimum_samples) is not int or minimum_samples < 2 or not callable(verify_receipt):
        raise ValueError("whole-path selection needs its runtime verifier and at least two matched samples")
    key, current_sha = comparison_key(identity), current_candidate["planSha256"]
    groups, ignored, seen_sequence = {}, 0, {}
    for receipt in receipts:
        try:
            payload = {name: value for name, value in receipt.items() if name != "receiptSha256"}
            if (not verify_receipt(receipt) or digest(payload) != receipt["receiptSha256"]
                    or receipt["format"] != "omni-declared-whole-path-observation" or receipt["version"] != 1
                    or receipt["origin"] != "runtime-source-callback" or not receipt["eligible"]
                    or receipt["status"] != "completed" or receipt["comparisonKeySha256"] != key
                    or _identity(receipt["identity"]) != identity or not _sha(receipt["outputContractSha256"])):
                ignored += 1
                continue
            candidate = _candidate(receipt["candidate"])
            if not candidate["supported"]:
                ignored += 1
                continue
            sequence = int(receipt["observationSequence"])
            observer = receipt["observerId"]
            if (sequence < 1 or str(sequence) != receipt["observationSequence"] or not isinstance(observer, str)
                    or sequence <= seen_sequence.get(observer, 0)):
                ignored += 1
                continue
            # Source iterator must be ordered within each observer. This fences
            # duplicate/page-retry receipts without an ever-growing ID set.
            seen_sequence[observer] = sequence
            duration = int(receipt["totalNanoseconds"])
            if duration <= 0 or str(duration) != receipt["totalNanoseconds"]:
                raise ValueError("invalid whole-path latency")
            # Different outputs cannot win just by finishing less/different work.
            group_key = (candidate["planSha256"], receipt["outputContractSha256"])
            row = groups.setdefault(group_key, {"candidate": candidate, "count": 0, "sum": 0, "min": duration, "max": duration,
                "evidenceHeadSha256": "0" * 64})
            if candidate != row["candidate"]:
                raise ValueError("same plan SHA has contradictory backend evidence")
            row["count"] += 1
            row["sum"] += duration
            row["min"], row["max"] = min(row["min"], duration), max(row["max"], duration)
            row["evidenceHeadSha256"] = digest([row["evidenceHeadSha256"], receipt["receiptSha256"]])
        except (KeyError, TypeError, ValueError):
            ignored += 1
    baseline = [(output, row) for (plan, output), row in groups.items() if plan == current_sha and row["count"] >= minimum_samples
                and row["candidate"] == current_candidate]
    proposed, reference, selected_output = None, None, None
    for output, current in baseline:
        for (plan, candidate_output), row in groups.items():
            if (plan == current_sha or candidate_output != output or row["count"] < minimum_samples
                    or row["max"] >= current["min"]):
                continue
            if proposed is None or row["sum"] * proposed["count"] < proposed["sum"] * row["count"]:
                proposed, reference, selected_output = row, current, output
    def summary(row):
        return None if row is None else {"samples": row["count"], "sumNanoseconds": str(row["sum"]),
            "minimumNanoseconds": str(row["min"]), "maximumNanoseconds": str(row["max"]),
            "evidenceHeadSha256": row["evidenceHeadSha256"]}
    result = {"format": "omni-declared-whole-path-selection-proposal", "version": 1,
        "comparisonKeySha256": key, "ownerId": identity["ownerId"], "ownerSessionId": identity["ownerSessionId"],
        "policySha256": identity["policySha256"], "topologySha256": identity["topologySha256"], "budgetSha256": identity["budgetSha256"],
        "currentCandidate": current_candidate, "proposedCandidate": proposed["candidate"] if proposed else current_candidate,
        "decision": "propose-at-quiescence" if proposed else "retain-supported-current-fallback",
        "reason": "matched-whole-declared-path-nonoverlapping-observed-latency-ranges" if proposed else "no-matched-admitted-dominant-whole-path-evidence",
        "baselineEvidence": summary(reference), "candidateEvidence": summary(proposed), "outputContractSha256": selected_output,
        "ignoredReceipts": ignored, "minimumSamples": minimum_samples,
        "automaticAdoptionPerformed": False, "savedManualRoutesArmed": False,
        "statisticalConfidenceClaimed": False, "neuralQualityMeasured": False}
    return {**result, "proposalSha256": digest(result)}


def apply_whole_path_proposal(proposal, *, identity, current_candidate, authorize_proposal, revalidate, quiescent, admit_candidate, apply_candidate):
    """Explicit trusted caller seam. Recheck every binding, then apply once.

    Caller authorization must consume its own one-use proposal authority. This function
    never changes preferences/arms regional routes by itself or retries a failed
    apply. Actual effects are only the explicitly supplied apply callback.
    """
    identity, current_candidate = _identity(identity), _candidate(current_candidate)
    payload = {name: value for name, value in proposal.items() if name != "proposalSha256"}
    if (digest(payload) != proposal.get("proposalSha256") or proposal.get("comparisonKeySha256") != comparison_key(identity)
            or proposal.get("ownerSessionId") != identity["ownerSessionId"] or proposal.get("currentCandidate") != current_candidate):
        raise ValueError("whole-path proposal is stale for current work/state/topology/budget/owner")
    if proposal.get("decision") != "propose-at-quiescence":
        return {"applied": False, "reason": "supported-current-fallback-retained"}
    candidate = _candidate(proposal["proposedCandidate"])
    if not candidate["supported"]:
        raise ValueError("whole-path proposed backend is unavailable")
    if not authorize_proposal(copy.deepcopy(proposal)):
        raise ValueError("whole-path proposal has no current one-use main authority")
    quiescent()  # Must refuse live cache/pins/transactions, never clear them.
    def current_bindings():
        fresh_identity, fresh_candidate = revalidate()
        if _identity(fresh_identity) != identity or _candidate(fresh_candidate) != current_candidate:
            raise ValueError("whole-path proposal changed at the current safe/admission boundary")
    current_bindings()
    admit_candidate(copy.deepcopy(candidate))
    current_bindings()
    apply_candidate(copy.deepcopy(candidate))
    return {"applied": True, "candidate": candidate, "proposalSha256": proposal["proposalSha256"],
            "savedManualRoutesArmed": False}
