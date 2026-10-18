"""Request-owned aggregate scheduling of fixed effects and complete compute.

No neural dependency or work on import. Memory effects stay on their actual
owner and execute once. Only authenticated complete-operation observations may
select compute placement after that already completed fixed prefix.
"""
from __future__ import annotations

import copy
import functools
import json
import time
import uuid
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar

from .operation_auto_registry import PRIVATE_OPERATION
from .whole_path_execution import comparison_key, digest

_CURRENT = ContextVar("omni_actual_owned_whole_path", default=None)
_WORLD_BARRIER = ContextVar("omni_actual_world_effect_barrier", default=0)
_CHOICE = ContextVar("omni_actual_prepared_operation_choice", default=None)
_QUALIFICATION_GATE = ContextVar("omni_actual_request_qualification_gate", default=None)
_WORK = {"chat", "resume_chat", "resume_inherited_chat", "consolidate_chat_learning", "learn_action_result",
    "learn_tool_route_outcome", "learn_capability_pack", "rebind_learning_source", "train", "ingest", "idle_cycle",
    "feedback", "observe_packet", "generate_modality", "generate_neural_speech", "full_merge_activate_ingestion",
    "evolution_propose", "evolution_evaluate", "evolution_promote", "evolution_rollback", "checkpoint", "snapshot",
    "evolution.propose", "evolution.evaluate", "evolution.promote", "evolution.rollback"}


def current_owned_path():
    """Runtime custody only: a saved receipt never recreates this scope."""
    path = _CURRENT.get()
    return None if path is None or path.closed else path


def qualification_allowed():
    if PRIVATE_OPERATION.get() or _WORLD_BARRIER.get(): return False
    gate = _QUALIFICATION_GATE.get()
    try: return gate is None or bool(gate())
    except Exception: return False


@contextmanager
def request_qualification_gate(predicate):
    previous = _QUALIFICATION_GATE.get()
    token = _QUALIFICATION_GATE.set(predicate if previous is None else lambda: previous() and predicate())
    try: yield
    finally: _QUALIFICATION_GATE.reset(token)


@contextmanager
def world_effect_barrier():
    """Real callbacks are once-only and cannot recursively qualify operations."""
    if PRIVATE_OPERATION.get():
        raise RuntimeError("private operation cannot enter a real world callback")
    token = _WORLD_BARRIER.set(_WORLD_BARRIER.get() + 1)
    try: yield
    finally: _WORLD_BARRIER.reset(token)


@contextmanager
def owned_phase(owner, stage, *, publication=False):
    """Execute an admitted actual-owner phase once, never a private replay."""
    if PRIVATE_OPERATION.get():
        raise RuntimeError("private operation cannot enter an actual owner phase")
    path = current_owned_path()
    if path is None:
        yield
        return
    try:
        belongs = path.scheduler.ops.owns(owner)
        current = path.scheduler.ops.binding() == path.binding
    except Exception:
        belongs = current = False
    if not belongs or not current:
        path.disabled = "actual-phase-owner-or-policy-changed"
        with world_effect_barrier(): yield
        return
    with path.scheduler.effect(path, stage, admit=not publication): yield


def current_operation_choice():
    value = _CHOICE.get()
    path = current_owned_path()
    return value if path is not None and value is not None and value["path"] is path else None


@contextmanager
def owned_operation_choice(controller, prepared_identity, exact_candidate):
    """Associate a *real* consumer's current prepared choice, not a proposal.

    The caller retains its native frontier/equality/admission checks. This
    context cannot authorize a route, load a model or turn an imported receipt
    into execution authority. A candidate may be a validated mixed route.
    """
    if PRIVATE_OPERATION.get():
        raise RuntimeError("private trials cannot publish an actual operation choice")
    path = current_owned_path()
    value = None
    if path is not None:
        try:
            binding = path.scheduler.ops.binding()
            brain = path.scheduler.ops.brain
            valid = (getattr(brain, "_native_operation_auto", None) is controller
                and binding == path.binding
                and prepared_identity["ownerSessionId"] == binding["ownerSessionId"]
                and prepared_identity["ownerId"] == brain.brain_id
                and prepared_identity["policySha256"] == binding["policySha256"]
                and prepared_identity["budgetSha256"] == binding["budgetSha256"])
            if valid:
                value = {"path": path, "controller": controller,
                    "identity": prepared_identity, "candidate": exact_candidate, "aborted": False}
        except Exception:
            pass
    token = _CHOICE.set(value)
    started, failed = None if path is None else path.scheduler.clock(), False
    try: yield value
    except BaseException:
        failed = True
        raise
    finally:
        _CHOICE.reset(token)
        if value is not None:
            # Bounded counters by declared kind, not retained per-token outputs.
            try:
                kind = prepared_identity["workKind"]
                row = path.consumed.setdefault(kind, {"entered": 0, "completed": 0, "aborted": 0, "nanoseconds": "0"})
                row["entered"] += 1
                row["completed"] += not failed and not value["aborted"]
                row["aborted"] += bool(value["aborted"])
                row["nanoseconds"] = str(int(row["nanoseconds"]) + max(0, path.scheduler.clock() - started))
            except Exception:
                path.disabled = "actual-choice-observation-unavailable"


@contextmanager
def owned_request_path(brain, operation, request_id):
    controller = getattr(brain, "_native_operation_auto", None)
    scheduler = getattr(getattr(controller, "ops", None), "whole_path_scheduler", None)
    if operation not in _WORK or scheduler is None or not scheduler.ops.armed():
        token = _CURRENT.set(None)
        try: yield
        finally: _CURRENT.reset(token)
        return
    with scheduler.request(operation, request_id): yield


def owned_memory_effect(stage):
    def decorate(method):
        @functools.wraps(method)
        def wrapped(owner, *args, **kwargs):
            if PRIVATE_OPERATION.get():
                raise RuntimeError("private placement trials cannot execute lasting memory or STDP effects")
            path = _CURRENT.get()
            if path is None: return method(owner, *args, **kwargs)
            try: belongs = path.scheduler.ops.owns(owner)
            except Exception as error:
                path.disabled = "memory-owner-observation:" + type(error).__name__
                return method(owner, *args, **kwargs)
            if not belongs:
                path.disabled = "unbound-memory-effect-owner"
                return method(owner, *args, **kwargs)
            with path.scheduler.effect(path, stage): return method(owner, *args, **kwargs)
        return wrapped
    return decorate


def select_complete_operation(controller, receipts, *, selector, **kwargs):
    scheduler = getattr(controller.ops, "whole_path_scheduler", None)
    if scheduler is None: return selector(receipts, **kwargs)
    return scheduler.select(receipts, selector=selector, **kwargs)


class _Path:
    def __init__(self, scheduler, operation, request_id):
        self.scheduler, self.operation = scheduler, operation
        self.request_id = str(request_id)
        self.id = uuid.uuid4().hex
        self.request_sha = digest([operation, str(request_id), self.id])
        self.started = scheduler.clock()
        self.depth, self.fixed_ns, self.effects, self.failed = 0, 0, 0, False
        self.excluded_drain_ns = 0
        self.phases, self.head, self.frontier = {}, "0" * 64, None
        self.consumed = {}
        self.resources, self.binding, self.disabled = None, scheduler.ops.binding(), None
        self.closed = False


def _resources(left, right):
    """Sequential phases share one physical ceiling and use max, never sum."""
    for value in (right,) if left is None else (left, right):
        if (not isinstance(value, dict) or not value.get("admissionComplete") or value.get("pressureObserved")
                or not isinstance(value.get("physicalGroups"), dict) or not value["physicalGroups"]):
            raise ValueError("fixed/compute resource envelope is missing or was not admitted")
        for row in value["physicalGroups"].values():
            if (type(row.get("observedPeakBytes")) is not int or type(row.get("ceilingBytes")) is not int
                    or row["observedPeakBytes"] > row["ceilingBytes"] or row.get("peakObservation") != "sampled-full-operation"):
                raise ValueError("fixed/compute physical peak is unproven or exceeds its ceiling")
    if left is None: return copy.deepcopy(right)
    if left["budgetSha256"] != right["budgetSha256"]:
        raise ValueError("fixed/compute resource envelope changed")
    result = copy.deepcopy(left)
    for group, value in right["physicalGroups"].items():
        current = result["physicalGroups"].get(group)
        if current is None:
            result["physicalGroups"][group] = copy.deepcopy(value)
            continue
        if current["ceilingBytes"] != value["ceilingBytes"]: raise ValueError("one physical group has contradictory ceilings")
        first, second = current["observedPeakBytes"], value["observedPeakBytes"]
        if type(first) is not int or type(second) is not int: raise ValueError("physical usage is unmeasured")
        current["observedPeakBytes"] = max(first, second)
        current["peakObservation"] = "sampled-full-operation"
    for row in result["physicalGroups"].values():
        if (type(row["observedPeakBytes"]) is not int or row["observedPeakBytes"] > row["ceilingBytes"]
                or row["peakObservation"] != "sampled-full-operation"):
            raise ValueError("fixed/compute physical peak is unproven or exceeds the ceiling")
    return result


class WholePathScheduler:
    def __init__(self, ops, *, clock_ns=time.monotonic_ns):
        self.ops, self.clock = ops, clock_ns

    @contextmanager
    def request(self, operation, request_id):
        # A nested actual request gets an independent scope; no receipts from
        # its parent authorize a changed owner or a later outer operation.
        try:
            path = _Path(self, operation, request_id)
            lease = self.ops.reserve_metadata(16384)
        except Exception:
            token = _CURRENT.set(None)
            try: yield
            finally: _CURRENT.reset(token)
            return
        token = _CURRENT.set(path)
        try: yield
        except BaseException:
            path.failed = True
            raise
        finally:
            path.closed = True
            _CURRENT.reset(token)
            record = {"format": "omni-owned-fixed-memory-path", "version": 1,
                "requestSha256": path.request_sha, "ownerSessionId": path.binding["ownerSessionId"],
                "operation": operation, "effectsObserved": path.effects, "effectHeadSha256": path.head,
                "fixedMemoryNanoseconds": str(path.fixed_ns), "elapsedNanoseconds": str(max(0, self.clock()-path.started)),
                "priorAsyncDrainExcludedNanoseconds": str(path.excluded_drain_ns),
                "phases": path.phases, "completed": not path.failed, "metadataIssue": path.disabled,
                "actualPreparedOperations": path.consumed,
                "replyExecution": getattr(path, "reply", None),
                "memoryPlacement": "actual-original-owners", "learningEffectsReplayed": False,
                "wholeReplyPlacementQualified": False, "wholeReplySpeedupMeasured": False}
            try: self.ops.record(record)
            except Exception: pass  # Diagnostics cannot retry/fail committed learning.
            finally: lease.release()

    @contextmanager
    def effect(self, path, stage, *, admit=True):
        if path.closed or PRIVATE_OPERATION.get(): raise RuntimeError("retired/private effect owner")
        # Admission happens at the real pre-mutation boundary. Auto Stop does
        # not cancel a genuine update; the original request's Stop remains its
        # existing controller's responsibility.
        outer = path.depth == 0
        path.depth += 1
        before, start, reading, stack = None, self.clock(), None, ExitStack()
        if outer:
            try:
                before = self.ops.frontier()
                reading = stack.enter_context(self.ops.resource_observation())
            except Exception as error: path.disabled = "memory-observation:" + type(error).__name__
            drain_started = self.clock()
            try: self.ops.drain_prior_effect(stage)
            except Exception as error: path.disabled = "memory-prior-drain:" + type(error).__name__
            path.excluded_drain_ns += max(0, self.clock()-drain_started)
            start = self.clock()
        try:
            if outer and reading is not None:
                try: reading()
                except Exception as error: path.disabled = "memory-resource-start:" + type(error).__name__
            try:
                if outer and admit: self.ops.admit_effect(stage)
                yield
            except BaseException:
                path.failed = True
                raise
            finally:
                if outer:
                    try: self.ops.complete_effect(stage)
                    except Exception as error: path.disabled = "memory-completion:" + type(error).__name__
                elapsed = max(0, self.clock()-start)
                phase = path.phases.setdefault(stage, {"calls": 0, "inclusiveNanoseconds": "0"})
                phase["calls"] += 1
                phase["inclusiveNanoseconds"] = str(int(phase["inclusiveNanoseconds"])+elapsed)
                if outer:
                    path.fixed_ns += elapsed
                    path.effects += 1
                    try:
                        after = self.ops.frontier()
                        resources = None if reading is None else reading()
                        path.resources = _resources(path.resources, resources)
                        path.frontier = after
                        path.head = digest([path.head, stage, elapsed, before, after, resources, path.failed])
                    except Exception as error: path.disabled = "memory-resource-finish:" + type(error).__name__
        finally:
            path.depth -= 1
            try: stack.close()
            except Exception as error: path.disabled = "memory-observation-release:" + type(error).__name__

    def select(self, receipts, *, selector, identity, current_candidate, verify_receipt, minimum_samples):
        path = _CURRENT.get()
        ordinary = dict(identity=identity, current_candidate=current_candidate,
            verify_receipt=verify_receipt, minimum_samples=minimum_samples)
        if path is None or path.scheduler is not self or not path.effects:
            return selector(receipts, **ordinary)
        try:
            if (path.closed or path.depth or path.failed or path.disabled or not self.ops.armed()
                    or self.ops.binding() != path.binding or self.ops.frontier() != path.frontier):
                raise ValueError("fixed memory prefix is incomplete, changed or retired")
            self.ops.admit_selection(identity, current_candidate)
            prefix_sha = digest([path.request_sha, path.head, path.fixed_ns, path.frontier, path.resources])
            combined = copy.deepcopy(identity)
            combined["stageCoverage"] = ["actual-fixed-owner-memory-prefix", *identity["stageCoverage"]]
            combined["workloadSha256"] = digest([identity["workloadSha256"], prefix_sha])
            issued = {}
            def derived():
                for original in receipts:
                    raw = {key: value for key, value in original.items() if key != "receiptSha256"}
                    if not verify_receipt(original) or digest(raw) != original.get("receiptSha256"):
                        continue
                    if original.get("identity") != identity: continue
                    lease = None
                    try:
                        lease = self.ops.reserve_metadata(len(json.dumps(original, allow_nan=False))*6+16384)
                        row = copy.deepcopy(original)
                        resources = _resources(path.resources, row["resourceAfter"])
                        elapsed = int(row["totalNanoseconds"])
                        if elapsed <= 0 or str(elapsed) != row["totalNanoseconds"]: continue
                        self.ops.admit_selection(identity, row["candidate"])
                        row.update(identity=combined, comparisonKeySha256=comparison_key(combined),
                            totalNanoseconds=str(elapsed+path.fixed_ns), resourceAfter=resources,
                            completedStages=list(combined["stageCoverage"]),
                            originalCompleteOperationReceiptSha256=original["receiptSha256"],
                            fixedMemoryPrefixSha256=prefix_sha, fixedMemoryReplayed=False,
                            latencyBasis="one-observed-fixed-prefix-plus-matched-complete-operation-samples",
                            scope="conditional-additive-fixed-owner-prefix-and-complete-compute-not-a-whole-reply-benchmark")
                        row["receiptSha256"] = digest({key: value for key, value in row.items() if key != "receiptSha256"})
                        # Stream custody, not an unbounded retained row/output list.
                        issued[row["observerId"]] = (row["observationSequence"], row["receiptSha256"])
                        yield row
                    except Exception: continue
                    finally:
                        if lease is not None: lease.release()
            def trusted(row):
                return issued.get(row.get("observerId")) == (row.get("observationSequence"), row.get("receiptSha256"))
            stream = derived()
            try:
                result = selector(stream, identity=combined, current_candidate=current_candidate,
                    verify_receipt=trusted, minimum_samples=minimum_samples)
            finally: stream.close()
            if (self.ops.frontier() != path.frontier or self.ops.binding() != path.binding or not self.ops.armed()):
                raise ValueError("fixed memory prefix changed before aggregate selection")
            result["aggregateSchedule"] = {"fixedMemoryPrefixSha256": prefix_sha,
                "fixedMemoryNanoseconds": str(path.fixed_ns), "fixedEffectsObserved": path.effects,
                "physicalGroupsCountedOnce": True, "fixedMemoryPlacement": "actual-original-owners",
                "computeWorkKind": identity["workKind"], "computeComparisonKeySha256": comparison_key(identity),
                "transferCosts": "included-in-actual-complete-operation-and-fixed-memory-timings",
                "latencyBasis": "one-observed-fixed-prefix-plus-matched-complete-operation-samples",
                "wholeReplySpeedupMeasured": False, "learningEffectsReplayed": False}
            result["proposalSha256"] = digest({key: value for key, value in result.items() if key != "proposalSha256"})
            return result
        except Exception as error:
            # Do not run another selector over a partially consumed iterator,
            # move a mutable memory owner, or repeat the real learning body.
            result = selector((), **ordinary)
            result["aggregateScheduleRefused"] = type(error).__name__ + ":" + str(error)
            result["proposalSha256"] = digest({key: value for key, value in result.items() if key != "proposalSha256"})
            return result
