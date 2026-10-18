"""Actual outer learning phases and canonical complete-operation consumption.

No neural dependency or execution on import. Mutable modality/replay/evolution
and publication phases stay with their real owners. A selected complete wave
is associated only while its existing genuine executor is running; private
comparisons, imported records and diagnostics never execute a phase or commit.
"""
from __future__ import annotations

import functools
import copy
import json
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar

from .operation_auto_registry import PRIVATE_OPERATION, STAGE_COVERAGE
from .whole_path_execution import digest
from .whole_path_scheduler import (
    current_operation_choice, current_owned_path, owned_phase,
    world_effect_barrier,
)

_PIPELINE = ContextVar("omni_actual_outer_learning_pipeline", default=None)
_PINNED = {
    "media-objectives": "mutable-modality-optimizer-sensory-and-embedded-audio-closure",
    "slow-candidate": "candidate-validation-update-stability-and-promotion-closure",
    "replay-objectives": "mutable-replay-cortical-liquid-and-optimizer-closure",
    "queued-replay": "queued-evidence-update-growth-cursor-and-generation-closure",
    "latent-rehearsal": "actual-recurrent-replay-and-stability-owner",
    "evolution-replay": "candidate-state-validation-decay-and-promotion-closure",
    "evolution-proposal": "isolated-candidate-files-training-and-lineage-owner",
    "evolution-evaluation": "candidate-holdouts-and-immutable-evaluation-owner",
    "evolution-promotion": "atomic-candidate-promotion-and-rollback-owner",
    "evolution-rollback": "atomic-recorded-rollback-owner",
    "checkpoint-publication": "actual-generation-and-atomic-brain-json-owner",
    "source-cursor-publication": "same-generation-neural-state-and-source-cursor-owner",
    "replay-cursor-publication": "same-generation-replay-tombstone-and-neural-owner",
}
_PUBLICATIONS = {"checkpoint-publication", "source-cursor-publication", "replay-cursor-publication",
    "evolution-promotion", "evolution-rollback"}


def _assert_genuine():
    if PRIVATE_OPERATION.get():
        raise RuntimeError("private operation cannot execute an outer learning or publication phase")


def _cancel(brain):
    check = getattr(brain, "_learning_cancel_check", None)
    if check is not None and check():
        # The worker supplies its current cancellation exception factory.
        # No saved plan or imported row supplies executable control here.
        factory = getattr(brain, "_outer_learning_cancel_error", None)
        raise factory() if callable(factory) else InterruptedError("learning stopped at an actual phase boundary")


class _Pipeline:
    def __init__(self, brain, path, operation):
        self.brain, self.path, self.operation = brain, path, str(operation)
        self.clock = path.scheduler.clock
        self.started = self.clock()
        self.depth = self.ordinal = 0
        self.commits = 0
        self.request_commits = 0
        self.publication_owner = None
        self.fixed_ns = self.compute_ns = 0
        self.phases, self.routes = {}, {}
        self.head = "0" * 64
        self.issue = None
        self.closed = False
        self.lease = path.scheduler.ops.reserve_metadata(16384)
        self.row_leases, self.resource_leases = {}, {}

    def _current(self):
        return not self.closed and current_owned_path() is self.path

    def _observe(self, stage, owner, elapsed, failed, *, candidate=None, publication=False):
        # One row per declared phase/work kind, not per record/window/output.
        if stage not in _PINNED and stage not in STAGE_COVERAGE:
            raise ValueError("outer phase has no declared finite work kind")
        if stage not in self.row_leases:
            self.row_leases[stage] = self.path.scheduler.ops.reserve_metadata(4096)
        row = self.phases.setdefault(stage, {"entered": 0, "completed": 0, "failed": 0,
            "inclusiveNanoseconds": "0", "publication": publication, "requestOwnerCalls": 0})
        row["entered"] += 1
        row["completed"] += not failed
        row["failed"] += failed
        row["inclusiveNanoseconds"] = str(int(row["inclusiveNanoseconds"]) + elapsed)
        row["actualOwnerId"] = str(getattr(owner, "brain_id", "runtime-owner"))
        row["lastActualOwnerObject"] = str(id(owner))
        row["ownerIsRequestBrain"] = owner is self.brain
        row["requestOwnerCalls"] += owner is self.brain
        row["placement"] = "current-qualified-complete-operation" if candidate is not None else "actual-owner-pinned"
        if candidate is None: row["pinnedReason"] = _PINNED[stage]
        self.ordinal += 1
        self.head = digest([self.head, self.ordinal, stage, row["actualOwnerId"], elapsed,
            bool(failed), None if candidate is None else candidate["planSha256"]])

    @contextmanager
    def pinned(self, owner, stage):
        if stage not in _PINNED: raise ValueError("unknown outer learning phase")
        publication = stage in _PUBLICATIONS
        # Never add a late Stop/diagnostic gate in front of a checkpoint that
        # finishes an already applied canonical update. Existing transactions
        # retain their original validation, admission and commit semantics.
        if not publication: _cancel(owner)
        outer, started, failed = self.depth == 0, self.clock(), False
        previous_publisher = self.publication_owner
        if publication: self.publication_owner = owner
        self.depth += 1
        try:
            with world_effect_barrier(), owned_phase(owner, "learning." + stage, publication=publication):
                yield
        except BaseException:
            failed = True
            raise
        finally:
            self.depth -= 1
            self.publication_owner = previous_publisher
            elapsed = max(0, self.clock() - started)
            if outer: self.fixed_ns += elapsed
            try: self._observe(stage, owner, elapsed, failed, publication=publication)
            except Exception as error: self.issue = "phase-observation:" + type(error).__name__

    @contextmanager
    def genuine(self, brain, controller, identity, candidate):
        choice = current_operation_choice()
        try:
            owner_current = (self.path.scheduler.ops.binding() == self.path.binding
                and getattr(brain, "_native_operation_auto", None) is controller)
        except Exception:
            owner_current = False
        if (not self._current() or brain is not self.brain or choice is None
                or not owner_current
                or choice["controller"] is not controller or choice["identity"] is not identity
                or choice["candidate"] is not candidate or choice["path"] is not self.path):
            # The real consumer keeps all its own native checks. A stale or
            # absent runtime association cannot be recorded as selected work.
            self.issue = "canonical-operation-association-unavailable"
            yield
            return
        _cancel(brain)
        # This is an execution admission, not a proposal/receipt acknowledgement.
        # The caller enters its exact genuine placement and verifies the full
        # state/RNG/effect contract inside this context before it can complete.
        controller.ops.admit_candidate(candidate)
        stage = str(identity["workKind"])
        stack, reading = ExitStack(), None
        try:
            try:
                reading = stack.enter_context(controller.ops.observation_resources(candidate, identity["budgetSha256"]))
                reading()
            except Exception as error: self.issue = "canonical-resource-observation:" + type(error).__name__
            started, failed = self.clock(), False
            try:
                yield
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = max(0, self.clock() - started)
                self.compute_ns += elapsed
                try:
                    self._observe(stage, brain, elapsed, failed, candidate=candidate)
                    row = self.routes.setdefault(stage, {"entered": 0, "completed": 0,
                        "failed": 0, "nanoseconds": "0"})
                    row["entered"] += 1; row["completed"] += not failed; row["failed"] += failed
                    row["nanoseconds"] = str(int(row["nanoseconds"]) + elapsed)
                    row["lastActualPlanSha256"] = candidate["planSha256"]
                    row["lastExecutionDevice"] = candidate["executionDevice"]
                    row["basis"] = "existing-authenticated-complete-operation-selection-and-genuine-consumer"
                    row["checkpointOrCursorReplayed"] = False
                    if reading is not None:
                        resource = reading()
                        groups = resource["physicalGroups"]
                        lease = self.path.scheduler.ops.reserve_metadata(len(json.dumps(groups, allow_nan=False))*4+1024)
                        try: snapshot = copy.deepcopy(groups)
                        except BaseException:
                            lease.release()
                            raise
                        previous = self.resource_leases.get(stage)
                        self.resource_leases[stage] = lease
                        row["lastPhysicalGroups"] = snapshot
                        if previous is not None: previous.release()
                        row["resourceAdmissionComplete"] = bool(resource["admissionComplete"])
                except Exception as error: self.issue = "canonical-observation:" + type(error).__name__
        finally:
            try: stack.close()
            except Exception as error: self.issue = "canonical-observation-release:" + type(error).__name__

    def close(self, failed):
        self.closed = True
        try:
            payload = {"format": "omni-actual-outer-learning-pipeline", "version": 1,
                "requestSha256": self.path.request_sha, "ownerSessionId": self.path.binding["ownerSessionId"],
                "operation": self.operation, "phaseCalls": self.ordinal, "phaseHeadSha256": self.head,
                "atomicGenerationPublicationsObserved": self.commits,
                "requestOwnerAtomicGenerationPublicationsObserved": self.request_commits,
                "phases": self.phases, "actualCompleteOperationRoutes": self.routes,
                "fixedOwnerNanoseconds": str(self.fixed_ns), "canonicalComputeNanoseconds": str(self.compute_ns),
                "elapsedNanoseconds": str(max(0, self.clock() - self.started)), "completed": not failed,
                "metadataIssue": self.issue, "publicationAuthority": "original-atomic-generation-and-source-cursor",
                "learningEffectsReplayed": False, "wholePipelinePrivateQualification": False,
                "wholePipelineSpeedupMeasured": False}
            self.path.scheduler.ops.record(payload)
        except Exception:
            pass  # An observation cannot roll back or replay a committed update.
        finally:
            for lease in [*self.resource_leases.values(), *self.row_leases.values(), self.lease]:
                try: lease.release()
                except Exception: pass


@contextmanager
def learning_pipeline(brain, operation, *, cancelled=None, cancel_error=None):
    """Bind real cancellation and one outer executor to the current request.

    No saved intent re-arms Auto. Unarmed callers retain ordinary execution and
    their current cancellation boundary without writing a pipeline receipt.
    """
    _assert_genuine()
    previous_present = "_learning_cancel_check" in vars(brain)
    error_present = "_outer_learning_cancel_error" in vars(brain)
    previous = getattr(brain, "_learning_cancel_check", None)
    previous_error = getattr(brain, "_outer_learning_cancel_error", None)
    combined = (lambda: bool(cancelled is not None and cancelled()) or bool(previous is not None and previous()))
    brain._learning_cancel_check = combined
    brain._outer_learning_cancel_error = cancel_error or previous_error
    active, token, created, failed = _PIPELINE.get(), None, None, False
    try:
        path = current_owned_path()
        if active is not None and active._current():
            # An isolated candidate remains a distinct actual owner. Its
            # phases may be accounted in this request, never inherit the
            # parent's selected-compute authority or owner identity.
            yield active
            return
        if path is not None:
            try:
                if path.scheduler.ops.brain is brain and path.scheduler.ops.binding() == path.binding:
                    created = _Pipeline(brain, path, operation)
            except Exception:
                created = None
        token = _PIPELINE.set(created)
        try: yield created
        except BaseException:
            failed = True
            raise
    finally:
        if token is not None: _PIPELINE.reset(token)
        if created is not None: created.close(failed)
        if getattr(brain, "_learning_cancel_check", None) is combined:
            if previous_present: brain._learning_cancel_check = previous
            else: vars(brain).pop("_learning_cancel_check", None)
            if error_present: brain._outer_learning_cancel_error = previous_error
            else: vars(brain).pop("_outer_learning_cancel_error", None)


@contextmanager
def learning_phase(owner, stage):
    _assert_genuine()
    if stage not in _PINNED: raise ValueError("unknown outer learning phase")
    active = _PIPELINE.get()
    if active is None or not active._current():
        # Direct brain calls still bind to a real current request when present.
        with learning_pipeline(owner, stage) as active:
            if active is None:
                with world_effect_barrier(): yield
            else:
                with active.pinned(owner, stage): yield
    else:
        with active.pinned(owner, stage): yield


def outer_learning_phase(stage):
    """The body is the genuine consumer; it is called exactly once."""
    if stage not in _PINNED: raise ValueError("unknown outer learning phase")
    def decorate(method):
        @functools.wraps(method)
        def wrapped(owner, *args, **kwargs):
            with learning_phase(owner, stage): return method(owner, *args, **kwargs)
        return wrapped
    return decorate


@contextmanager
def genuine_learning_operation(brain, controller, identity, candidate):
    _assert_genuine()
    active = _PIPELINE.get()
    if active is None or not active._current():
        with learning_pipeline(brain, "prepared-wave") as active:
            if active is None: yield
            else:
                with active.genuine(brain, controller, identity, candidate): yield
    else:
        with active.genuine(brain, controller, identity, candidate): yield


def note_learning_commit(brain):
    """Called only by the existing atomic generation commit callback.

    This observes the real publication, including attached replay/source
    cursors. It never validates, retries, authorizes or delays that publication.
    """
    try:
        active = _PIPELINE.get()
        if active is not None and active._current() and active.publication_owner is brain:
            active.commits += 1
            active.request_commits += brain is active.brain
    except Exception:
        pass
