"""NEW outer learning custody tests: inert clocks and tiny real SQLite only.

No neural import, model construction, Load, forward/backward, training,
hardware benchmark, app launch or old closed test case executes here.
"""
from __future__ import annotations

import ast
import copy
import importlib
import sqlite3
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "owned_outer_learning_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "omni_core")]
sys.modules[PACKAGE] = package
S = importlib.import_module(PACKAGE + ".whole_path_scheduler")
P = importlib.import_module(PACKAGE + ".outer_learning_pipeline")
R = importlib.import_module(PACKAGE + ".operation_auto_registry")
W = importlib.import_module(PACKAGE + ".whole_path_execution")


class Stopped(Exception):
    pass


class Clock:
    def __init__(self): self.now = 0
    def __call__(self): return self.now


class Lease:
    def __init__(self, size): self.size, self.closed, self.fail = size, False, False
    def release(self):
        self.closed = True
        if self.fail: raise RuntimeError("inert metadata release failed")


class ActualOps:
    def __init__(self):
        self.clock, self.brain = Clock(), types.SimpleNamespace(brain_id="actual")
        self.current = {"ownerSessionId": "s", "policySha256": "1"*64, "budgetSha256": "4"*64}
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.execute("CREATE TABLE effects(stage TEXT)")
        self.rows, self.leases, self.admissions, self.compute_admissions = [], [], [], []
        self.fail_admission = self.fail_record = self.fail_sampler = self.fail_sampler_release = False
        self.fail_metadata_at = None
        self.controller = types.SimpleNamespace(ops=self)
        self.brain._native_operation_auto = self.controller
        self.whole_path_scheduler = S.WholePathScheduler(self, clock_ns=self.clock)
        self.candidate = {"executionDevice": "cpu", "backendMode": "native-cpu", "asyncDevices": [],
            "supported": True, "fallbackAcknowledged": False, "planSha256": "a"*64}
        self.identity = {"ownerId": "actual", "ownerSessionId": "s", "policySha256": "1"*64,
            "budgetSha256": "4"*64, "workKind": R.TRAINING_KINDS[0]}
    def armed(self): return True
    def binding(self): return copy.deepcopy(self.current)
    def owns(self, owner): return owner is self.brain
    def frontier(self): return {"actualSQLiteChanges": self.db.total_changes}
    def reserve_metadata(self, size):
        if self.fail_metadata_at == len(self.leases) + 1: raise RuntimeError("inert metadata pressure")
        lease = Lease(size); self.leases.append(lease); return lease
    def admit_effect(self, stage):
        self.admissions.append(stage)
        if self.fail_admission: raise RuntimeError("real phase admission denied")
    def admit_candidate(self, candidate):
        if self.fail_admission: raise RuntimeError("real compute admission denied")
        self.compute_admissions.append(candidate)
    def admit_selection(self, identity, candidate):
        if (identity["ownerId"] != self.brain.brain_id or identity["ownerSessionId"] != self.current["ownerSessionId"]
                or identity["policySha256"] != self.current["policySha256"]
                or identity["budgetSha256"] != self.current["budgetSha256"]):
            raise RuntimeError("actual current binding changed")
    def complete_effect(self, _stage): pass
    def drain_prior_effect(self, _stage): pass
    @contextmanager
    def observation_resources(self, candidate, sha):
        if self.fail_sampler: raise RuntimeError("inert sampler failed")
        first = [True]
        def read():
            row = {"budgetSha256": sha, "admissionComplete": True, "pressureObserved": False,
                "physicalGroups": {"ram:system": {"ceilingBytes": 1024, "observedPeakBytes": 64,
                    "peakObservation": "operation-start" if first[0] else "sampled-full-operation"}}}
            first[0] = False
            return row
        try: yield read
        finally:
            if self.fail_sampler_release: raise RuntimeError("inert sampler release failed")
    def resource_observation(self): return self.observation_resources(self.candidate, "4"*64)
    def record(self, row):
        if self.fail_record: raise RuntimeError("inert journal failed")
        self.rows.append(copy.deepcopy(row))
    def write(self, stage, elapsed=1):
        self.db.execute("INSERT INTO effects VALUES(?)", (stage,)); self.clock.now += elapsed
    def summary(self):
        return next(row for row in reversed(self.rows) if row["format"] == "omni-actual-outer-learning-pipeline")
    def values(self): return [value for value, in self.db.execute("SELECT stage FROM effects ORDER BY rowid")]


class OuterLearningPipelineCases(unittest.TestCase):
    def fixture(self):
        ops = ActualOps(); self.addCleanup(ops.db.close); return ops

    def test_actual_nested_media_replay_and_atomic_cursor_publish_once_on_original_owner(self):
        f = self.fixture()
        with f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
            with P.learning_phase(f.brain, "media-objectives"):
                self.assertFalse(S.qualification_allowed())
                f.write("image-window", 50)
                with P.learning_phase(f.brain, "replay-objectives"): f.write("replay", 20)
                with P.learning_phase(f.brain, "checkpoint-publication"):
                    f.write("generation-and-source-cursor", 7)
                    P.note_learning_commit(f.brain)
        row = f.summary()
        self.assertEqual(f.values(), ["image-window", "replay", "generation-and-source-cursor"])
        self.assertEqual(row["fixedOwnerNanoseconds"], "77")
        self.assertEqual(row["phases"]["replay-objectives"]["inclusiveNanoseconds"], "20")
        self.assertEqual(row["atomicGenerationPublicationsObserved"], 1)
        self.assertEqual(f.rows[-1]["fixedMemoryNanoseconds"], "77")
        self.assertEqual(f.admissions, ["learning.media-objectives"])
        self.assertFalse(row["learningEffectsReplayed"])
        self.assertFalse(row["wholePipelinePrivateQualification"])
        self.assertTrue(all(lease.closed for lease in f.leases))

    def test_real_selected_genuine_wave_uses_admission_not_a_receipt_and_commits_cursor_once(self):
        f = self.fixture()
        with f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
            with P.learning_phase(f.brain, "latent-rehearsal"): f.write("fixed-prefix", 5)
            with S.owned_operation_choice(f.controller, f.identity, f.candidate), \
                    P.genuine_learning_operation(f.brain, f.controller, f.identity, f.candidate):
                f.write("actual-native-update-and-verification", 30)
            with P.learning_phase(f.brain, "checkpoint-publication"):
                f.write("same-generation-source-cursor", 3)
                P.note_learning_commit(f.brain)
        row = f.summary(); route = row["actualCompleteOperationRoutes"][R.TRAINING_KINDS[0]]
        self.assertEqual(f.compute_admissions, [f.candidate])
        self.assertEqual(route["entered"], 1); self.assertEqual(route["completed"], 1)
        self.assertEqual(route["lastActualPlanSha256"], "a"*64)
        self.assertEqual(route["lastPhysicalGroups"]["ram:system"]["observedPeakBytes"], 64)
        self.assertEqual(row["canonicalComputeNanoseconds"], "30")
        self.assertEqual(row["fixedOwnerNanoseconds"], "8")
        self.assertEqual(row["atomicGenerationPublicationsObserved"], 1)
        self.assertFalse(route["checkpointOrCursorReplayed"])
        self.assertEqual(len(f.values()), 3)

    def test_new_outer_prefix_is_consumed_by_actual_aggregate_selector_and_genuine_consumer(self):
        f = self.fixture()
        identity = {**f.identity, "stageCoverage": list(R.TRAINING_STAGES), "workloadSha256": "5"*64,
            "geometrySha256": "6"*64, "samplingContractSha256": "7"*64, "topologySha256": "8"*64,
            "cache": {"heat": "warm", "logicalStateSha256": "9"*64, "backendInitialization": "warm"},
            "nativeState": {"kind": "live-owner-epochs", "bindingSha256": "0"*64,
                "ownerSessionId": "s", "complete": True, "ownerVersions": {"inert-current-owner": 1}}}
        other = {**f.candidate, "executionDevice": "mps", "backendMode": "native-metal",
            "asyncDevices": ["mps"], "planSha256": "b"*64}
        def primitive(candidate):
            return {key: value for key, value in candidate.items() if key != "executionDevice"}
        recorder = W.WholePathRecorder(clock_ns=f.clock)
        rows = []
        with f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
            with P.learning_phase(f.brain, "latent-rehearsal"): f.write("genuine-fixed-prefix", 15)
            # Inert complete-operation observations compare only declared pure
            # compute. The already-applied SQL prefix is never run in a trial.
            for _ in range(3):
                for candidate in (f.candidate, other):
                    with f.observation_resources(candidate, identity["budgetSha256"]) as resources:
                        observation = recorder.begin(identity, primitive(candidate), synchronize=lambda devices: list(devices),
                            counters=lambda: dict.fromkeys(W.COUNTERS, 0), resources=resources)
                        f.clock.now += 100 if candidate is f.candidate else 10
                        for stage in R.TRAINING_STAGES: observation.stage_completed(stage)
                        rows.append(observation.finish(native_state_after=identity["nativeState"], output_contract_sha256="c"*64))
            selected = S.select_complete_operation(f.controller, iter(rows), selector=W.select_whole_path_candidate,
                identity=identity, current_candidate=primitive(f.candidate), minimum_samples=3,
                verify_receipt=lambda row: row["observerId"] == recorder.observer_id)
            self.assertEqual(selected["decision"], "propose-at-quiescence")
            self.assertEqual(selected["aggregateSchedule"]["fixedMemoryNanoseconds"], "15")
            self.assertEqual(selected["proposedCandidate"], primitive(other))
            with S.owned_operation_choice(f.controller, identity, other), \
                    P.genuine_learning_operation(f.brain, f.controller, identity, other):
                f.write("genuine-selected-update", 10)
            with P.learning_phase(f.brain, "checkpoint-publication"):
                f.write("real-cursor-and-generation"); P.note_learning_commit(f.brain)
        self.assertEqual(f.values(), ["genuine-fixed-prefix", "genuine-selected-update", "real-cursor-and-generation"])
        self.assertEqual(f.compute_admissions, [other])
        self.assertEqual(f.summary()["actualCompleteOperationRoutes"][R.TRAINING_KINDS[0]]["lastExecutionDevice"], "mps")

    def test_forged_copied_stale_or_foreign_choice_cannot_claim_selected_execution(self):
        for changed in ("copied-candidate", "copied-identity", "no-choice", "foreign-controller", "changed-policy"):
            with self.subTest(changed=changed):
                f = self.fixture()
                with f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
                    with S.owned_operation_choice(f.controller, f.identity, f.candidate):
                        controller, identity, candidate = f.controller, f.identity, f.candidate
                        if changed == "copied-candidate": candidate = copy.deepcopy(candidate)
                        if changed == "copied-identity": identity = copy.deepcopy(identity)
                        if changed == "foreign-controller": controller = types.SimpleNamespace(ops=f)
                        if changed == "changed-policy": f.current["policySha256"] = "2"*64
                        token = S._CHOICE.set(None) if changed == "no-choice" else None
                        try:
                            with P.genuine_learning_operation(f.brain, controller, identity, candidate): f.write("ordinary-body-once")
                        finally:
                            if token is not None: S._CHOICE.reset(token)
                self.assertEqual(f.values(), ["ordinary-body-once"])
                self.assertEqual(f.compute_admissions, [])
                self.assertEqual(f.summary()["actualCompleteOperationRoutes"], {})

    def test_real_stop_retains_applied_update_publication_then_stops_before_next_compute(self):
        f = self.fixture(); stop = [False]
        prior = lambda: False
        f.brain._learning_cancel_check = prior
        with self.assertRaises(Stopped), f.whole_path_scheduler.request("ingest", "request"), \
                P.learning_pipeline(f.brain, "ingest", cancelled=lambda: stop[0], cancel_error=Stopped):
            with P.learning_phase(f.brain, "replay-objectives"):
                f.write("applied-update"); stop[0] = True
            with P.learning_phase(f.brain, "checkpoint-publication"):
                f.write("durable-generation-and-cursor"); P.note_learning_commit(f.brain)
            with P.learning_phase(f.brain, "media-objectives"): f.write("must-not-run")
        self.assertEqual(f.values(), ["applied-update", "durable-generation-and-cursor"])
        self.assertEqual(f.summary()["atomicGenerationPublicationsObserved"], 1)
        self.assertFalse(f.summary()["completed"])
        self.assertIs(f.brain._learning_cancel_check, prior)
        self.assertNotIn("_outer_learning_cancel_error", vars(f.brain))

    def test_actual_admission_can_stop_compute_but_new_publication_admission_cannot_block_commit(self):
        f = self.fixture(); f.fail_admission = True
        with f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
            with self.assertRaisesRegex(RuntimeError, "real phase admission"):
                with P.learning_phase(f.brain, "replay-objectives"): f.write("must-not-run")
            with P.learning_phase(f.brain, "checkpoint-publication"):
                f.write("original-publication"); P.note_learning_commit(f.brain)
        self.assertEqual(f.values(), ["original-publication"])
        self.assertEqual(f.summary()["atomicGenerationPublicationsObserved"], 1)
        other = self.fixture(); other.fail_admission = True
        with self.assertRaisesRegex(RuntimeError, "real compute admission"), \
                other.whole_path_scheduler.request("ingest", "second"), P.learning_pipeline(other.brain, "ingest"), \
                S.owned_operation_choice(other.controller, other.identity, other.candidate), \
                P.genuine_learning_operation(other.brain, other.controller, other.identity, other.candidate):
            other.write("must-not-run")
        self.assertEqual(other.values(), [])

    def test_private_mutation_and_publication_refused_before_any_SQL_effect(self):
        f = self.fixture(); token = R.PRIVATE_OPERATION.set(True)
        try:
            for context in (P.learning_pipeline(f.brain, "ingest"),
                    P.learning_phase(f.brain, "checkpoint-publication"),
                    P.genuine_learning_operation(f.brain, f.controller, f.identity, f.candidate)):
                with self.assertRaisesRegex(RuntimeError, "private operation"):
                    with context: f.write("must-not-run")
        finally: R.PRIVATE_OPERATION.reset(token)
        self.assertEqual(f.values(), [])

    def test_diagnostic_failure_never_replays_or_invalidates_real_learning_or_cursor(self):
        for failure in ("sampler", "sampler-release", "record", "phase-metadata", "outer-release"):
            with self.subTest(failure=failure):
                f = self.fixture()
                f.fail_sampler = failure == "sampler"
                f.fail_sampler_release = failure == "sampler-release"
                f.fail_record = failure == "record"
                if failure == "phase-metadata": f.fail_metadata_at = 3
                with f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
                    if failure == "outer-release": f.leases[1].fail = True
                    with P.learning_phase(f.brain, "replay-objectives"): f.write("applied-once")
                    with P.learning_phase(f.brain, "checkpoint-publication"):
                        f.write("cursor-once"); P.note_learning_commit(f.brain)
                self.assertEqual(f.values(), ["applied-once", "cursor-once"])
                self.assertTrue(all(lease.closed for lease in f.leases))

    def test_unarmed_scope_and_foreign_candidate_keep_normal_body_without_imported_authority(self):
        f = self.fixture()
        with P.learning_pipeline(f.brain, "unarmed") as pipeline:
            self.assertIsNone(pipeline)
            with P.learning_phase(f.brain, "media-objectives"): f.write("ordinary")
        self.assertEqual(f.rows, [])
        self.assertNotIn("_learning_cancel_check", vars(f.brain))
        foreign = types.SimpleNamespace(brain_id="actual")
        with f.whole_path_scheduler.request("evolution.propose", "request"), P.learning_pipeline(f.brain, "evolution"):
            with P.learning_phase(foreign, "slow-candidate"): f.write("isolated-candidate")
            P.note_learning_commit(foreign)  # No actual publication scope: not an observed commit.
            with P.learning_phase(foreign, "checkpoint-publication"):
                f.write("isolated-candidate-generation"); P.note_learning_commit(foreign)
        row = f.summary()["phases"]["slow-candidate"]
        self.assertFalse(row["ownerIsRequestBrain"])
        self.assertEqual(row["lastActualOwnerObject"], str(id(foreign)))
        self.assertEqual(row["requestOwnerCalls"], 0)
        self.assertEqual(f.summary()["atomicGenerationPublicationsObserved"], 1)
        self.assertEqual(f.summary()["requestOwnerAtomicGenerationPublicationsObserved"], 0)

    def test_isolated_candidate_inherits_only_real_job_stop_not_parent_compute_authority(self):
        f = self.fixture(); child = types.SimpleNamespace(brain_id="actual"); stop = [False]
        with self.assertRaises(Stopped), f.whole_path_scheduler.request("evolution.propose", "request"), \
                P.learning_pipeline(f.brain, "evolution", cancelled=lambda: stop[0], cancel_error=Stopped):
            with P.learning_pipeline(child, "isolated-evolution-training", cancelled=lambda: stop[0], cancel_error=Stopped):
                with P.learning_phase(child, "slow-candidate"): f.write("candidate-once")
                stop[0] = True
                with P.learning_phase(child, "slow-candidate"): f.write("must-not-run")
        self.assertEqual(f.values(), ["candidate-once"])
        self.assertEqual(f.summary()["phases"]["slow-candidate"]["lastActualOwnerObject"], str(id(child)))
        self.assertEqual(f.summary()["actualCompleteOperationRoutes"], {})
        self.assertNotIn("_learning_cancel_check", vars(child))

    def test_real_body_failure_propagates_once_without_recorded_success(self):
        f = self.fixture()
        with self.assertRaisesRegex(ValueError, "actual failure"), \
                f.whole_path_scheduler.request("ingest", "request"), P.learning_pipeline(f.brain, "ingest"):
            with P.learning_phase(f.brain, "media-objectives"):
                f.write("actual-once"); raise ValueError("actual failure")
        self.assertEqual(f.values(), ["actual-once"])
        self.assertEqual(f.summary()["phases"]["media-objectives"]["failed"], 1)
        self.assertFalse(f.summary()["completed"])

    def test_actual_worker_scope_helper_uses_current_request_and_job_stop_without_loading(self):
        tree = ast.parse((ROOT / "worker.py").read_text())
        methods = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            methods["_request_cancel_check"], methods["_outer_learning_scope"]], type_ignores=[])
        class Fault(Exception):
            def __init__(self, code, message, data): self.code, self.data = code, data; super().__init__(message)
        namespace = {"learning_pipeline": P.learning_pipeline, "RpcFault": Fault}
        exec(compile(ast.fix_missing_locations(module), "actual-worker-learning-scope", "exec"), namespace)
        worker = types.SimpleNamespace(_active_request_lock=threading.Lock(), _active_request=("ingest", "current"),
            _cooperative_cancel=threading.Event(), cancelled_jobs=set())
        for name in ("_request_cancel_check", "_outer_learning_scope"):
            setattr(worker, name, types.MethodType(namespace[name], worker))
        f = self.fixture(); worker._cooperative_cancel.set()
        with worker._outer_learning_scope(f.brain, {"jobId": "job"}, "older", "ingest"):
            self.assertFalse(f.brain._learning_cancel_check())
        with self.assertRaises(Fault) as stopped:
            worker._outer_learning_scope(f.brain, {"jobId": "job"}, "current", "ingest")
        self.assertEqual(stopped.exception.code, -32800)
        self.assertTrue(stopped.exception.data["safeBoundary"])
        worker._cooperative_cancel.clear()
        with worker._outer_learning_scope(f.brain, {"jobId": "job"}, "current", "ingest"):
            worker.cancelled_jobs.add("job")
            with self.assertRaises(Fault): P._cancel(f.brain)
        self.assertEqual(f.values(), [])
        self.assertNotIn("_learning_cancel_check", vars(f.brain))

    def test_source_hooks_cover_real_objectives_commit_and_request_cancel_not_status_or_new_Load(self):
        brain = ast.parse((ROOT / "omni_core/brain.py").read_text())
        worker = ast.parse((ROOT / "worker.py").read_text())
        training = ast.parse((ROOT / "omni_core/training_whole_operation.py").read_text())
        evolution = ast.parse((ROOT / "omni_core/evolution.py").read_text())
        methods = {item.name: item for item in ast.walk(brain) if isinstance(item, ast.FunctionDef)}
        expected = {"_train_media": "media-objectives", "train": "slow-candidate",
            "_optimize_experience": "replay-objectives", "_latent_rehearsal_step": "latent-rehearsal",
            "consolidate_pending_chat_learning": "queued-replay", "_train_evolution_replay_candidate": "evolution-replay",
            "save": "checkpoint-publication"}
        for name, stage in expected.items():
            self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "outer_learning_phase" and node.args[0].value == stage
                for node in methods[name].decorator_list), name)
        commit = next(node for node in ast.walk(methods["save"]) if isinstance(node, ast.FunctionDef) and node.name == "mark_committed")
        self.assertEqual(ast.unparse(commit.body[1]), "committed = True")
        self.assertEqual(ast.unparse(commit.body[2]), "note_learning_commit(self)")
        consume = next(node for node in training.body if isinstance(node, ast.FunctionDef) and node.name == "consume_streaming_training_wave")
        self.assertEqual(ast.unparse(consume.body[0].test), "not qualification_allowed()")
        genuine = next(node for node in ast.walk(training) if isinstance(node, ast.With)
            and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "genuine_learning_operation" for item in node.items))
        names = [item.context_expr.func.id for item in genuine.items if isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)]
        self.assertLess(names.index("owned_operation_choice"), names.index("genuine_learning_operation"))
        self.assertIn("ops.genuine_placement(selected, closure, GenuineFlags())", ast.unparse(genuine))
        actual = {node.name: node for node in ast.walk(worker) if isinstance(node, ast.FunctionDef)}
        work = {"train", "ingest", "checkpoint", "snapshot", "evolution.propose", "evolution_propose",
            "evolution.evaluate", "evolution_evaluate", "evolution.promote", "evolution_promote", "evolution.rollback", "evolution_rollback"}
        for name in ("dispatch", "request_cooperative_cancel"):
            declared = {node.value for node in ast.walk(actual[name]) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
            self.assertTrue(work <= declared, name)
        for name in ("train", "ingest", "checkpoint", "snapshot", "consolidate_chat_learning",
                "learn_action_result", "learn_capability_pack", "learn_tool_route_outcome",
                "evolution_propose", "evolution_evaluate", "evolution_promote", "evolution_rollback"):
            self.assertIn("self._outer_learning_scope(", ast.unparse(actual[name]), name)
        helper = ast.unparse(actual["_outer_learning_scope"])
        self.assertNotIn("_get(", helper); self.assertNotIn("load(", helper)
        self.assertIn("job_id in self.cancelled_jobs", helper)
        self.assertIn("_request_cancel_check(request_id)", helper)
        self.assertTrue(any(isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name)
            and node.type.id == "ChatGenerationCancelled" for node in ast.walk(actual["dispatch"])))
        propose = next(node for node in ast.walk(evolution) if isinstance(node, ast.FunctionDef) and node.name == "propose")
        candidate_scopes = [node for node in ast.walk(propose) if isinstance(node, ast.With)
            and any(isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "learning_pipeline" for item in node.items)]
        self.assertEqual(len(candidate_scopes), 2)
        self.assertTrue(all("cancelled=cancel_check" in ast.unparse(node) for node in candidate_scopes))
        self.assertIn("cancel_check=brain._learning_cancel_check", ast.unparse(actual["evolution_propose"]))


if __name__ == "__main__": unittest.main()
