"""NEW prepared learner-wave source seams; no neural computation.

Only raw CPU uint8/control tensors, actual temporary SQLite/files, fake owners
and inert wave callbacks. No brain/model construction, Load, forward, backward,
training, kernel, device benchmark or performance/quality proof is performed.
"""
import ast
import hashlib
import importlib
import json
import sqlite3
import tempfile
import threading
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from test_operation_auto_boundaries import PACKAGE, ROOT, methods
from test_media_canvas_streaming import Policy

COW = importlib.import_module(PACKAGE+".training_cow")
AUTO = importlib.import_module(PACKAGE+".training_whole_operation")
RNG = importlib.import_module(PACKAGE+".training_randomness")
PAGING = importlib.import_module(PACKAGE+".native_core_paging")
DIAGNOSTICS = importlib.import_module(PACKAGE+".parameter_diagnostics")
TRANSACTION = importlib.import_module(PACKAGE+".connected_transaction")
WHOLE = importlib.import_module(PACKAGE+".whole_path_execution")


class Owner:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)


class RawOwner:
    def __init__(self, pager, data):
        self._buffers = {"codes": data, "trigger": torch.tensor(0., requires_grad=True)}
        self._parameters, self._modules = {}, {}
        self._non_persistent_buffers_set = {"trigger"}
        self._native_cached_buffers = ()
        self._native_core_pager, self.training = pager, False
        self._pending_stability_events, self._revision = 0, 0
        self.metadata = {"paths": [1, 2]}
    def named_modules(self):
        yield "", self
        for key, value in self._modules.items(): yield key, value
    def authoritative_packed_tensors(self): return (self._buffers["codes"],)


def policy():
    result = Policy()
    result.require_device = lambda *_args: None
    result.status = lambda **_kw: {"memoryPressure": False, "diskPressure": False}
    return result


def native_fixture(root):
    root = Path(root); selected = policy()
    pager = PAGING.NativeCorePager(root/"original", cpu_hot_bytes=65536, accelerator_hot_bytes=0,
        resource_policy=selected, reserve_disk=selected.require_disk, chunk_bytes=8)
    owner = RawOwner(pager, torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8))
    pager.attach(owner, loaded=True)
    brain = Owner(brain_id="owned", storage_path=root, _live_paging_cache_directory=root/"cache",
        core_pager=pager, resource_policy=selected, device=torch.device("cpu"),
        counters={"training_steps": 0}, slow_anchors={}, slow_importance={}, liquid_state=torch.tensor([[.25]]),
        _optimizer=Owner(state={}, param_groups=[]), _optimizer_offloaded=False,
        working_attention_pager=Owner(resident_budget_bytes=65536, scratch_budget_bytes=1<<20,
            device_tile_budget_bytes=65536, page_tokens=8), events=Owner(append=lambda *_args: None))
    return brain, owner, selected


def candidate(name="cpu"):
    return {"deviceId": name, "executionDevice": name, "planSha256": ("1" if name=="cpu" else "2")*64,
        "backendMode": "native-cpu" if name=="cpu" else "native-metal", "asyncDevices": [] if name=="cpu" else ["mps"],
        "supported": True, "fallbackAcknowledged": False}


def inert_current_consumer(root, *, private_failure=False, stop_private=False, genuine_failure=False):
    """Actual policy body, inert wave callback, no neural model/learner."""
    selected, clock, history = policy(), [0], []
    binding = {"policySha256": "1"*64, "topologySha256": "2"*64, "budgetSha256": "3"*64}
    home, alternative = candidate(), candidate("mps")
    review = {"binding": binding, "currentCandidate": home, "candidates": [home, alternative], "supportedWorkKinds": [AUTO.TRAINING_KIND]}
    versions = {"raw-owner": 0}
    scope = Owner(snapshot=lambda: dict(versions), close=lambda: history.append("scope-closed"))
    brain = Owner(brain_id="owned", storage_path=Path(root), _live_paging_cache_directory=Path(root)/"cache",
        device=torch.device("cpu"), resource_policy=selected, core_pager=Owner(_owners={}, observe_learning_owners=lambda _owners: scope),
        counters={"training_steps": 0}, events=Owner(append=lambda *args, **kwargs: history.append((args, kwargs))))
    def genuine(*args, **kwargs):
        history.append("genuine"); brain.counters["training_steps"] += 1
        if genuine_failure: raise RuntimeError("inert genuine update failed")
        return {"loss": 0.125}
    brain._optimize_streaming_experience_batch = genuine
    closure = {"modules": {}, "learningRoots": [], "bindings": []}
    def tick(): clock[0] += 1; return clock[0]
    controller = Owner(_lock=threading.RLock(), _cancel=threading.Event(), _closed=False, _active=False, _phase="active",
        _authority_revision=4, _operation="owned-op", _samples=3, owner_session="owned-session",
        _review=review, _plans={}, recorder=WHOLE.WholePathRecorder(clock_ns=tick, origin="inert-fixture"))
    controller.primitive = lambda value: {key: value[key] for key in ("planSha256", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged")}
    controller._store_plan = lambda key, operation, route, _reference, issued, **kwargs: history.append(("plan", route["executionDevice"], operation))
    def resources(_candidate, budget):
        return nullcontext(lambda: {"budgetSha256": budget, "admissionComplete": True, "pressureObserved": False,
            "physicalGroups": {"ram:system": {"ceilingBytes": 1<<30, "observedPeakBytes": 4096, "peakObservation": "sampled-full-operation"}}})
    issued = ["0"*64, 0]
    def append(_job, payload):
        history.append(payload)
        if payload["phase"] == "measured":
            issued[:] = [WHOLE.digest([issued[0], payload["observation"]["receiptSha256"]]), issued[1]+1]
    host = Owner(hardware_binding=lambda _session: binding, parent_cancelled=lambda: False,
        observation_resources=resources, synchronize=lambda devices: list(devices),
        append_receipt=append, receipt_head=lambda _job: tuple(issued), auto_allowed=lambda _candidate: True)
    controller.ops = host; brain._native_operation_auto = controller
    flags = AUTO._Flags(controller)
    rng = RNG.TrainingRandomScope("cpu", torch.get_rng_state().clone(), transfer=lambda value, _target: value, cancelled=lambda: False)
    effects = hashlib.sha256().hexdigest()
    baseline = {"contract": "a"*64, "state": "b"*64, "report": {"loss": 0.125}, "rng": rng.summary(), "graph": [[], []], "effects": effects}
    placements = []
    @contextmanager
    def placement(value, *_args): placements.append(value["executionDevice"]); yield
    ops = Owner(_closure=lambda: closure, registry=lambda: "c"*64, graph_guard=lambda *_args: [],
        genuine_graph_summary=lambda *_args: [], counters=lambda: {name: 0 for name in WHOLE.COUNTERS},
        move=lambda value, _target: value, genuine_placement=placement, state=lambda *_args: baseline["state"],
        recurrent_transfer=lambda _size: None)
    def private(value, _wave, _closure, _rng, _invoke, observer, _flags):
        history.append(("private", value["executionDevice"]))
        for name in AUTO.TRAINING_STAGES: observer.stage_completed(name)
        if stop_private: controller._cancel.set(); raise InterruptedError("inert stop")
        if private_failure: raise RuntimeError("inert comparison refused")
        return baseline
    ops.execute_private = private; host.training_operations = ops
    intent, authority = AUTO.capture_training_intent(brain, "owned-job")
    job = {"brain": brain, "intent": intent, "authority": authority}
    wave = Owner(identity={"geometrySha256": "d"*64, "windows": 3}, batches=lambda: iter(()), close=lambda: history.append("wave-closed"))
    return brain, controller, job, history, placements, wave


class TrainingWholeWaveCases(unittest.TestCase):
    def test_native_auto_operations_do_not_keep_unloaded_brain_alive(self):
        brain = Owner()
        reference = weakref.ref(brain)
        ops = AUTO.NativeTrainingWaveOps(brain, Owner())
        self.assertIs(ops.brain, brain); del brain
        self.assertIsNone(reference())
        with self.assertRaisesRegex(RuntimeError, "owner was released"): _ = ops.brain

    def test_unexecuted_fixed_controls_are_borrowed_read_only_while_metadata_remains_private(self):
        with tempfile.TemporaryDirectory() as root:
            brain, owner, selected = native_fixture(root)
            fixed = RawOwner(None, torch.tensor([[5, 6]], dtype=torch.uint8)); original = fixed._buffers["codes"]
            with COW.TrainingNamespace(brain, [owner, fixed], Path(root)/"private", selected, lambda: False,
                    mutable_controls={id(owner)}) as namespace:
                self.assertIs(fixed._buffers["codes"], original); fixed.training = True; fixed.metadata["paths"].append(3)
                with COW.private_training_diagnostics(namespace.pager, namespace.controls), self.assertRaisesRegex(RuntimeError, "original parameter"):
                    with DIAGNOSTICS.packed_diagnostic_write(fixed, original, original): self.fail("borrowed learned write admitted")
            self.assertFalse(fixed.training); self.assertEqual(fixed.metadata["paths"], [1, 2])
            self.assertEqual(original.tolist(), [[5, 6]]); brain.core_pager.close()

    def test_actual_consumer_holds_connected_router_representation_through_private_and_genuine_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, job, history, _placements, wave = inert_current_consumer(root)
            active = [0]
            @contextmanager
            def operation():
                active[0] += 1
                try: yield
                finally: active[0] -= 1
            closure = controller.ops.training_operations._closure()
            closure["regions"] = [("regional", Owner(router=Owner(synapses=Owner(_router_state_pager=Owner(operation=operation)))))]
            old_private = controller.ops.training_operations.execute_private
            def private(*args): self.assertEqual(active[0], 1); return old_private(*args)
            controller.ops.training_operations.execute_private = private
            old_genuine = brain._optimize_streaming_experience_batch
            def genuine(*args, **kwargs): self.assertEqual(active[0], 1); return old_genuine(*args, **kwargs)
            brain._optimize_streaming_experience_batch = genuine
            with patch.object(AUTO, "_PreparedWave", return_value=wave):
                self.assertEqual(AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller), {"loss": .125})
            self.assertEqual(active[0], 0); self.assertEqual(history.count("genuine"), 1)

    def test_stop_auto_during_genuine_update_does_not_cancel_or_repeat_it_and_reports_stopped(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, job, history, _placements, wave = inert_current_consumer(root)
            original = brain._optimize_streaming_experience_batch
            def genuine(*args, **kwargs):
                controller._cancel.set(); return original(*args, **kwargs)
            brain._optimize_streaming_experience_batch = genuine
            with patch.object(AUTO, "_PreparedWave", return_value=wave):
                result = AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller)
            self.assertEqual(result, {"loss": .125}); self.assertEqual(history.count("genuine"), 1)
            self.assertEqual(controller._phase, "cancelled"); self.assertFalse(controller._active); self.assertFalse(controller._committing)

    def test_scoped_input_dtype_and_typed_recall_use_admitted_transfer_endpoint_without_model_or_context_text(self):
        recall = importlib.import_module(PACKAGE+".connected_recall_binding")
        transfers, calls, selected = [], [], policy()
        source = Owner(recall_vector=lambda cue, **kwargs: calls.append((cue.clone(), kwargs)) or (torch.tensor([.25, -.5]), None))
        binding = recall.ConnectedRecallBinding(source, 9)
        scope = RNG.TrainingRandomScope("cpu", torch.get_rng_state().clone(),
            transfer=lambda value, target: transfers.append((str(target), value.dtype)) or value, cancelled=lambda: False,
            reserve_conversion=lambda size: selected.reserve_ram(size, "raw input conversion"))
        with scope.activate(), RNG.training_execution("cpu", private=True):
            converted = RNG.learning_transfer(torch.tensor([1, 2], dtype=torch.uint8), "cpu", dtype=torch.float32)
            result = binding(torch.tensor([.5, .75]))
        self.assertEqual(converted.tolist(), [1., 2.]); self.assertEqual(result.tolist(), [.25, -.5])
        self.assertEqual(len(transfers), 3); self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], {"workspace_slots": 9, "record_activity": False})
        self.assertTrue(selected.leases[0].closed)

    def test_scoped_explicit_home_generator_preserves_advanced_state_and_publishes_it_only_once(self):
        source = torch.Generator(device="cpu").manual_seed(7)
        torch.rand((3,), generator=source)
        before = source.get_state().clone(); reference = torch.Generator(device="cpu"); reference.set_state(before)
        expected = torch.rand((4,), generator=reference)
        scope = RNG.TrainingRandomScope("cpu", torch.get_rng_state().clone(), transfer=lambda value, _target: value, cancelled=lambda: False)
        with scope.activate(): actual = RNG.learning_rand((4,), device="cpu", generator=source)
        self.assertTrue(torch.equal(actual, expected)); self.assertTrue(torch.equal(source.get_state(), before))
        scope.commit(); self.assertTrue(torch.equal(source.get_state(), reference.get_state()))
        with self.assertRaisesRegex(RuntimeError, "explicit learner generator changed"):
            scope.commit()

    def test_actual_checkpoint_and_native_draw_sites_preserve_unscoped_rng_without_resetting_scoped_trials(self):
        tree = ast.parse((ROOT/"omni_core"/"model.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id=="checkpoint"]
        self.assertEqual(len(calls), 1)
        rng_flag = next(item.value for item in calls[0].keywords if item.arg=="preserve_rng_state")
        self.assertIsInstance(rng_flag, ast.UnaryOp); self.assertIsInstance(rng_flag.op, ast.Not)
        self.assertEqual(rng_flag.operand.func.id, "training_wave_active")
        self.assertEqual(RNG.training_execution_device("cpu"), torch.device("cpu"))
        with RNG.training_execution("mps", private=True):
            self.assertEqual(RNG.training_execution_device("cpu"), torch.device("mps")); self.assertTrue(RNG.training_private())
        self.assertFalse(RNG.training_private()); self.assertFalse(RNG.training_wave_active())

    def test_ninth_registry_kind_declares_only_the_actual_six_stage_wave_not_complete_training_or_cross_rank(self):
        registry = importlib.import_module(PACKAGE+".operation_auto_registry")
        public = importlib.import_module(PACKAGE+".operation_auto_policy")
        self.assertEqual(registry.TRAINING_KINDS, [AUTO.TRAINING_KIND])
        self.assertEqual(registry.coverage([AUTO.TRAINING_KIND]), [{"workKind": AUTO.TRAINING_KIND, "stages": AUTO.TRAINING_STAGES}])
        prior = [kind for kind in public.WORK_KINDS if kind not in registry.PREFILL_KINDS]
        self.assertEqual(len(prior), 9); self.assertEqual(prior[-1], AUTO.TRAINING_KIND)
        self.assertNotIn("training-backward-optimizer", public.UNSUPPORTED)
        self.assertIn("unfenced-cross-rank-or-complete-training-checkpoint-pipeline", public.UNSUPPORTED)

    def test_private_cleanup_error_after_verified_genuine_update_does_not_undo_or_repeat_it_or_leave_running_flags(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, job, history, placements, wave = inert_current_consumer(root)
            wave.close = lambda: (_ for _ in ()).throw(OSError("inert private backing is still open"))
            with patch.object(AUTO, "_PreparedWave", return_value=wave):
                result = AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller)
            self.assertEqual(result, {"loss": .125}); self.assertEqual(history.count("genuine"), 1)
            self.assertFalse(controller._active); self.assertIsNone(controller._progress)
            self.assertIn("resource-retirement-pending", controller._reason)

    def test_actual_wave_graph_read_scope_rejects_unreviewed_writes_in_its_private_copy(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"original.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE data(value)"); connection.execute("INSERT INTO data VALUES (7)")
            before = path.read_bytes()
            scope = COW.PrivateTrainingSQLite([path], Path(root)/"private", policy(), lambda: False, read_only=True)
            with scope.activate():
                connection = TRANSACTION.borrow_paged_connection(path)
                connection.execute("BEGIN"); self.assertEqual(connection.execute("SELECT value FROM data").fetchone()[0], 7)
                with self.assertRaises(sqlite3.OperationalError): connection.execute("UPDATE data SET value=8")
                connection.commit(); connection.close()
            scope.close(); self.assertEqual(path.read_bytes(), before)

    def test_app_job_scope_persists_exact_reviewed_policy_as_provenance_not_restart_authority(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, _job, _history, _placements, _wave = inert_current_consumer(root)
            with AUTO.training_job(brain, "owned-job", {"manifestId": "manifest", "manifestHash": "f"*64, "epoch": 2}) as intent:
                self.assertTrue(intent["enabled"]); self.assertIs(AUTO._JOB.get()["brain"], brain)
                self.assertEqual(intent["operationId"], "owned-op")
            self.assertIsNone(AUTO._JOB.get())
            files = list((Path(root)/"engine"/"native-operation-jobs").glob("*.json"))
            self.assertEqual(len(files), 1); saved = json.loads(files[0].read_text())
            self.assertEqual(saved, intent); self.assertFalse(saved["savedIntentRestoresAuthority"])
            self.assertEqual(saved["binding"], controller._review["binding"])

    def test_cross_rank_or_missing_actual_job_identity_stays_ordinary_without_parent_review_authority(self):
        with tempfile.TemporaryDirectory() as root:
            brain, _controller, _job, _history, _placements, _wave = inert_current_consumer(root)
            with patch.dict(AUTO.os.environ, {"WORLD_SIZE": "2"}):
                intent, authority = AUTO.capture_training_intent(brain, "job")
            self.assertFalse(intent["enabled"]); self.assertEqual(intent["worldSize"], 2); self.assertIsNone(authority)
            intent, authority = AUTO.capture_training_intent(brain, "")
            self.assertFalse(intent["enabled"]); self.assertIsNone(authority)

    def test_stop_inside_actual_private_raw_wave_restores_original_without_reset_or_worker_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            brain, owner, _selected = native_fixture(root)
            stopped, stages = threading.Event(), []
            host = Owner(brain=brain, admit_candidate=lambda _candidate: None)
            ops = AUTO.NativeTrainingWaveOps(brain, host)
            closure = {"modules": {id(owner): owner}, "learningRoots": [("raw", owner)], "bindings": []}
            original, before = owner._buffers["codes"], torch.get_rng_state().clone()
            def inert_wave():
                with brain.core_pager.use(owner, "cpu"): owner._buffers["codes"].fill_(9)
                stopped.set(); return {"fixture": "raw"}
            wave = Owner(identity={"fixture": "raw-only"}, batches=lambda: iter(()))
            with self.assertRaises(InterruptedError):
                ops.execute_private(candidate(), wave, closure, before, inert_wave,
                    Owner(stage_completed=lambda stage: stages.append(stage)), stopped)
            self.assertIs(owner._buffers["codes"], original); self.assertEqual(original.tolist(), [[1, 2, 3, 4]])
            self.assertTrue(torch.equal(torch.get_rng_state(), before)); self.assertEqual(stages, AUTO.TRAINING_STAGES[:4])
            self.assertEqual(list((brain._live_paging_cache_directory/"training-auto-private").iterdir()), [])
            self.assertFalse(hasattr(brain, "_native_execution_restore_required")); brain.core_pager.close()

    def test_actual_prepared_input_visits_every_row_once_then_replays_exact_spilled_windows_without_cursor_changes(self):
        with tempfile.TemporaryDirectory() as root:
            selected, calls = policy(), []
            rows = [(torch.tensor([1, 2, 3]), torch.tensor([.25, .5])),
                (torch.tensor([4, 5]), torch.tensor([-.25, .75])), (torch.tensor([6, 7, 8, 9]), torch.tensor([.125, 0.]))]
            def source(*_args, **_kwargs):
                calls.append("prepare")
                yield [item[0] for item in rows[:2]], [item[1] for item in rows[:2]]
                yield [rows[2][0]], [rows[2][1]]
            brain = Owner(resource_policy=selected, working_attention_pager=Owner(resident_budget_bytes=0),
                _streaming_experience_window_batches=source)
            prepared = AUTO._PreparedWave(brain, [], {"physicalBatchRecords": 2, "trainingSequenceTokens": 8}, root, threading.Event())
            self.assertIsNotNone(prepared.wave.path); path = prepared.wave.path
            for _repeat in range(2):
                actual = [(ids.tolist(), vector.tolist()) for encoded, vectors in prepared.batches() for ids, vector in zip(encoded, vectors)]
                self.assertEqual(actual, [(ids.tolist(), vector.tolist()) for ids, vector in rows])
            self.assertEqual(calls, ["prepare"]); self.assertEqual(prepared.identity["windows"], 3)
            self.assertEqual(prepared.identity["labels"], 6); prepared.close(); self.assertFalse(path.exists())

    def test_actual_consumer_applies_chosen_current_wave_target_once_and_does_not_replay_outer_publication(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, job, history, placements, wave = inert_current_consumer(root)
            proposal = {"decision": "propose-at-quiescence", "proposedCandidate": {"planSha256": "2"*64}}
            with patch.object(AUTO, "_PreparedWave", return_value=wave), patch.object(AUTO, "select_whole_path_candidate", return_value=proposal):
                result = AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller)
            self.assertEqual(result, {"loss": .125}); self.assertEqual(history.count("genuine"), 1)
            self.assertEqual(brain.counters["training_steps"], 1); self.assertEqual(placements, ["mps"])
            self.assertEqual(sum(isinstance(item, tuple) and item[0]=="private" for item in history), 8)
            self.assertFalse(controller._active); self.assertIn("wave-closed", history)

    def test_private_stop_or_failure_preserves_one_ordinary_fallback_and_no_canonical_update(self):
        for setting in ("private_failure", "stop_private"):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as root:
                brain, controller, job, history, placements, wave = inert_current_consumer(root, **{setting: True})
                with patch.object(AUTO, "_PreparedWave", return_value=wave):
                    result = AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller)
                self.assertIsNone(result); self.assertEqual(history.count("genuine"), 0); self.assertEqual(placements, [])
                self.assertTrue(job["retired"]); self.assertFalse(controller._active)
                # The existing caller executes this once only after None.
                brain._optimize_streaming_experience_batch(); self.assertEqual(history.count("genuine"), 1)

    def test_genuine_failure_after_selected_update_begins_cannot_fall_through_into_a_second_update(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, job, history, placements, wave = inert_current_consumer(root, genuine_failure=True)
            with patch.object(AUTO, "_PreparedWave", return_value=wave), self.assertRaisesRegex(RuntimeError, "inert genuine update failed"):
                AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller)
            self.assertEqual(history.count("genuine"), 1); self.assertEqual(brain.counters["training_steps"], 1)
            self.assertEqual(placements, ["cpu"]); self.assertFalse(controller._active)

    def test_current_job_owner_retirement_declines_before_preparation_without_reusing_saved_intent(self):
        with tempfile.TemporaryDirectory() as root:
            brain, controller, job, history, placements, wave = inert_current_consumer(root)
            controller.owner_session = "replacement-session"
            with patch.object(AUTO, "_PreparedWave", side_effect=AssertionError("stale intent reached preparation")):
                result = AUTO._consume_prepared(brain, [("inert", None)], {"physicalBatchRecords": 1}, job, controller)
            self.assertIsNone(result); self.assertEqual(history.count("genuine"), 0); self.assertEqual(placements, [])

    def test_same_objects_private_packed_and_control_changes_restore_original_bytes_epochs_and_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            brain, owner, selected = native_fixture(root)
            originals = brain.__dict__, owner.__dict__, brain.core_pager._owners[owner]
            original = owner._buffers["codes"]; version = original._version
            with COW.TrainingNamespace(brain, [owner], Path(root)/"private", selected, lambda: False) as namespace:
                self.assertIs(brain.core_pager, namespace.pager)
                self.assertNotIn(owner, namespace.pager._materialized)
                owner.metadata["paths"].append(9); brain.counters["training_steps"] += 1
                with namespace.pager.use(owner, "cpu"), COW.private_training_diagnostics(namespace.pager, namespace.controls):
                    target = owner._buffers["codes"]
                    with DIAGNOSTICS.packed_diagnostic_write(owner, target, target): target[0, 0] = 7
                    self.assertEqual(int(original[0, 0]), 1)
            self.assertIs(brain.__dict__, originals[0]); self.assertIs(owner.__dict__, originals[1])
            self.assertIs(brain.core_pager._owners[owner], originals[2])
            self.assertIs(owner._buffers["codes"], original); self.assertEqual(original._version, version)
            self.assertEqual(owner.metadata, {"paths": [1, 2]}); self.assertEqual(brain.counters, {"training_steps": 0})
            self.assertFalse(any(Path(root).rglob("*.packed-page")))
            brain.core_pager.close()

    def test_private_diagnostics_reject_original_storage_before_outer_journal_receives_it(self):
        with tempfile.TemporaryDirectory() as root:
            brain, owner, selected = native_fixture(root); original = owner._buffers["codes"]
            seen = []
            prior = DIAGNOSTICS._OBSERVERS.set((Owner(before_write=lambda *_args: seen.append(True)),))
            try:
                with COW.TrainingNamespace(brain, [owner], Path(root)/"private", selected, lambda: False) as namespace:
                    with namespace.pager.use(owner, "cpu"), COW.private_training_diagnostics(namespace.pager, namespace.controls):
                        with self.assertRaisesRegex(RuntimeError, "original parameter"):
                            with DIAGNOSTICS.packed_diagnostic_write(owner, original, original): self.fail("write body admitted")
            finally: DIAGNOSTICS._OBSERVERS.reset(prior)
            self.assertEqual(seen, []); self.assertEqual(original.tolist(), [[1, 2, 3, 4]])
            brain.core_pager.close()

    def test_control_storage_aliases_and_views_are_preserved_without_original_writes(self):
        with tempfile.TemporaryDirectory() as root:
            brain, owner, selected = native_fixture(root)
            pager = COW.TrainingCowPager({}, Path(root)/"private", selected, lambda: False, cpu_hot_bytes=0, accelerator_hot_bytes=0)
            controls = COW.TrainingControlCow(pager, selected, lambda: False)
            source = torch.arange(12, dtype=torch.uint8); view = source[2:8:2]
            first, second = controls.tensor(source), controls.tensor(view)
            second[1] = 77
            self.assertEqual(int(first[4]), 77); self.assertEqual(int(source[4]), 4)
            self.assertEqual(second.stride(), view.stride()); self.assertEqual(second.storage_offset(), view.storage_offset())
            controls.close(); pager.close(); brain.core_pager.close()

    def test_private_sqlite_uses_actual_paged_store_connection_seam_and_never_updates_original(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"original.sqlite3"
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE data (id INTEGER PRIMARY KEY, value TEXT)")
                connection.execute("INSERT INTO data VALUES (1,'old')")
            selected = policy(); before = path.read_bytes()
            scope = COW.PrivateTrainingSQLite([path], Path(root)/"private", selected, lambda: False)
            with scope.activate():
                first = TRANSACTION.borrow_paged_connection(path)
                first.execute("BEGIN IMMEDIATE"); first.execute("UPDATE data SET value='private' WHERE id=1"); first.commit(); first.close()
                second = TRANSACTION.borrow_paged_connection(path)
                self.assertEqual(second.execute("SELECT value FROM data").fetchone()[0], "private"); second.close()
                self.assertEqual(len(scope.connections), 1)
                with self.assertRaisesRegex(RuntimeError, "uninventoried"):
                    TRANSACTION.borrow_paged_connection(Path(root)/"other.sqlite3")
            self.assertEqual(path.read_bytes(), before)
            self.assertNotEqual(scope.summary(), COW.PrivateTrainingSQLite([path], root, selected, lambda: False).summary())
            scope.close(); self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list((Path(root)/"private").iterdir()), [])

    def test_private_home_rng_does_not_advance_global_and_genuine_commit_advances_once(self):
        before = torch.get_rng_state().clone()
        try:
            first = RNG.TrainingRandomScope("cpu", before, transfer=lambda value, _target: value, cancelled=lambda: False)
            second = RNG.TrainingRandomScope("cpu", before, transfer=lambda value, _target: value, cancelled=lambda: False)
            with first.activate(): values = RNG.learning_rand((4,), device="cpu")
            with second.activate(): repeated = RNG.learning_rand((4,), device="cpu")
            self.assertTrue(torch.equal(values, repeated)); self.assertTrue(torch.equal(torch.get_rng_state(), before))
            first.commit(); self.assertTrue(torch.equal(torch.get_rng_state(), first.raw.get_state()))
            with self.assertRaisesRegex(RuntimeError, "changed before genuine"):
                second.commit()
        finally: torch.set_rng_state(before)

    def test_current_review_intent_is_source_bound_and_does_not_authorize_a_restart_or_unarmed_job(self):
        binding = {"policySha256": "1"*64, "budgetSha256": "2"*64, "topologySha256": "3"*64}
        brain = Owner(brain_id="owned")
        off, authority = AUTO.capture_training_intent(brain, "job")
        self.assertFalse(off["enabled"]); self.assertIsNone(authority)
        controller = Owner(_lock=threading.RLock(), _closed=False, _cancel=threading.Event(), _active=False,
            _phase="active", _review={"binding": binding, "currentCandidate": candidate(), "supportedWorkKinds": [AUTO.TRAINING_KIND]},
            _operation="reviewed-op", _authority_revision=4, owner_session="session",
            ops=Owner(hardware_binding=lambda _session: binding))
        brain._native_operation_auto = controller
        intent, authority = AUTO.capture_training_intent(brain, "job", {"manifestId": "manifest", "epoch": 2, "text": "never persist raw source"})
        self.assertTrue(intent["enabled"]); self.assertFalse(intent["savedIntentRestoresAuthority"])
        self.assertEqual(intent["source"], {"manifestId": "manifest", "epoch": 2})
        self.assertEqual(intent["ownerSessionId"], "session"); self.assertIs(authority["controller"](), controller)
        controller._cancel.set(); off, authority = AUTO.capture_training_intent(brain, "job")
        self.assertFalse(off["enabled"]); self.assertIsNone(authority)

    def test_actual_private_complete_wave_consumer_uses_raw_cow_and_discards_effects_once(self):
        with tempfile.TemporaryDirectory() as root:
            brain, owner, selected = native_fixture(root)
            original_attention, original_events = brain.working_attention_pager, brain.events
            host = Owner(brain=brain, admit_candidate=lambda _candidate: None)
            ops = AUTO.NativeTrainingWaveOps(brain, host)
            closure = {"modules": {id(owner): owner}, "learningRoots": [("raw", owner)], "bindings": []}
            calls, stages = [], []
            def inert_wave():
                calls.append("private")
                with brain.core_pager.use(owner, "cpu"):
                    value = owner._buffers["codes"]
                    with DIAGNOSTICS.packed_diagnostic_write(owner, value, value): value[0, 1] = 8
                brain.counters["training_steps"] += 1
                brain.events.append("inert-wave", {"raw-fixture": True})
                return {"loss": 0.125, "optimizer_steps": 1}
            wave = Owner(identity={"fixture": "raw-only"}, batches=lambda: iter(()))
            before = torch.get_rng_state().clone()
            first = ops.execute_private(candidate(), wave, closure, before, inert_wave,
                Owner(stage_completed=lambda name: stages.append(name)), threading.Event())
            second = ops.execute_private(candidate(), wave, closure, before, inert_wave, None, threading.Event())
            self.assertEqual(first["contract"], second["contract"]); self.assertEqual(stages, AUTO.TRAINING_STAGES)
            self.assertEqual(calls, ["private", "private"]); self.assertEqual(owner._buffers["codes"].tolist(), [[1, 2, 3, 4]])
            self.assertEqual(brain.counters["training_steps"], 0); self.assertIs(brain.events, original_events)
            self.assertIs(brain.working_attention_pager, original_attention); self.assertTrue(torch.equal(torch.get_rng_state(), before))
            self.assertEqual(list((brain._live_paging_cache_directory/"training-auto-private").iterdir()), [])
            brain.core_pager.close()

    def test_actual_existing_learner_hook_consumes_only_current_job_result_without_running_neural_body(self):
        namespace = {"__package__": PACKAGE}
        methods("omni_core/brain.py", "AdaptiveBrain", {"_optimize_streaming_experience_batch"}, namespace)
        sentinel, calls = {"fixture": "no-neural-work"}, []
        fake = Owner()
        with patch.object(AUTO, "consume_streaming_training_wave", side_effect=lambda *args: calls.append(args) or sentinel):
            result = namespace["_optimize_streaming_experience_batch"](fake, [("inert", torch.tensor([1], dtype=torch.uint8))],
                learning_schedule={"physicalBatchRecords": 1}, schedule_locked=True)
        self.assertIs(result, sentinel); self.assertEqual(len(calls), 1)
        worker = ast.parse((ROOT/"worker.py").read_text())
        cls = next(node for node in worker.body if isinstance(node, ast.ClassDef) and any(isinstance(item, ast.FunctionDef) and item.name=="ingest" for item in node.body))
        ingest = next(item for item in cls.body if isinstance(item, ast.FunctionDef) and item.name=="ingest")
        scoped = [node for node in ast.walk(ingest) if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Name) and item.context_expr.func.id=="training_job" for item in node.items)]
        self.assertEqual(len(scoped), 1)
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr=="ingest" for node in ast.walk(scoped[0])))


if __name__ == "__main__": unittest.main()
