"""New inert aggregate scheduling and tiny real SQLite effect custody.

No model/brain/decoder construction, neural forward/backward, Load, training,
hardware benchmark or prior closed operation case runs.
"""
from __future__ import annotations

import ast
import copy
import importlib
import sqlite3
import threading
import types
import unittest
from contextlib import contextmanager
from pathlib import Path

_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "owned_ci_operation_auto_helpers", Path(__file__).with_name("test_operation_auto_policy.py"))
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
Ops, AUTO, WHOLE = _HELPERS.Ops, _HELPERS.AUTO, _HELPERS.WHOLE

SCHEDULE = importlib.import_module(AUTO.__package__ + ".whole_path_scheduler")
REGISTRY = importlib.import_module(AUTO.__package__ + ".operation_auto_registry")
NATIVE = importlib.import_module(AUTO.__package__ + ".whole_path_native")
ROOT = Path(__file__).resolve().parents[1]


class FixedOps:
    def __init__(self, host):
        self.host, self.current, self.active = host, {"ownerSessionId": "s", "policy": "p", "slowStorage": False}, True
        self.rows, self.leases, self.admissions, self.completed = [], [], [], []
        self.db = sqlite3.connect(":memory:", isolation_level=None)
        self.db.execute("CREATE TABLE effects(value INTEGER)")
        self.fail_observation = self.fail_record = self.fail_release = False
        self.revision = 0
        self.owner = types.SimpleNamespace()
    def armed(self): return self.active
    def binding(self): return copy.deepcopy(self.current)
    def owns(self, owner): return owner is self.owner
    def frontier(self): return {"sqlChanges": self.db.total_changes, "revision": self.revision}
    def reserve_metadata(self, size):
        held = types.SimpleNamespace(closed=False)
        held.release = lambda: setattr(held, "closed", True)
        self.leases.append(held); return held
    def admit_effect(self, stage): self.admissions.append(stage)
    def complete_effect(self, stage): self.completed.append(stage)
    def drain_prior_effect(self, _stage): pass
    def admit_selection(self, identity, candidate):
        if not self.active: raise RuntimeError("actual current permission stopped")
        self.admissions.append((identity["workKind"], candidate["planSha256"]))
    @contextmanager
    def resource_observation(self):
        if self.fail_observation: raise RuntimeError("inert sampler unavailable")
        count = [0]
        def read():
            count[0] += 1
            return {"budgetSha256": "4" * 64, "admissionComplete": True, "pressureObserved": False,
                "physicalGroups": {"ram:system": {"ceilingBytes": 1024, "observedPeakBytes": 80,
                    "peakObservation": "operation-start" if count[0] == 1 else "sampled-full-operation"}}}
        try: yield read
        finally:
            if self.fail_release: raise RuntimeError("inert sampler close failed")
    def record(self, row):
        if self.fail_record: raise RuntimeError("inert diagnostic write unavailable")
        self.rows.append(copy.deepcopy(row))


class MemorySchedulerCases(unittest.TestCase):
    def fixture(self):
        host = Ops(); fixed = FixedOps(host)
        self.addCleanup(fixed.db.close)
        scheduler = SCHEDULE.WholePathScheduler(fixed, clock_ns=host.clock)
        recorder = WHOLE.WholePathRecorder(clock_ns=host.clock)
        return host, fixed, scheduler, recorder

    def receipts(self, host, recorder):
        identity = host.capture_operation(None, "image", [1], "s", {})["identity"]
        rows = []
        for _ in range(3):
            for candidate in (host.current, host.other):
                with host.observation_resources(candidate, "4" * 64) as resources:
                    observer = recorder.begin(identity, AUTO.OperationAutoPolicy.primitive(candidate),
                        synchronize=host.synchronize, counters=host.counters, resources=resources)
                    host.clock.now += 100 if candidate is host.current else 10
                    for stage in identity["stageCoverage"]: observer.stage_completed(stage)
                    rows.append(observer.finish(native_state_after=identity["nativeState"], output_contract_sha256="9" * 64))
        return identity, rows

    def select(self, scheduler, host, recorder, identity, rows, *, selector=None):
        return scheduler.select(iter(rows), identity=identity, current_candidate=AUTO.OperationAutoPolicy.primitive(host.current),
            selector=selector or WHOLE.select_whole_path_candidate, minimum_samples=3,
            verify_receipt=lambda row: row["observerId"] == recorder.observer_id)

    def test_actual_sql_effect_and_nested_STDP_working_scratch_phases_run_once_without_double_counting(self):
        host, fixed, scheduler, recorder = self.fixture()
        @SCHEDULE.owned_memory_effect("working-memory-admission")
        def working(owner):
            fixed.db.execute("INSERT INTO effects VALUES(1)"); host.clock.now += 30
        @SCHEDULE.owned_memory_effect("organic-memory-settling")
        def settle(owner):
            fixed.db.execute("INSERT INTO effects VALUES(2)"); host.clock.now += 20
        @SCHEDULE.owned_memory_effect("experience-learning")
        def experience(owner):
            fixed.db.execute("INSERT INTO effects VALUES(3)"); host.clock.now += 50
            working(owner); settle(owner)
        with scheduler.request("chat", "actual-request"):
            experience(fixed.owner)
            identity, receipts = self.receipts(host, recorder)
            result = self.select(scheduler, host, recorder, identity, receipts)
            self.assertEqual(result["decision"], "propose-at-quiescence")
            self.assertEqual(result["proposedCandidate"], AUTO.OperationAutoPolicy.primitive(host.other))
            self.assertEqual(result["aggregateSchedule"]["fixedMemoryNanoseconds"], "100")
            self.assertEqual(result["baselineEvidence"]["minimumNanoseconds"], "200")
            self.assertEqual(result["candidateEvidence"]["maximumNanoseconds"], "110")
        self.assertEqual(fixed.db.execute("SELECT COUNT(*) FROM effects").fetchone()[0], 3)
        self.assertEqual(fixed.rows[-1]["effectsObserved"], 1)
        self.assertEqual(set(fixed.rows[-1]["phases"]), {"experience-learning", "working-memory-admission", "organic-memory-settling"})
        self.assertEqual(fixed.completed, ["experience-learning"])
        self.assertTrue(all(lease.closed for lease in fixed.leases))

    def test_shared_RAM_and_Metal_group_peaks_use_max_and_actual_complete_transfer_costs(self):
        host, fixed, scheduler, recorder = self.fixture()
        captured = []
        def inspect(rows, **kwargs):
            rows = list(rows); captured.extend(rows)
            # This fixture only inspects derived source records. Real selector
            # streaming custody is covered by the preceding actual selector.
            return WHOLE.select_whole_path_candidate((), **kwargs)
        with scheduler.request("chat", "r"):
            with scheduler.effect(SCHEDULE._CURRENT.get(), "experience-learning"):
                fixed.db.execute("INSERT INTO effects VALUES(1)"); host.clock.now += 100
            identity, receipts = self.receipts(host, recorder)
            self.select(scheduler, host, recorder, identity, receipts, selector=inspect)
        self.assertEqual(len(captured), 6)
        for row in captured:
            self.assertEqual(set(row["resourceAfter"]["physicalGroups"]), {"ram:system"})
            self.assertEqual(row["resourceAfter"]["physicalGroups"]["ram:system"]["observedPeakBytes"], 80)
            self.assertEqual(row["counterDeltas"]["activationTransferBytes"], "0")
            self.assertFalse(row["fixedMemoryReplayed"])
            self.assertEqual(row["completedStages"][0], "actual-fixed-owner-memory-prefix")

    def test_stale_policy_context_owner_or_stop_never_authorizes_aggregate_adoption(self):
        for changed in ("policy", "slowStorage", "frontier", "stop"):
            with self.subTest(changed=changed):
                host, fixed, scheduler, recorder = self.fixture()
                with scheduler.request("chat", "r"):
                    with scheduler.effect(SCHEDULE._CURRENT.get(), "memory"):
                        fixed.db.execute("INSERT INTO effects VALUES(1)"); host.clock.now += 100
                    identity, receipts = self.receipts(host, recorder)
                    if changed == "frontier": fixed.revision += 1
                    elif changed == "stop": fixed.active = False
                    else: fixed.current[changed] = "changed"
                    result = self.select(scheduler, host, recorder, identity, receipts)
                    self.assertEqual(result["decision"], "retain-supported-current-fallback")
                    self.assertIn("aggregateScheduleRefused", result)
                self.assertEqual(fixed.db.execute("SELECT COUNT(*) FROM effects").fetchone()[0], 1)

    def test_prefix_cannot_cross_requests_or_be_replayed_from_an_exported_record(self):
        host, fixed, scheduler, recorder = self.fixture()
        with scheduler.request("chat", "first"):
            with scheduler.effect(SCHEDULE._CURRENT.get(), "memory"):
                fixed.db.execute("INSERT INTO effects VALUES(1)"); host.clock.now += 100
        saved = copy.deepcopy(fixed.rows[-1])
        with scheduler.request("chat", "second"):
            identity, receipts = self.receipts(host, recorder)
            result = self.select(scheduler, host, recorder, identity, receipts)
            self.assertNotIn("aggregateSchedule", result)
        self.assertNotEqual(saved["requestSha256"], fixed.rows[-1]["requestSha256"])
        self.assertIsNone(SCHEDULE._CURRENT.get())

    def test_sampler_or_receipt_failure_does_not_repeat_or_fail_committed_learning(self):
        for failed in ("fail_observation", "fail_release", "fail_record"):
            with self.subTest(failed=failed):
                host, fixed, scheduler, recorder = self.fixture(); setattr(fixed, failed, True)
                with scheduler.request("chat", "r"):
                    with scheduler.effect(SCHEDULE._CURRENT.get(), "memory"):
                        fixed.db.execute("INSERT INTO effects VALUES(1)"); host.clock.now += 100
                    identity, receipts = self.receipts(host, recorder)
                    result = self.select(scheduler, host, recorder, identity, receipts)
                    if failed != "fail_record":
                        self.assertEqual(result["decision"], "retain-supported-current-fallback")
                self.assertEqual(fixed.db.execute("SELECT COUNT(*) FROM effects").fetchone()[0], 1)
                self.assertTrue(all(lease.closed for lease in fixed.leases))

    def test_actual_error_after_SQL_effect_propagates_once_and_private_effects_are_refused_before_writes(self):
        host, fixed, scheduler, recorder = self.fixture()
        @SCHEDULE.owned_memory_effect("memory")
        def actual(owner):
            fixed.db.execute("INSERT INTO effects VALUES(1)"); raise ValueError("real memory failure")
        with self.assertRaisesRegex(ValueError, "real memory failure"), scheduler.request("chat", "r"):
            actual(fixed.owner)
        self.assertEqual(fixed.db.total_changes, 1); self.assertFalse(fixed.rows[-1]["completed"])
        token = REGISTRY.PRIVATE_OPERATION.set(True)
        try:
            with self.assertRaisesRegex(RuntimeError, "cannot execute lasting"):
                actual(fixed.owner)
        finally: REGISTRY.PRIVATE_OPERATION.reset(token)
        self.assertEqual(fixed.db.total_changes, 1)

    def test_changed_physical_ceiling_and_tampered_raw_compute_receipts_cannot_win(self):
        for changed in ("ceiling", "checksum"):
            with self.subTest(changed=changed):
                host, fixed, scheduler, recorder = self.fixture()
                with scheduler.request("chat", "r"):
                    with scheduler.effect(SCHEDULE._CURRENT.get(), "memory"):
                        host.clock.now += 100
                    identity, receipts = self.receipts(host, recorder)
                    for row in receipts:
                        if changed == "ceiling":
                            row["resourceAfter"]["physicalGroups"]["ram:system"]["ceilingBytes"] = 2048
                            row["receiptSha256"] = WHOLE.digest({key: value for key, value in row.items() if key != "receiptSha256"})
                        else: row["totalNanoseconds"] = "1"
                    result = self.select(scheduler, host, recorder, identity, receipts)
                    self.assertEqual(result["decision"], "retain-supported-current-fallback")

    def test_actual_sources_bind_one_request_and_reuse_thirteen_complete_operation_selectors(self):
        worker = (ROOT / "worker.py").read_text()
        self.assertIn("owned_request_path(loaded, method, correlated_id)", worker)
        expectations = {"brain.py": ("_learn_experience_impl", "_append_working_memory", "_settle_memory_automatically", "_append_replay",
                "_admit_sensory_embedding", "_preview_chat_experience", "_append_recent_dialogue", "_cool_recent_dialogue"),
            "spiking.py": ("route", "apply_feedback"), "connected_regions.py": ("observe",)}
        for filename, names in expectations.items():
            tree = ast.parse((ROOT / "omni_core" / filename).read_text())
            for name in names:
                node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertTrue(any(isinstance(value, ast.Call) and ast.unparse(value.func) == "owned_memory_effect" for value in node.decorator_list))
        for file in ("operation_auto_policy.py", "cached_whole_operation_auto.py", "prefill_whole_operation_auto.py",
                "forward_whole_operation_auto.py", "media_whole_operation_auto.py", "training_whole_operation.py"):
            source = (ROOT / "omni_core" / file).read_text()
            self.assertIn("select_complete_operation(", source)
        self.assertEqual(len(AUTO.WORK_KINDS), 13)

    def test_real_native_adapter_uses_current_review_context_budget_and_reclaim_boundary(self):
        host = Ops(); calls = []
        config = types.SimpleNamespace(max_seq_len=32, working_memory_slots=16, to_dict=lambda: {"width": 2})
        policy = types.SimpleNamespace(device_resources=types.SimpleNamespace(settings={"allowStorageSpill": False}),
            check_active_pressure=lambda label, **kwargs: calls.append((label, kwargs)))
        def tensor(version=0):
            return types.SimpleNamespace(_version=version, device="cpu", dtype="metadata-only", shape=(1, 2))
        brain = types.SimpleNamespace(brain_id="native-owner", device="cpu", config=config, resource_policy=policy,
            decoder=types.SimpleNamespace(), _regional_policy_sha=lambda: "policy",
            _maintain_neural_state_resources=lambda: calls.append("original-reclaim"),
            memory=types.SimpleNamespace(state_revision=7), working_memory=[], workspace_items=[],
            memory_lifecycle=types.SimpleNamespace(cycle=1, afterimage_vectors=[], active_focus=[]),
            recent_token_context=[1, 2], liquid_state=tensor(), counters={"experiences": 5}, replay=object(), paged_working_memory=object(),
            router=types.SimpleNamespace(population=types.SimpleNamespace(membrane=tensor()),
                synapses=types.SimpleNamespace(plasticity_events=tensor())))
        host._budgets = lambda: [{"memoryGroupId": "ram:system", "ceilingBytes": 1024}]
        host.admit_candidate = lambda candidate: calls.append(("candidate", candidate))
        binding = {"ownerSessionId": "s", "policySha256": WHOLE.digest({"resource": "policy", "configuration": config.to_dict()}),
            "budgetSha256": WHOLE.digest(host._budgets()), "topologySha256": "2" * 64, "registrySha256": "3" * 64}
        brain._native_operation_auto = types.SimpleNamespace(ops=host, owner_session="s", _lock=threading.RLock(), _closed=False,
            _cancel=threading.Event(), _phase="active", _review={"binding": binding, "currentCandidate": host.current, "candidates": [host.current]},
            primitive=AUTO.OperationAutoPolicy.primitive)
        # SimpleNamespace has no weak-reference support. This empty ordinary
        # Python holder is an operational owner only; no neural constructor.
        class Owner: pass
        owner = Owner(); owner.__dict__.update(vars(brain)); brain = owner
        ops = NATIVE.NativeFixedMemoryOps(brain, host)
        self.assertTrue(ops.armed()); self.assertTrue(ops.owns(brain.router)); self.assertFalse(ops.owns(object()))
        identity = {"ownerId": brain.brain_id, "ownerSessionId": "s", "policySha256": binding["policySha256"],
            "budgetSha256": binding["budgetSha256"], "topologySha256": binding["topologySha256"]}
        ops.admit_selection(identity, AUTO.OperationAutoPolicy.primitive(host.current))
        ops.admit_effect("memory")
        self.assertIs(calls[-1][1]["reclaim"], brain._maintain_neural_state_resources)
        before = ops.frontier(); brain.memory.state_revision += 1; brain.liquid_state._version += 1
        self.assertNotEqual(before, ops.frontier())
        before_policy = ops.binding(); policy.device_resources.settings["allowStorageSpill"] = True
        self.assertNotEqual(before_policy, ops.binding())
        brain.recent_token_context = list(range(33))
        with self.assertRaisesRegex(RuntimeError, "context exceeds"): ops.admit_selection(identity, AUTO.OperationAutoPolicy.primitive(host.current))
        brain._native_operation_auto._cancel.set(); self.assertFalse(ops.armed())

    def test_status_polling_and_unarmed_nested_requests_do_not_write_or_inherit_memory_receipts(self):
        host, fixed, scheduler, recorder = self.fixture()
        brain = types.SimpleNamespace(_native_operation_auto=types.SimpleNamespace(ops=types.SimpleNamespace(whole_path_scheduler=scheduler)))
        with SCHEDULE.owned_request_path(brain, "native_operation_auto_status", "poll"):
            self.assertIsNone(SCHEDULE._CURRENT.get())
        self.assertEqual(fixed.rows, [])
        with scheduler.request("chat", "outer"):
            outer = SCHEDULE._CURRENT.get()
            fixed.active = False
            with SCHEDULE.owned_request_path(brain, "chat", "unarmed"):
                self.assertIsNone(SCHEDULE._CURRENT.get())
            self.assertIs(SCHEDULE._CURRENT.get(), outer)

    def test_native_connected_owner_walk_preserves_shared_borrowing_and_admits_only_depth_metadata(self):
        class Owner: pass
        brain = Owner(); held = []
        def reserve(size, label):
            lease = types.SimpleNamespace(size=size, closed=False)
            lease.release = lambda: setattr(lease, "closed", True)
            held.append(lease); return lease
        leaf = types.SimpleNamespace(regions=[])
        shared = types.SimpleNamespace(regions=[types.SimpleNamespace(decoder=leaf)])
        root = types.SimpleNamespace(regions=[types.SimpleNamespace(decoder=shared), types.SimpleNamespace(decoder=shared)])
        brain.decoder, brain.resource_policy = root, types.SimpleNamespace(reserve_ram=reserve)
        ops = NATIVE.NativeFixedMemoryOps(brain, types.SimpleNamespace())
        rows = list(ops._regions())
        self.assertEqual(rows, [root.regions[0], shared.regions[0], root.regions[1], shared.regions[0]])
        self.assertTrue(all(lease.closed for lease in held)); self.assertEqual(len(held), 1)
        shared.regions.append(types.SimpleNamespace(decoder=root))
        with self.assertRaisesRegex(ValueError, "cyclic"): list(ops._regions())
        self.assertTrue(all(lease.closed for lease in held))

    def test_actual_resource_sampler_covers_distinct_home_target_and_one_shared_RAM_group(self):
        tree = ast.parse((ROOT / "omni_core/decoder_execution_native.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NativeDecoderExecutionOps")
        method = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "observation_resources"))
        namespace = {"contextmanager": contextmanager, "threading": threading, "digest": WHOLE.digest,
            "canonical_device": lambda value: "mps" if str(value) == "mps:0" else str(value)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "real_resource_sampler", "exec"), namespace)
        budgets = [{"deviceId": "ram", "memoryGroupId": "ram:system", "enabled": True, "ceilingBytes": 1024},
            {"deviceId": "home", "memoryGroupId": "gpu:0", "enabled": True, "ceilingBytes": 100},
            {"deviceId": "target", "memoryGroupId": "gpu:1", "enabled": True, "ceilingBytes": 100},
            {"deviceId": "unused", "memoryGroupId": "gpu:2", "enabled": True, "ceilingBytes": 100}]
        allocated = {"cpu": 40, "cuda:0": 18, "cuda:1": 25, "cuda:2": 90}
        names = dict(zip(("cpu", "cuda:0", "cuda:1", "cuda:2"), ("ram", "home", "target", "unused")))
        calls = []
        def device(name, **_kwargs):
            calls.append(name)
            return {"deviceId": names[name], "allocatedBytes": allocated[name], "enabled": True, "capacityMeasured": True}
        policy = types.SimpleNamespace(status=lambda **_kwargs: {"ramAdmissionVerified": True, "memoryPressure": False,
            "admissionResidentMemoryBytes": 40, "systemRamBudgetBytes": 1024}, device_budget=device)
        owner = types.SimpleNamespace(brain=types.SimpleNamespace(device="cuda:0", resource_policy=policy), _budgets=lambda: copy.deepcopy(budgets))
        candidate = {"executionDevice": "cuda:1", "deviceId": "target", "asyncDevices": ["cuda:0", "cuda:1"]}
        with namespace["observation_resources"](owner, candidate, WHOLE.digest(budgets)) as read:
            before = read(); allocated["cuda:0"] = 35; after = read()
        self.assertTrue(after["admissionComplete"])
        self.assertEqual(set(after["physicalGroups"]), {"ram:system", "gpu:0", "gpu:1"})
        self.assertEqual(after["physicalGroups"]["gpu:0"]["observedPeakBytes"], 35)
        self.assertNotIn("cuda:2", calls)
        budgets[1]["memoryGroupId"] = budgets[2]["memoryGroupId"] = "ram:system"
        with namespace["observation_resources"](owner, candidate, WHOLE.digest(budgets)) as read:
            read(); shared = read()
        self.assertEqual(set(shared["physicalGroups"]), {"ram:system"})
        self.assertEqual(shared["physicalGroups"]["ram:system"]["observedPeakBytes"], 40)


if __name__ == "__main__": unittest.main()
