"""New inert mixed-region contracts: no Torch, model, Load or hardware work.

Fake tensors/clocks exercise the actual route/receipt control flow only. They
are not throughput, neural equality or hardware-placement qualification.
"""
from __future__ import annotations

import ast
import copy
import importlib
import sys
import threading
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "owned_mixed_forward_helpers", Path(__file__).with_name("test_forward_whole_operation_auto.py"))
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
AUTO, WHOLE, Ops = _HELPERS.AUTO, _HELPERS.WHOLE, _HELPERS.Ops
FORWARD, ForwardOps, Trial = _HELPERS.FORWARD, _HELPERS.ForwardOps, _HELPERS.Trial

SCHEDULE = importlib.import_module(AUTO.__package__ + ".mixed_region_schedule")
NATIVE = importlib.import_module(AUTO.__package__ + ".mixed_region_native")
REGIONAL = importlib.import_module(AUTO.__package__ + ".regional_execution")
REGISTRY = importlib.import_module(AUTO.__package__ + ".operation_auto_registry")
ROOT = Path(__file__).resolve().parents[1] / "omni_core"


def base(device="cpu"):
    value = {"executionDevice": device, "deviceId": device + ":interface", "memoryGroupId": "vram:0" if device.startswith("cuda:") else "ram:system",
        "backendMode": "native-cpu" if device == "cpu" else "native-metal" if device == "mps" else "explicit-cuda-cpu-tile",
        "asyncDevices": [] if device == "cpu" else [device], "supported": True, "fallbackAcknowledged": device.startswith("cuda:")}
    return {**value, "planSha256": WHOLE.digest(value)}


def rehash(value):
    value["planSha256"] = WHOLE.digest({key: item for key, item in value.items() if key != "planSha256"})
    return value


class Lease:
    def __init__(self, policy, size): self.policy, self.size, self.closed = policy, size, False; policy.leases.append(self)
    def __enter__(self): return self
    def __exit__(self, *_args): self.release()
    def mark_allocated(self, size): self.allocated = size
    def release(self): self.closed = True


class Resource:
    def __init__(self): self.leases, self.refuse = [], False
    def reserve_ram(self, size, _reason):
        if self.refuse: raise RuntimeError("inert actual metadata budget refused")
        return Lease(self, size)


class Tensor:
    def __init__(self, device="cpu"): self.device = device
    def numel(self): return 8
    def element_size(self): return 4


class Packed:
    def __init__(self): self._buffers, self._parameters = {}, {}; self._native_compute_device = "cpu"


class Block:
    def __init__(self):
        self.training, self.width, self.packed = False, 2, Packed()
        self._buffers, self._parameters, self.calls, self.failure = {"gain": Tensor()}, {}, [], None
    def modules(self): return iter([self, self.packed])
    @REGIONAL.regional_block_call
    def forward(self, hidden):
        self.calls.append((hidden.device, self._buffers["gain"].device, self.packed._native_compute_device))
        if self.failure: raise self.failure
        return Tensor(hidden.device)
    @REGIONAL.regional_block_call
    def forward_cached(self, hidden, cache=None): return self.forward.__wrapped__(self, hidden), cache


class PoolOps:
    def __init__(self, blocks):
        self.host, self.admissions, self.transfers, self.computes = types.SimpleNamespace(), [], [], []
        policy = Resource()
        self.brain = types.SimpleNamespace(resource_policy=policy, core_pager=types.SimpleNamespace(_owners={block.packed: object() for block in blocks}),
            _regional_describe=lambda block: {"dimensions": block.width, "feedForward": 4},
            working_attention_pager=types.SimpleNamespace(admit_compute=lambda *args, **kwargs: self.computes.append((args, kwargs))))
    def admit_candidate(self, candidate):
        if candidate["backendMode"] == SCHEDULE.MIXED_MODE: return NATIVE.admit_mixed_candidate(self, candidate)
        self.admissions.append(copy.deepcopy(candidate))
        return {"selection": "manual" if candidate["executionDevice"] == "cpu" else "auto", "memoryGroupId": candidate["memoryGroupId"]}
    def move_activity(self, tensor, target): self.transfers.append((tensor.device, target)); return tensor if tensor.device == target else Tensor(target)


class MixedRegionCases(unittest.TestCase):
    def pool(self, n=2): return SCHEDULE.MixedRegionCandidatePool(base(), [base(), base("mps"), base("cuda:0")], "3" * 64, n)

    def native(self):
        blocks = [Block(), Block()]; ops = PoolOps(blocks)
        host = Ops(); host.current = base()
        op = ForwardOps(host, FORWARD.FORWARD_KINDS[0]).capture_forward([1, 2], None, True, None, "session", {}, stopped=None)
        op["closure"] = {"modules": {id(block): block for block in blocks}, "names": {id(block): "decoder.blocks." + str(i) for i, block in enumerate(blocks)}}
        op["stack"] = ExitStack(); self.addCleanup(op["stack"].close)
        review, fences = {"currentCandidate": base(), "candidates": [base(), base("mps")]}, []
        fake_model = types.SimpleNamespace(DecoderBlock=Block)
        with patch.dict(sys.modules, {AUTO.__package__ + ".model": fake_model}):
            pool = NATIVE.prepare_mixed_operation_candidates(ops, op, review, current=lambda: fences.append(True))
        self.assertIsInstance(pool, NATIVE.NativeMixedRegionPool)
        return ops, op, pool, blocks, review, fences

    def test_candidate_family_is_linear_in_real_regions_and_targets_not_cartesian(self):
        for n in (1, 4, 65):
            pool = self.pool(n); rows = list(pool)
            self.assertEqual(len(rows), 3 + 2 * (2 * n - 1) + int(n > 1))
            self.assertEqual(len({row["planSha256"] for row in rows}), len(rows))
            for row in rows[3:]:
                self.assertEqual(SCHEDULE.checked_mixed_candidate(row, bases=pool.bases, registry_sha="3" * 64, region_count=n), row)
                self.assertEqual(row["executionDevice"], "cpu")
            if n > 1:
                self.assertEqual([row["candidate"]["executionDevice"] for row in rows[-1]["regionSchedule"]["segments"]], ["mps", "cuda:0"])
        with self.assertRaises(ValueError): SCHEDULE.MixedRegionCandidatePool(base(), [base(), base()], "3" * 64, 2)
        with self.assertRaises(ValueError): SCHEDULE.MixedRegionCandidatePool(base(), [base("mps")], "3" * 64, 2)
        with self.assertRaises(ValueError): self.pool(0)

    def test_hash_participants_ranges_and_exact_review_cannot_manufacture_authority(self):
        candidate = list(self.pool(4))[-1]
        variants = []
        for field, value in (("executionDevice", "mps"), ("memoryGroupId", "vram:fake"), ("asyncDevices", ["mps"]), ("fallbackAcknowledged", False)):
            row = copy.deepcopy(candidate); row[field] = value; variants.append(rehash(row))
        for first, count in ((-1, 1), (0, 0), (0, 5), (0.5, 1), (True, 1)):
            row = copy.deepcopy(candidate); row["regionSchedule"]["segments"][0].update(first=first, count=count); variants.append(rehash(row))
        row = copy.deepcopy(candidate); row["regionSchedule"]["segments"][1]["first"] = 0; variants.append(rehash(row))
        row = copy.deepcopy(candidate); row["regionSchedule"]["segments"][0]["candidate"] = base(); variants.append(rehash(row))
        row = copy.deepcopy(candidate); row["sourcePath"] = "/not-authority"; variants.append(rehash(row))
        for value in variants:
            with self.subTest(value=value), self.assertRaises(ValueError): SCHEDULE.checked_mixed_candidate(value)
        with self.assertRaises(ValueError): SCHEDULE.checked_mixed_candidate(candidate, bases=[base(), base("mps")])
        with self.assertRaises(ValueError): SCHEDULE.checked_mixed_candidate(candidate, registry_sha="4" * 64)
        with self.assertRaises(ValueError): SCHEDULE.checked_mixed_candidate(candidate, region_count=5)
        checked = SCHEDULE.checked_mixed_candidate(candidate); checked["regionSchedule"]["segments"].clear()
        self.assertTrue(candidate["regionSchedule"]["segments"])

    def test_current_captured_operation_and_fresh_base_admissions_are_required_not_saved_json(self):
        ops, op, pool, blocks, review, fences = self.native()
        candidate = next(row for row in pool if row["backendMode"] == SCHEDULE.MIXED_MODE)
        primitive = {name: candidate[name] for name in ("planSha256", "backendMode", "asyncDevices", "supported", "fallbackAcknowledged")}
        self.assertTrue(NATIVE.admit_mixed_primitive(ops.host, op["identity"], primitive, review))
        self.assertEqual([row["executionDevice"] for row in ops.admissions], ["cpu", "mps"])
        self.assertEqual(len(fences), 0)  # Full frontier fences surround the source stream, not every metadata row.
        pool.require(candidate); self.assertEqual(len(fences), 1)
        wrong = {**op["identity"], "workloadSha256": "8" * 64}
        with self.assertRaises(RuntimeError): NATIVE.admit_mixed_primitive(ops.host, wrong, primitive, review)
        with self.assertRaises(ValueError): NATIVE.admit_mixed_primitive(ops.host, op["identity"], primitive, {"candidates": [base()]})
        changed = copy.deepcopy(candidate); changed["regionSchedule"]["registrySha256"] = "4" * 64; rehash(changed)
        with self.assertRaises(ValueError): pool.require(changed)
        op["stack"].close()
        self.assertTrue(all(lease.closed for lease in ops.brain.resource_policy.leases))
        with self.assertRaises(RuntimeError): NATIVE.admit_mixed_primitive(ops.host, op["identity"], primitive, review)

    def test_metadata_pressure_keeps_reviewed_homogeneous_fallback_without_any_model_work(self):
        ops, op, pool, blocks, review, _fences = self.native(); pool.close()
        ops.brain.resource_policy.refuse = True
        with patch.dict(sys.modules, {AUTO.__package__ + ".model": types.SimpleNamespace(DecoderBlock=Block)}):
            self.assertIs(NATIVE.prepare_mixed_operation_candidates(ops, op, review, current=lambda: None), review["candidates"])
        self.assertEqual(op["mixedScheduleUnavailable"], "RuntimeError")
        self.assertTrue(all(not block.calls for block in blocks)); self.assertEqual(getattr(ops.host, "_mixed_region_pools"), {})

    def test_actual_decorated_block_moves_only_selected_activity_then_restores_controls_and_hints(self):
        ops, op, pool, blocks, review, fences = self.native()
        candidate = next(row for row in pool if row["backendMode"] == SCHEDULE.MIXED_MODE)
        stopped = threading.Event(); consumer = NATIVE.MixedDecoderBlockConsumer(ops, candidate, op, stopped)
        class Target:
            active, _lease = True, {"epoch": "inert-current-operation"}
            def policy_identity(self): return ops.brain.resource_policy
            def execute(self, module, method, hidden, args, kwargs): return consumer.execute(self, module, method, hidden, args, kwargs)
        target, original = Target(), blocks[0]._buffers["gain"]
        torch = types.SimpleNamespace(Tensor=Tensor, device=lambda value: value, is_grad_enabled=lambda: False)
        attention = types.SimpleNamespace(PagedTensorSequence=type("Paged", (), {}))
        with patch.dict(sys.modules, {"torch": torch, AUTO.__package__ + ".working_attention_paging": attention}), REGIONAL.transient_regional_scope(target):
            output = blocks[0].forward(Tensor()); unchanged = blocks[1].forward(Tensor())
            self.assertEqual((output.device, unchanged.device), ("cpu", "cpu"))
            self.assertEqual(blocks[0].calls, [("mps", "mps", "mps")]); self.assertEqual(blocks[1].calls, [("cpu", "cpu", "cpu")])
            self.assertIs(blocks[0]._buffers["gain"], original); self.assertEqual(blocks[0].packed._native_compute_device, "cpu")
            self.assertEqual(len(ops.computes), 1)
            blocks[0].failure = ValueError("inert block body failed")
            with self.assertRaisesRegex(ValueError, "body failed"): blocks[0].forward(Tensor())
            self.assertIs(blocks[0]._buffers["gain"], original); self.assertEqual(blocks[0].packed._native_compute_device, "cpu")
            blocks[0].failure = None; blocks[0].width = 3
            with self.assertRaisesRegex(RuntimeError, "geometry"): blocks[0].forward(Tensor())
            blocks[0].width = 2; stopped.set()
            with self.assertRaises(InterruptedError): blocks[0].forward(Tensor())
        self.assertIsNone(REGIONAL.current_regional_target(ops.brain.resource_policy))

    def test_cached_route_refuses_materialized_kv_and_retired_owner_before_calling_body(self):
        ops, op, pool, blocks, _review, _fences = self.native()
        candidate = next(row for row in pool if row["backendMode"] == SCHEDULE.MIXED_MODE)
        consumer = NATIVE.MixedDecoderBlockConsumer(ops, candidate, op, threading.Event())
        target = types.SimpleNamespace(active=True, _lease={"epoch": "inert"}, policy_identity=lambda: ops.brain.resource_policy)
        torch = types.SimpleNamespace(Tensor=Tensor, device=lambda value: value, is_grad_enabled=lambda: False)
        Paged = type("Paged", (), {})
        with patch.dict(sys.modules, {"torch": torch, AUTO.__package__ + ".working_attention_paging": types.SimpleNamespace(PagedTensorSequence=Paged)}):
            method = Block.forward_cached.__wrapped__
            with self.assertRaisesRegex(RuntimeError, "paged KV"):
                consumer.execute(target, blocks[0], method, Tensor(), (types.SimpleNamespace(key=Tensor(), value=Tensor()),), {})
            self.assertEqual(blocks[0].calls, [])
            cache = types.SimpleNamespace(key=Paged(), value=Paged())
            output, returned = consumer.execute(target, blocks[0], method, Tensor(), (cache,), {})
            self.assertEqual(output.device, "cpu"); self.assertIs(returned, cache)
            pool.close()
            with self.assertRaises(InterruptedError): consumer.execute(target, blocks[0], method, Tensor(), (), {})

    def test_six_real_text_paths_share_the_runtime_seam_without_advertising_mixed_media_or_training(self):
        for filename in ("cached_whole_operation_auto.py", "prefill_whole_operation_auto.py", "forward_whole_operation_auto.py"):
            tree = ast.parse((ROOT / filename).read_text())
            calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
            self.assertIn("prepare_candidates", calls); self.assertIn("ops.execute", calls)
            self.assertIn("mixed_pool", {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
        native = (ROOT / "cached_whole_operation_native.py").read_text()
        self.assertIn("_Target(self, candidate, stopped, operation)", native)
        self.assertIn("return self._mixed.execute(self, module, method, hidden, args, kwargs)", native)
        for filename in ("media_whole_operation_auto.py", "training_whole_operation.py"):
            self.assertNotIn("mixed_candidates", (ROOT / filename).read_text())
        self.assertEqual(len(REGISTRY.STAGE_COVERAGE), 13)


class MixedCompleteForwardCases(unittest.TestCase):
    def fixture(self, *, bad=False):
        host = Ops(); host.current, host.other = base(), base("mps")
        kind = FORWARD.FORWARD_KINDS[0]
        gate = patch.dict(REGISTRY.STAGE_COVERAGE, {kind: list(FORWARD.FORWARD_STAGES)}); gate.start(); self.addCleanup(gate.stop)
        class MixedOps(ForwardOps):
            max_live = 0
            def mixed_candidates(self, op, review, *, current):
                current(); self.pool = SCHEDULE.MixedRegionCandidatePool(review["currentCandidate"], review["candidates"], "3" * 64, 2)
                return self.pool
            def execute(self, candidate, op, observer, flags):
                self.calls.append(candidate["planSha256"])
                self.host.clock.now += 1 if candidate.get("backendMode") == SCHEDULE.MIXED_MODE else 100
                if observer is not None:
                    for stage in FORWARD.FORWARD_STAGES: observer.stage_completed(stage)
                output = {"hidden": list(op["window"]), "logits": [sum(op["window"])]}
                if bad and observer is None: output["logits"][0] += 1
                trial = Trial(output, []); self.trials.append(trial)
                self.max_live = max(self.max_live, sum(not value.closed for value in self.trials))
                return trial
        ops = host.forward_operations = MixedOps(host, kind); host.work_kinds = lambda: [kind]
        policy = AUTO.OperationAutoPolicy("owned", host, recorder=WHOLE.WholePathRecorder(clock_ns=host.clock, origin="inert-fixture"))
        self.addCleanup(policy.close); review = policy.prepare(); policy.arm(operation_id="mixed-inert", reviewed=review["binding"])
        return policy, ops

    def test_injected_inert_proposal_executes_its_exact_issued_mixed_route_and_publishes_only_once(self):
        policy, ops = self.fixture(); fallbacks = []
        def proposal(*_args, **_kwargs):
            selected = next(row for row in ops.pool if row["backendMode"] == SCHEDULE.MIXED_MODE)
            return {"decision": "propose-at-quiescence", "proposedCandidate": policy.primitive(selected)}
        with patch.object(FORWARD, "select_whole_path_candidate", side_effect=proposal):
            output = policy.consume_full_forward(window=[4, 8], cue=None, use_global_workspace=False, attention_mask=None,
                fallback=lambda: fallbacks.append(True))
        self.assertEqual((output, fallbacks), ({"hidden": [4, 8], "logits": [12]}, []))
        plan = policy.plans()["plans"][0]
        self.assertEqual(plan["candidate"]["backendMode"], SCHEDULE.MIXED_MODE)
        self.assertEqual(plan["candidate"]["executionDevice"], "cpu")
        self.assertEqual(len(ops.calls), 4 * len(ops.pool) + 1)  # Warmup, three actual inert full samples, selected reexecution.
        self.assertEqual((ops.max_live, len(ops.published), sum(trial.taken for trial in ops.trials)), (2, 1, 1))
        self.assertTrue(all(trial.closed for trial in ops.trials))
        self.assertEqual(ops.released, 1)

    def test_inert_fixture_receipts_do_not_claim_a_real_hardware_mixed_winner(self):
        policy, ops = self.fixture()
        policy.consume_full_forward(window=[4, 8], cue=None, use_global_workspace=False, attention_mask=None, fallback=lambda: None)
        self.assertEqual(policy.plans()["plans"][0]["candidate"], base())
        selection = next(row for _job, row in ops.host.rows if row["phase"] == "selection")
        self.assertEqual(selection["comparison"]["decision"], "retain-supported-current-fallback")

    def test_reexecuted_result_mismatch_cannot_publish_or_reuse_the_old_trial(self):
        policy, ops = self.fixture(bad=True); fallbacks = []
        output = policy.consume_full_forward(window=[4, 8], cue=None, use_global_workspace=False, attention_mask=None,
            fallback=lambda: fallbacks.append("ordinary") or "ordinary")
        self.assertEqual((output, fallbacks, ops.published), ("ordinary", ["ordinary"], []))
        self.assertTrue(all(trial.closed for trial in ops.trials)); self.assertFalse(policy._active)
        self.assertEqual(policy.status()["phase"], "failed-fallback")


if __name__ == "__main__": unittest.main()
