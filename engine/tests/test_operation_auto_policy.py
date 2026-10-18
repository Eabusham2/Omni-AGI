"""NEW inert perception policy cases. No Torch, model, Load or device work."""
from __future__ import annotations

import copy
import importlib
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path

PACKAGE = "owned_perception_auto_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "omni_core")]
sys.modules[PACKAGE] = package
AUTO = importlib.import_module(PACKAGE + ".operation_auto_policy")
WHOLE = importlib.import_module(PACKAGE + ".whole_path_execution")


class Clock:
    now = 100
    def __call__(self): return self.now


class Lease:
    def __init__(self, ops): self.ops, self.closed = ops, False; ops.leases.append(self)
    def mark_allocated(self, size): self.size = size
    def release(self): self.closed = True


class Output:
    def __init__(self, values): self.value, self.contract = list(values), WHOLE.digest(values); self.closed = self.taken = False
    def take(self): self.taken = True; return list(self.value)
    def close(self): self.closed = True


class Ops:
    def __init__(self):
        self.clock, self.epoch, self.policy = Clock(), 1, "1" * 64
        self.rows, self.calls, self.outputs, self.leases = [], [], [], []
        self.current, self.other = self.candidate("cpu", "native-cpu", "a"), self.candidate("mps", "native-metal", "b")
        self.on_candidate = self.on_supported = self.failure = None
        self.eligible_now, self.mismatch, self.release_count, self.retired, self.plan_capacity = True, False, 0, 0, None
        self.counter = dict.fromkeys(WHOLE.COUNTERS, 0)
    @staticmethod
    def candidate(device, mode, letter):
        return {"executionDevice": device, "deviceId": letter, "memoryGroupId": "ram:system", "backendMode": mode,
            "asyncDevices": [] if device == "cpu" else [device], "supported": True, "fallbackAcknowledged": False, "planSha256": letter * 64}
    def quiescent(self): pass
    def supported(self):
        if self.on_supported: self.on_supported()
        return True, "inert", False
    def work_kinds(self): return list(AUTO.WORK_KINDS)
    def candidates(self, acknowledge): return self.current, [self.current, self.other]
    def hardware_binding(self, session, topology=None):
        return {"ownerSessionId": session, "policySha256": self.policy, "topologySha256": "2" * 64,
            "registrySha256": "3" * 64, "budgetSha256": "4" * 64, "homeExecutionDevice": "cpu"}
    def state(self, session):
        return {"kind": "live-owner-epochs", "bindingSha256": WHOLE.digest(self.epoch), "ownerSessionId": session,
            "complete": True, "ownerVersions": {"actual-selected-packed-owner": self.epoch}}
    def eligible(self, *_args): return self.eligible_now
    def parent_cancelled(self): return False
    def capture_operation(self, hub, modality, tensor, session, review):
        identity = {"ownerId": "owned", "ownerSessionId": session, "workKind": "native.perception." + modality + "-embedding",
            "stageCoverage": list(AUTO.STAGES), "workloadSha256": WHOLE.digest(tensor), "geometrySha256": WHOLE.digest(len(tensor)),
            "samplingContractSha256": "5" * 64, "policySha256": self.policy, "topologySha256": "2" * 64, "budgetSha256": "4" * 64,
            "cache": {"heat": "warm", "logicalStateSha256": "6" * 64, "backendInitialization": "warm"}, "nativeState": self.state(session)}
        return {"values": list(tensor), "identity": identity, "stratumKey": WHOLE.digest([modality, len(tensor)])}
    def unchanged(self, operation, session): return operation["identity"]["nativeState"] == self.state(session)
    def operation_state(self, operation, session): return self.state(session)
    def pure_scope(self, stopped): return nullcontext()
    def calibration_reservation(self, *_args): return nullcontext()
    @contextmanager
    def observation_resources(self, candidate, sha):
        first = [True]
        def reading():
            value = {"budgetSha256": sha, "admissionComplete": True, "pressureObserved": False,
                "physicalGroups": {"ram:system": {"ceilingBytes": 1024, "observedPeakBytes": 64,
                    "peakObservation": "operation-start" if first[0] else "sampled-full-operation"}}}
            first[0] = False
            return value
        yield reading
    def synchronize(self, devices): return list(devices)
    def counters(self): return dict(self.counter)
    def execute(self, candidate, operation, observer, flags):
        self.calls.append((candidate["executionDevice"], list(operation["values"]), observer is not None))
        self.clock.now += 100 if candidate == self.current else 10
        self.counter["activationTransferBytes"] += 8
        if self.failure: raise self.failure
        if observer is not None:
            for stage in AUTO.STAGES: observer.stage_completed(stage)
        values = [value * 2 for value in operation["values"]]
        if self.mismatch and candidate == self.other: values[0] += 1
        output = Output(values); self.outputs.append(output)
        if self.on_candidate: self.on_candidate(candidate)
        return output
    def exact_output(self, reference, trial): return reference.contract == trial.contract
    def output_descriptor(self, output): return {"size": len(output.value)}
    def check_output(self, output, descriptor):
        if self.output_descriptor(output) != descriptor: raise RuntimeError("output geometry changed")
    def reserve_plan(self, size):
        live = sum(not lease.closed for lease in self.leases)
        if self.plan_capacity is not None and live >= self.plan_capacity: raise RuntimeError("actual fixture RAM pressure")
        return Lease(self)
    def auto_allowed(self, candidate): return True
    def release_operation(self, operation): self.release_count += 1
    def retire_observations(self): self.retired += 1
    def append_receipt(self, job, row): self.rows.append((job, copy.deepcopy(row)))
    def receipts(self, job): return (row["observation"] for key, row in self.rows if key == job and row["phase"] == "measured")
    def receipt_head(self, job):
        head, count = "0" * 64, 0
        for row in self.receipts(job): head, count = WHOLE.digest([head, row["receiptSha256"]]), count + 1
        return head, count


class PerceptionAutoPolicyCases(unittest.TestCase):
    def fixture(self, *, arm=True):
        ops = Ops(); policy = AUTO.OperationAutoPolicy("owned", ops, recorder=WHOLE.WholePathRecorder(clock_ns=ops.clock))
        self.addCleanup(policy.close)
        review = policy.prepare()
        if arm: policy.arm(operation_id="exact-main-policy", reviewed=review["binding"])
        return policy, ops, review

    def consume(self, policy, values, *, normal=None):
        calls = []
        def ordinary(): calls.append("normal"); return ["ordinary", *values]
        result = policy.consume(None, "image", values, normal or ordinary)
        return result, calls

    def test_real_new_input_uses_qualified_operator_not_an_old_embedding_or_token_proof(self):
        policy, ops, _review = self.fixture()
        value, calls = self.consume(policy, [2, 3])
        self.assertEqual((value, calls), ([4, 6], [])); self.assertEqual(len(ops.calls), 8)
        plan = policy.plans()["plans"][0]
        self.assertEqual(len(plan["id"]), 64)
        self.assertEqual(plan["qualifiedSamples"], 3)
        value, calls = self.consume(policy, [11, 12])
        self.assertEqual((value, calls), ([22, 24], []))
        self.assertEqual(ops.calls[-1], ("mps", [11, 12], False)); self.assertEqual(len(ops.calls), 9)
        self.assertFalse(plan["unseenInputOrFrontierWasPreviouslyMeasured"])
        self.assertFalse(plan["wholeBrainOrTrainingCovered"])
        self.assertTrue(all(output.closed for output in ops.outputs))
        reused = ops.rows[-1][1]
        self.assertTrue(reused["reuseIsInferenceForUnseenInput"]); self.assertFalse(reused["previousOutputWasReused"])

    def test_exact_mismatch_keeps_tested_ordinary_route_without_requalifying_each_frame(self):
        policy, ops, _review = self.fixture(); ops.mismatch = True
        value, calls = self.consume(policy, [2, 3])
        self.assertEqual((value, calls), ([4, 6], []))
        self.assertEqual(policy.plans()["plans"][0]["candidate"], ops.current)
        value, calls = self.consume(policy, [9, 10])
        self.assertEqual((value, calls), (["ordinary", 9, 10], ["normal"]))
        self.assertEqual(len(ops.calls), 8)

    def test_selected_learned_epoch_change_requalifies_but_stale_hardware_does_not_adopt(self):
        policy, ops, _review = self.fixture(); self.consume(policy, [2, 3])
        old = policy.plans()["plans"][0]["qualifiedNativeStateSha256"]
        ops.epoch += 1; self.consume(policy, [5, 6]); self.assertEqual(len(ops.calls), 16)
        self.assertNotEqual(old, policy.plans()["plans"][0]["qualifiedNativeStateSha256"])
        ops.policy = "9" * 64
        value, calls = self.consume(policy, [7, 8])
        self.assertEqual((value, calls), (["ordinary", 7, 8], ["normal"]))
        self.assertEqual(policy.status()["phase"], "failed-fallback")
        self.assertEqual(policy.plans()["total"], 0); self.assertTrue(all(lease.closed for lease in ops.leases))

    def test_stop_and_pressure_preserve_public_body_once_and_release_private_outputs(self):
        for cause in ("stop", "pressure"):
            with self.subTest(cause=cause):
                policy, ops, _review = self.fixture()
                if cause == "stop": ops.on_candidate = lambda _candidate: policy.stop(operation_id="exact-main-policy")
                else: ops.failure = RuntimeError("fixture resource reserve refused")
                value, calls = self.consume(policy, [2, 3])
                self.assertEqual((value, calls), (["ordinary", 2, 3], ["normal"]))
                self.assertTrue(all(output.closed for output in ops.outputs))
                self.assertEqual(ops.release_count, 1)
                self.assertEqual(policy.status()["qualifiedPlanCount"], 0)
                self.assertFalse(policy.status()["globalPlacementChanged"])

    def test_genuine_original_error_is_not_replayed_and_unsafe_restore_cannot_fall_through(self):
        policy, ops, _review = self.fixture(); ops.mismatch = True; self.consume(policy, [1, 2])
        calls = []
        def broken(): calls.append("original"); raise ValueError("genuine public operation error")
        with self.assertRaisesRegex(ValueError, "genuine public"): self.consume(policy, [3, 4], normal=broken)
        self.assertEqual(calls, ["original"])
        other, callbacks, _review = self.fixture(); callbacks.failure = AUTO.OperationAutoRestoreRequired("fixture exact restore failed")
        normal = []
        with self.assertRaises(AUTO.OperationAutoRestoreRequired): self.consume(other, [3], normal=lambda: normal.append("unsafe-body"))
        self.assertEqual(normal, [])

    def test_prepare_ticket_stop_exact_queued_arm_and_active_policy_refusal(self):
        policy, ops, review = self.fixture(arm=False)
        policy.stop(operation_id="old-unrelated-id")
        self.assertEqual(policy.status()["phase"], "prepared")
        policy.stop(operation_id="new-main-id")
        self.assertEqual(policy.arm(operation_id="new-main-id", reviewed=review["binding"])["phase"], "cancelled")
        policy.release_stopped(operation_id="new-main-id")
        ops.on_supported = lambda: policy.stop()
        with self.assertRaises(InterruptedError): policy.prepare()
        self.assertEqual(policy.status()["phase"], "cancelled")
        ops.on_supported = None; review = policy.prepare(); policy.arm(operation_id="replacement", reviewed=review["binding"])
        with self.assertRaisesRegex(RuntimeError, "Stop the current"): policy.prepare()
        policy.stop(operation_id="old-unrelated-id")
        self.assertEqual(policy.status()["phase"], "awaiting-operation")

    def test_pages_bind_owner_revision_filter_and_physical_metadata_pressure_evicts_derived_only(self):
        policy, ops, _review = self.fixture()
        for length in range(1, 36): self.consume(policy, [1] * length)
        page = policy.plans(); self.assertEqual(len(page["plans"]), 32); self.assertEqual(page["total"], 35)
        self.assertEqual(len(policy.plans(after=page["nextAfter"])["plans"]), 3)
        other, _callbacks, _ = self.fixture()
        with self.assertRaises(ValueError): other.plans(after=page["nextAfter"])
        with self.assertRaises(ValueError): policy.plans(after=page["nextAfter"][:-1] + "x")
        self.consume(policy, [2] * 36)
        with self.assertRaises(ValueError): policy.plans(after=page["nextAfter"])
        ops.plan_capacity = 36; self.consume(policy, [3] * 37)
        self.assertEqual(policy.plans()["total"], 36)
        self.assertTrue(ops.leases[0].closed)
        with self.assertRaises(ValueError): policy.plans(limit=129)

    def test_ineligible_live_boundary_never_runs_private_body_or_changes_normal_input(self):
        policy, ops, _review = self.fixture(); ops.eligible_now = False
        values = [8, 9]; value, calls = self.consume(policy, values)
        self.assertEqual((value, calls), (["ordinary", 8, 9], ["normal"]))
        self.assertEqual(values, [8, 9]); self.assertEqual(ops.calls, []); self.assertEqual(policy.plans()["total"], 0)
        self.assertIn("connected-text-recall-recurrent", policy.status()["unsupportedRoutes"])

    def test_no_plan_metadata_room_falls_back_once_instead_of_repeating_an_unstorable_pool(self):
        policy, ops, _review = self.fixture(); ops.plan_capacity = 0
        value, calls = self.consume(policy, [7, 8])
        self.assertEqual((value, calls), (["ordinary", 7, 8], ["normal"]))
        self.assertEqual(policy.status()["phase"], "failed-fallback")
        before = len(ops.calls); self.consume(policy, [9, 10]); self.assertEqual(len(ops.calls), before)


if __name__ == "__main__": unittest.main()
