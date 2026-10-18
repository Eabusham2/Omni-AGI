"""NEW inert owner/controller tests; no Torch/model/Load/hardware execution."""
import copy
import importlib
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path

PACKAGE = "owned_decoder_execution_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "omni_core")]
sys.modules[PACKAGE] = package
PATH = importlib.import_module(PACKAGE + ".decoder_path_execution")
WHOLE = importlib.import_module(PACKAGE + ".whole_path_execution")


class FakeClock:
    now = 1_000
    def __call__(self): return self.now


class InertOps:
    relative_tolerance, absolute_tolerance = 2e-5, 1e-4
    def __init__(self):
        self.clock, self.epoch = FakeClock(), 1
        self.rows, self.calls = [], []
        self.callback = None
        self.allow_auto = True
        self.released, self.reference_calls = 0, 0
        self.current = self.plan("cpu", "native-cpu", "a")
        self.other = self.plan("mps", "native-metal", "b")
        self.counter = dict.fromkeys(WHOLE.COUNTERS, 0)
    def plan(self, device, mode, code):
        return {"executionDevice": device, "deviceId": code, "memoryGroupId": "ram:system",
            "backendMode": mode, "asyncDevices": [] if device == "cpu" else [device],
            "supported": True, "fallbackAcknowledged": False, "planSha256": code * 64}
    def supported(self): return True, "inert-owned-decoder"
    def quiescent(self): self.calls.append("quiescent")
    def retain_input(self, window, cue): return {"window": window, "cue": cue}
    def materialize_input(self, capture): return capture
    def release_input(self, capture): self.released += 1
    def retire_observation(self): pass
    def describe_input(self, capture): return {"inputShape": [1, 4], "cueShape": None}
    def native_state(self, session):
        return {"kind": "live-owner-epochs", "bindingSha256": WHOLE.digest(self.epoch),
            "ownerSessionId": session, "complete": True, "ownerVersions": {"fake-packed": self.epoch}}
    def binding(self, capture, session, capture_id, topology=None):
        return {"captureId": capture_id, "ownerSessionId": session, "policySha256": "1" * 64,
            "topologySha256": "2" * 64, "registrySha256": "3" * 64,
            "nativeStateSha256": WHOLE.digest(self.epoch), "workloadSha256": WHOLE.digest(capture),
            "geometrySha256": "4" * 64, "budgetSha256": "5" * 64}
    def candidates(self, acknowledge_cuda): return self.current, [self.current, self.other]
    def identity(self, binding, session, work_kind, stages):
        return {"ownerId": "owned", "ownerSessionId": session, "workKind": work_kind,
            "stageCoverage": list(stages), **{key: binding[key] for key in ("workloadSha256", "geometrySha256", "policySha256", "topologySha256", "budgetSha256")},
            "samplingContractSha256": "6" * 64, "cache": {"heat": "warm", "logicalStateSha256": "7" * 64,
                "backendInitialization": "warm"}, "nativeState": self.native_state(session)}
    def pure_scope(self, cancelled): return nullcontext()
    def calibration_reservation(self, samples, candidates): return nullcontext()
    def reference_output(self, capture, cancelled):
        self.reference_calls += 1
        return {"values": "same"}
    def output_contract(self, reference): return "8" * 64
    def execute(self, candidate, capture, observation, cancelled):
        self.calls.append(candidate["executionDevice"])
        self.clock.now += 100 if candidate["executionDevice"] == "cpu" else 10
        self.counter["coreReadBytes"] += 16
        if observation:
            for stage in PATH.STAGES: observation.stage_completed(stage)
        if self.callback: self.callback(candidate)
        if cancelled.is_set(): raise InterruptedError("inert exact calibration stop")
        return {"values": "same"}
    def compare(self, reference, output):
        return {"equivalent": reference == output, "maximumAbsoluteLogitError": 0.0,
            "maximumAbsoluteStateError": 0.0, "actualLogitsSha256": "9" * 64}
    def release_output(self, output): pass
    def consumer_result(self, output): return output["values"]
    @contextmanager
    def observation_resources(self, candidate, budget_sha):
        calls = {"n": 0}
        def reading():
            calls["n"] += 1
            return {"budgetSha256": budget_sha, "admissionComplete": True, "pressureObserved": False,
                "physicalGroups": {"ram:system": {"ceilingBytes": 1024, "observedPeakBytes": 100,
                    "peakObservation": "operation-start" if calls["n"] == 1 else "sampled-full-operation"}}}
        yield reading
    def synchronize(self, devices): return devices
    def counters(self): return dict(self.counter)
    def append_receipt(self, job, value):
        self.rows.append((job, copy.deepcopy(value)))
        return "receipt-" + str(len(self.rows))
    def receipts(self, job):
        return (value["observation"] for row_job, value in self.rows if row_job == job and value["phase"] == "measured")
    def receipt_head(self, job):
        head, count = "0" * 64, 0
        for receipt in self.receipts(job): head, count = WHOLE.digest([head, receipt["receiptSha256"]]), count + 1
        return head, count
    def admit_candidate(self, candidate): self.calls.append("admit:" + candidate["executionDevice"])
    def auto_allowed(self, candidate): return self.allow_auto
    def input_may_match(self, capture, window, cue): return len(window) == len(capture["window"])
    def matches_input(self, capture, window, cue): return capture == {"window": window, "cue": cue}


class DecoderExecutionFixtures(unittest.TestCase):
    def fixture(self):
        ops = InertOps()
        consumer = PATH.DecoderPathExecution("owned", ops, recorder=WHOLE.WholePathRecorder(clock_ns=ops.clock))
        self.addCleanup(consumer.close)
        consumer.capture_next()
        self.assertEqual(consumer.capture_call([1, 2, 3, 4], None, lambda: "normal-reply"), "normal-reply")
        review = consumer.prepare()
        return consumer, ops, review

    def test_actual_reachable_capture_compare_propose_apply_exact_operation_only(self):
        consumer, ops, review = self.fixture()
        result = consumer.calibrate(operation_id="main-call-1", reviewed=review["binding"])
        self.assertEqual(result["status"]["phase"], "proposal")
        self.assertEqual(result["proposal"]["candidate"]["executionDevice"], "mps")
        self.assertEqual(result["outputEquivalence"]["checked"], True)
        self.assertEqual(ops.reference_calls, 1)
        self.assertEqual(len([value for _job, value in ops.rows if value["phase"] == "warmup"]), 2)
        self.assertEqual(len([value for _job, value in ops.rows if value["phase"] == "measured"]), 6)
        for _job, value in ops.rows:
            self.assertEqual(value["operationId"], "main-call-1")
            self.assertNotIn("window", value)
        proposal = result["proposal"]
        status = consumer.apply(proposal_id=proposal["proposalId"], proposal_sha256=proposal["proposalSha256"])
        self.assertFalse(status["globalPlacementChanged"])
        self.assertFalse(status["retainedKvContinuationMeasured"])
        fallback = []
        self.assertEqual(consumer.execute_if_matching([1, 2, 3, 4], None, lambda: fallback.append(1)), "same")
        self.assertEqual(fallback, [])
        self.assertEqual(consumer.execute_if_matching([9], None, lambda: "original-placement"), "original-placement")
        self.assertEqual(consumer.status()["adoptedOperationCalls"], 1)
        with self.assertRaisesRegex(ValueError, "already-consumed"):
            consumer.apply(proposal_id=proposal["proposalId"], proposal_sha256=proposal["proposalSha256"])

    def test_capture_stop_race_preserves_exactly_one_normal_body_and_releases_borrow(self):
        ops = InertOps()
        consumer = PATH.DecoderPathExecution("owned", ops)
        self.addCleanup(consumer.close)
        consumer.capture_next()
        bodies = []
        def normal():
            bodies.append(1)
            consumer.stop()
            return "completed-normal-chat"
        self.assertEqual(consumer.capture_call([1], None, normal), "completed-normal-chat")
        self.assertEqual(bodies, [1])
        self.assertEqual(ops.released, 1)
        self.assertEqual(consumer.status()["phase"], "cancelled")

    def test_exact_queued_stop_refuses_only_that_calibration_without_running_reference(self):
        consumer, ops, review = self.fixture()
        consumer.stop(operation_id="queued-call")
        with self.assertRaises(InterruptedError):
            consumer.calibrate(operation_id="queued-call", reviewed=review["binding"])
        self.assertEqual(ops.reference_calls, 0)
        self.assertEqual(consumer.status()["phase"], "cancelled")
        other, other_ops, other_review = self.fixture()
        other.stop(operation_id="different-call")
        self.assertIsNotNone(other.calibrate(operation_id="actual-call", reviewed=other_review["binding"])["proposal"])

    def test_live_exact_stop_and_status_use_no_neural_callback_and_preserve_failed_receipt(self):
        consumer, ops, review = self.fixture()
        polls = []
        def stop(_candidate):
            polls.append(consumer.status()["phase"])
            consumer.stop(operation_id="unrelated-call")
            self.assertFalse(consumer.status()["cancellationRequested"])
            consumer.stop(operation_id="exact-call")
        ops.callback = stop
        with self.assertRaises(InterruptedError):
            consumer.calibrate(operation_id="exact-call", reviewed=review["binding"])
        self.assertEqual(polls, ["calibrating"])
        self.assertEqual(consumer.status()["phase"], "cancelled")
        self.assertEqual(ops.rows[0][1]["observation"]["status"], "cancelled")

    def test_changed_live_learned_epoch_never_adopts_or_passes_receipt(self):
        consumer, ops, review = self.fixture()
        ops.callback = lambda _candidate: setattr(ops, "epoch", ops.epoch + 1)
        with self.assertRaisesRegex(RuntimeError, "learned state"):
            consumer.calibrate(operation_id="epoch-change", reviewed=review["binding"])
        self.assertEqual(consumer.status()["phase"], "failed")
        row = ops.rows[0][1]["observation"]
        self.assertFalse(row["eligible"])
        self.assertIn("native-owner-fence-changed-during-operation", row["ineligibleReasons"])

    def test_real_output_mismatch_records_failure_and_does_not_silently_loosen_tolerance(self):
        consumer, ops, review = self.fixture()
        ops.compare = lambda _ref, _output: {"equivalent": False,
            "maximumAbsoluteLogitError": 1.0, "maximumAbsoluteStateError": 2.0}
        with self.assertRaisesRegex(RuntimeError, "float32 equivalence"):
            consumer.calibrate(operation_id="different-output", reviewed=review["binding"])
        self.assertEqual(consumer.status()["phase"], "failed")
        self.assertEqual(ops.rows[0][1]["observation"]["outputEquivalence"]["maximumAbsoluteStateError"], 2.0)
        self.assertIsNone(ops.rows[0][1]["observation"]["outputContractSha256"])

    def test_recomputed_imported_receipt_cannot_supply_live_private_issued_head(self):
        consumer, ops, review = self.fixture()
        original = ops.receipt_head
        ops.receipt_head = lambda job: ("0" * 64, original(job)[1])
        with self.assertRaisesRegex(RuntimeError, "custody"):
            consumer.calibrate(operation_id="forged-custody", reviewed=review["binding"])
        self.assertIsNone(consumer.status()["proposal"])

    def test_manual_policy_and_stale_input_do_not_become_a_global_auto_override(self):
        consumer, ops, review = self.fixture()
        proposal = consumer.calibrate(operation_id="manual-policy", reviewed=review["binding"])["proposal"]
        ops.allow_auto = False
        with self.assertRaisesRegex(RuntimeError, "policy"):
            consumer.apply(proposal_id=proposal["proposalId"], proposal_sha256=proposal["proposalSha256"])
        self.assertIsNone(consumer._lease)
        self.assertFalse(consumer.status()["globalPlacementChanged"])

    def test_new_cpu_compute_receipt_synchronizes_actual_metal_home_output_both_before_and_after(self):
        ops = InertOps()
        consumer = PATH.DecoderPathExecution("owned", ops)
        self.addCleanup(consumer.close)
        binding = ops.binding({"window": [1], "cue": None}, consumer.owner_session, "captured")
        identity = ops.identity(binding, consumer.owner_session, PATH.WORK_KIND, PATH.STAGES)
        plan = {**consumer._primitive(ops.current), "asyncDevices": ["mps"]}
        calls = []
        def synchronize(devices):
            calls.append(devices)
            ops.clock.now += 7
            return devices
        recorder = WHOLE.WholePathRecorder(clock_ns=ops.clock)
        with ops.observation_resources(ops.current, binding["budgetSha256"]) as readings:
            observation = recorder.begin(identity, plan, synchronize=synchronize, counters=ops.counters, resources=readings)
            for stage in PATH.STAGES: observation.stage_completed(stage)
            ops.clock.now += 10
            receipt = observation.finish(native_state_after=ops.native_state(consumer.owner_session), output_contract_sha256="8" * 64)
        self.assertTrue(receipt["eligible"])
        self.assertEqual(calls, [("mps",), ("mps",)])
        self.assertEqual(receipt["totalNanoseconds"], "17")
        self.assertEqual(receipt["priorWorkDrainExcludedNanoseconds"], "7")
        self.assertEqual(receipt["candidate"]["backendMode"], "native-cpu")

    def test_new_capture_admission_failure_does_not_retry_or_cancel_the_normal_body(self):
        ops = InertOps()
        consumer = PATH.DecoderPathExecution("owned", ops)
        self.addCleanup(consumer.close)
        consumer.capture_next()
        def refuse(_window, _cue): raise RuntimeError("inert resource pressure")
        ops.retain_input = refuse
        calls = []
        for _ in range(2):
            self.assertEqual(consumer.capture_call([1], None, lambda: calls.append(1) or "normal"), "normal")
        self.assertEqual(calls, [1, 1])
        self.assertEqual(consumer.status()["phase"], "failed")
        self.assertFalse(consumer.status()["cancellationRequested"])

    def test_new_serial_stop_cleanup_releases_input_and_cannot_retire_a_new_capture(self):
        consumer, ops, review = self.fixture()
        consumer.stop()
        self.assertEqual(ops.released, 0)  # No tensor cleanup on flags-only lane.
        consumer.release_stopped()
        self.assertEqual(ops.released, 1)
        self.assertIsNone(consumer.status()["captureId"])
        consumer.capture_next()
        with self.assertRaisesRegex(RuntimeError, "active/new"):
            consumer.release_stopped()
        self.assertEqual(consumer.status()["phase"], "awaiting-normal-input")

    def test_new_cancelled_calibration_releases_input_at_serial_completion(self):
        consumer, ops, review = self.fixture()
        ops.callback = lambda _candidate: consumer.stop(operation_id="cancel-and-release")
        with self.assertRaises(InterruptedError):
            consumer.calibrate(operation_id="cancel-and-release", reviewed=review["binding"])
        self.assertEqual(ops.released, 1)
        self.assertIsNone(consumer._capture)
        self.assertIsNone(consumer._review)


if __name__ == "__main__": unittest.main()
