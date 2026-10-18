"""NEW inert cached-frontier controller cases; no Torch/model/Load import."""
import copy
import importlib
import sys
import threading
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path

PACKAGE = "owned_cached_decoder_fixture"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1] / "omni_core")]
sys.modules[PACKAGE] = package
CACHED = importlib.import_module(PACKAGE + ".cached_decoder_execution")
WHOLE = importlib.import_module(PACKAGE + ".whole_path_execution")


class FakeClock:
    now = 100
    def __call__(self): return self.now


class Cache:
    def __init__(self, values): self.values, self.closed = list(values), False
    def close(self): self.closed = True


class Trial:
    def __init__(self, cache, sample, rng):
        self.cache, self.sample, self.rng, self.taken = cache, sample, rng, False
        self.neural_state = "exact-native-state"
        self.contract = WHOLE.digest([cache.values, sample, rng])
    def take_cache(self): self.taken = True; return self.cache
    def close(self):
        if not self.taken: self.cache.close()


class InertCachedOps:
    def __init__(self):
        self.clock, self.epoch = FakeClock(), 1
        self.rows, self.trials, self.actions = [], [], []
        self.callback, self.mismatch, self.valid = None, None, True
        self.current = self.plan("cpu", "native-cpu", "a")
        self.other = self.plan("mps", "native-metal", "b")
        self.counter = dict.fromkeys(WHOLE.COUNTERS, 0)
        self.released = self.retired = 0
        self.policy = "1" * 64
    def plan(self, device, mode, code):
        return {"executionDevice": device, "deviceId": code, "memoryGroupId": "ram:system", "backendMode": mode,
            "asyncDevices": [] if device == "cpu" else [device], "supported": True,
            "fallbackAcknowledged": False, "planSha256": code * 64}
    def quiescent(self): pass
    def supported(self): return True, "inert"
    def candidates(self, acknowledge): return self.current, [self.current, self.other]
    def hardware_binding(self, session, topology=None):
        return {"ownerSessionId": session, "policySha256": self.policy, "topologySha256": "2" * 64,
            "registrySha256": "3" * 64, "budgetSha256": "4" * 64, "homeExecutionDevice": "cpu"}
    def native_state(self, session):
        return {"kind": "live-owner-epochs", "bindingSha256": WHOLE.digest(self.epoch), "ownerSessionId": session,
            "complete": True, "ownerVersions": {"fake-packed": self.epoch}}
    def valid_frontier(self, window, cue, cache): return self.valid
    def capture_frontier(self, window, cue, cache, generator, sampling, session, review):
        state = self.native_state(session)
        identity = {"ownerId": "owned", "ownerSessionId": session, "workKind": CACHED.WORK_KIND,
            "stageCoverage": list(CACHED.STAGES), "workloadSha256": WHOLE.digest([window, cue, cache.values, generator.state]),
            "geometrySha256": "5" * 64, "samplingContractSha256": "6" * 64, "policySha256": self.policy,
            "topologySha256": "2" * 64, "budgetSha256": "4" * 64,
            "cache": {"heat": "warm", "logicalStateSha256": WHOLE.digest(cache.values), "backendInitialization": "warm"}, "nativeState": state}
        return {"cache": cache, "values": list(cache.values), "generator": generator, "rng": generator.state,
            "identity": identity, "epoch": self.epoch}
    def frontier_unchanged(self, frontier, session):
        return frontier["cache"].values == frontier["values"] and frontier["generator"].state == frontier["rng"] and self.epoch == frontier["epoch"]
    def calibration_reservation(self, samples, candidates): return nullcontext()
    def pure_scope(self, stopped): return nullcontext()
    @contextmanager
    def observation_resources(self, candidate, sha):
        first = {"value": True}
        def reading():
            value = {"budgetSha256": sha, "admissionComplete": True, "pressureObserved": False,
                "physicalGroups": {"ram:system": {"ceilingBytes": 1024, "observedPeakBytes": 100,
                    "peakObservation": "operation-start" if first["value"] else "sampled-full-operation"}}}
            first["value"] = False
            return value
        yield reading
    def synchronize(self, devices): return devices
    def counters(self): return dict(self.counter)
    def execute_cached(self, candidate, frontier, observation, stopped):
        self.clock.now += 100 if candidate["executionDevice"] == "cpu" else 10
        self.counter["attentionReadBytes"] += 16
        for stage in CACHED.STAGES: observation.stage_completed(stage)
        values, sample, rng = frontier["values"] + [7], ("token", 0.0, 0.0), frontier["rng"] + 1
        if candidate["executionDevice"] == "mps":
            if self.mismatch == "cache": values[-1] = 8
            if self.mismatch == "sample": sample = ("different-token", 0.0, 0.0)
            if self.mismatch == "rng": rng += 1
        trial = Trial(Cache(values), sample, rng)
        self.trials.append(trial)
        if self.callback: self.callback(candidate)
        return trial
    def exact_output(self, reference, trial): return reference.contract == trial.contract
    def append_receipt(self, job, value): self.rows.append((job, copy.deepcopy(value))); return "receipt"
    def receipts(self, job):
        return (row["observation"] for row_job, row in self.rows if row_job == job and row["phase"] == "measured")
    def receipt_head(self, job):
        head, count = "0" * 64, 0
        for row in self.receipts(job): head, count = WHOLE.digest([head, row["receiptSha256"]]), count + 1
        return head, count
    def auto_allowed(self, candidate): return True
    def commit_sampling(self, frontier, trial):
        if frontier["generator"].state != frontier["rng"]: return None
        frontier["generator"].state = trial.rng
        return trial.sample
    def rebind_original(self, frontier, session): pass
    def release_frontier(self, frontier): self.released += 1
    def retire_observation(self): self.retired += 1


class CachedDecoderFixtures(unittest.TestCase):
    def fixture(self):
        ops = InertCachedOps()
        consumer = CACHED.CachedDecoderExecution("owned", ops, recorder=WHOLE.WholePathRecorder(clock_ns=ops.clock))
        self.addCleanup(consumer.close)
        review = consumer.prepare()
        consumer.arm(operation_id="exact-call", reviewed=review["binding"])
        return consumer, ops, review

    def consume(self, consumer, ops, *, boundary_mutation=None):
        original, generator, owner = Cache([1, 2, 3]), types.SimpleNamespace(state=0), {"cache": None}
        calls = []
        def boundary(cache, state):
            calls.append("world-boundary")
            owner["cache"] = cache
            if boundary_mutation: boundary_mutation(generator)
            return True
        def ordinary(serve=True):
            calls.append("original-forward")
            self.assertFalse(original.closed)
            original.values.append(7)
            if serve: boundary(original, "ordinary-state")
            generator.state += 1
            return ("ordinary-token", 0.0, 0.0)
        result = consumer.consume(window=[1, 2, 3, 4], cue=None, cache=original, generator=generator,
            sampling={"fixture": True}, flags_cancelled=lambda: False, before_sample=boundary,
            fallback=ordinary, fallback_after_boundary=lambda: ordinary(False))
        return result, original, generator, owner, calls

    def test_exact_current_frontier_can_select_and_consume_once_without_replaying_world(self):
        consumer, ops, _review = self.fixture()
        result, original, rng, owner, calls = self.consume(consumer, ops)
        self.assertEqual(result, ("token", 0.0, 0.0))
        self.assertEqual(calls, ["world-boundary"])
        self.assertTrue(original.closed)
        self.assertEqual(owner["cache"].values, [1, 2, 3, 7])
        self.assertFalse(owner["cache"].closed)
        self.assertEqual(rng.state, 1)
        self.assertEqual(len(ops.trials), 8)
        self.assertTrue(all(trial.cache.closed or trial.taken for trial in ops.trials))
        status = consumer.status()
        self.assertEqual(status["phase"], "completed")
        self.assertTrue(status["actualOperationConsumed"])
        self.assertTrue(status["exactCacheAndSamplingMatched"])
        self.assertFalse(status["globalPlacementChanged"])
        self.assertFalse(status["reusableAcrossWorkloads"])
        before = len(ops.trials)
        self.consume(consumer, ops)
        self.assertEqual(len(ops.trials), before)  # One-shot, not an invented reusable lease.

    def test_any_cache_sample_or_rng_mismatch_retains_original_normal_step(self):
        for mismatch in ("cache", "sample", "rng"):
            with self.subTest(mismatch=mismatch):
                consumer, ops, _review = self.fixture()
                ops.mismatch = mismatch
                result, original, rng, owner, calls = self.consume(consumer, ops)
                self.assertEqual(result[0], "ordinary-token")
                self.assertEqual(calls, ["original-forward", "world-boundary"])
                self.assertEqual(original.values, [1, 2, 3, 7])
                self.assertEqual(rng.state, 1)
                self.assertFalse(consumer.status()["actualOperationConsumed"])

    def test_private_stop_cancels_comparison_not_chat_and_preserves_unadvanced_rng(self):
        consumer, ops, _review = self.fixture()
        def stop(_candidate):
            consumer.stop(operation_id="other-call")
            self.assertFalse(consumer.status()["cancellationRequested"])
            consumer.stop(operation_id="exact-call")
        ops.callback = stop
        result, original, rng, owner, calls = self.consume(consumer, ops)
        self.assertEqual(result[0], "ordinary-token")
        self.assertEqual(calls, ["original-forward", "world-boundary"])
        self.assertEqual(rng.state, 1)
        self.assertFalse(original.closed)
        self.assertEqual(consumer.status()["phase"], "cancelled")
        self.assertEqual(ops.rows[0][1]["observation"]["status"], "cancelled")

    def test_real_boundary_rng_change_falls_back_without_replaying_boundary_or_resetting_rng(self):
        consumer, ops, _review = self.fixture()
        result, original, rng, owner, calls = self.consume(consumer, ops, boundary_mutation=lambda rng: setattr(rng, "state", 41))
        self.assertEqual(result[0], "ordinary-token")
        self.assertEqual(calls, ["world-boundary", "original-forward"])
        self.assertEqual(rng.state, 42)
        self.assertFalse(consumer.status()["actualOperationConsumed"])

    def test_changed_learned_epoch_is_not_reset_or_adopted(self):
        consumer, ops, _review = self.fixture()
        ops.callback = lambda _candidate: setattr(ops, "epoch", ops.epoch + 1)
        _result, _original, rng, _owner, calls = self.consume(consumer, ops)
        self.assertEqual(ops.epoch, 2)
        self.assertEqual(rng.state, 1)
        self.assertEqual(calls, ["original-forward", "world-boundary"])
        self.assertEqual(consumer.status()["phase"], "failed-fallback")

    def test_hardware_change_and_invalid_cache_never_start_private_work(self):
        consumer, ops, _review = self.fixture()
        ops.policy = "f" * 64
        self.consume(consumer, ops)
        self.assertEqual(ops.trials, [])
        other, other_ops, _review = self.fixture()
        other_ops.valid = False
        self.consume(other, other_ops)
        self.assertEqual(other_ops.trials, [])
        self.assertEqual(other.status()["phase"], "awaiting-retained-frontier")

    def test_exact_queued_stop_prevents_only_that_arm_and_control_does_not_wait_on_normal_body(self):
        ops = InertCachedOps()
        consumer = CACHED.CachedDecoderExecution("owned", ops)
        self.addCleanup(consumer.close)
        review = consumer.prepare()
        consumer.stop(operation_id="queued")
        self.assertEqual(consumer.arm(operation_id="queued", reviewed=review["binding"])["phase"], "cancelled")
        entered, released = threading.Event(), threading.Event()
        def body(): entered.set(); released.wait(0.5); return "normal"
        worker = threading.Thread(target=lambda: consumer.consume(window=[], cue=None, cache=Cache([]), generator=None,
            sampling={}, flags_cancelled=None, before_sample=None, fallback=body, fallback_after_boundary=body))
        worker.start()
        try:
            self.assertTrue(entered.wait(0.2))
            self.assertEqual(consumer.status()["phase"], "cancelled")
            consumer.stop(operation_id="queued")
        finally: released.set(); worker.join(0.5)
        self.assertFalse(worker.is_alive())

    def test_genuine_boundary_exception_is_not_caught_as_private_failure_and_replayed(self):
        consumer, ops, _review = self.fixture()
        original, generator, calls = Cache([1, 2, 3]), types.SimpleNamespace(state=0), []
        def boundary(_cache, _state): calls.append("real-effect"); raise ValueError("genuine world callback failed")
        def fallback(): calls.append("MUST-NOT-REPLAY"); self.fail("real boundary was replayed")
        with self.assertRaisesRegex(ValueError, "genuine world callback failed"):
            consumer.consume(window=[1, 2, 3, 4], cue=None, cache=original, generator=generator,
                sampling={}, flags_cancelled=lambda: False, before_sample=boundary,
                fallback=fallback, fallback_after_boundary=fallback)
        self.assertEqual(calls, ["real-effect"]); self.assertEqual(generator.state, 0)
        self.assertEqual((ops.released, ops.retired), (1, 1))
        self.assertTrue(all(trial.cache.closed for trial in ops.trials))

    def test_private_pager_cancel_exception_keeps_chat_fallback_but_reports_cancelled(self):
        consumer, ops, _review = self.fixture()
        class PagerCancel(RuntimeError): pass
        def stop_with_pager(_candidate):
            consumer.stop(operation_id="exact-call")
            raise PagerCancel("actual pager cancellation class is not InterruptedError")
        ops.callback = stop_with_pager
        result, original, rng, _owner, calls = self.consume(consumer, ops)
        self.assertEqual(result[0], "ordinary-token"); self.assertFalse(original.closed)
        self.assertEqual(rng.state, 1); self.assertEqual(calls, ["original-forward", "world-boundary"])
        self.assertEqual(consumer.status()["phase"], "cancelled")
        self.assertEqual(ops.rows[0][1]["observation"]["status"], "cancelled")

    def test_pending_prepare_stop_close_and_old_operation_fence_late_publication(self):
        for retirement in ("stop", "close", "unrelated-old-operation"):
            with self.subTest(retirement=retirement):
                consumer, ops, _review = self.fixture()
                # The old operation belongs to the replaced arm, not this new
                # metadata review. No neural operation is executing.
                entered, release, results, failures = threading.Event(), threading.Event(), [], []
                original = ops.hardware_binding
                def blocked(*args, **kwargs):
                    entered.set()
                    if not release.wait(.5): raise AssertionError("owned metadata callback did not release")
                    return original(*args, **kwargs)
                ops.hardware_binding = blocked
                def prepare():
                    try: results.append(consumer.prepare())
                    except BaseException as error: failures.append(error)
                thread = threading.Thread(target=prepare); thread.start()
                try:
                    self.assertTrue(entered.wait(.2))
                    if retirement == "stop":
                        self.assertEqual(consumer.stop()["phase"], "cancelled")
                    elif retirement == "close": consumer.close()
                    else: consumer.stop(operation_id="exact-call")
                finally: release.set(); thread.join(.5)
                self.assertFalse(thread.is_alive())
                ops.hardware_binding = original
                if retirement == "unrelated-old-operation":
                    self.assertEqual(len(results), 1); self.assertEqual(failures, [])
                    self.assertEqual(consumer.status()["phase"], "prepared")
                    self.assertFalse(consumer.status()["cancellationRequested"])
                    self.assertEqual(consumer.arm(operation_id="new-main-call", reviewed=results[0]["binding"])["phase"], "awaiting-retained-frontier")
                else:
                    self.assertEqual(results, []); self.assertEqual(len(failures), 1)
                    self.assertIsInstance(failures[0], InterruptedError)
                    self.assertNotEqual(consumer.status()["phase"], "prepared")
                    self.assertTrue(consumer.status()["cancellationRequested"])


if __name__ == "__main__": unittest.main()
