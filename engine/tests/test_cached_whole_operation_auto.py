"""NEW unified cached-controller cases; fake clocks/cache only, no Torch."""
import copy
import importlib
import types
import unittest
from contextlib import nullcontext

from test_operation_auto_policy import Ops, AUTO, WHOLE

REGISTRY = importlib.import_module(AUTO.__package__ + ".operation_auto_registry")
LAYOUT = importlib.import_module(AUTO.__package__ + ".cached_operation_layout")


class Cache:
    def __init__(self, values): self.values, self.closed = list(values), False
    def close(self): self.closed = True


class Trial:
    def __init__(self, values, sample, rng, activity):
        self.cache, self.sample, self.rng_state, self.activity = Cache(values), sample, rng, activity
        self.neural_state, self.taken = activity, False
        self.contract = WHOLE.digest([values, sample, rng, activity])
    def take_cache(self): self.taken = True; return self.cache
    def close(self):
        if not self.taken: self.cache.close()


class TextOps:
    def __init__(self, host):
        self.host, self.calls, self.trials, self.mismatch, self.callback = host, [], [], None, None
        self.valid, self.released = True, 0
        self.kind = REGISTRY.CACHED_KINDS[0]
        self.activity = "original-activity"
    def valid_frontier(self, *_args): return self.valid
    def capture_frontier(self, window, cue, cache, generator, sampling, session, review, *, stopped=None):
        length = len(window)
        ends = ([length - 1] if length <= 17 else [16, length - 1])
        row = {"owner": "actual.attention.key", "axis": 2, "shape": [1, 2, length - 1, 4], "dtype": "float32", "tokenBytes": 32,
            "pageTokens": 16, "tileBudgetBytes": 65536, "previousPageTokens": 16, "ends": ends,
            "configuredKeyTokens": 64, "maximumSafeKeyTokens": 64}
        context = LAYOUT.cached_operation_range(length, 256, [row])
        state = self.host.state(session)
        binding = review["binding"]
        identity = {"ownerId": "owned", "ownerSessionId": session, "workKind": self.kind, "stageCoverage": REGISTRY.CACHED_STAGES,
            "workloadSha256": WHOLE.digest([window, cue, cache.values, generator.state]), "geometrySha256": WHOLE.digest(length),
            "samplingContractSha256": WHOLE.digest(sampling), "policySha256": self.host.policy, "topologySha256": "2" * 64, "budgetSha256": "4" * 64,
            "cache": {"heat": "warm", "logicalStateSha256": WHOLE.digest(cache.values), "backendInitialization": "warm"}, "nativeState": state}
        return {"window": list(window), "cue": cue, "cache": cache, "values": list(cache.values), "generator": generator, "rng": generator.state,
            "sampling": sampling, "identity": identity, "contextRange": context, "originalActivity": self.activity,
            "stratumKey": WHOLE.digest([self.kind, context, LAYOUT.sampling_inference_class(sampling), binding]),
            "planContract": {"geometrySha256": WHOLE.digest(context), "contextRange": context,
                "numericGate": "exact-cache-connected-activity-neural-sampling-equality",
                "reuseBasis": "same-validated-operator-page-layout-range-current-owner-policy-inference"}}
    def frontier_unchanged(self, op, session):
        return (op["cache"].values == op["values"] and not op["cache"].closed and op["generator"].state == op["rng"]
            and self.operation_state(op, session) == op["identity"]["nativeState"])
    def operation_state(self, op, session): return self.host.state(session)
    def calibration_reservation(self, *_args): return nullcontext()
    def pure_scope(self, flags): return nullcontext()
    def observation_resources(self, *args): return self.host.observation_resources(*args)
    def synchronize(self, devices): return devices
    def counters(self): return self.host.counters()
    def auto_allowed(self, candidate): return True
    def execute(self, candidate, op, observer, flags):
        self.calls.append((candidate["executionDevice"], list(op["window"]), observer is not None))
        self.host.clock.now += 100 if candidate == self.host.current else 10
        self.host.counter["attentionReadBytes"] += 8
        if observer is not None:
            for stage in REGISTRY.CACHED_STAGES: observer.stage_completed(stage)
        values, token, rng = [*op["values"], op["window"][-1]], op["window"][-1] + 1, op["rng"] + 1
        sample, activity = (token, -0.5, 0.25), ["fresh-neural", op["window"][-1]]
        if candidate == self.host.other:
            if self.mismatch == "cache": values[-1] += 1
            if self.mismatch == "token": sample = (token + 1, -0.5, 0.25)
            if self.mismatch == "rng": rng += 1
            if self.mismatch == "activity": activity[-1] += 1
            if self.mismatch == "entropy": sample = (token, -0.5, 0.25000000000000006)
        result = Trial(values, sample, rng, activity); self.trials.append(result)
        if self.callback: self.callback(candidate)
        return result
    def exact_output(self, reference, trial): return reference.contract == trial.contract
    def output_descriptor(self, trial): return {"kind": self.kind}
    def check_output(self, trial, descriptor): assert descriptor == {"kind": self.kind}
    def publish_activity(self, trial): self.activity = trial.activity
    def restore_published_activity(self, op): self.activity = op["originalActivity"]
    def commit_sampling(self, op, trial):
        if op["generator"].state != op["rng"]: return None
        op["generator"].state = trial.rng_state
        return trial.sample
    def rebind_original(self, op, session): pass
    def release_frontier(self, op): self.released += 1


class CompleteCachedPolicyCases(unittest.TestCase):
    def fixture(self):
        host = Ops(); host.cached_operations = TextOps(host)
        policy = AUTO.OperationAutoPolicy("owned", host, recorder=WHOLE.WholePathRecorder(clock_ns=host.clock))
        self.addCleanup(policy.close); review = policy.prepare()
        policy.arm(operation_id="exact-unified-main", reviewed=review["binding"])
        return policy, host, host.cached_operations, review

    def consume(self, policy, *, values=None, remaining=90, boundary=None):
        window = values or [1, 2, 3, 4, 5]
        cache, rng, calls, owner = Cache(window[:-1]), types.SimpleNamespace(state=12), [], {}
        sampling = {"temperature": 0.65, "top_k": 8, "noise": 0.0, "text_boundaries": None, "remaining_tokens": remaining}
        def serve(new_cache, neural):
            calls.append("world"); owner["cache"], owner["neural"] = new_cache, neural
            if boundary: return boundary(rng)
            return True
        def original(serve_boundary=True):
            calls.append("normal")
            self.assertFalse(cache.closed)
            cache.values.append(window[-1])
            if serve_boundary: serve(cache, ["ordinary"])
            else: owner["cache"] = cache
            rng.state += 1
            return ("ordinary", -0.5, 0.25)
        result = policy.consume_cached(window=window, cue=None, cache=cache, generator=rng, sampling=sampling,
            flags_cancelled=lambda: False, before_sample=serve, fallback=original, fallback_after_boundary=lambda: original(False))
        return result, cache, rng, calls, owner

    def test_unified_v2_registry_and_complete_fresh_range_reuse_not_per_token_requalification(self):
        policy, host, ops, review = self.fixture()
        self.assertEqual(review["version"], 2); self.assertNotIn("stageCoverage", review)
        self.assertEqual(len(review["stageCoverageByWorkKind"]), 5)
        result, old, rng, calls, owner = self.consume(policy)
        self.assertEqual(result, (6, -0.5, 0.25)); self.assertEqual(calls, ["world"]); self.assertTrue(old.closed)
        self.assertEqual(rng.state, 13); self.assertEqual(len(ops.calls), 8)
        self.assertEqual(owner["cache"].values, [1, 2, 3, 4, 5])
        result, old, rng, calls, owner = self.consume(policy, values=[9, 8, 7, 6, 5, 4], remaining=89)
        self.assertEqual(result, (5, -0.5, 0.25)); self.assertEqual(calls, ["world"])
        self.assertEqual(len(ops.calls), 9); self.assertEqual(owner["cache"].values, [9, 8, 7, 6, 5, 4])
        plan = policy.plans()["plans"][0]
        self.assertEqual((plan["contextRange"]["minimumTokens"], plan["contextRange"]["maximumTokens"]), (2, 16))
        self.assertEqual(plan["numericGate"], "exact-cache-connected-activity-neural-sampling-equality")
        self.assertFalse(plan["unseenInputOrFrontierWasPreviouslyMeasured"]); self.assertFalse(plan["wholeBrainOrTrainingCovered"])

    def test_real_page_transition_and_participating_epoch_requalify_not_old_output_reuse(self):
        policy, host, ops, _review = self.fixture(); self.consume(policy)
        self.consume(policy, values=list(range(17))); self.assertEqual(len(ops.calls), 16)
        host.epoch += 1; self.consume(policy); self.assertEqual(len(ops.calls), 24)

    def test_exact_cache_activity_entropy_token_rng_mismatches_keep_home_then_skip_same_pool(self):
        for mismatch in ("cache", "activity", "entropy", "token", "rng"):
            with self.subTest(mismatch=mismatch):
                policy, host, ops, _review = self.fixture(); ops.mismatch = mismatch
                result, old, rng, calls, owner = self.consume(policy)
                self.assertEqual(result[0], "ordinary"); self.assertEqual(calls, ["normal", "world"])
                self.assertFalse(old.closed); self.assertEqual(rng.state, 13)
                self.consume(policy, values=[2, 3, 4, 5, 6], remaining=89)
                self.assertEqual(len(ops.calls), 8)

    def test_private_stop_before_delivery_preserves_original_chat_once_and_no_future_pool(self):
        policy, host, ops, _review = self.fixture()
        ops.callback = lambda _candidate: policy.stop(operation_id="exact-unified-main")
        result, old, rng, calls, owner = self.consume(policy)
        self.assertEqual(result[0], "ordinary"); self.assertEqual(calls, ["normal", "world"])
        self.assertFalse(old.closed); self.assertEqual(rng.state, 13); self.assertEqual(ops.released, 1)
        self.assertEqual(policy.status()["phase"], "cancelled")
        before = len(ops.calls); self.consume(policy); self.assertEqual(len(ops.calls), before)

    def test_world_rng_change_uses_original_new_rng_without_repeating_callback_or_retaining_trials(self):
        policy, host, ops, _review = self.fixture()
        def boundary(rng): rng.state = 100; return True
        result, old, rng, calls, owner = self.consume(policy, boundary=boundary)
        self.assertEqual(result[0], "ordinary"); self.assertEqual(calls, ["world", "normal"])
        self.assertEqual(rng.state, 101); self.assertIs(owner["cache"], old)
        self.assertTrue(all(trial.cache.closed for trial in ops.trials))

    def test_real_boundary_exception_is_not_caught_and_replayed(self):
        policy, host, ops, _review = self.fixture()
        def broken(_rng): raise ValueError("actual user-boundary error")
        with self.assertRaisesRegex(ValueError, "actual user-boundary"): self.consume(policy, boundary=broken)
        self.assertEqual(len(ops.calls), 8); self.assertEqual(ops.released, 1)

    def test_late_private_stop_applies_after_linearized_real_delivery_not_to_chat(self):
        policy, host, ops, _review = self.fixture()
        def stop_after_commit_started(_rng): policy.stop(operation_id="exact-unified-main"); return True
        result, old, rng, calls, owner = self.consume(policy, boundary=stop_after_commit_started)
        self.assertEqual(result, (6, -0.5, 0.25)); self.assertEqual(calls, ["world"])
        self.assertTrue(old.closed); self.assertFalse(owner["cache"].closed); self.assertEqual(rng.state, 13)
        self.assertEqual(policy.status()["phase"], "cancelled")

    def test_unavailable_frontier_or_changed_hardware_uses_unchanged_original_once(self):
        policy, host, ops, _review = self.fixture(); ops.valid = False
        result, old, rng, calls, owner = self.consume(policy)
        self.assertEqual(result[0], "ordinary"); self.assertEqual(ops.calls, [])
        ops.valid = True; host.policy = "9" * 64
        result, old, rng, calls, owner = self.consume(policy)
        self.assertEqual(result[0], "ordinary"); self.assertEqual(calls, ["normal", "world"])
        self.assertEqual(policy.status()["phase"], "failed-fallback")

    def test_plan_admission_failure_releases_trials_and_does_not_repeat_an_unstorable_qualification_pool(self):
        policy, host, ops, _review = self.fixture(); host.plan_capacity = 0
        result, old, rng, calls, owner = self.consume(policy)
        self.assertEqual(result[0], "ordinary"); self.assertEqual(calls, ["normal", "world"])
        self.assertFalse(old.closed); self.assertEqual(rng.state, 13)
        self.assertTrue(all(trial.cache.closed for trial in ops.trials)); self.assertEqual(policy.plans()["total"], 0)
        self.assertEqual(policy.status()["phase"], "failed-fallback")
        before = len(ops.calls); self.consume(policy)
        self.assertEqual(len(ops.calls), before)

    def test_failed_exact_restore_raises_without_replaying_original_live_computation(self):
        policy, host, ops, _review = self.fixture()
        class RestoreRequired(RuntimeError): native_owner_restore_required = True
        def fail(_candidate): raise RestoreRequired("source packed owner could not be restored")
        ops.callback = fail
        with self.assertRaisesRegex(RestoreRequired, "could not be restored"): self.consume(policy)
        self.assertEqual(ops.released, 1); self.assertEqual(len(ops.calls), 1)
        self.assertEqual(policy.status()["phase"], "failed-fallback")


if __name__ == "__main__": unittest.main()
