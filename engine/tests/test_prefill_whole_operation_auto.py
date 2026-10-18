"""NEW whole-prefix controller/source cases; inert callbacks and raw tensors.

No brain/decoder construction, native forward/backward, Load, training,
hardware route, benchmark or previously closed test case is executed.
"""
from __future__ import annotations

import ast
import copy
import importlib
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_operation_auto_policy import Ops, AUTO, WHOLE

REGISTRY = importlib.import_module(AUTO.__package__ + ".operation_auto_registry")
PREFIX = importlib.import_module(AUTO.__package__ + ".prefill_whole_operation_auto")
ROOT = Path(__file__).resolve().parents[1]


class Cache:
    def __init__(self, values): self.values, self.closed, self.close_count = list(values), False, 0
    def close(self): self.closed = True; self.close_count += 1


class Trial:
    def __init__(self, values, sample, rng, activity):
        self.cache, self.sample, self.rng_state, self.activity = Cache(values), sample, rng, activity
        self.neural_state, self.taken, self.closed = activity, False, False
        self.contract = WHOLE.digest([values, sample, rng, activity])
    def take_cache(self): self.taken = True; return self.cache
    def close(self):
        if not self.closed and not self.taken: self.cache.close()
        self.closed = True


class PrefixOps:
    def __init__(self, host, kind):
        self.host, self.kind, self.valid = host, kind, True
        self.calls, self.trials, self.callback, self.mismatch = [], [], None, None
        self.activity, self.released = "original-neural", 0
    def valid_prefix(self, *_args): return self.valid
    def capture_prefix(self, window, cue, cache, generator, sampling, session, review, *, stopped):
        state = self.host.state(session)
        old = None if cache is None else list(cache.values)
        geometry = WHOLE.digest([self.kind, len(window)])
        identity = {"ownerId": "owned", "ownerSessionId": session, "workKind": self.kind,
            "stageCoverage": list(PREFIX.PREFILL_STAGES), "geometrySha256": geometry,
            "workloadSha256": WHOLE.digest([window, cue, old, generator.state]),
            "samplingContractSha256": WHOLE.digest(sampling), "policySha256": self.host.policy,
            "topologySha256": "2" * 64, "budgetSha256": "4" * 64,
            "cache": {"heat": "cold", "logicalStateSha256": WHOLE.digest(old), "backendInitialization": "warm"},
            "nativeState": state}
        return {"identity": identity, "stratumKey": geometry, "window": list(window), "cue": cue,
            "cache": cache, "original": old, "generator": generator, "rng": generator.state,
            "originalActivity": self.activity, "sampling": copy.deepcopy(sampling),
            "planContract": {"prefillLayout": {"shapeSha256": geometry},
                "numericGate": PREFIX.PREFILL_NUMERIC_GATE, "reuseBasis": PREFIX.PREFILL_REUSE_BASIS}}
    def prefix_unchanged(self, op, session):
        return (op["generator"].state == op["rng"] and self.operation_state(op, session) == op["identity"]["nativeState"]
            and (op["cache"] is None or not op["cache"].closed and op["cache"].values == op["original"]))
    def operation_state(self, op, session): return self.host.state(session)
    def calibration_reservation(self, *args): return self.host.calibration_reservation(*args)
    def pure_scope(self, flags): return self.host.pure_scope(flags)
    def observation_resources(self, *args): return self.host.observation_resources(*args)
    def synchronize(self, devices): return self.host.synchronize(devices)
    def counters(self): return self.host.counters()
    def auto_allowed(self, candidate): return True
    def execute(self, candidate, op, observer, stopped):
        self.calls.append((candidate["executionDevice"], list(op["window"])))
        if op["cache"] is not None:
            assert not op["cache"].closed and op["cache"].values == op["original"]
        self.host.clock.now += 100 if candidate == self.host.current else 10
        self.host.counter["attentionReadBytes"] += len(op["window"]) * 8
        for stage in PREFIX.PREFILL_STAGES: observer.stage_completed(stage)
        values, sample, rng, activity = list(op["window"]), (op["window"][-1] + 1, -0.5, 0.25), op["rng"] + 1, ["new-neural", *op["window"]]
        if candidate == self.host.other:
            if self.mismatch == "cache": values[0] += 1
            if self.mismatch == "token": sample = (sample[0] + 1, sample[1], sample[2])
            if self.mismatch == "logProbability": sample = (sample[0], -0.5000000000000001, sample[2])
            if self.mismatch == "entropy": sample = (sample[0], sample[1], 0.25000000000000006)
            if self.mismatch == "rng": rng += 1
            if self.mismatch == "activity": activity[0] = "wrong-neural"
        trial = Trial(values, sample, rng, activity); self.trials.append(trial)
        if self.callback: self.callback(candidate)
        return trial
    def exact_output(self, reference, trial): return reference.contract == trial.contract
    def output_descriptor(self, trial): return {"kind": self.kind}
    def publish_activity(self, trial): self.activity = trial.activity
    def restore_published_activity(self, op): self.activity = op["originalActivity"]
    def commit_sampling(self, op, trial):
        if op["generator"].state != op["rng"]: return None
        op["generator"].state = trial.rng_state
        return trial.sample
    def rebind_original(self, op, session): pass
    def release_prefix(self, op): self.released += 1


class FullPrefixPolicyCases(unittest.TestCase):
    def fixture(self, *, connected=False, publicly_composed=True):
        kind = PREFIX.PREFILL_KINDS[int(connected)]
        if publicly_composed:
            gate = patch.dict(REGISTRY.STAGE_COVERAGE, {name: list(PREFIX.PREFILL_STAGES) for name in PREFIX.PREFILL_KINDS})
            gate.start(); self.addCleanup(gate.stop)
        host = Ops(); host.prefill_operations = PrefixOps(host, kind)
        if publicly_composed: host.work_kinds = lambda: [kind]
        else: host.work_kinds = lambda: [value for value in AUTO.WORK_KINDS if value not in PREFIX.PREFILL_KINDS]
        policy = AUTO.OperationAutoPolicy("owned", host, recorder=WHOLE.WholePathRecorder(clock_ns=host.clock))
        self.addCleanup(policy.close)
        review = policy.prepare(); policy.arm(operation_id="actual-prefix-job", reviewed=review["binding"])
        return policy, host, host.prefill_operations

    def consume(self, policy, *, window=None, old=False, boundary=None):
        values = window or [1, 2, 3, 4, 5]
        cache = Cache([7, 8, 9]) if old else None
        rng, calls, owner = types.SimpleNamespace(state=10), [], {}
        def real_boundary(new, neural):
            calls.append("world"); owner.update(cache=new, neural=neural)
            if cache is not None:
                self.assertFalse(cache.closed); self.assertEqual(cache.values, [7, 8, 9])
            return True if boundary is None else boundary(rng)
        def ordinary(serve=True):
            calls.append("ordinary")
            if cache is not None:
                self.assertFalse(cache.closed); cache.close()
            fresh = Cache(values)
            if serve: real_boundary(fresh, ["ordinary"])
            else: owner.update(cache=fresh, neural=["ordinary"])
            rng.state += 1
            return ("ordinary", -0.5, 0.25)
        result = policy.consume_prefill(window=values, cue=None, cache=cache, generator=rng,
            sampling={"temperature": 0.65, "top_k": 8, "noise": 0.0, "text_boundaries": None, "remaining_tokens": 20},
            flags_cancelled=lambda: False, before_sample=real_boundary, fallback=ordinary,
            fallback_after_boundary=lambda: ordinary(False))
        return result, cache, rng, calls, owner

    def test_old_nine_kind_review_cannot_acquire_new_full_prefix_authority(self):
        policy, host, ops = self.fixture(publicly_composed=False)
        self.assertFalse(set(PREFIX.PREFILL_KINDS).intersection(policy.status()["supportedWorkKinds"]))
        result, _old, rng, calls, _owner = self.consume(policy)
        self.assertEqual((result[0], rng.state, calls, ops.calls), ("ordinary", 11, ["ordinary", "world"], []))

    def test_initial_ordinary_and_connected_full_prefix_adopt_once_after_private_pool(self):
        for connected in (False, True):
            with self.subTest(connected=connected):
                policy, host, ops = self.fixture(connected=connected)
                result, old, rng, calls, owner = self.consume(policy)
                self.assertEqual((result, old, rng.state, calls), ((6, -0.5, 0.25), None, 11, ["world"]))
                self.assertEqual(owner["cache"].values, [1, 2, 3, 4, 5]); self.assertFalse(owner["cache"].closed)
                self.assertEqual(len(ops.calls), 8); self.assertEqual(ops.released, 1)
                self.assertTrue(all(trial.closed for trial in ops.trials))
                self.assertEqual(sum(trial.taken for trial in ops.trials), 1)
                plan = policy.plans()["plans"][-1]
                self.assertEqual(plan["candidate"], host.other); self.assertEqual(plan["numericGate"], PREFIX.PREFILL_NUMERIC_GATE)
                self.assertEqual(plan["reuseBasis"], PREFIX.PREFILL_REUSE_BASIS)
                self.assertEqual(set(plan["prefillLayout"]), {"shapeSha256"}); self.assertFalse(plan["wholeBrainOrTrainingCovered"])

    def test_existing_invalid_cache_stays_unchanged_until_genuine_adoption(self):
        policy, host, ops = self.fixture()
        result, old, _rng, calls, owner = self.consume(policy, old=True)
        self.assertEqual(result[0], 6); self.assertEqual(calls, ["world"])
        self.assertTrue(old.closed); self.assertEqual(old.close_count, 1); self.assertEqual(old.values, [7, 8, 9])
        self.assertIsNot(owner["cache"], old)

    def test_other_prefix_same_geometry_does_not_reuse_unseen_work(self):
        policy, host, ops = self.fixture()
        self.consume(policy); result, _old, _rng, calls, owner = self.consume(policy, window=[6, 7, 8, 9, 10])
        self.assertEqual((result[0], calls, len(ops.calls)), (11, ["world"], 16))
        self.assertEqual(owner["cache"].values, [6, 7, 8, 9, 10])
        self.assertFalse(any(row.get("phase") == "reused-route" for _job, row in host.rows))

    def test_each_complete_result_mismatch_falls_back_once_without_real_partial_publication(self):
        for field in ("cache", "token", "logProbability", "entropy", "rng", "activity"):
            with self.subTest(field=field):
                policy, host, ops = self.fixture(); ops.mismatch = field
                result, _old, rng, calls, owner = self.consume(policy)
                self.assertEqual((result[0], rng.state, calls), ("ordinary", 11, ["ordinary", "world"]))
                self.assertEqual(ops.activity, "original-neural")
                self.assertTrue(all(trial.closed and not trial.taken for trial in ops.trials))
                self.assertEqual(policy.plans()["total"], 0)

    def test_stop_owner_and_policy_change_retain_original_fallback_once(self):
        for cause in ("stop", "owner", "policy"):
            with self.subTest(cause=cause):
                policy, host, ops = self.fixture()
                def interrupt(candidate):
                    if candidate == host.other:
                        if cause == "stop": policy.stop()
                        elif cause == "owner": host.epoch += 1
                        else: host.policy = "9" * 64
                ops.callback = interrupt
                result, _old, rng, calls, _owner = self.consume(policy)
                self.assertEqual((result[0], rng.state, calls), ("ordinary", 11, ["ordinary", "world"]))
                self.assertTrue(all(trial.closed and not trial.taken for trial in ops.trials))

    def test_real_boundary_stop_keeps_adopted_cache_without_consuming_sampling_rng(self):
        policy, host, ops = self.fixture()
        result, _old, rng, calls, owner = self.consume(policy, boundary=lambda _rng: False)
        self.assertIsNone(result); self.assertEqual((rng.state, calls), (10, ["world"]))
        self.assertFalse(owner["cache"].closed); self.assertEqual(sum(trial.taken for trial in ops.trials), 1)

    def test_real_boundary_rng_change_does_not_repeat_world_callback(self):
        policy, host, ops = self.fixture()
        def change(rng): rng.state += 9; return True
        result, _old, rng, calls, owner = self.consume(policy, boundary=change)
        self.assertEqual((result[0], rng.state, calls), ("ordinary", 20, ["world", "ordinary"]))
        self.assertEqual(owner["cache"].values, [1, 2, 3, 4, 5])
        self.assertTrue(all(trial.closed and not trial.taken for trial in ops.trials))

    def test_genuine_boundary_exception_propagates_without_repeating_actual_chat(self):
        policy, host, ops = self.fixture(); calls = []
        def fail(_rng): calls.append("boundary"); raise RuntimeError("real request failed")
        with self.assertRaisesRegex(RuntimeError, "real request failed"):
            self.consume(policy, boundary=fail)
        self.assertEqual(calls, ["boundary"]); self.assertTrue(all(trial.closed for trial in ops.trials))

    def test_stop_after_real_boundary_does_not_undo_sample_or_repeat_original(self):
        policy, host, ops = self.fixture()
        def stop(_rng): policy.stop(); return True
        result, _old, rng, calls, owner = self.consume(policy, boundary=stop)
        self.assertEqual((result, rng.state, calls), ((6, -0.5, 0.25), 11, ["world"]))
        self.assertFalse(owner["cache"].closed); self.assertEqual(policy.status()["phase"], "cancelled")
        self.assertEqual(policy.plans()["total"], 0)

    def test_failed_private_source_release_clears_running_flags_without_replaying_real_boundary(self):
        policy, host, ops = self.fixture(); boundaries = []
        def release(_operation): raise RuntimeError("private prefix source release failed")
        ops.release_prefix = release
        def observe(_rng): boundaries.append("world"); return True
        with self.assertRaisesRegex(RuntimeError, "private prefix source release failed"):
            self.consume(policy, boundary=observe)
        self.assertEqual(boundaries, ["world"])
        self.assertFalse(policy._active); self.assertFalse(policy._committing)
        self.assertEqual(policy.status()["phase"], "failed-fallback")

    def test_wrapper_disabled_private_unbound_and_retained_dispatch_do_not_add_work(self):
        class Decoder: pass
        class Controller:
            def consume_prefill(self, **_kwargs): return "actual-prefill-consumer"
        decoder, controller = Decoder(), Controller(); calls = []
        args = dict(allowed=True, window=[1], cue=None, cache=None, generator=None, sampling={},
            flags_cancelled=None, before_sample=None, fallback=lambda: calls.append("ordinary"), fallback_after_boundary=None)
        PREFIX.full_prefix_sampling_operation(decoder, **args)
        PREFIX.bind_prefill_whole_operation(decoder, controller)
        self.assertEqual(PREFIX.full_prefix_sampling_operation(decoder, **args), "actual-prefill-consumer")
        PREFIX.full_prefix_sampling_operation(decoder, **{**args, "allowed": False})
        token = REGISTRY.PRIVATE_OPERATION.set(True)
        try: PREFIX.full_prefix_sampling_operation(decoder, **args)
        finally: REGISTRY.PRIVATE_OPERATION.reset(token)
        self.assertEqual(calls, ["ordinary", "ordinary", "ordinary"])

    def test_real_caller_and_native_full_prefix_source_seams_are_composed_without_new_model(self):
        model = ast.parse((ROOT / "omni_core/model.py").read_text())
        decoder = next(node for node in model.body if isinstance(node, ast.ClassDef) and node.name == "OmniDecoder")
        loop = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == "_generate_scoped_iter")
        calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertEqual(sum(call.func.id == "full_prefix_sampling_operation" for call in calls), 1)
        entry = next(call for call in calls if call.func.id == "full_prefix_sampling_operation")
        self.assertEqual(ast.unparse(next(k.value for k in entry.keywords if k.arg == "before_sample")), "record_forward")
        self.assertEqual(ast.unparse(next(k.value for k in entry.keywords if k.arg == "cache")), "original_cache")
        source = (ROOT / "omni_core/prefill_whole_operation_native.py").read_text()
        self.assertIn("._generation_step_bounded(window, cue, None)", source)
        self.assertNotIn(".manual_seed(", source); self.assertNotIn("brain.device =", source)
        self.assertNotIn("deepcopy(", source); self.assertNotIn("prepare_native_execution(", source)
        binding = (ROOT / "omni_core/operation_auto_native.py").read_text()
        self.assertIn("bind_prefill_whole_operation(brain.decoder, controller)", binding)
        self.assertIn("kind in STAGE_COVERAGE for kind in PREFILL_KINDS", binding)


if __name__ == "__main__": unittest.main()
