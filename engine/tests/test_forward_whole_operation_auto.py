"""New inert full-forward controller cases; no Torch or learned operation."""
from __future__ import annotations

import ast
import copy
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

_HELPERS_SPEC = importlib.util.spec_from_file_location(
    "owned_forward_operation_helpers", Path(__file__).with_name("test_operation_auto_policy.py"))
_HELPERS = importlib.util.module_from_spec(_HELPERS_SPEC)
_HELPERS_SPEC.loader.exec_module(_HELPERS)
Ops, AUTO, WHOLE = _HELPERS.Ops, _HELPERS.AUTO, _HELPERS.WHOLE

REGISTRY = importlib.import_module(AUTO.__package__ + ".operation_auto_registry")
FORWARD = importlib.import_module(AUTO.__package__ + ".forward_whole_operation_auto")
ROOT = Path(__file__).resolve().parents[1]


class Trial:
    def __init__(self, output, activity):
        self.output, self.activity = copy.deepcopy(output), copy.deepcopy(activity)
        self.contract = WHOLE.digest([output, activity])
        self.closed = self.taken = False
    def take(self): self.taken = True; return self.output
    def close(self): self.closed = True


class ForwardOps:
    def __init__(self, host, kind):
        self.host, self.kind, self.valid = host, kind, True
        self.calls, self.trials, self.published, self.released = [], [], [], 0
        self.callback, self.mismatch, self.failure = None, None, None
        self.rng, self.original_rng = 19, 19
        self.restore_required = False
    def valid_forward(self, *_args): return self.valid
    def capture_forward(self, window, cue, workspace, mask, session, review, *, stopped):
        geometry = WHOLE.digest([self.kind, len(window), workspace, mask is not None])
        state = self.host.state(session)
        identity = {"ownerId": "owned", "ownerSessionId": session, "workKind": self.kind,
            "stageCoverage": list(FORWARD.FORWARD_STAGES), "geometrySha256": geometry,
            "workloadSha256": WHOLE.digest([window, cue, workspace, mask, self.rng]),
            "samplingContractSha256": WHOLE.digest("no-sampling"), "policySha256": self.host.policy,
            "topologySha256": "2" * 64, "budgetSha256": "4" * 64,
            "cache": {"heat": "unknown", "logicalStateSha256": None, "backendInitialization": "warm"}, "nativeState": state}
        return {"identity": identity, "stratumKey": WHOLE.digest([geometry, window]), "window": list(window),
            "cue": cue, "workspace": workspace, "mask": mask, "rng": self.rng,
            "planContract": {"forwardLayout": {"shapeSha256": geometry},
                "numericGate": FORWARD.FORWARD_NUMERIC_GATE, "reuseBasis": FORWARD.FORWARD_REUSE_BASIS}}
    def require_original_frontiers(self, op, session):
        if self.rng != op["rng"] or self.restore_required:
            raise AUTO.OperationAutoRestoreRequired("original deterministic frontier changed")
    def forward_unchanged(self, op, session):
        return op["identity"]["nativeState"] == self.host.state(session)
    def operation_state(self, op, session): return self.host.state(session)
    def calibration_reservation(self, *args): return self.host.calibration_reservation(*args)
    def pure_scope(self, flags): return self.host.pure_scope(flags)
    def observation_resources(self, *args): return self.host.observation_resources(*args)
    def synchronize(self, devices): return self.host.synchronize(devices)
    def counters(self): return self.host.counters()
    def auto_allowed(self, candidate): return True
    def execute(self, candidate, op, observer, flags):
        self.calls.append((candidate["executionDevice"], list(op["window"]), op["workspace"], op["mask"]))
        self.host.clock.now += 100 if candidate == self.host.current else 10
        self.host.counter["activationTransferBytes"] += len(op["window"]) * 8
        for stage in FORWARD.FORWARD_STAGES: observer.stage_completed(stage)
        if self.failure: raise self.failure
        output = {"hidden": list(op["window"]), "logits": [sum(op["window"])], "action_logits": [1, 2]}
        if op["workspace"]: output["workspace"] = [sum(op["window"]) / len(op["window"])]
        activity = ["connected", list(op["window"])] if "connected" in self.kind else []
        if candidate == self.host.other:
            if self.mismatch == "activity": activity.append("wrong")
            elif self.mismatch is not None: output[self.mismatch][0] += 1
        trial = Trial(output, activity); self.trials.append(trial)
        if self.callback: self.callback(candidate)
        return trial
    def exact_output(self, reference, trial): return reference.contract == trial.contract
    def output_descriptor(self, trial): return {"outputs": sorted(trial.output), "kind": self.kind}
    def publish_activity(self, trial): self.published.append(copy.deepcopy(trial.activity))
    def release_forward(self, op): self.released += 1


class FullForwardPolicyCases(unittest.TestCase):
    def fixture(self, *, connected=False, publicly_composed=True):
        kind = FORWARD.FORWARD_KINDS[int(connected)]
        gate = patch.dict(REGISTRY.STAGE_COVERAGE, {value: list(FORWARD.FORWARD_STAGES) for value in FORWARD.FORWARD_KINDS})
        gate.start(); self.addCleanup(gate.stop)
        host = Ops(); host.forward_operations = ForwardOps(host, kind)
        host.work_kinds = lambda: [kind] if publicly_composed else [REGISTRY.PERCEPTION_KINDS[0]]
        policy = AUTO.OperationAutoPolicy("owned", host, recorder=WHOLE.WholePathRecorder(clock_ns=host.clock, origin="inert-fixture"))
        self.addCleanup(policy.close)
        reviewed = policy.prepare(); policy.arm(operation_id="inert-full-forward", reviewed=reviewed["binding"])
        return policy, host, host.forward_operations

    def consume(self, policy, *, values=None, workspace=True, ordinary=None):
        calls = []
        def normal():
            calls.append("ordinary")
            return {"ordinary": list(values or [3, 5, 7])}
        return (policy.consume_full_forward(window=values or [3, 5, 7], cue=[0.25],
            use_global_workspace=workspace, attention_mask=[True, True, False], fallback=ordinary or normal), calls)

    def test_old_review_or_unavailable_boundary_retains_one_actual_forward(self):
        policy, host, ops = self.fixture(publicly_composed=False)
        result, calls = self.consume(policy)
        self.assertEqual((result, calls, ops.calls), ({"ordinary": [3, 5, 7]}, ["ordinary"], []))
        policy, host, ops = self.fixture(); ops.valid = False
        result, calls = self.consume(policy)
        self.assertEqual((calls, ops.calls), (["ordinary"], []))

    def test_complete_ordinary_and_connected_outputs_publish_once_without_sampling(self):
        for connected in (False, True):
            with self.subTest(connected=connected):
                policy, host, ops = self.fixture(connected=connected)
                output, calls = self.consume(policy)
                self.assertEqual(output, {"hidden": [3, 5, 7], "logits": [15], "action_logits": [1, 2], "workspace": [5.0]})
                self.assertEqual((calls, len(ops.calls), len(ops.published), ops.released, ops.rng), ([], 8, 1, 1, 19))
                self.assertTrue(all(trial.closed for trial in ops.trials))
                self.assertEqual(sum(trial.taken for trial in ops.trials), 1)
                plan = policy.plans()["plans"][-1]
                self.assertEqual((plan["candidate"], plan["numericGate"], plan["reuseBasis"]),
                    (host.current, FORWARD.FORWARD_NUMERIC_GATE, FORWARD.FORWARD_REUSE_BASIS))
                self.assertEqual(set(plan["forwardLayout"]), {"shapeSha256"})
                self.assertFalse(plan["wholeBrainOrTrainingCovered"])

    def test_same_shaped_new_input_is_a_new_complete_operation_not_output_or_route_reuse(self):
        policy, host, ops = self.fixture()
        self.consume(policy)
        output, calls = self.consume(policy, values=[11, 13, 17])
        self.assertEqual((output["logits"], calls, len(ops.calls), len(ops.published)), ([41], [], 16, 2))
        self.assertEqual(len(policy.plans()["plans"]), 2)

    def test_inert_explicit_selector_proposal_takes_exact_selected_trial_once(self):
        policy, host, ops = self.fixture(connected=True)
        proposal = {"decision": "propose-at-quiescence", "proposedCandidate": policy.primitive(host.other)}
        with patch.object(FORWARD, "select_whole_path_candidate", return_value=proposal):
            result, calls = self.consume(policy)
        self.assertEqual((result["logits"], calls, len(ops.published)), ([15], [], 1))
        self.assertEqual(policy.plans()["plans"][-1]["candidate"], host.other)
        self.assertEqual(sum(value.taken for value in ops.trials), 1)

    def test_every_output_and_real_connected_activity_mismatch_refuses_adoption(self):
        for key in ("hidden", "logits", "action_logits", "workspace", "activity"):
            with self.subTest(key=key):
                policy, host, ops = self.fixture(connected=True); ops.mismatch = key
                result, calls = self.consume(policy)
                self.assertEqual((result, calls, ops.published), ({"ordinary": [3, 5, 7]}, ["ordinary"], []))
                self.assertTrue(all(trial.closed for trial in ops.trials))
                self.assertFalse(policy._active); self.assertIsNone(policy._progress)

    def test_private_stop_or_policy_change_keeps_original_forward_once(self):
        for change in ("stop", "policy", "owner"):
            with self.subTest(change=change):
                policy, host, ops = self.fixture()
                def alter(_candidate):
                    if change == "stop": policy.stop(operation_id="inert-full-forward")
                    elif change == "policy": host.policy = "9" * 64
                    else: host.epoch += 1
                ops.callback = alter
                result, calls = self.consume(policy)
                self.assertEqual((calls, ops.published), (["ordinary"], []))
                self.assertFalse(policy._active); self.assertTrue(all(trial.closed for trial in ops.trials))

    def test_unknown_rng_draw_pauses_without_reset_or_retrying_changed_real_input(self):
        policy, host, ops = self.fixture()
        ops.callback = lambda _candidate: setattr(ops, "rng", ops.rng + 1)
        with self.assertRaises(AUTO.OperationAutoRestoreRequired): self.consume(policy)
        self.assertEqual((ops.rng, ops.published, ops.released), (20, [], 1))
        self.assertFalse(policy._active); self.assertTrue(all(trial.closed for trial in ops.trials))

    def test_pressure_and_genuine_fallback_failure_are_not_repeated(self):
        policy, host, ops = self.fixture(); ops.failure = RuntimeError("inert allocation pressure")
        calls = []
        def fail(): calls.append("actual"); raise ValueError("actual forward refused")
        with self.assertRaisesRegex(ValueError, "actual forward refused"):
            self.consume(policy, ordinary=fail)
        self.assertEqual(calls, ["actual"]); self.assertEqual(ops.released, 1); self.assertFalse(policy._active)

    def test_publication_and_cleanup_failures_cannot_retry_after_real_delivery(self):
        policy, host, ops = self.fixture()
        def publication(trial):
            ops.published.append(trial.activity); raise RuntimeError("real publication failed")
        ops.publish_activity = publication
        with self.assertRaisesRegex(RuntimeError, "real publication failed"): self.consume(policy)
        self.assertEqual(len(ops.published), 1); self.assertFalse(policy._active)
        policy, host, ops = self.fixture()
        def release(_op): ops.released += 1; raise RuntimeError("private release failed")
        ops.release_forward = release
        with self.assertRaisesRegex(RuntimeError, "private release failed"): self.consume(policy)
        self.assertEqual((len(ops.published), ops.released), (1, 1)); self.assertFalse(policy._active)

    def test_bound_decorator_preserves_arguments_labels_and_private_bypass(self):
        class Decoder: pass
        class Controller:
            def consume_full_forward(self, **kwargs): calls.append(("controller", kwargs)); return "qualified"
        decoder, controller, calls = Decoder(), Controller(), []
        @FORWARD.measured_full_forward
        def inert(decoder, window, cue=None, labels=None, use_global_workspace=None, attention_mask=None):
            calls.append(("ordinary", window, cue, labels, use_global_workspace, attention_mask)); return "ordinary"
        FORWARD.bind_full_forward_operation(decoder, controller)
        self.assertEqual(inert(decoder, "full", "cue", None, True, "mask"), "qualified")
        self.assertEqual(calls[-1][1]["attention_mask"], "mask")
        self.assertEqual(inert(decoder, "labels", labels="training"), "ordinary")
        token = REGISTRY.PRIVATE_OPERATION.set(True)
        try: self.assertEqual(inert(decoder, "nested"), "ordinary")
        finally: REGISTRY.PRIVATE_OPERATION.reset(token)
        self.assertEqual([row[0] for row in calls], ["controller", "ordinary", "ordinary"])

    def test_actual_public_source_and_initial_chat_keep_full_window_admission_and_routing(self):
        model = ast.parse((ROOT / "omni_core/model.py").read_text())
        decoder = next(node for node in model.body if isinstance(node, ast.ClassDef) and node.name == "OmniDecoder")
        forward = next(node for node in decoder.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        self.assertIn("measured_full_forward", [ast.unparse(value) for value in forward.decorator_list])
        body = ast.unparse(forward)
        self.assertIn("maximum_forward_tokens", body); self.assertIn("pager.admit_compute", body)
        connected = ast.parse((ROOT / "omni_core/connected_regions.py").read_text())
        cls = next(node for node in connected.body if isinstance(node, ast.ClassDef) and node.name == "ConnectedOmniDecoder")
        self.assertFalse(any(isinstance(node, ast.FunctionDef) and node.name == "forward" for node in cls.body))
        source = (ROOT / "omni_core/brain.py").read_text()
        brain = ast.parse(source)
        adaptive = next(node for node in brain.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        chat = next(node for node in adaptive.body if isinstance(node, ast.FunctionDef) and node.name == "chat")
        self.assertTrue(any(isinstance(node, ast.Call) and ast.unparse(node.func) == "self.decoder"
            and any(value.arg == "use_global_workspace" and isinstance(value.value, ast.Constant)
                and value.value.value is True for value in node.keywords) for node in ast.walk(chat)))
        native = (ROOT / "omni_core/forward_whole_operation_native.py").read_text()
        tree = ast.parse(native)
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertFalse(any(name.endswith(("set_rng_state", "manual_seed", "seed", "backward")) for name in calls))
        self.assertNotIn("last_generation_neural_state =", native)


if __name__ == "__main__": unittest.main()
