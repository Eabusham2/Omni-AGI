"""NEW scheduling/once-only policy tests: inert owners and fake clocks only.

No brain/module constructors, native kernels, Load, training, codec process or
hardware calibration. These are source-boundary tests, not neural quality proof.
"""
import importlib
import types
import unittest
from unittest.mock import patch

from test_operation_auto_policy import AUTO, Ops, WHOLE

MEDIA = importlib.import_module(AUTO.__package__ + ".media_whole_operation_auto")
REGISTRY = importlib.import_module(AUTO.__package__ + ".operation_auto_registry")
KINDS = ["native.media.image-artifact", "native.media.audio-artifact", "native.media.video-artifact"]
STAGES = ["prepared-media-owner-admission", "home-RNG-and-activity-transfer", "all-native-media-regions",
    "overlap-composition-and-preview", "deterministic-codec-artifact-checksum", "original-owner-restore"]


class MediaOps:
    def __init__(self, host):
        self.host, self.calls, self.released, self.mismatch = host, [], 0, False
        self.enabled, self.post_failure, self.live_failure, self.private_callback = True, False, None, None
    def eligible(self, *_args): return self.enabled
    def capture_operation(self, prepared, producer, previews, session, review):
        identity = {"ownerId": "owned", "ownerSessionId": session, "workKind": KINDS[0], "stageCoverage": STAGES,
            "workloadSha256": WHOLE.digest(prepared["idea"]), "geometrySha256": WHOLE.digest(len(prepared["idea"])),
            "samplingContractSha256": "5"*64, "policySha256": self.host.policy, "topologySha256": "2"*64, "budgetSha256": "4"*64,
            "cache": {"heat": "warm", "logicalStateSha256": "6"*64, "backendInitialization": "warm"}, "nativeState": self.host.state(session)}
        return {"identity": identity, "stratumKey": WHOLE.digest(len(prepared["idea"])), "producer": producer,
            "prepared": prepared, "previews": previews, "released": False,
            "planContract": {"numericGate": "exact-media-neural-timeline-rng-and-encoded-artifact-equality",
                "reuseBasis": "same-validated-media-operator-output-layout-current-owner-policy-inference",
                "mediaLayout": {"modality": "image", "shapeSha256": "7"*64}}}
    def trial(self, op):
        return types.SimpleNamespace(contract=WHOLE.digest(op["prepared"]["idea"]), descriptor={"shape": [len(op["prepared"]["idea"])]})
    def execute_genuine(self, op, preview, directory, cancelled, started):
        self.calls.append("genuine")
        return op["producer"](op["prepared"]["idea"], preview, directory, cancelled, started), self.trial(op)
    def after_genuine(self, *_args):
        if self.post_failure: raise RuntimeError("optional post-result proof unavailable")
    def unchanged(self, op, session): return self.host.state(session) == op["identity"]["nativeState"]
    def operation_state(self, op, session): return self.host.state(session)
    def release_operation(self, op):
        assert not op["released"]
        self.released += 1; op["released"] = True
    def auto_allowed(self, candidate): return True
    def execute_private(self, candidate, op, observer, flags):
        self.calls.append(("private", candidate["executionDevice"]))
        self.host.clock.now += 100 if candidate == self.host.current else 10
        if self.private_callback: self.private_callback(candidate)
        for stage in STAGES: observer.stage_completed(stage)
        result = self.trial(op)
        if self.mismatch and candidate == self.host.other: result.contract = "e"*64
        return result
    def execute_live(self, candidate, op, preview, directory, cancelled, started, begin):
        self.calls.append(("live", candidate["executionDevice"]))
        if self.live_failure == "before": raise RuntimeError("candidate admission declined")
        begin()
        if self.live_failure == "after": raise RuntimeError("actual generation error")
        return op["producer"](op["prepared"]["idea"], preview, directory, cancelled, started), self.trial(op)
    def check_output(self, trial, descriptor): assert trial.descriptor == descriptor
    def observation_resources(self, *args): return self.host.observation_resources(*args)
    def synchronize(self, devices): return devices
    def counters(self): return self.host.counters()


class WholeMediaPolicyCases(unittest.TestCase):
    def test_review_registry_composes_eight_exact_kinds_and_six_media_stages_without_changing_prior_coverage(self):
        self.assertEqual(AUTO.WORK_KINDS, [*REGISTRY.PERCEPTION_KINDS, *REGISTRY.CACHED_KINDS, *KINDS,
            *REGISTRY.TRAINING_KINDS, *REGISTRY.PREFILL_KINDS])
        self.assertEqual(REGISTRY.MEDIA_KINDS, KINDS); self.assertEqual(REGISTRY.MEDIA_STAGES, STAGES)
        coverage = REGISTRY.coverage(AUTO.WORK_KINDS)
        self.assertEqual(len(coverage), 13)
        self.assertEqual(coverage[:3], [{"workKind": kind, "stages": REGISTRY.PERCEPTION_STAGES} for kind in REGISTRY.PERCEPTION_KINDS])
        self.assertEqual(coverage[3:5], [{"workKind": kind, "stages": REGISTRY.CACHED_STAGES} for kind in REGISTRY.CACHED_KINDS])
        self.assertEqual(coverage[5:8], [{"workKind": kind, "stages": STAGES} for kind in KINDS])
        self.assertEqual(coverage[8:9], [{"workKind": kind, "stages": REGISTRY.TRAINING_STAGES} for kind in REGISTRY.TRAINING_KINDS])
        self.assertEqual(coverage[9:], [{"workKind": kind, "stages": REGISTRY.PREFILL_STAGES} for kind in REGISTRY.PREFILL_KINDS])
        self.assertNotIn("whole-imagination-preview-codec-files", AUTO.UNSUPPORTED)
        self.assertIn("opaque-unpinned-or-nondeterministic-media-codec", AUTO.UNSUPPORTED)
        self.assertIn("unfenced-cross-rank-or-complete-training-checkpoint-pipeline", AUTO.UNSUPPORTED)

    def fixture(self, *, reviewed_media=True):
        declaration = patch.dict(REGISTRY.STAGE_COVERAGE, {kind: STAGES for kind in KINDS})
        declaration.start(); self.addCleanup(declaration.stop)
        host = Ops(); host.work_kinds = lambda: [kind for kind in AUTO.WORK_KINDS if reviewed_media or kind not in KINDS]
        host.media_operations = MediaOps(host)
        policy = AUTO.OperationAutoPolicy("owned", host, recorder=WHOLE.WholePathRecorder(clock_ns=host.clock))
        self.addCleanup(policy.close); review = policy.prepare(); policy.arm(operation_id="media-owned", reviewed=review["binding"])
        return policy, host, host.media_operations
    def consume(self, policy, values=None, callback=None):
        values, published = values or [1, 2], []
        def preview(*args): published.append(("preview", args))
        def producer(idea, emit, directory, cancelled, started):
            published.append(("result", list(idea)))
            if emit is not None: emit(1.0, "image/png", b"inert-not-a-real-image", {"fixture": True})
            if callback: callback()
            return {"result": {"pixels": [value*2 for value in idea]}, "previewCount": 1}
        result = policy.consume_media(None, {"modality": "image", "idea": values}, producer, preview, None, lambda: False, 0)
        return result, published
    def finish(self, policy):
        steps = 0
        while getattr(policy, "_media_pending", None) is not None:
            policy.step_private_media(); steps += 1
            self.assertLess(steps, 10)
        return steps

    def test_genuine_delivery_precedes_all_private_work_and_fresh_output_reuses_only_route(self):
        policy, host, ops = self.fixture()
        result, published = self.consume(policy)
        self.assertEqual(result["result"]["pixels"], [2, 4]); self.assertEqual(ops.calls, ["genuine"])
        self.assertEqual([row[0] for row in published], ["result", "preview"])
        self.assertEqual(self.finish(policy), 8)
        self.assertEqual([row[0] for row in published], ["result", "preview"])
        plan = policy.plans()["plans"][0]
        self.assertEqual(plan["qualifiedSamples"], 3); self.assertEqual(plan["numericGate"], "exact-media-neural-timeline-rng-and-encoded-artifact-equality")
        result, published = self.consume(policy, [8, 9]); self.assertEqual(result["result"]["pixels"], [16, 18])
        self.assertEqual(ops.calls[-1], ("live", "mps")); self.assertIsNone(policy._media_pending)
        self.assertFalse(plan["unseenInputOrFrontierWasPreviouslyMeasured"])

    def test_five_kind_review_keeps_original_media_path_without_new_capture_or_calibration(self):
        policy, _host, ops = self.fixture(reviewed_media=False)
        ops.eligible = lambda *_args: self.fail("unreviewed media attempted capture/closure admission")
        result, published = self.consume(policy)
        self.assertEqual(result["result"]["pixels"], [2, 4]); self.assertEqual(ops.calls, [])
        self.assertEqual((len(published), ops.released), (2, 0)); self.assertIsNone(getattr(policy, "_media_pending", None))

    def test_new_request_during_genuine_delivery_prevents_late_pool_and_does_not_cancel_result(self):
        policy, _host, ops = self.fixture()
        result, published = self.consume(policy, callback=policy.retire_private_media)
        self.assertEqual(result["result"]["pixels"], [2, 4]); self.assertIsNone(getattr(policy, "_media_pending", None))
        self.assertEqual((ops.calls, ops.released), (["genuine"], 1)); self.assertEqual(len(published), 2)

    def test_new_user_work_retires_one_pending_pool_without_touching_real_public_work(self):
        policy, _host, ops = self.fixture(); _result, published = self.consume(policy)
        policy.retire_private_media(); self.assertFalse(policy.step_private_media())
        self.assertEqual(ops.calls, ["genuine"]); self.assertEqual(ops.released, 1); self.assertEqual(len(published), 2)

    def test_stop_during_private_work_keeps_original_result_once_and_releases_owners(self):
        policy, _host, ops = self.fixture(); result, published = self.consume(policy)
        ops.private_callback = lambda _candidate: policy.stop(operation_id="media-owned")
        self.assertFalse(policy.step_private_media()); self.assertEqual(len(published), 2)
        self.assertEqual(result["result"]["pixels"], [2, 4]); self.assertEqual(ops.released, 1)
        self.assertEqual(policy.status()["phase"], "cancelled"); self.assertEqual(policy.plans()["plans"], [])

    def test_neural_rng_timeline_or_codec_contract_mismatch_never_forges_qualification(self):
        policy, _host, ops = self.fixture(); self.consume(policy); ops.mismatch = True
        self.assertTrue(policy.step_private_media()); self.assertFalse(policy.step_private_media())
        self.assertEqual(policy.plans()["plans"], []); self.assertEqual(policy.status()["phase"], "failed-fallback")
        self.consume(policy); self.assertEqual(ops.calls.count("genuine"), 1)

    def test_changed_learned_state_or_policy_retires_without_private_execution(self):
        for change in ("epoch", "policy"):
            with self.subTest(change=change):
                policy, host, ops = self.fixture(); self.consume(policy)
                if change == "epoch": host.epoch += 1
                else: host.policy = "9"*64
                self.assertFalse(policy.step_private_media()); self.assertEqual(ops.calls, ["genuine"])
                self.assertEqual(policy.plans()["plans"], []); self.assertEqual(ops.released, 1)

    def test_post_genuine_proof_failure_preserves_already_produced_result_without_replay(self):
        policy, _host, ops = self.fixture(); ops.post_failure = True
        result, published = self.consume(policy)
        self.assertEqual(result["result"]["pixels"], [2, 4]); self.assertEqual(ops.calls, ["genuine"])
        self.assertEqual((len(published), ops.released), (2, 1)); self.assertEqual(policy.plans()["plans"], [])

    def test_admission_failure_falls_back_once_but_real_generation_failure_is_not_replayed(self):
        for stage in ("before", "after"):
            with self.subTest(stage=stage):
                policy, _host, ops = self.fixture(); self.consume(policy); self.finish(policy); ops.live_failure = stage
                if stage == "before":
                    result, published = self.consume(policy, [3, 4]); self.assertEqual(result["result"]["pixels"], [6, 8])
                    self.assertEqual(len(published), 2)
                else:
                    with self.assertRaisesRegex(RuntimeError, "actual generation error"): self.consume(policy, [3, 4])
                self.assertEqual(ops.calls[-1], ("live", "mps"))
