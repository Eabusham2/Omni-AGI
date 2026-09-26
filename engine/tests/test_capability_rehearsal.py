import tempfile
import unittest
from pathlib import Path

import torch


import sys

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.capability_rehearsal import (
    CapabilityRehearsalPolicy,
    _declared_action_candidate_rank,
    _arguments_valid,
    probe_capabilities,
    rehearse_capabilities,
    structural_capability_schemas,
)
from omni_core.ground_up import GROUND_UP_TOOL_TRAJECTORIES


class DeclaredCandidateRetentionTests(unittest.TestCase):
    def test_new_routes_cannot_replace_canonical_anchors_or_no_action(self):
        route_targets = torch.tensor([0, 1])
        all_targets = torch.tensor([0, 1])
        anchor_targets = torch.tensor([0])
        margins = {
            "minimumLanguageThresholdMargin": 0.02,
            "minimumInternalThresholdMargin": 0.06,
            "minimumDeployedThresholdMargin": 0.06,
        }
        baseline = _declared_action_candidate_rank(
            torch.tensor([[0.8, 0.2], [0.7, 0.3]]),
            route_targets,
            torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
            all_targets,
            torch.tensor([[0.8, 0.2]]),
            anchor_targets,
            margins,
            minimum_anchor_correct=1,
            minimum_negative_correct=1,
            negative_count=1,
        )
        more_routes_but_forgot_anchor = _declared_action_candidate_rank(
            torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            route_targets,
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            all_targets,
            torch.tensor([[0.2, 0.8]]),
            anchor_targets,
            margins,
            minimum_anchor_correct=1,
            minimum_negative_correct=1,
            negative_count=1,
        )
        more_routes_but_forgot_no_action = _declared_action_candidate_rank(
            torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            route_targets,
            torch.tensor([[0.9, 0.1], [0.9, 0.1]]),
            all_targets,
            torch.tensor([[0.8, 0.2]]),
            anchor_targets,
            margins,
            minimum_anchor_correct=1,
            minimum_negative_correct=1,
            negative_count=1,
        )
        more_routes_but_forgot_canonical = _declared_action_candidate_rank(
            torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            route_targets,
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            all_targets,
            torch.tensor([[0.8, 0.2]]),
            anchor_targets,
            {**margins, "minimumLanguageThresholdMargin": -0.001},
            minimum_anchor_correct=1,
            minimum_negative_correct=1,
            negative_count=1,
        )
        retained_improvement = _declared_action_candidate_rank(
            torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            route_targets,
            torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            all_targets,
            torch.tensor([[0.8, 0.2]]),
            anchor_targets,
            margins,
            minimum_anchor_correct=1,
            minimum_negative_correct=1,
            negative_count=1,
        )
        self.assertEqual(baseline[0], 1)
        self.assertLess(more_routes_but_forgot_anchor, baseline)
        self.assertLess(more_routes_but_forgot_no_action, baseline)
        self.assertLess(more_routes_but_forgot_canonical, baseline)
        self.assertGreater(retained_improvement, baseline)


class CapabilitySchemaTests(unittest.TestCase):
    def test_frozen_collection_arguments_preserve_json_schema_types(self):
        schemas = structural_capability_schemas()
        by_id = {str(schema["id"]): schema for schema in schemas}

        self.assertEqual(
            by_id["browser.automation"]["inputSchema"]["properties"][
                "steps"
            ]["type"],
            "array",
        )
        self.assertEqual(
            by_id["device.input"]["inputSchema"]["properties"][
                "modifiers"
            ]["type"],
            "array",
        )
        for trajectory in GROUND_UP_TOOL_TRAJECTORIES:
            with self.subTest(
                tool=trajectory["toolId"], action=trajectory["action"]
            ):
                self.assertTrue(
                    _arguments_valid(
                        schemas,
                        str(trajectory["toolId"]),
                        dict(trajectory.get("arguments", {})),
                    )
                )


class CapabilityRehearsalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(701)
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="omni-capability-rehearsal-"
        )
        cls.brain = AdaptiveBrain.create(
            "capability-retention",
            Path(cls.temporary.name) / "brain",
            OmniConfig.micro(
                origin_kind="ground-up",
                max_seq_len=16,
                train_batch_size=1,
                gradient_accumulation=1,
            ),
            initialize_ground_up=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.brain.events.close()
        cls.temporary.cleanup()

    def test_structural_probe_has_no_description_or_system_prompt(self):
        report = probe_capabilities(self.brain)
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["correct"], 8)
        self.assertTrue(report["structuralSchemasOnly"])
        self.assertFalse(report["toolDescriptionProse"])
        self.assertFalse(report["systemPrompt"])
        self.assertFalse(report["rewardModel"])
        self.assertFalse(report["rlhf"])
        self.assertTrue(
            all(record["schemaValidArguments"] for record in report["records"])
        )
        all_routes = report["allToolTrajectories"]
        self.assertTrue(all_routes["passed"])
        self.assertEqual(all_routes["correctRoutes"], all_routes["routeCount"])
        self.assertEqual(all_routes["routeCount"], 27)
        self.assertEqual(
            all_routes["exactMaterializedRoutes"]
            + all_routes["stateDependentDeferredRoutes"],
            27,
        )
        self.assertEqual(
            all_routes["negativeNoActionCount"], all_routes["negativeCount"]
        )
        self.assertEqual(all_routes["negativeCount"], 6)

    def test_ground_up_action_retention_requires_authenticated_origin_and_manifest(self):
        self.assertTrue(self.brain._ground_up_action_origin_verified)
        self.assertTrue(self.brain._can_retain_bundled_action_policy())
        manifest = self.brain.ground_up_training_manifest
        assert manifest is not None
        original_sha = manifest["sha256"]
        try:
            manifest["sha256"] = "0" * 64
            self.assertFalse(self.brain._can_retain_bundled_action_policy())
        finally:
            manifest["sha256"] = original_sha
        self.brain.save()
        reloaded = AdaptiveBrain.load(
            self.brain.storage_path, self.brain.brain_id
        )
        try:
            self.assertTrue(reloaded._ground_up_action_origin_verified)
            self.assertTrue(reloaded._can_retain_bundled_action_policy())
        finally:
            reloaded.close()

    def test_late_catastrophic_action_forgetting_is_repaired_before_promotion(self):
        baseline = probe_capabilities(self.brain)["minimumExpectedProbability"]
        snapshot = {
            "language": {
                key: value.detach().clone()
                for key, value in self.brain.decoder.action_policy.state_dict().items()
            },
            "internal": {
                key: value.detach().clone()
                for key, value in self.brain.decoder.internal_action_policy.state_dict().items()
            },
        }
        try:
            with torch.no_grad():
                for parameter in (
                    *self.brain.decoder.action_policy.parameters(),
                    *self.brain.decoder.internal_action_policy.parameters(),
                ):
                    parameter.zero_()
            forgotten = probe_capabilities(self.brain)
            self.assertFalse(forgotten["passed"])
            self.assertLess(forgotten["correct"], forgotten["probeCount"])
            self.assertTrue(
                all(
                    record["rawDeployedKind"] == "talk"
                    for record in forgotten["records"]
                )
            )
            self.assertTrue(
                any(
                    record["correct"]
                    and record.get("supportEvidence", {}).get("kind")
                    == "authenticated-ground-up-route-assembly"
                    for record in forgotten["records"]
                )
            )

            receipt = rehearse_capabilities(
                self.brain,
                phase="final",
                committed_global_waves=9,
                policy=CapabilityRehearsalPolicy(periodic_global_waves=3),
                baseline_minimum_probability=baseline,
            )
            self.assertTrue(receipt["after"]["passed"])
            self.assertEqual(receipt["after"]["correct"], 8)
            self.assertTrue(receipt["regressionGatePassed"])
            self.assertTrue(receipt["actionHeadReinitializedFromGroundUpSeed"])
            self.assertEqual(receipt["recordsVisited"], receipt["expectedRecords"])
            self.assertFalse(receipt["systemPrompt"])
            self.assertFalse(receipt["rlhf"])
        finally:
            self.brain.decoder.action_policy.load_state_dict(snapshot["language"])
            self.brain.decoder.internal_action_policy.load_state_dict(snapshot["internal"])


if __name__ == "__main__":
    unittest.main()
