import unittest
from pathlib import Path

import torch


import sys

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.capability_rehearsal import (
    CapabilityRehearsalPolicy,
    _declared_action_candidate_rank,
    _arguments_valid,
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
    """Real packed production heads on tiny fixed neural-feature inputs.

    This tests the replay transaction, not a trained brain's tool competence.
    """
    def setUp(self):
        from types import SimpleNamespace
        from omni_core.model import ActionPolicyHead, ToolRouteHead, ActionArgumentHead, ACTION_KINDS
        from omni_core.ground_up import GROUND_UP_ACTION_EXAMPLES, ground_up_v3_curriculum_manifest
        torch.manual_seed(701)
        torch.set_num_threads(1)
        decoder = torch.nn.Module()
        decoder.action_policy = ActionPolicyHead(8)
        decoder.internal_action_policy = ActionPolicyHead(8)
        decoder.tool_route_head = ToolRouteHead(8)
        decoder.action_argument_head = ActionArgumentHead(8)
        for row in GROUND_UP_TOOL_TRAJECTORIES:
            decoder.tool_route_head.register_route(row["toolId"], row["action"])
        features = torch.randn((len(GROUND_UP_ACTION_EXAMPLES), 8))
        targets = torch.tensor([ACTION_KINDS.index(kind) for _text, kind in GROUND_UP_ACTION_EXAMPLES])
        self.released = 0
        self.reserved = []
        def reserve(size, purpose):
            self.reserved.append((size, purpose))
            def release():
                self.released += 1
            return SimpleNamespace(release=release)
        def vector(text):
            import hashlib
            return torch.tensor(list(hashlib.sha256(text.encode()).digest())[:8], dtype=torch.float32) / 127.5 - 1
        def forbidden_episode(*args, **kwargs):
            raise AssertionError("rehearsal must not forge an ordinary episodic observation")
        self.brain = SimpleNamespace(
            decoder=decoder, device=torch.device("cpu"),
            config=SimpleNamespace(origin_kind="ground-up", max_seq_len=64),
            ground_up_training_manifest=ground_up_v3_curriculum_manifest(),
            counters={"training_steps": 7, "metaplastic_updates": 0,
                "action_retention_checks": 0, "action_retention_replays": 0},
            resource_policy=SimpleNamespace(reserve_ram=reserve),
            memory=SimpleNamespace(vector_for_text=vector, learn=forbidden_episode),
            _idea_model_vector=lambda value: value.reshape(1, -1),
            _starter_action_features=lambda: (features.clone(), features.clone(), targets.clone()),
            _normalize_tool_schemas=lambda schemas: schemas,
            ingestion_checkpoints={"source": {"nextRecordIndex": 3, "window": 2}},
            tool_permissions={"system.files": "off"},
        )
        self.policy = CapabilityRehearsalPolicy(maximum_action_steps=2, minimum_action_steps=1)
        self.assertEqual(tuple(decoder.parameters()), ())
        self.assertFalse(hasattr(decoder.action_policy.norm, "scale"))
        self.assertFalse(hasattr(decoder.action_policy.hidden, "weight"))

    def state(self):
        return {name: value.clone() for name, value in self.brain.decoder.state_dict().items()}

    def assert_state(self, expected):
        observed = self.brain.decoder.state_dict()
        self.assertEqual(set(observed), set(expected))
        for name, value in expected.items():
            self.assertTrue(torch.equal(observed[name], value), name)

    def test_declared_native_replay_measures_candidate_without_quiz_or_seed_reset(self):
        from unittest.mock import patch
        from omni_core import vsa
        now = vsa._now
        before = self.state()
        with patch("omni_core.capability_rehearsal.probe_capabilities", side_effect=AssertionError("no quiz")), \
             patch("torch.nn.init.xavier_uniform_", side_effect=AssertionError("no reset")):
            receipt = rehearse_capabilities(self.brain, phase="middle", committed_global_waves=3, policy=self.policy)
        self.assertEqual(receipt["action"]["attemptedSteps"], 2)
        self.assertNotEqual(receipt["ownerChecksumBefore"], receipt["ownerChecksumCandidate"])
        self.assertFalse(receipt["readinessProbeExecuted"])
        self.assertFalse(receipt["capabilityOutcomeVerified"])
        self.assertFalse(receipt["actionHeadReinitializedFromGroundUpSeed"])
        self.assertEqual(receipt["episodicMemoryRecordsAdded"], 0)
        self.assertEqual(receipt["candidateOwners"], ["action_policy", "internal_action_policy", "tool_route_head", "action_argument_head"])
        self.assertGreater(receipt["before"]["argumentRecords"], 0)
        self.assertEqual(len(receipt["beforePerRecord"]["actionCorrect"]), receipt["before"]["actionRecords"])
        self.assertNotIn("passed", receipt["after"])
        self.assertIs(vsa._now, now)
        self.assertEqual(self.brain.ingestion_checkpoints, {"source": {"nextRecordIndex": 3, "window": 2}})
        self.assertEqual(self.brain.tool_permissions, {"system.files": "off"})
        self.assertEqual(int(self.brain.decoder.action_argument_head.grounded_steps), 0)
        self.assertEqual(self.released, len(self.reserved))
        self.assertGreater(self.reserved[0][0], 0)
        if receipt["rollbackVerified"]:
            self.assert_state(before)

    def test_historical_probability_shortfall_is_not_overridden_or_a_blocking_quiz(self):
        from omni_core.capability_rehearsal import CapabilityScheduleState, advance_schedule_state
        before = self.state()
        receipt = rehearse_capabilities(self.brain, phase="final", committed_global_waves=9,
            policy=self.policy, baseline_minimum_probability=1.0)
        self.assertFalse(receipt["candidateAccepted"])
        self.assertFalse(receipt["candidateBaselineRetentionPassed"])
        self.assertFalse(receipt["baselineRetentionMet"])
        self.assertFalse(receipt["regressionGatePassed"])
        self.assertTrue(receipt["rollbackVerified"])
        self.assertEqual(receipt["ownerChecksumAfter"], receipt["ownerChecksumBefore"])
        self.assertEqual(receipt["parameterSteps"], 0)
        self.assert_state(before)
        state = advance_schedule_state(CapabilityScheduleState(start_completed=True), receipt)
        self.assertTrue(state.final_completed)
        self.assertEqual(state.event_count, 1)

    def test_mid_replay_failure_restores_native_contacts_counters_pending_events_and_modes(self):
        before = self.state()
        counters = dict(self.brain.counters)
        projection = self.brain.decoder.action_policy.projection
        projection._pending_stability_events = 11
        self.brain.decoder.train()
        self.brain.decoder.action_policy.hidden.eval()
        modes = [module.training for module in self.brain.decoder.modules()]
        boundaries = 0
        def boundary(_label):
            nonlocal boundaries
            boundaries += 1
            if boundaries == 2:
                projection.add_parallel_connection_(0, 0, sign=1)
                projection._pending_stability_events = 99
                self.brain.counters["uncommitted"] = 1
                raise RuntimeError("injected native rehearsal boundary")
        self.brain._check_neural_learning_boundary = boundary
        with self.assertRaisesRegex(RuntimeError, "injected native rehearsal boundary"):
            rehearse_capabilities(self.brain, phase="middle", committed_global_waves=5, policy=self.policy)
        self.assert_state(before)
        self.assertEqual(self.brain.counters, counters)
        self.assertEqual(projection._pending_stability_events, 11)
        self.assertEqual([module.training for module in self.brain.decoder.modules()], modes)
        self.assertEqual(self.released, len(self.reserved))

    def test_reservation_refusal_happens_before_any_native_mutation(self):
        before = self.state()
        def refuse(*args):
            raise RuntimeError("rollback reserve refused")
        self.brain.resource_policy.reserve_ram = refuse
        with self.assertRaisesRegex(RuntimeError, "rollback reserve refused"):
            rehearse_capabilities(self.brain, phase="start", committed_global_waves=0, policy=self.policy)
        self.assert_state(before)

    def test_maintenance_protects_grounded_and_experienced_native_owners_only(self):
        from omni_core.model import PackedAdaptiveBitLinear, packed_online_step
        route = self.brain.decoder.tool_route_head
        argument = self.brain.decoder.action_argument_head
        route.experience_updates.fill_(3)
        argument.grounded_steps.fill_(2)
        route.register_route("fixture.private", "saved")
        index = argument.register_route("fixture.private", "saved")
        argument.grounded_route_updates[index] = 2
        route.internal_query._pending_stability_events = 5
        argument.output._pending_stability_events = 7
        protected = {name: {key: value.clone() for key, value in head.state_dict().items()}
            for name, head in (("tool_route_head", route), ("action_argument_head", argument))}
        handles = []
        def assert_readonly(module, _inputs):
            self.assertFalse(module.training, "generic maintenance attempted to train a private owner")
        for head in (route, argument):
            for module in head.modules():
                if isinstance(module, PackedAdaptiveBitLinear):
                    handles.append(module.register_forward_pre_hook(assert_readonly))
        try:
            receipt = rehearse_capabilities(self.brain, phase="middle", committed_global_waves=4, policy=self.policy)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(set(receipt["protectedOwners"]), set(protected))
        self.assertEqual(receipt["candidateOwners"], ["action_policy", "internal_action_policy"])
        self.assertEqual(receipt["argumentStepsAttempted"], 0)
        self.assertFalse(receipt["ordinarySchemaAndExperienceLearningChanged"])
        self.assertEqual(route.internal_query._pending_stability_events, 5)
        self.assertEqual(argument.output._pending_stability_events, 7)
        for name, expected in protected.items():
            observed = getattr(self.brain.decoder, name).state_dict()
            self.assertEqual(set(observed), set(expected))
            self.assertTrue(all(torch.equal(observed[key], value) for key, value in expected.items()))
        # The policy is temporary, not a freeze installed on these owners:
        # a later actual-schema teaching call still registers and learns.
        argument.register_route("fixture.new", "added")
        argument.train()
        before = {key: value.clone() for key, value in argument.state_dict().items()}
        encoded_schema = argument.schema_features("fixture.new", "added", {
            "id": "fixture.new", "actions": ["added"], "inputSchema": {
                "type": "object", "properties": {"value": {"type": "string"}},
                "required": ["value"], "additionalProperties": False,
            },
        })
        self.assertIsNotNone(encoded_schema)
        features = encoded_schema[None]
        with packed_online_step((argument,)):
            argument.supervised_loss(torch.ones((1, 8)), features, {"value": "new"}).backward()
        self.assertTrue(any(not torch.equal(argument.state_dict()[key], value) for key, value in before.items()))


if __name__ == "__main__":
    unittest.main()
