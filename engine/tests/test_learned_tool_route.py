"""Small route-head checks; deliberately do not run a production Build."""

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.capability_rehearsal import structural_capability_schemas
from omni_core.config import OmniConfig
from omni_core.ground_up import GROUND_UP_TOOL_NEGATIVE_EXAMPLES, GROUND_UP_TOOL_TRAJECTORIES
from omni_core.model import OmniDecoder, ToolRouteHead, pack_ternary_weight
from omni_core.persistence import atomic_save_tensors, load_tensors
from omni_core.ternary_packing import collect_module_ternary_tensors


SCHEMAS = [{
    "id": "system.files", "actions": ["read", "list", "write"], "grant": "ask",
    "inputSchema": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"},
    }, "required": ["path"]},
}]


def tiny_brain(*, with_memory=True):
    brain = object.__new__(AdaptiveBrain)
    brain.config = SimpleNamespace(origin_kind="ground-up")
    brain.decoder = SimpleNamespace(tool_route_head=ToolRouteHead())
    brain.device = torch.device("cpu")
    brain.counters = {"training_steps": 0}
    if with_memory:
        def vector_for_text(text):
            # Opaque, deterministic source-only neural fixture. It is neither
            # an answer table nor the route head's hashed-ngram input.
            seed = int.from_bytes(
                hashlib.sha256(text.encode("utf-8")).digest()[:8], "little"
            )
            generator = torch.Generator().manual_seed(seed)
            return torch.randn(64, generator=generator)
        brain.memory = SimpleNamespace(vector_for_text=vector_for_text)
        brain.memory_bridge = torch.nn.Identity()
    return brain


def neural_state(brain, text):
    return brain._idea_model_vector(brain.memory.vector_for_text(text))


@torch.no_grad()
def bias_packed_internal_query_toward(head: ToolRouteHead, route_index: int) -> None:
    """Change only authoritative ternary bytes to favor one learned route."""

    candidate = head.candidate(head.route_features[route_index][None])
    levels = torch.sign(candidate).to(torch.int8)
    head.internal_query.fill_ternary_(0)
    packed_bias = head.internal_query._packed_forward_bias
    assert packed_bias is not None
    packed_bias.copy_(pack_ternary_weight(levels))
    head.internal_query.packed_forward_weight()  # Validate the exact codes.


class ActionInputSchemaBoundaryTests(unittest.TestCase):
    """Schema-only source fixtures; no model creation, training, or Build."""

    def test_builtin_selected_action_schema_survives_normalization(self):
        schemas = AdaptiveBrain._normalize_tool_schemas([{
            "id": "system.files", "actions": ["read", "write"], "grant": "ask",
            "actionInputSchemas": {
                "read": {"type": "object", "properties": {
                    "path": {"type": "string", "description": "NO PROSE"},
                    "maxBytes": {"type": "number"},
                }, "required": ["path"]},
                "write": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"},
                }, "required": ["path", "content"]},
            },
        }])
        self.assertNotIn("inputSchema", schemas[0])
        self.assertEqual(schemas[0]["actionInputSchemas"]["read"]["properties"]["path"], {"type": "string"})
        validate = AdaptiveBrain._materialized_tool_action_matches_schema
        self.assertTrue(validate(schemas, {"toolId": "system.files", "action": "read", "arguments": {"path": "/tmp/a"}}))
        self.assertFalse(validate(schemas, {"toolId": "system.files", "action": "write", "arguments": {"path": "/tmp/a"}}))
        self.assertTrue(validate(schemas, {"toolId": "system.files", "action": "write", "arguments": {"path": "/tmp/a", "content": "hello"}}))
        brain = object.__new__(AdaptiveBrain)
        self.assertEqual(brain._literal_tool_route_arguments(
            "system.files", "write", '{"path":"/tmp/a","content":"hello"}', schemas[0],
        ), {"path": "/tmp/a", "content": "hello"})

    def test_missing_action_schema_fails_closed_and_mcp_keeps_single_schema(self):
        shared = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
        schemas = AdaptiveBrain._normalize_tool_schemas([{
            "id": "system.files", "actions": ["read", "write"], "grant": "ask",
            "inputSchema": shared, "actionInputSchemas": {"read": shared},
        }, {
            "id": "mcp.example.read", "actions": ["call"], "grant": "ask",
            "inputSchema": shared, "actionInputSchemas": {"call": {"properties": {}}},
        }])
        by_id = {value["id"]: value for value in schemas}
        self.assertIsNone(AdaptiveBrain._tool_action_input_schema(by_id["system.files"], "write"))
        self.assertNotIn("actionInputSchemas", by_id["mcp.example.read"])
        self.assertTrue(AdaptiveBrain._materialized_tool_action_matches_schema(
            schemas, {"toolId": "mcp.example.read", "action": "call", "arguments": {"path": "/tmp/a"}},
        ))


class LearnedToolRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        cls.trained = tiny_brain()
        cls.training = cls.trained._train_tool_route_head(
            GROUND_UP_TOOL_TRAJECTORIES,
            negative_examples=GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
        )

    def setUp(self):
        self.brain = tiny_brain()
        self.brain.decoder.tool_route_head.load_state_dict(
            copy.deepcopy(self.trained.decoder.tool_route_head.state_dict())
        )

    def test_fails_closed_before_training_even_with_exact_tool_words(self):
        brain = tiny_brain()
        self.assertFalse(hasattr(brain, "_materialize_legacy_tool_action"))
        selected = brain._materialize_generic_tool_action(
            schemas=SCHEMAS, input_text="read /tmp/file.txt", assembly_ids=(), organic_state={},
            neural_state=neural_state(brain, "read /tmp/file.txt"),
        )
        self.assertIsNone(selected)
        self.assertEqual(brain._last_tool_route_evidence["reason"], "untrained-internal-route")

    def test_all_27_authored_targets_and_negatives_receive_supervised_loss(self):
        self.assertTrue(self.training["ready"])
        self.assertEqual(self.training["routeCount"], 27)
        self.assertEqual(self.training["records"], 33)
        self.assertFalse(self.training["generalizationVerified"])
        self.assertGreaterEqual(
            self.training["minimumInternalTrainingTargetProbability"], 0.70
        )
        head = self.brain.decoder.tool_route_head
        head.eval()
        schemas = structural_capability_schemas()
        for example in GROUND_UP_TOOL_TRAJECTORIES:
            with self.subTest(route=(example["toolId"], example["action"])):
                actual = head.select_internal(
                    neural_state(self.brain, example["utterance"]), schemas
                )
                self.assertEqual(actual["selected"], {"toolId": example["toolId"], "action": example["action"]})
                self.assertFalse(actual["sourceTextLookup"])
        for example in GROUND_UP_TOOL_NEGATIVE_EXAMPLES:
            self.assertIsNone(head.select_internal(
                neural_state(self.brain, example["utterance"]), schemas
            )["selected"])

    def test_build_readiness_depends_only_on_internal_neural_head(self):
        brain = tiny_brain()
        example = GROUND_UP_TOOL_TRAJECTORIES[0]
        head = brain.decoder.tool_route_head
        with patch.object(
            head.query, "forward",
            side_effect=AssertionError("text query must not train"),
        ):
            result = brain._train_tool_route_head([example], maximum_steps=24)
        self.assertTrue(result["ready"])
        self.assertEqual(int(head.training_steps.item()), 0)
        self.assertGreaterEqual(
            result["minimumInternalTrainingTargetProbability"], 0.70
        )

    def test_weak_internal_head_cannot_claim_build_readiness(self):
        brain = tiny_brain()
        example = GROUND_UP_TOOL_TRAJECTORIES[0]
        head = brain.decoder.tool_route_head
        def uniform_internal(state):
            return (
                torch.zeros((state.shape[0], 2), device=state.device)
                + head.internal_query._autograd_trigger * 0
            )
        with patch.object(head, "forward_internal", side_effect=uniform_internal):
            result = brain._train_tool_route_head([example], maximum_steps=2)
        self.assertEqual(result["minimumInternalTrainingTargetProbability"], 0.5)
        self.assertFalse(result["ready"])

    def test_missing_neural_state_cannot_fall_back_to_text_route(self):
        brain = self.brain
        selected = brain._materialize_generic_tool_action(
            schemas=SCHEMAS, input_text="read /tmp/file.txt",
            assembly_ids=(), organic_state={},
        )
        self.assertIsNone(selected)
        self.assertEqual(
            brain._last_tool_route_evidence["reason"],
            "missing-active-neural-state",
        )

    def test_build_tool_curriculum_calls_supervised_route_training(self):
        brain = tiny_brain()
        brain.memory.learn = Mock()
        brain.parameter_checksum = Mock(return_value="tiny-checksum")
        brain.learn_experience = Mock()
        with patch.object(brain, "_train_tool_route_head", wraps=brain._train_tool_route_head) as train:
            report = brain._train_starter_tool_curriculum(
                trajectories=GROUND_UP_TOOL_TRAJECTORIES,
                negative_examples=GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
                files_tool_id="system.files",
            )
        train.assert_called_once_with(
            GROUND_UP_TOOL_TRAJECTORIES, negative_examples=GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
        )
        self.assertTrue(report["routeHeadTraining"]["ready"])
        self.assertEqual(report["routeHeadTraining"]["objective"],
                         "typed-internal-route-cross-entropy")

    def test_changed_neural_weights_change_winner_without_changing_input(self):
        brain = self.brain
        head = brain.decoder.tool_route_head
        text = "read /tmp/explicit.txt"
        state = neural_state(brain, text)
        index = head.route_index("system.files", "list")
        self.assertIsNotNone(index)
        bias_packed_internal_query_toward(head, index)
        self.assertFalse(hasattr(brain, "_materialize_legacy_tool_action"))
        after = brain._materialize_generic_tool_action(
            schemas=SCHEMAS, input_text=text, assembly_ids=(), organic_state={},
            neural_state=state,
        )
        self.assertEqual(after["action"], "list")
        self.assertEqual(after["arguments"]["path"], "/tmp/explicit.txt")
        self.assertEqual(after["routeEvidence"]["reason"], "trained-internal-weights")

    def test_missing_arguments_do_not_fall_back_or_get_fabricated(self):
        head = self.brain.decoder.tool_route_head
        index = head.route_index("system.files", "write")
        self.assertIsNotNone(index)
        bias_packed_internal_query_toward(head, index)
        actual = self.brain._materialize_generic_tool_action(
            schemas=SCHEMAS, input_text="read /tmp/file.txt", assembly_ids=(), organic_state={},
            neural_state=neural_state(self.brain, "read /tmp/file.txt"),
        )
        self.assertIsNone(actual)
        self.assertEqual(self.brain._last_tool_route_evidence["selected"]["action"], "write")
        self.assertEqual(self.brain._last_tool_route_evidence["materialization"], "missing-explicit-arguments")

    def test_off_and_unknown_schema_candidates_cannot_be_chosen(self):
        example = GROUND_UP_TOOL_TRAJECTORIES[0]
        head = self.brain.decoder.tool_route_head
        state = neural_state(self.brain, example["utterance"])
        self.assertIsNone(head.select_internal(
            state, [{**SCHEMAS[0], "grant": "off"}]
        )["selected"])
        self.assertIsNone(head.select_internal(
            state, [{"id": "mcp.untrained", "actions": ["call"]}]
        )["selected"])

    def test_online_host_episode_updates_and_is_idempotent_across_checkpoint(self):
        brain = self.brain
        example = GROUND_UP_TOOL_TRAJECTORIES[0]
        params = dict(utterance=example["utterance"], tool_id=example["toolId"],
                      action=example["action"], outcome="success", source="host-tool-outcome", event_id="receipt-123")
        updated = brain.learn_tool_route_experience(**params)
        self.assertTrue(updated["applied"])
        head = brain.decoder.tool_route_head
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.safetensors"
            atomic_save_tensors(path, head.state_dict())
            loaded = tiny_brain()
            loaded.decoder.tool_route_head.load_state_dict(load_tensors(path))
        duplicate = loaded.learn_tool_route_experience(**params)
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["steps"], 0)
        self.assertEqual(int(loaded.decoder.tool_route_head.experience_updates), 1)

    def test_untrusted_or_failed_outcomes_do_not_train(self):
        head = self.brain.decoder.tool_route_head
        before = copy.deepcopy(head.state_dict())
        for source, outcome in (("self", "success"), ("host-tool-outcome", "error"), ("host-tool-outcome", "permission-denied")):
            result = self.brain.learn_tool_route_experience(
                utterance="read /tmp/file.txt", tool_id="system.files", action="read", source=source, outcome=outcome,
            )
            self.assertFalse(result["processed"])
        for key, value in before.items():
            self.assertTrue(torch.equal(value, head.state_dict()[key]))

    def test_default_native_route_and_argument_heads_are_ternary_checkpointed(self):
        native = OmniDecoder(OmniConfig.micro())
        state = native.state_dict()
        self.assertIn("tool_route_head.query._packed_forward_weight", state)
        self.assertIn("tool_route_head.internal_query._packed_forward_weight", state)
        self.assertIn("action_argument_head.condition._packed_forward_weight", state)
        self.assertIn("action_argument_head.output._packed_forward_weight", state)
        self.assertFalse(any(key.endswith(".weight") for key in state))
        tensors = collect_module_ternary_tensors(native)
        self.assertTrue(any("tool_route_head.query" in spec.name for spec in tensors))
        self.assertTrue(any("tool_route_head.candidate" in spec.name for spec in tensors))
        self.assertTrue(any("action_argument_head.condition" in spec.name for spec in tensors))
        self.assertTrue(any("action_argument_head.output" in spec.name for spec in tensors))


if __name__ == "__main__":
    unittest.main()
