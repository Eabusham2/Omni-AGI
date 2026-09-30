"""Protocol/method fixtures only: no brain/model constructor or training."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.model import ACTION_KINDS, ActionArgumentHead, ToolRouteHead
from omni_core.native_action_protocol import (
    NativeActionEmissionLedger,
    observed_argument_schema,
    structural_schema,
    validate_structural_value,
)


class ZeroState(nn.Module):
    def forward(self, values):
        return torch.zeros((values.shape[0], 4), device=values.device)


class FixtureByteLogits(nn.Module):
    """Programmed logits test transport; they are not learned intelligence."""
    def __init__(self, payload):
        super().__init__()
        self.bytes = [*json.dumps(payload).encode(), 256]
        self.position = 0

    def forward(self, values):
        logits = torch.full((values.shape[0], 257), -30.0)
        logits[:, self.bytes[self.position]] = 30.0
        self.position += 1
        return logits


class ResetFixtureState(ZeroState):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, values):
        self.output.position = 0
        return super().forward(values)


def argument_method_fixture(payload, tool="mcp.fixture", action="call", trained=1):
    head = ActionArgumentHead.__new__(ActionArgumentHead)
    nn.Module.__init__(head)
    head.dimensions = 4
    head.transition = ZeroState()
    head.token_embedding = ZeroState()
    head.output = FixtureByteLogits(payload)
    head.condition = ResetFixtureState(head.output)
    head.register_buffer("training_steps", torch.tensor(trained))
    head.register_buffer("grounded_steps", torch.tensor(0))
    head.register_buffer("route_keys", torch.tensor([list(ToolRouteHead.identity(tool, action))], dtype=torch.uint8))
    head.register_buffer("grounded_route_updates", torch.tensor([0]))
    return head


class NativeActionProtocolFixtures(unittest.TestCase):
    def test_recursive_schema_discards_prose_not_nested_structure(self):
        schema = structural_schema({
            "type": "object", "description": "not a model instruction", "required": ["steps"],
            "properties": {"steps": {"type": "array", "items": {"type": "object", "properties": {
                "kind": {"enum": ["click", "extract"]}, "selector": {"type": "string"}
            }, "required": ["kind"]}}},
        })
        self.assertNotIn("description", json.dumps(schema))
        self.assertTrue(validate_structural_value({"steps": [{"kind": "click", "selector": "#a"}, {"kind": "extract"}]}, schema))
        self.assertFalse(validate_structural_value({"steps": [{"kind": "delete"}]}, schema))
        self.assertFalse(validate_structural_value({"steps": [{"kind": "click", "selector": 4}]}, schema))
        self.assertFalse(validate_structural_value({"steps": [], "ungrantedField": 1}, schema))

    def test_local_refs_unions_and_numeric_guards_are_not_weakened(self):
        schema = structural_schema({
            "type": "object", "$defs": {"value": {"type": ["integer", "null"], "minimum": 2}},
            "properties": {"count": {"$ref": "#/$defs/value"}}, "required": ["count"],
        })
        self.assertTrue(validate_structural_value({"count": 3}, schema))
        self.assertTrue(validate_structural_value({"count": None}, schema))
        self.assertFalse(validate_structural_value({"count": True}, schema))
        self.assertFalse(validate_structural_value({"count": 1}, schema))
        self.assertFalse(validate_structural_value({"count": 10 ** 1000}, schema))
        unsupported = structural_schema({"type": "string", "pattern": "unhandled-assertion"})
        self.assertFalse(validate_structural_value("anything", unsupported))

    def test_schema_condition_changes_with_nested_types_not_description(self):
        schema = {"id": "mcp.fixture", "actions": ["call"], "inputSchema": {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}}}
        first = ActionArgumentHead.schema_features("mcp.fixture", "call", schema)
        schema["inputSchema"]["description"] = "prose never controls the native model"
        second = ActionArgumentHead.schema_features("mcp.fixture", "call", schema)
        self.assertTrue(torch.equal(first, second))
        schema["inputSchema"]["properties"]["items"]["items"]["type"] = "integer"
        self.assertFalse(torch.equal(first, ActionArgumentHead.schema_features("mcp.fixture", "call", schema)))

    def test_declared_trained_arguments_can_decode_without_a_host_success_quiz(self):
        head = argument_method_fixture({"query": "native fixture"})
        schema = structural_schema({"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})
        result = head.decode(torch.zeros(1, 4), torch.zeros(1, ToolRouteHead.feature_width), tool_id="mcp.fixture", action="call", input_schema=schema)
        self.assertEqual(result["arguments"], {"query": "native fixture"})
        self.assertEqual(result["trainingProvenance"], "declared-native-typed-trajectory")
        self.assertFalse(result["hostCapabilityVerified"])
        self.assertFalse(result["generalizationVerified"])
        untrained = argument_method_fixture({"query": "native fixture"}, trained=0)
        self.assertIsNone(untrained.decode(torch.zeros(1, 4), torch.zeros(1, ToolRouteHead.feature_width), tool_id="mcp.fixture", action="call")["arguments"])

    def test_generic_materialization_uses_same_head_for_natural_multistep_browser_arguments(self):
        payload = {"url": "https://example.org", "steps": [{"kind": "click", "selector": "#docs"}, {"kind": "extract", "selector": "main"}]}
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.device = torch.device("cpu")
        brain.config = SimpleNamespace(max_seq_len=8192)
        brain.decoder = SimpleNamespace(
            action_argument_head=argument_method_fixture(payload, "browser.automation", "task"),
            tool_route_head=SimpleNamespace(select_internal=lambda *_: {"selected": {"toolId": "browser.automation", "action": "task"}}),
        )
        schemas = [{"id": "browser.automation", "actions": ["task"], "grant": "ask", "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}, "steps": {"type": "array"}}, "required": ["url", "steps"]}}]
        result = brain._materialize_generic_tool_action(
            schemas=schemas, input_text="Please inspect the documentation and collect the main section.",
            assembly_ids=[], organic_state={}, neural_state=torch.zeros(1, 4),
        )
        self.assertEqual(result["arguments"], payload)
        self.assertEqual(result["routeEvidence"]["materialization"], "same-cortex-learned-json-schema-validated")
        self.assertTrue(AdaptiveBrain._materialized_tool_action_matches_schema(schemas, result))

    def test_emission_identity_is_stable_and_never_replays_across_turns(self):
        action = {"kind": "tool", "toolId": "web.search", "action": "search", "arguments": {"query": "same native state", "assemblyIds": ["a"], "organic": True}}
        first, replay = NativeActionEmissionLedger("turn-one"), NativeActionEmissionLedger("turn-one")
        initial = first.register(action, step=0, phase="pre-speech")
        changed_metadata = {**action, "arguments": {**action["arguments"], "assemblyIds": ["b"]}}
        self.assertIsNone(first.register(changed_metadata, step=12, phase="mid-generation"))
        different_arguments = {**action, "arguments": {"query": "a distinct native decision"}}
        self.assertIsNotNone(first.register(different_arguments, step=13, phase="mid-generation"))
        self.assertEqual(initial["actionId"], replay.register(action, step=0, phase="pre-speech")["actionId"])
        self.assertNotEqual(initial["actionId"], NativeActionEmissionLedger("turn-two").register(action, step=0, phase="pre-speech")["actionId"])

    def test_observed_schema_has_no_argument_literal_answers(self):
        value = {"query": "not a lookup answer", "steps": [{"kind": "click", "selector": "#private-source"}, {"milliseconds": 3}]}
        schema = observed_argument_schema(value)
        self.assertNotIn("not a lookup answer", json.dumps(schema))
        self.assertNotIn("#private-source", json.dumps(schema))
        self.assertTrue(validate_structural_value(value, schema))

    def test_unseen_schema_candidates_use_the_same_projection_without_registering_answers(self):
        head = ToolRouteHead.__new__(ToolRouteHead)
        nn.Module.__init__(head)
        head.internal_query = nn.Identity()
        head.candidate = nn.Identity()
        head.register_buffer("internal_training_steps", torch.tensor(1))
        head.register_buffer("route_keys", torch.empty((0, 32), dtype=torch.uint8))
        head.register_buffer("route_features", torch.empty((0, head.feature_width)))
        head.register_buffer("null_features", head.encode("<no-action>"))
        state = head.encode("mcp.novel call")[None]
        result = head.select_internal(state, [{"id": "mcp.novel", "actions": ["call"], "grant": "ask"}])
        self.assertEqual(result["selected"], {"toolId": "mcp.novel", "action": "call"})
        self.assertEqual(result["unseenStructuralCandidates"], 1)
        self.assertEqual(head.route_keys.shape[0], 0)
        self.assertFalse(result["generalizationVerified"])
        argument_head = argument_method_fixture({"query": "fixture only"})
        decoded = argument_head.decode(torch.zeros(1, 4), torch.zeros(1, head.feature_width), tool_id="mcp.novel", action="call")
        self.assertEqual(decoded["arguments"], {"query": "fixture only"})
        self.assertFalse(decoded["routeSeenInDeclaredTraining"])
        self.assertFalse(decoded["generalizationVerified"])

    def test_live_action_callback_evaluates_heads_but_only_visible_invocation_emits_once(self):
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.device = torch.device("cpu")
        brain.config = SimpleNamespace(max_seq_len=8192, vocab_size=261, online_learning=True, idle_cognition=True)
        kind = "tool"
        observed = []

        def policy(hidden):
            observed.append(hidden.detach().clone())
            logits = torch.full((1, len(ACTION_KINDS)), -30.0)
            logits[0, ACTION_KINDS.index(kind)] = 30.0
            return logits

        def decoder():
            return SimpleNamespace(
                action_argument_head=argument_method_fixture({"query": "method fixture"}, "web.search", "search"),
                tool_route_head=SimpleNamespace(select_internal=lambda *_: {"selected": {"toolId": "web.search", "action": "search"}}),
                action_policy=policy, internal_action_policy=policy,
            )

        schemas = [{"id": "web.search", "actions": ["search"], "grant": "ask", "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}]
        inputs = dict(schemas=schemas, input_text="a natural request", assembly_ids=[], organic_state={}, action_cue=torch.zeros(1, 4), seed=3)
        brain.decoder = decoder()
        private = NativeActionEmissionLedger("turn")
        callback = brain._generation_activity_callback(**inputs, ledger=private)
        callback(torch.ones(1, 4), 1, 0.5)
        self.assertEqual(len(private.actions), 1)
        brain.decoder = decoder()
        visible = NativeActionEmissionLedger("turn")
        events = []
        callback = brain._generation_activity_callback(**inputs, ledger=visible, emit=lambda *event: events.append(event))
        callback(torch.ones(1, 4), 1, 0.5)
        callback(torch.ones(1, 4) * 2, 2, 0.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][1]["actionId"], next(iter(private.actions.values()))["actionId"])
        kind = "stop"
        self.assertTrue(callback(torch.ones(1, 4) * 3, 3, 0.5)["stop"])
        self.assertEqual(len(events), 2)
        self.assertGreaterEqual(len(observed), 8)

    def test_learned_source_edits_use_typed_existing_authority_path_not_backend_synthesis(self):
        payload = {
            "objective": "record an isolated measured helper", "candidateKind": "source",
            "sourceEdits": [{"path": "src/measurement.ts", "content": "export const measurement = 1;\n", "expectedSha256": None}],
        }
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.device = torch.device("cpu")
        brain.config = SimpleNamespace(max_seq_len=8192, recursive_improvement=True, online_learning=True, idle_cognition=True)
        brain.decoder = SimpleNamespace(action_argument_head=argument_method_fixture(payload, "source.self-modify", "propose"))
        schemas = [{"id": "source.self-modify", "actions": ["propose"], "grant": "ask", "inputSchema": {
            "type": "object", "properties": {
                "objective": {"type": "string"}, "candidateKind": {"type": "string"},
                "sourceEdits": {"type": "array", "items": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"}, "expectedSha256": {"type": ["string", "null"]}
                }, "required": ["path", "content", "expectedSha256"]}},
            }, "required": ["objective"],
        }}]
        logits = torch.full((1, len(ACTION_KINDS)), -30.0)
        logits[0, ACTION_KINDS.index("evolve")] = 30.0
        _scores, actions = brain._select_structured_actions(
            logits, schemas=schemas, input_text="", assembly_ids=[], organic_state={}, neural_state=torch.zeros(1, 4),
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["arguments"]["sourceEdits"], payload["sourceEdits"])
        self.assertEqual(actions[0]["arguments"]["candidateKind"], "source")
        self.assertTrue(brain._valid_neural_source_edits(payload["sourceEdits"]))
        self.assertFalse(brain._valid_neural_source_edits([{**payload["sourceEdits"][0], "path": "../outside.ts"}]))
        self.assertFalse(brain._valid_neural_source_edits([{**payload["sourceEdits"][0], "expectedSha256": "not-a-digest"}]))


if __name__ == "__main__":
    unittest.main()
