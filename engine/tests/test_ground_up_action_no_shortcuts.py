"""Fast action-boundary checks without constructing or training a brain."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.model import ACTION_KINDS


def winning_logits(kind: str) -> torch.Tensor:
    values = torch.full((1, len(ACTION_KINDS)), -8.0)
    values[0, ACTION_KINDS.index(kind)] = 12.0
    return values


class GroundUpActionNoShortcutsTests(unittest.TestCase):
    def make_brain(self) -> AdaptiveBrain:
        brain = object.__new__(AdaptiveBrain)
        brain.config = SimpleNamespace(
            idle_cognition=True,
            recursive_improvement=True,
            image_enabled=True,
            audio_enabled=True,
            video_enabled=True,
        )
        brain.modality_training = {"image": 0, "audio": 0, "video": 0}
        brain.installed_modality_packs = []
        brain.liquid_state = torch.zeros((1, 64))
        brain.modalities = SimpleNamespace(select_imagination=Mock())
        return brain

    def test_legacy_lexical_tool_router_is_absent(self):
        self.assertFalse(hasattr(AdaptiveBrain, "_materialize_legacy_tool_action"))
        self.assertFalse(hasattr(AdaptiveBrain, "_ground_up_tool_route_assembly_evidence"))

    def test_idle_tool_route_cannot_invent_arguments_from_focus_labels(self):
        brain = self.make_brain()
        neural_head = Mock()
        brain.decoder = SimpleNamespace(tool_route_head=neural_head)
        selected = brain._materialize_generic_tool_action(
            schemas=[{"id": "system.files", "actions": ["list"], "grant": "ask"}],
            input_text="",
            assembly_ids=["learned-assembly"],
            organic_state={"promptFree": 1.0},
        )
        self.assertIsNone(selected)
        self.assertEqual(brain._last_tool_route_evidence["reason"],
                         "no-explicit-action-arguments")
        neural_head.select_internal.assert_not_called()

    def test_stale_neural_route_cannot_revive_an_off_tool(self):
        brain = self.make_brain()
        head = Mock()
        head.select_internal.return_value = {
            "kind": "trained-internal-schema-route",
            "selected": {"toolId": "system.files", "action": "read"},
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        selected = brain._materialize_generic_tool_action(
            schemas=[{"id": "system.files", "actions": ["read"], "grant": "off"}],
            input_text="read /tmp/report.txt", assembly_ids=[], organic_state={},
            neural_state=torch.zeros(1, 64),
        )
        self.assertIsNone(selected)
        self.assertEqual(brain._last_tool_route_evidence["materialization"],
                         "route-not-enabled")
        head.select.assert_not_called()

    def test_browser_prose_does_not_synthesize_side_effect_steps(self):
        brain = self.make_brain()
        schema = brain._normalize_tool_schemas([{
            "id": "browser.automation", "actions": ["task"], "grant": "ask",
            "actionInputSchemas": {"task": {
                "type": "object", "properties": {
                    "url": {"type": "string"}, "steps": {"type": "array"},
                }, "required": ["url"],
            }},
        }])[0]
        arguments = brain._literal_tool_route_arguments(
            "browser.automation", "task",
            'Open https://example.com; do not click "Delete" and do not take a screenshot.',
            schema,
        )
        self.assertEqual(arguments, {"url": "https://example.com"})
        self.assertFalse(hasattr(AdaptiveBrain, "_explicit_browser_steps"))
        head = Mock()
        head.select_internal.return_value = {
            "selected": {"toolId": "browser.automation", "action": "task"},
            "kind": "trained-internal-schema-route",
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        selected = brain._materialize_generic_tool_action(
            schemas=[schema],
            input_text='Open https://example.com; do not click "Delete" and do not take a screenshot.',
            assembly_ids=[], organic_state={},
            neural_state=torch.zeros(1, 64),
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["arguments"], {"url": "https://example.com"})

    def test_browser_accepts_only_recursively_validated_explicit_steps(self):
        brain = self.make_brain()
        schemas = brain._normalize_tool_schemas([{
            "id": "browser.automation", "actions": ["task"], "grant": "ask",
            "actionInputSchemas": {"task": {
                "type": "object", "properties": {
                    "url": {"type": "string"}, "steps": {"type": "array"},
                }, "required": ["url"],
            }},
        }])
        arguments = brain._literal_tool_route_arguments(
            "browser.automation", "task",
            '{"url":"https://example.com","steps":[{"kind":"click","selector":"#go"}]}',
            schemas[0],
        )
        candidate = {
            "toolId": "browser.automation", "action": "task",
            "arguments": arguments,
        }
        self.assertTrue(brain._materialized_tool_action_matches_schema(
            schemas, candidate
        ))
        multi = brain._literal_tool_route_arguments(
            "browser.automation", "task",
            '{"url":"https://example.com","steps":['
            '{"kind":"click","selector":"#go"},'
            '{"kind":"press","key":"Enter"}]}',
            schemas[0],
        )
        self.assertEqual(multi["steps"], [
            {"kind": "click", "selector": "#go"},
            {"kind": "press", "key": "Enter"},
        ])
        self.assertTrue(brain._materialized_tool_action_matches_schema(
            schemas, {**candidate, "arguments": multi}
        ))
        self.assertTrue(brain._valid_explicit_browser_steps([{
            "kind": "type", "selector": "#editor", "value": "line 1\nline 2",
            "clear": False,
        }]))
        for invalid in (
            [{"kind": "click", "selector": "#go", "unexpected": "side effect"}],
            [{"kind": "type", "selector": "#field", "value": "x", "clear": "yes"}],
            [{"kind": "wait", "milliseconds": 60_000}],
            [{"kind": ["click"], "selector": "#go"}],
        ):
            with self.subTest(invalid=invalid):
                self.assertFalse(brain._materialized_tool_action_matches_schema(
                    schemas, {**candidate, "arguments": {
                        "url": "https://example.com", "steps": invalid,
                    }}
                ))

    def test_browser_prose_step_requires_grounded_neural_operation(self):
        brain = self.make_brain()
        schema = brain._normalize_tool_schemas([{
            "id": "browser.automation", "actions": ["task"], "grant": "ask",
            "actionInputSchemas": {"task": {
                "type": "object", "properties": {
                    "url": {"type": "string"}, "steps": {"type": "array"},
                }, "required": ["url"],
            }},
        }])[0]
        route = Mock()
        route.select_internal.return_value = {
            "selected": {"toolId": "browser.automation", "action": "task"},
        }
        operation = Mock()
        operation.grounded_for.return_value = 0
        operation.decode.return_value = {
            "arguments": {"operation": "click"},
            "reason": "learned-typed-arguments", "meanTokenProbability": 0.94,
        }
        brain.decoder = SimpleNamespace(
            tool_route_head=route, action_argument_head=operation,
        )
        state = torch.zeros(1, 64)
        utterance = 'Open https://example.com and click "#go"'

        ungrounded = brain._materialize_generic_tool_action(
            schemas=[schema], input_text=utterance, neural_state=state,
            assembly_ids=[], organic_state={},
        )
        self.assertEqual(ungrounded["arguments"], {"url": "https://example.com"})
        operation.decode.assert_not_called()

        operation.grounded_for.side_effect = lambda tool, action: (
            1 if (tool, action) in {
                ("browser.operation", "select"),
                ("browser.operation", "select:none"),
                ("browser.operation", "select:click"),
            } else 0
        )
        operation.schema_features.return_value = torch.zeros(1024)
        selected = brain._materialize_generic_tool_action(
            schemas=[schema], input_text=utterance, neural_state=state,
            assembly_ids=[], organic_state={},
        )
        self.assertEqual(selected["arguments"]["steps"], [
            {"kind": "click", "selector": "#go"}
        ])
        self.assertEqual(
            selected["routeEvidence"]["materialization"],
            "grounded-neural-browser-operation-literal-operands-schema-validated",
        )
        operation.decode.assert_called_once()
        self.assertEqual(operation.decode.call_args.kwargs["tool_id"],
                         "browser.operation")

        operation.decode.return_value = {
            "arguments": {"operation": "click"},
            "reason": "uncertain-argument-decoding", "meanTokenProbability": 0.6,
        }
        uncertain = brain._materialize_generic_tool_action(
            schemas=[schema], input_text=utterance, neural_state=state,
            assembly_ids=[], organic_state={},
        )
        self.assertEqual(uncertain["arguments"], {"url": "https://example.com"})

        operation.decode.return_value = {
            "arguments": {"operation": "none"},
            "reason": "learned-typed-arguments", "meanTokenProbability": 0.94,
        }
        negated = brain._materialize_generic_tool_action(
            schemas=[schema],
            input_text='Open https://example.com but do not click "#go"',
            neural_state=state, assembly_ids=[], organic_state={},
        )
        self.assertEqual(negated["arguments"], {"url": "https://example.com"})
        self.assertFalse(hasattr(AdaptiveBrain, "_explicit_browser_steps"))

    def test_chat_tool_and_independent_support_share_active_neural_route(self):
        brain = self.make_brain()
        head = Mock()
        head.select_internal.return_value = {
            "kind": "trained-internal-schema-route",
            "selected": {"toolId": "system.files", "action": "read"},
            "reason": "trained-internal-weights",
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        schemas = brain._normalize_tool_schemas([{
            "id": "system.files", "actions": ["read"], "grant": "ask",
            "actionInputSchemas": {"read": {
                "type": "object", "properties": {
                    "path": {"type": "string"},
                }, "required": ["path"],
            }},
        }])
        state = torch.randn(1, 64)
        for kind, support in (
            ("tool", ()),
            ("talk", (winning_logits("tool"),)),
        ):
            with self.subTest(kind=kind):
                head.reset_mock()
                _scores, actions = brain._select_structured_actions(
                    winning_logits(kind), schemas=schemas,
                    input_text="read /tmp/explicit.txt", neural_state=state,
                    assembly_ids=[], organic_state={"computeDemand": 0.5},
                    supporting_action_logits=support,
                )
                self.assertEqual(len(actions), 1)
                self.assertEqual(actions[0]["toolId"], "system.files")
                self.assertEqual(actions[0]["arguments"]["path"],
                                 "/tmp/explicit.txt")
                head.select_internal.assert_called_once()
                self.assertTrue(torch.equal(
                    head.select_internal.call_args.args[0], state
                ))
                head.select.assert_not_called()

    def test_untrained_media_does_not_become_organic_imagination(self):
        brain = self.make_brain()
        scores, actions = brain._select_structured_actions(
            winning_logits("imagine"),
            schemas=[{"id": "modality.imagine", "actions": ["generate"], "grant": "ask"}],
            input_text="",
            assembly_ids=["scene-assembly"],
            organic_state={"computeDemand": 0.9},
        )
        self.assertGreater(scores["imagine"], 0.99)
        self.assertEqual(actions, [])
        brain.modalities.select_imagination.assert_not_called()

    def test_trained_media_routes_through_same_brain_selector(self):
        brain = self.make_brain()
        brain.modality_training["image"] = 1
        brain.modalities.select_imagination.return_value = (
            "image", {"image": 1.0, "audio": 0.0, "video": 0.0}
        )
        with patch.object(brain, "_modality_idea", return_value=torch.zeros((1, 64))):
            _scores, actions = brain._select_structured_actions(
                winning_logits("imagine"),
                schemas=[{"id": "modality.imagine", "actions": ["generate"], "grant": "ask"}],
                input_text="",
                assembly_ids=["scene-assembly"],
                organic_state={"computeDemand": 0.9},
            )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["arguments"]["modality"], "image")
        self.assertTrue(actions[0]["arguments"]["trainedPackAvailable"])
        self.assertEqual(brain.modalities.select_imagination.call_args.kwargs["enabled"],
                         ["image"])

    def test_agent_and_evolution_need_a_real_objective(self):
        brain = self.make_brain()
        for kind, tool_id, action in (
            ("agent", "agent.fork", "start"),
            ("evolve", "source.self-modify", "propose"),
        ):
            with self.subTest(kind=kind):
                schemas = [{"id": tool_id, "actions": [action], "grant": "ask"}]
                _scores, absent = brain._select_structured_actions(
                    winning_logits(kind), schemas=schemas, input_text="",
                    assembly_ids=["assembly"], organic_state={"computeDemand": 0.9},
                )
                self.assertEqual(absent, [])
                _scores, explicit = brain._select_structured_actions(
                    winning_logits(kind), schemas=schemas,
                    input_text="Investigate the renderer crash",
                    assembly_ids=["assembly"], organic_state={"computeDemand": 0.9},
                )
                if kind == "evolve":
                    # Objective prose alone must not invent a mutation kind.
                    self.assertEqual(explicit, [])
                    continue
                self.assertEqual(len(explicit), 1)
                self.assertEqual(explicit[0]["arguments"]["objective"],
                                 "Investigate the renderer crash")

    def test_explicit_evolution_kind_is_not_replaced_by_a_fixed_substrate_choice(self):
        brain = self.make_brain()
        schema = brain._normalize_tool_schemas([{
            "id": "source.self-modify", "actions": ["propose"], "grant": "ask",
            "actionInputSchemas": {"propose": {
                "type": "object", "properties": {
                    "objective": {"type": "string"},
                    "candidateKind": {"type": "string"},
                    "addExperts": {"type": "integer"},
                }, "required": ["objective"],
            }},
        }])
        _scores, actions = brain._select_structured_actions(
            winning_logits("evolve"), schemas=schema,
            input_text='{"objective":"Improve recurrent memory","candidateKind":"architecture","addExperts":1}',
            assembly_ids=[], organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["arguments"]["objective"],
                         "Improve recurrent memory")
        self.assertEqual(actions[0]["arguments"]["candidateKind"],
                         "architecture")
        self.assertEqual(actions[0]["arguments"]["addExperts"], 1)

    def test_tool_kind_cannot_bypass_imagination_agent_or_evolution_gates(self):
        brain = self.make_brain()
        for tool_id, action in (
            ("modality.imagine", "generate"),
            ("agent.fork", "start"),
            ("source.self-modify", "propose"),
        ):
            with self.subTest(tool_id=tool_id), patch.object(
                brain, "_materialize_generic_tool_action",
                return_value={"toolId": tool_id, "action": action, "arguments": {}},
            ):
                _scores, actions = brain._select_structured_actions(
                    winning_logits("tool"),
                    schemas=[{"id": tool_id, "actions": [action], "grant": "ask"}],
                    input_text="Please make this happen",
                    assembly_ids=[], organic_state={"computeDemand": 0.9},
                )
                self.assertEqual(actions, [])


if __name__ == "__main__":
    unittest.main()
