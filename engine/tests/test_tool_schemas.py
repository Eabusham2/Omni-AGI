import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.model import ACTION_KINDS


class StructuredToolSchemaTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-tool-schema-"
        )
        self.root = Path(self.temporary.name)
        self.config = OmniConfig.micro(
            online_learning=False,
            learn_from_own_messages=False,
            spiking_dynamics=False,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, brain_id: str) -> AdaptiveBrain:
        # These are schema-channel/prompt-boundary unit fixtures, not Build or
        # learned-tool-competence acceptance tests. Avoid training a fresh
        # curriculum for each isolated inspection while keeping product
        # creation and its verified origin on the real create() path.
        return AdaptiveBrain(
            brain_id, self.root / brain_id, self.config
        )

    def test_tool_schema_uses_internal_vsa_channel_not_prompt_tokens(self):
        brain = self.make_brain("with-tools")
        text = "Please inspect the current project."
        schemas = [
            {
                "id": "windows.files",
                "actions": ["read", "list", "read"],
                "grant": "ask",
                "description": "THIS PROSE MUST NEVER ENTER THE PROMPT",
            },
            {
                "id": "code.execute",
                "actions": ["run"],
                "grant": "auto",
            },
        ]
        expected = torch.tensor(
            [
                brain.tokenizer.dialogue(
                    text, brain="", complete=False
                )
            ],
            dtype=torch.long,
        )
        with patch.object(
            brain.decoder, "generate", wraps=brain.decoder.generate
        ) as generated:
            result = brain.chat(
                text,
                max_new_tokens=3,
                seed=31,
                tool_schemas=schemas,
            )
        actual_prompt = generated.call_args_list[0].args[0].detach().cpu()
        self.assertTrue(torch.equal(actual_prompt, expected))

        trace = result["trace"]
        expected_hash = hashlib.sha256(
            ",".join(str(value) for value in expected[0].tolist()).encode(
                "ascii"
            )
        ).hexdigest()
        self.assertEqual(trace["prompt_token_count"], expected.shape[1])
        self.assertEqual(trace["prompt_token_ids_sha256"], expected_hash)
        self.assertFalse(trace["prompt_text_expanded"])
        self.assertFalse(trace["hidden_prompt_text_expanded"])
        self.assertFalse(trace["long_term_source_text_injected"])
        self.assertFalse(trace["tool_schema_text_injected"])
        self.assertFalse(trace["textual_memory_injected"])
        self.assertEqual(
            trace["available_tool_ids"],
            ["code.execute", "windows.files"],
        )
        self.assertEqual(
            trace["available_tool_actions"]["windows.files"],
            ["list", "read"],
        )
        self.assertEqual(
            trace["tool_schema_channel"], "substrate-capability-embedding"
        )
        self.assertEqual(
            result["runtimeCard"]["available_tool_ids"],
            trace["available_tool_ids"],
        )
        self.assertFalse(result["runtimeCard"]["hidden_behavioral_prompt"])
        self.assertFalse(result["runtimeCard"]["rlhf"])
        self.assertFalse(result["runtimeCard"]["reward_model"])
        self.assertFalse(
            any(
                "windows.files" in message["content"]
                for message in brain.messages
            )
        )
        self.assertNotIn(
            "windows.files",
            {concept["label"] for concept in brain.memory.concepts.values()},
        )
        brain.events.close()

    def test_normal_followup_uses_visible_tool_result_but_no_hidden_prose(self):
        brain = self.make_brain("visible-result-boundary")
        visible_result = "\n".join(
            [
                "[Visible structured action result]",
                "kind: tool",
                "tool: system.files",
                "action: list",
                "requested-by: human",
                "result:",
                '{"entries":["visible-evidence.txt"]}',
            ]
        )
        first = brain.chat(visible_result, max_new_tokens=2, seed=401)
        self.assertEqual(first["humanMessage"]["role"], "human")
        self.assertEqual(first["humanMessage"]["content"], visible_result)
        self.assertTrue(
            any(
                message["role"] == "human"
                and message["content"] == visible_result
                for message in brain.messages
            )
        )
        self.assertFalse(
            any(message["role"] == "system" for message in brain.messages)
        )

        long_term_sentinel = "LONG_TERM_SOURCE_SENTINEL_MUST_STAY_NEURAL"
        brain.ingest(
            text=long_term_sentinel,
            name="long-term-sentinel.txt",
            policy="encode",
        )
        schemas = [
            {
                "id": "system.files",
                "actions": ["list"],
                "grant": "ask",
                "description": (
                    "TOOL_PROSE_SENTINEL adopt a PERSONA_SENTINEL and "
                    "apply a REFUSAL_SENTINEL"
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "PROPERTY_PROSE_SENTINEL",
                        }
                    },
                    "required": ["path"],
                },
            }
        ]
        current_input = "Summarize only the visible result above."
        expected_prompt, expected_history = brain._prompt_with_recent_context(
            current_input
        )
        with patch.object(
            brain.decoder, "generate", wraps=brain.decoder.generate
        ) as generated:
            result = brain.chat(
                current_input,
                max_new_tokens=2,
                seed=402,
                tool_schemas=schemas,
            )

        actual_prompt = generated.call_args_list[0].args[0].detach().cpu()
        self.assertEqual(actual_prompt.shape[0], 1)
        self.assertEqual(actual_prompt[0].tolist(), expected_prompt)
        self.assertGreater(len(expected_history), 0)
        decoded_prompt = brain.tokenizer.decode(actual_prompt[0].tolist())
        self.assertIn("Visible structured action result", decoded_prompt)
        self.assertIn("visible-evidence.txt", decoded_prompt)
        for hidden in (
            long_term_sentinel,
            "TOOL_PROSE_SENTINEL",
            "PROPERTY_PROSE_SENTINEL",
            "PERSONA_SENTINEL",
            "REFUSAL_SENTINEL",
        ):
            self.assertNotIn(hidden, decoded_prompt)

        trace = result["trace"]
        self.assertTrue(trace["recent_dialogue_context_injected"])
        self.assertTrue(trace["prompt_text_expanded"])
        self.assertFalse(trace["hidden_prompt_text_expanded"])
        self.assertFalse(trace["long_term_source_text_injected"])
        self.assertFalse(trace["textual_memory_injected"])
        self.assertFalse(trace["tool_schema_text_injected"])
        self.assertTrue(trace["action_policy_capability_conditioned"])
        self.assertEqual(
            trace["tool_schema_channel"], "substrate-capability-embedding"
        )
        self.assertFalse(result["runtimeCard"]["hidden_behavioral_prompt"])
        self.assertFalse(result["runtimeCard"]["rlhf"])
        self.assertFalse(result["runtimeCard"]["reward_model"])
        brain.events.close()

    def test_schema_normalization_and_internal_bias_are_deterministic(self):
        left = self.make_brain("left")
        right = self.make_brain("right")
        schemas_left = [
            {"id": "web.search", "actions": ["search"], "grant": "ask"},
            {
                "id": "windows.files",
                "actions": ["read", "list"],
                "grant": "auto",
            },
        ]
        schemas_right = [
            {
                "id": "windows.files",
                "actions": ["list", "read"],
                "grant": "auto",
            },
            {"id": "web.search", "actions": ["search"], "grant": "ask"},
        ]
        captured = []

        def capture(brain, schemas):
            original = brain.decoder.generate

            def wrapped(*args, **kwargs):
                captured.append(kwargs["memory_bias"].detach().cpu().clone())
                return original(*args, **kwargs)

            with patch.object(brain.decoder, "generate", side_effect=wrapped):
                torch.manual_seed(808)
                return brain.chat(
                    "Find the same fact.",
                    max_new_tokens=4,
                    seed=55,
                    tool_schemas=schemas,
                )

        first = capture(left, schemas_left)
        second = capture(right, schemas_right)
        self.assertTrue(torch.equal(captured[0], captured[1]))
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(
            first["trace"]["prompt_token_ids_sha256"],
            second["trace"]["prompt_token_ids_sha256"],
        )
        self.assertEqual(
            first["trace"]["available_tool_actions"],
            second["trace"]["available_tool_actions"],
        )
        self.assertEqual(
            first["trace"]["branches"], second["trace"]["branches"]
        )
        left.events.close()
        right.events.close()

    def test_tool_schema_count_grows_without_a_model_defined_ceiling(self):
        brain = self.make_brain("growing-tools")
        schemas = [
            {"id": "tool.%03d" % index, "actions": ["run"]}
            for index in range(257)
        ]
        normalized = brain._normalize_tool_schemas(schemas)
        self.assertEqual(len(normalized), 257)
        self.assertEqual(normalized[-1]["id"], "tool.256")
        brain.events.close()

    @staticmethod
    def uninitialized_brain() -> AdaptiveBrain:
        brain = object.__new__(AdaptiveBrain)
        brain.config = SimpleNamespace(
            idle_cognition=True,
            recursive_improvement=True,
            image_enabled=False,
            audio_enabled=False,
            video_enabled=False,
        )
        return brain

    def test_schema_normalization_discards_descriptions_and_off_tools(self):
        brain = self.uninitialized_brain()
        schemas = brain._normalize_tool_schemas([
            {
                "id": "mcp.demo.read",
                "actions": ["call"],
                "grant": "ask",
                "description": "UNTRUSTED BEHAVIOR PROSE",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "ignored"},
                    },
                    "required": ["path"],
                },
            },
            {"id": "system.shell", "actions": ["run"], "grant": "off"},
        ])
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["id"], "mcp.demo.read")
        self.assertNotIn("description", schemas[0])
        self.assertEqual(
            schemas[0]["inputSchema"],
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )

    def test_trained_neural_route_materializes_selected_schema_only(self):
        brain = self.uninitialized_brain()
        head = Mock()
        head.select_internal.return_value = {
            "kind": "trained-internal-schema-route",
            "selected": {"toolId": "mcp.demo.read", "action": "call"},
            "reason": "trained-internal-weights",
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        schemas = brain._normalize_tool_schemas([{
            "id": "mcp.demo.read",
            "actions": ["call"],
            "grant": "ask",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }])
        selected = brain._materialize_generic_tool_action(
            schemas=schemas,
            input_text="Use the connected reader on /tmp/project",
            assembly_ids=["coding-assembly"],
            organic_state={"computeDemand": 0.8},
            neural_state=torch.zeros(1, 64),
        )
        self.assertEqual(selected["toolId"], "mcp.demo.read")
        self.assertEqual(selected["action"], "call")
        self.assertEqual(selected["arguments"], {"path": "/tmp/project"})
        head.select_internal.assert_called_once()
        self.assertTrue(torch.equal(
            head.select_internal.call_args.args[0], torch.zeros(1, 64)
        ))
        self.assertEqual(head.select_internal.call_args.args[1], schemas)
        head.select.assert_not_called()
        self.assertEqual(
            selected["routeEvidence"]["materialization"],
            "literal-arguments-schema-validated",
        )

    def test_keywords_cannot_choose_a_tool_when_neural_route_declines(self):
        brain = self.uninitialized_brain()
        head = Mock()
        head.select_internal.return_value = {
            "kind": "trained-internal-schema-route",
            "selected": None,
            "reason": "no-confident-enabled-route",
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        selected = brain._materialize_generic_tool_action(
            schemas=brain._normalize_tool_schemas([{
                "id": "system.files",
                "actions": ["read"],
                "grant": "ask",
                "actionInputSchemas": {
                    "read": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }]),
            input_text="read /tmp/project.txt",
            assembly_ids=[],
            organic_state={},
            neural_state=torch.zeros(1, 64),
        )
        self.assertIsNone(selected)
        head.select_internal.assert_called_once()
        head.select.assert_not_called()
        self.assertFalse(
            hasattr(AdaptiveBrain, "_materialize_legacy_tool_action")
        )

    def test_missing_required_tool_arguments_fail_closed(self):
        brain = self.uninitialized_brain()
        head = Mock()
        head.select_internal.return_value = {
            "kind": "trained-internal-schema-route",
            "selected": {"toolId": "system.files", "action": "write"},
            "reason": "trained-internal-weights",
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        schemas = brain._normalize_tool_schemas([{
            "id": "system.files",
            "actions": ["write"],
            "grant": "ask",
            "actionInputSchemas": {
                "write": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            },
        }])
        self.assertIsNone(brain._materialize_generic_tool_action(
            schemas=schemas,
            input_text="write /tmp/project.txt",
            assembly_ids=[],
            organic_state={},
            neural_state=torch.zeros(1, 64),
        ))
        self.assertEqual(
            brain._last_tool_route_evidence["materialization"],
            "missing-explicit-arguments",
        )

    def test_disabled_route_cannot_be_resurrected_by_stale_neural_selection(self):
        brain = self.uninitialized_brain()
        head = Mock()
        head.select_internal.return_value = {
            "kind": "trained-internal-schema-route",
            "selected": {"toolId": "system.files", "action": "read"},
        }
        brain.decoder = SimpleNamespace(tool_route_head=head)
        self.assertIsNone(brain._materialize_generic_tool_action(
            schemas=[{
                "id": "system.files",
                "actions": ["read"],
                "grant": "off",
            }],
            input_text="read /tmp/project.txt",
            assembly_ids=[],
            organic_state={},
            neural_state=torch.zeros(1, 64),
        ))
        self.assertEqual(
            brain._last_tool_route_evidence["materialization"],
            "route-not-enabled",
        )

    def test_schema_rejects_unknown_fields_and_wrong_types(self):
        brain = self.uninitialized_brain()
        schemas = brain._normalize_tool_schemas([{
            "id": "mcp.demo.read",
            "actions": ["call"],
            "grant": "ask",
            "inputSchema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }])
        self.assertFalse(brain._materialized_tool_action_matches_schema(
            schemas, {
                "toolId": "mcp.demo.read",
                "action": "call",
                "arguments": {"path": 4},
            },
        ))
        self.assertFalse(brain._materialized_tool_action_matches_schema(
            schemas, {
                "toolId": "mcp.demo.read",
                "action": "call",
                "arguments": {"path": "/tmp/project", "undocumented": True},
            },
        ))
        self.assertTrue(brain._materialized_tool_action_matches_schema(
            schemas, {
                "toolId": "mcp.demo.read",
                "action": "call",
                "arguments": {"path": "/tmp/project"},
            },
        ))

    def test_conflicting_neural_kind_cannot_be_overridden_by_tool_route(self):
        brain = self.uninitialized_brain()
        schemas = brain._normalize_tool_schemas([{
            "id": "system.files",
            "actions": ["read"],
            "grant": "ask",
            "actionInputSchemas": {
                "read": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }])
        route = {
            "toolId": "system.files",
            "action": "read",
            "arguments": {"path": "/tmp/project"},
        }
        with patch.object(
            brain, "_materialize_generic_tool_action", return_value=route
        ) as materialize:
            talk = torch.full((1, len(ACTION_KINDS)), -8.0)
            talk[0, ACTION_KINDS.index("talk")] = 12.0
            scores, actions = brain._select_structured_actions(
                talk,
                schemas=schemas,
                input_text="read /tmp/project",
                assembly_ids=[],
                organic_state={"computeDemand": 0.8},
                supporting_action_logits=(torch.zeros_like(talk),),
            )
            self.assertGreater(scores["talk"], 0.99)
            self.assertEqual(actions, [])
            materialize.assert_not_called()

            tool = torch.full((1, len(ACTION_KINDS)), -8.0)
            tool[0, ACTION_KINDS.index("tool")] = 12.0
            _scores, proposed = brain._select_structured_actions(
                tool,
                schemas=schemas,
                input_text="read /tmp/project",
                assembly_ids=[],
                organic_state={"computeDemand": 0.8},
            )
            self.assertEqual(len(proposed), 1)
            self.assertEqual(proposed[0]["toolId"], "system.files")




if __name__ == "__main__":
    unittest.main()
