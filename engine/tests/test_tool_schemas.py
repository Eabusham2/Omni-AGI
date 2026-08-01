import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig


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
        return AdaptiveBrain.create(
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

    def test_tool_schema_limits_reject_oversized_capability_lists(self):
        brain = self.make_brain("bounded")
        oversized = [
            {"id": "tool.%03d" % index, "actions": ["run"]}
            for index in range(101)
        ]
        with self.assertRaisesRegex(ValueError, "at most 100"):
            brain.chat("hello", tool_schemas=oversized)
        brain.events.close()

    def test_learned_tool_head_materializes_typed_standard_actions(self):
        brain = self.make_brain("materialized")
        logits = torch.tensor(
            [[-8.0, 12.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
        )
        scores, actions = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "web.search",
                    "actions": ["search"],
                    "grant": "ask",
                },
                {
                    "id": "windows.files",
                    "actions": ["read", "write"],
                    "grant": "auto",
                },
            ],
            input_text="Search the web for liquid neural networks",
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertGreater(scores["tool"], 0.99)
        self.assertEqual(
            actions,
            [
                {
                    "kind": "tool",
                    "toolId": "web.search",
                    "action": "search",
                    "arguments": {
                        "assemblyIds": [],
                        "organic": True,
                        "query": "liquid neural networks",
                    },
                    "confidence": scores["tool"],
                }
            ],
        )
        brain.events.close()

    def test_tool_materialization_never_guesses_required_arguments(self):
        brain = self.make_brain("no-guesses")
        logits = torch.tensor(
            [[-8.0, 12.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
        )
        _, actions = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "windows.files",
                    "actions": ["write"],
                    "grant": "auto",
                }
            ],
            input_text="Write /tmp/omni.txt",
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(actions, [])
        brain.events.close()

    def test_powershell_materialization_requires_explicit_command_and_absolute_cwd(self):
        brain = self.make_brain("typed-powershell")
        logits = torch.tensor(
            [[-8.0, 12.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
        )
        scores, actions = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "windows.powershell",
                    "actions": ["run"],
                    "grant": "ask",
                }
            ],
            input_text=(
                'Run PowerShell command "Get-ChildItem -Force" with '
                'cwd "C:\\Users\\Eyad\\Omni".'
            ),
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(
            actions,
            [
                {
                    "kind": "tool",
                    "toolId": "windows.powershell",
                    "action": "run",
                    "arguments": {
                        "assemblyIds": [],
                        "organic": True,
                        "command": "Get-ChildItem -Force",
                        "cwd": "C:\\Users\\Eyad\\Omni",
                    },
                    "confidence": scores["tool"],
                }
            ],
        )

        _, missing_absolute_cwd = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "windows.powershell",
                    "actions": ["run"],
                    "grant": "auto",
                }
            ],
            input_text=(
                'Run PowerShell command "Get-ChildItem -Force" with '
                'cwd "relative-project".'
            ),
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(missing_absolute_cwd, [])
        brain.events.close()

    def test_browser_materialization_preserves_explicit_ordered_steps(self):
        brain = self.make_brain("typed-browser")
        logits = torch.tensor(
            [[-8.0, 12.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
        )
        scores, actions = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "browser.automation",
                    "actions": ["task"],
                    "grant": "ask",
                }
            ],
            input_text=(
                "In the browser, open https://example.com/login then "
                'click "#email", type "user@example.com" into "#email", '
                'press "Enter", wait for ".ready", extract text from '
                '".result", then take a screenshot.'
            ),
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(
            actions,
            [
                {
                    "kind": "tool",
                    "toolId": "browser.automation",
                    "action": "task",
                    "arguments": {
                        "assemblyIds": [],
                        "organic": True,
                        "url": "https://example.com/login",
                        "steps": [
                            {"kind": "click", "selector": "#email"},
                            {
                                "kind": "type",
                                "selector": "#email",
                                "value": "user@example.com",
                                "clear": True,
                            },
                            {"kind": "press", "key": "Enter"},
                            {"kind": "wait", "selector": ".ready"},
                            {"kind": "extract", "selector": ".result"},
                            {"kind": "screenshot"},
                        ],
                    },
                    "confidence": scores["tool"],
                }
            ],
        )

        _, vague = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "browser.automation",
                    "actions": ["task"],
                    "grant": "auto",
                }
            ],
            input_text="Use the browser to log me into my account.",
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(vague, [])
        brain.events.close()

    def test_organic_evolution_routes_to_viable_substrate_replay(self):
        brain = self.make_brain("organic-evolution")
        brain.config.recursive_improvement = True
        logits = torch.tensor(
            [[-8.0, -8.0, -8.0, -8.0, -8.0, -8.0, 12.0, -8.0]]
        )
        scores, actions = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "source.self-modify",
                    "actions": ["propose"],
                    "grant": "ask",
                }
            ],
            input_text="Reduce this measured prediction error.",
            assembly_ids=["active-assembly"],
            organic_state={"computeDemand": 0.9, "predictionError": 0.8},
        )
        self.assertGreater(scores["evolve"], 0.99)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["kind"], "evolve")
        self.assertEqual(
            actions[0]["arguments"]["candidateKind"], "substrate"
        )
        self.assertTrue(actions[0]["arguments"]["latentReplay"])
        self.assertNotIn("sourceEdits", actions[0]["arguments"])
        brain.events.close()

    def test_off_tool_schema_cannot_participate_in_action_selection(self):
        brain = self.make_brain("off-schema")
        logits = torch.tensor(
            [[-8.0, 12.0, -8.0, -8.0, -8.0, -8.0, -8.0, -8.0]]
        )
        _, actions = brain._select_structured_actions(
            logits,
            schemas=[
                {
                    "id": "web.search",
                    "actions": ["search"],
                    "grant": "off",
                }
            ],
            input_text="Search the web for ternary kernels",
            assembly_ids=[],
            organic_state={"computeDemand": 0.8},
        )
        self.assertEqual(actions, [])
        brain.events.close()


if __name__ == "__main__":
    unittest.main()
