import json
import gc
import os
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.ground_up import (
    GROUND_UP_TOOL_TRAJECTORIES,
    ground_up_curriculum_manifest,
)
from omni_core.model import TERNARY_PROJECTION_TYPES


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_PATH = (
    REPOSITORY_ROOT / "architecture" / "omnicortex-ground-up-v1.json"
)


class GroundUpArchitectureContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.brains = []

    def tearDown(self):
        for brain in self.brains:
            brain.events.close()
        self.temporary.cleanup()

    def make_brain(self, name, config):
        brain = AdaptiveBrain(name, self.root / name, config)
        self.brains.append(brain)
        return brain

    def test_public_config_never_switches_to_persisted_shape_parsing(self):
        config = OmniConfig.from_external(
            {
                "hardwareTier": "micro",
                "d_model": 4096,
                "n_layers": 96,
                "vsa_dim": 8192,
            }
        )

        self.assertEqual(config.d_model, 32)
        self.assertEqual(config.n_layers, 1)
        self.assertEqual(config.vsa_dim, 128)
        self.assertEqual(config.origin_kind, "ground-up")
        self.assertFalse(hasattr(config, "foundation_model_id"))

    def test_public_profile_instantiates_the_canonical_native_parameter_count(self):
        architecture = json.loads(ARCHITECTURE_PATH.read_text("utf-8"))
        profile = architecture["profiles"]["micro"]
        items = int(profile["autoWorkingMemoryItems"])
        config = OmniConfig.from_external(
            {
                "hardwareTier": "micro",
                "workingMemoryMode": "auto",
                "workingMemorySlots": items,
                "device": "cpu",
            }
        )
        brain = self.make_brain("profile", config)
        workspace_latents = max(
            int(architecture["workspaceLatentRule"]["minimumLatents"]),
            items
            // int(
                architecture["workspaceLatentRule"][
                    "workingMemoryItemsPerLatent"
                ]
            ),
        )
        expected = int(profile["baseParametersExcludingWorkspaceLatents"]) + (
            workspace_latents * int(profile["dModel"])
        )

        self.assertEqual(config.origin_kind, "ground-up")
        self.assertFalse(hasattr(config, "foundation_model_id"))
        self.assertEqual(
            brain.parameter_accounting()["totalNeuralParameters"], expected
        )
        self.assertEqual(expected, 339_768)
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in brain._trainable_modules()
                for parameter in module.parameters()
            )
        )

    def test_every_hardware_tier_matches_the_versioned_counting_contract(self):
        architecture = json.loads(ARCHITECTURE_PATH.read_text("utf-8"))
        for tier, profile in architecture["profiles"].items():
            with self.subTest(tier=tier), tempfile.TemporaryDirectory() as root:
                items = int(profile["autoWorkingMemoryItems"])
                config = OmniConfig.from_external(
                    {
                        "hardwareTier": tier,
                        "workingMemoryMode": "auto",
                        "workingMemorySlots": items,
                        "device": "cpu",
                    }
                )
                brain = AdaptiveBrain(tier, Path(root) / tier, config)
                try:
                    divisor = int(
                        architecture["workspaceLatentRule"][
                            "workingMemoryItemsPerLatent"
                        ]
                    )
                    minimum = int(
                        architecture["workspaceLatentRule"]["minimumLatents"]
                    )
                    workspace_latents = max(minimum, items // divisor)
                    expected = int(
                        profile["baseParametersExcludingWorkspaceLatents"]
                    ) + workspace_latents * int(profile["dModel"])
                    self.assertEqual(config.d_model, int(profile["dModel"]))
                    self.assertEqual(config.n_layers, int(profile["layers"]))
                    self.assertEqual(config.d_ff, int(profile["feedForward"]))
                    self.assertEqual(config.vsa_dim, int(profile["vsaDimensions"]))
                    self.assertEqual(
                        config.router_neurons, int(profile["routerNeurons"])
                    )
                    self.assertEqual(
                        brain.parameter_accounting()["totalNeuralParameters"],
                        expected,
                    )
                    seen = set()
                    ternary_parameters = 0
                    for root_module in brain._ternary_export_roots().values():
                        for module in root_module.modules():
                            if not isinstance(module, TERNARY_PROJECTION_TYPES):
                                continue
                            identity = id(module.weight)
                            if identity in seen:
                                continue
                            seen.add(identity)
                            ternary_parameters += int(module.weight.numel())
                    self.assertEqual(
                        ternary_parameters,
                        int(profile["exactTernaryProjectionParameters"]),
                    )
                finally:
                    brain.events.close()
                    del brain
                    gc.collect()

    def test_ground_up_construction_has_only_native_modules(self):
        config = OmniConfig(
            name="No inherited model",
            seed=91,
            origin_kind="ground-up",
        )
        brain = self.make_brain("isolated", config)

        runtime = brain.runtime_card()
        self.assertFalse(hasattr(brain, "foundation_cortex"))
        self.assertNotIn("foundation_adapter", brain._ternary_export_roots())
        self.assertFalse(runtime["pretrained"])
        self.assertFalse(runtime["baseFrozen"])
        self.assertNotIn("pretrained_text_cortex", runtime)
        self.assertNotIn("foundationModelId", runtime)
        self.assertEqual(runtime["origin_kind"], "ground-up")

    def test_seeded_master_weights_are_reproducible_but_not_imported(self):
        first = self.make_brain(
            "first",
            OmniConfig.micro(
                seed=317,
                origin_kind="ground-up",
            ),
        )
        second = self.make_brain(
            "second",
            OmniConfig.micro(
                seed=317,
                origin_kind="ground-up",
            ),
        )
        different = self.make_brain(
            "different",
            OmniConfig.micro(
                seed=318,
                origin_kind="ground-up",
            ),
        )

        self.assertEqual(first.parameter_checksum(), second.parameter_checksum())
        self.assertNotEqual(first.parameter_checksum(), different.parameter_checksum())
        audit = first.runtime_card()["ternary_audit"]
        self.assertEqual(audit["coverage"], 1.0)
        self.assertEqual(audit["violations"], [])
        self.assertEqual(audit["forwardPrecision"], "exact ternary {-1,0,+1}")
        self.assertTrue(set(audit["observedLevels"]).issubset({-1, 0, 1}))
        self.assertTrue(
            all(
                torch.isfinite(parameter).all().item()
                for module in first._trainable_modules()
                for parameter in module.parameters()
            )
        )

    def test_built_in_curriculum_is_transparent_local_data_not_model_output(self):
        manifest = ground_up_curriculum_manifest()
        encoded = json.dumps(manifest, sort_keys=True).lower()

        self.assertEqual(manifest["format"], "omni-ground-up-curriculum-manifest")
        self.assertEqual(manifest["sourceKind"], "ordinary-training-examples")
        self.assertEqual(manifest["languageCorpusRecords"], 0)
        self.assertEqual(manifest["dialogueAnswerRecords"], 0)
        self.assertFalse(manifest["pretrainedWeights"])
        self.assertFalse(manifest["apiTeacher"])
        self.assertIsNone(manifest["externalFoundation"])
        self.assertIsNone(manifest["upstreamModel"])
        self.assertNotIn("falcon", encoded)
        self.assertNotIn("llama", encoded)
        self.assertTrue(
            all(entry["upstreamModel"] is None for entry in manifest["datasetLedger"])
        )
        trajectories = json.dumps(
            GROUND_UP_TOOL_TRAJECTORIES,
            sort_keys=True,
        ).lower()
        for legacy in (
            "windows.",
            "powershell",
            "get-childitem",
            "c:\\",
            "/workspace",
        ):
            self.assertNotIn(legacy, trajectories)
        self.assertIn("system.files", trajectories)
        self.assertIn("system.shell", trajectories)

    def test_untrained_native_tool_route_does_not_use_keyword_shortcuts(self):
        brain = self.make_brain(
            "system-tools",
            OmniConfig.micro(
                origin_kind="ground-up",
            ),
        )
        schemas = [
            {"id": "system.files", "actions": ["list", "read", "write"]},
            {"id": "system.shell", "actions": ["run"]},
        ]
        path = "C:\\Users\\Public" if os.name == "nt" else "/tmp"
        command = "Get-Date" if os.name == "nt" else "date"
        file_text = 'read the file "%s/example.txt"' % path.rstrip("/\\")
        file_action = brain._materialize_generic_tool_action(
            schemas=schemas,
            input_text=file_text,
            assembly_ids=[],
            organic_state={},
            neural_state=brain._idea_model_vector(
                brain.memory.vector_for_text(file_text)
            ),
        )
        shell_text = (
            'run system shell command "%s" with working directory "%s"'
            % (command, path)
        )
        shell_action = brain._materialize_generic_tool_action(
            schemas=schemas,
            input_text=shell_text,
            assembly_ids=[],
            organic_state={},
            neural_state=brain._idea_model_vector(
                brain.memory.vector_for_text(shell_text)
            ),
        )

        self.assertIsNone(file_action)
        self.assertIsNone(shell_action)
        if os.name != "nt":
            windows_text = (
                'run PowerShell command "Get-Date" with working directory '
                '"/tmp"'
            )
            windows_dialect = brain._materialize_generic_tool_action(
                schemas=schemas,
                input_text=windows_text,
                assembly_ids=[],
                organic_state={},
                neural_state=brain._idea_model_vector(
                    brain.memory.vector_for_text(windows_text)
                ),
            )
            self.assertIsNone(windows_dialect)

if __name__ == "__main__":
    unittest.main()
