import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig


RETIRED_BETA_FIELDS = {
    "ternary_weights",
    "spiking_dynamics",
    "stdp_plasticity",
    "liquid_dynamics",
    "vector_symbolic_memory",
    "consolidation_enabled",
    "metaplasticity",
    "noise",
    "memory_injection",
    "learn_from_own_messages",
    "max_concepts",
    "max_ideas",
    "max_synapses",
    "novelty_drive",
    "coherence_drive",
    "curiosity_drive",
    "parallel_thoughts",
    "growth_policy",
    "max_experts",
}


class StableConfigTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_stable_payload_hides_retired_beta_controls_but_loads_legacy_dicts(self):
        legacy = OmniConfig.micro(ternary_weights=False)
        payload = legacy.to_dict()
        self.assertTrue(RETIRED_BETA_FIELDS.isdisjoint(payload))
        self.assertEqual(payload["hardware_tier"], "micro")
        self.assertEqual(payload["router_neurons"], 24)
        self.assertEqual(payload["working_memory_slots"], 128)

        # The compatibility parser still accepts internal/legacy architecture
        # dictionaries. New saves deliberately normalize away those controls.
        loaded = OmniConfig.from_dict(
            {
                **payload,
                "ternary_weights": False,
                "noise": 0.7,
                "curiosity_drive": 0.2,
                "parallel_thoughts": 5,
                "max_concepts": 13,
                "growth_policy": "elastic",
            }
        )
        self.assertFalse(loaded.ternary_weights)
        self.assertFalse(hasattr(loaded, "noise"))
        self.assertFalse(hasattr(loaded, "parallel_thoughts"))
        self.assertFalse(hasattr(loaded, "max_concepts"))
        self.assertFalse(hasattr(loaded, "growth_policy"))

    def test_external_population_is_hardware_derived_and_beta_sliders_are_ignored(self):
        retired_values = {
            "initialNeuronBudget": 2**40,
            "maxConcepts": 1,
            "maxIdeas": 1,
            "maxSynapses": 1,
            "maxExperts": 1,
            "growthPolicy": "fixed",
            "noveltyDrive": 0.0,
            "coherenceDrive": 0.0,
            "curiosityDrive": 0.0,
            "noise": 1.0,
            "parallelThoughts": 99,
            "consolidation": False,
            "metaplasticity": False,
            "learnFromOwnMessages": False,
            "firingThreshold": 1.9,
            "membraneLeak": 0.1,
            "stdpWindow": 200,
            "shortTermHalfLifeMinutes": 1,
            "longTermThreshold": 0.01,
            "forgettingRate": 0.9,
            "consolidationRate": 0.9,
        }
        expected_routers = {
            "micro": 24,
            "personal": 64,
            "gpu": 96,
            "workstation": 128,
        }
        for tier, expected in expected_routers.items():
            with self.subTest(tier=tier):
                low = OmniConfig.from_external(
                    {
                        "hardwareTier": tier,
                        "initialNeuronBudget": 64,
                    }
                )
                high = OmniConfig.from_external(
                    {"hardwareTier": tier, **retired_values}
                )
                self.assertEqual(low.router_neurons, expected)
                self.assertEqual(high.router_neurons, expected)
                self.assertFalse(hasattr(high, "noise"))
                self.assertFalse(hasattr(high, "curiosity_drive"))
                self.assertFalse(hasattr(high, "parallel_thoughts"))
                self.assertTrue(high.consolidation_enabled)
                self.assertTrue(high.metaplasticity)
                self.assertTrue(high.learn_from_own_messages)
                self.assertEqual(
                    high.firing_threshold, OmniConfig().firing_threshold
                )
                self.assertEqual(
                    high.short_term_half_life_minutes,
                    OmniConfig().short_term_half_life_minutes,
                )

    def test_engine_manifest_uses_filtered_stable_config_and_reloads(self):
        with tempfile.TemporaryDirectory(prefix="omni-config-test-") as folder:
            root = Path(folder)
            brain = AdaptiveBrain.create(
                "stable-config-brain",
                root,
                OmniConfig.micro(),
            )
            brain.events.close()
            metadata = json.loads(
                (root / "engine" / "brain.json").read_text("utf-8")
            )
            self.assertTrue(
                RETIRED_BETA_FIELDS.isdisjoint(metadata["config"])
            )
            self.assertEqual(metadata["release_format"], "stable-1.0")

            reloaded = AdaptiveBrain.load(root, "stable-config-brain")
            self.assertEqual(reloaded.config.d_model, 32)
            self.assertEqual(reloaded.config.router_neurons, 24)
            self.assertTrue(reloaded.config.ternary_weights)
            reloaded.events.close()


if __name__ == "__main__":
    unittest.main()
