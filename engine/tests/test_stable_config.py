import sys
import unittest
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.config import OmniConfig


RETIRED_BETA_CONTROLS = {
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
    def test_external_beta_controls_cannot_change_mandatory_substrate(self):
        raw = {
            "name": "Stable mind",
            "hardwareTier": "personal",
            "initialNeuronBudget": 1,
            "ternaryWeights": False,
            "spikingDynamics": False,
            "stdpPlasticity": False,
            "liquidDynamics": False,
            "vectorSymbolicMemory": False,
            "consolidation": False,
            "metaplasticity": False,
            "noise": 1.0,
            "curiosityDrive": 0.0,
            "parallelThoughts": 999,
            "maxConcepts": 1,
            "maxIdeas": 1,
            "maxSynapses": 1,
            "maxExperts": 1,
        }
        config = OmniConfig.from_external(raw)
        self.assertEqual(config.router_neurons, 64)
        self.assertTrue(config.ternary_weights)
        self.assertTrue(config.spiking_dynamics)
        self.assertTrue(config.stdp_plasticity)
        self.assertTrue(config.liquid_dynamics)
        self.assertTrue(config.vector_symbolic_memory)
        self.assertTrue(config.consolidation_enabled)
        self.assertTrue(config.metaplasticity)
        self.assertEqual(config.growth_policy, "unbounded")
        self.assertTrue(RETIRED_BETA_CONTROLS.isdisjoint(config.to_dict()))

    def test_extended_workspace_is_hardware_derived_not_a_numeric_slider(self):
        ordinary = OmniConfig.from_external(
            {"hardwareTier": "gpu", "extendedWorkingMemory": False}
        )
        extended = OmniConfig.from_external(
            {
                "hardwareTier": "gpu",
                "extendedWorkingMemory": True,
                "workingMemorySlots": 1,
            }
        )
        self.assertEqual(ordinary.max_seq_len, 256)
        self.assertEqual(extended.max_seq_len, 512)
        self.assertEqual(ordinary.working_memory_slots, 96)
        self.assertEqual(extended.working_memory_slots, 192)

    def test_legacy_internal_dictionary_loads_but_retires_controls_on_save(self):
        config = OmniConfig.from_dict(
            {
                "d_model": 32,
                "idea_dim": 32,
                "n_heads": 4,
                "vsa_dim": 64,
                "router_neurons": 24,
                "max_seq_len": 48,
                "max_concepts": 7,
                "max_ideas": 8,
                "max_synapses": 9,
                "curiosity_drive": 0.1,
                "noise": 0.9,
                "parallel_thoughts": 2,
            }
        )
        self.assertEqual(config.max_concepts, 7)
        self.assertEqual(config.parallel_thoughts, 2)
        self.assertTrue(RETIRED_BETA_CONTROLS.isdisjoint(config.to_dict()))


if __name__ == "__main__":
    unittest.main()
