"""A degenerate sensory bridge must not invent an unrelated memory vector."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.model import PackedAdaptiveBitLinear


class SensoryProjectionNoFabricationTests(unittest.TestCase):
    @staticmethod
    def bridge(weights: torch.Tensor) -> SimpleNamespace:
        projection = PackedAdaptiveBitLinear(int(weights.shape[1]), int(weights.shape[0]), scale=1.0)
        projection.set_ternary_weight_(weights)
        return SimpleNamespace(
            device=torch.device("cpu"),
            config=SimpleNamespace(idea_dim=int(weights.shape[0])),
            memory_bridge=projection,
        )

    def test_zero_bridge_fails_without_making_a_hash_symbol(self):
        brain = self.bridge(torch.zeros((3, 4), dtype=torch.int8))
        with self.assertRaisesRegex(RuntimeError, "not admitted"):
            AdaptiveBrain._sensory_substrate_vector(
                brain, torch.ones(3), "image", "fixture-fingerprint"
            )

    def test_nonzero_ternary_bridge_uses_the_sensory_input(self):
        brain = self.bridge(
            torch.tensor(
                [[1, 0, -1, 0], [0, 1, 0, -1], [1, 1, 0, 0]],
                dtype=torch.int8,
            )
        )
        first = AdaptiveBrain._sensory_substrate_vector(
            brain, torch.tensor([1.0, 2.0, 3.0]), "image", "same-fingerprint"
        )
        second = AdaptiveBrain._sensory_substrate_vector(
            brain, torch.tensor([3.0, 2.0, 1.0]), "image", "same-fingerprint"
        )
        self.assertEqual(tuple(first.shape), (4,))
        self.assertAlmostEqual(float(first.norm()), 1.0, places=6)
        self.assertFalse(torch.allclose(first, second))

    def test_sensory_admission_uses_strengths_and_repeated_contacts_without_decoded_matrix(self):
        weights = torch.tensor([[1, 0, -1, 0], [0, 1, 0, -1], [1, 1, 0, 0]], dtype=torch.int8)
        brain = self.bridge(weights)
        brain.memory_bridge.set_connection_strengths_(torch.tensor([[4, 1, 2, 1], [1, 3, 1, 1], [1, 2, 1, 1]]))
        brain.memory_bridge.add_parallel_connection_(0, 0, sign=1, strength=2)
        brain.memory_bridge.add_parallel_connection_(1, 3, sign=-1, strength=3)
        before = {name: tensor.clone() for name, tensor in brain.memory_bridge.state_dict().items()}
        with patch.object(brain.memory_bridge, "effective_weight", side_effect=AssertionError("expanded weight path used")):
            vector = AdaptiveBrain._sensory_substrate_vector(brain, torch.tensor([1.0, 2.0, 3.0]), "image", "fixture")
        expected = torch.tensor([9.0, 12.0, -2.0, -8.0])
        self.assertTrue(torch.allclose(vector, expected / expected.norm()))
        for name, tensor in brain.memory_bridge.state_dict().items():
            self.assertTrue(torch.equal(before[name], tensor), name)
        self.assertEqual(tuple(brain.memory_bridge.parameters()), ())

    @unittest.skipUnless(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"), "native Metal is unavailable")
    def test_native_metal_reverse_projection_uses_the_same_weighted_state(self):
        brain = self.bridge(torch.tensor([[1, -1, 0], [0, 1, -1]], dtype=torch.int8))
        brain.memory_bridge.set_connection_strengths_(torch.tensor([[4, 3, 1], [1, 2, 5]]))
        brain.memory_bridge.add_parallel_connection_(0, 2, sign=1, strength=2)
        brain.memory_bridge.to("mps")
        brain.device = torch.device("mps")
        vector = AdaptiveBrain._sensory_substrate_vector(brain, torch.tensor([2., 3.], device="mps"), "audio", "fixture")
        expected = torch.tensor([8., 0., -11.])
        self.assertTrue(torch.allclose(vector, expected / expected.norm(), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
