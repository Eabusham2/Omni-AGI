"""A degenerate sensory bridge must not invent an unrelated memory vector."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain


class SensoryProjectionNoFabricationTests(unittest.TestCase):
    @staticmethod
    def bridge(weights: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            device=torch.device("cpu"),
            config=SimpleNamespace(idea_dim=int(weights.shape[0])),
            memory_bridge=SimpleNamespace(effective_weight=lambda: weights),
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


if __name__ == "__main__":
    unittest.main()
