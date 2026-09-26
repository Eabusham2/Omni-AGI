"""Focused evidence for bounded packed-synapse metaplastic resistance.

The native Build/reload acceptance test is separate; these small tests make
the direct ternary transition and checkpointed resistance contract explicit.
"""

import unittest
from unittest.mock import patch

import torch

from omni_core.model import (
    PackedAdaptiveBitConv1d,
    PackedAdaptiveBitLinear,
    PackedAdaptiveTernaryEmbedding,
    _apply_packed_gradient_rows,
    pack_ternary_weight,
    unpack_ternary_weight_rows,
)


class PackedRowStabilityTests(unittest.TestCase):
    def test_prior_row_activity_reduces_but_does_not_forbid_level_changes(self):
        zero = torch.zeros((2, 8), dtype=torch.int8)
        unstabilized = pack_ternary_weight(zero)
        stabilized = pack_ternary_weight(zero)
        new_row = torch.zeros(2, dtype=torch.uint8)
        old_row = torch.full((2,), 15, dtype=torch.uint8)

        def fixed_draw(shape, **kwargs):
            return torch.full(shape, 0.5, dtype=kwargs["dtype"], device=kwargs["device"])

        with patch.object(torch, "rand", side_effect=fixed_draw):
            new_changes = _apply_packed_gradient_rows(
                unstabilized,
                8,
                0,
                torch.full((2, 8), -0.8),
                1.0,
                torch.tensor(1.0),
                row_stability=new_row,
                stability_strength=0.1,
            )
            old_changes = _apply_packed_gradient_rows(
                stabilized,
                8,
                0,
                torch.full((2, 8), -0.8),
                1.0,
                torch.tensor(1.0),
                row_stability=old_row,
                stability_strength=0.1,
            )
        self.assertEqual(new_changes, 16)
        self.assertEqual(old_changes, 0)
        self.assertTrue(torch.equal(new_row, torch.ones_like(new_row)))
        self.assertTrue(torch.equal(old_row, torch.full_like(old_row, 15)))
        self.assertTrue(
            torch.equal(
                unpack_ternary_weight_rows(unstabilized, 8),
                torch.ones((2, 8), dtype=torch.int8),
            )
        )

    def test_all_packed_families_checkpoint_only_bounded_row_state(self):
        modules = (
            PackedAdaptiveBitLinear(8, 3, bias=True),
            PackedAdaptiveTernaryEmbedding(11, 8),
            PackedAdaptiveBitConv1d(2, 3, 3, bias=True),
        )
        expected_rows = (3, 11, 3)
        for module, rows in zip(modules, expected_rows):
            with self.subTest(module=type(module).__name__):
                module.configure_packed_stability(enabled=True, strength=0.025)
                status = module.packed_stability_status()
                self.assertEqual(module._row_stability.dtype, torch.uint8)
                self.assertEqual(module._row_stability.numel(), rows)
                self.assertEqual(
                    status["metaplasticityCheckpointBytes"],
                    rows + int(module.has_bias),
                )
                checkpoint = module.state_dict()
                self.assertIn("_row_stability", checkpoint)
                self.assertNotIn("weight", checkpoint)
                self.assertTrue(status["metaplasticityEnabled"])
                self.assertFalse(module.packed_forward_status()["latentMasterLearningState"])
                module._row_stability[0] = 3
                reloaded = type(module)(
                    *(
                        (8, 3) if isinstance(module, PackedAdaptiveBitLinear)
                        else (11, 8) if isinstance(module, PackedAdaptiveTernaryEmbedding)
                        else (2, 3, 3)
                    ),
                    **({"bias": True} if module.has_bias else {}),
                )
                reloaded.load_state_dict(module.state_dict(), strict=True)
                self.assertEqual(int(reloaded._row_stability[0].item()), 3)
                reloaded.configure_packed_stability(enabled=False, strength=0.025)
                self.assertFalse(
                    reloaded.packed_stability_status()["metaplasticityEnabled"]
                )


if __name__ == "__main__":
    unittest.main()
