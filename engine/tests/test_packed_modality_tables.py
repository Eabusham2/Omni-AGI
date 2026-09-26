"""Focused proof that modality tables own and update packed ternary values."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from omni_core.modalities import TernaryLatentTransformer, VectorQuantizer


class PackedModalityTableTests(unittest.TestCase):
    def test_vq_codebook_learns_without_a_float_master(self):
        torch.manual_seed(17)
        quantizer = VectorQuantizer(8, 4)
        quantizer.codebook.projection.online_learning_rate = 100.0
        self.assertEqual(tuple(quantizer.parameters()), ())
        original = quantizer.codebook.projection._packed_forward_weight.clone()

        _quantized, commitment = quantizer(torch.randn(2, 4, 5))
        commitment.backward()

        self.assertFalse(
            torch.equal(original, quantizer.codebook.projection._packed_forward_weight)
        )
        levels = quantizer.codebook.projection.effective_weight()
        self.assertTrue(bool(((levels >= -1) & (levels <= 1)).all()))
        self.assertEqual(levels.dtype, torch.int8)
        self.assertFalse(quantizer.state_dict()["codebook.projection._packed_forward_weight"].is_floating_point())

    def test_learned_positions_update_packed_state_and_reload_exactly(self):
        torch.manual_seed(7)
        model = TernaryLatentTransformer(4, 4, 3, layers=0)
        model.positions.projection.online_learning_rate = 100.0
        self.assertEqual(tuple(model.positions.parameters()), ())
        original = model.positions.projection._packed_forward_weight.clone()
        latent = torch.randn(2, 4, 3)
        idea = torch.randn(2, 4)
        model(latent, idea).square().mean().backward()
        self.assertFalse(
            torch.equal(original, model.positions.projection._packed_forward_weight)
        )

        model.eval()
        expected = model(latent, idea)
        with tempfile.TemporaryDirectory(prefix="omni-ternary-table-") as folder:
            path = Path(folder) / "modality.safetensors"
            save_file(
                {
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                },
                str(path),
            )
            reloaded = TernaryLatentTransformer(4, 4, 3, layers=0)
            reloaded.load_state_dict(load_file(str(path)))
            reloaded.eval()
            self.assertTrue(torch.equal(expected, reloaded(latent, idea)))
            self.assertEqual(
                reloaded.state_dict()["positions.projection._packed_forward_weight"].dtype,
                torch.uint8,
            )


if __name__ == "__main__":
    unittest.main()
