"""Tiny exact arithmetic checks for backend-safe packed ternary matmul."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (
    _packed_ternary_forward,
    _packed_ternary_input_gradient,
    pack_ternary_weight,
)


class PackedIntegerKernelTests(unittest.TestCase):
    def test_direct_gradient_works_without_decoded_weights_or_torch_matmul(self):
        inputs = torch.tensor([[2., -3.], [-1., 5.]])
        levels = torch.tensor([[1, 0, -1], [-1, 1, 1]], dtype=torch.int8)
        expected = inputs @ levels.float()
        packed = pack_ternary_weight(levels)
        with patch.object(torch.Tensor, "__matmul__", side_effect=AssertionError("Torch matmul forbidden")), \
             patch('omni_core.model.unpack_ternary_weight_rows', side_effect=AssertionError('decoded weights forbidden')):
            actual = _packed_ternary_input_gradient(inputs, packed, 3, 2, torch.tensor(1.))
        self.assertTrue(torch.equal(actual, expected))

    def test_cpu_never_calls_unavailable_int_mm_and_preserves_exact_levels(self):
        levels = torch.tensor([[1, 0, -1, 1], [-1, 1, 1, 0]], dtype=torch.int8)
        packed = pack_ternary_weight(levels)
        before = packed.clone()
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 3.0, 0.0, 1.0]])
        activation_scale = (inputs.abs().amax(dim=-1, keepdim=True) / 127).clamp_min(
            torch.finfo(torch.float32).eps
        )
        a8 = (inputs / activation_scale).round().clamp(-127, 127).to(torch.int32)
        expected = (a8 @ levels.to(torch.int32).t()).float() * activation_scale * 0.5
        with patch.object(torch, "_int_mm", side_effect=AssertionError("CPU _int_mm called"), create=True), \
             patch.object(torch.Tensor, "__matmul__", side_effect=AssertionError("Torch matmul forbidden")), \
             patch('omni_core.model.unpack_ternary_weight_rows', side_effect=AssertionError('decoded weights forbidden')):
            actual = _packed_ternary_forward(inputs, packed, 4, 2, torch.tensor(0.5))
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(packed, before))

    def test_output_blocks_and_cpu_row_chunks_remain_exact(self):
        levels = torch.arange(65 * 5, dtype=torch.int16).remainder(3).sub(1).to(torch.int8)
        levels = levels.reshape(65, 5)
        packed = pack_ternary_weight(levels)
        inputs = torch.arange(257 * 5, dtype=torch.int16).remainder(17).sub(8).to(torch.int8)
        inputs = inputs.reshape(257, 5)
        real_inputs = inputs.float()
        activation_scale = (
            real_inputs.abs().amax(dim=-1, keepdim=True) / 127
        ).clamp_min(torch.finfo(torch.float32).eps)
        a8 = (real_inputs / activation_scale).round().clamp(-127, 127).to(torch.int32)
        expected = (a8 @ levels.to(torch.int32).t()).float() * activation_scale
        with patch.object(torch.Tensor, "__matmul__", side_effect=AssertionError("Torch matmul forbidden")), \
             patch('omni_core.model.unpack_ternary_weight_rows', side_effect=AssertionError('decoded weights forbidden')):
            actual = _packed_ternary_forward(real_inputs, packed, 5, 65, torch.tensor(1.0))
        self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
