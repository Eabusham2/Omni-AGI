import unittest
from unittest.mock import patch

import torch

from omni_core.model import pack_ternary_weight
from omni_core.packed_native_kernel import (
    native_packed_kernel_available, packed_forward_i8, packed_input_gradient_f32,
)


@unittest.skipUnless(native_packed_kernel_available(), "build native kernel before running native checks")
class NativePackedKernelTest(unittest.TestCase):
    def test_odd_widths_and_every_strength_match_exact_integer_reference(self):
        generator = torch.Generator().manual_seed(91)
        for width in (1, 3, 4, 5, 19, 257):
            signs = torch.randint(-1, 2, (7, width), dtype=torch.int8, generator=generator)
            states = torch.randint(0, 256, signs.shape, dtype=torch.uint8, generator=generator)
            activity = torch.randint(-127, 128, (3, width), dtype=torch.int8, generator=generator)
            expected = activity.to(torch.int64) @ (signs.to(torch.int64) * ((states.to(torch.int64) & 15) + 1)).t()
            with patch("omni_core.model.unpack_ternary_weight_rows", side_effect=AssertionError("weight decoding must stay inside kernel")):
                actual = packed_forward_i8(activity, pack_ternary_weight(signs), states, width, 7)
            self.assertTrue(torch.equal(actual, expected), width)

    def test_unit_strength_uses_original_native_signs(self):
        signs = torch.tensor([[-1, 0, 1], [1, -1, 0]], dtype=torch.int8)
        activity = torch.tensor([[127, -127, 13]], dtype=torch.int8)
        self.assertTrue(torch.equal(packed_forward_i8(activity, pack_ternary_weight(signs), None, 3, 2),
            activity.to(torch.int64) @ signs.to(torch.int64).t()))

    def test_direct_packed_input_gradient(self):
        signs = torch.tensor([[-1, 1, 0], [1, -1, 1]], dtype=torch.int8)
        states = torch.tensor([[0, 1, 255], [15, 240, 8]], dtype=torch.uint8)
        gradient = torch.tensor([[.5, -2], [4, 3]], dtype=torch.float32)
        expected = gradient @ (signs.float() * ((states & 15).float() + 1))
        self.assertTrue(torch.equal(packed_input_gradient_f32(gradient, pack_ternary_weight(signs), states, 3, 2), expected))

    def test_rejects_reserved_signs_and_wrong_buffers(self):
        activity = torch.ones(1, 3, dtype=torch.int8)
        with self.assertRaisesRegex(ValueError, "reserved"):
            packed_forward_i8(activity, torch.tensor([[0x57]], dtype=torch.uint8), None, 3, 1)
        with self.assertRaisesRegex(ValueError, "connection state"):
            packed_forward_i8(activity, torch.tensor([[0x55]], dtype=torch.uint8), torch.ones(2, 3, dtype=torch.uint8), 3, 1)
        with self.assertRaisesRegex(ValueError, "int8 activations"):
            packed_forward_i8(activity.float(), torch.tensor([[0x55]], dtype=torch.uint8), None, 3, 1)

    def test_empty_batch_is_valid(self):
        result = packed_forward_i8(torch.empty(0, 3, dtype=torch.int8), torch.tensor([[0x55]], dtype=torch.uint8), None, 3, 1)
        self.assertEqual(tuple(result.shape), (0, 1))


@unittest.skipUnless(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"), "Metal compiler unavailable")
class MetalPackedKernelTest(unittest.TestCase):
    def test_metal_matches_exact_cpu_for_negative_and_positive_sums(self):
        generator = torch.Generator().manual_seed(122)
        for width in (1, 7, 256, 1001):
            signs = torch.randint(-1, 2, (9, width), dtype=torch.int8, generator=generator)
            states = torch.randint(0, 256, signs.shape, dtype=torch.uint8, generator=generator)
            activity = torch.randint(-127, 128, (4, width), dtype=torch.int8, generator=generator)
            packed = pack_ternary_weight(signs)
            expected = packed_forward_i8(activity, packed, states, width, 9)
            actual = packed_forward_i8(activity.to("mps"), packed.to("mps"), states.to("mps"), width, 9).cpu()
            self.assertTrue(torch.equal(actual, expected), width)
            gradients = torch.randn(4, 9, generator=generator)
            expected_gradient = packed_input_gradient_f32(gradients, packed, states, width, 9)
            actual_gradient = packed_input_gradient_f32(gradients.to("mps"), packed.to("mps"), states.to("mps"), width, 9).cpu()
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-5, atol=2e-5)

    def test_metal_accumulates_beyond_int32_without_float_rounding(self):
        width = 1_100_000
        signs = torch.full((1, (width + 3) // 4), 0xaa, dtype=torch.uint8, device="mps")
        states = torch.full((1, width), 15, dtype=torch.uint8, device="mps")
        activity = torch.full((1, width), 127, dtype=torch.int8, device="mps")
        actual = packed_forward_i8(activity, signs, states, width, 1).cpu().item()
        self.assertEqual(actual, width * 127 * 16)
        self.assertGreater(actual, 2**31)


if __name__ == "__main__":
    unittest.main()
