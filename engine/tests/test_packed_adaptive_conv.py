"""Packed-authoritative spatial projections: geometry, learning, and storage."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (  # noqa: E402
    PackedAdaptiveBitConv1d,
    PackedAdaptiveBitConv2d,
    PackedAdaptiveBitConv3d,
    PackedAdaptiveBitConvTranspose1d,
    PackedAdaptiveBitConvTranspose2d,
    PackedAdaptiveBitConvTranspose3d,
    pack_ternary_weight,
    packed_runtime_status,
)
from omni_core.ternary_packing import (  # noqa: E402
    export_module_ternary_shards,
    inspect_module_ternary_layout,
    verify_ternary_shards,
)


class PackedAdaptiveConvolutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def test_all_dimensions_and_transposes_match_dense_ternary_geometry(self):
        cases = (
            (
                PackedAdaptiveBitConv1d(2, 4, 3, stride=2, padding=1, groups=2),
                torch.randn(1, 2, 7),
                F.conv1d,
            ),
            (
                PackedAdaptiveBitConv2d(2, 4, 3, padding=1, groups=2),
                torch.randn(1, 2, 4, 5),
                F.conv2d,
            ),
            (
                PackedAdaptiveBitConv3d(2, 4, (2, 1, 1), groups=2),
                torch.randn(1, 2, 3, 3, 2),
                F.conv3d,
            ),
            (
                PackedAdaptiveBitConvTranspose1d(
                    2, 4, 3, stride=2, padding=1, output_padding=1, groups=2
                ),
                torch.randn(1, 2, 3),
                F.conv_transpose1d,
            ),
            (
                PackedAdaptiveBitConvTranspose2d(
                    2, 4, 3, stride=2, padding=1, output_padding=1, groups=2
                ),
                torch.randn(1, 2, 2, 3),
                F.conv_transpose2d,
            ),
            (
                PackedAdaptiveBitConvTranspose3d(
                    2, 4, (2, 1, 1), groups=2
                ),
                torch.randn(1, 2, 2, 2, 2),
                F.conv_transpose3d,
            ),
        )
        for layer, values, operation in cases:
            with self.subTest(layer=type(layer).__name__):
                self.assertEqual(tuple(layer.parameters()), ())
                self.assertFalse(layer.packed_forward_status()["latentMasterLearningState"])
                self.assertEqual(
                    layer.logical_ternary_parameter_count,
                    layer.effective_weight().numel() + layer.out_channels,
                )
                self.assertTrue(packed_runtime_status(layer)["complete"])
                inputs = values.clone().requires_grad_(True)
                reference_input = values.clone().requires_grad_(True)
                layer.eval()
                actual = layer(inputs)
                reference = operation(
                    reference_input,
                    layer.effective_weight().float()
                    * layer._packed_forward_scale.detach(),
                    layer.bias.detach(),
                    layer.stride,
                    layer.padding,
                    *(
                        (layer.output_padding, layer.groups, layer.dilation)
                        if layer.transposed
                        else (layer.dilation, layer.groups)
                    ),
                )
                self.assertEqual(actual.shape, reference.shape)
                self.assertTrue(
                    torch.allclose(actual, reference, atol=0.07),
                    type(layer).__name__,
                )
                gradient = torch.ones_like(actual)
                actual.backward(gradient)
                reference.backward(gradient)
                self.assertTrue(
                    torch.allclose(inputs.grad, reference_input.grad, atol=1e-5),
                    type(layer).__name__,
                )

    def test_normal_backward_changes_packed_conv_without_adam_parameters(self):
        for layer, inputs in (
            (
                PackedAdaptiveBitConv1d(
                    1, 1, 1, bias=False, scale=1.0, online_learning_rate=100.0
                ),
                torch.ones((1, 1, 1)),
            ),
            (
                PackedAdaptiveBitConvTranspose1d(
                    1, 1, 1, bias=False, scale=1.0, online_learning_rate=100.0
                ),
                torch.ones((1, 1, 1)),
            ),
        ):
            with self.subTest(layer=type(layer).__name__):
                with torch.no_grad():
                    layer._packed_forward_weight.copy_(
                        pack_ternary_weight(torch.zeros((1, 1), dtype=torch.int8))
                    )
                before = layer._packed_forward_weight.clone()
                loss = F.mse_loss(layer(inputs), torch.ones((1, 1, 1)))
                self.assertTrue(loss.requires_grad)
                loss.backward()
                self.assertFalse(torch.equal(layer._packed_forward_weight, before))
                self.assertEqual(int(layer.effective_weight().flatten()[0]), 1)
                self.assertEqual(float(layer.eval()(inputs).flatten()[0]), 1.0)
                self.assertEqual(tuple(layer.parameters()), ())

    def test_packed_bias_export_reload_and_corruption(self):
        layer = PackedAdaptiveBitConv2d(1, 2, 1, bias=True)
        layout = inspect_module_ternary_layout(
            {"vision": layer}, dynamic_synapses={}
        )
        self.assertEqual(
            layout,
            {
                "vision.bias": ((2,), "projection"),
                "vision.weight": ((2, 1, 1, 1), "projection"),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            export_module_ternary_shards(
                Path(temporary),
                {"vision": layer},
                expected_names={"vision.weight", "vision.bias"},
            )
            verified = verify_ternary_shards(
                Path(temporary),
                expected_names={"vision.weight", "vision.bias"},
            )
            self.assertTrue(
                torch.equal(verified.tensors["vision.weight"], layer.effective_weight())
            )
        restored = PackedAdaptiveBitConv2d(1, 2, 1, bias=True)
        restored.load_state_dict(layer.state_dict())
        self.assertTrue(torch.equal(restored.effective_weight(), layer.effective_weight()))
        corrupted = {name: value.clone() for name, value in layer.state_dict().items()}
        corrupted["_packed_forward_bias"].fill_(0xFF)
        restored.load_state_dict(corrupted)
        with self.assertRaisesRegex(ValueError, "reserved code"):
            restored(torch.ones((1, 1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
