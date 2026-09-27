"""Focused proof for an opt-in packed-authoritative cortical projection."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (  # noqa: E402
    PackedAdaptiveBitLinear,
    pack_ternary_weight,
    packed_runtime_status,
)
from omni_core.ternary_packing import (  # noqa: E402
    collect_module_ternary_tensors,
    export_module_ternary_shards,
    inspect_module_ternary_layout,
    verify_ternary_shards,
)


class PackedAdaptiveLinearTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def test_only_packed_synapses_are_resident_and_reload_exactly(self):
        layer = PackedAdaptiveBitLinear(7, 5, online_learning_rate=0.125)
        self.assertEqual(tuple(layer.parameters()), ())
        self.assertEqual(layer._packed_forward_weight.numel(), 10)
        self.assertEqual(
            set(layer.state_dict()),
            {
                "_packed_forward_weight",
                "_packed_forward_scale",
                "_online_learning_rate",
                "_row_stability",
            },
        )
        self.assertTrue(
            set(layer.effective_weight().flatten().tolist()).issubset({-1, 0, 1})
        )
        self.assertTrue(layer.packed_forward_status()["authoritativePackedWeight"])
        self.assertFalse(layer.packed_forward_status()["latentMasterLearningState"])
        self.assertTrue(packed_runtime_status(layer)["complete"])

        inputs = torch.randn(3, 7)
        expected = layer.eval()(inputs)
        restored = PackedAdaptiveBitLinear(7, 5)
        restored.load_state_dict(layer.state_dict())
        self.assertAlmostEqual(restored.online_learning_rate, 0.125)
        self.assertTrue(torch.equal(restored.effective_weight(), layer.effective_weight()))
        self.assertTrue(torch.equal(restored.eval()(inputs), expected))

    def test_normal_backward_mutates_packed_synapses_and_reduces_tiny_loss(self):
        layer = PackedAdaptiveBitLinear(
            1, 1, scale=1.0, online_learning_rate=100.0
        )
        with torch.no_grad():
            layer._packed_forward_weight.copy_(
                pack_ternary_weight(torch.tensor([[0]], dtype=torch.int8))
            )
        before = layer._packed_forward_weight.clone()
        inputs = torch.ones((1, 1))  # No requires_grad: first-layer case.
        target = torch.ones((1, 1))
        initial_loss = torch.nn.functional.mse_loss(layer(inputs), target)
        self.assertTrue(initial_loss.requires_grad)
        initial_loss.backward()
        self.assertFalse(torch.equal(layer._packed_forward_weight, before))
        self.assertEqual(int(layer.effective_weight()[0, 0]), 1)
        self.assertLess(
            float(torch.nn.functional.mse_loss(layer.eval()(inputs), target)),
            float(initial_loss.detach()),
        )
        self.assertEqual(tuple(layer.parameters()), ())

    def test_input_gradient_and_explicit_local_update_without_master(self):
        layer = PackedAdaptiveBitLinear(2, 1, scale=1.0).eval()
        with torch.no_grad():
            layer._packed_forward_weight.copy_(
                pack_ternary_weight(torch.tensor([[1, -1]], dtype=torch.int8))
            )
        inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)
        layer(inputs).sum().backward()
        self.assertTrue(torch.allclose(inputs.grad, torch.tensor([[1.0, -1.0]])))
        unchanged = layer.effective_weight().clone()
        changed = layer.learn_from_gradient(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[1.0]]),
            100.0,
            generator=torch.Generator().manual_seed(9),
        )
        self.assertEqual(changed, 1)
        self.assertEqual(int(layer.effective_weight()[0, 0]), 0)
        self.assertEqual(int(unchanged[0, 0]), 1)

    def test_bias_is_packed_ternary_and_learns_through_backward(self):
        layer = PackedAdaptiveBitLinear(
            2, 1, bias=True, scale=1.0, online_learning_rate=100.0
        )
        with torch.no_grad():
            layer._packed_forward_weight.copy_(
                pack_ternary_weight(torch.zeros((1, 2), dtype=torch.int8))
            )
        self.assertEqual(int(layer.effective_bias()[0]), 0)
        self.assertEqual(tuple(layer.parameters()), ())
        self.assertEqual(layer.packed_forward_status()["packedBytes"], 2)
        self.assertIn("_packed_forward_bias", layer.state_dict())
        inputs = torch.zeros((1, 2))
        loss = torch.nn.functional.mse_loss(layer(inputs), torch.ones((1, 1)))
        loss.backward()
        self.assertEqual(int(layer.effective_bias()[0]), 1)
        self.assertEqual(int(layer.effective_weight().abs().sum()), 0)
        self.assertEqual(float(layer.eval()(inputs)[0, 0]), 1.0)
        self.assertEqual(
            inspect_module_ternary_layout({"cortex": layer}, dynamic_synapses={}),
            {
                "cortex.bias": ((1,), "projection"),
                "cortex.weight": ((1, 2), "projection"),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            export_module_ternary_shards(
                Path(temporary),
                {"cortex": layer},
                expected_names={"cortex.weight", "cortex.bias"},
            )
            verified = verify_ternary_shards(
                Path(temporary),
                expected_names={"cortex.weight", "cortex.bias"},
            )
            self.assertEqual(int(verified.tensors["cortex.bias"][0]), 1)

    def test_packing_export_and_corrupt_load_rejection(self):
        layer = PackedAdaptiveBitLinear(5, 3)
        layout = inspect_module_ternary_layout(
            {"cortex": layer}, dynamic_synapses={}
        )
        self.assertEqual(layout, {"cortex.weight": ((3, 5), "projection")})
        specs = collect_module_ternary_tensors({"cortex": layer})
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].source_dtype, "packed-2bit")
        self.assertTrue(torch.equal(specs[0].values, layer.effective_weight()))
        with tempfile.TemporaryDirectory() as temporary:
            manifest = export_module_ternary_shards(
                Path(temporary), {"cortex": layer}, expected_names={"cortex.weight"}
            )
            verified = verify_ternary_shards(
                Path(temporary), expected_names={"cortex.weight"}
            )
            self.assertEqual(manifest["format"], "omni-packed-ternary")
            self.assertTrue(
                torch.equal(verified.tensors["cortex.weight"], layer.effective_weight())
            )

        corrupted = {name: value.clone() for name, value in layer.state_dict().items()}
        corrupted["_packed_forward_weight"].fill_(0xFF)
        layer.load_state_dict(corrupted)
        with self.assertRaisesRegex(ValueError, "reserved code"):
            layer(torch.ones((1, 5)))

    def test_fill_is_exact_and_zeroes_packed_bias_without_a_dense_weight(self):
        layer = PackedAdaptiveBitLinear(5, 3, bias=True)
        self.assertEqual(layer.logical_ternary_parameter_count, 18)
        self.assertEqual(len(layer.authoritative_packed_tensors()), 2)
        layer.fill_ternary_(1)
        self.assertTrue(bool((layer.effective_weight() == 1).all()))
        self.assertTrue(bool((layer.effective_bias() == 0).all()))
        with self.assertRaisesRegex(ValueError, "-1, 0, or \\+1"):
            layer.fill_ternary_(2)


if __name__ == "__main__":
    unittest.main()
