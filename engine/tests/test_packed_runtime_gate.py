"""Focused packed-runtime coverage checks without building a brain."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.model import (
    PackedAdaptiveBitLinear,
    packed_runtime_status,
    require_packed_runtime_complete,
)


class PackedRuntimeGateTests(unittest.TestCase):
    def test_trainable_float_scalar_is_a_runtime_blocker(self):
        class FloatingControl(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(0.15))

        status = packed_runtime_status(FloatingControl())
        self.assertEqual(status["floatingLearnedParameterBlockers"], ["gain"])
        self.assertFalse(status["complete"])
        with self.assertRaisesRegex(RuntimeError, "gain"):
            require_packed_runtime_complete(FloatingControl())

    def test_bare_linear_is_a_runtime_blocker(self):
        module = torch.nn.Sequential(PackedAdaptiveBitLinear(3, 4), torch.nn.Linear(4, 2))
        status = packed_runtime_status(module)
        self.assertEqual(status["packedBitLinearModules"], 1)
        self.assertEqual(status["denseLinearBlockers"], ["1"])
        self.assertFalse(status["complete"])
        with self.assertRaisesRegex(RuntimeError, "1"):
            require_packed_runtime_complete(module)

    def test_frozen_floating_weight_cannot_bypass_native_representation_gate(self):
        class FrozenProjection(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frozen_weight = torch.nn.Parameter(torch.ones(4, 3), requires_grad=False)
            def forward(self, value):
                return value @ self.frozen_weight.t()

        module = FrozenProjection()
        audit = packed_runtime_status(module)
        self.assertEqual(audit["floatingLearnedParameterBlockers"], ["frozen_weight"])
        self.assertFalse(audit["complete"])
        with self.assertRaisesRegex(RuntimeError, "frozen_weight"):
            require_packed_runtime_complete(module)

    def test_brain_card_aggregates_dense_linear_across_roots(self):
        brain = object.__new__(AdaptiveBrain)
        roots = {
            "decoder": torch.nn.Sequential(PackedAdaptiveBitLinear(3, 4)),
            "modalities": torch.nn.Sequential(torch.nn.Linear(3, 4)),
        }
        with patch.object(AdaptiveBrain, "_ternary_export_roots", return_value=roots):
            status = brain.packed_runtime_audit()
            self.assertFalse(status["complete"])
            self.assertEqual(status["denseLinearBlockers"], ["modalities.0"])
            with self.assertRaisesRegex(RuntimeError, "modalities.0"):
                brain.require_complete_packed_runtime()

    def test_bf16_flag_describes_actual_dense_dtype(self):
        fp32 = packed_runtime_status(torch.nn.Linear(3, 4))
        bf16 = packed_runtime_status(torch.nn.Linear(3, 4).to(torch.bfloat16))
        self.assertFalse(fp32["denseBf16LinearWeightMaterialized"])
        self.assertTrue(bf16["denseBf16LinearWeightMaterialized"])

    def test_detached_or_unregistered_float_projection_mirrors_are_blocked(self):
        for registered in (True, False):
            module = PackedAdaptiveBitLinear(3, 4)
            mirror = torch.zeros(4, 3, dtype=torch.bfloat16)
            if registered:
                module.register_buffer("detached_weight_mirror", mirror)
            else:
                module.detached_weight_mirror = mirror
            audit = packed_runtime_status(module)
            self.assertFalse(audit["complete"])
            self.assertEqual(audit["residentFloatMasterBlockers"], ["<root>.detached_weight_mirror"])

    def test_native_scalar_controls_are_not_mistaken_for_weight_mirrors(self):
        module = PackedAdaptiveBitLinear(3, 4)
        self.assertTrue(packed_runtime_status(module)["complete"])
        module._buffers["_packed_forward_scale"] = torch.ones(4, 3)
        self.assertEqual(packed_runtime_status(module)["residentFloatMasterBlockers"], ["<root>._packed_forward_scale"])


if __name__ == "__main__":
    unittest.main()
