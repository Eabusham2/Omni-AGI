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


if __name__ == "__main__":
    unittest.main()
