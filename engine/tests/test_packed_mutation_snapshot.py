"""Small packed-only rollback checks; no brain construction or corpus training."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (
    PackedAdaptiveBitConv1d,
    PackedAdaptiveBitLinear,
    PackedAdaptiveTernaryEmbedding,
)
from omni_core.optimizers import PackedMutationSnapshot


class PackedMutationSnapshotTests(unittest.TestCase):
    def test_row_resistance_corruption_invalidates_all_packed_forward_caches(self):
        modules = (
            PackedAdaptiveBitLinear(2, 2, bias=True),
            PackedAdaptiveTernaryEmbedding(3, 2),
            PackedAdaptiveBitConv1d(1, 1, 1),
        )
        for module in modules:
            with self.subTest(module=type(module).__name__):
                module.packed_forward_weight()
                module._row_stability.fill_(255)
                with self.assertRaisesRegex(ValueError, "stability state is invalid"):
                    module.packed_forward_weight()

    def test_restores_multiple_packed_owners_and_row_resistance(self):
        linear = PackedAdaptiveBitLinear(2, 2, bias=True)
        embedding = PackedAdaptiveTernaryEmbedding(3, 2)
        root = nn.ModuleList([linear, embedding])
        original = {
            key: value.clone() for key, value in root.state_dict().items()
        }
        reserved = []
        snapshot = PackedMutationSnapshot.capture(
            [root], reserve=lambda size: reserved.append(size)
        )
        self.assertEqual(reserved, [snapshot.byte_count])
        self.assertGreater(snapshot.byte_count, 0)
        linear.fill_ternary_(1)
        embedding.fill_ternary_(1)
        linear._row_stability.fill_(3)
        embedding._row_stability.fill_(4)
        linear._pending_stability_events = 5
        snapshot.restore()
        for key, value in original.items():
            self.assertTrue(torch.equal(root.state_dict()[key], value), key)
        self.assertEqual(linear._pending_stability_events, 0)

    def test_failed_microbatch_can_retry_from_unchanged_codes(self):
        layer = PackedAdaptiveBitLinear(
            1, 1, scale=1.0, online_learning_rate=100.0
        )
        layer.fill_ternary_(0)
        snapshot = PackedMutationSnapshot.capture([layer])
        layer(torch.ones((1, 1))).sum().backward()
        self.assertEqual(int(layer.effective_weight()[0, 0]), -1)
        snapshot.restore()
        self.assertEqual(int(layer.effective_weight()[0, 0]), 0)
        layer(torch.ones((1, 1))).sum().backward()
        self.assertEqual(int(layer.effective_weight()[0, 0]), -1)

    def test_reserve_refusal_precedes_any_copy_or_mutation(self):
        layer = PackedAdaptiveBitLinear(1, 1)
        before = layer.packed_forward_weight().clone()
        with self.assertRaisesRegex(RuntimeError, "reserve refused"):
            PackedMutationSnapshot.capture(
                [layer],
                reserve=lambda _size: (_ for _ in ()).throw(
                    RuntimeError("reserve refused")
                ),
            )
        self.assertTrue(torch.equal(layer.packed_forward_weight(), before))


if __name__ == "__main__":
    unittest.main()
