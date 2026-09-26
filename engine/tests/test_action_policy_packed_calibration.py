"""Small action-head checks independent of a full ground-up brain Build."""

import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (  # noqa: E402
    ActionPolicyHead,
    PackedAdaptiveBitLinear,
    pack_ternary_weight,
    packed_online_step,
)
from omni_core.optimizers import adamw_for_remaining_parameters  # noqa: E402


class PackedActionPolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(97)
        torch.set_num_threads(1)

    def test_uniform_gain_never_changes_selected_action(self):
        head = ActionPolicyHead(16).eval()
        features = torch.randn(5, 16)
        with torch.no_grad():
            raw = head.projection(F.silu(head.hidden(head.norm(features))))
            actual = head(features)
        self.assertEqual(head.logit_gain, 2.0)
        self.assertTrue(torch.allclose(actual, raw * 2.0))
        self.assertTrue(torch.equal(actual.argmax(-1), raw.argmax(-1)))
        self.assertEqual(tuple(head.hidden.parameters()), ())
        self.assertEqual(tuple(head.projection.parameters()), ())

    def test_packed_action_synapses_change_and_loss_falls_on_tiny_fixture(self):
        head = ActionPolicyHead(8).train()
        features = torch.eye(8) * 3.0
        targets = torch.arange(8)
        projections = [
            module for module in head.modules()
            if isinstance(module, PackedAdaptiveBitLinear)
        ]
        before = [
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors())
            for module in projections
        ]
        optimizer = adamw_for_remaining_parameters(
            head.parameters(), lr=0.02, weight_decay=1e-5
        )
        with torch.no_grad():
            initial = float(F.cross_entropy(head(features), targets))
        for _ in range(96):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(features), targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = float(F.cross_entropy(head(features), targets))
        self.assertLess(final, initial)
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for module, snapshot in zip(projections, before)
                for previous, current in zip(
                    snapshot, module.authoritative_packed_tensors()
                )
            )
        )

    def test_reused_head_commits_once_after_backward_and_discards_on_error(self):
        head = ActionPolicyHead(8).train()
        features = torch.eye(8) * 3.0
        targets = torch.arange(8)
        projections = [head.hidden, head.projection]
        before = [
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors())
            for module in projections
        ]
        with packed_online_step((head,), max_scratch_synapses=1000):
            loss = F.cross_entropy(head(features), targets)
            loss = loss + F.cross_entropy(head(features * 0.8), targets)
            loss.backward()
            self.assertTrue(
                all(
                    torch.equal(previous, current)
                    for module, snapshot in zip(projections, before)
                    for previous, current in zip(
                        snapshot, module.authoritative_packed_tensors()
                    )
                )
            )
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for module, snapshot in zip(projections, before)
                for previous, current in zip(
                    snapshot, module.authoritative_packed_tensors()
                )
            )
        )
        committed = [
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors())
            for module in projections
        ]
        with self.assertRaisesRegex(RuntimeError, "cancel candidate"):
            with packed_online_step((head,), max_scratch_synapses=1000):
                F.cross_entropy(head(features), targets).backward()
                raise RuntimeError("cancel candidate")
        self.assertTrue(
            all(
                torch.equal(previous, current)
                for module, snapshot in zip(projections, committed)
                for previous, current in zip(
                    snapshot, module.authoritative_packed_tensors()
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "bounded scratch"):
            with packed_online_step((head,), max_scratch_synapses=1):
                pass

    def test_repeated_opposing_uses_cancel_before_a_discrete_flip(self):
        layer = PackedAdaptiveBitLinear(
            1, 1, scale=1.0, online_learning_rate=100.0
        )
        with torch.no_grad():
            layer._packed_forward_weight.copy_(
                pack_ternary_weight(torch.zeros((1, 1), dtype=torch.int8))
            )
        initial = layer._packed_forward_weight.clone()
        with packed_online_step((layer,), max_scratch_synapses=1):
            first = layer(torch.ones((1, 1)))
            second = layer(torch.ones((1, 1)))
            (first - second).sum().backward()
        self.assertTrue(torch.equal(layer._packed_forward_weight, initial))

    def test_wider_packed_head_learns_new_routes_without_forgetting_old_ones(self):
        head = ActionPolicyHead(8).train()
        self.assertEqual(head.hidden_width, 32)
        self.assertEqual(
            head.hidden.logical_ternary_parameter_count
            + head.projection.logical_ternary_parameter_count,
            552,
        )
        old_features = torch.eye(8) * 3.0
        new_features = -torch.eye(8) * 3.0
        old_targets = torch.arange(8)
        new_targets = (torch.arange(8) + 1) % 8
        optimizer = adamw_for_remaining_parameters(
            head.parameters(), lr=0.02, weight_decay=0.0
        )
        for _ in range(128):
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(head(old_features), old_targets).backward()
            optimizer.step()
        with torch.no_grad():
            self.assertTrue(
                bool((head(old_features).argmax(-1) == old_targets).all())
            )
        for _ in range(64):
            optimizer.zero_grad(set_to_none=True)
            with packed_online_step((head,)):
                loss = F.cross_entropy(head(old_features), old_targets)
                loss = loss + F.cross_entropy(head(new_features), new_targets)
                loss.backward()
            optimizer.step()
        with torch.no_grad():
            old_probabilities = head(old_features).softmax(-1).gather(
                1, old_targets[:, None]
            )
            new_probabilities = head(new_features).softmax(-1).gather(
                1, new_targets[:, None]
            )
        self.assertGreater(float(old_probabilities.min()), 0.95)
        self.assertGreater(float(new_probabilities.min()), 0.95)


if __name__ == "__main__":
    unittest.main()
