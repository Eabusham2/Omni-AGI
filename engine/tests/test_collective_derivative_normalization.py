"""Actual derivative-method/storage stubs: no constructors, updates or ranks."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from omni_core.model import PackedAdaptiveBitLinear, PackedAdaptiveTernaryEmbedding, _PackedAdaptiveConvBase
from omni_core.modalities import PackedTernaryTable
from omni_core.packed_collective_hooks import packed_derivative_sink, packed_derivative_collective_active


def storage_owner(width, rows, *, bias=True):
    value = SimpleNamespace(in_features=width, out_features=rows,
        _packed_forward_weight=torch.full((rows, (width + 3) // 4), 0x55, dtype=torch.uint8),
        _packed_forward_bias=torch.full((1, (rows + 3) // 4), 0x55, dtype=torch.uint8) if bias else None,
        _packed_forward_scale=torch.tensor(1.0), _row_stability=torch.zeros(rows, dtype=torch.uint8),
        _bias_row_stability=torch.zeros(1, dtype=torch.uint8), _packed_stability_strength=0.0,
        _pending_stability_events=0, _online_transaction=None,
        _native_core_pager=None, _native_compute_device=torch.device("cpu"),
        _packed_residency_scope=lambda _device: nullcontext(), online_learning_rate=0.1)
    value.packed_forward_weight = lambda: value._packed_forward_weight
    return value


class Collector:
    def __init__(self):
        self.records = []

    def stage(self, owner, packed, width, row_start, gradient, *_args):
        role = "bias" if packed is owner._packed_forward_bias else "weight"
        self.records.append((role, row_start, gradient.clone()))
        return 0  # Derivative staging only: zero packed mutations.

    def total(self, role, rows, width):
        value = torch.zeros(rows, width)
        for recorded_role, start, gradient in self.records:
            if role == recorded_role:
                value[start:start + gradient.shape[0]].add_(gradient)
        return value


def ordinary_capture(collector, owner):
    def capture(packed, width, row_start, gradient, *_args, **_kwargs):
        return collector.stage(owner, packed, width, row_start, gradient)
    return capture


class CollectiveDerivativeNormalizationFixtures(unittest.TestCase):
    def test_linear_global_objective_sum_is_independent_of_rank_batch_and_zero_padding(self):
        inputs = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        downstream = torch.tensor([[.2, -.1], [.4, .3], [-.2, .6]]) / 7  # Already global label normalized.
        expected = downstream.t() @ inputs
        owner = storage_owner(3, 2)
        merged, split = Collector(), Collector()
        with packed_derivative_sink(merged):
            self.assertTrue(packed_derivative_collective_active())
            PackedAdaptiveBitLinear.learn_from_gradient(owner, inputs, downstream, .1)
        with packed_derivative_sink(split):
            PackedAdaptiveBitLinear.learn_from_gradient(owner, inputs[:1], downstream[:1], .1)
            # The second rank has more physical examples, including a padding
            # row with zero global objective derivative. Do not divide again.
            PackedAdaptiveBitLinear.learn_from_gradient(owner, torch.cat((inputs[1:], torch.ones(1, 3))),
                torch.cat((downstream[1:], torch.zeros(1, 2))), .1)
        self.assertFalse(packed_derivative_collective_active())
        self.assertTrue(torch.allclose(merged.total("weight", 2, 3), expected))
        self.assertTrue(torch.allclose(split.total("weight", 2, 3), expected))
        self.assertTrue(torch.allclose(split.total("bias", 1, 2), downstream.sum(0, keepdim=True)))
        self.assertTrue(bool(owner._packed_forward_weight.eq(0x55).all()))

    def test_ordinary_linear_local_mean_remains_unchanged(self):
        owner, collector = storage_owner(3, 2), Collector()
        inputs, downstream = torch.arange(9).reshape(3, 3).float(), torch.arange(6).reshape(3, 2).float()
        with patch("omni_core.model._apply_packed_gradient_rows", side_effect=ordinary_capture(collector, owner)):
            PackedAdaptiveBitLinear.learn_from_gradient(owner, inputs, downstream, .1)
        self.assertTrue(torch.equal(collector.total("weight", 2, 3), downstream.t() @ inputs / 3))
        self.assertTrue(torch.equal(collector.total("bias", 1, 2), downstream.sum(0, keepdim=True) / 3))

    def test_bounded_online_accumulation_stages_global_derivatives_without_private_mean(self):
        owner, collector = storage_owner(3, 2), Collector()
        owner._online_transaction = {"weight": torch.zeros(2, 3), "bias": torch.zeros(2), "rate": None, "calls": 0}
        owner._commit_online_step_impl = lambda: PackedAdaptiveBitLinear._commit_online_step_impl(owner)
        owner._release_online_pager_scope = lambda: None
        inputs, downstream = torch.arange(9).reshape(3, 3).float(), torch.arange(6).reshape(3, 2).float() / 11
        with packed_derivative_sink(collector):
            PackedAdaptiveBitLinear.learn_from_gradient(owner, inputs[:1], downstream[:1], .1)
            PackedAdaptiveBitLinear.learn_from_gradient(owner, inputs[1:], downstream[1:], .1)
            PackedAdaptiveBitLinear.commit_online_step(owner)
        self.assertTrue(torch.allclose(collector.total("weight", 2, 3), downstream.t() @ inputs))
        self.assertTrue(torch.allclose(collector.total("bias", 1, 2), downstream.sum(0, keepdim=True)))

    def test_embedding_occurrence_and_table_derivatives_are_already_raw_and_stay_raw(self):
        owner, collector = storage_owner(3, 4, bias=False), Collector()
        owner.num_embeddings, owner.embedding_dim, owner.padding_idx = 4, 3, 0
        indices = torch.tensor([1, 1, 2, 0])
        gradient = torch.arange(12).reshape(4, 3).float() / 9
        with packed_derivative_sink(collector):
            PackedAdaptiveTernaryEmbedding.learn_from_gradient(owner, indices[:1], gradient[:1], .1)
            PackedAdaptiveTernaryEmbedding.learn_from_gradient(owner, indices[1:], gradient[1:], .1)
        expected = torch.zeros(4, 3)
        expected[1] = gradient[:2].sum(0); expected[2] = gradient[2]
        self.assertTrue(torch.allclose(collector.total("weight", 4, 3), expected))
        projection, table_collector = storage_owner(4, 3, bias=False), Collector()
        table = SimpleNamespace(rows=4, dimensions=3, projection=projection)
        table._learn_from_gradient_impl = lambda values: PackedTernaryTable._learn_from_gradient_impl(table, values)
        with packed_derivative_sink(table_collector):
            PackedTernaryTable.learn_from_gradient(table, gradient)
        self.assertTrue(torch.equal(table_collector.total("weight", 3, 4), gradient.t()))

    def test_standard_and_transposed_conv_preserve_global_sum_across_heterogeneous_batches(self):
        inputs = torch.arange(18).reshape(3, 2, 3).float() / 10
        downstream = torch.arange(18).reshape(3, 2, 3).float() / 17
        expected = downstream.transpose(1, 2).reshape(-1, 2).t() @ inputs.transpose(1, 2).reshape(-1, 2)
        for transposed in (False, True):
            owner, collector = storage_owner(2, 2), Collector()
            owner.dimensions, owner.in_channels, owner.out_channels, owner.groups = 1, 2, 2, 1
            owner.transposed, owner.kernel_size, owner.stride, owner.padding = transposed, (1,), (1,), (0,)
            owner.dilation, owner.output_padding, owner.padding_mode, owner._matrix_width = (1,), (0,), "zeros", 2
            with packed_derivative_sink(collector):
                _PackedAdaptiveConvBase.learn_from_gradient(owner, inputs[:1], downstream[:1], .1)
                _PackedAdaptiveConvBase.learn_from_gradient(owner, torch.cat((inputs[1:], torch.ones(1, 2, 3))),
                    torch.cat((downstream[1:], torch.zeros(1, 2, 3))), .1)
            self.assertTrue(torch.allclose(collector.total("weight", 2, 2), expected), transposed)
            self.assertTrue(torch.allclose(collector.total("bias", 1, 2), downstream.sum((0, 2))[None]), transposed)

    def test_ordinary_conv_local_position_mean_remains_unchanged(self):
        owner, collector = storage_owner(2, 2), Collector()
        owner.dimensions, owner.in_channels, owner.out_channels, owner.groups = 1, 2, 2, 1
        owner.transposed, owner.kernel_size, owner.stride, owner.padding = False, (1,), (1,), (0,)
        owner.dilation, owner.output_padding, owner.padding_mode, owner._matrix_width = (1,), (0,), "zeros", 2
        inputs, downstream = torch.arange(12).reshape(2, 2, 3).float(), torch.ones(2, 2, 3)
        with patch("omni_core.model._apply_packed_gradient_rows", side_effect=ordinary_capture(collector, owner)):
            _PackedAdaptiveConvBase.learn_from_gradient(owner, inputs, downstream, .1)
        expected = downstream.transpose(1, 2).reshape(-1, 2).t() @ inputs.transpose(1, 2).reshape(-1, 2) / 6
        self.assertTrue(torch.equal(collector.total("weight", 2, 2), expected))
        self.assertTrue(torch.equal(collector.total("bias", 1, 2), torch.ones(1, 2)))


if __name__ == "__main__":
    unittest.main()
