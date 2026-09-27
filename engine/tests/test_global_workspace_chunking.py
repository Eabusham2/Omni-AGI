from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import model as model_module
from omni_core.model import (
    GlobalWorkspace,
    PACKED_AUTHORITATIVE_PROJECTION_TYPES,
    packed_online_step,
)


def original_workspace_forward(
    workspace: GlobalWorkspace,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor | None,
):
    """The pre-chunk implementation, kept independent of the new loop."""

    latents = workspace.latents.unsqueeze(0).expand(inputs.shape[0], -1, -1)
    keys = workspace.key(inputs)
    values = workspace.value(inputs)
    for _ in range(workspace.iterations):
        queries = workspace.query(workspace.norm(latents))
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(float(workspace.dimensions))
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask[:, None, :],
                -torch.inf,
            )
        attention = F.softmax(scores.float(), dim=-1).to(inputs.dtype)
        latents = latents + workspace.update(
            torch.matmul(attention, values)
        )
    summary = workspace.broadcast(workspace.norm(latents)).mean(dim=1)
    return latents, summary


class GlobalWorkspaceChunkingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1701)
        torch.set_num_threads(1)

    def test_summary_chunks_preserve_original_mean_outputs_and_gradients(self):
        reference = GlobalWorkspace(dimensions=8, slots=9, iterations=3).train()
        chunked = GlobalWorkspace(dimensions=8, slots=9, iterations=3).train()
        chunked.load_state_dict(reference.state_dict(), strict=True)
        for workspace in (reference, chunked):
            for module in workspace.modules():
                if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES):
                    module.online_learning_rate = 0.0
        chunked.query_chunk_slots = 2

        base_inputs = torch.randn(2, 6, 8)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
            ]
        )
        reference_inputs = base_inputs.clone().requires_grad_(True)
        chunked_inputs = base_inputs.clone().requires_grad_(True)
        summary_probe = torch.linspace(-0.5, 0.5, 16).reshape(2, 8)

        _reference_latents, reference_summary = original_workspace_forward(
            reference,
            reference_inputs,
            mask,
        )
        chunked_summary = chunked.summarize(
            chunked_inputs,
            attention_mask=mask,
        )
        self.assertEqual(
            type(chunked_summary.grad_fn).__name__,
            "_SequentialWorkspaceSummaryBackward",
        )
        torch.testing.assert_close(
            chunked_summary,
            reference_summary,
            rtol=1e-6,
            atol=1e-7,
        )

        reference_loss = (reference_summary * summary_probe).sum()
        chunked_loss = (chunked_summary * summary_probe).sum()
        # Exact derivative parity is defined at one fixed packed state.
        with packed_online_step((reference,)):
            reference_loss.backward()
        chunked_loss.backward()
        self.assertEqual(
            chunked.last_chunk_backward_mode,
            "exact-bounded-packed-transaction",
        )
        torch.testing.assert_close(
            chunked_inputs.grad,
            reference_inputs.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        self.assertEqual(tuple(reference.named_parameters()), ())
        self.assertEqual(tuple(chunked.named_parameters()), ())
        self.assertIn("query._packed_forward_weight", chunked.state_dict())

        repeated = GlobalWorkspace(dimensions=8, slots=9, iterations=3).train()
        repeated.load_state_dict(chunked.state_dict(), strict=True)
        repeated.query_chunk_slots = chunked.query_chunk_slots
        for workspace in (chunked, repeated):
            for module in workspace.modules():
                if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES):
                    module.online_learning_rate = 100.0
        before_learning = {
            name: value.clone()
            for name, value in chunked.state_dict().items()
            if name.endswith("._packed_forward_weight")
        }
        chunked.zero_grad(set_to_none=True)
        continued_inputs = base_inputs.clone().requires_grad_(True)
        repeated_inputs = base_inputs.clone().requires_grad_(True)
        torch.manual_seed(1702)
        continued_summary = chunked.summarize(
            continued_inputs,
            attention_mask=mask,
        )
        torch.manual_seed(1702)
        repeated_summary = repeated.summarize(
            repeated_inputs,
            attention_mask=mask,
        )
        # The reloaded copy starts with the same authoritative packed state.
        self.assertTrue(torch.equal(repeated_summary, continued_summary))
        torch.manual_seed(1703)
        (continued_summary * summary_probe).sum().backward()
        torch.manual_seed(1703)
        (repeated_summary * summary_probe).sum().backward()
        self.assertTrue(torch.equal(repeated_inputs.grad, continued_inputs.grad))
        for name, chunked_parameter in chunked.named_parameters():
            repeated_parameter = dict(repeated.named_parameters())[name]
            self.assertTrue(
                torch.equal(repeated_parameter.grad, chunked_parameter.grad),
                name,
            )
        for name, current in chunked.state_dict().items():
            self.assertTrue(torch.equal(current, repeated.state_dict()[name]), name)
        self.assertTrue(
            any(
                not torch.equal(before, chunked.state_dict()[name])
                for name, before in before_learning.items()
            ),
            "online backward must change packed workspace synapses",
        )

    def test_sequential_execution_visits_every_slot_in_order_and_is_bounded(
        self,
    ):
        workspace = GlobalWorkspace(dimensions=8, slots=10, iterations=2).train()
        workspace.query_chunk_slots = 3
        inputs = torch.randn(2, 7, 8, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, False, False],
            ]
        )
        query_widths = []
        summary_chunks = []

        def observe_query_width(_module, arguments):
            query_widths.append(int(arguments[0].shape[1]))

        original_summary_chunk = workspace._summary_chunk

        def observe_summary_chunk(
            replay_inputs, replay_mask, start, end
        ):
            self.assertTrue(torch.equal(replay_inputs, inputs))
            self.assertTrue(torch.equal(replay_mask, mask))
            summary_chunks.append(
                (
                    int(start),
                    int(end),
                    int(replay_inputs.shape[1]),
                    tuple(replay_mask.shape),
                )
            )
            return original_summary_chunk(
                replay_inputs, replay_mask, start, end
            )

        handle = workspace.query.register_forward_pre_hook(observe_query_width)
        try:
            with patch.object(
                workspace,
                "_ordered_summary",
                wraps=workspace._ordered_summary,
            ) as ordered_call, patch.object(
                workspace,
                "_summary_chunk",
                side_effect=observe_summary_chunk,
            ):
                summary = workspace.summarize(inputs, attention_mask=mask)
                summary.square().mean().backward()
        finally:
            handle.remove()

        expected_chunks = math.ceil(
            workspace.slots / workspace.query_chunk_slots
        )
        ordered_call.assert_called_once_with(
            inputs,
            mask,
            release_mps_cache=True,
        )
        self.assertEqual(len(summary_chunks), expected_chunks)
        self.assertTrue(query_widths)
        self.assertLessEqual(max(query_widths), workspace.query_chunk_slots)
        expected_ranges = [
            (
                start,
                min(workspace.slots, start + workspace.query_chunk_slots),
            )
            for start in range(
                0, workspace.slots, workspace.query_chunk_slots
            )
        ]
        self.assertEqual(
            [(start, end) for start, end, _tokens, _mask in summary_chunks],
            expected_ranges,
        )
        self.assertTrue(
            all(tokens == inputs.shape[1] for _, _, tokens, _ in summary_chunks)
        )
        self.assertTrue(
            all(shape == tuple(mask.shape) for _, _, _, shape in summary_chunks)
        )
        ordered_widths = [
            end - start
            for start, end in expected_ranges
            for _ in range(workspace.iterations)
        ]
        self.assertEqual(query_widths, ordered_widths + ordered_widths)

    def test_ordinary_forward_still_returns_the_original_full_latents(self):
        workspace = GlobalWorkspace(
            dimensions=8, slots=10, iterations=2
        ).train()
        workspace.query_chunk_slots = 3
        inputs = torch.randn(2, 7, 8, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, False, False],
            ]
        )
        expected_latents, expected_summary = original_workspace_forward(
            workspace, inputs, mask
        )
        query_widths = []
        handle = workspace.query.register_forward_pre_hook(
            lambda _module, arguments: query_widths.append(
                int(arguments[0].shape[1])
            )
        )
        try:
            with patch.object(
                workspace,
                "_ordered_summary",
                side_effect=AssertionError(
                    "ordinary forward must not use summary-only execution"
                ),
            ), patch.object(
                workspace,
                "_summary_chunk",
                side_effect=AssertionError(
                    "ordinary forward must not use sequential recomputation"
                ),
            ):
                actual_latents, actual_summary = workspace(
                    inputs, attention_mask=mask
                )
        finally:
            handle.remove()

        self.assertEqual(
            query_widths,
            [workspace.slots for _ in range(workspace.iterations)],
        )
        torch.testing.assert_close(
            actual_latents,
            expected_latents,
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            actual_summary,
            expected_summary,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_none_mask_frozen_and_unused_gradients_do_not_mutate_leaf_grads(
        self,
    ):
        workspace = GlobalWorkspace(
            dimensions=8, slots=10, iterations=2
        ).train()
        workspace.query_chunk_slots = 3
        workspace.unused = nn.Parameter(torch.randn(3))
        workspace.query.online_learning_rate = 0.0
        frozen_query = tuple(
            tensor.clone() for tensor in workspace.query.authoritative_packed_tensors()
        )
        inputs = torch.randn(2, 6, 8, requires_grad=True)

        input_sentinel = torch.full_like(inputs, 11.0)
        inputs.grad = input_sentinel.clone()
        active_parameters = tuple(
            parameter
            for parameter in workspace.parameters()
            if parameter.requires_grad
        )
        parameter_sentinels = {}
        for parameter in active_parameters:
            sentinel = torch.full_like(parameter, 13.0)
            parameter.grad = sentinel.clone()
            parameter_sentinels[id(parameter)] = sentinel

        summary = workspace.summarize(inputs, attention_mask=None)
        gradients = torch.autograd.grad(
            summary.square().mean(),
            (inputs, *active_parameters),
            allow_unused=True,
        )

        self.assertIsNotNone(gradients[0])
        self.assertTrue(torch.isfinite(gradients[0]).all())
        gradient_by_parameter = {
            id(parameter): gradient
            for parameter, gradient in zip(
                active_parameters, gradients[1:]
            )
        }
        self.assertIsNone(gradient_by_parameter[id(workspace.unused)])
        for name, parameter in workspace.named_parameters():
            if parameter is workspace.unused:
                self.assertIsNone(gradient_by_parameter[id(parameter)], name)
            else:
                gradient = gradient_by_parameter[id(parameter)]
                self.assertIsNotNone(gradient, name)
                self.assertTrue(torch.isfinite(gradient).all(), name)

        self.assertTrue(
            all(
                torch.equal(previous, current)
                for previous, current in zip(
                    frozen_query, workspace.query.authoritative_packed_tensors()
                )
            )
        )

        self.assertTrue(torch.equal(inputs.grad, input_sentinel))
        for parameter in active_parameters:
            self.assertTrue(
                torch.equal(
                    parameter.grad,
                    parameter_sentinels[id(parameter)],
                )
            )

    def test_sequential_summary_is_explicitly_first_order(self):
        workspace = GlobalWorkspace(
            dimensions=6, slots=7, iterations=2
        ).train()
        workspace.query_chunk_slots = 2
        workspace.requires_grad_(False)
        inputs = torch.randn(2, 5, 6, requires_grad=True)

        summary = workspace.summarize(inputs)
        input_gradient = torch.autograd.grad(
            summary.square().mean(),
            inputs,
            create_graph=True,
        )[0]

        self.assertTrue(input_gradient.requires_grad)
        self.assertIsNotNone(input_gradient.grad_fn)
        with self.assertRaisesRegex(RuntimeError, "once_differentiable"):
            input_gradient.sum().backward()

    def test_workspace_cache_release_never_clears_live_mps_graph_cache(self):
        empty_cache = Mock()
        fake_mps = SimpleNamespace(empty_cache=empty_cache)
        workspace = GlobalWorkspace(
            dimensions=6, slots=7, iterations=2
        ).train()
        workspace.query_chunk_slots = 2
        inputs = torch.randn(2, 5, 6, requires_grad=True)

        with patch.object(model_module.torch, "mps", fake_mps):
            workspace.summarize(inputs).sum().backward()
            model_module._release_mps_workspace_cache(
                torch.device("cpu")
            )
            model_module._release_mps_workspace_cache(
                torch.device("cuda")
            )
            empty_cache.assert_not_called()

            model_module._release_mps_workspace_cache(
                torch.device("mps")
            )
            empty_cache.assert_not_called()

    def test_chunk_execution_control_is_not_serialized(self):
        workspace = GlobalWorkspace(
            dimensions=8, slots=10, iterations=2
        ).train()
        workspace.query_chunk_slots = 3
        inputs = torch.randn(2, 7, 8)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, False, False],
            ]
        )

        state = workspace.state_dict()
        for part in (
            "latent_table", "query", "key", "value", "update", "broadcast"
        ):
            self.assertIn(f"{part}._packed_forward_weight", state)
            self.assertIn(f"{part}._packed_forward_scale", state)
            self.assertIn(f"{part}._online_learning_rate", state)
        self.assertGreaterEqual(
            sum(name.endswith("._packed_forward_weight") for name in state), 6
        )
        self.assertNotIn("latents", state)
        self.assertNotIn("query.weight", state)
        self.assertIn("norm.scale_delta._packed_forward_weight", state)
        self.assertNotIn("norm.scale", state)
        self.assertFalse(
            any("query_chunk_slots" in key for key in state)
        )
        restored = GlobalWorkspace(
            dimensions=workspace.dimensions,
            slots=workspace.slots,
            iterations=workspace.iterations,
        ).eval()
        restored.load_state_dict(state, strict=True)
        restored.query_chunk_slots = 3
        inference_widths = []
        inference_handle = restored.query.register_forward_pre_hook(
            lambda _module, arguments: inference_widths.append(
                int(arguments[0].shape[1])
            )
        )
        try:
            with torch.no_grad():
                first = restored.summarize(
                    inputs.detach(), attention_mask=mask
                )
                second = restored.summarize(
                    inputs.detach(), attention_mask=mask
                )
        finally:
            inference_handle.remove()

        self.assertLessEqual(max(inference_widths), restored.query_chunk_slots)
        self.assertTrue(torch.equal(first, second))
        with torch.no_grad():
            forward_latents, forward_summary = restored(
                inputs.detach(), attention_mask=mask
            )
        self.assertEqual(
            tuple(forward_latents.shape),
            (inputs.shape[0], restored.slots, restored.dimensions),
        )
        torch.testing.assert_close(
            first,
            forward_summary,
            rtol=1e-6,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
