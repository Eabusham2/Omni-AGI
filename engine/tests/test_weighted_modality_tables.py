"""Small exact native table contracts, never a user brain or full training."""

import copy
from contextlib import ExitStack
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from omni_core.config import OmniConfig
from omni_core.modalities import ModalityHub, PackedTernaryTable, TinyAudioCodec, VectorQuantizer, _codec_token_logits
from omni_core.native_core_paging import NativeCorePager
from omni_core.packed_collective_hooks import packed_derivative_sink
from omni_core.packed_connections import encode_connection_state


class WeightedModalityTableTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def weighted_table(self):
        table = PackedTernaryTable(5, 3, scale=.25)
        signs = torch.tensor([[1, 0, -1, 1, 0], [0, 1, -1, 0, 1], [-1, 1, 0, 1, -1]], dtype=torch.int8)
        strengths = torch.tensor([[3, 1, 4, 2, 1], [1, 2, 5, 1, 6], [2, 7, 1, 3, 8]], dtype=torch.int16)
        table.projection.set_ternary_weight_(signs)
        table.projection.set_connection_strengths_(strengths)
        table.projection.add_parallel_connection_(0, 1, sign=1, strength=5)
        table.projection.add_parallel_connection_(0, 1, sign=-1, strength=2)
        table.projection.add_parallel_connection_(2, 3, sign=-1, strength=4)
        expected = signs.to(torch.float32) * strengths
        expected[0, 1] += 3
        expected[2, 3] -= 4
        return table, expected.t().contiguous() * .25

    def test_lookup_and_exact_scores_include_gains_and_repeated_contacts_without_expansion(self):
        table, expected = self.weighted_table()
        selected = torch.tensor([[3, 1, 3], [0, 4, 2]])
        inputs = torch.tensor([[[.17, .21, -.73], [1.2, -.23, .1]]], requires_grad=True)
        with patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')):
            values = table(selected)
            logits = table.scores(inputs)
            self.assertTrue(torch.equal(values, expected[selected]))
            self.assertTrue(torch.allclose(logits, 2 * inputs @ expected.t() - expected.square().sum(-1), atol=1e-6))
            self.assertTrue(torch.equal(table.nearest_indices(inputs), logits.argmax(-1)))
            gradient = torch.arange(logits.numel()).reshape_as(logits).float() / 11
            logits.backward(gradient)
            self.assertTrue(torch.allclose(inputs.grad, 2 * gradient @ expected, atol=1e-6))
        self.assertEqual(tuple(table.parameters()), ())

    def test_audio_logits_and_input_gradient_match_weighted_distance_reference(self):
        table, expected = self.weighted_table()
        predicted = torch.tensor([[[.11, .9], [-.7, 1.3], [.27, -.3]]], requires_grad=True)
        before = {key: value.clone() for key, value in table.state_dict().items()}
        scores = _codec_token_logits(predicted, table)
        reference_input = predicted.detach().clone().requires_grad_()
        reference = (2 * reference_input.transpose(1, 2) @ expected.t() - expected.square().sum(-1)) * (16 / 3**.5)
        self.assertTrue(torch.allclose(scores, reference, atol=2e-6))
        scores.square().mean().backward()
        reference.square().mean().backward()
        self.assertTrue(torch.allclose(predicted.grad, reference_input.grad, atol=1e-4))
        for key, value in table.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_selected_backward_updates_strength_resistance_and_parallel_contacts_only_at_touched_codes(self):
        table = PackedTernaryTable(3, 2, scale=1.)
        table.projection.fill_ternary_(1)
        strengths = torch.ones((2, 3), dtype=torch.int16)
        strengths[0, 1] = 16
        table.projection.set_connection_strengths_(strengths)
        old_key = table.projection.add_parallel_connection_(0, 1, sign=1, strength=2)
        before = table(torch.tensor([0, 2])).detach().clone()
        states = table.projection._connection_state[:, [0, 2]].clone()
        table.projection.online_learning_rate = 100.
        table(torch.tensor([1, 1])).sum().neg().backward()
        self.assertTrue(torch.equal(table(torch.tensor([0, 2])), before))
        self.assertTrue(torch.equal(table.projection._connection_state[:, [0, 2]], states))
        self.assertEqual(int(table.projection.connection_strengths()[1, 1]), 2)
        self.assertGreater(int(table.projection._connection_state[1, 1]) >> 4, 0)
        records = table.projection.parallel_connection_records()
        old = next(item for item in records if item[0] == old_key)
        self.assertEqual((old[-1] & 15) + 1, 3)
        self.assertEqual(len(records), 2)
        self.assertEqual(float(table(torch.tensor([1]))[0, 0].detach()), 20.)

    def test_connection_resistance_changes_update_probability(self):
        table = PackedTernaryTable(1, 2, scale=1.)
        table.projection.fill_ternary_(0)
        table.projection.online_learning_rate = .1
        table.projection._connection_state.copy_(encode_connection_state(
            torch.ones((2, 1), dtype=torch.int16), torch.tensor([[0], [15]], dtype=torch.int16)))
        with patch('omni_core.model.torch.rand', side_effect=lambda shape, **kwargs: torch.full(shape, .07, **{key: val for key, val in kwargs.items() if key != 'generator'})):
            table(torch.tensor([0])).sum().backward()
        self.assertEqual(table(torch.tensor([0])).tolist(), [[-1., 0.]])

    def test_selected_derivatives_preserve_collective_orientation_and_duplicate_sum(self):
        table = PackedTernaryTable(5, 3, scale=1.)
        class Sink:
            def __init__(self): self.records = []
            def stage(self, owner, packed, width, start, gradient, *_args):
                self.records.append((owner, start, gradient.clone()))
                return 0
        sink = Sink()
        ids = torch.tensor([3, 1, 3])
        gradients = torch.arange(9).reshape(3, 3).float()
        before = copy.deepcopy(table.state_dict())
        with packed_derivative_sink(sink):
            table(ids).backward(gradients)
        expected = torch.zeros(3, 5)
        expected.index_add_(1, ids, gradients.t())
        total = torch.zeros_like(expected)
        for owner, start, derivative in sink.records:
            self.assertIs(owner, table.projection)
            total[start:start + derivative.shape[0]].add_(derivative)
        self.assertTrue(torch.equal(expected, total))
        for key, value in table.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_vq_uses_weighted_nearest_ids_and_touched_lookup_not_all_codes(self):
        quantizer = VectorQuantizer(5, 3)
        quantizer.codebook, expected = self.weighted_table()
        latents = expected[[1, 3, 4]].t()[None].contiguous()
        with patch.object(quantizer.codebook.projection, 'effective_weight', side_effect=AssertionError('full expansion')), \
             patch.object(quantizer.codebook, '_read_values', wraps=quantizer.codebook._read_values) as reads:
            values, _loss, ids = quantizer.encode_with_indices(latents)
        self.assertEqual(ids.tolist(), [[1, 3, 4]])
        self.assertTrue(torch.equal(values, latents))
        self.assertEqual(reads.call_count, 1)
        self.assertEqual(reads.call_args.args[0].tolist(), [[1, 3, 4]])

    def test_code_norm_and_backward_tiles_are_bounded_in_both_axes(self):
        table = PackedTernaryTable(137, 131, scale=.02)
        activity = torch.randn(2, 131, requires_grad=True)
        with patch.object(table, '_column_tile', wraps=table._column_tile) as tile, \
             patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')):
            table.scores(activity).sum().backward()
        self.assertGreater(tile.call_count, 1)
        for call in tile.call_args_list:
            _packed, indices, start, end = call.args
            self.assertLessEqual(indices.numel(), table._CODE_BLOCK)
            self.assertLessEqual(end - start, 64)

    def test_checkpoint_retains_orientation_weighted_state_and_unit_native_migration(self):
        table, expected = self.weighted_table()
        with tempfile.TemporaryDirectory(prefix='omni-weighted-codebook-') as root:
            path = Path(root) / 'codebook.safetensors'
            save_file({key: value.detach().contiguous() for key, value in table.state_dict().items()}, str(path))
            saved = load_file(str(path))
            clone = PackedTernaryTable(5, 3, scale=.25)
            clone.load_state_dict(saved)
            self.assertTrue(torch.equal(clone(torch.arange(5)), expected))
            self.assertEqual(tuple(saved['projection._packed_forward_weight'].shape), (3, 2))
        original = PackedTernaryTable(5, 3, scale=.25)
        signs = torch.tensor([[1, 0, -1, 1, 0], [0, 1, -1, 0, 1], [-1, 1, 0, 1, -1]], dtype=torch.int8)
        original.projection.set_ternary_weight_(signs)
        state = {key: value for key, value in original.state_dict().items() if key != 'projection._connection_state'}
        migrated = PackedTernaryTable(5, 3, scale=.25)
        migrated.load_state_dict(state)
        self.assertTrue(torch.equal(migrated(torch.arange(5)), signs.t().float() * .25))
        self.assertTrue(bool(migrated.projection._connection_state.eq(0).all()))

    def test_cold_paged_table_keeps_weighted_values_and_updates(self):
        with tempfile.TemporaryDirectory(prefix='omni-paged-codebook-') as root:
            pager = NativeCorePager(Path(root) / 'pages', cpu_hot_bytes=0, chunk_bytes=64)
            try:
                with pager.construction():
                    table, expected = self.weighted_table()
                self.assertGreater(pager.status()['cpuMappedLogicalBytes'], 0)
                with patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')):
                    self.assertTrue(torch.equal(table(torch.tensor([1, 4])), expected[[1, 4]]))
                    self.assertEqual(table.nearest_indices(expected[[1, 4]]).tolist(), [1, 4])
                    table.projection.online_learning_rate = 100.
                    table(torch.tensor([2])).sum().neg().backward()
                    changed = table(torch.tensor([2])).detach().clone()
                    self.assertFalse(torch.equal(changed, expected[[2]]))
                    pager.cool_to_budget(0)
                    self.assertTrue(torch.equal(table(torch.tensor([2])), changed))
            finally:
                pager.close()

    def test_audio_forward_backward_and_generation_use_the_packed_tables(self):
        torch.manual_seed(41)
        codec = TinyAudioCodec(4, 16, channels=2)
        idea = torch.randn(1, 4)
        waveform = torch.randn(1, 16).clamp(-1, 1)
        codebook = codec.quantizer_a.codebook.projection
        codebook.online_learning_rate = 1e6
        original = {key: value.clone() for key, value in codebook.state_dict().items()}
        with ExitStack() as stack:
            for table in (codec.quantizer_a.codebook, codec.quantizer_b.codebook, codec.token_generator.positions):
                stack.enter_context(patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')))
            result = codec(waveform, idea)
            self.assertTrue(bool(torch.isfinite(result['loss'])))
            result['loss'].backward()
            self.assertTrue(any(key not in original or not torch.equal(value, original[key])
                                for key, value in codebook.state_dict().items()))
            codec.eval()
            revisions = []
            with torch.no_grad():
                output = codec.generate(idea, torch.Generator().manual_seed(13),
                    preview_callback=lambda progress, values: revisions.append((progress, values.shape[-1])), maximum_previews=2)
            self.assertEqual(tuple(output.shape), (1, 16))
            self.assertTrue(bool(torch.isfinite(output).all()))
            self.assertEqual(len(revisions), 2)
            self.assertEqual(revisions[-1], (1., 16))

    @unittest.skipUnless(torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader'), 'native Metal unavailable')
    def test_actual_metal_lookup_scores_and_activity_backward_match_cpu(self):
        table, expected = self.weighted_table()
        table.to('mps')
        inputs = torch.tensor([[[.17, .21, -.73], [1.2, -.23, .1]]], device='mps', requires_grad=True)
        with patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')):
            values = table(torch.tensor([1, 3], device='mps'))
            scores = table.scores(inputs)
            self.assertTrue(torch.equal(values.detach().cpu(), expected[[1, 3]]))
            reference = 2 * inputs.detach().cpu() @ expected.t() - expected.square().sum(-1)
            self.assertTrue(torch.allclose(scores.detach().cpu(), reference, atol=2e-6))
            scores.sum().backward()
            self.assertTrue(torch.allclose(inputs.grad.cpu(), 2 * expected.sum(0).expand(1, 2, 3), atol=2e-6))

    def test_tiny_modality_hub_all_regions_backward_generate_and_exact_safe_reload(self):
        torch.manual_seed(53)
        config = OmniConfig.micro(d_model=4, idea_dim=4, n_heads=2, d_ff=8, modality_channels=2,
            image_size=8, audio_samples=16, video_frames=2)
        hub = ModalityHub(config)
        idea = torch.randn(1, 4)
        fixtures = {
            'image': torch.randn(1, 3, 8, 8).clamp(-1, 1),
            'audio': torch.randn(1, 1, 16).clamp(-1, 1),
            'video': torch.randn(1, 3, 2, 8, 8).clamp(-1, 1),
        }
        # Forbid the removed reader in every actual production table, not
        # just a manually constructed helper. Exercise non-unit state too.
        tables = [module for module in hub.modules() if isinstance(module, PackedTernaryTable)]
        for table in tables:
            table.projection.set_connection_strengths_(torch.full((table.dimensions, table.rows), 2, dtype=torch.int16))
            table.projection.add_parallel_connection_(0, 0, sign=1, strength=2)
            table.projection.online_learning_rate = 100.
        with ExitStack() as stack:
            for table in tables:
                stack.enter_context(patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')))
            for name, target in fixtures.items():
                result = getattr(hub, name)(target, idea)
                self.assertTrue(bool(torch.isfinite(result['loss'])), name)
                self.assertEqual(tuple(result['reconstruction'].shape), tuple(target.shape), name)
                result['loss'].backward()
            expected = {name: hub.generate(name, idea, seed=91) for name in fixtures}
        with tempfile.TemporaryDirectory(prefix='omni-packed-media-hub-') as root:
            path = Path(root) / 'media.safetensors'
            save_file({key: value.detach().cpu().contiguous() for key, value in hub.state_dict().items()}, str(path))
            restored = ModalityHub(config)
            restored.load_state_dict(load_file(str(path)))
            with ExitStack() as stack:
                for table in restored.modules():
                    if isinstance(table, PackedTernaryTable):
                        stack.enter_context(patch.object(table.projection, 'effective_weight', side_effect=AssertionError('full expansion')))
                for name in fixtures:
                    self.assertTrue(torch.equal(restored.generate(name, idea, seed=91), expected[name]), name)
        self.assertEqual(tuple(hub.parameters()), ())


if __name__ == '__main__':
    unittest.main()
