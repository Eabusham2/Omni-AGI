"""Pure tensor/store fixtures: never construct a brain or a full OmniDecoder."""

import gc
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (
    CausalSelfAttention,
    DecoderBlock,
    GlobalWorkspace,
    OmniDecoder,
    RotaryEmbedding,
)
from omni_core.offload import NeuralStateResourcePause
from omni_core.brain import AdaptiveBrain
from omni_core.tokenizer import ByteTokenizer
from omni_core.working_attention_paging import (
    WorkingAttentionCancelled,
    WorkingAttentionPager,
    _dropout_mask,
    attention_tile_bytes,
    exact_causal_attention,
    kv_context_bytes,
)


def dense_attention(query, key, value, offset=0, dropout_mask=None, causal=True):
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if causal:
        mask = torch.arange(key.shape[-2])[None, :] > torch.arange(
            offset, offset + query.shape[-2]
        )[:, None]
        scores = scores.masked_fill(mask, -torch.inf)
    probabilities = scores.softmax(-1)
    if dropout_mask is not None:
        probabilities = probabilities * dropout_mask
    return probabilities @ value


class FixedProjection(nn.Module):
    """Small derived fixture matrices, not a learned native weight master."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, inputs):
        return inputs @ self.weight.transpose(-2, -1)


class FixedTable(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, indices):
        return self.values[indices]

    def packed_forward_weight(self):
        return self.values


class FixedScalar(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.register_buffer("value", torch.tensor(value))

    def forward(self):
        return self.value


def decoder_method_fixture(pager):
    """Attach real context-processing methods to bounded, fixed tensor stubs."""
    generator = torch.Generator().manual_seed(107)
    decoder = OmniDecoder.__new__(OmniDecoder)
    nn.Module.__init__(decoder)
    decoder.config = SimpleNamespace(
        d_model=8, d_ff=16, vocab_size=17, max_seq_len=100_000,
    )
    decoder.embedding = FixedTable(torch.randn(17, 8, generator=generator) * 0.25)
    workspace = GlobalWorkspace.__new__(GlobalWorkspace)
    nn.Module.__init__(workspace)
    workspace.slots = 4
    workspace.dimensions = 8
    workspace.iterations = 2
    workspace.query_chunk_slots = 2
    workspace.latent_table = FixedTable(torch.randn(4, 8, generator=generator) * 0.05)
    workspace.norm = nn.Identity()
    for name in ("query", "key", "value", "update", "broadcast"):
        setattr(workspace, name, FixedProjection(torch.eye(8) * 0.5))
    decoder.global_workspace = workspace
    decoder.workspace_strength = FixedScalar(0.12)
    decoder.memory_strength = FixedScalar(0.15)
    decoder.memory_projection = FixedProjection(torch.eye(8))
    attention = CausalSelfAttention.__new__(CausalSelfAttention)
    nn.Module.__init__(attention)
    attention.n_heads = 2
    attention.head_dim = 4
    attention.query_chunk_tokens = 4
    attention.key_chunk_tokens = 3
    attention.dropout = 0.0
    attention.rotary = RotaryEmbedding(4, 100_000)
    attention.qkv = FixedProjection(torch.randn(24, 8, generator=generator) * 0.05)
    attention.output = FixedProjection(torch.eye(8) * 0.2)
    block = DecoderBlock.__new__(DecoderBlock)
    nn.Module.__init__(block)
    block.attention_norm = nn.Identity()
    block.attention = attention
    block.feed_forward_norm = nn.Identity()
    block.feed_forward = FixedProjection(torch.zeros(8, 8))
    decoder.blocks = nn.ModuleList([block])
    decoder.final_norm = nn.Identity()
    decoder.language_head = FixedProjection(torch.randn(17, 8, generator=generator) * 0.1)
    decoder.experts = nn.ModuleList()
    decoder.expert_prototypes = nn.ModuleList()
    decoder.native_core_pager = None
    decoder.last_generation_cache_mode = "not-started"
    decoder.configure_working_attention(pager)
    return decoder.eval()


class WorkingAttentionPagingFixtures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)
        torch.set_num_threads(1)

    def pager(self, path, resident=64, scratch=1_000_000, tile=131072, page=4):
        return WorkingAttentionPager(
            path, resident_budget_bytes=resident, scratch_budget_bytes=scratch,
            device_tile_budget_bytes=tile, page_tokens=page,
        )

    def test_exact_forward_and_first_order_gradients_match_dense(self):
        query, key, value = [
            torch.randn(2, 3, 9, 4, dtype=torch.float64, requires_grad=True)
            for _ in range(3)
        ]
        gradient = torch.randn_like(query)
        expected = dense_attention(query, key, value)
        expected_gradients = torch.autograd.grad(expected, (query, key, value), gradient)
        actual = exact_causal_attention(query, key, value, query_chunk_tokens=3, key_chunk_tokens=2)
        actual_gradients = torch.autograd.grad(actual, (query, key, value), gradient)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        for left, right in zip(actual_gradients, expected_gradients):
            self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))

    def test_paged_key_tiles_match_dense_with_absolute_causal_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=48)
            query = torch.randn(1, 2, 3, 4)
            key, value = torch.randn(1, 2, 17, 4), torch.randn(1, 2, 17, 4)
            keys, values = pager.sequence(key), pager.sequence(value)
            with pager.activate(), torch.no_grad():
                actual = exact_causal_attention(
                    query, keys, values, position_offset=14,
                    query_chunk_tokens=2, key_chunk_tokens=3,
                )
            self.assertTrue(torch.allclose(actual, dense_attention(query, key, value, 14), atol=1e-6))
            status = pager.status()
            self.assertGreater(status["bytesRead"], 0)
            self.assertGreater(status["spillBytes"], 0)
            self.assertLessEqual(status["residentBytes"], 48)
            self.assertLessEqual(status["largestScoreElements"], 1 * 2 * 2 * 3)
            self.assertLessEqual(status["largestKeyTileTokens"], 3)
            keys.close()
            values.close()
            self.assertEqual(pager.status()["livePages"], 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_saved_training_qkv_pages_preserve_exact_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=32, page=3)
            tensors = [torch.randn(1, 2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
            gradient = torch.randn_like(tensors[0])
            expected = dense_attention(*tensors)
            wanted = torch.autograd.grad(expected, tensors, gradient)
            with pager.activate():
                actual = exact_causal_attention(*tensors, query_chunk_tokens=2, key_chunk_tokens=3)
            got = torch.autograd.grad(actual, tensors, gradient)
            for left, right in zip(got, wanted):
                self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))
            self.assertGreater(pager.status()["bytesWritten"], 0)
            del actual, got
            gc.collect()
            self.assertEqual(pager.status()["livePages"], 0)

    def test_dropout_recomputation_uses_mathematically_correct_mask_gradient(self):
        query, key, value = [torch.randn(1, 2, 7, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        gradient = torch.randn_like(query)
        torch.manual_seed(51)
        seed = int(torch.randint(0, 2 ** 31 - 1, ()).item())
        mask = torch.zeros(1, 2, 7, 7, dtype=torch.float64)
        for q_start in range(0, 7, 2):
            q_end = min(7, q_start + 2)
            for k_start in range(0, q_end, 3):
                k_end = min(q_end, k_start + 3)
                mask[..., q_start:q_end, k_start:k_end] = _dropout_mask(
                    (1, 2, q_end - q_start, k_end - k_start), torch.device("cpu"),
                    seed, q_start, k_start, 0.3, torch.float64,
                )
        expected = dense_attention(query, key, value, dropout_mask=mask)
        wanted = torch.autograd.grad(expected, (query, key, value), gradient)
        torch.manual_seed(51)
        actual = exact_causal_attention(
            query, key, value, query_chunk_tokens=2, key_chunk_tokens=3,
            dropout=0.3, training=True,
        )
        got = torch.autograd.grad(actual, (query, key, value), gradient)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-12))
        for left, right in zip(got, wanted):
            self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))

    def test_noncausal_whole_input_tiles_match_dense(self):
        query = torch.randn(1, 1, 9, 4, dtype=torch.float64, requires_grad=True)
        key, value = [torch.randn(1, 1, 5, 4, dtype=torch.float64, requires_grad=True) for _ in range(2)]
        wanted = dense_attention(query, key, value, causal=False)
        actual = exact_causal_attention(query, key, value, causal=False, query_chunk_tokens=2, key_chunk_tokens=2)
        self.assertTrue(torch.allclose(actual, wanted, atol=1e-12, rtol=1e-12))
        expected_gradients = torch.autograd.grad(wanted.sum(), (query, key, value))
        actual_gradients = torch.autograd.grad(actual.sum(), (query, key, value))
        for left, right in zip(actual_gradients, expected_gradients):
            self.assertTrue(torch.allclose(left, right, atol=1e-12, rtol=1e-12))

    def test_lossless_noncontiguous_activation_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=32, tile=4096)
            source = torch.randn(4, 9, 7, requires_grad=True)
            with pager.activate(), pager.saved_activation_hooks():
                output = source.transpose(1, 2).square().sum()
            output.backward()
            self.assertTrue(torch.equal(source.grad, 2 * source.detach()))
            self.assertGreater(pager.status()["savedActivationBytes"], 0)
            del output
            gc.collect()
            self.assertEqual(pager.status()["livePages"], 0)

    def test_spill_exhaustion_rejects_without_discarding_requested_tokens(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=0, scratch=8)
            with self.assertRaisesRegex(NeuralStateResourcePause, "not truncated"):
                pager.sequence(torch.randn(1, 1, 4, 4))
            self.assertEqual(pager.status()["livePages"], 0)

    def test_corrupt_private_page_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=0)
            sequence = pager.sequence(torch.randn(1, 1, 4, 4))
            path = next(Path(directory).iterdir())
            with path.open("r+b") as handle:
                handle.write(b"bad!")
            with self.assertRaisesRegex(ValueError, "checksum"):
                sequence.read(0, 1, torch.device("cpu"))
            sequence.close()

    def test_cancellation_is_checked_inside_key_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=0)
            keys, values = pager.sequence(torch.randn(1, 1, 19, 4)), pager.sequence(torch.randn(1, 1, 19, 4))
            count = 0

            def cancelled():
                nonlocal count
                count += 1
                return count > 5

            with self.assertRaises(WorkingAttentionCancelled), pager.activate(cancelled), torch.no_grad():
                exact_causal_attention(torch.randn(1, 1, 1, 4), keys, values, position_offset=18, key_chunk_tokens=2)
            self.assertLess(count, 15)
            keys.close()
            values.close()

    def test_large_rotary_offset_does_not_allocate_prefix_position_tables(self):
        rotary = RotaryEmbedding(4, 1_000_000)
        query = torch.randn(1, 2, 3, 4)
        rotated, _ = rotary(query, query, position_offset=999_000)
        self.assertEqual(rotated.shape, query.shape)
        self.assertLessEqual(rotary.cached_seq_len, 256)
        self.assertTrue(torch.isfinite(rotated).all())

    def test_rotary_high_positions_keep_adjacent_tokens_distinct_without_whole_prefix(self):
        rotary = RotaryEmbedding(4, 100_000_000)
        query = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]])
        offset = 2 ** 24 + 1
        rotated, _ = rotary(query, query, position_offset=offset)
        expected = torch.arange(offset, offset + 2, dtype=torch.float64).cos().float()
        self.assertTrue(torch.equal(rotated[0, 0, :, 0], expected))
        self.assertNotEqual(float(rotated[0, 0, 0, 0]), float(rotated[0, 0, 1, 0]))
        self.assertEqual(rotary.cached_seq_len, 0)

    def test_compute_pressure_shrinks_tiles_not_context(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=64, tile=4096)
            query = torch.randn(1, 2, 3, 4)
            key, value = torch.randn(1, 2, 33, 4), torch.randn(1, 2, 33, 4)
            keys, values = pager.sequence(key), pager.sequence(value)
            with pager.activate(), torch.no_grad():
                result = exact_causal_attention(
                    query, keys, values, position_offset=30,
                    query_chunk_tokens=128, key_chunk_tokens=128,
                )
            self.assertTrue(torch.allclose(result, dense_attention(query, key, value, 30), atol=1e-6))
            self.assertEqual(keys.length, 33)
            self.assertLessEqual(pager.status()["largestKeyTileTokens"], 4)
            self.assertLessEqual(pager.status()["peakComputeTileBytes"], 4096)
            keys.close()
            values.close()

    def test_live_resource_admission_pauses_before_allocation(self):
        policy = SimpleNamespace(status=lambda **kwargs: {
            "availableMemoryBytes": 1024,
            "ramReserveBytes": 1024,
            "processMemoryBytes": 0,
            "systemRamBudgetBytes": 32768,
            "diskFreeBytes": 100000,
            "diskReserveBytes": 1,
            "acceleratorFreeMemoryBytes": 0,
        })
        with tempfile.TemporaryDirectory() as directory:
            pager = WorkingAttentionPager(
                Path(directory), resident_budget_bytes=64,
                scratch_budget_bytes=1024, device_tile_budget_bytes=4096,
                resource_policy=policy,
            )
            with self.assertRaisesRegex(NeuralStateResourcePause, "RAM reserve"):
                pager.admit_compute(32)
            self.assertEqual(pager.status()["livePages"], 0)
            policy.status = lambda **kwargs: {
                "availableMemoryBytes": 100000, "ramReserveBytes": 1,
                "processMemoryBytes": 0, "systemRamBudgetBytes": 100000,
                "acceleratorFreeMemoryBytes": 0,
            }
            pager._policy_at = 0.0
            # A CPU fixture must not be paused by an unrelated GPU reading.
            pager.admit_compute(32, device=torch.device("cpu"))
            with self.assertRaisesRegex(NeuralStateResourcePause, "accelerator"):
                pager.admit_compute(32, device=torch.device("cuda"))

    def test_decoder_methods_stream_prefill_and_append_with_stub_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            ids = torch.arange(19).remainder(17).unsqueeze(0)

            def reference(prefix):
                hidden = decoder.embedding(prefix)
                hidden = hidden + decoder.workspace_strength() * decoder.global_workspace.causal_summaries(hidden)
                for block in decoder.blocks:
                    hidden = block(hidden)
                return decoder.language_head(decoder.final_norm(hidden))[:, -1]

            with torch.no_grad():
                expected = reference(ids)
                actual, cache = decoder._generation_step(ids, None, None)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
                self.assertEqual(cache.length, 19)
                self.assertEqual(cache.attention[0].key.shape[-2], 19)
                self.assertEqual(pager.status()["processedPrefillTokens"], 19)
                self.assertLessEqual(pager.status()["largestPrefillBlockTokens"], 4)
                self.assertGreater(pager.status()["spillBytes"], 0)
                extended = torch.cat((ids, torch.tensor([[3]])), dim=1)
                expected = reference(extended)
                actual, cache = decoder._generation_step(extended, None, cache)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
                self.assertEqual(pager.status()["processedPrefillTokens"], 20)
                cache.close()
            self.assertEqual(pager.status()["livePages"], 0)

    def test_whole_encoding_streams_all_tokens_with_stub_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            ids = torch.arange(29).remainder(17).unsqueeze(0)
            with torch.no_grad():
                expected = decoder.global_workspace.summarize(decoder.embedding(ids))
                actual = decoder.encode_whole(ids)
            self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
            self.assertEqual(pager.status()["processedPrefillTokens"], 29)
            self.assertEqual(pager.status()["livePages"], 0)

    def test_generation_cleanup_and_full_output_admission_with_stub_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            ids = torch.arange(13).remainder(17).unsqueeze(0)
            generated, entropies = decoder.generate(ids, max_new_tokens=2, printable_only=False, top_k=0)
            self.assertGreater(len(entropies), 0)
            self.assertEqual(generated.shape[1], ids.shape[1] + len(entropies))
            self.assertEqual(decoder.last_generation_cache_mode, "paged-incremental-v2")
            self.assertEqual(pager.status()["livePages"], 0)
            maximum = decoder.maximum_forward_tokens()
            oversized = torch.zeros((1, maximum + 1), dtype=torch.long)
            with self.assertRaisesRegex(NeuralStateResourcePause, "not truncated"):
                decoder(oversized)
            with self.assertRaisesRegex(ValueError, "not truncated"):
                decoder.generate(torch.zeros((1, 100_001), dtype=torch.long))

    def test_costs_scale_linearly_with_capacity_and_quadratically_only_with_tiles(self):
        self.assertEqual(kv_context_bytes(100_000, 1, 4, 128), 409_600_000)
        self.assertEqual(kv_context_bytes(200_000, 1, 4, 128), 819_200_000)
        small = attention_tile_bytes(1, 4, 32, 16, 16)
        large = attention_tile_bytes(1, 4, 32, 32, 32)
        self.assertGreater(large, small)
        self.assertLess(large, small * 4)

    def test_live_activity_receives_native_prefix_and_can_stop_before_next_token(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            ids = torch.tensor([[1, 3, 4]])
            activity, emitted = [], []

            def inspect(hidden, step, entropy):
                activity.append((hidden.clone(), step, entropy))
                return {"stop": step == 1}

            generated, entropies = decoder.generate(
                ids, max_new_tokens=8, printable_only=False,
                activity_callback=inspect,
                token_callback=lambda token, *_: emitted.extend(token.flatten().tolist()),
            )
            self.assertEqual(len(activity), 2)
            self.assertEqual(activity[0][0].shape, (1, 8))
            self.assertEqual(generated.shape[1], ids.shape[1] + 1)
            self.assertEqual(emitted, generated[0, ids.shape[1]:].tolist())
            self.assertEqual(len(entropies), 1)
            self.assertEqual(decoder.last_generation_stop_reason, "native-action-stop")
            self.assertEqual(pager.status()["livePages"], 0)

    def test_steer_yields_exact_emitted_prefix_and_closes_only_its_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            ids = torch.tensor([[1, 3, 4]])
            emitted = []
            generated, entropies = decoder.generate(
                ids, max_new_tokens=8, printable_only=False,
                steer_check=lambda: len(emitted) >= 2,
                token_callback=lambda token, *_: emitted.extend(token.flatten().tolist()),
            )
            self.assertEqual(emitted, generated[0, ids.shape[1]:].tolist())
            self.assertEqual(len(entropies), len(emitted))
            self.assertEqual(len(emitted), 2)
            self.assertEqual(decoder.last_generation_stop_reason, "steered")
            self.assertEqual(pager.status()["livePages"], 0)

    def test_steer_inside_prefill_has_zero_visible_tokens_and_cleans_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            ids = torch.arange(19).remainder(17).unsqueeze(0)
            calls = 0

            def steer():
                nonlocal calls
                calls += 1
                return calls >= 12

            generated, entropies = decoder.generate(ids, max_new_tokens=8, printable_only=False, steer_check=steer)
            self.assertTrue(torch.equal(generated, ids))
            self.assertEqual(entropies, [])
            self.assertEqual(decoder.last_generation_stop_reason, "steered")
            self.assertEqual(pager.status()["livePages"], 0)

    def test_single_token_appends_coalesce_exact_values_into_bounded_tail_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=0, page=4)
            values = torch.randn(1, 2, 21, 4)
            sequence = pager.sequence(values[..., :1, :])
            for position in range(1, 21):
                sequence.append(values[..., position:position + 1, :])
                self.assertEqual(len(sequence.pages), math.ceil((position + 1) / 4))
                self.assertTrue(torch.equal(sequence.read(0, position + 1, torch.device("cpu")), values[..., :position + 1, :]))
            self.assertEqual(sequence.length, 21)
            self.assertLessEqual(pager.status()["largestPageBytes"], 4 * 1 * 2 * 4 * 4)
            self.assertEqual(pager.status()["pageMetadataEstimatedBytes"], 6 * 2048)
            sequence.close()
            self.assertEqual(pager.status()["livePages"], 0)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_coalescing_uses_physical_allocation_delta_at_a_full_spill_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            probe = self.pager(Path(directory), resident=0, page=4)
            allocation = probe.status()["filesystemAllocationUnitBytes"]
            pager = self.pager(Path(directory), resident=0, scratch=allocation, page=4)
            source = torch.randn(1, 1, 4, 4)
            sequence = pager.sequence(source[..., :1, :])
            for position in range(1, 4):
                sequence.append(source[..., position:position + 1, :])
                self.assertEqual(pager.status()["livePages"], 1)
                self.assertLessEqual(pager.status()["spillBytes"], allocation)
            self.assertTrue(torch.equal(sequence.read(0, 4, torch.device("cpu")), source))
            sequence.close()
            self.assertEqual(pager.status()["spillBytes"], 0)

    def test_candidate_suffix_scoring_streams_context_larger_than_full_output_window(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(Path(directory), resident=96, tile=32768)
            decoder = decoder_method_fixture(pager)
            brain = AdaptiveBrain.__new__(AdaptiveBrain)
            brain.decoder = decoder
            brain.config = decoder.config
            brain.tokenizer = ByteTokenizer()
            prompt = torch.arange(29).remainder(17).unsqueeze(0)
            generated = torch.cat((prompt, torch.tensor([[3, 4, 5]])), dim=1)
            self.assertGreater(prompt.shape[1], decoder.maximum_forward_tokens())

            def reference(prefix):
                hidden = decoder.embedding(prefix)
                hidden = hidden + decoder.workspace_strength() * decoder.global_workspace.causal_summaries(hidden)
                for block in decoder.blocks:
                    hidden = block(hidden)
                return decoder.language_head(decoder.final_norm(hidden))[:, -1]

            with torch.no_grad():
                expected = sum(float(torch.nn.functional.cross_entropy(reference(generated[:, :position]), generated[:, position]).item()) for position in range(29, 32)) / 3
                actual = brain._candidate_nll(prompt, generated, None)
            self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(pager.status()["livePages"], 0)


if __name__ == "__main__":
    unittest.main()
