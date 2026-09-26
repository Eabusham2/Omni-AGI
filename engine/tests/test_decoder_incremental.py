"""Decoder-only cache equivalence; no brain creation, persistence, or training."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.config import OmniConfig
from omni_core.model import OmniDecoder, RotaryEmbedding


class IncrementalDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(108)
        torch.set_num_threads(1)

    def make_decoder(self, *, experts=0, device="cpu", **overrides):
        config = OmniConfig.micro(
            max_seq_len=overrides.pop("max_seq_len", 64),
            working_memory_slots=32,
            dropout=0.0,
            **overrides,
        )
        model = OmniDecoder(config).to(device)
        for _ in range(experts):
            model.grow_expert(torch.randn(config.d_model, device=device))
        return model.eval()

    def test_rotary_offset_matches_the_corresponding_full_positions(self):
        rotary = RotaryEmbedding(8, 128)
        query, key = torch.randn(2, 2, 3, 19, 8)
        expected_query, expected_key = rotary(query, key)
        actual_query, actual_key = rotary(query[..., 17:, :], key[..., 17:, :], 17)
        torch.testing.assert_close(actual_query, expected_query[..., 17:, :], rtol=0, atol=0)
        torch.testing.assert_close(actual_key, expected_key[..., 17:, :], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            rotary(query, key, -1)

    def test_cached_logits_match_complete_prefix_with_workspace_memory_and_experts(self):
        for experts in (0, 2):
            for prompt_length in (1, 7, 19):
                with self.subTest(experts=experts, prompt_length=prompt_length):
                    model = self.make_decoder(experts=experts, n_layers=2)
                    model.global_workspace.query_chunk_slots = 3
                    for block in model.blocks:
                        block.attention.query_chunk_tokens = 3
                    ids = torch.randint(3, 260, (2, prompt_length + 6))
                    memory = torch.randn(2, model.config.idea_dim)
                    cache = None
                    with torch.no_grad():
                        for length in range(prompt_length, ids.shape[1] + 1):
                            expected = model(ids[:, :length], memory_bias=memory)["logits"][:, -1]
                            actual, cache = model._generation_step(ids[:, :length], memory, cache)
                            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                            self.assertEqual(cache.length, length)
                            for layer in cache.attention:
                                self.assertEqual(layer.key.shape[-2], length)
                                self.assertEqual(layer.value.shape[-2], length)

    def test_cache_evaluates_only_one_new_token_after_prefill(self):
        model = self.make_decoder()
        ids = torch.randint(3, 260, (1, 11))
        memory = torch.randn(model.config.idea_dim)
        with mock.patch.object(model.embedding, "forward", wraps=model.embedding.forward) as embed:
            with mock.patch.object(model.memory_projection, "forward", wraps=model.memory_projection.forward) as cue:
                with mock.patch.object(model.blocks[0].attention.qkv, "forward", wraps=model.blocks[0].attention.qkv.forward) as qkv:
                    with torch.no_grad():
                        cache = None
                        for length in (8, 9, 10, 11):
                            _, cache = model._generation_step(ids[:, :length], memory, cache)
        self.assertEqual([call.args[0].shape[1] for call in embed.call_args_list], [8, 1, 1, 1])
        self.assertEqual([call.args[0].shape[1] for call in qkv.call_args_list], [8, 1, 1, 1])
        self.assertEqual(cue.call_count, 1)

    def test_cache_storage_tracks_live_tokens_not_selected_context_capacity(self):
        model = self.make_decoder(max_seq_len=1_000_000)
        ids = torch.randint(3, 260, (1, 17))
        with torch.no_grad():
            _, cache = model._generation_step(ids[:, :16], None, None)
            _, cache = model._generation_step(ids, None, cache)
        self.assertEqual(cache.length, 17)
        self.assertEqual(model.blocks[0].attention.rotary.cached_seq_len, 32)
        for layer in cache.attention:
            for tensor in (layer.key, layer.value):
                self.assertEqual(tensor.shape[-2], 17)
                self.assertEqual(
                    tensor.untyped_storage().nbytes(),
                    tensor.numel() * tensor.element_size(),
                )
        for chunk in cache.workspace:
            for tensor in (chunk.mass, chunk.weighted_values):
                self.assertEqual(
                    tensor.untyped_storage().nbytes(),
                    tensor.numel() * tensor.element_size(),
                )

    def test_shifted_context_rebuilds_workspace_and_attention(self):
        model = self.make_decoder(experts=2, max_seq_len=8)
        ids = torch.randint(3, 260, (1, 15))
        cache = None
        with mock.patch.object(model.global_workspace, "causal_summaries", wraps=model.global_workspace.causal_summaries) as prefill:
            with torch.no_grad():
                for length in range(6, 16):
                    window = ids[:, :length][:, -8:]
                    expected = model(window)["logits"][:, -1]
                    # Reference forward also calls causal_summaries; count
                    # only the cached path's prefill calls below.
                    before = prefill.call_count
                    actual, cache = model._generation_step(window, None, cache)
                    self.assertEqual(prefill.call_count - before, int(length == 6 or length > 8))
                    self.assertEqual(cache.length, min(length, 8))
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_seeded_sampling_noise_printable_filter_and_callbacks_match_full_forward(self):
        for experts in (0, 2):
            for printable_only in (False, True):
                for top_k in (1, 7, 0):
                    with self.subTest(experts=experts, printable=printable_only, top_k=top_k):
                        model = self.make_decoder(experts=experts, max_seq_len=12)
                        ids = torch.tensor([[1, 259, 42, 43, 44, 45, 46, 47, 48, 260]])
                        memory = torch.randn(model.config.idea_dim)
                        kwargs = dict(
                            memory_bias=memory,
                            max_new_tokens=6,
                            temperature=0.8,
                            top_k=top_k,
                            noise=0.07,
                            seed=314,
                            printable_only=printable_only,
                        )
                        cached_callbacks, reference_callbacks = [], []

                        def observe(destination):
                            return lambda token, step, entropy: destination.append((token.clone(), step, entropy))

                        actual, entropy = model.generate(ids, token_callback=observe(cached_callbacks), **kwargs)
                        expected, expected_entropy = model.generate(
                            ids,
                            token_callback=observe(reference_callbacks),
                            use_cache=False,
                            **kwargs,
                        )
                        self.assertTrue(torch.equal(actual, expected))
                        torch.testing.assert_close(torch.tensor(entropy), torch.tensor(expected_entropy), rtol=1e-5, atol=1e-6)
                        self.assertEqual(len(cached_callbacks), len(reference_callbacks))
                        for left, right in zip(cached_callbacks, reference_callbacks):
                            self.assertTrue(torch.equal(left[0], right[0]))
                            self.assertEqual(left[1], right[1])
                            self.assertAlmostEqual(left[2], right[2], places=5)

    def test_cached_candidate_and_replay_are_exactly_reproducible(self):
        for context in (8, 64):
            with self.subTest(context=context):
                model = self.make_decoder(experts=2, max_seq_len=context)
                ids = torch.tensor([[1, 259, 42, 43, 44, 260]])
                kwargs = dict(
                    memory_bias=torch.randn(model.config.idea_dim),
                    max_new_tokens=16,
                    temperature=0.8,
                    top_k=7,
                    noise=0.13,
                    seed=118,
                )
                candidate, candidate_entropy = model.generate(ids, **kwargs)
                replay, replay_entropy = model.generate(ids, **kwargs)
                self.assertTrue(torch.equal(candidate, replay))
                self.assertEqual(candidate_entropy, replay_entropy)

    def test_full_prefix_reference_option_bypasses_the_incremental_path(self):
        model = self.make_decoder()
        ids = torch.tensor([[1, 259, 42, 260]])
        with mock.patch.object(model, "_generation_step", side_effect=AssertionError("cache bypass required")):
            result, entropy = model.generate(ids, max_new_tokens=2, use_cache=False)
        self.assertEqual(result.shape[1], ids.shape[1] + 2)
        self.assertEqual(len(entropy), 2)
        self.assertEqual(model.last_generation_cache_mode, "full-prefix-reference")

    def test_parameter_cue_and_expert_growth_invalidate_cached_states(self):
        for changed in ("embedding", "linear", "load", "memory", "grow"):
            with self.subTest(changed=changed):
                model = self.make_decoder()
                memory = torch.randn(model.config.idea_dim)
                ids = torch.randint(3, 260, (1, 6))
                with torch.no_grad():
                    _, cache = model._generation_step(ids[:, :5], memory, None)
                    if changed == "embedding":
                        model.embedding.weight.add_(0.01)
                    elif changed == "linear":
                        model.blocks[0].attention.qkv.weight.mul_(-1)
                    elif changed == "load":
                        state = {name: value.clone() for name, value in model.state_dict().items()}
                        state["embedding.weight"].mul_(-1)
                        model.load_state_dict(state)
                    elif changed == "memory":
                        memory.mul_(-1)
                    else:
                        model.grow_expert(torch.randn(model.config.d_model))
                        model.grow_expert(torch.randn(model.config.d_model))
                        model.eval()
                    with mock.patch.object(model.global_workspace, "causal_step", side_effect=AssertionError("stale cache reused")):
                        actual, _ = model._generation_step(ids, memory, cache)
                    expected = model(ids, memory_bias=memory)["logits"][:, -1]
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_inference_mode_cue_without_version_counter_uses_full_forward(self):
        model = self.make_decoder()
        with torch.inference_mode():
            memory = torch.randn(model.config.idea_dim)
        ids = torch.tensor([[1, 259, 42, 260]])
        with torch.no_grad():
            actual, cache = model._generation_step(ids, memory, None)
            expected = model(ids, memory_bias=memory)["logits"][:, -1]
        self.assertIsNone(cache)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        model.generate(ids, memory_bias=memory, max_new_tokens=2)
        self.assertEqual(model.last_generation_cache_mode, "full-prefix-fallback")

    def test_cancel_before_prefill_or_after_callback_preserves_returned_prefix(self):
        model = self.make_decoder()
        ids = torch.tensor([[1, 259, 42, 260]])
        with mock.patch.object(model, "_generation_step", side_effect=AssertionError("already cancelled")):
            actual, entropy = model.generate(ids, cancelled=lambda: True)
        self.assertTrue(torch.equal(actual, ids))
        self.assertEqual(entropy, [])
        self.assertEqual(model.last_generation_cache_mode, "not-started")
        observed = []
        actual, entropy = model.generate(
            ids,
            max_new_tokens=4,
            token_callback=lambda token, step, entropy: observed.append(step),
            cancelled=lambda: bool(observed),
        )
        self.assertEqual(actual.shape[1], ids.shape[1] + 1)
        self.assertEqual(observed, [0])
        self.assertEqual(len(entropy), 1)
        self.assertEqual(model.last_generation_cache_mode, "incremental-v1")

    def test_cancel_during_prefill_does_not_emit_a_sample(self):
        model = self.make_decoder()
        ids = torch.tensor([[1, 259, 42, 260]])
        checks = iter((False, True))
        callbacks = []
        actual, entropy = model.generate(
            ids,
            cancelled=lambda: next(checks),
            token_callback=lambda *args: callbacks.append(args),
        )
        self.assertTrue(torch.equal(actual, ids))
        self.assertEqual(entropy, [])
        self.assertEqual(callbacks, [])

    def test_cache_is_ephemeral_and_not_a_training_or_persisted_model_state(self):
        model = self.make_decoder()
        ids = torch.tensor([[1, 259, 42, 260]])
        original_keys = set(model.state_dict())
        original_parameters = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
        }
        with self.assertRaisesRegex(RuntimeError, "no_grad"):
            model._generation_step(ids, None, None)
        model.generate(ids, max_new_tokens=2)
        self.assertEqual(set(model.state_dict()), original_keys)
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, original_parameters[name]))
        self.assertFalse(any("inference_cache" in key for key in vars(model)))
        with torch.no_grad():
            model.train()
            _, cache = model._generation_step(ids, None, None)
        self.assertIsNone(cache)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_cached_logits_match_full_prefix_and_shifted_window(self):
        model = self.make_decoder(experts=2, device="mps", max_seq_len=8)
        ids = torch.tensor([[1, 259, 42, 43, 44, 45, 46, 260, 47]], device="mps")
        memory = torch.randn(model.config.idea_dim, device="mps")
        cache = None
        with torch.no_grad():
            for length in (5, 6, 7, 8, 9):
                window = ids[:, :length][:, -8:]
                expected = model(window, memory_bias=memory)["logits"][:, -1]
                actual, cache = model._generation_step(window, memory, cache)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        kwargs = dict(
            memory_bias=memory,
            max_new_tokens=4,
            top_k=7,
            noise=0.13,
            seed=315,
        )
        candidate, candidate_entropy = model.generate(ids[:, :5], **kwargs)
        replay, replay_entropy = model.generate(ids[:, :5], **kwargs)
        self.assertTrue(torch.equal(candidate, replay))
        self.assertEqual(candidate_entropy, replay_entropy)


if __name__ == "__main__":
    unittest.main()
