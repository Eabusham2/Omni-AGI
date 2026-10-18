"""Short own-native drafting mechanism fixtures, not trained conversation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from omni_core.config import OmniConfig
from omni_core.model import OmniDecoder
from omni_core.native_draft_evaluation import (DraftEvaluationInterrupted, NativeDraftRegion,
    acceptance_and_residual, evaluate_round, sampling_distribution, verify_proposal)
from omni_core.persistence import tensor_checksum
from omni_core.working_attention_paging import WorkingAttentionPager


def tiny_target(seed=4129, experts=False):
    torch.manual_seed(seed)
    target = OmniDecoder(OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1,
        d_ff=16, max_seq_len=64, working_memory_slots=4, vsa_dim=16, router_neurons=4,
        liquid_steps=1))
    if experts:
        prototype = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0.])
        target.grow_expert(prototype)
        target.grow_expert(-prototype)
    return target.eval()


class NativeDraftSamplingTests(unittest.TestCase):
    def test_acceptance_plus_positive_residual_is_exact_target_law_with_support_mismatch(self):
        for p, q in (([.1, .2, .7], [.7, .2, .1]), ([0., 1., 0.], [.5, 0., .5]),
                     ([.25, .75, 0.], [.25, .75, 0.]), ([.2, .3, .5], [0., 1., 0.])):
            with self.subTest(p=p, q=q):
                target, proposal, accepted, residual = acceptance_and_residual(
                    torch.tensor(p, dtype=torch.float64), torch.tensor(q, dtype=torch.float64))
                actual = accepted if residual is None else accepted + (1 - accepted.sum()) * residual
                self.assertTrue(torch.allclose(actual, target, atol=1e-15, rtol=1e-15))
                self.assertTrue(bool((actual[proposal == 0] >= 0).all()))

    def test_actual_sampling_rejects_disjoint_proposal_then_uses_target_residual(self):
        generator = torch.Generator().manual_seed(17)
        token, accepted = verify_proposal(0, torch.tensor([0., 1., 0.]),
            torch.tensor([1., 0., 0.]), generator)
        self.assertFalse(accepted)
        self.assertEqual(token, 1)
        for token in (0, 2):
            selected, accepted = verify_proposal(token, torch.tensor([.5, 0., .5]),
                torch.tensor([.5, 0., .5]), generator)
            self.assertTrue(accepted)
            self.assertEqual(selected, token)

    def test_bad_proposal_has_no_rng_effect_and_sampling_filters_have_actual_support(self):
        generator = torch.Generator().manual_seed(18)
        state = generator.get_state().clone()
        with self.assertRaisesRegex(ValueError, 'no probability'):
            verify_proposal(1, torch.tensor([.2, .8]), torch.tensor([1., 0.]), generator)
        self.assertTrue(torch.equal(state, generator.get_state()))
        logits = torch.tensor([0., 2., 1.])
        self.assertEqual(sampling_distribution(logits, .7, 1).tolist(), [0., 1., 0.])
        self.assertEqual(sampling_distribution(logits, 0).tolist(), [0., 1., 0.])

    def test_rejected_sample_uses_positive_residual_not_an_unmodified_target_fallback(self):
        seen = []
        def sample(law, *_args, **_kwargs):
            seen.append(law.clone())
            return torch.tensor([2])
        with patch('torch.rand', return_value=torch.tensor(.99)), patch('torch.multinomial', side_effect=sample):
            token, accepted = verify_proposal(0, torch.tensor([.1, .2, .7]),
                torch.tensor([.7, .2, .1]), torch.Generator().manual_seed(31))
        self.assertFalse(accepted)
        self.assertEqual(token, 2)
        self.assertEqual(seen[0].tolist(), [0., 0., 1.])


class NativeDraftRegionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.target = tiny_target()
        self.region = NativeDraftRegion(8, block_size=3, selector_rank=4, candidate_count=8).eval()
        self.prefix = torch.tensor([[259, 65, 260]])

    def hidden(self):
        with torch.no_grad():
            return self.target(self.prefix)['hidden'][:, -1]

    def test_own_auxiliary_is_packed_and_actual_proposal_law_matches_its_selected_path(self):
        before = tensor_checksum(self.target.state_dict().values())
        tokens, probabilities = self.region.propose(self.target, self.hidden(), self.prefix[:, -1], 3,
            torch.Generator().manual_seed(17))
        self.assertEqual(tokens.shape, (3,))
        self.assertEqual(probabilities.shape, (3, self.target.config.vocab_size))
        self.assertTrue(torch.allclose(probabilities.sum(-1), torch.ones(3)))
        self.assertTrue(bool((probabilities.gather(-1, tokens[:, None]) > 0).all()))
        self.assertTrue(bool((probabilities.ne(0).sum(-1) <= 8).all()))
        self.assertEqual(tuple(self.region.parameters()), ())
        self.assertEqual(tensor_checksum(self.target.state_dict().values()), before)
        self.assertFalse(any(module is self.target for module in self.region.modules()))

    def test_mixed_shared_head_training_cannot_be_written_by_auxiliary_imitation(self):
        hidden = self.hidden()
        before = tensor_checksum(self.target.state_dict().values())
        self.target.language_head.train()
        self.region.train()
        with self.assertRaisesRegex(ValueError, 'read-only'):
            self.region.teacher_loss(self.target, hidden, self.prefix[:, -1], torch.tensor([66, 67, 68]))
        self.assertEqual(before, tensor_checksum(self.target.state_dict().values()))

    def test_native_auxiliary_training_mutates_its_sign_strength_bytes_not_the_shared_brain(self):
        hidden = self.hidden()
        target_before = tensor_checksum(self.target.state_dict().values())
        before = tensor_checksum(self.region.state_dict().values())
        teacher = torch.tensor([66, 67, 68])
        self.region.train()
        for module in self.region.modules():
            if hasattr(module, 'online_learning_rate'):
                module.online_learning_rate = 1.
        for _ in range(3):
            loss = self.region.teacher_loss(self.target, hidden, self.prefix[:, -1], teacher)
            self.assertTrue(bool(torch.isfinite(loss)))
            loss.backward()
        self.assertNotEqual(tensor_checksum(self.region.state_dict().values()), before)
        self.assertEqual(tensor_checksum(self.target.state_dict().values()), target_before)
        self.assertEqual(tuple(self.region.parameters()), ())

    def test_native_auxiliary_safe_reload_reproduces_same_seeded_proposals(self):
        hidden = self.hidden()
        expected = self.region.propose(self.target, hidden, self.prefix[:, -1], 3,
            torch.Generator().manual_seed(22))
        with tempfile.TemporaryDirectory(prefix='native-draft-reload-') as directory:
            path = Path(directory) / 'draft.safetensors'
            save_file({key: value.contiguous() for key, value in self.region.state_dict().items()}, str(path))
            restored = NativeDraftRegion(8, 3, 4, 8).eval()
            restored.load_state_dict(load_file(str(path)))
            actual = restored.propose(self.target, hidden, self.prefix[:, -1], 3,
                torch.Generator().manual_seed(22))
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_private_round_preserves_global_rng_neural_bytes_and_visible_metadata(self):
        before = tensor_checksum(self.target.state_dict().values())
        auxiliary = tensor_checksum(self.region.state_dict().values())
        self.target.last_generation_stop_reason = 'visible outside turn'
        self.target.last_generation_neural_state = torch.tensor([9.])
        previous = self.target.last_generation_neural_state
        global_rng = torch.get_rng_state()
        result = evaluate_round(self.target, self.region, self.prefix, 4,
            draft_generator=torch.Generator().manual_seed(23), verifier_generator=torch.Generator().manual_seed(24))
        self.assertGreaterEqual(result.tokens.numel(), 1)
        self.assertLessEqual(result.tokens.numel(), 4)
        self.assertEqual(result.target_calls, result.tokens.numel())
        self.assertEqual(before, tensor_checksum(self.target.state_dict().values()))
        self.assertEqual(auxiliary, tensor_checksum(self.region.state_dict().values()))
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))
        self.assertEqual(self.target.last_generation_stop_reason, 'visible outside turn')
        self.assertIs(self.target.last_generation_neural_state, previous)

    def test_interruption_restores_both_rng_frontiers_and_releases_real_paged_cache(self):
        with tempfile.TemporaryDirectory(prefix='native-draft-private-pages-') as directory:
            pager = WorkingAttentionPager(Path(directory), resident_budget_bytes=4096,
                scratch_budget_bytes=2 * 1024**2, device_tile_budget_bytes=2 * 1024**2, page_tokens=2)
            self.target.configure_working_attention(pager)
            draft, verifier = torch.Generator().manual_seed(25), torch.Generator().manual_seed(26)
            draft_before, verifier_before = draft.get_state(), verifier.get_state()
            calls = 0
            def stop():
                nonlocal calls
                calls += 1
                return calls == 4  # Both RNGs advanced; no result was published.
            try:
                with self.assertRaises(DraftEvaluationInterrupted):
                    evaluate_round(self.target, self.region, self.prefix, 4,
                        draft_generator=draft, verifier_generator=verifier, stop_check=stop,
                        admit=lambda size, label: pager.admit_compute(size, label))
                self.assertTrue(torch.equal(draft_before, draft.get_state()))
                self.assertTrue(torch.equal(verifier_before, verifier.get_state()))
                self.assertEqual(pager.status()['livePages'], 0)
                self.assertIsNone(pager._cancelled)
            finally:
                pager.close()

    def test_admission_or_proposal_failure_cannot_leak_metadata_rng_or_cache(self):
        draft, verifier = torch.Generator().manual_seed(27), torch.Generator().manual_seed(28)
        before = (draft.get_state(), verifier.get_state())
        self.target.last_generation_stop_reason = 'untouched'
        with self.assertRaisesRegex(MemoryError, 'admission'):
            evaluate_round(self.target, self.region, self.prefix, 4, draft_generator=draft,
                verifier_generator=verifier,
                admit=lambda *_: (_ for _ in ()).throw(MemoryError('admission refused')))
        with patch.object(self.region, 'propose', side_effect=RuntimeError('private proposal failed')), \
                self.assertRaisesRegex(RuntimeError, 'proposal failed'):
            evaluate_round(self.target, self.region, self.prefix, 4, draft_generator=draft, verifier_generator=verifier)
        self.assertEqual(self.target.last_generation_stop_reason, 'untouched')
        self.assertTrue(torch.equal(draft.get_state(), before[0]))
        self.assertTrue(torch.equal(verifier.get_state(), before[1]))

    def test_single_long_forward_is_not_an_exact_verifier_for_grown_prefix_routed_experts(self):
        target = tiny_target(experts=True)
        sequence = torch.tensor([[259, 65, 260, 66, 67, 68]])
        with torch.no_grad():
            block = target(sequence)['logits'][0]
            prefixes = torch.stack([target(sequence[:, :end])['logits'][0, -1]
                for end in range(3, sequence.shape[1] + 1)])
        maximum = float((block[2:] - prefixes).abs().max())
        self.assertGreater(maximum, 1e-5,
            'this fixture must expose complete-prefix expert routing, not silently declare block verification exact')

    def test_no_expert_forward_prefix_distributions_match_within_float_tolerance(self):
        sequence = torch.tensor([[259, 65, 260, 66, 67, 68]])
        with torch.no_grad():
            block = self.target(sequence)['logits'][0]
            prefixes = torch.stack([self.target(sequence[:, :end])['logits'][0, -1]
                for end in range(3, sequence.shape[1] + 1)])
        self.assertTrue(torch.allclose(block[2:], prefixes, atol=2e-5, rtol=2e-5))


if __name__ == '__main__':
    unittest.main()
