import contextvars
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.config import OmniConfig
from omni_core.model import OmniDecoder
from omni_core.cooperative_generation import CooperativeNativeGeneration
from omni_core.working_attention_paging import WorkingAttentionPager


class CooperativeGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(912)
        self.decoder = OmniDecoder(OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1,
            n_layers=1, d_ff=16, max_seq_len=16, working_memory_slots=4,
            vsa_dim=16, router_neurons=4)).eval()
        self.prompt = torch.tensor([[259, 65, 260]])

    def finish(self, session):
        while not session.done:
            session.advance()
        return session.result

    def test_interleaved_seeded_candidates_match_independent_native_generate(self):
        kwargs = dict(max_new_tokens=5, printable_only=True, top_k=12, noise=.05)
        expected = [self.decoder.generate(self.prompt, seed=seed, temperature=temp, **kwargs)
                    for seed, temp in ((5, .2), (9, .95))]
        previous_stop = self.decoder.last_generation_stop_reason
        candidates = [CooperativeNativeGeneration(self.decoder, self.prompt, seed=seed, temperature=temp, **kwargs)
                      for seed, temp in ((5, .2), (9, .95))]
        events = []
        try:
            while not all(candidate.done for candidate in candidates):
                for index, candidate in enumerate(candidates):
                    if not candidate.done:
                        event = candidate.advance()
                        if event is not None:
                            events.append((index, event['step']))
                        self.assertEqual(self.decoder.last_generation_stop_reason, previous_stop)
            for actual, (tokens, entropies) in zip(candidates, expected):
                self.assertTrue(torch.equal(actual.result[0], tokens))
                self.assertEqual(actual.result[1], entropies)
                self.assertTrue(actual.closed)
                self.assertIsNone(actual.cache_owner[0])
                self.assertLessEqual(actual.mean_log_probability, 0.)
            self.assertEqual(events[:4], [(0, 0), (1, 0), (0, 1), (1, 1)])
        finally:
            for candidate in candidates:
                candidate.close()

    def test_cancellation_and_steer_release_cache_and_keep_private_prefix(self):
        for reason in ('cancelled', 'steer_check'):
            state = {'stop': False}
            candidate = CooperativeNativeGeneration(self.decoder, self.prompt, seed=5, max_new_tokens=8,
                printable_only=False,
                **{reason: lambda: state['stop']})
            candidate.advance()
            state['stop'] = True
            tokens, _ = self.finish(candidate)
            self.assertEqual(tokens.shape[1], self.prompt.shape[1] + 1)
            self.assertEqual(candidate.stop_reason, 'cancelled' if reason == 'cancelled' else 'steered')
            self.assertIsNone(candidate.cache_owner[0])

    def test_each_advance_exits_its_context_scope_before_other_branch(self):
        marker = contextvars.ContextVar('private_generation_test', default=None)
        class Scope:
            def __init__(self):
                self.entries = 0
            @contextmanager
            def activate(self, *_args):
                token = marker.set(self.entries)
                self.entries += 1
                try:
                    yield self
                finally:
                    marker.reset(token)
            def admit_compute(self, *_args, **_kwargs):
                self.assert_active()
            def assert_active(self):
                if marker.get() is None:
                    raise AssertionError('branch entered computation without scope')
        scope = Scope()
        original = self.decoder._generation_step
        def step(*args):
            scope.assert_active()
            with patch.object(self.decoder, '_working_pager', return_value=None):
                return original(*args)
        with patch.object(self.decoder, '_working_pager', return_value=scope), patch.object(self.decoder, '_generation_step', side_effect=step):
            left = CooperativeNativeGeneration(self.decoder, self.prompt, max_new_tokens=2, seed=1)
            right = CooperativeNativeGeneration(self.decoder, self.prompt, max_new_tokens=2, seed=2)
            try:
                while not left.done or not right.done:
                    for candidate in (left, right):
                        candidate.advance()
                        self.assertIsNone(marker.get())
            finally:
                left.close(); right.close()
        self.assertGreater(scope.entries, 2)

    def test_private_session_rejects_world_observation_or_output_emitters(self):
        for field in ('token_callback', 'conditioning_callback'):
            with self.assertRaisesRegex(ValueError, 'private candidates'):
                CooperativeNativeGeneration(self.decoder, self.prompt, **{field: lambda *_: None})

    def test_regional_decoder_metadata_is_also_private_between_advances(self):
        self.decoder.fixture_region = torch.nn.Module()
        region = self.decoder.fixture_region
        region.last_generation_stop_reason = 'regional-outside'
        actual = self.decoder._generation_step
        def step(*args):
            region.last_generation_stop_reason = 'regional-private'
            return actual(*args)
        session = CooperativeNativeGeneration(self.decoder, self.prompt, max_new_tokens=2, seed=7)
        with patch.object(self.decoder, '_generation_step', side_effect=step):
            try:
                while not session.done:
                    session.advance()
                    self.assertEqual(region.last_generation_stop_reason, 'regional-outside')
                self.assertEqual(session.regional_metadata[id(region)]['last_generation_stop_reason'], 'regional-private')
            finally:
                session.close()

    def test_real_paged_candidates_release_all_activity_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            pager = WorkingAttentionPager(Path(folder), resident_budget_bytes=4096,
                scratch_budget_bytes=2 * 1024**2, device_tile_budget_bytes=2 * 1024**2, page_tokens=2)
            self.decoder.configure_working_attention(pager)
            kwargs = dict(seed=91, max_new_tokens=5, temperature=.5, printable_only=False)
            expected, entropy = self.decoder.generate(self.prompt, **kwargs)
            self.assertEqual(pager.status()['livePages'], 0)
            session = CooperativeNativeGeneration(self.decoder, self.prompt, **kwargs)
            try:
                actual, actual_entropy = self.finish(session)
                self.assertTrue(torch.equal(actual, expected))
                self.assertEqual(actual_entropy, entropy)
                self.assertEqual(pager.status()['livePages'], 0)
                self.assertIsNone(pager._cancelled)
            finally:
                session.close(); pager.close()

    def test_exception_closes_private_pages_and_restores_decoder_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            pager = WorkingAttentionPager(Path(folder), resident_budget_bytes=4096,
                scratch_budget_bytes=2 * 1024**2, device_tile_budget_bytes=2 * 1024**2, page_tokens=2)
            self.decoder.configure_working_attention(pager)
            self.decoder.last_generation_stop_reason = 'outside-state'
            session = CooperativeNativeGeneration(self.decoder, self.prompt, max_new_tokens=4, seed=71,
                activity_callback=lambda *_: (_ for _ in ()).throw(RuntimeError('fixture callback failed')))
            try:
                with self.assertRaisesRegex(RuntimeError, 'fixture callback'):
                    session.advance()
                self.assertEqual(pager.status()['livePages'], 0)
                self.assertEqual(self.decoder.last_generation_stop_reason, 'outside-state')
                self.assertIsNone(pager._cancelled)
            finally:
                session.close(); pager.close()


if __name__ == '__main__':
    unittest.main()
