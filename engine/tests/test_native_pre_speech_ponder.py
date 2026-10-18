"""Adaptive native Ponder contracts without constructing or training a brain.

Fixed pass/deadline and floating-head fixtures were superseded by native
weighted heads. Ordinary and selected chat behavior is checked separately.
"""
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from omni_core.brain import AdaptiveBrain, ChatGenerationCancelled
from omni_core.model import ACTION_KINDS, PackedAdaptiveBitLinear


class NativePreSpeechPonderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.brain = object.__new__(AdaptiveBrain)
        self.brain.config = SimpleNamespace(liquid_dynamics=False, spiking_dynamics=True)
        self.brain.liquid_state = torch.zeros((1, 2))
        self.brain.idea_adapter = nn.Identity()
        # The packed head chooses Ponder while coordinate0 is negative and
        # Talk when it becomes positive. No float learned weight is installed.
        head = PackedAdaptiveBitLinear(2, len(ACTION_KINDS), bias=False)
        levels = torch.zeros((len(ACTION_KINDS), 2), dtype=torch.int8)
        levels[ACTION_KINDS.index('talk'), 0] = 1
        levels[ACTION_KINDS.index('ponder'), 0] = -1
        head.set_ternary_weight_(levels)
        self.brain.decoder = SimpleNamespace(internal_action_policy=head)
        self.population = SimpleNamespace(membrane=torch.tensor([.3]), spike_count=torch.tensor([4.]))
        self.routes = 0
        self.stop = False
        self.brain.resource_policy = SimpleNamespace(status=lambda: {})
        self.brain.router = SimpleNamespace(population=self.population, route=self.route)
        self.initial = torch.tensor([[-1., 0.]])

    def route(self, driven, **kwargs):
        self.assertFalse(kwargs['learn'])
        self.routes += 1
        self.population.membrane.add_(1.)
        self.population.spike_count.add_(2.)
        return torch.tensor([[-1. + .05 * self.routes, 0.]]), {}

    def ponder(self, **kwargs):
        return self.brain._native_pre_speech_ponder(self.initial, self.initial,
            seed=719, confidence=.99, noise=0., **kwargs)

    def test_learned_transition_can_continue_beyond_old_pass_and_time_caps(self):
        updates = []
        clock = itertools.count(0., 11.)
        with patch('omni_core.brain.time.perf_counter', side_effect=lambda: next(clock)):
            _, trace = self.ponder(progress_callback=updates.append)
        self.assertGreater(trace['passes'], 16)
        self.assertGreater(trace['elapsed_ms'], 10000)
        self.assertEqual(trace['stop_reason'], 'learned-action-transition')
        self.assertEqual(trace['continuationAction'], 'talk')
        self.assertTrue(trace['completed'])
        self.assertIsNone(trace['fixedPassLimit'])
        self.assertIsNone(trace['fixedTimeLimitMs'])
        self.assertTrue(updates[0]['active'])
        self.assertFalse(updates[-1]['active'])
        self.assertEqual(updates[-1]['stopReason'], trace['stop_reason'])

    def test_private_recurrence_preserves_native_weights_rng_and_scratch(self):
        before = {key: value.clone() for key, value in self.brain.decoder.internal_action_policy.state_dict().items()}
        liquid = self.brain.liquid_state.clone()
        membrane, spikes = self.population.membrane.clone(), self.population.spike_count.clone()
        rng = torch.random.get_rng_state().clone()
        refined, trace = self.ponder()
        self.assertGreater(trace['memory_bias_delta'], 0.)
        self.assertEqual(trace['router_steps'], 2 * trace['passes'])
        self.assertTrue(torch.equal(self.brain.liquid_state, liquid))
        self.assertTrue(torch.equal(self.population.membrane, membrane))
        self.assertTrue(torch.equal(self.population.spike_count, spikes))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        for key, value in self.brain.decoder.internal_action_policy.state_dict().items():
            self.assertTrue(torch.equal(before[key], value), key)
        self.routes = 0
        repeated, again = self.ponder()
        self.assertTrue(torch.equal(repeated, refined))
        self.assertEqual(again['passes'], trace['passes'])

    def test_convergence_stops_recurrence_even_when_head_still_selects_ponder(self):
        self.brain.router.route = lambda *_args, **_kwargs: (self.initial, {})
        _, trace = self.ponder()
        self.assertEqual(trace['passes'], 1)
        self.assertTrue(trace['converged'])
        self.assertEqual(trace['continuationAction'], 'ponder')
        self.assertEqual(trace['stop_reason'], 'converged')

    def test_resource_pressure_is_not_mislabeled_as_completed_thinking(self):
        self.brain.resource_policy.status = lambda: {'memoryPressure': True}
        updates = []
        refined, trace = self.ponder(progress_callback=updates.append)
        self.assertEqual(self.routes, 0)
        self.assertEqual(trace['passes'], 0)
        self.assertEqual(trace['stop_reason'], 'resource-pressure')
        self.assertFalse(trace['completed'])
        self.assertFalse(updates[-1]['active'])
        self.assertTrue(torch.equal(refined, self.initial))

    def test_steer_is_checked_after_router_before_another_private_operation(self):
        membrane = self.population.membrane.clone()
        _, trace = self.ponder(steer_check=lambda: self.routes >= 1)
        self.assertEqual(self.routes, 1)
        self.assertEqual(trace['stop_reason'], 'steered')
        self.assertEqual(trace['passes'], 0)
        self.assertEqual(trace['router_steps'], 2)
        self.assertFalse(trace['completed'])
        self.assertTrue(torch.equal(self.population.membrane, membrane))

    def test_cancel_restores_private_state_and_emits_terminal_cognition_only(self):
        membrane = self.population.membrane.clone()
        updates = []
        with self.assertRaises(ChatGenerationCancelled):
            self.ponder(cancel_check=lambda: self.routes >= 1, progress_callback=updates.append)
        self.assertTrue(torch.equal(self.population.membrane, membrane))
        self.assertFalse(updates[-1]['active'])
        self.assertEqual(updates[-1]['stopReason'], 'cancelled')
        self.assertNotIn('replyComplete', updates[-1])


if __name__ == '__main__':
    unittest.main()
