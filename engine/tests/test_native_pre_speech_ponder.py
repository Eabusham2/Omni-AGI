"""Native Ponder must be optional, measured, and condition same-turn speech."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.brain import ChatGenerationCancelled
from omni_core.model import ACTION_KINDS
from omni_core.offload import ResourcePolicy


class NativePreSpeechPonderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="omni-native-ponder-")
        self.addCleanup(temporary.cleanup)
        reserve = mock.patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)
        # An in-memory micro fixture exercises native Ponder without Build.
        self.brain = AdaptiveBrain(
            "native-ponder-fixture",
            Path(temporary.name) / "brain",
            OmniConfig.micro(
                max_seq_len=32,
                working_memory_slots=12,
                online_learning=False,
                learn_from_own_messages=False,
                growth_novelty_threshold=1.0,
                vision_enabled=False,
                image_enabled=False,
                audio_enabled=False,
                video_enabled=False,
            ),
        )
        self.addCleanup(self.brain.close)

    def select_head(self, kind, confidence_bias=20.0):
        with torch.no_grad():
            for head in (
                self.brain.decoder.action_policy,
                self.brain.decoder.internal_action_policy,
            ):
                head.projection.weight.zero_()
                head.projection.bias.zero_()
                head.projection.bias[ACTION_KINDS.index(kind)] = confidence_bias

    def latent_inputs(self):
        with torch.no_grad():
            combined = torch.linspace(
                -0.3, 0.7, self.brain.config.idea_dim,
                device=self.brain.device,
            ).unsqueeze(0)
            return combined, self.brain.idea_adapter(combined)

    def ponder(self, **overrides):
        arguments = {
            "seed": 719,
            "confidence": 0.99,
            "pass_budget": 4,
            "resource_budget_ms": 60_000,
            "noise": 0.1,
        }
        arguments.update(overrides)
        return self.brain._native_pre_speech_ponder(
            *self.latent_inputs(), **arguments
        )

    def test_private_recurrence_changes_memory_without_training_or_scratch_leak(self):
        brain = self.brain
        checksum = brain.parameter_checksum()
        liquid = brain.liquid_state.clone()
        membrane = brain.router.population.membrane.clone()
        spikes = brain.router.population.spike_count.clone()
        rng = torch.random.get_rng_state().clone()
        with mock.patch.object(
            brain.resource_policy, "status", return_value={}
        ), mock.patch.object(brain.router, "route", wraps=brain.router.route) as route:
            refined, trace = self.ponder()
        self.assertTrue(trace["activated"])
        self.assertEqual(trace["activated_by"], "learned-action-head")
        self.assertEqual(trace["passes"], route.call_count)
        self.assertEqual(trace["router_steps"], 2 * trace["passes"])
        self.assertGreater(trace["passes"], 0)
        self.assertGreater(trace["memory_bias_delta"], 0.0)
        self.assertEqual(trace["seed"], 719)
        self.assertEqual(trace["phase"], "pre-speech")
        self.assertTrue(trace["private"])
        self.assertFalse(trace["visibleMagicTags"])
        self.assertTrue(all(call.kwargs["learn"] is False for call in route.call_args_list))
        self.assertEqual(brain.parameter_checksum(), checksum)
        self.assertTrue(torch.equal(brain.liquid_state, liquid))
        self.assertTrue(torch.equal(brain.router.population.membrane, membrane))
        self.assertTrue(torch.equal(brain.router.population.spike_count, spikes))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        with mock.patch.object(brain.resource_policy, "status", return_value={}):
            repeated, repeated_trace = self.ponder()
        self.assertTrue(torch.equal(refined, repeated))
        self.assertEqual(trace["passes"], repeated_trace["passes"])
        self.assertEqual(trace["memory_bias_delta"], repeated_trace["memory_bias_delta"])

    def test_pressure_measures_zero_passes_and_preserves_generation_memory(self):
        with mock.patch.object(
            self.brain.resource_policy, "status", return_value={"memoryPressure": True}
        ), mock.patch.object(self.brain.router, "route") as route:
            refined, trace = self.ponder()
        route.assert_not_called()
        self.assertEqual(trace["passes"], 0)
        self.assertEqual(trace["router_steps"], 0)
        self.assertEqual(trace["memory_bias_delta"], 0.0)
        self.assertEqual(trace["stop_reason"], "resource-pressure")
        self.assertTrue(torch.equal(refined, self.latent_inputs()[1]))

    def test_convergence_stops_before_spending_the_requested_pass_budget(self):
        def stable_liquid(current, **_kwargs):
            return current, {"threshold_offset": torch.zeros(1)}

        def stable_route(driven, **_kwargs):
            return driven, {}

        with mock.patch.object(
            self.brain.resource_policy, "status", return_value={}
        ), mock.patch.object(
            self.brain.liquid, "forward", side_effect=stable_liquid
        ), mock.patch.object(
            self.brain.router, "route", side_effect=stable_route
        ):
            _refined, trace = self.ponder(noise=0.0)
        self.assertTrue(trace["converged"])
        self.assertEqual(trace["passes"], 1)
        self.assertEqual(trace["stop_reason"], "converged")

    def test_resource_deadline_is_not_reported_as_executed_passes(self):
        with mock.patch.object(
            self.brain.resource_policy, "status", return_value={}
        ), mock.patch("omni_core.brain.time.perf_counter", side_effect=[0.0, 2.0, 2.0]):
            _refined, trace = self.ponder(resource_budget_ms=1)
        self.assertEqual(trace["passes"], 0)
        self.assertEqual(trace["stop_reason"], "resource-budget")

    def test_cancel_after_first_pass_restores_private_scratch(self):
        brain = self.brain
        liquid = brain.liquid_state.clone()
        membrane = brain.router.population.membrane.clone()
        spikes = brain.router.population.spike_count.clone()
        cancellation = mock.Mock(side_effect=[False, True])
        with mock.patch.object(brain.resource_policy, "status", return_value={}):
            with self.assertRaises(ChatGenerationCancelled):
                self.ponder(cancel_check=cancellation)
        self.assertTrue(torch.equal(brain.liquid_state, liquid))
        self.assertTrue(torch.equal(brain.router.population.membrane, membrane))
        self.assertTrue(torch.equal(brain.router.population.spike_count, spikes))

    def test_learned_ponder_conditions_all_candidates_and_stream_replay(self):
        brain = self.brain
        self.select_head("ponder")
        measurements = {}
        stream = []
        actual_ponder = brain._native_pre_speech_ponder
        actual_generate = brain.decoder.generate

        def ponder(*args, **kwargs):
            refined, trace = actual_ponder(*args, **kwargs)
            measurements["bias"] = refined.clone()
            measurements["trace"] = dict(trace)
            return refined, trace

        def generate(*args, **kwargs):
            self.assertIn("bias", measurements, "speech ran before selected Ponder")
            self.assertTrue(torch.equal(kwargs["memory_bias"], measurements["bias"]))
            return actual_generate(*args, **kwargs)

        with mock.patch.object(
            brain.resource_policy, "status", return_value={}
        ), mock.patch.object(
            brain, "_native_pre_speech_ponder", side_effect=ponder
        ) as recurrent, mock.patch.object(
            brain.decoder, "generate", side_effect=generate
        ) as decode, mock.patch.object(
            brain, "idle_cycle", side_effect=AssertionError("post-response Ponder")
        ):
            result = brain.chat(
                "Compare both routes.", max_new_tokens=2, seed=719,
                stream_callback=lambda kind, payload: stream.append((kind, payload)),
            )
        recurrent.assert_called_once()
        self.assertGreaterEqual(decode.call_count, 2)
        trace = result["trace"]
        self.assertEqual(trace["ponder"], measurements["trace"])
        self.assertEqual(trace["ponder_steps"], trace["ponder"]["passes"])
        actions = [payload["action"] for kind, payload in stream if kind == "action"]
        ponder_actions = [action for action in actions if action["kind"] == "ponder"]
        self.assertEqual(len(ponder_actions), 1)
        self.assertTrue(ponder_actions[0]["arguments"]["completedInTurn"])
        self.assertEqual(ponder_actions[0]["arguments"]["ponderTrace"], trace["ponder"])
        self.assertLess(
            next(i for i, (kind, _) in enumerate(stream) if kind == "action"),
            next(i for i, (kind, _) in enumerate(stream) if kind == "token"),
        )
        self.assertFalse(trace["textual_memory_injected"])
        self.assertFalse(result["runtimeCard"]["hidden_behavioral_prompt"])

    def test_talk_stop_and_indecisive_heads_do_not_force_ponder(self):
        for kind, bias in (("talk", 20.0), ("stop", 20.0), ("ponder", 0.01)):
            with self.subTest(kind=kind, bias=bias):
                self.select_head(kind, bias)
                with mock.patch.object(
                    self.brain, "_native_pre_speech_ponder",
                    side_effect=AssertionError("Ponder bypassed learned selection gate"),
                ):
                    result = self.brain.chat(
                        "A new ordinary observation.", max_new_tokens=1, seed=31
                    )
                trace = result["trace"]
                self.assertFalse(trace["ponder"]["activated"])
                self.assertEqual(trace["ponder_steps"], 0)
                self.assertEqual(trace["ponder"]["stop_reason"], "not-selected")
                self.assertNotIn(
                    "recurrent-refinement", [step["stage"] for step in trace["steps"]]
                )

    def test_chat_cancellation_during_ponder_commits_nothing_and_emits_no_speech(self):
        brain = self.brain
        self.select_head("ponder")
        cancelled = False
        actual_route = brain.router.route
        before = brain.parameter_checksum()
        before_count = brain.counters["inference_count"]
        stream = []

        def run_ponder(*args, **kwargs):
            def route(*route_args, **route_kwargs):
                nonlocal cancelled
                result = actual_route(*route_args, **route_kwargs)
                cancelled = True
                return result
            with mock.patch.object(brain.router, "route", side_effect=route):
                return AdaptiveBrain._native_pre_speech_ponder(brain, *args, **kwargs)

        with mock.patch.object(
            brain.resource_policy, "status", return_value={}
        ), mock.patch.object(
            brain, "_native_pre_speech_ponder", side_effect=run_ponder
        ), mock.patch.object(brain.decoder, "generate") as decode:
            with self.assertRaises(ChatGenerationCancelled):
                brain.chat(
                    "Compare both routes.", max_new_tokens=1, seed=719,
                    cancel_check=lambda: cancelled,
                    stream_callback=lambda kind, payload: stream.append((kind, payload)),
                )
        decode.assert_not_called()
        self.assertEqual(stream, [])
        self.assertEqual(brain.parameter_checksum(), before)
        self.assertEqual(brain.counters["inference_count"], before_count)


if __name__ == "__main__":
    unittest.main()
