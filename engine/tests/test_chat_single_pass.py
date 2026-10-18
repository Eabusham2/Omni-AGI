"""A normal native reply must stream before completion without hidden re-decodes."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.offload import ResourcePolicy


class SinglePassChatTest(unittest.TestCase):
    def test_talk_streams_once_and_counts_native_boundary_tokens(self):
        self.run_reply(steer=False)

    def test_steer_keeps_already_streamed_reply_without_restarting_generation(self):
        self.run_reply(steer=True)

    def test_best_work_search_keeps_private_candidates_and_replays_only_one_selected_branch(self):
        self.run_reply(steer=False, best_work=True)

    def test_selected_latent_ponder_does_not_force_a_second_answer_decode(self):
        self.run_reply(steer=False, ponder_selected=True)

    def run_reply(self, *, steer, best_work=False, ponder_selected=False):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="omni-single-pass-") as root, mock.patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        ):
            brain = AdaptiveBrain("single-pass-fixture", Path(root) / "brain", OmniConfig.micro(
                max_seq_len=32, working_memory_slots=12, online_learning=False,
                d_model=8, idea_dim=8, n_heads=2, d_ff=16,
                vsa_dim=16, router_neurons=4, modality_channels=2,
                learn_from_own_messages=False, growth_novelty_threshold=1.0,
                vision_enabled=False, image_enabled=False, audio_enabled=False, video_enabled=False,
                best_work_enabled=best_work, best_work_exploration_branches=2,
            ))
            try:
                if ponder_selected:
                    # This fixture controls entry independently; the actual
                    # packed continuation head then transitions to Talk.
                    brain.decoder.internal_action_policy.projection.fill_ternary_(0)
                stream = []
                steer_requested = False
                def emit(kind, value):
                    nonlocal steer_requested
                    stream.append((kind, value))
                    if steer and kind == "token" and value.get("delta"):
                        steer_requested = True
                actual_generate = brain.decoder.generate
                def decode(*args, **kwargs):
                    self.assertIsNotNone(kwargs.get("token_callback"))
                    result = actual_generate(*args, **kwargs)
                    self.assertTrue(any(kind == "token" for kind, _ in stream), "no live token before generate returned")
                    return result
                selection = ({"talk": 0., "ponder": .99}, [{"kind": "ponder", "confidence": .99, "arguments": {}}]) if ponder_selected else ({"talk": 1.0, "ponder": 0.0}, [])
                actual_ponder = brain._native_pre_speech_ponder
                with mock.patch.object(brain, "_select_structured_actions", return_value=selection), \
                     mock.patch.object(brain, "_generation_activity_callback", return_value=lambda *args: None), \
                     mock.patch.object(brain, "_candidate_nll", side_effect=AssertionError("ordinary reply rescored privately")), \
                     mock.patch.object(brain, "_native_pre_speech_ponder", side_effect=actual_ponder if ponder_selected else AssertionError("unselected Ponder")), \
                     mock.patch.object(brain.decoder, "generate", side_effect=decode) as generation:
                    result = brain.chat("Hello", max_new_tokens=8 if steer else 3, seed=719,
                        stream_callback=emit, steer_check=lambda: steer_requested)
                self.assertEqual(generation.call_count, 1)
                self.assertEqual(result['trace']['ponder']['activated'], ponder_selected)
                if ponder_selected:
                    self.assertTrue(result['trace']['ponder']['completed'])
                    ponder_actions = [payload['action'] for kind, payload in stream if kind == 'action' and payload['action']['kind'] == 'ponder']
                    self.assertEqual(len(ponder_actions), 1)
                    self.assertTrue(ponder_actions[0]['arguments']['handledInTurn'])
                self.assertEqual(result["trace"]["single_pass_generation"], not best_work)
                if best_work:
                    self.assertGreater(len(result["trace"]["branches"]), 1)
                    self.assertTrue(result['trace']['best_work']['enabled'])
                    self.assertEqual(result['trace']['best_work']['candidateHostEffects'], 0)
                    self.assertEqual(result['trace']['best_work']['concurrency'], 'cooperative-interleaved')
                    activity = [payload for kind, payload in stream if kind == 'cognition-progress']
                    self.assertTrue(activity[0]['active'])
                    self.assertFalse(activity[-1]['active'])
                    self.assertFalse(result['trace']['best_work']['selection']['correctnessProven'])
                else:
                    self.assertEqual(len(result["trace"]["branches"]), 1)
                    self.assertIsNone(result["trace"]["branches"][0]["selfNll"])
                tokens = [value for kind, value in stream if kind == "token"]
                self.assertEqual(tokens[-1]["outputTokens"], result["trace"]["generated_token_count"])
                self.assertEqual([item["outputTokens"] for item in tokens], sorted(item["outputTokens"] for item in tokens))
                self.assertEqual("".join(item["delta"] for item in tokens), result["text"])
                if steer:
                    self.assertTrue(steer_requested)
                    self.assertTrue(result["steered"])
                    self.assertTrue(result["text"])
            finally:
                brain.close()


if __name__ == "__main__":
    unittest.main()
