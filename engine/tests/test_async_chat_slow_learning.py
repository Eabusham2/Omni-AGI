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
from omni_core.offload import ResourcePolicy


class AsyncChatSlowLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(781)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-chat-slow-")
        self.path = Path(self.temporary.name) / "brain"
        # These tiny learning fixtures do not test the desktop's production
        # disk-reserve policy, which can reject all writes on a busy host.
        reserve = mock.patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self) -> AdaptiveBrain:
        return AdaptiveBrain.create(
            "async-chat-slow",
            self.path,
            OmniConfig.micro(
                max_seq_len=32,
                working_memory_slots=12,
                online_learning=True,
                online_steps=1,
                learn_from_own_messages=False,
                growth_novelty_threshold=1.0,
                vision_enabled=False,
                image_enabled=False,
                audio_enabled=False,
                video_enabled=False,
            ),
            initialize_ground_up=True,
        )

    def test_fast_turn_commits_before_slow_replay_and_job_survives_restart(self):
        brain = self.make_brain()
        experiences_before = brain.counters["experiences"]
        fast_before = brain._fast_synapse_checksum()
        assemblies_before = len(brain.memory.assemblies)

        with mock.patch.object(
            brain,
            "_optimize_experience",
            side_effect=AssertionError("slow backward must not run in chat"),
        ), mock.patch.object(brain, "save", wraps=brain.save) as save:
            result = brain.chat(
                "Hi there. What are you thinking about?",
                max_new_tokens=1,
                seed=91,
                turn_id="fast-before-slow",
                defer_slow_learning=True,
            )

        self.assertTrue(result["turnCommitted"])
        self.assertEqual(result["trace"]["slow_mutation_stage"], "queued-background")
        self.assertFalse(result["trace"]["slow_mutation_applied"])
        self.assertEqual(brain.counters["experiences"], experiences_before + 1)
        self.assertGreater(len(brain.memory.assemblies), assemblies_before)
        self.assertNotEqual(brain._fast_synapse_checksum(), fast_before)
        self.assertEqual(len(brain.pending_chat_slow_learning), 1)
        self.assertEqual(save.call_count, 1)
        job_id = brain.pending_chat_slow_learning[0]["jobId"]
        brain.close()

        restarted = AdaptiveBrain.load(self.path, "async-chat-slow")
        self.assertEqual(
            [item["jobId"] for item in restarted.pending_chat_slow_learning],
            [job_id],
        )
        with mock.patch.object(
            restarted,
            "_optimize_experience",
            return_value={
                "loss": 0.25,
                "language_loss": 0.20,
                "idea_loss": 0.05,
                "stability_loss": 0.0,
            },
        ), mock.patch.object(restarted, "_maybe_grow", return_value=False):
            committed = restarted.consolidate_pending_chat_learning(job_id)
        self.assertTrue(committed["processed"])
        self.assertEqual(committed["transaction"], "atomic-candidate-promoted")
        self.assertEqual(restarted.pending_chat_slow_learning, [])
        restarted.close()

        verified = AdaptiveBrain.load(self.path, "async-chat-slow")
        self.assertIn(job_id, verified.completed_chat_slow_learning)
        repeated = verified.consolidate_pending_chat_learning(job_id)
        self.assertFalse(repeated["processed"])
        self.assertTrue(repeated["idempotent"])
        verified.close()

    def test_post_commit_finalization_failure_never_replays_optimizer_step(self):
        brain = self.make_brain()
        brain.chat(
            "A river bell rang after the rain.",
            max_new_tokens=1,
            seed=97,
            turn_id="post-commit-finalization",
            defer_slow_learning=True,
        )
        job_id = brain.pending_chat_slow_learning[0]["jobId"]
        with mock.patch.object(
            brain, "_optimize_experience", return_value={"loss": 0.25}
        ) as optimize, mock.patch.object(
            brain, "_maybe_grow", return_value=False
        ), mock.patch.object(
            brain.state_store,
            "publish",
            side_effect=OSError("simulated pointer publication failure"),
        ):
            committed = brain.consolidate_pending_chat_learning(job_id)

        self.assertTrue(committed["processed"])
        self.assertIn("simulated pointer", committed["postCommitFinalizationError"])
        self.assertEqual(optimize.call_count, 1)
        self.assertEqual(brain.pending_chat_slow_learning, [])
        repeated = brain.consolidate_pending_chat_learning(job_id)
        self.assertFalse(repeated["processed"])
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(optimize.call_count, 1)
        brain.close()

        restarted = AdaptiveBrain.load(self.path, "async-chat-slow")
        self.assertIn(job_id, restarted.completed_chat_slow_learning)
        self.assertFalse(
            restarted.consolidate_pending_chat_learning(job_id)["processed"]
        )
        restarted.close()

    def test_pre_commit_failure_keeps_the_exact_job_pending_for_retry(self):
        brain = self.make_brain()
        brain.chat(
            "A silver orchard grew beside the station.",
            max_new_tokens=1,
            seed=98,
            turn_id="pre-commit-failure",
            defer_slow_learning=True,
        )
        job_id = brain.pending_chat_slow_learning[0]["jobId"]
        checksum_before = brain._slow_parameter_checksum()
        pointer_before = brain.mutable_state_manifest
        with mock.patch.object(
            brain, "_optimize_experience", return_value={"loss": 0.25}
        ), mock.patch.object(
            brain, "_maybe_grow", return_value=False
        ), mock.patch.object(
            brain.state_store,
            "stage_generation",
            side_effect=OSError("simulated pre-commit storage pause"),
        ):
            with self.assertRaisesRegex(OSError, "pre-commit storage pause"):
                brain.consolidate_pending_chat_learning(job_id)

        self.assertEqual(brain._slow_parameter_checksum(), checksum_before)
        self.assertEqual(brain.mutable_state_manifest, pointer_before)
        self.assertEqual(
            [item["jobId"] for item in brain.pending_chat_slow_learning],
            [job_id],
        )
        brain.close()
        restarted = AdaptiveBrain.load(self.path, "async-chat-slow")
        self.assertEqual(
            [item["jobId"] for item in restarted.pending_chat_slow_learning],
            [job_id],
        )
        restarted.close()

    def test_fast_episode_has_no_question_or_reply_grammar_gate(self):
        brain = self.make_brain()
        text = "What changed in the harbor? Reply with the nearby signal."
        segments = brain.memory._segments(text)
        report = brain._learn_sequence_associations(text, retain_exact=True)
        self.assertEqual(report["exactEpisodicSegments"], len(segments))
        self.assertEqual(report["temporalLinks"], max(0, len(segments) - 1))
        self.assertEqual(report["retentionMode"], "detailed-exact-and-statistical")
        brain.close()

    def test_unmocked_background_replay_changes_and_persists_decoder_weights(self):
        # This is a tiny isolated fixture. A host below the desktop's disk
        # reserve must still be able to verify the gradient path without
        # weakening the application's real storage policy.
        reserve = mock.patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)
        brain = self.make_brain()
        result = brain.chat(
            "A copper kite crossed the harbor at dawn.",
            max_new_tokens=1,
            seed=93,
            turn_id="real-slow-parameter-step",
            defer_slow_learning=True,
        )
        self.assertTrue(result["turnCommitted"])
        self.assertEqual(len(brain.pending_chat_slow_learning), 1)
        job_id = brain.pending_chat_slow_learning[0]["jobId"]
        slow_before = brain._slow_parameter_checksum()
        cortical_before = brain._cortical_parameter_checksum()
        embedding_before = brain.decoder.embedding.weight.detach().cpu().clone()
        steps_before = brain.counters["training_steps"]

        committed = brain.consolidate_pending_chat_learning(job_id)
        self.assertTrue(committed["processed"])
        self.assertTrue(committed["corticalParametersUpdated"])
        self.assertIn("decoder", committed["updatedTrainableModules"])
        self.assertNotEqual(
            committed["corticalParameterChecksumBefore"],
            committed["corticalParameterChecksumAfter"],
        )
        self.assertNotEqual(
            committed["parameterChecksumBefore"],
            committed["parameterChecksumAfter"],
        )
        self.assertNotEqual(brain._slow_parameter_checksum(), slow_before)
        self.assertNotEqual(brain._cortical_parameter_checksum(), cortical_before)
        self.assertTrue(committed["corticalParametersUpdated"])
        self.assertEqual(committed["corticalParameterChecksumBefore"], cortical_before)
        self.assertEqual(
            committed["corticalParameterChecksumAfter"],
            brain._cortical_parameter_checksum(),
        )
        self.assertFalse(
            torch.equal(
                brain.decoder.embedding.weight.detach().cpu(), embedding_before
            )
        )
        self.assertGreater(brain.counters["training_steps"], steps_before)
        self.assertEqual(brain.pending_chat_slow_learning, [])
        brain.close()

        restarted = AdaptiveBrain.load(self.path, "async-chat-slow")
        self.assertEqual(restarted._slow_parameter_checksum(), committed["parameterChecksumAfter"])
        self.assertEqual(
            restarted._cortical_parameter_checksum(),
            committed["corticalParameterChecksumAfter"],
        )
        self.assertIn(job_id, restarted.completed_chat_slow_learning)
        restarted.close()

    def test_continuous_retention_changes_slow_replay_strength(self):
        brain = self.make_brain()
        low = brain._enqueue_chat_slow_learning(
            turn_id="low-turn",
            input_sha256="a" * 64,
            human_message_id="low-message",
            experience={
                "assembly_id": "missing-low",
                "novelty": 0.05,
                "retention_prediction_error": 0.05,
                "memory_settling": {
                    "retentionScore": 0.05,
                    "reinforcementDrive": 0.01,
                    "unfinishedScore": 0.01,
                    "signals": {
                        "reuse": 0.0,
                        "salience": 0.02,
                        "stability": 0.0,
                        "recurrence": 0.0,
                        "activation": 0.05,
                        "interference": 0.95,
                    },
                },
            },
        )
        high = brain._enqueue_chat_slow_learning(
            turn_id="high-turn",
            input_sha256="b" * 64,
            human_message_id="high-message",
            experience={
                "assembly_id": "missing-high",
                "novelty": 0.95,
                "retention_prediction_error": 0.9,
                "memory_settling": {
                    "retentionScore": 0.9,
                    "reinforcementDrive": 0.9,
                    "unfinishedScore": 0.8,
                    "signals": {
                        "reuse": 0.9,
                        "salience": 0.95,
                        "stability": 0.8,
                        "recurrence": 0.9,
                        "activation": 0.9,
                        "interference": 0.02,
                    },
                },
            },
        )
        assert low is not None and high is not None
        self.assertGreater(high["priority"], low["priority"])
        self.assertGreater(high["replayStrength"], low["replayStrength"])
        self.assertGreater(low["replayStrength"], 0.0)
        self.assertEqual(low["fastEpisodePolicy"], "unconditional-generic")
        brain.close()


if __name__ == "__main__":
    unittest.main()
