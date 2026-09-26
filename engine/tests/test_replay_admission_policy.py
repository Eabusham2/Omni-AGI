import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.memory_lifecycle import OrganicMemoryLifecycle
from omni_core.offload import DurableReplayBuffer, NeuralStateResourcePause


class _RecordingReplay:
    def __init__(self):
        self.values = []

    def append(self, value):
        self.values.append(value.detach().clone())


class _DiskPolicy:
    def __init__(self):
        self.blocked = False
        self.checks = []

    def require_disk(self, estimated_write_bytes, operation):
        self.checks.append((estimated_write_bytes, operation))
        if self.blocked:
            raise NeuralStateResourcePause(
                "latent replay paused at disk reserve",
                {"diskPressure": True, "paused": True},
            )
        return {"diskPressure": False}


class _BrainHarness:
    """Exercise only replay policy methods, without creating a model brain."""

    _replay_admission_probability = staticmethod(
        AdaptiveBrain._replay_admission_probability
    )
    _organic_replay_priority = AdaptiveBrain._organic_replay_priority
    _replay_priority_from_signals = staticmethod(
        AdaptiveBrain._replay_priority_from_signals
    )
    _append_replay = AdaptiveBrain._append_replay
    _append_selected_replay_batch = AdaptiveBrain._append_selected_replay_batch
    _enqueue_chat_slow_learning = AdaptiveBrain._enqueue_chat_slow_learning

    def __init__(self, replay=None):
        self.brain_id = "stable-replay-test-brain"
        self.memory_lifecycle = OrganicMemoryLifecycle()
        self.memory = SimpleNamespace(
            assembly_by_id={"assembly": {"rehearsals": 1}},
            assemblies=[{"id": "assembly", "rehearsals": 1}],
        )
        self.counters = {"experiences": 0, "idle_cognition_cycles": 0}
        self.replay = replay if replay is not None else _RecordingReplay()
        self.resource_pause = None
        self.config = SimpleNamespace(
            online_learning=True,
            online_steps=1,
            consolidation_rate=0.06,
        )
        self.pending_chat_slow_learning = []
        self.completed_chat_slow_learning = []


class ReplayAdmissionPolicyTests(unittest.TestCase):
    def test_low_experience_has_nonzero_chance_and_organic_reuse_raises_it(self):
        brain = _BrainHarness()
        weak = {
            "reuse": 0.0,
            "salience": 0.02,
            "stability": 0.0,
            "recurrence": 0.0,
            "activation": 0.02,
            "interference": 0.95,
        }
        recurring = {
            "reuse": 0.9,
            "salience": 0.8,
            "stability": 0.7,
            "recurrence": 0.9,
            "activation": 0.8,
            "interference": 0.01,
        }
        weak_priority = brain._replay_priority_from_signals(
            weak, rehearsals=1, novelty=0.05, prediction_error=0.05
        )
        recurring_priority = brain._replay_priority_from_signals(
            recurring, rehearsals=3, novelty=0.75, prediction_error=0.7
        )
        low = brain._replay_admission_probability(0.3, weak_priority)
        recurrent = brain._replay_admission_probability(0.3, recurring_priority)
        self.assertGreater(low, 0.0)
        self.assertLess(low, recurrent)
        self.assertGreater(
            brain._replay_admission_probability(0.58),
            brain._replay_admission_probability(0.3),
        )
        self.assertEqual(brain._replay_admission_probability(1.0), 1.0)
        self.assertNotIn("long_term_threshold", OmniConfig().to_dict())
        self.assertEqual(
            OmniConfig.from_dict({"long_term_threshold": 0.99}).long_term_threshold,
            0.99,
        )
        self.assertEqual(
            OmniConfig.from_external({"longTermThreshold": 0.99}).long_term_threshold,
            OmniConfig().long_term_threshold,
        )

    def test_seeded_low_replay_is_stable_but_later_cycles_get_new_chances(self):
        first = _BrainHarness()
        second = _BrainHarness()
        idea = torch.arange(8, dtype=torch.float32)
        decisions = []
        for cycle in range(1, 129):
            first.memory_lifecycle.cycle = cycle
            second.memory_lifecycle.cycle = cycle
            left = first._append_replay(
                idea, importance=0.3, assembly_id="assembly"
            )
            right = second._append_replay(
                idea, importance=0.3, assembly_id="assembly"
            )
            self.assertEqual(left, right)
            decisions.append(left)
        self.assertIn(True, decisions)
        self.assertIn(False, decisions)
        self.assertEqual(len(first.replay.values), sum(decisions))

    def test_replay_checkpoint_survives_reopen_and_reserve_pauses_before_write(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-policy-") as folder:
            path = Path(folder) / "replay.sqlite3"
            policy = _DiskPolicy()
            replay = DurableReplayBuffer(path, policy)
            brain = _BrainHarness(replay)
            idea = torch.linspace(-1.0, 1.0, 8)
            self.assertTrue(
                brain._append_replay(idea, importance=1.0, assembly_id="assembly")
            )
            checkpoint = replay.checkpoint()
            reopened = DurableReplayBuffer(path, policy, read_only=True)
            self.assertEqual(reopened.verify_checkpoint(checkpoint)["pendingExamples"], 0)
            self.assertTrue(torch.equal(reopened[0], idea))
            policy.blocked = True
            brain.memory_lifecycle.cycle = 9
            brain.memory_lifecycle.afterimage_items = [
                {"assemblyId": "assembly", "strength": 0.4}
            ]
            with mock.patch.object(
                brain.memory_lifecycle,
                "_measure_signals",
                return_value=(
                    {
                        "reuse": 0.2,
                        "salience": 0.3,
                        "stability": 0.2,
                        "recurrence": 0.1,
                        "activation": 0.3,
                        "interference": 0.1,
                    },
                    1,
                    (),
                ),
            ):
                priority = brain._organic_replay_priority(
                    assembly_id="assembly",
                    salience=0.3,
                    novelty=0.2,
                    prediction_error=0.2,
                    spike_rate=0.3,
                )
            with self.assertRaises(NeuralStateResourcePause):
                brain._append_replay(
                    idea + 1.0, importance=1.0, replay_priority=priority
                )
            self.assertEqual(len(replay), 1)
            self.assertEqual(reopened.verify_checkpoint(checkpoint)["pendingExamples"], 0)
            self.assertTrue(brain.resource_pause["readings"]["diskPressure"])
            self.assertEqual(brain.memory_lifecycle.cycle, 9)
            self.assertEqual(
                brain.memory_lifecycle.afterimage_items,
                [{"assemblyId": "assembly", "strength": 0.4}],
            )
            self.assertEqual(len(policy.checks), 2)

    def test_selected_microbatch_preserves_pause_evidence_and_stub_fallback(self):
        stub_brain = _BrainHarness()
        ideas = [torch.ones(2), torch.ones(2) * 2]
        stub_brain._append_selected_replay_batch(iter(ideas))
        self.assertEqual(len(stub_brain.replay.values), 2)

        with tempfile.TemporaryDirectory(prefix="omni-replay-batch-policy-") as folder:
            policy = _DiskPolicy()
            replay = DurableReplayBuffer(Path(folder) / "replay.sqlite3", policy)
            brain = _BrainHarness(replay)
            policy.blocked = True
            with self.assertRaises(NeuralStateResourcePause):
                brain._append_selected_replay_batch(iter(ideas))
            self.assertEqual(len(replay), 0)
            self.assertTrue(brain.resource_pause["readings"]["diskPressure"])
            self.assertEqual(len(policy.checks), 1)

            policy.blocked = False
            brain._append_selected_replay_batch(iter(ideas))
            self.assertEqual(len(replay), 2)
            self.assertEqual(len(policy.checks), 2)

    def test_chat_slow_job_still_queues_weak_turn_with_positive_weight(self):
        brain = _BrainHarness()
        record = brain._enqueue_chat_slow_learning(
            turn_id="weak-turn",
            input_sha256="a" * 64,
            human_message_id="human-message",
            experience={
                "assembly_id": "assembly",
                "novelty": 0.01,
                "retention_prediction_error": 0.01,
                "memory_settling": {
                    "signals": {
                        "reuse": 0.0,
                        "salience": 0.01,
                        "stability": 0.0,
                        "recurrence": 0.0,
                        "activation": 0.01,
                        "interference": 0.99,
                    }
                },
            },
        )
        self.assertIsNotNone(record)
        self.assertGreater(record["replayStrength"], 0.0)
        self.assertEqual(len(brain.pending_chat_slow_learning), 1)


if __name__ == "__main__":
    unittest.main()
