"""Constructor-free checks of chat replay's exact structural lookup."""

import unittest
from types import SimpleNamespace

from omni_core.brain import AdaptiveBrain


class NoAssemblyTraversal:
    def __iter__(self):
        raise AssertionError("foreground replay admission scanned all assemblies")


class RetentionRecorder:
    def __init__(self):
        self.calls = []

    def assess_retention_candidate(self, **signals):
        self.calls.append(signals)
        return {
            "retentionScore": 0.4,
            "slowReplayPriority": 0.3,
            "fadePressure": 0.2,
            "evidence": {},
            "reasons": [],
            "fastEpisodePolicy": "whole-experience",
        }


class ChatReplayIndexContract(unittest.TestCase):
    def state(self, records):
        return SimpleNamespace(
            brain_id="fixture-identity",
            config=SimpleNamespace(
                online_learning=True, online_steps=1, consolidation_rate=0.06
            ),
            memory=SimpleNamespace(
                assemblies=NoAssemblyTraversal(), assembly_by_id=records
            ),
            memory_lifecycle=RetentionRecorder(),
            pending_chat_slow_learning=[],
            completed_chat_slow_learning=[],
        )

    def enqueue(self, state, assembly_id="current-episode"):
        return AdaptiveBrain._enqueue_chat_slow_learning(
            state,
            turn_id="turn-1",
            input_sha256="a" * 64,
            human_message_id="human-1",
            experience={
                "assembly_id": assembly_id,
                "novelty": 0.7,
                "memory_settling": {"signals": {"recurrence": 0.2}},
            },
        )

    def test_exact_episode_lookup_without_substrate_traversal(self):
        state = self.state({"current-episode": {"rehearsals": 17}})
        first = self.enqueue(state)
        self.assertEqual(state.memory_lifecycle.calls[0]["rehearsals"], 17)
        self.assertEqual(state.memory_lifecycle.calls[0]["observations"], 17)
        self.assertIs(self.enqueue(state), first)
        self.assertEqual(len(state.pending_chat_slow_learning), 1)
        self.assertEqual(len(state.memory_lifecycle.calls), 1)

    def test_missing_episode_uses_existing_conservative_default(self):
        state = self.state({})
        self.assertIsNotNone(self.enqueue(state, "unknown-episode"))
        self.assertEqual(state.memory_lifecycle.calls[0]["rehearsals"], 1)
        self.assertEqual(state.memory_lifecycle.calls[0]["observations"], 1)


if __name__ == "__main__":
    unittest.main()
