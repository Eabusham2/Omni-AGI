import json
import tempfile
import unittest
from pathlib import Path

import torch

from omni_core import AdaptiveBrain, OmniConfig


class ContinuousChatLearningTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(812)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-continuous-chat-"
        )
        self.root = Path(self.temporary.name)
        self.brains = []

    def tearDown(self):
        for brain in self.brains:
            brain.events.close()
        self.temporary.cleanup()

    def make_brain(self, name: str, *, online_learning: bool) -> AdaptiveBrain:
        brain = AdaptiveBrain(
            name,
            self.root / name,
            OmniConfig.micro(
                max_seq_len=80,
                working_memory_slots=16,
                memory_resident_items=16,
                online_learning=online_learning,
                online_steps=1 if online_learning else 0,
                learn_from_own_messages=False,
                vision_enabled=False,
                image_enabled=False,
                audio_enabled=False,
                video_enabled=False,
            ),
        )
        self.brains.append(brain)
        return brain

    @staticmethod
    def afterimage(brain: AdaptiveBrain, assembly_id: str):
        return next(
            item
            for item in brain.memory_lifecycle.afterimage_items
            if str(item.get("assemblyId", "")) == assembly_id
        )

    def test_ordinary_narrative_chat_always_updates_fast_organic_memory(self):
        brain = self.make_brain("fast-narrative", online_learning=False)
        one_off = (
            "A pale moth crossed the lower window while the rain softened."
        )
        recurring = (
            "The cedar lantern beside the canal glowed amber after dusk."
        )
        unrelated = (
            "A brass kite drifted above the quiet winter orchard."
        )
        experiences_before = int(brain.counters["experiences"])
        plasticity_before = int(brain.router.synapses.plasticity_events.item())

        weak_turn = brain.chat(one_off, max_new_tokens=1, seed=1201)
        recurring_first = brain.chat(recurring, max_new_tokens=1, seed=1202)
        recurring_second = brain.chat(recurring, max_new_tokens=1, seed=1203)
        weak_id = next(
            item["assemblyId"]
            for item in brain.memory_lifecycle.afterimage_items
            if item.get("source") == "conversation"
            and int(item.get("rehearsals", 0)) == 1
        )
        weak_strength_before_interference = float(
            self.afterimage(brain, str(weak_id))["strength"]
        )
        unrelated_turn = brain.chat(unrelated, max_new_tokens=1, seed=1204)

        turns = (
            weak_turn,
            recurring_first,
            recurring_second,
            unrelated_turn,
        )
        self.assertEqual(
            int(brain.counters["experiences"]), experiences_before + len(turns)
        )
        self.assertEqual(
            brain.memory_lifecycle.settled_experiences, len(turns)
        )
        self.assertEqual(brain.memory_lifecycle.last_source, "conversation")
        self.assertGreater(
            int(brain.router.synapses.plasticity_events.item()),
            plasticity_before,
        )
        for turn in turns:
            trace = turn["trace"]
            self.assertFalse(trace["slow_mutation_requested"])
            self.assertFalse(trace["slow_mutation_applied"])
            self.assertTrue(trace["memory_settling"]["automatic"])
            self.assertFalse(trace["memory_settling"]["fixedStage"])
            self.assertTrue(
                trace["memory_settling"]["scoresRecomputedEachCycle"]
            )
            self.assertGreater(trace["spike_rate"], 0.0)
            self.assertGreater(trace["stdp_update"], 0.0)
            self.assertFalse(trace["textual_memory_injected"])
            self.assertFalse(trace["long_term_source_text_injected"])

        first_signals = recurring_first["trace"]["memory_settling"]["signals"]
        repeated_signals = recurring_second["trace"]["memory_settling"]["signals"]
        self.assertGreater(repeated_signals["reuse"], first_signals["reuse"])
        self.assertGreater(
            repeated_signals["recurrence"], first_signals["recurrence"]
        )
        self.assertLess(
            repeated_signals["decayPressure"], first_signals["decayPressure"]
        )
        self.assertGreater(
            recurring_second["trace"]["memory_settling"][
                "reinforcementDrive"
            ],
            recurring_first["trace"]["memory_settling"][
                "reinforcementDrive"
            ],
        )
        self.assertGreater(
            unrelated_turn["trace"]["memory_settling"]["signals"][
                "interference"
            ],
            0.0,
        )
        self.assertLess(
            float(self.afterimage(brain, str(weak_id))["strength"]),
            weak_strength_before_interference,
        )
        lifecycle = brain.memory_lifecycle.metadata()
        self.assertFalse(lifecycle["rawTextStored"])
        self.assertFalse(lifecycle["rawTokenIdsStored"])
        self.assertNotIn(one_off, json.dumps(lifecycle))
        self.assertFalse(hasattr(brain, "consolidate"))

if __name__ == "__main__":
    unittest.main()
