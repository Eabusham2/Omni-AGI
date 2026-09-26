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


class OrganicMemoryLifecycleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(491)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-memory-life-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(
        self, name: str = "memory-life", *, persisted_origin=False
    ) -> AdaptiveBrain:
        constructor = AdaptiveBrain.create if persisted_origin else AdaptiveBrain
        return constructor(
            name,
            self.root / name,
            OmniConfig.micro(
                max_seq_len=48,
                working_memory_slots=24,
                online_learning=False,
                learn_from_own_messages=False,
                vision_enabled=False,
                image_enabled=False,
                audio_enabled=False,
                video_enabled=False,
            ),
            **({"initialize_ground_up": True} if persisted_origin else {}),
        )

    @staticmethod
    def assembly(brain: AdaptiveBrain, assembly_id: str):
        return next(
            item for item in brain.memory.assemblies if item["id"] == assembly_id
        )

    def test_settling_is_automatic_and_high_use_outlasts_a_cooling_trace(self):
        brain = self.make_brain()
        with mock.patch.object(
            brain,
            "_train_evolution_replay_candidate",
            side_effect=AssertionError("evolution replay must not run organically"),
        ):
            low = brain.learn_experience(
                "A passing grey fleck crossed the lower window.",
                source="conversation",
                steps=0,
                importance=0.05,
            )
            low_id = str(low["assembly_id"])
            immediate = brain.workspace_snapshot()["memory"]
            self.assertFalse(low["memory_settling"]["fixedStage"])
            self.assertTrue(
                low["memory_settling"]["scoresRecomputedEachCycle"]
            )
            self.assertNotIn("stage", low["memory_settling"])
            self.assertEqual(immediate["automaticSettling"]["experienceCycles"], 1)
            self.assertFalse(
                immediate["automaticSettling"]["requiredManualAction"]
            )
            self.assertFalse(
                immediate["automaticSettling"]["visiblePonderForced"]
            )
            low_signals = low["memory_settling"]["signals"]
            self.assertEqual(
                set(low_signals),
                {
                    "activation",
                    "recurrence",
                    "reuse",
                    "salience",
                    "stability",
                    "interference",
                    "decayPressure",
                },
            )
            scratch_before = next(
                item
                for item in immediate["afterimageTrail"]["items"]
                if item["assemblyId"] == low_id
            )["strength"]

            high = None
            for iteration in range(4):
                high = brain.learn_experience(
                    "The amber harbor key opens the emergency gate.",
                    source="conversation",
                    steps=0,
                    importance=0.95,
                )
                if iteration == 0:
                    self.assertGreaterEqual(
                        float(high["memory_settling"]["signals"]["interference"]),
                        0.0,
                    )
            assert high is not None
            high_id = str(high["assembly_id"])
            self.assertNotIn("memory_stage", self.assembly(brain, high_id))
            self.assertIn("retention_score", self.assembly(brain, high_id))
            self.assertGreater(
                float(high["memory_settling"]["reinforcementDrive"]),
                float(low["memory_settling"]["reinforcementDrive"]),
            )
            self.assertGreater(
                float(high["memory_settling"]["signals"]["recurrence"]),
                float(low_signals["recurrence"]),
            )
            self.assertGreater(
                float(high["memory_settling"]["signals"]["reuse"]),
                float(low_signals["reuse"]),
            )
            self.assertLess(
                float(high["memory_settling"]["signals"]["decayPressure"]),
                float(low_signals["decayPressure"]),
            )

            for index in range(12):
                brain.learn_experience(
                    "Transient margin note %d passes through attention." % index,
                    source="reading",
                    steps=0,
                    importance=0.03,
                )

        # The low-use assembly remains in the authoritative substrate. Its
        # short-lived afterimage loses influence instead of being erased at
        # the instant a new item arrives.
        low_record = self.assembly(brain, low_id)
        high_record = self.assembly(brain, high_id)
        self.assertIn(low_id, brain.memory.assembly_vectors)
        scratch_after = next(
            (
                item
                for item in brain.memory_lifecycle.afterimage_items
                if item["assemblyId"] == low_id
            ),
            None,
        )
        if scratch_after is not None:
            self.assertLess(float(scratch_after["strength"]), scratch_before)
            self.assertGreaterEqual(int(scratch_after["ageCycles"]), 8)
            self.assertEqual(
                int(scratch_after["lastRescoredCycle"]),
                brain.memory_lifecycle.cycle,
            )
        self.assertGreater(
            float(high_record["memory_strength"]),
            float(low_record["memory_strength"]),
        )
        self.assertGreater(
            float(high_record["importance"]), float(low_record["importance"])
        )

        cooled_strength = float((scratch_after or {}).get("strength", 0.0))
        reheated = brain.learn_experience(
            "A passing grey fleck crossed the lower window.",
            source="conversation",
            steps=0,
            importance=0.90,
        )
        self.assertEqual(str(reheated["assembly_id"]), low_id)
        reheated_item = next(
            item
            for item in brain.memory_lifecycle.afterimage_items
            if item["assemblyId"] == low_id
        )
        self.assertGreater(float(reheated_item["strength"]), cooled_strength)
        brain.events.close()

    def test_continuous_scores_focus_and_afterimages_survive_restart(self):
        brain = self.make_brain("restart-life", persisted_origin=True)
        weak = brain.learn_experience(
            "A weak temporary pencil mark.", steps=0, importance=0.04
        )
        repeated = None
        for _ in range(4):
            repeated = brain.learn_experience(
                "The recurring safety route uses the north stair.",
                steps=0,
                importance=0.96,
            )
        assert repeated is not None
        before = brain.workspace_snapshot()["memory"]
        brain.save()
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root / "restart-life", "restart-life")
        after = reloaded.workspace_snapshot()["memory"]
        self.assertEqual(
            before["automaticSettling"]["cycles"],
            after["automaticSettling"]["cycles"],
        )
        self.assertEqual(
            before["afterimageTrail"]["count"],
            after["afterimageTrail"]["count"],
        )
        self.assertEqual(
            before["retentionDynamics"]["trackedAssemblies"],
            after["retentionDynamics"]["trackedAssemblies"],
        )
        self.assertIn(
            str(weak["assembly_id"]), reloaded.memory.assembly_vectors
        )
        repeated_record = self.assembly(
            reloaded, str(repeated["assembly_id"])
        )
        self.assertIn("retention_score", repeated_record)
        self.assertNotIn("memory_stage", repeated_record)
        self.assertGreater(after["activeFocus"]["count"], 0)
        self.assertGreater(after["workingThoughts"]["count"], 0)
        reloaded.events.close()

    def test_multimodal_vectors_use_the_same_automatic_memory_lifecycle(self):
        brain = self.make_brain("multimodal-life")
        result = brain._admit_sensory_embedding(
            torch.linspace(-1.0, 1.0, brain.config.idea_dim).reshape(1, -1),
            kind="image",
            source_name="sample-frame.png",
            fingerprint="fixture-image-sha",
            importance=0.90,
        )
        snapshot = brain.workspace_snapshot()["memory"]
        self.assertEqual(result["memorySettling"]["cycle"], 1)
        self.assertNotIn("stage", result["memorySettling"])
        self.assertGreater(result["memorySettling"]["retentionScore"], 0.0)
        self.assertIn(
            result["assemblyId"],
            {item["assemblyId"] for item in snapshot["activeFocus"]["items"]},
        )
        self.assertEqual(snapshot["workingThoughts"]["count"], 1)
        self.assertFalse(result["memorySettling"]["visiblePonderForced"])
        brain.events.close()

    def test_reading_visits_all_sections_but_weights_and_binds_them_as_a_whole(self):
        brain = self.make_brain("reading-life")
        text = (
            "CHAPTER ONE: SIGNAL ROUTING\n\n"
            + ("The iris relay carries cobalt packets through the north gate. " * 22)
            + "\n\n"
            + ("ok ok ok ok ok ok ok ok. " * 8)
        )
        chunks = brain._experience_chunks(text)
        weights = brain._reading_chunk_importances(chunks)
        self.assertEqual(len(weights), len(chunks))
        self.assertGreater(len(set(round(value, 6) for value in weights)), 1)

        result = brain.ingest(
            text=text,
            name="brain-like-reading.txt",
            policy="encode",
        )
        self.assertEqual(result["coverage"]["processedRecords"], 1)
        self.assertTrue(result["readingIntegration"]["everySectionVisited"])
        self.assertEqual(result["readingIntegration"]["wholeRecordAssemblies"], 1)
        whole = [
            record
            for record in brain.memory.assemblies
            if record.get("kind") == "document"
            and record.get("source") == "reading"
            and record.get("child_assembly_ids")
        ]
        self.assertEqual(len(whole), 1)
        self.assertGreaterEqual(len(whole[0]["child_assembly_ids"]), len(chunks))
        self.assertIn("retention_score", whole[0])
        self.assertNotIn("memory_stage", whole[0])
        self.assertFalse(result["readingIntegration"]["rawSourceTextStored"])
        brain.events.close()

    def test_idle_rest_settles_memory_without_forcing_visible_ponder(self):
        brain = self.make_brain("rest-life")
        brain.learn_experience(
            "An unfinished route still has a missing junction.",
            steps=0,
            importance=0.45,
        )
        before = brain.memory_lifecycle.settled_rest_cycles
        zero_scores = {
            "talk": 0.0,
            "tool": 0.0,
            "imagine": 0.0,
            "agent": 0.0,
            "ponder": 0.0,
            "learn": 0.0,
            "evolve": 0.0,
            "stop": 1.0,
        }
        with mock.patch.object(
            brain,
            "_latent_rehearsal_step",
            return_value={
                "loss": 0.0,
                "reconstructionLoss": 0.0,
                "temporalLoss": 0.0,
                "stabilityLoss": 0.0,
            },
        ), mock.patch.object(
            brain,
            "_select_structured_actions",
            return_value=(zero_scores, []),
        ):
            result = brain.idle_cycle(minimum_idle_seconds=0)
        self.assertTrue(result["ran"])
        self.assertEqual(result["actions"], [])
        self.assertEqual(brain.memory_lifecycle.settled_rest_cycles, before + 1)
        self.assertFalse(
            result["trace"]["memorySettling"]["visiblePonderForced"]
        )
        brain.events.close()


if __name__ == "__main__":
    unittest.main()
