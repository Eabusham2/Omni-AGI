import math
import sys
import unittest
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.memory_lifecycle import OrganicMemoryLifecycle


class AdaptiveRetentionCandidateTests(unittest.TestCase):
    @staticmethod
    def assess(**signals):
        return OrganicMemoryLifecycle.assess_retention_candidate(**signals)

    def test_weak_one_off_has_fade_pressure_without_a_binary_admission_gate(self):
        decision = self.assess(
            observations=1,
            rehearsals=1,
            novelty=0.30,
            salience=0.20,
            prediction_error=0.10,
            activation=0.40,
            recurrence=0.10,
            interference=0.10,
        )

        self.assertFalse(decision["binaryAdmissionGate"])
        self.assertEqual(
            decision["fastEpisodePolicy"], "unconditional-generic"
        )
        self.assertFalse(decision["manualConsolidationRequired"])
        self.assertNotIn("durableEligible", decision)
        self.assertNotIn("adaptiveThreshold", decision)
        self.assertIn("weak-one-off", decision["reasons"])
        self.assertGreater(
            decision["fadePressure"], decision["retentionScore"]
        )

    def test_second_matching_observation_smoothly_raises_replay_priority(self):
        signals = {
            "rehearsals": 1,
            "novelty": 0.55,
            "salience": 0.55,
            "prediction_error": 0.55,
            "activation": 0.60,
            "recurrence": 0.60,
            "stability": 0.15,
            "related_coactivation": 0.20,
            "interference": 0.10,
        }
        first = self.assess(observations=1, **signals)
        second = self.assess(observations=2, **signals)

        self.assertIn("repeated-or-rehearsed", second["reasons"])
        self.assertGreater(
            second["retentionScore"], first["retentionScore"]
        )
        self.assertGreater(
            second["slowReplayPriority"], first["slowReplayPriority"]
        )
        self.assertLess(second["fadePressure"], first["fadePressure"])
        self.assertFalse(first["binaryAdmissionGate"])
        self.assertFalse(second["binaryAdmissionGate"])

    def test_salient_surprising_one_shot_gets_high_priority_without_special_casing(self):
        weak = self.assess(
            observations=1,
            rehearsals=1,
            novelty=0.20,
            salience=0.20,
            prediction_error=0.10,
            activation=0.40,
            recurrence=0.10,
            interference=0.05,
        )
        salient = self.assess(
            observations=1,
            rehearsals=1,
            novelty=0.90,
            salience=0.95,
            prediction_error=0.90,
            activation=0.70,
            recurrence=0.10,
            stability=0.05,
            related_coactivation=0.10,
            interference=0.05,
        )

        self.assertGreater(
            salient["retentionScore"], weak["retentionScore"]
        )
        self.assertGreater(
            salient["slowReplayPriority"], weak["slowReplayPriority"]
        )
        self.assertNotIn("weak-one-off", salient["reasons"])
        self.assertIn("novelty", salient["reasons"])
        self.assertIn("salience", salient["reasons"])
        self.assertIn("prediction-error", salient["reasons"])

    def test_unrelated_competition_suppresses_retention_and_replay(self):
        common = {
            "observations": 2,
            "rehearsals": 1,
            "novelty": 0.55,
            "salience": 0.55,
            "prediction_error": 0.55,
            "activation": 0.60,
            "recurrence": 0.60,
            "stability": 0.15,
            "related_coactivation": 0.20,
        }
        quiet = self.assess(interference=0.10, **common)
        competed = self.assess(interference=0.90, **common)

        self.assertIn("unrelated-interference", competed["reasons"])
        self.assertLess(
            competed["retentionScore"], quiet["retentionScore"]
        )
        self.assertLess(
            competed["slowReplayPriority"], quiet["slowReplayPriority"]
        )
        self.assertGreater(competed["fadePressure"], quiet["fadePressure"])

    def test_related_coactivation_reinforces_instead_of_interfering(self):
        common = {
            "observations": 2,
            "rehearsals": 1,
            "novelty": 0.45,
            "salience": 0.45,
            "prediction_error": 0.45,
            "activation": 0.55,
            "recurrence": 0.55,
            "reuse": 0.45,
            "stability": 0.25,
            "interference": 0.20,
        }
        isolated = self.assess(related_coactivation=0.0, **common)
        related = self.assess(related_coactivation=0.85, **common)

        self.assertGreater(
            related["retentionScore"], isolated["retentionScore"]
        )
        self.assertGreater(
            related["slowReplayPriority"], isolated["slowReplayPriority"]
        )
        self.assertLess(related["fadePressure"], isolated["fadePressure"])
        self.assertIn("related-coactivation", related["reasons"])

    def test_invalid_scalar_inputs_are_bounded_and_deterministic(self):
        decision = self.assess(
            observations="invalid",
            rehearsals=-4,
            novelty=float("nan"),
            reuse=float("inf"),
            salience=-8,
            prediction_error=9,
            stability=None,
            related_coactivation=-1,
            interference=4,
            recurrence=None,
            activation=2,
        )

        self.assertEqual(decision["evidence"]["observations"], 1)
        self.assertEqual(decision["evidence"]["rehearsals"], 1)
        for name, value in decision["evidence"].items():
            if name in {"observations", "rehearsals"}:
                continue
            if isinstance(value, bool):
                continue
            self.assertTrue(math.isfinite(value), name)
            self.assertGreaterEqual(value, 0.0, name)
            self.assertLessEqual(value, 1.0, name)


if __name__ == "__main__":
    unittest.main()
