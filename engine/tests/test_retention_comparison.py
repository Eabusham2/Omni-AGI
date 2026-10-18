"""Tiny harness correctness tests, separate from measured comparison results."""
import copy
import unittest

import torch

from omni_core.retention_comparison import (
    CONDITIONS, VARIANTS, Protocol, compare_trials, dataset, run_trial,
)


class RetentionComparisonTests(unittest.TestCase):
    def test_fixed_partitions_are_disjoint_reproducible_and_bounded(self):
        spec = Protocol(seeds=(17,), steps_per_task=2)
        train = dataset(spec, "shortcut-shift", 17, 0, "train")
        repeated = dataset(spec, "shortcut-shift", 17, 0, "train")
        heldout = dataset(spec, "shortcut-shift", 17, 0, "heldout")
        self.assertTrue(torch.equal(train[0], repeated[0]))
        self.assertFalse(torch.equal(train[0], heldout[0][:len(train[0])]))
        self.assertEqual(torch.bincount(train[1]).tolist(), [16] * 4)
        with self.assertRaises(ValueError):
            Protocol(steps_per_task=10000).validate()
        with self.assertRaises(ValueError):
            Protocol(seeds=(17, 17)).validate()

    def test_real_updates_are_reproducible_without_floating_masters(self):
        spec = Protocol(seeds=(17,), steps_per_task=4)
        first = run_trial(spec, "shortcut-shift", 17, "native-rehearsal")
        again = run_trial(spec, "shortcut-shift", 17, "native-rehearsal")
        self.assertEqual(first["finalStateSha256"], again["finalStateSha256"])
        self.assertEqual(first["afterEachTask"], again["afterEachTask"])
        self.assertNotEqual(first["finalStateSha256"], first["initialStateSha256"])
        self.assertGreater(sum(first["changedContactsByTask"]), 0)
        self.assertEqual(first["storage"]["floatingParameterBytes"], 0)
        self.assertEqual(first["storage"]["peakReplayTensorBytes"], 3 * 16 * (12 * 4 + 8))
        self.assertFalse(first["evaluationEnteredReplay"])
        self.assertEqual(first["newAndReplayExposure"], [
            {"newTaskExamples": 64, "replayedExamples": 0, "updates": 4},
            {"newTaskExamples": 48, "replayedExamples": 16, "updates": 4},
            {"newTaskExamples": 48, "replayedExamples": 16, "updates": 4},
        ])

    @staticmethod
    def gate_rows():
        # Deliberately constructed gate unit inputs: never experimental evidence.
        spec = Protocol(seeds=(17,), steps_per_task=1)
        base = {"dataSha256": "data", "sourceSha256": {"model": "same"}, "initialStateSha256": "initial",
            "acquisition": [.9, .9, .9], "retention": [.8, .8], "acquisitionGain": [.5, .5, .5],
            "trainingSeconds": 1.0, "storage": {"peakCheckpointTensorBytes": 80,
                "peakReplayTensorBytes": 2688, "peakWorkerRssBytes": 100000000, "floatingParameterBytes": 0},
            "evaluationEnteredReplay": False, "nativeContract": {"latentMasterLearningState": False}}
        rows = [{**copy.deepcopy(base), "condition": condition, "seed": 17, "variant": variant}
                for condition in CONDITIONS for variant in VARIANTS]
        return spec, rows

    def test_gate_never_authorizes_defaults_and_rejects_any_paired_regression(self):
        spec, rows = self.gate_rows()
        clean = compare_trials(rows, spec)
        self.assertTrue(clean["native-rehearsal"]["componentScreenPassed"])
        self.assertFalse(clean["native-rehearsal"]["defaultChangeAuthorized"])
        for field, changed in (("acquisition", [.9, .8, .9]), ("retention", [.7, .8])):
            with self.subTest(field=field):
                candidate = copy.deepcopy(rows)
                next(row for row in candidate if row["variant"] == "native-rehearsal")[field] = changed
                outcome = compare_trials(candidate, spec)["native-rehearsal"]
                self.assertFalse(outcome["componentScreenPassed"])
                self.assertTrue(any("regression" in reason for reason in outcome["reasons"]))

    def test_gate_rejects_missing_pairs_source_drift_and_resource_or_authority_failures(self):
        spec, rows = self.gate_rows()
        with self.assertRaises(ValueError):
            compare_trials(rows[:-1], spec)
        for field in ("dataSha256", "initialStateSha256", "sourceSha256"):
            with self.subTest(field=field):
                changed = copy.deepcopy(rows); changed[1][field] = "different"
                with self.assertRaises(ValueError):
                    compare_trials(changed, spec)
        candidate = next(row for row in rows if row["variant"] == "native-rehearsal")
        candidate["trainingSeconds"] = 3
        candidate["storage"]["peakReplayTensorBytes"] = 100000
        candidate["evaluationEnteredReplay"] = True
        reasons = compare_trials(rows, spec)["native-rehearsal"]["reasons"]
        self.assertTrue(any("memory budget" in reason for reason in reasons))
        self.assertTrue(any("time budget" in reason for reason in reasons))
        self.assertTrue(any("isolation" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
