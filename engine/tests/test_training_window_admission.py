"""Neural-free measured-budget tests for exhaustive small-window admission."""

import sys
import tempfile
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.offload import GIB, MIB, ResourcePolicy, ResourceReading
from omni_core.tokenizer import ByteTokenizer


class TrainingWindowAdmissionFixtures(unittest.TestCase):
    def plan(self, partition, max_window=4096, activation=MIB):
        with tempfile.TemporaryDirectory() as directory:
            reading = ResourceReading(
                total_memory_bytes=8 * GIB,
                available_memory_bytes=7 * GIB,
                process_memory_bytes=128 * MIB,
                disk_total_bytes=200 * GIB,
                disk_free_bytes=180 * GIB,
            )
            policy = ResourcePolicy(Path(directory), reading_provider=lambda: reading)
            return policy.training_plan(
                max_window_tokens=max_window,
                requested_batch_size=1,
                requested_gradient_accumulation=1,
                trainable_parameter_bytes=0,
                activation_bytes_per_token=activation,
                resource_mode="manual",
                manual_ram_budget_bytes=partition,
            )

    def test_a_real_two_token_window_fits_without_forcing_sixty_four(self):
        plan = self.plan(4 * MIB)
        self.assertFalse(plan["pauseBeforeStep"])
        self.assertEqual(plan["windowTokens"], 3)
        self.assertEqual(plan["admittedWindowTokens"], 3)
        self.assertEqual(plan["minimumWindowTokens"], 2)
        self.assertEqual(plan["windowOverlapTokens"], 1)
        self.assertLess(plan["memory"]["allocatorMarginBytes"], 128 * MIB)
        self.assertFalse(plan["coverageConfirmed"])

    def test_exactly_two_tokens_are_supported_and_one_token_pauses(self):
        admitted = self.plan(3 * MIB, max_window=2)
        self.assertFalse(admitted["pauseBeforeStep"])
        self.assertEqual(admitted["admittedWindowTokens"], 2)
        paused = self.plan(2 * MIB)
        self.assertTrue(paused["pauseBeforeStep"])
        self.assertEqual(paused["admittedWindowTokens"], 0)
        self.assertEqual(paused["requestedWindowTokens"], 4096)
        self.assertEqual(paused["maximumAffordableWindowTokens"], 1)

    def test_invalid_label_window_is_rejected_not_silently_enlarged(self):
        with self.assertRaisesRegex(ValueError, "two tokens"):
            self.plan(64 * MIB, max_window=1)

    def test_larger_budget_keeps_preferred_floor_metadata_without_imposing_it(self):
        plan = self.plan(96 * MIB)
        self.assertFalse(plan["pauseBeforeStep"])
        self.assertEqual(plan["preferredWindowTokens"], 64)
        self.assertEqual(plan["windowTokens"], 72)
        self.assertTrue(plan["allSourceBytesVisited"])
        self.assertTrue(plan["labelTargetsCoveredOnce"])

    def test_two_token_unicode_windows_cover_each_target_once(self):
        tokenizer = ByteTokenizer()
        text = "one 🌊 café\nlast"
        wanted = tokenizer.encode(text, add_bos=True, add_eos=True)
        for maximum in (2, 3, 8, 64):
            windows = list(tokenizer.windows(text, maximum))
            actual = windows[0][:1] + [token for window in windows for token in window[1:]]
            self.assertEqual(actual, wanted)
            self.assertTrue(all(2 <= len(window) <= maximum for window in windows))
            for previous, current in zip(windows, windows[1:]):
                self.assertEqual(previous[-1], current[0])
            self.assertEqual(sum(len(window) - 1 for window in windows), len(wanted) - 1)

    def test_window_boundaries_and_optional_specials_are_not_repeated(self):
        tokenizer = ByteTokenizer()
        for text in ("", "x", "xy", "some UTF8 é bytes"):
            for bos, eos in ((True, True), (True, False), (False, True), (False, False)):
                wanted = tokenizer.encode(text, add_bos=bos, add_eos=eos)
                windows = list(tokenizer.windows(text, 2, bos, eos))
                actual = windows[0][:1] + [token for window in windows for token in window[1:]]
                self.assertEqual(actual, wanted)
        with self.assertRaisesRegex(ValueError, "two tokens"):
            list(tokenizer.windows("retained", 1))


if __name__ == "__main__":
    unittest.main()
