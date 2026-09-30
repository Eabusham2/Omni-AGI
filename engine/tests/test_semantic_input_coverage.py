"""Pure indexing coverage; no brain/model construction or learned response."""
import unittest

from omni_core.vsa import NeuralSubstrate


class SemanticInputCoverage(unittest.TestCase):
    def test_single_letters_scripts_and_symbols_are_real_fast_learning_cues(self):
        labels = NeuralSubstrate.extract_concepts("Q = 我 🙂 ++ __init__ 中文")
        for value in ("q", "我", "🙂", "++", "__init__", "中文"):
            self.assertIn(value, labels)
        self.assertIn("q::=", labels)

    def test_ordered_cue_keeps_repeated_short_elements(self):
        self.assertEqual(NeuralSubstrate.extract_ordered_atomic_concepts("A A 🙂"), ["a", "a", "🙂"])


if __name__ == "__main__": unittest.main()
