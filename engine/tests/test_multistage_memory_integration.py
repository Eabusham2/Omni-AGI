"""Native, no-Build checks for experience-driven substrate integration.

The retired cue-to-answer sequence fixture is intentionally absent. A fluent
generated answer still requires a separately trained brain acceptance run.
"""

import unittest

import torch

from omni_core.vsa import NeuralSubstrate, SubstrateResourcePause


class NativeMemoryIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_question_story_and_slang_all_change_the_same_substrate(self):
        memory = NeuralSubstrate(dimensions=64, seed=73)
        experiences = (
            "Where did that cobalt bus go?",
            "The cobalt bus rolled behind the old rec center.",
            "ngl that cobalt bus dipped behind the rec center fr.",
        )
        revisions = []
        for text in experiences:
            before = memory.state_revision
            report = memory.learn(text, retain_source_text=False)
            self.assertIn(str(report["assembly_id"]), memory.assembly_vectors)
            self.assertGreater(memory.state_revision, before)
            revisions.append(memory.state_revision)

        cobalt = next(
            neuron for neuron in memory.neurons.values()
            if neuron.get("label") == "cobalt"
        )
        self.assertGreaterEqual(int(cobalt["exposures"]), 3)
        self.assertEqual(revisions, sorted(revisions))
        self.assertFalse(any("source_text" in row for row in memory.assemblies))
        self.assertTrue(all(
            row["effective_weight"] in {-1, 0, 1}
            for row in memory.synapses.values()
        ))

    def test_related_reuse_adapts_vectors_and_recurrent_recall(self):
        memory = NeuralSubstrate(dimensions=64, seed=79)
        text = "A copper tram crossed the violet harbor beside three lanterns."
        first = memory.learn(text, retain_source_text=False)
        assembly_id = str(first["assembly_id"])
        before_vector = memory.assembly_vectors[assembly_id].clone()
        before_uses = sum(int(row["uses"]) for row in memory.synapses.values())

        memory.learn(text, retain_source_text=False)
        self.assertFalse(torch.equal(before_vector, memory.assembly_vectors[assembly_id]))
        self.assertGreater(
            sum(int(row["uses"]) for row in memory.synapses.values()),
            before_uses,
        )
        cue = memory.vector_for_text("copper tram near the violet harbor")
        signal, recalled = memory.recall_vector(cue, workspace_slots=16)
        self.assertEqual(tuple(signal.shape), tuple(cue.shape))
        self.assertTrue(recalled)
        self.assertFalse(any("source_text" in row for row in memory.assemblies))

    def test_resource_pause_prevents_partial_first_experience(self):
        memory = NeuralSubstrate(
            dimensions=64, seed=83, growth_guard=lambda _bytes: False
        )
        with self.assertRaises(SubstrateResourcePause):
            memory.learn("A first whole experience should be atomic.")
        self.assertEqual(memory.neurons, {})
        self.assertEqual(memory.assemblies, [])
        self.assertEqual(memory.synapses, {})


if __name__ == "__main__":
    unittest.main()
