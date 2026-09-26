import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.vsa import NeuralSubstrate


class AdaptiveSubstrateVectorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def _semantic_id(label: str) -> str:
        return hashlib.sha256(("semantic:%s" % label).encode("utf-8")).hexdigest()[:24]

    def test_plain_experience_adapts_neuron_and_assembly_without_lookup(self):
        memory = NeuralSubstrate(dimensions=64, seed=117)
        sentence = "A cobalt tram wandered beside the violet harbor."
        learned = memory.learn(sentence, retain_source_text=False)
        semantic_id = self._semantic_id("cobalt")
        assembly_id = str(learned["assembly_id"])
        first_neuron = memory.neuron_vectors[semantic_id].clone()
        first_assembly = memory.assembly_vectors[assembly_id].clone()
        first_cue = memory.vector_for_text(sentence).clone()

        memory.learn(sentence, retain_source_text=False)

        self.assertFalse(torch.equal(first_neuron, memory.neuron_vectors[semantic_id]))
        self.assertFalse(torch.equal(first_assembly, memory.assembly_vectors[assembly_id]))
        self.assertGreater(
            float((first_cue - memory.vector_for_text(sentence)).norm()), 0.0
        )
        self.assertTrue(torch.equal(
            memory.neuron_vectors[assembly_id], memory.assembly_vectors[assembly_id]
        ))
        self.assertTrue(all(
            synapse["effective_weight"] in {-1, 0, 1}
            for synapse in memory.synapses.values()
        ))
        self.assertFalse(any("source_text" in item for item in memory.assemblies))

    def test_statistical_experience_adapts_and_survives_sharded_reload(self):
        memory = NeuralSubstrate(dimensions=64, seed=119)
        sentence = "Cobalt trams cross violet harbors each morning."
        first = memory.learn_statistical(sentence)
        semantic_id = self._semantic_id("cobalt")
        first_neuron = memory.neuron_vectors[semantic_id].clone()
        first_field = memory.assembly_vectors[first["assembly_id"]].clone()

        memory.learn_statistical(sentence)
        self.assertFalse(torch.equal(first_neuron, memory.neuron_vectors[semantic_id]))
        self.assertFalse(torch.equal(first_field, memory.assembly_vectors[first["assembly_id"]]))

        with tempfile.TemporaryDirectory(prefix="omni-adaptive-vectors-") as folder:
            store = Path(folder) / "substrate"
            memory.save_sharded(store, records_per_shard=3)
            restored = NeuralSubstrate.load_sharded(
                store, memory.metadata(include_records=False)
            )
            self.assertTrue(torch.equal(
                memory.neuron_vectors[semantic_id],
                restored.neuron_vectors[semantic_id],
            ))
            self.assertTrue(torch.equal(
                memory.assembly_vectors[first["assembly_id"]],
                restored.assembly_vectors[first["assembly_id"]],
            ))
            self.assertTrue(torch.equal(
                restored.neuron_vectors[first["assembly_id"]],
                restored.assembly_vectors[first["assembly_id"]],
            ))


if __name__ == "__main__":
    unittest.main()
