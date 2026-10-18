import unittest
from types import SimpleNamespace

from torch import nn

from omni_core.brain import AdaptiveBrain
from omni_core.model import PackedAdaptiveBitLinear


class WeightedBrainAccountingTest(unittest.TestCase):
    def test_strength_bytes_and_shared_contacts_are_not_counted_as_extra_synapses(self):
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.decoder = PackedAdaptiveBitLinear(3, 2, bias=False)
        brain.memory_bridge = brain.decoder
        brain.idea_adapter = nn.Identity()
        brain.router = nn.Identity()
        brain.liquid = nn.Identity()
        brain.modalities = nn.Identity()
        brain.memory = SimpleNamespace(synapses={}, neuron_vectors={}, space=SimpleNamespace(dimensions=4))
        initial = brain.parameter_accounting()
        self.assertEqual(initial["totalNeuralParameters"], 6)
        self.assertEqual(initial["connectionStateBytes"], 6)
        self.assertEqual(initial["packedLearnedStateBytes"], 8)
        self.assertEqual(initial["packedStructuralStateBytes"], 8)
        self.assertEqual(initial["packedAllocatedStateBytes"], 16)
        brain.decoder.add_parallel_connection_(0, 1, sign=1, strength=3)
        grown = brain.parameter_accounting()
        self.assertEqual(grown["totalNeuralParameters"], 7)
        self.assertEqual(grown["nativeRepeatedConnections"], 1)
        self.assertEqual(grown["connectionStateBytes"], 1030)
        self.assertEqual(grown["packedLearnedStateBytes"], 1288)
        self.assertEqual(grown["packedStructuralStateBytes"], 10248)
        self.assertEqual(grown["packedAllocatedStateBytes"], 11536)
        self.assertEqual(grown["mutableDenseParameters"] - grown["nativeRepeatedConnections"], 6)


if __name__ == "__main__":
    unittest.main()
