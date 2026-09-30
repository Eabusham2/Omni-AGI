"""Parameter-count arithmetic only; no brain construction or training."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omni_core.brain import AdaptiveBrain


class ParameterInventoryMathTests(unittest.TestCase):
    def test_shared_memory_rows_are_counted_once_without_loading_tensors(self):
        brain = object.__new__(AdaptiveBrain)
        brain.memory = SimpleNamespace(
            neuron_vectors={"a": None, "b": None, "c": None},
            assembly_vectors={"a": None, "b": None},
            synapses={"a>b": None, "b>c": None},
            space=SimpleNamespace(dimensions=64),
        )
        brain._trainable_modules = lambda: ()
        with patch.object(AdaptiveBrain, "_packed_logical_parameter_count", return_value=100):
            result = brain.parameter_accounting()
        self.assertEqual(result["mutableDenseParameters"], 100)
        self.assertEqual(result["substrateVectorParameters"], 192)
        self.assertEqual(result["dynamicSparseSynapses"], 2)
        self.assertEqual(result["totalNeuralParameters"], 294)


if __name__ == "__main__":
    unittest.main()
