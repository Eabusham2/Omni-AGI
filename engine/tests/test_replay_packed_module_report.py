"""Packed replay reporting without constructing or training a brain/model."""
import inspect
from types import SimpleNamespace
import unittest

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.persistence import tensor_checksum


class PackedReplayModuleReport(unittest.TestCase):
    def test_authoritative_buffer_mutation_is_visible_without_floating_parameters(self):
        packed = torch.tensor([0x55, 0x55], dtype=torch.uint8)
        module = SimpleNamespace(parameters=lambda: iter(()),
                                 authoritative_packed_tensors=lambda: (packed,))
        module.named_modules = lambda: iter((("", module),))
        before = tensor_checksum(AdaptiveBrain._learned_parameter_tensors((module,)))
        packed[0] = 0x56
        after = tensor_checksum(AdaptiveBrain._learned_parameter_tensors((module,)))
        self.assertNotEqual(before, after)
        self.assertEqual(list(module.parameters()), [])

    def test_both_production_module_reports_use_authoritative_packed_collector(self):
        source = inspect.getsource(AdaptiveBrain.consolidate_pending_chat_learning)
        self.assertEqual(source.count("tensor_checksum(self._learned_parameter_tensors((module,)))"), 2)
        self.assertNotIn("tensor_checksum(module.parameters())", source)


if __name__ == "__main__":
    unittest.main()
