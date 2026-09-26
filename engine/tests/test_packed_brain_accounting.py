"""The app must count and hash packed synapses, not only float Parameters."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.liquid import LiquidController
from omni_core.modalities import ModalityHub
from omni_core.model import OmniDecoder, PackedAdaptiveBitLinear, packed_runtime_status
from omni_core.optimizers import PackedOnlyOptimizer, adamw_for_remaining_parameters
from omni_core.spiking import AssociativeSpikingRouter


class PackedBrainAccountingTest(unittest.TestCase):
    def test_learned_float_gain_cannot_pass_native_audit(self) -> None:
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.decoder = nn.Module()
        brain.decoder.register_parameter("gain", nn.Parameter(torch.ones(1)))
        brain.memory_bridge = nn.Identity()
        brain.idea_adapter = nn.Identity()
        brain.router = nn.Identity()
        brain.liquid = nn.Identity()
        brain.modalities = nn.Identity()

        runtime = brain.packed_runtime_audit()
        self.assertFalse(runtime["complete"])
        self.assertEqual(runtime["floatingLearnedParameterBlockers"], ["decoder.gain"])
        with self.assertRaisesRegex(RuntimeError, "decoder.gain"):
            brain.export_packed_ternary()

    def test_dense_embedding_cannot_pass_brain_or_export_audit(self) -> None:
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.decoder = nn.Embedding(4, 3)
        brain.memory_bridge = nn.Identity()
        brain.idea_adapter = nn.Identity()
        brain.router = nn.Identity()
        brain.liquid = nn.Identity()
        brain.modalities = nn.Identity()

        runtime = brain.packed_runtime_audit()
        self.assertFalse(runtime["complete"])
        self.assertEqual(runtime["denseEmbeddingBlockers"], ["decoder.<root>"])
        with self.assertRaisesRegex(RuntimeError, "decoder.<root>"):
            brain.require_complete_packed_runtime()
        with self.assertRaisesRegex(RuntimeError, "decoder.<root>"):
            brain.export_packed_ternary()

    def test_packed_only_objective_does_not_allocate_adam_moments(self) -> None:
        optimizer = adamw_for_remaining_parameters(
            [{"params": []}], lr=0.01, weight_decay=0.0
        )
        self.assertIsInstance(optimizer, PackedOnlyOptimizer)
        self.assertEqual(optimizer.state_dict()["state"], {})

    def test_native_components_have_no_resident_projection_master(self) -> None:
        config = OmniConfig.micro()
        roots = (
            OmniDecoder(config),
            ModalityHub(config),
            LiquidController(config.idea_dim),
            AssociativeSpikingRouter(config.idea_dim, config.router_neurons),
        )
        for root in roots:
            audit = packed_runtime_status(root)
            self.assertTrue(audit["complete"], audit)
            self.assertEqual(audit["residentFloatMasterBlockers"], [])

    def test_packed_change_counts_as_parameter_change(self) -> None:
        config = OmniConfig.micro()
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.decoder = OmniDecoder(config)
        brain.memory_bridge = PackedAdaptiveBitLinear(
            config.vsa_dim, config.idea_dim, bias=True
        )
        brain.idea_adapter = nn.Sequential(
            PackedAdaptiveBitLinear(config.idea_dim, config.idea_dim, bias=True)
        )
        brain.router = AssociativeSpikingRouter(
            config.idea_dim, config.router_neurons
        )
        brain.liquid = LiquidController(config.idea_dim)
        brain.modalities = ModalityHub(config)
        brain.memory = SimpleNamespace(synapses={})

        runtime = brain.packed_runtime_audit()
        self.assertTrue(runtime["complete"], runtime)
        self.assertEqual(runtime["residentFloatMasterBlockers"], [])
        accounting = brain.parameter_accounting()
        self.assertGreater(accounting["packedTernaryParameters"], 0)
        self.assertEqual(
            accounting["totalNeuralParameters"],
            accounting["floatingTrainableParameters"]
            + accounting["packedTernaryParameters"],
        )
        before = brain.parameter_checksum()
        brain.memory_bridge.fill_ternary_(0)
        self.assertNotEqual(before, brain.parameter_checksum())
        packed = tuple(
            tensor for tensor in brain._learned_parameter_tensors(
                brain._trainable_modules()
            ) if tensor.dtype == torch.uint8
        )
        self.assertTrue(packed)


if __name__ == "__main__":
    unittest.main()
