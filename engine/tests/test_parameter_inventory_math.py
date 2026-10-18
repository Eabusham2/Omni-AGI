"""Parameter-count arithmetic only; no brain construction or training."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.native_architecture import native_core_inventory, packed_learning_scratch_bytes


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

    def test_weighted_owner_constructor_bytes_match_the_exact_tiny_native_inventory(self):
        import torch
        from torch import nn
        from omni_core.model import OmniDecoder, PackedAdaptiveBitLinear
        from omni_core.modalities import ModalityHub
        from omni_core.liquid import LiquidController
        from omni_core.spiking import AssociativeSpikingRouter
        config = OmniConfig(d_model=32, idea_dim=32, n_layers=1, n_heads=4, d_ff=97,
            vsa_dim=128, router_neurons=24, modality_channels=8, image_size=8,
            audio_samples=64, video_frames=2, working_memory_slots=32)
        roots = (OmniDecoder(config), PackedAdaptiveBitLinear(128, 32, bias=True),
            nn.Sequential(PackedAdaptiveBitLinear(32, 64, bias=True), nn.SiLU(), PackedAdaptiveBitLinear(64, 32, bias=True)),
            AssociativeSpikingRouter(32, 24), LiquidController(32), ModalityHub(config))
        shape = {"dModel": 32, "layers": 1, "feedForward": 97, "nHeads": 4,
            "vsaDimensions": 128, "routerNeurons": 24, "modalityChannels": 8,
            "imageSize": 8, "audioSamples": 64, "videoFrames": 2,
            "workingMemoryItems": 32, "workspaceLatents": 8, "vocabSize": 261, "liquidMode": "cfc"}
        expected = native_core_inventory(shape, "block-sparse-v1")
        owners = [module for root in roots for module in root.modules() if hasattr(module, "_packed_forward_weight")]
        signs = sum(value.numel() * value.element_size() for owner in owners
            for name in ("_packed_forward_weight", "_packed_forward_bias")
            for value in (getattr(owner, name, None),) if isinstance(value, torch.Tensor))
        state = sum(value.numel() * value.element_size() for owner in owners
            for name in ("_connection_state", "_bias_connection_state")
            for value in (getattr(owner, name, None),) if isinstance(value, torch.Tensor))
        maximum = max(sum(value.numel() * value.element_size() for value in owner._buffers.values()
            if isinstance(value, torch.Tensor)) for owner in owners)
        self.assertEqual(len(owners), expected["packedOwners"])
        self.assertEqual(signs, expected["packedWeightBytes"])
        self.assertEqual(state, expected["connectionStateBytes"])
        self.assertEqual(signs + state, expected["packedLearnedStateBytes"])
        self.assertEqual(maximum, expected["maximumProjectionBytes"])
        scratch = max(packed_learning_scratch_bytes(int(value.shape[0]), int(value.shape[1]) * 4)
            for owner in owners for name in ("_packed_forward_weight", "_packed_forward_bias")
            for value in (getattr(owner, name, None),) if isinstance(value, torch.Tensor))
        self.assertEqual(scratch, expected["maximumUpdateScratchBytes"])
        self.assertEqual(sum(owner.parallel_connections._next_page.numel() for owner in owners), expected["parallelTopologyControlBytes"])
        self.assertLess(signs + state, 1024 * 1024)

    def test_live_training_plan_uses_weighted_scratch_without_dense_optimizer_bytes(self):
        from omni_core.model import PackedAdaptiveBitLinear
        brain = object.__new__(AdaptiveBrain)
        owner = PackedAdaptiveBitLinear(17, 3, bias=True)
        brain.config = OmniConfig(d_model=32, idea_dim=32, n_heads=4)
        brain._trainable_modules = lambda: (owner,)
        brain.decoder = SimpleNamespace()
        brain._runtime_training_max_seq_len = brain.config.max_seq_len
        brain._runtime_train_batch_size = brain.config.train_batch_size
        brain._optimizer = SimpleNamespace(state={})
        brain._optimizer_offloaded = False
        brain.resource_policy = SimpleNamespace(training_plan=lambda **value: value)
        result = brain._training_resource_plan()
        self.assertEqual(result["packed_update_scratch_bytes"], packed_learning_scratch_bytes(3, 20))
        self.assertEqual(result["trainable_parameter_bytes"], 0)
        skinny = PackedAdaptiveBitLinear(1, 513, bias=True)
        brain._trainable_modules = lambda: (skinny,)
        self.assertEqual(brain._training_resource_plan()["packed_update_scratch_bytes"], packed_learning_scratch_bytes(1, 516))

    def test_growth_reserve_includes_strength_and_only_allocated_router_blocks(self):
        from omni_core.evolution import _native_growth_reserve
        config = SimpleNamespace(d_model=32, d_ff=97, n_layers=1, router_neurons=24)
        experts = _native_growth_reserve(config, {"mutation": "grow-experts", "addExperts": 2})
        self.assertEqual(experts["connectionStateBytes"], 2 * (2 * 32 + 3 * 32 * 48))
        depth = _native_growth_reserve(config, {"mutation": "grow-depth", "addLayers": 2})
        self.assertEqual(depth["connectionStateBytes"], 2 * (4 * 32 ** 2 + 3 * 32 * 97 + 2 * 32))
        empty = SimpleNamespace(synapses=SimpleNamespace(blocks={}), region_ends=[24])
        allocated = SimpleNamespace(synapses=SimpleNamespace(blocks={"r0_c0": None, "r0_c0_p1": None}), region_ends=[24])
        mutation = {"mutation": "grow-router", "addNeurons": 24}
        blank = _native_growth_reserve(config, mutation, empty)
        used = _native_growth_reserve(config, mutation, allocated)
        self.assertEqual(blank["connectionStateBytes"], 2 * 32 * 48 + 48 + 32)
        self.assertEqual(used["nonweightTransientBytes"] - blank["nonweightTransientBytes"], 2 * 2 * (48 * 12 + 11 * 48 ** 2 + 8192))
        self.assertGreater(experts["estimatedBytes"], 3 * experts["packedLearnedStateBytes"])


if __name__ == "__main__":
    unittest.main()
