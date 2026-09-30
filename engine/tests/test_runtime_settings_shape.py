"""Production-method stubs only: never construct or train a native brain/model."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.native_architecture import native_architecture_sha256, native_core_inventory
from omni_core.tokenizer import ByteTokenizer


def fixture():
    config = OmniConfig.micro(max_seq_len=32, working_memory_slots=128)
    shape = {"dModel": config.d_model, "layers": config.n_layers, "feedForward": config.d_ff,
        "nHeads": config.n_heads, "vsaDimensions": config.vsa_dim, "routerNeurons": config.router_neurons,
        "modalityChannels": config.modality_channels, "imageSize": config.image_size, "audioSamples": config.audio_samples,
        "videoFrames": config.video_frames, "workingMemoryItems": 128, "workspaceLatents": 32,
        "vocabSize": config.vocab_size, "liquidMode": config.liquid_mode}
    descriptor = {"format": "omni-main-selected-native-architecture", "formatVersion": 1,
        "architecture": "OmniCortex", "externalPretrainedWeights": False, "hardwareTier": "micro",
        "shape": shape, "inventory": native_core_inventory(shape), "sizing": {"policy": "fixture"},
        "qualityEvidence": "unmeasured-native-quality-deferred"}
    descriptor["sha256"] = native_architecture_sha256(descriptor)
    config.native_architecture = descriptor
    brain = AdaptiveBrain.__new__(AdaptiveBrain)
    brain.config, brain.engine_path = config, Path("/fixture")
    rotary = SimpleNamespace(configure_max_seq_len=Mock())
    brain.decoder = SimpleNamespace(blocks=[SimpleNamespace(attention=SimpleNamespace(rotary=rotary))])
    brain._runtime_training_max_seq_len = 2
    brain.recent_token_context = []
    brain.tokenizer = ByteTokenizer()
    brain.counters = {"context_token_evictions": 0}
    brain._recent_token_activity = lambda: SimpleNamespace(fit_capacity=lambda tokens, limit, **kwargs: tokens[:limit])
    brain._sync_recent_dialogue_counts = Mock()
    brain.resource_policy = object()
    brain.state_store = SimpleNamespace(policy=brain.resource_policy)
    brain.replay = SimpleNamespace(policy=brain.resource_policy)
    brain.paged_working_memory = SimpleNamespace(policy=brain.resource_policy)
    brain._configure_working_attention_resources = Mock()
    brain.core_pager = SimpleNamespace(refresh_budget=Mock())
    brain._optimizer = object()
    brain._replace_optimizer = Mock(side_effect=AssertionError("runtime resources must not rebuild the optimizer"))
    brain.population_controls_from_config = Mock()
    brain.save = Mock()
    brain.events = SimpleNamespace(append=Mock())
    brain.runtime_card = lambda: {"contextTokens": brain.config.max_seq_len}
    brain.summary = lambda: {"config": brain.config.to_dict()}
    return brain, descriptor, rotary


class RuntimeSettingsShapeFixtures(unittest.TestCase):
    def test_stale_auto_extended_item_derivation_does_not_resize_learned_tables(self):
        brain, descriptor, rotary = fixture()
        before_optimizer, before_descriptor = brain._optimizer, brain.config.native_architecture
        with patch("omni_core.brain.ResourcePolicy", side_effect=lambda *args, **kwargs: SimpleNamespace(**kwargs)):
            result = brain.update_config({"workingMemorySlots": 65_536, "contextWindowTokens": 64,
                "workingMemoryMode": "extended", "extendedWorkingMemory": True,
                "systemRamMode": "manual", "systemRamSharePercent": 54,
                "storagePoolBytes": 1_073_741_824, "contextOffloadBudgetBytes": 4096})
        self.assertEqual(brain.config.working_memory_slots, 128)
        self.assertIs(brain.config.native_architecture, before_descriptor)
        self.assertEqual(brain.config.native_architecture["sha256"], descriptor["sha256"])
        self.assertEqual(brain.config.max_seq_len, 64)
        self.assertTrue(brain.config.extended_working_memory)
        self.assertEqual(brain._runtime_training_max_seq_len, 2)
        self.assertIs(brain._optimizer, before_optimizer)
        self.assertEqual(result["ignoredLearnedShapeFields"], ["workingMemorySlots"])
        self.assertNotIn("working_memory_slots", result["changed"])
        rotary.configure_max_seq_len.assert_called_once_with(64)
        brain._configure_working_attention_resources.assert_called_once()
        brain.core_pager.refresh_budget.assert_called_once_with(force=True)
        for owner in (brain.state_store, brain.replay, brain.paged_working_memory):
            self.assertIs(owner.policy, brain.resource_policy)
        brain.save.assert_called_once()

    def test_pool_only_update_rebinds_policy_and_context_allowance(self):
        brain, _, _ = fixture()
        original_policy = brain.resource_policy
        with patch("omni_core.brain.ResourcePolicy", side_effect=lambda *args, **kwargs: SimpleNamespace(**kwargs)):
            brain.update_config({"storagePoolBytes": 987_654, "contextOffloadBudgetBytes": 65_536})
        self.assertIsNot(original_policy, brain.resource_policy)
        self.assertEqual(brain.config.storage_pool_bytes, 987_654)
        self.assertEqual(brain.config.working_attention_scratch_budget_bytes, 65_536)
        brain._configure_working_attention_resources.assert_called_once()

    def test_liquid_shape_change_and_bad_advisory_slots_reject_before_any_mutation(self):
        for payload in ({"liquidMode": "ltc", "contextWindowTokens": 64}, {"workingMemorySlots": 0}, {"workingMemorySlots": True}, {"workingMemorySlots": 1.5}):
            brain, _, rotary = fixture()
            with self.assertRaises(ValueError):
                brain.update_config(payload)
            self.assertEqual(brain.config.max_seq_len, 32)
            self.assertEqual(brain.config.working_memory_slots, 128)
            rotary.configure_max_seq_len.assert_not_called()
            brain.save.assert_not_called()

    def test_name_only_update_keeps_policy_and_live_attention_budgets(self):
        brain, _, _ = fixture()
        old_policy = brain.resource_policy
        brain.update_config({"name": "Renamed"})
        self.assertIs(old_policy, brain.resource_policy)
        brain._configure_working_attention_resources.assert_not_called()
        brain.core_pager.refresh_budget.assert_not_called()

    def test_existing_activity_pager_rebinds_new_policy_not_an_old_cached_ram_cap(self):
        brain, _, _ = fixture()
        brain.device = torch.device("cpu")
        brain.resource_policy = SimpleNamespace(status=lambda: {"systemRamBudgetBytes": 512 * 1024**2,
            "availableSafeRamBytes": 256 * 1024**2, "processMemoryBytes": 32 * 1024**2,
            "diskFreeBytes": 2 * 1024**3, "diskReserveBytes": 512 * 1024**2})
        brain.core_pager.status = lambda: {"cpuHeapBytes": 0}
        pager = SimpleNamespace(resource_policy=object(), _policy_snapshot={"systemRamBudgetBytes": 1},
            _policy_at=99.0, status=lambda: {"residentBytes": 0})
        brain.working_attention_pager = pager
        brain.decoder.configure_working_attention = Mock()
        AdaptiveBrain._configure_working_attention_resources(brain)
        self.assertIs(pager.resource_policy, brain.resource_policy)
        self.assertIsNone(pager._policy_snapshot)
        self.assertEqual(pager._policy_at, 0.0)
        self.assertGreater(pager.resident_budget_bytes, 0)
        brain.decoder.configure_working_attention.assert_called_once_with(pager)


if __name__ == "__main__":
    unittest.main()
