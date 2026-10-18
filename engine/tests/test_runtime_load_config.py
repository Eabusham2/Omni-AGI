"""No-model tests of reviewed resource settings before native allocation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from omni_core.config import OmniConfig
from omni_core.runtime_load_config import apply_runtime_load_config
from worker_lifecycle import WorkerLifecycle


class RuntimeLoadConfigTests(unittest.TestCase):
    def test_resource_override_is_visible_before_construction_without_geometry_substitution(self):
        base = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, d_ff=16, max_seq_len=32)
        changed = apply_runtime_load_config(base, {"contextWindowTokens": 64, "storagePoolBytes": 20 * 1024**3,
            "systemRamMode": "manual", "systemRamSharePercent": 40, "memoryOffloadBytes": 1024,
            "dimensions": 4096, "d_model": 4096, "onlineLearning": False, "temperature": .01})
        self.assertEqual(changed.storage_pool_bytes, 20 * 1024**3)
        self.assertEqual(changed.max_seq_len, 64)
        self.assertEqual(changed.system_ram_share_percent, 40)
        self.assertEqual(changed.d_model, 8)
        self.assertEqual(changed.online_learning, base.online_learning)
        self.assertEqual(changed.temperature, base.temperature)
        self.assertEqual(base.storage_pool_bytes, 0)
        self.assertEqual(base.max_seq_len, 32)

    def test_bad_context_ram_and_type_are_rejected_before_any_allocation(self):
        base = OmniConfig.micro(max_seq_len=32)
        for raw in ({"contextWindowTokens": 8}, {"storagePoolBytes": True}, {"systemRamMode": "manual", "systemRamSharePercent": 25}, {"extendedWorkingMemory": 1}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                apply_runtime_load_config(base, raw, minimum_context_tokens=16)

    def test_pending_ancestral_turn_keeps_source_policy_but_accepts_resource_review(self):
        brain = SimpleNamespace(current_context={"tokenCount": 9}, config=SimpleNamespace(max_seq_len=64))
        status = {"minimumContextTokens": 24, "rootMutationHeld": True}
        with patch("omni_core.connected_execution.connected_execution_status", return_value=status):
            reviewed = WorkerLifecycle().reviewed_execution_runtime_config(brain,
                {"contextWindowTokens": 32, "storagePoolBytes": 20 * 1024**3, "onlineLearning": False,
                    "temperature": .01, "bestWorkEnabled": True}, store=SimpleNamespace(current=lambda: None))
            self.assertEqual(reviewed, {"contextWindowTokens": 32, "storagePoolBytes": 20 * 1024**3})
            with self.assertRaisesRegex(ValueError, "frontier"):
                WorkerLifecycle().reviewed_execution_runtime_config(brain, {"contextWindowTokens": 16},
                    store=SimpleNamespace(current=lambda: None))


if __name__ == "__main__": unittest.main()
