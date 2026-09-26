import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.offload import ResourceReading


class ResourceConfigWiringTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(1)

    def test_external_ram_mode_has_unambiguous_auto_and_manual_encoding(self):
        automatic = OmniConfig.from_external(
            {
                "hardwareTier": "micro",
                "systemRamMode": "auto",
                "systemRamSharePercent": 92,
                "storageBytesPerSecond": 456_000_000.6,
            }
        )
        self.assertEqual(automatic.system_ram_share_percent, 0.0)
        self.assertEqual(automatic.storage_bytes_per_second, 456_000_001)

        for percent in (30, 61, 100):
            with self.subTest(percent=percent):
                manual = OmniConfig.from_external(
                    {
                        "hardwareTier": "micro",
                        "systemRamMode": "manual",
                        "systemRamSharePercent": percent,
                    }
                )
                self.assertEqual(manual.system_ram_share_percent, float(percent))

        for percent in (0, 29, 101):
            with self.subTest(rejected=percent):
                with self.assertRaisesRegex(ValueError, r"\[30, 100\]"):
                    OmniConfig.from_external(
                        {
                            "systemRamMode": "manual",
                            "systemRamSharePercent": percent,
                        }
                    )

    def test_gpu_auto_policy_uses_tier_base_without_persisting_a_manual_share(self):
        with tempfile.TemporaryDirectory(prefix="omni-gpu-auto-config-") as folder:
            root = Path(folder)
            config = OmniConfig.from_external(
                {
                    "name": "GPU Auto",
                    "hardwareTier": "gpu",
                    "systemRamMode": "auto",
                    "systemRamSharePercent": 92,
                    "storageBytesPerSecond": 734_003_200.6,
                }
            )
            # This is a low-level resource-policy persistence fixture, not a
            # public Build. Public ground-up persistence must run the local
            # curriculum and is covered by test_ground_up_build_gate.py.
            brain = AdaptiveBrain(
                "gpu-auto-brain",
                root,
                config,
            )
            brain.resource_policy.reading_provider = lambda: ResourceReading(
                total_memory_bytes=16 * 1024**3,
                available_memory_bytes=13 * 1024**3,
                process_memory_bytes=256 * 1024**2,
                disk_total_bytes=500 * 1024**3,
                disk_free_bytes=300 * 1024**3,
            )

            status = brain.resource_policy.status()

            self.assertEqual(brain.config.system_ram_share_percent, 0.0)
            self.assertEqual(brain.config.storage_bytes_per_second, 734_003_201)
            self.assertEqual(brain.resource_policy.hardware_tier, "gpu")
            self.assertEqual(status["systemRamMode"], "auto")
            self.assertEqual(status["hardwareTier"], "gpu")
            self.assertEqual(status["systemRamSharePercent"], 75.0)
            brain.save()
            metadata = json.loads(
                (root / "engine" / "brain.json").read_text("utf-8")
            )
            self.assertEqual(metadata["config"]["system_ram_share_percent"], 0.0)
            self.assertEqual(
                metadata["config"]["storage_bytes_per_second"], 734_003_201
            )
            brain.events.close()

    def test_auto_training_uses_safe_divisor_four_without_growing_logical_batch(self):
        with tempfile.TemporaryDirectory(prefix="omni-auto-batch-") as folder:
            brain = AdaptiveBrain(
                "auto-batch-brain",
                Path(folder),
                OmniConfig.micro(
                    max_seq_len=64,
                    train_batch_size=2,
                    gradient_accumulation=8,
                    training_resource_mode="auto",
                ),
            )
            brain.resource_policy.reading_provider = lambda: ResourceReading(
                total_memory_bytes=64 * 1024**3,
                available_memory_bytes=56 * 1024**3,
                process_memory_bytes=256 * 1024**2,
                disk_total_bytes=500 * 1024**3,
                disk_free_bytes=300 * 1024**3,
                accelerator_total_memory_bytes=16 * 1024**3,
                accelerator_free_memory_bytes=15 * 1024**3,
            )

            automatic = brain._training_resource_plan()
            self.assertTrue(automatic["autoPhysicalBatchDivisorPolicy"])
            self.assertEqual(automatic["autoPhysicalBatchCandidate"], 4)
            self.assertEqual(automatic["physicalBatchRecords"], 4)
            self.assertEqual(automatic["gradientAccumulation"], 4)
            self.assertEqual(automatic["effectiveBatchTarget"], 16)
            self.assertEqual(
                automatic["physicalBatchRecords"]
                * automatic["gradientAccumulation"],
                automatic["effectiveBatchTarget"],
            )

            # A prior allocator downgrade is an operational hint, not a
            # permanent cap on a later transaction after headroom returns.
            brain._runtime_train_batch_size = 1
            recovered_headroom = brain._training_resource_plan()
            self.assertEqual(recovered_headroom["physicalBatchRecords"], 4)
            self.assertEqual(recovered_headroom["gradientAccumulation"], 4)

            # Auto also chooses the largest non-power-of-two divisor rather
            # than expanding a six-record logical update to eight records.
            brain.config.gradient_accumulation = 3
            composite_target = brain._training_resource_plan()
            self.assertEqual(composite_target["autoPhysicalBatchCandidate"], 4)
            self.assertEqual(composite_target["physicalBatchRecords"], 3)
            self.assertEqual(composite_target["gradientAccumulation"], 2)
            self.assertEqual(composite_target["effectiveBatchTarget"], 6)

            # Manual mode retains the explicitly configured physical shape.
            brain.config.training_resource_mode = "manual"
            brain.config.gradient_accumulation = 8
            brain._runtime_train_batch_size = 2
            manual = brain._training_resource_plan()
            self.assertFalse(manual["autoPhysicalBatchDivisorPolicy"])
            self.assertIsNone(manual["autoPhysicalBatchCandidate"])
            self.assertEqual(manual["physicalBatchRecords"], 2)
            self.assertEqual(manual["gradientAccumulation"], 8)
            self.assertEqual(manual["effectiveBatchTarget"], 16)
            brain.events.close()

    def test_resource_update_rebuilds_policy_without_changing_neural_state(self):
        with tempfile.TemporaryDirectory(prefix="omni-resource-config-") as folder:
            root = Path(folder)
            brain = AdaptiveBrain.create(
                "resource-config-brain",
                root,
                OmniConfig.micro(max_seq_len=32),
                initialize_ground_up=True,
            )
            before_parameters = brain.parameter_checksum()
            before_optimizer = brain._optimizer
            before_policy = brain.resource_policy

            result = brain.update_config(
                {
                    "systemRamMode": "manual",
                    "systemRamSharePercent": 44,
                    "storageBytesPerSecond": 321_000_000,
                    "memoryOffloadBytes": 987_654,
                    "memoryResidentItems": 64,
                    "contextWindowTokens": 48,
                }
            )

            self.assertEqual(brain.parameter_checksum(), before_parameters)
            self.assertIs(brain._optimizer, before_optimizer)
            self.assertIsNot(brain.resource_policy, before_policy)
            self.assertIs(brain.state_store.policy, brain.resource_policy)
            self.assertIs(brain.replay.policy, brain.resource_policy)
            self.assertIs(brain.paged_working_memory.policy, brain.resource_policy)
            self.assertEqual(brain.config.system_ram_share_percent, 44.0)
            self.assertEqual(brain.config.storage_bytes_per_second, 321_000_000)
            self.assertEqual(brain.config.max_seq_len, 48)
            self.assertEqual(
                brain.decoder.blocks[0].attention.rotary.max_seq_len, 48
            )
            self.assertEqual(
                brain.decoder.blocks[0].attention.rotary.cached_seq_len, 0
            )
            self.assertIn("max_seq_len", result["changed"])
            self.assertIn("system_ram_share_percent", result["changed"])
            self.assertIn("storage_bytes_per_second", result["changed"])

            metadata = json.loads(
                (root / "engine" / "brain.json").read_text("utf-8")
            )
            self.assertEqual(
                metadata["config"]["system_ram_share_percent"], 44.0
            )
            self.assertEqual(
                metadata["config"]["storage_bytes_per_second"], 321_000_000
            )
            brain.events.close()

            restarted = AdaptiveBrain.load(root, "resource-config-brain")
            self.assertEqual(restarted.config.system_ram_share_percent, 44.0)
            self.assertEqual(
                restarted.config.storage_bytes_per_second, 321_000_000
            )
            self.assertEqual(restarted.config.max_seq_len, 48)

            restarted.update_config(
                {
                    "systemRamMode": "auto",
                    # Full desktop payloads may contain a stale slider value.
                    "systemRamSharePercent": 88,
                }
            )
            self.assertEqual(restarted.config.system_ram_share_percent, 0.0)
            restarted.events.close()

    def test_partial_non_resource_update_keeps_manual_ram_policy(self):
        with tempfile.TemporaryDirectory(prefix="omni-resource-partial-") as folder:
            brain = AdaptiveBrain(
                "resource-partial-brain",
                Path(folder),
                OmniConfig.micro(system_ram_share_percent=52),
            )
            policy = brain.resource_policy

            brain.update_config({"name": "Renamed only"})

            self.assertEqual(brain.config.system_ram_share_percent, 52.0)
            self.assertIs(brain.resource_policy, policy)
            brain.events.close()


if __name__ == "__main__":
    unittest.main()
