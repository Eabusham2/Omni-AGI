"""Opt-in saved-native rehearsal proof; reuses a declared temporary fixture.

Never initializes/trains a user brain or reruns the full initial curriculum.
OMNI_REHEARSAL_NATIVE_PARENT must name an already initialized tiny test parent.
"""
import hashlib
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.capability_rehearsal import (
    CapabilityRehearsalPolicy, CapabilityScheduleState, advance_schedule_state,
    eligible_ground_up_rehearsal, rehearse_capabilities,
)
from omni_core.offload import ResourcePolicy, ResourceReading
from omni_core.shared_resource_ledger import SharedResourceLedger


@unittest.skipUnless(os.environ.get("OMNI_REHEARSAL_NATIVE_PARENT"), "opt-in existing tiny native fixture only")
class SavedNativeCapabilityRehearsal(unittest.TestCase):
    def test_saved_native_replay_preserves_origin_and_cursor_then_reloads(self):
        source = Path(os.environ["OMNI_REHEARSAL_NATIVE_PARENT"]).resolve()
        if not source.parent.name.startswith("omni-verified-curriculum."):
            self.fail("refusing to use anything except the dedicated initialized test fixture")
        tracked = ("brain.json", "core.safetensors", "plasticity.safetensors", "origin/brain.json")
        hashes = {name: hashlib.sha256((source / "engine" / name).read_bytes()).hexdigest() for name in tracked}
        started = time.perf_counter()
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="omni-native-rehearsal-") as temporary:
            root = Path(temporary)
            storage = root / "brain"
            shutil.copytree(source, storage)
            quota = SharedResourceLedger(root / "fixture-quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(total_memory_bytes=8 * 1024**3,
                        available_memory_bytes=6 * 1024**3, process_memory_bytes=128 * 1024**2,
                        disk_total_bytes=128 * 1024**3, disk_free_bytes=100 * 1024**3,
                        managed_process_memory_bytes=128 * 1024**2, managed_memory_verified=True))
            with patch("omni_core.brain.ResourcePolicy", policy):
                brain = AdaptiveBrain.load(storage, expected_brain_id="a")
                try:
                    self.assertTrue(eligible_ground_up_rehearsal(brain))
                    self.assertEqual(tuple(brain.decoder.action_policy.parameters()), ())
                    before_cursors = json.dumps(brain.ingestion_checkpoints, sort_keys=True)
                    before_memory = (len(brain.memory.neurons), len(brain.memory.assemblies), len(brain.memory.synapses))
                    before_origin = (storage / "engine/origin/brain.json").read_bytes()
                    grounded = int(brain.decoder.action_argument_head.grounded_steps)
                    with patch("omni_core.capability_rehearsal.probe_capabilities", side_effect=AssertionError("no post-training quiz")):
                        receipt = rehearse_capabilities(brain, phase="start", committed_global_waves=0,
                            policy=CapabilityRehearsalPolicy(maximum_action_steps=2))
                    self.assertEqual(json.dumps(brain.ingestion_checkpoints, sort_keys=True), before_cursors)
                    self.assertEqual((len(brain.memory.neurons), len(brain.memory.assemblies), len(brain.memory.synapses)), before_memory)
                    self.assertEqual(int(brain.decoder.action_argument_head.grounded_steps), grounded)
                    self.assertFalse(receipt["readinessProbeExecuted"])
                    self.assertFalse(receipt["corticalLanguageTrainingApplied"])
                    self.assertEqual(receipt["action"]["attemptedSteps"], 2)
                    schedule = advance_schedule_state(CapabilityScheduleState(), receipt)
                    self.assertTrue(schedule.start_completed)
                    brain.save()
                    self.assertEqual((storage / "engine/origin/brain.json").read_bytes(), before_origin)
                    parameter_checksum = brain.parameter_checksum()
                finally:
                    brain.close()
                reloaded = AdaptiveBrain.load(storage, expected_brain_id="a")
                try:
                    self.assertTrue(eligible_ground_up_rehearsal(reloaded))
                    self.assertEqual(reloaded.parameter_checksum(), parameter_checksum)
                    self.assertEqual(json.dumps(reloaded.ingestion_checkpoints, sort_keys=True), before_cursors)
                finally:
                    reloaded.close()
            for name, expected in hashes.items():
                self.assertEqual(hashlib.sha256((source / "engine" / name).read_bytes()).hexdigest(), expected)
            print(json.dumps({"scope": "existing tiny initialized native fixture; two declared replay steps and reload",
                "candidateAccepted": receipt["candidateAccepted"], "rollbackVerified": receipt["rollbackVerified"],
                "before": receipt["before"], "candidate": receipt["candidate"], "after": receipt["after"],
                "sourceUnchanged": True, "originUnchanged": True, "cursorUnchanged": True,
                "elapsedSeconds": time.perf_counter() - started}, sort_keys=True), flush=True)


if __name__ == "__main__":
    unittest.main()
