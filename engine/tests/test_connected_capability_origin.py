"""Real tiny initial curriculum and connected replay; no fluency claims."""
import copy
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
from omni_core.capability_rehearsal import CapabilityRehearsalPolicy, eligible_ground_up_rehearsal, rehearse_capabilities
from omni_core.config import OmniConfig
from omni_core.full_merge import assemble_full_merge
from omni_core.offload import ResourcePolicy, ResourceReading
from omni_core.shared_resource_ledger import SharedResourceLedger
from omni_core.persistence import tensor_checksum
from omni_core.connected_regions import NativeRegion


class ConnectedCapabilityOriginTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("OMNI_RUN_NATIVE_CURRICULUM") == "1", "opt-in real initial curriculum; not a routine unit test")
    def test_inherited_curriculum_uses_real_parent_origin_and_joined_learning(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="omni-connected-curriculum-") as temporary:
            root = Path(temporary)
            quota = SharedResourceLedger(root / "quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(total_memory_bytes=8 * 1024**3,
                        available_memory_bytes=6 * 1024**3, process_memory_bytes=256 * 1024**2,
                        disk_total_bytes=128 * 1024**3, disk_free_bytes=100 * 1024**3,
                        managed_process_memory_bytes=256 * 1024**2, managed_memory_verified=True))
            config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1,
                d_ff=16, vsa_dim=16, router_neurons=4, max_seq_len=32,
                working_memory_slots=4, memory_resident_items=4, modality_channels=4,
                image_size=8, audio_samples=16, video_frames=2,
                train_batch_size=1, gradient_accumulation=1)
            first = root / "merged/merge/parents/a"
            second = root / "merged/merge/parents/b"
            started = time.monotonic()
            with patch("omni_core.brain.ResourcePolicy", policy):
                saved_fixture = os.environ.get("OMNI_CONNECTED_CURRICULUM_FIXTURE")
                if saved_fixture:
                    # Optional reuse of a previously, genuinely initialized
                    # tiny fixture. Load still authenticates every receipt;
                    # this never fabricates a manifest or touches user data.
                    shutil.copytree(Path(saved_fixture), first)
                    original = AdaptiveBrain.load(first, expected_brain_id="a")
                else:
                    original = AdaptiveBrain.create("a", first, copy.deepcopy(config),
                        progress=lambda phase, *_: print("curriculum fixture:", phase, round(time.monotonic() - started, 2), flush=True))
                try:
                    self.assertTrue(original._ground_up_action_origin_verified)
                    self.assertFalse(original.ground_up_training_manifest["publicCapabilityReadiness"]["readinessProbeExecuted"])
                finally:
                    original.close()
                shutil.copytree(first, second)
                fork = AdaptiveBrain.load(second, expected_brain_id="a")
                try:
                    fork.brain_id = "b"
                    # Match the host's private-fork ledger rekey; changing
                    # only brain.json would rightly fail merge ownership.
                    fork.conversation.connection.execute("UPDATE meta SET value=? WHERE key='brain_id'", ("b",))
                    fork.conversation.connection.commit()
                    fork.conversation.brain_id = "b"
                    # A genuine fast neural difference is enough to prevent
                    # proven-identical-owner dedup. Unrelated single-parent
                    # slow-learning qualification is outside this fixture.
                    fork.memory.learn("new local episode", kind="fixture", retain_source_text=False)
                    fork.save()
                    fork.export_packed_ternary()
                finally:
                    fork.close()
                def native_hashes():
                    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for directory in (first, second) for path in directory.rglob("*") if path.is_file()}
                before = native_hashes()
                print("curriculum fixture: parents ready", round(time.monotonic() - started, 2), flush=True)
                declarations = [{"brainId": name, "enginePath": "merge/parents/" + name + "/engine",
                    "generationSha256": hashlib.sha256(name.encode()).hexdigest()} for name in ("a", "b")]
                assemble_full_merge(root / "merged", "merged", declarations, copy.deepcopy(config))
                print("curriculum fixture: merged", round(time.monotonic() - started, 2), flush=True)
                self.assert_loaded_replay(root / "merged")
                self.assertEqual(native_hashes(), before)

    @unittest.skipUnless(os.environ.get("OMNI_CONNECTED_MERGED_FIXTURE"), "optional continuation of an actually assembled fixture")
    def test_completed_merge_reloads_and_replays_inherited_curriculum(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="omni-connected-replay-") as temporary:
            root = Path(temporary)
            shutil.copytree(Path(os.environ["OMNI_CONNECTED_MERGED_FIXTURE"]), root / "merged")
            # No allocation exemption is needed for this already initialized
            # tiny native state. Production resource checks stay active.
            parents = root / "merged/merge/parents"
            def hashes():
                return {str(path.relative_to(parents)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in parents.rglob("*") if path.is_file()}
            before = hashes()
            self.assert_loaded_replay(root / "merged")
            self.assertEqual(hashes(), before)

    def assert_loaded_replay(self, directory):
        started = time.monotonic()
        merged = AdaptiveBrain.load(directory, expected_brain_id="merged")
        try:
            self.assertIsNone(merged.ground_up_training_manifest)
            self.assertTrue(merged._ground_up_action_origin_verified)
            self.assertTrue(eligible_ground_up_rehearsal(merged))
            self.assertIsNone(merged._starter_action_language_cache)
            self.assertEqual(len(merged._inherited_capability_curriculum["origins"]), 2)
            print("curriculum fixture: authenticated load", round(time.monotonic() - started, 2), flush=True)
            activity = [merged.liquid_state, merged.router.population.membrane, merged.router.population.spike_count]
            for region in merged.decoder.modules():
                if isinstance(region, NativeRegion):
                    activity.extend((region.liquid_state, region.router.population.membrane, region.router.population.spike_count))
            activity_before = tensor_checksum(activity)
            workspace_before = (copy.deepcopy(merged.current_context), list(merged.recent_token_context), copy.deepcopy(merged.recent_token_activity))
            # A selected, declared trajectory exercises the real connected
            # forward and native head mutation. Full35-row rehearsal retains
            # its default coverage but is performance-unqualified here.
            features = merged._starter_action_features(indices=(0,))
            self.assertEqual(features[0].shape, (1, 16))
            self.assertEqual(tensor_checksum(activity), activity_before)
            self.assertEqual((merged.current_context, merged.recent_token_context, merged.recent_token_activity), workspace_before)
            head_before = tensor_checksum([*merged.decoder.action_policy.state_dict().values(), *merged.decoder.internal_action_policy.state_dict().values()])
            with patch.object(merged, "_starter_action_features", side_effect=AssertionError("duplicate feature forward")):
                receipt = merged._calibrate_starter_action_policy(max_steps=2, minimum_steps=2, strict=False, _feature_batch=features)
            self.assertEqual(receipt["examples"], 1)
            self.assertEqual(receipt["steps"], 2)
            self.assertFalse(receipt["rolledBack"])
            self.assertNotEqual(tensor_checksum([*merged.decoder.action_policy.state_dict().values(),
                *merged.decoder.internal_action_policy.state_dict().values()]), head_before)
            print("curriculum fixture: one declared trajectory learned", round(time.monotonic() - started, 2), flush=True)
            merged.save()
        finally:
            merged.close()
        reloaded = AdaptiveBrain.load(directory, expected_brain_id="merged")
        try:
            self.assertTrue(eligible_ground_up_rehearsal(reloaded))
            self.assertIsNone(reloaded.ground_up_training_manifest)
        finally:
            reloaded.close()


if __name__ == "__main__":
    unittest.main()
