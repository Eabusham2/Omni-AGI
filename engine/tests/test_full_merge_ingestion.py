"""Short real cursor adoption: no corpus run and no capability claims."""
import copy
import hashlib
import sys
import tempfile
import unittest
import json
import uuid
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.connected_runtime import activate_pending_ingestion, inherited_ingestion_records
from omni_core.full_merge import assemble_full_merge
from omni_core.offload import ResourcePolicy, ResourceReading
from omni_core.persistence import snapshot_files
from omni_core.shared_resource_ledger import SharedResourceLedger


class FullMergeIngestionTests(unittest.TestCase):
    def test_partial_native_source_resumes_at_its_same_record_window_after_merge(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_evidence = hashlib.sha256(b"original-action-result").hexdigest()
            new_evidence = hashlib.sha256(b"merged-action-result").hexdigest()
            dataset = root / (original_evidence + ".jsonl")
            dataset.write_text('{"text":"A"}\n{"text":"B"}\n{"text":"C"}\n', encoding="utf-8")
            evidence_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
            source_name = "action-result-" + original_evidence
            quota = SharedResourceLedger(root / "quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(total_memory_bytes=8 * 1024**3,
                        available_memory_bytes=6 * 1024**3, process_memory_bytes=256 * 1024**2,
                        disk_total_bytes=128 * 1024**3, disk_free_bytes=100 * 1024**3,
                        managed_process_memory_bytes=256 * 1024**2, managed_memory_verified=True))
            config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1,
                d_ff=16, vsa_dim=16, router_neurons=4, max_seq_len=16,
                working_memory_slots=4, memory_resident_items=4, modality_channels=4,
                image_size=8, audio_samples=16, video_frames=2,
                train_batch_size=1, gradient_accumulation=1)
            declarations = []
            with patch("omni_core.brain.ResourcePolicy", policy):
                for identifier in ("a", "b"):
                    parent_root = root / "merged" / "merge" / "parents" / identifier
                    parent = AdaptiveBrain(identifier, parent_root, copy.deepcopy(config))
                    try:
                        parent.save()
                        if identifier == "a":
                            parent._ingestion_checkpoint_records = 1
                            save = parent.save
                            stopped = False
                            def stop_at_checkpoint():
                                nonlocal stopped
                                save()
                                if parent.ingestion_checkpoints and not stopped:
                                    stopped = True
                                    raise RuntimeError("fixture stopped at the committed source boundary")
                            parent.save = stop_at_checkpoint
                            with self.assertRaisesRegex(RuntimeError, "fixture stopped"):
                                parent.ingest(path=str(dataset), name=source_name, policy="pretrain", transaction_key=original_evidence)
                            checkpoint = copy.deepcopy(next(iter(parent.ingestion_checkpoints.values())))
                            self.assertLess(checkpoint["committedRecords"], 3)
                    finally:
                        parent.close()
                    declarations.append({"brainId": identifier, "enginePath": "merge/parents/" + identifier + "/engine",
                        "generationSha256": hashlib.sha256(identifier.encode()).hexdigest()})
                receipt = assemble_full_merge(root / "merged", "merged", declarations, copy.deepcopy(config))
                self.assertEqual(receipt["runtimeAdoption"]["counts"]["pendingDatasetJobs"], 1)
                saved = json.loads((root / "merged/engine/brain.json").read_text())
                next_parent = SimpleNamespace(parent_id="merged", region_id="regional-merge", engine_relative_path="merge/parents/merged/engine", generation_sha256="f" * 64)
                inherited = list(inherited_ingestion_records(next_parent, saved, AdaptiveBrain._validated_ingestion_checkpoints))
                self.assertEqual(len(inherited), 1)
                self.assertEqual(inherited[0]["checkpoint"], saved["connected_pending_ingestions"][0]["checkpoint"])
                self.assertEqual(inherited[0]["status"], "queued")
                # This is the actual evolution clone shape. No original host
                # root or merge/parents tree exists beside the final engine.
                snapshot_files(root / "merged/engine", root / "stable")
                snapshot_files(root / "stable", root / "isolated/engine")
                self.assertFalse((root / "isolated/merge").exists())
                isolated = AdaptiveBrain.load(root / "isolated", expected_brain_id="merged")
                try:
                    self.assertEqual(isolated._connected_regions.descriptor["sha256"], receipt["descriptorSha256"])
                    self.assertEqual(isolated.pending_chat_slow_learning, [])
                    self.assertEqual(isolated.connected_pending_ingestions, saved["connected_pending_ingestions"])
                finally:
                    isolated.close()
                merged = AdaptiveBrain.load(root / "merged", expected_brain_id="merged")
                try:
                    pending = merged.connected_pending_ingestions[0]
                    continuation = {"pendingId": pending["id"], "originalEvidenceId": original_evidence, "parentId": "a"}
                    # Keep original evidence bytes/header while assigning the
                    # new identity its own owned spool path and receipt IDs.
                    evidence_path = merged.engine_path / "action-result-learning/evidence" / (new_evidence + ".jsonl")
                    evidence_path.parent.mkdir(parents=True)
                    evidence_path.write_bytes(dataset.read_bytes())
                    provenance = {"format": "omni-completed-action-result-v1", "actionEventId": "observed-parent-action",
                        "toolId": "files", "action": "read", "completedAt": "2026-10-01T00:00:00Z"}
                    execution_id = str(uuid.uuid4())
                    # First bind to the old path, as a saved active job would
                    # before export/import moves its exact evidence file.
                    initial = activate_pending_ingestion(merged, pending["id"], dataset, source_name)
                    with patch.object(merged, "ingest", side_effect=RuntimeError("fixture stopped after path rebind")):
                        with self.assertRaisesRegex(RuntimeError, "after path rebind"):
                            merged.learn_action_result(new_evidence, execution_id, str(evidence_path), evidence_sha,
                                provenance, merge_continuation=continuation)
                    active = {"checkpoint": next(iter(merged.ingestion_checkpoints.values()))}
                    self.assertNotEqual(active["checkpoint"]["sourceIdentity"], initial["checkpoint"]["sourceIdentity"])
                    self.assertEqual(active["checkpoint"]["committedRecords"], checkpoint["committedRecords"])
                    self.assertEqual(active["checkpoint"].get("activeRecordWindow"), checkpoint.get("activeRecordWindow"))
                    self.assertEqual(active["checkpoint"]["learningSchedule"], checkpoint["learningSchedule"])
                    self.assertEqual(active["checkpoint"]["neuralStateChecksum"], merged.parameter_checksum())
                    inherited_active = list(inherited_ingestion_records(next_parent, merged._metadata(), AdaptiveBrain._validated_ingestion_checkpoints))
                    self.assertEqual(len(inherited_active), 1)
                    self.assertEqual(inherited_active[0]["checkpoint"], active["checkpoint"])
                    self.assertEqual(inherited_active[0]["status"], "queued")
                finally:
                    merged.close()
                resumed = AdaptiveBrain.load(root / "merged", expected_brain_id="merged")
                try:
                    self.assertTrue(activate_pending_ingestion(resumed, pending["id"], evidence_path, source_name)["idempotent"])
                    # These cursor fixtures deliberately use low-level random
                    # constructors, not claimed initialized parents. Only the
                    # unrelated capability finalization is excluded here;
                    # real inherited eligibility/replay has its own created-
                    # parent fixture. All source learning, cursor rebinding,
                    # record traversal and completion saves remain real.
                    self.assertFalse(resumed._ground_up_action_origin_verified)
                    with patch.object(resumed, "_can_retain_native_action_policy", return_value=True), patch.object(
                            resumed, "_calibrate_starter_action_policy", return_value={
                                "steps": 0, "calibrated": False, "applied": False}):
                        result = resumed.learn_action_result(new_evidence, execution_id, str(evidence_path), evidence_sha,
                            provenance, merge_continuation=continuation)
                    self.assertFalse(resumed.ingestion_checkpoints)
                    self.assertEqual(resumed.connected_pending_ingestions[0]["status"], "complete")
                    self.assertEqual(result["source"]["coverage"]["processedRecords"], 3)
                    self.assertEqual(result["evidenceId"], new_evidence)
                    checksum = resumed.parameter_checksum()
                    duplicate = resumed.learn_action_result(new_evidence, execution_id, str(evidence_path), evidence_sha,
                        provenance, merge_continuation=continuation)
                    self.assertTrue(duplicate["duplicate"])
                    self.assertEqual(resumed.parameter_checksum(), checksum)
                finally:
                    resumed.close()


if __name__ == "__main__":
    unittest.main()
