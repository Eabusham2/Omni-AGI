"""Bounded real ingestion-cursor relocation; no initial curriculum or corpus."""
import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.offload import ResourcePolicy, ResourceReading
from omni_core.shared_resource_ledger import SharedResourceLedger
from omni_core.source_relocation import rebind_retained_source


class SourceRelocationTests(unittest.TestCase):
    def test_owned_exact_bytes_rebind_the_committed_cursor_without_replay(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            original = root / "selected.jsonl"
            original.write_text('{"text":"A"}\n{"text":"B"}\n{"text":"C"}\n')
            content_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            source_id = hashlib.sha256(("file\0" + str(original)).encode()).hexdigest()
            transaction = hashlib.sha256(b"actual-selected-manifest-transaction").hexdigest()
            quota = SharedResourceLedger(root / "quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(total_memory_bytes=8*1024**3,
                        available_memory_bytes=6*1024**3, process_memory_bytes=256*1024**2,
                        disk_total_bytes=128*1024**3, disk_free_bytes=100*1024**3,
                        managed_process_memory_bytes=256*1024**2, managed_memory_verified=True))
            config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1, d_ff=16, vsa_dim=16,
                router_neurons=4, max_seq_len=16, working_memory_slots=4, memory_resident_items=4,
                modality_channels=4, image_size=8, audio_samples=16, video_frames=2,
                train_batch_size=1, gradient_accumulation=1)
            with patch("omni_core.brain.ResourcePolicy", policy):
                brain = AdaptiveBrain("brain", root / "brain", config)
                try:
                    brain.save(); brain._ingestion_checkpoint_records = 1
                    save = brain.save
                    def stop_after_commit():
                        save()
                        if brain.ingestion_checkpoints: raise RuntimeError("fixture committed boundary")
                    brain.save = stop_after_commit
                    with self.assertRaisesRegex(RuntimeError, "fixture committed boundary"):
                        brain.ingest(path=str(original), name="selected.jsonl", policy="pretrain", transaction_key=transaction)
                    brain.save = save
                    before = copy.deepcopy(next(iter(brain.ingestion_checkpoints.values())))
                    checksum = brain.parameter_checksum()
                    owned = root / "brain" / "learning" / "source-blobs" / (content_hash + ".data")
                    owned.parent.mkdir(parents=True); owned.write_bytes(original.read_bytes()); original.unlink()
                    result = rebind_retained_source(brain, path=str(owned), original_source_identity=source_id,
                        content_hash=content_hash, source_name="selected.jsonl", transaction_id=transaction)
                    self.assertEqual(result["recordsReplayed"], 0)
                    self.assertTrue(result["rebound"])
                    after = next(iter(brain.ingestion_checkpoints.values()))
                    self.assertEqual(after["committedRecords"], before["committedRecords"])
                    self.assertEqual(after["learningSchedule"], before["learningSchedule"])
                    self.assertEqual(after.get("activeRecordWindow"), before.get("activeRecordWindow"))
                    self.assertEqual(brain.parameter_checksum(), checksum)
                    duplicate = rebind_retained_source(brain, path=str(owned), original_source_identity=source_id,
                        content_hash=content_hash, source_name="selected.jsonl", transaction_id=transaction)
                    self.assertTrue(duplicate["idempotent"])
                finally: brain.close()
                restored = AdaptiveBrain.load(root / "brain", expected_brain_id="brain")
                try:
                    self.assertEqual(restored.parameter_checksum(), checksum)
                    self.assertEqual(next(iter(restored.ingestion_checkpoints.values()))["committedRecords"], before["committedRecords"])
                    owned.write_text("changed")
                    with self.assertRaisesRegex(ValueError, "bytes changed|byte count"):
                        rebind_retained_source(restored, path=str(owned), original_source_identity=source_id,
                            content_hash=content_hash, source_name="selected.jsonl", transaction_id=transaction)
                    self.assertEqual(restored.parameter_checksum(), checksum)
                finally: restored.close()


if __name__ == "__main__": unittest.main()
