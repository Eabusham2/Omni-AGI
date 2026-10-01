"""Actual ledger and boundary controls, without creating a brain/model."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy

from omni_core.brain import AdaptiveBrain
from omni_core.conversation_ledger import NeuralConversationLedger


class ReceivedInputBoundary(unittest.TestCase):
    def test_retry_reads_only_owned_receipt_not_an_answer(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = NeuralConversationLedger(Path(folder) / "conversation.sqlite3", "brain")
            self.addCleanup(ledger.close)
            text = "The actual received input"
            checksum = hashlib.sha256(text.encode()).hexdigest()
            receipt = {"turnId": "turn", "inputSha256": checksum,
                "humanMessageId": "human", "afterimageId": "episode", "committed": True}
            key = hashlib.sha256(b"brain\0turn\0accepted-input-v1").hexdigest()
            ledger.append("message", [{"id": "human", "role": "human", "content": text,
                "turn_id": "turn", "created_at": "now", "input_accepted_before_reply": True}])
            ledger.append("trace", [{"id": key, "created_at": "now", "inputAccepted": receipt}])
            owner = SimpleNamespace(brain_id="brain", accepted_chat_inputs=[], conversation=ledger)
            self.assertEqual(AdaptiveBrain._accepted_input_receipt(owner, "turn", checksum), receipt)
            self.assertNotIn("answer", receipt)
            with self.assertRaises(ValueError):
                AdaptiveBrain._accepted_input_receipt(owner, "turn", "a" * 64)

    def test_chat_changes_neural_generation_without_advancing_data_prefix(self):
        checkpoint = {"neuralStateChecksum": "before", "committedRecords": 7,
                      "recordPrefixSha256": "prefix", "activeRecordWindow": {"offset": 19}}
        saved = []
        owner = SimpleNamespace(parameter_checksum=lambda: "after",
            ingestion_checkpoints={"source": deepcopy(checkpoint)}, ingestion_joint_generation={"old": True},
            save=lambda: saved.append(deepcopy(owner.ingestion_checkpoints)))
        AdaptiveBrain._save_observed_neural_boundary(owner, "before", "observed input")
        self.assertEqual(saved[0]["source"], {**checkpoint, "neuralStateChecksum": "after"})
        with self.assertRaises(RuntimeError):
            AdaptiveBrain._save_observed_neural_boundary(owner, "wrong", "unowned mutation")

    def test_failed_publication_restores_the_paused_cursor(self):
        with tempfile.TemporaryDirectory() as folder:
            old = {"source": {"neuralStateChecksum": "before", "committedRecords": 7}}
            owner = SimpleNamespace(engine_path=Path(folder), parameter_checksum=lambda: "after",
                ingestion_checkpoints=deepcopy(old), ingestion_joint_generation={"old": True},
                mutable_state_manifest=None, accepted_chat_inputs=[],
                save=lambda: (_ for _ in ()).throw(OSError("precommit failure")))
            with self.assertRaises(OSError):
                AdaptiveBrain._save_observed_neural_boundary(owner, "before", "observed")
            self.assertEqual(owner.ingestion_checkpoints, old)
            self.assertEqual(owner.ingestion_joint_generation, {"old": True})


if __name__ == "__main__":
    unittest.main()
