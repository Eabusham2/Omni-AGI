"""Mocked chat failure recovery; no neural model is constructed or trained."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from omni_core.conversation_ledger import NeuralConversationLedger
from worker import Worker


class FailedChatCommitRecoveryTests(unittest.TestCase):
    def _fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        engine = root / "engine"
        engine.mkdir()
        ledger = NeuralConversationLedger(engine / "conversation.sqlite3", "brain")
        ledger.append("action", [{"id": "prior", "created_at": "t0"}])
        (engine / "brain.json").write_text(
            json.dumps({"completed_chat_turns": []}), encoding="utf-8"
        )
        brain = MagicMock()
        brain.brain_id = "brain"
        brain.storage_path = root
        brain.conversation = ledger
        brain.close.side_effect = ledger.close
        worker = Worker()
        worker.brains["brain"] = brain
        return temporary, worker, brain, ledger, engine

    def test_save_failure_discards_unacknowledged_chat_and_reload_committed_state(self):
        temporary, worker, brain, ledger, engine = self._fixture()
        restored = MagicMock()

        def failed_chat(*_args, **_kwargs):
            ledger.append(
                "message",
                [{"id": "failed", "role": "human", "content": "not committed", "created_at": "t1"}],
            )
            raise OSError("simulated checkpoint save failure")

        brain.chat.side_effect = failed_chat
        try:
            with patch.object(worker, "_get", return_value=brain), patch(
                "worker.AdaptiveBrain.load", return_value=restored
            ) as load:
                with self.assertRaisesRegex(OSError, "save failure"):
                    worker.chat(
                        {"brainId": "brain", "input": "not committed", "toolSchemas": []},
                        "turn-failed",
                    )
            load.assert_called_once_with(engine.parent, expected_brain_id="brain")
            self.assertIs(worker.brains["brain"], restored)
            verified = NeuralConversationLedger(engine / "conversation.sqlite3", "brain")
            try:
                self.assertIsNone(verified.payload_by_id("message", "failed"))
                self.assertEqual(verified.summary()["actionCount"], 1)
                verified.integrity()
            finally:
                verified.close()
        finally:
            worker.brains.clear()
            worker.shutdown({}, "shutdown")
            temporary.cleanup()

    def test_post_commit_failure_preserves_matching_receipt_and_chat_rows(self):
        temporary, worker, brain, ledger, engine = self._fixture()
        restored = MagicMock()

        def committed_then_failed(*_args, **kwargs):
            ledger.append(
                "message",
                [{"id": "committed", "role": "human", "content": "kept", "created_at": "t1"}],
            )
            (engine / "brain.json").write_text(
                json.dumps({
                    "completed_chat_turns": [{
                        "turnId": kwargs["turn_id"],
                        "inputSha256": hashlib.sha256(b"kept").hexdigest(),
                    }]
                }),
                encoding="utf-8",
            )
            raise OSError("simulated post-commit acknowledgement failure")

        brain.chat.side_effect = committed_then_failed
        try:
            with patch.object(worker, "_get", return_value=brain), patch(
                "worker.AdaptiveBrain.load", return_value=restored
            ):
                with self.assertRaisesRegex(OSError, "acknowledgement"):
                    worker.chat(
                        {"brainId": "brain", "input": "kept", "toolSchemas": []},
                        "turn-committed",
                    )
            verified = NeuralConversationLedger(engine / "conversation.sqlite3", "brain")
            try:
                self.assertIsNotNone(verified.payload_by_id("message", "committed"))
                verified.integrity()
            finally:
                verified.close()
        finally:
            worker.brains.clear()
            worker.shutdown({}, "shutdown")
            temporary.cleanup()

    def test_failed_feedback_reloads_committed_state_not_live_stdp_mutation(self):
        temporary, worker, brain, ledger, engine = self._fixture()
        restored = MagicMock()
        brain.feedback.side_effect = OSError("simulated feedback save failure")
        try:
            with patch.object(worker, "_get", return_value=brain), patch(
                "worker.AdaptiveBrain.load", return_value=restored
            ) as load:
                with self.assertRaisesRegex(OSError, "feedback save failure"):
                    worker.feedback(
                        {"brainId": "brain", "text": "feedback", "direction": "up"},
                        "feedback-request",
                    )
            load.assert_called_once_with(engine.parent, expected_brain_id="brain")
            self.assertIs(worker.brains["brain"], restored)
        finally:
            worker.brains.clear()
            worker.shutdown({}, "shutdown")
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
