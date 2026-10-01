"""Receipt/ledger recovery fixtures only; no AdaptiveBrain construction/forward."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import Worker
from omni_core.conversation_ledger import NeuralConversationLedger


class ReceivedInputRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-input-recovery-")
        self.addCleanup(temporary.cleanup)
        self.storage = Path(temporary.name) / "brain"
        self.engine = self.storage / "engine"
        self.engine.mkdir(parents=True)
        self.path = self.engine / "conversation.sqlite3"
        self.input = "actual received human fixture input"
        self.sha = hashlib.sha256(self.input.encode()).hexdigest()
        self.human = {"id": "admitted-human", "role": "human", "content": self.input, "turn_id": "turn",
            "created_at": "2026-09-30T12:00:00.000Z", "attention_epoch": 1, "input_accepted_before_reply": True}
        self.receipt = {"turnId": "turn", "inputSha256": self.sha, "humanMessageId": "admitted-human",
            "afterimageId": "a" * 64, "committed": True}
        ledger = NeuralConversationLedger(self.path, "brain")
        ledger.append("message", [{"id": "prior-human", "role": "human", "content": "prior", "created_at": "2026-09-30T11:00:00.000Z", "attention_epoch": 1}])
        self.prior = ledger.summary()
        ledger.append("message", [self.human])
        ledger.append("trace", [{"id": "input-trace", "kind": "chat-input-accepted", "inputAccepted": self.receipt,
            "created_at": self.human["created_at"], "attention_epoch": 1}])
        self.admitted = ledger.summary()
        ledger.append("message", [{"id": "unsaved-assistant", "role": "brain", "content": "actual fixture orphan, not committed", "created_at": self.human["created_at"], "attention_epoch": 1}])
        ledger.close()
        self.brain = SimpleNamespace(brain_id="brain", storage_path=self.storage, close=lambda: None)
        self.worker = Worker.__new__(Worker)
        self.worker.brains = {"brain": self.brain}
        self.worker._discard_inline_generations = lambda _brain: None

    def save(self, receipt):
        (self.engine / "brain.json").write_text(json.dumps({"brain_id": "brain", "completed_chat_turns": [],
            "accepted_chat_inputs": [receipt] if receipt else [], "conversation": self.admitted}), encoding="utf-8")

    def test_failure_after_admission_preserves_exact_human_and_committed_input_head_not_pre_turn_head(self):
        self.save(self.receipt)
        loaded = SimpleNamespace(brain_id="brain", receipt_reused=True)
        with patch("worker.AdaptiveBrain.load", return_value=loaded) as load:
            self.assertTrue(self.worker._restore_committed_brain_after_failed_chat(self.brain,
                prior_ledger_head=self.prior, turn_id="turn", input_sha256=self.sha))
            load.assert_called_once_with(self.storage, expected_brain_id="brain")
        ledger = NeuralConversationLedger(self.path, "brain")
        try:
            self.assertEqual(ledger.payload_by_id("message", "admitted-human"), self.human)
            self.assertIsNotNone(ledger.payload_by_id("message", "prior-human"))
            self.assertIsNone(ledger.payload_by_id("message", "unsaved-assistant"))
            self.assertEqual(ledger.summary()["headSha256"], self.admitted["headSha256"])
        finally: ledger.close()
        self.assertEqual(json.loads((self.engine / "brain.json").read_text())["accepted_chat_inputs"], [self.receipt])

    def test_no_durable_admission_drops_only_uncommitted_new_rows(self):
        self.save(None)
        with patch("worker.AdaptiveBrain.load", return_value=SimpleNamespace(brain_id="brain")):
            self.assertTrue(self.worker._restore_committed_brain_after_failed_chat(self.brain,
                prior_ledger_head=self.prior, turn_id="turn", input_sha256=self.sha))
        ledger = NeuralConversationLedger(self.path, "brain")
        try:
            self.assertIsNone(ledger.payload_by_id("message", "admitted-human"))
            self.assertEqual(ledger.summary()["headSha256"], self.prior["headSha256"])
        finally: ledger.close()

    def test_forged_receipt_for_a_different_human_never_loads_unverified_state(self):
        self.save({**self.receipt, "humanMessageId": "missing"})
        with patch("worker.AdaptiveBrain.load") as load:
            self.assertFalse(self.worker._restore_committed_brain_after_failed_chat(self.brain,
                prior_ledger_head=self.prior, turn_id="turn", input_sha256=self.sha))
            load.assert_not_called()
        self.assertNotIn("brain", self.worker.brains)


if __name__ == "__main__": unittest.main()
