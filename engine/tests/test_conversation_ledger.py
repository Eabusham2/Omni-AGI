import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.conversation_ledger import NeuralConversationLedger
from omni_core.offload import ResourcePolicy
from worker import Worker


class NeuralConversationLedgerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(131)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-neural-conversation-"
        )
        # The tiny ledger fixture must not depend on the host's desktop disk
        # reserve; production reserve behavior has separate resource tests.
        reserve = mock.patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def test_failed_turn_tail_rollback_preserves_prior_actions(self):
        path = Path(self.temporary.name) / "rollback.sqlite3"
        ledger = NeuralConversationLedger(path, "ledger-brain")
        ledger.append("action", [{"id": "approved-tool", "created_at": "t0"}])
        before = ledger.summary()
        ledger.append(
            "message",
            [{"id": "uncommitted", "role": "human", "content": "secret", "created_at": "t1"}],
        )
        self.assertEqual(
            ledger.truncate_after_head(before["headSequence"], before["headSha256"]),
            1,
        )
        self.assertEqual(ledger.summary(), before)
        self.assertIsNone(ledger.payload_by_id("message", "uncommitted"))
        self.assertEqual(ledger.integrity(), before)
        with self.assertRaisesRegex(ValueError, "head does not match"):
            ledger.truncate_after_head(before["headSequence"], "f" * 64)
        ledger.close()

    def test_omitted_export_ledger_accepts_only_empty_committed_head(self):
        path = Path(self.temporary.name) / "portable-empty.sqlite3"
        ledger = NeuralConversationLedger(path, "portable-brain")
        try:
            self.assertEqual(ledger.truncate_after_head(0, "0" * 64), 0)
            with self.assertRaisesRegex(ValueError, "head does not match"):
                ledger.truncate_after_head(1, "0" * 64)
        finally:
            ledger.close()

    def test_uncapped_hash_chain_pages_interleaved_legacy_rows_once(self):
        path = Path(self.temporary.name) / "conversation.sqlite3"
        ledger = NeuralConversationLedger(path, "ledger-brain")
        messages = [
            {
                "id": "message-%05d" % index,
                "role": "human" if index % 2 == 0 else "brain",
                "content": "full-message-%d-%s" % (index, "x" * 80),
                "created_at": "2026-01-01T00:%02d:%02dZ"
                % ((index // 60) % 60, index % 60),
                "attention_epoch": index // 500,
            }
            for index in range(2_000)
        ]
        traces = [
            {
                "id": "trace-middle",
                "created_at": "2026-01-01T00:00:00.500Z",
                "attention_epoch": 0,
                "input_sha256": hashlib.sha256(b"input").hexdigest(),
            }
        ]
        ledger.backfill(messages, traces)
        first = ledger.summary()
        ledger.backfill(messages, traces)
        self.assertEqual(ledger.summary(), first)
        self.assertEqual(first["messageCount"], 2_000)
        self.assertEqual(first["traceCount"], 1)
        self.assertEqual(
            ledger.counts(attention_epoch=0),
            {
                "totalEntries": 501,
                "messageCount": 500,
                "actionCount": 0,
                "traceCount": 1,
            },
        )
        self.assertEqual(ledger.counts(attention_epoch=3)["messageCount"], 500)
        page = ledger.page(limit=120)
        self.assertEqual(len(page), 120)
        self.assertIn("full-message-1999", page[-1]["payload"]["content"])
        self.assertEqual(ledger.integrity(), first)
        ledger.close()

    def test_rejects_opening_a_copied_ledger_as_another_brain(self):
        path = Path(self.temporary.name) / "identity-bound.sqlite3"
        ledger = NeuralConversationLedger(path, "source-brain")
        ledger.close()

        with self.assertRaisesRegex(ValueError, "belongs to another brain"):
            NeuralConversationLedger(path, "different-brain")

    def test_brain_metadata_is_summary_only_and_receipt_reads_ledger(self):
        root = Path(self.temporary.name) / "brain"
        brain = AdaptiveBrain("receipt-brain", root, OmniConfig.micro())
        result = brain.chat(
            "hello ledger",
            max_new_tokens=2,
            seed=9,
            turn_id="ledger-turn",
        )
        self.assertTrue(result["turnCommitted"])
        metadata = json.loads((root / "engine" / "brain.json").read_text("utf-8"))
        self.assertNotIn("messages", metadata)
        self.assertNotIn("traces", metadata)
        self.assertEqual(metadata["conversation"]["messageCount"], 2)
        self.assertEqual(metadata["conversation"]["traceCount"], 1)
        brain.close()

        worker = Worker()
        try:
            receipt = worker.chat_receipt(
                {
                    "brainId": "receipt-brain",
                    "storagePath": str(root),
                    "turnId": "ledger-turn",
                    "inputSha256": hashlib.sha256(
                        b"hello ledger"
                    ).hexdigest(),
                    "minimumInferenceCount": 0,
                },
                "receipt-query",
            )
            self.assertTrue(receipt["committed"])
            self.assertEqual(receipt["humanMessage"]["content"], "hello ledger")
            self.assertEqual(receipt["trace"]["id"], result["trace"]["id"])
        finally:
            worker._shutdown_inline_generations()


if __name__ == "__main__":
    unittest.main()
