"""Owned-source/receipt contract; no brain, model or training is constructed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from omni_core.brain import AdaptiveBrain
from omni_core.persistence import BoundedChecksumSource, tensor_checksum
import torch


class ActionResultReceipt(unittest.TestCase):
    def state(self, folder):
        state = SimpleNamespace(engine_path=Path(folder), brain_id="fixture", metrics=lambda: {},
            _sha256_identifier=AdaptiveBrain._sha256_identifier)
        calls, committed = [], {}
        def ingest(**request):
            calls.append(request)
            key = request["transaction_key"]
            duplicate = key in committed
            committed[key] = {"completionReceipt": {"transactionId": "c" * 64,
                "contentHash": request["expected_hash"], "sourceIdentity": "d" * 64}}
            return {"duplicate": duplicate, "source": committed[key], "metrics": {}}
        state.ingest = ingest
        return state, calls

    def test_lost_ack_reuses_exact_durable_ingestion_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            state, calls = self.state(directory)
            identity = "a" * 64
            path = Path(directory) / "action-result-learning" / "evidence" / (identity + ".jsonl")
            path.parent.mkdir(parents=True)
            path.write_text('{"actualOutput":"observed result"}\n', encoding="utf8")
            args = dict(evidence_id=identity, execution_id="12345678-1234-4234-8234-123456789abc",
                evidence_path=str(path), evidence_sha256="b" * 64,
                provenance={"format": "omni-completed-action-result-v1", "actionEventId": "event",
                    "toolId": "system.files", "action": "read", "completedAt": "2026-09-30T00:00:00Z"})
            first = AdaptiveBrain.learn_action_result(state, **args)
            retry = AdaptiveBrain.learn_action_result(state, **args)
            self.assertTrue(first["processed"] and first["committed"])
            self.assertTrue(retry["duplicate"] and retry["committed"])
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(calls[0]["transaction_key"], identity)
            self.assertFalse(calls[0]["allow_replay"])
            self.assertEqual(calls[0]["kind"], "jsonl")

    def test_unowned_or_symlink_evidence_never_reaches_ingestion(self):
        with tempfile.TemporaryDirectory() as directory:
            state, calls = self.state(directory)
            args = dict(evidence_id="a" * 64, execution_id="12345678-1234-4234-8234-123456789abc",
                evidence_path=str(Path(directory) / "other.jsonl"), evidence_sha256="b" * 64,
                provenance={"format": "omni-completed-action-result-v1", "actionEventId": "event",
                    "toolId": "system.files", "action": "read", "completedAt": "now"})
            with self.assertRaises(ValueError):
                AdaptiveBrain.learn_action_result(state, **args)
            self.assertEqual(calls, [])

    def test_missing_neural_completion_is_not_a_successful_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            state, _ = self.state(directory)
            identity = "a" * 64
            path = Path(directory) / "action-result-learning" / "evidence" / (identity + ".jsonl")
            path.parent.mkdir(parents=True); path.write_text("{}\n")
            state.ingest = lambda **_: {"source": {}}
            with self.assertRaises(RuntimeError):
                AdaptiveBrain.learn_action_result(state, identity,
                    "12345678-1234-4234-8234-123456789abc", str(path), "b" * 64,
                    {"format": "omni-completed-action-result-v1", "actionEventId": "event",
                     "toolId": "system.files", "action": "read", "completedAt": "now"})

    def test_bounded_original_checksum_preserves_shape_and_bytes(self):
        original = torch.arange(15, dtype=torch.uint8).reshape(3, 5)
        source = BoundedChecksumSource(tuple(original.shape), original.dtype,
            lambda: iter((original.reshape(-1)[:4], original.reshape(-1)[4:11], original.reshape(-1)[11:])))
        self.assertEqual(tensor_checksum([original]), tensor_checksum([source]))
        incomplete = BoundedChecksumSource(tuple(original.shape), original.dtype,
            lambda: iter((original.reshape(-1)[:4],)))
        with self.assertRaises(ValueError):
            tensor_checksum([incomplete])


if __name__ == "__main__":
    unittest.main()
