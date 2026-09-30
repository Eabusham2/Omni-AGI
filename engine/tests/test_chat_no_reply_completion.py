"""Empty-output method/receipt fixtures, without a neural constructor or run."""
import ast
import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.chat_steering import (
    generation_completion, mark_no_reply_turn, validate_no_reply_turn,
)
from omni_core.conversation_ledger import NeuralConversationLedger
from omni_core.recent_token_activity import RecentTokenActivity
from worker import Worker


def saved_turn():
    input_text = "actual human message"
    input_hash = hashlib.sha256(input_text.encode()).hexdigest()
    checksum = "a" * 64
    human = {"id": "human", "role": "human", "content": input_text,
             "created_at": "2026-09-30T12:00:00.000Z", "turn_id": "turn"}
    completion = {"id": "completion", "role": "brain", "content": "",
                  "created_at": "2026-09-30T12:00:01.000Z", "turn_id": "turn"}
    trace = {"id": "trace", "turn_id": "turn", "input_sha256": input_hash,
             "parameter_checksum_before": "b" * 64, "parameter_checksum_after": checksum,
             "generation_stop_reason": "no-reply", "generation_decoder_stop_reason": "eos",
             "generation_printable_text_characters": 0, "generated_token_count": 1,
             "generation_no_reply_reason": "no-decoded-text",
             "created_at": "2026-09-30T12:00:02.000Z"}
    receipt = {"format": "omni-completed-chat-turn", "formatVersion": 1,
               "turnId": "turn", "inputSha256": input_hash,
               "humanMessageId": "human", "brainMessageId": "completion", "traceId": "trace",
               "inferenceCount": 1, "parameterChecksumAfter": checksum,
               "committedAt": "2026-09-30T12:00:03.000Z"}
    mark_no_reply_turn(human, completion, receipt)
    return human, completion, trace, receipt


class ChatNoReplyCompletionTests(unittest.TestCase):
    def test_observed_empty_special_whitespace_and_nonprintable_output(self):
        for ids, decoded, reason in [
            ([], "", "no-generated-tokens"), ([2], "", "no-decoded-text"),
            ([35, 12], " \t\n", "whitespace-only"), ([0], "\u200b", "no-printable-text"),
        ]:
            with self.subTest(reason=reason):
                result = generation_completion(ids, lambda _ids: decoded)
                self.assertEqual(result["text"], "")
                self.assertEqual(result["noReplyReason"], reason)
                self.assertTrue(result["noReply"])
                self.assertFalse(result["zeroTokenYield"])
                self.assertFalse(result["steered"])
                self.assertFalse(result["nativeStopped"])

    def test_actual_question_mark_is_not_suppressed_or_manufactured(self):
        result = generation_completion([66], lambda _ids: "?")
        self.assertEqual(result["text"], "?")
        self.assertNotIn("noReply", result)

    def test_no_reply_marker_requires_exact_zero_text_evidence(self):
        human, completion, trace, receipt = saved_turn()
        self.assertTrue(validate_no_reply_turn(human, completion, trace, receipt))
        completion["content"] = "?"
        with self.assertRaises(ValueError):
            validate_no_reply_turn(human, completion, trace, receipt)
        with self.assertRaises(ValueError):
            mark_no_reply_turn(human, completion, receipt)

    def test_recent_context_has_only_real_human_input_and_no_assistant_target(self):
        tokenizer = SimpleNamespace(encode=lambda value: list(value.encode()), human_id=259, brain_id=260, eos_id=2)
        owner = SimpleNamespace(tokenizer=tokenizer, config=SimpleNamespace(max_seq_len=10))
        tokens, removed = AdaptiveBrain._bounded_completed_turn_tokens(owner, "actual input", "")
        self.assertEqual(tokens, [259] + list(b"actual input")[-8:] + [2])
        self.assertNotIn(260, tokens)
        self.assertEqual(removed, 4)
        activity = RecentTokenActivity.from_state(None, tokens, human_id=259)
        fitted = activity.fit_capacity(tokens, 5, human_id=259, brain_id=260, eos_id=2)
        self.assertEqual(fitted, [259] + list(b"put") + [2])

    def test_idempotent_receipt_returns_same_zero_text_without_replaying_effects(self):
        human, completion, trace, receipt = saved_turn()
        owner = SimpleNamespace(brain_id="brain", messages=[human, completion], traces=[trace],
                                counters={"inference_count": 1}, metrics=lambda: {}, runtime_card=lambda: {})
        result = AdaptiveBrain._completed_chat_result(owner, receipt)
        self.assertTrue(result["noReply"])
        self.assertTrue(result["idempotentCompletion"])
        self.assertTrue(result["turnCommitted"])
        self.assertEqual(result["text"], "")
        self.assertEqual(result["humanMessage"]["content"], human["content"])
        self.assertEqual(result["actions"], [])
        self.assertEqual(AdaptiveBrain._validated_completed_chat_turns([receipt]), [receipt])

    def test_worker_no_reply_is_successful_commit_not_control_yield_or_reload(self):
        worker = Worker.__new__(Worker)
        worker._cooperative_cancel = threading.Event()
        worker.notify = lambda *_args, **_kwargs: None
        result = {"text": "", "noReply": True, "turnCommitted": True, "trace": {"id": "trace"}}
        worker._get = lambda _params: SimpleNamespace(brain_id="brain", conversation=SimpleNamespace(summary=lambda: {}),
                                                     chat=lambda *_args, **_kwargs: result)
        worker._restore_committed_brain_after_failed_chat = lambda *_args, **_kwargs: self.fail("committed no-reply reloaded")
        self.assertIs(worker._chat({"brainId": "brain", "input": "actual human message", "streamId": "turn"}, "rpc"), result)
        self.assertFalse(worker._cooperative_cancel.is_set())

    def test_load_free_receipt_survives_metadata_window_eviction_and_reopen(self):
        human, completion, trace, receipt = saved_turn()
        with tempfile.TemporaryDirectory(prefix="omni-no-reply-") as directory:
            root = Path(directory)
            engine = root / "engine"
            engine.mkdir()
            ledger = NeuralConversationLedger(engine / "conversation.sqlite3", "brain")
            ledger.append("message", [human, completion])
            ledger.append("trace", [trace])
            ledger.close()
            metadata = {"brain_id": "brain", "messages": [], "traces": [],
                        "completed_chat_turns": [receipt], "updated_at": receipt["committedAt"],
                        "counters": {"inference_count": 1, "plasticity_events": 0, "consolidation_cycles": 0},
                        "substrate": {"persistence": {"activeGeneration": "b" * 64}},
                        "mutable_state": {"activeGeneration": "c" * 64}}
            (engine / "brain.json").write_text(json.dumps(metadata), encoding="utf-8")
            worker = Worker.__new__(Worker)
            worker._storage = lambda _params, _brain_id: root
            worker._get = lambda _params: self.fail("receipt query loaded a brain")
            result = worker.chat_receipt({"brainId": "brain", "turnId": "turn", "inputSha256": receipt["inputSha256"]}, "receipt")
            self.assertTrue(result["noReply"])
            self.assertEqual(result["generationEnd"], "no-reply")
            self.assertEqual(result["brainMessage"]["content"], "")
            self.assertEqual(result["humanMessage"]["content"], human["content"])
            self.assertEqual(result["trace"], trace)

    def test_production_chat_has_no_decode_special_or_literal_fallback(self):
        source = Path(__file__).resolve().parents[1] / "omni_core" / "brain.py"
        tree = ast.parse(source.read_text())
        brain = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        chat = next(node for node in brain.body if isinstance(node, ast.FunctionDef) and node.name == "chat")
        for node in ast.walk(chat):
            if isinstance(node, ast.Call):
                self.assertFalse(any(keyword.arg == "skip_special" and isinstance(keyword.value, ast.Constant)
                                     and keyword.value.value is False for keyword in node.keywords))
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "response" for target in node.targets):
                self.assertFalse(any(isinstance(value, ast.Constant) and value.value == "?" for value in ast.walk(node.value)))


if __name__ == "__main__":
    unittest.main()
