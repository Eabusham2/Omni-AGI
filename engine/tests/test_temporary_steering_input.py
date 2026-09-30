"""Pure input/hash/role/admission/source fixtures; no neural constructors or runs."""
import hashlib
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.offload import NeuralStateResourcePause
from omni_core.tokenizer import ByteTokenizer
from omni_core.temporary_steering_context import temporary_steering_inputs, prompt_with_temporary_user_inputs
from worker import Worker


def descriptor(text="The actual unfinished user request", epoch=4):
    return {"format": "omni-temporary-steering-input-v1", "brainId": "brain", "successorTurnId": "new",
        "attentionEpoch": epoch, "inputs": [{"turnId": "old", "content": text,
            "inputSha256": hashlib.sha256(text.encode()).hexdigest()}]}


class TemporarySteeringInputTests(unittest.TestCase):
    def setUp(self):
        self.policy = SimpleNamespace(status=lambda **_kw: {"memoryPressure": False})

    def bind(self, value, epoch=4):
        return temporary_steering_inputs(value, brain_id="brain", turn_id="new", attention_epoch=epoch, policy=self.policy)

    def test_exact_current_request_identity_hash_and_audit_without_retaining_raw_history(self):
        text = "Original current human input العربية"
        inputs, audit = self.bind(descriptor(text))
        self.assertEqual(inputs, [text])
        self.assertNotIn("content", audit["inputs"][0])
        self.assertFalse(audit["durablePredecessorLearning"])
        self.assertFalse(audit["longTermHistoryRead"])

    def test_fresh_epoch_discards_already_queued_temporary_input(self):
        inputs, audit = self.bind(descriptor(), epoch=5)
        self.assertEqual(inputs, [])
        self.assertTrue(audit["freshAttentionDiscarded"])

    def test_changed_input_or_brain_successor_identity_is_rejected(self):
        for field, replacement in (("brainId", "other"), ("successorTurnId", "other"), ("attentionEpoch", True)):
            value = descriptor()
            value[field] = replacement
            with self.assertRaises(ValueError): self.bind(value)
        value = descriptor()
        value["inputs"][0]["content"] = "not the bound actual human input"
        with self.assertRaises(ValueError): self.bind(value)

    def test_prompt_has_actual_human_role_segments_and_no_synthesized_direction(self):
        tokenizer = ByteTokenizer()
        prior, current = "Use the quoted data 中文", "Change only its order"
        recent = tokenizer.dialogue("previous saved turn", "previous saved output")
        result, retained = prompt_with_temporary_user_inputs(tokenizer, current, [prior], recent, capacity=256, policy=self.policy)
        expected_current = [259, *tokenizer.encode(prior), 259, *tokenizer.encode(current), 260]
        self.assertEqual(result, [1, *retained, *expected_current])
        self.assertEqual(result[-len(expected_current):], expected_current)

    def test_token_minimum_is_admitted_before_encoding_any_payload(self):
        policy = SimpleNamespace(status=lambda **_kw: {"memoryPressure": True})
        with patch.object(ByteTokenizer, "encode", side_effect=AssertionError("unadmitted allocation")):
            with self.assertRaises(NeuralStateResourcePause):
                prompt_with_temporary_user_inputs(ByteTokenizer(), "new", ["old"], [], capacity=64, policy=policy)

    def test_overflow_never_suffix_truncates_the_unfinished_input_or_successor(self):
        with self.assertRaises(NeuralStateResourcePause) as raised:
            prompt_with_temporary_user_inputs(ByteTokenizer(), "new direction", ["old important direction"], [], capacity=8, policy=self.policy)
        self.assertFalse(raised.exception.status["inputTruncated"])
        self.assertFalse(raised.exception.status["temporaryInputDiscarded"])

    def test_actual_brain_prompt_method_uses_transient_payload_without_mutating_saved_context(self):
        brain = SimpleNamespace(config=SimpleNamespace(max_seq_len=128), tokenizer=ByteTokenizer(),
            recent_token_context=[259, 100, 260, 101, 2], resource_policy=self.policy)
        saved = list(brain.recent_token_context)
        result, _ = AdaptiveBrain._prompt_with_recent_context(brain, "new", ["old"])
        self.assertEqual(result[-9:], [259, 114, 111, 103, 259, 113, 104, 122, 260])
        self.assertEqual(brain.recent_token_context, saved)
        self.assertEqual(set(vars(brain)), {"config", "tokenizer", "recent_token_context", "resource_policy"})

    def test_production_worker_and_chat_signature_forward_private_payload_not_a_dead_helper(self):
        self.assertIn("temporary_steering_context=params.get(\"temporarySteeringContext\")", inspect.getsource(Worker._chat))
        self.assertIn("temporary_steering_context", inspect.signature(AdaptiveBrain.chat).parameters)
        source = inspect.getsource(AdaptiveBrain.chat)
        self.assertIn("clean, temporary_user_inputs", source)
        self.assertIn("\"temporary_steering_input\": temporary_input_audit", source)


if __name__ == "__main__": unittest.main()
