"""Actual method admission fixtures without any native model or forward."""

import hashlib
import unittest
from types import SimpleNamespace

from omni_core.brain import AdaptiveBrain
from omni_core.offload import NeuralStateResourcePause
from omni_core.tokenizer import ByteTokenizer


class CurrentChatInputCapacityTests(unittest.TestCase):
    def test_oversized_current_message_blocks_before_recall_or_token_allocation(self):
        # Deliberately has no model, memory, tokenizer, counters, or persistence.
        state = SimpleNamespace(config=SimpleNamespace(max_seq_len=8),
                                _validated_chat_turn_id=AdaptiveBrain._validated_chat_turn_id)
        with self.assertRaisesRegex(NeuralStateResourcePause, "blocked without truncation"):
            AdaptiveBrain.chat(state, "a complete oversized current message")

    def test_helper_blocks_current_input_instead_of_suffix_cropping(self):
        state = SimpleNamespace(config=SimpleNamespace(max_seq_len=8),
                                tokenizer=ByteTokenizer(), recent_token_context=[])
        with self.assertRaisesRegex(NeuralStateResourcePause, "blocked without truncation"):
            AdaptiveBrain._prompt_with_recent_context(state, "中文🙂")

    def test_full_current_input_fits_with_old_prompt_words_yielding_space(self):
        tokenizer = ByteTokenizer()
        human = "中文🙂"
        payload = tokenizer.encode(human)
        recent = tokenizer.dialogue("previous experience", "previous response")[1:]
        state = SimpleNamespace(config=SimpleNamespace(max_seq_len=len(payload) + 3),
                                tokenizer=tokenizer, recent_token_context=list(recent))
        prompt, history = AdaptiveBrain._prompt_with_recent_context(state, human)
        self.assertEqual(prompt, [tokenizer.bos_id, tokenizer.human_id, *payload, tokenizer.brain_id])
        self.assertEqual(history, [])
        # Selection does not erase learned state or old transient bookkeeping;
        # existing lifecycle admission/settling remains the authority.
        self.assertEqual(state.recent_token_context, recent)

    def test_non_scalar_input_fails_before_any_neural_work(self):
        state = SimpleNamespace(config=SimpleNamespace(max_seq_len=100))
        with self.assertRaisesRegex(ValueError, "Unicode scalar"):
            AdaptiveBrain.chat(state, "\ud800")

    def test_completed_exact_retry_remains_idempotent_after_context_reduction(self):
        text = "An already completed long human input 中文🙂" * 1000
        receipt = {"turnId": "completed", "inputSha256": hashlib.sha256(text.encode()).hexdigest()}
        state = SimpleNamespace(config=SimpleNamespace(max_seq_len=8), completed_chat_turns=[receipt],
                                _validated_chat_turn_id=AdaptiveBrain._validated_chat_turn_id,
                                _completed_chat_result=lambda value: {"receipt": value})
        self.assertEqual(AdaptiveBrain.chat(state, text, turn_id="completed"), {"receipt": receipt})
        with self.assertRaises(NeuralStateResourcePause):
            AdaptiveBrain.chat(state, text + "changed", turn_id="completed")


if __name__ == "__main__":
    unittest.main()
