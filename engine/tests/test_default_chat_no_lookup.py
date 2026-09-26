"""Source contract for native chat without answer-key storage.

No brain is created or trained here. End-to-end recall belongs to the next
authorized ground-up Build/training acceptance run.
"""

import inspect
import sys
import unittest
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig


class DefaultChatNoLookupTests(unittest.TestCase):
    def test_native_configuration_has_no_imported_foundation_selector(self):
        fields = OmniConfig.__dataclass_fields__
        self.assertEqual(fields["origin_kind"].default, "ground-up")
        self.assertNotIn("foundation_model_id", fields)
        self.assertFalse(hasattr(AdaptiveBrain, "_create_foundation_cortex"))

    def test_chat_routes_recall_through_shared_substrate(self):
        source = inspect.getsource(AdaptiveBrain.chat)
        self.assertIn("self.memory.recall_vector(", source)
        self.assertIn("self._idea_model_vector(recalled_vector)", source)
        self.assertIn("memory_bias=internal_memory", source)
        self.assertNotIn("sequence_memory", source)
        self.assertNotIn("exact_response", source)
        self.assertNotIn("foundation_cortex", source)

    def test_each_turn_has_fast_learning_and_optional_durable_slow_replay(self):
        source = inspect.getsource(AdaptiveBrain.chat)
        self.assertIn("return self.learn_experience(", source)
        self.assertIn("self._enqueue_chat_slow_learning(", source)
        self.assertIn("self._optimize_experience(", source)
        self.assertNotIn("_learn_sequence_associations", source)
        self.assertNotIn("train_dialogue_adapter", source)

    def test_typed_learning_changes_native_cortex_not_a_separate_answer_table(self):
        source = inspect.getsource(AdaptiveBrain._apply_supervised_dialogue)
        self.assertIn("self._optimize_dialogue_pair(", source)
        self.assertIn("self._snapshot_slow_transaction_state()", source)
        self.assertNotIn("sequence_memory", source)
        self.assertNotIn("foundation", source)


if __name__ == "__main__":
    unittest.main()
