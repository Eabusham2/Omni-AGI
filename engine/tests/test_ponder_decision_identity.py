"""Per-prefix internal computation identity, with no brain/model instance."""
import unittest

from omni_core.native_action_protocol import NativeActionEmissionLedger


class PonderDecisionIdentityTests(unittest.TestCase):
    def test_new_prefix_can_select_ponder_again_but_exact_replay_is_idempotent(self):
        ledger = NativeActionEmissionLedger("turn")
        action = {"kind": "ponder", "arguments": {"organic": True}}
        first = ledger.register(action, step=3, phase="mid-generation")
        self.assertIsNone(ledger.register(action, step=3, phase="mid-generation"))
        second = ledger.register(action, step=4, phase="mid-generation")
        self.assertNotEqual(first["actionId"], second["actionId"])
        replay = NativeActionEmissionLedger("turn")
        self.assertEqual(first["actionId"], replay.register(action, step=3, phase="mid-generation")["actionId"])

    def test_repeated_external_effect_stays_suppressed_across_prefixes(self):
        ledger = NativeActionEmissionLedger("turn")
        action = {"kind": "tool", "toolId": "web.search", "action": "search",
                  "arguments": {"query": "actual native choice"}}
        self.assertIsNotNone(ledger.register(action, step=3, phase="mid-generation"))
        self.assertIsNone(ledger.register(action, step=4, phase="mid-generation"))

    def test_kind_is_not_inferred_from_words_or_tool_name(self):
        ledger = NativeActionEmissionLedger("turn")
        action = {"kind": "tool", "toolId": "mcp.ponder", "action": "call",
                  "arguments": {"query": "ponder again"}}
        self.assertIsNotNone(ledger.register(action, step=3, phase="mid-generation"))
        self.assertIsNone(ledger.register(action, step=4, phase="mid-generation"))


if __name__ == "__main__":
    unittest.main()
