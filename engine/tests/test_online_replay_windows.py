"""Pure coverage/scheduling checks; no brain or model is constructed."""
import unittest

from omni_core.online_replay import initial_replay_cursor, replay_window, rehearsal_due
from omni_core.tokenizer import ByteTokenizer
from omni_core.recent_token_activity import RecentTokenActivity
from omni_core.chat_steering import mark_no_reply_turn, validate_no_reply_turn


class OnlineReplayWindows(unittest.TestCase):
    def test_changed_window_sizes_preserve_every_unicode_target_once(self):
        text, tokenizer = "Aé🙂文z", ByteTokenizer()
        cursor, targets, windows = initial_replay_cursor(), [], 0
        while True:
            values, cursor, done = replay_window(text, cursor, (5, 2, 3)[windows % 3], tokenizer)
            targets.extend(values[1:])
            windows += 1
            if done:
                break
        expected = [value + tokenizer.byte_offset for value in text.encode("utf-8")] + [tokenizer.eos_id]
        self.assertEqual(targets, expected)
        self.assertGreater(windows, 1)

    def test_empty_source_has_real_bos_eos_without_manufactured_words(self):
        ids, cursor, done = replay_window("", None, 2, ByteTokenizer())
        self.assertEqual(ids, [ByteTokenizer.bos_id, ByteTokenizer.eos_id])
        self.assertTrue(done)
        self.assertEqual(cursor["phase"], "done")

    def test_cursor_cannot_hide_invalid_position(self):
        with self.assertRaises(ValueError):
            replay_window("a", {"phase": "text", "character": 0, "byte": 3}, 2, ByteTokenizer())

    def test_single_record_rehearses_on_committed_tail_progress(self):
        cadence = {"mode": "finite-midpoint", "middleWave": None, "checkpointRecords": 32}
        self.assertTrue(rehearsal_due(cadence, committed_records=0, previous_records=0,
            committed_windows=2, active_window=True, last_middle_wave=0, periodic_due=False))
        self.assertFalse(rehearsal_due(cadence, committed_records=0, previous_records=0,
            committed_windows=2, active_window=True, last_middle_wave=2, periodic_due=False))
        self.assertFalse(rehearsal_due(cadence, committed_records=1, previous_records=0,
            committed_windows=1, active_window=False, last_middle_wave=0, periodic_due=False))

    def test_pending_cortex_keeps_weak_old_raw_activity_until_handoff(self):
        tokens = [4, 5, 6, 7]
        activity = RecentTokenActivity([{"tokenCount": 2, "afterimageId": "older"},
                                        {"tokenCount": 2, "afterimageId": "current"}])
        self.assertEqual(activity.cool(tokens, [], 10, protected_episode_ids={"older"}), tokens)
        self.assertEqual(activity.cool(tokens, [], 10), [6, 7])

    def test_no_reply_does_not_rewrite_already_observed_human(self):
        human = {"content": "observed input", "input_accepted_before_reply": True}
        original = dict(human)
        assistant, receipt = {"content": ""}, {}
        mark_no_reply_turn(human, assistant, receipt)
        trace = {"input_accepted_before_reply": True, "generation_stop_reason": "no-reply",
                 "generation_no_reply_reason": "no-decoded-text", "generation_printable_text_characters": 0,
                 "generated_token_count": 1, "generation_decoder_stop_reason": "eos"}
        self.assertEqual(human, original)
        self.assertTrue(validate_no_reply_turn(human, assistant, trace, receipt))


if __name__ == "__main__":
    unittest.main()
