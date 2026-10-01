"""Typed observation inbox fixtures only; no brain/model or forward."""
import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.chat_tool_observation import ChatToolObservationInbox, FORMAT


class FailedToolObservationTests(unittest.TestCase):
    def inbox(self):
        value = ChatToolObservationInbox("brain", "turn", lambda _size: True, wait_seconds=0)
        value.register("a" * 32, {"kind": "tool", "toolId": "system.shell", "action": "run"})
        return value

    def observation(self, payload):
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        value = {"format": FORMAT, "brainId": "brain", "turnId": "turn", "neuralActionId": "a" * 32,
            "actionEventId": "event", "executionId": "receipt", "toolId": "system.shell", "action": "run",
            "completedAt": "2026-09-30T12:00:00.000Z", "payloadJson": raw, "payloadSha256": sha}
        fields = [FORMAT, "brain", "turn", value["neuralActionId"], "event", "receipt", "system.shell", "run", value["completedAt"], sha]
        value["observationId"] = hashlib.sha256("\0".join(fields).encode()).hexdigest()
        return value

    def test_actual_failed_attempt_is_queued_as_data_and_resolves_pending_eos_ownership_not_positive_supervision(self):
        inbox = self.inbox()
        payload = {"outputPresent": True, "output": {"exitCode": 7, "stderr": "actual fixture error"},
            "executionState": "failed", "dispatchStarted": True, "executionError": "actual nonzero exit"}
        event = self.observation(payload)
        ack = inbox.offer(event)
        self.assertTrue(ack["accepted"])
        self.assertEqual(inbox.awaiting, set())
        used = inbox.wait_pending()
        self.assertEqual(json.loads(used[0]["payloadJson"]), payload)
        self.assertNotIn("outcome", payload)

    def test_success_wrapper_stays_compatible_and_failure_without_output_keeps_actual_error(self):
        for payload in [{"outputPresent": True, "output": {"actual": True}},
            {"outputPresent": False, "executionState": "failed", "dispatchStarted": True, "executionError": "actual transport failure"}]:
            inbox = self.inbox(); event = self.observation(payload)
            self.assertTrue(inbox.offer(event)["accepted"])
            self.assertTrue(inbox.offer(event)["duplicate"])
            self.assertEqual(len(inbox.drain()), 1)

    def test_no_dispatch_or_forged_positive_status_is_rejected(self):
        for extra in [{"executionState": "failed", "dispatchStarted": False},
            {"executionState": "complete", "dispatchStarted": True},
            {"executionState": "failed", "dispatchStarted": True, "executionError": {"invented": True}}]:
            inbox = self.inbox()
            with self.assertRaises(ValueError): inbox.offer(self.observation({"outputPresent": False, **extra}))
            self.assertEqual(len(inbox.pending), 0)

    def test_missing_completion_timestamp_and_changed_payload_do_not_end_pending_real_tool(self):
        inbox = self.inbox(); event = self.observation({"outputPresent": False, "executionState": "failed", "dispatchStarted": True})
        with self.assertRaises(ValueError): inbox.offer({**event, "completedAt": ""})
        with self.assertRaises(ValueError): inbox.offer({**event, "payloadJson": '{"outputPresent":true,"output":"invented"}'})
        self.assertEqual(inbox.awaiting, {"a" * 32})


if __name__ == "__main__": unittest.main()
