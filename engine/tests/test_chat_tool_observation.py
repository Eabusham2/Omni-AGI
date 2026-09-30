"""Inbox and scripted decoding control only; no neural model or training."""
import hashlib
import json
import threading
import unittest
from types import SimpleNamespace

import torch
from omni_core.chat_tool_observation import ChatToolObservationInbox, FORMAT
from omni_core.offload import NeuralStateResourcePause
from omni_core.model import OmniDecoder
from engine.tests.test_native_text_conditioning_boundary import ScriptedTokenLoop


def observation(output="actual external 中文 result"):
    value = {"format": FORMAT, "brainId": "brain", "turnId": "turn", "neuralActionId": "a" * 32,
        "actionEventId": "event", "executionId": "execution", "toolId": "web.search", "action": "search",
        "completedAt": "2026-09-30T12:00:00.000Z", "payloadJson": json.dumps({"outputPresent": True, "output": output}, ensure_ascii=False)}
    value["payloadSha256"] = hashlib.sha256(value["payloadJson"].encode()).hexdigest()
    fields = [value[name] for name in ("format", "brainId", "turnId", "neuralActionId", "actionEventId",
        "executionId", "toolId", "action", "completedAt", "payloadSha256")]
    value["observationId"] = hashlib.sha256("\x00".join(fields).encode()).hexdigest()
    return value


class ChatToolObservationTests(unittest.TestCase):
    def inbox(self, **kwargs):
        inbox = ChatToolObservationInbox("brain", "turn", kwargs.pop("admit", lambda _size: True), **kwargs)
        inbox.register("a" * 32, {"kind": "tool", "toolId": "web.search", "action": "search"})
        return inbox

    def test_only_actual_emitted_action_can_offer_exact_bound_evidence(self):
        inbox = self.inbox()
        value = observation()
        self.assertTrue(inbox.offer(value)["accepted"])
        self.assertTrue(inbox.offer(value)["duplicate"])
        self.assertEqual(inbox.drain(), [value])
        self.assertEqual(inbox.drain(), [])

    def test_wrong_turn_action_or_changed_payload_does_not_enter_native_work(self):
        for field, bad in (("turnId", "other"), ("neuralActionId", "b" * 32), ("payloadJson", "{}")):
            inbox, value = self.inbox(), observation()
            value[field] = bad
            with self.assertRaises(ValueError): inbox.offer(value)
            self.assertEqual(inbox.drain(), [])

    def test_closed_and_resource_pressure_keep_evidence_unused_not_truncated(self):
        inbox = self.inbox(admit=lambda _size: False)
        self.assertFalse(inbox.offer(observation())["accepted"])
        inbox.close()
        self.assertEqual(inbox.offer(observation())["reason"], "turn-output-closed")

    def test_accepted_decoded_rpc_lease_is_held_until_neural_consume_or_close(self):
        class Lease:
            released = 0
            def release(self): self.released += 1

        inbox, lease = self.inbox(), Lease()
        self.assertTrue(inbox.offer(observation(), lease)["accepted"])
        self.assertEqual(lease.released, 0)
        delivered = inbox.drain()
        self.assertEqual(lease.released, 0)
        delivered[0].release()
        delivered[0].release()
        self.assertEqual(lease.released, 1)

        another, pending_lease = self.inbox(), Lease()
        another.offer(observation(), pending_lease)
        another.close()
        self.assertEqual(pending_lease.released, 1)

    def test_nested_json_validation_uses_an_atomic_parse_reservation(self):
        events = []
        class ParseLease:
            def __enter__(self): events.append("enter")
            def __exit__(self, *_args): events.append("exit")
        inbox = self.inbox(reserve_parse=lambda amount, name:
            (events.append((amount, name)), ParseLease())[1])
        self.assertTrue(inbox.offer(observation({"nested": [1, 2, 3]}))["accepted"])
        self.assertEqual(events[1:], ["enter", "exit"])
        self.assertGreater(events[0][0], 0)

        def blocked(*_args):
            raise NeuralStateResourcePause("fixture resource pause", {"paused": True})
        refused = self.inbox(reserve_parse=blocked)
        self.assertEqual(refused.offer(observation())["reason"], "observation-resource-pressure")
        self.assertEqual(refused.drain(), [])

    def test_wait_can_receive_a_complete_result_without_a_fake_message(self):
        inbox = self.inbox(wait_seconds=.05)
        timer = threading.Timer(.002, lambda: inbox.offer(observation()))
        timer.start()
        try: self.assertEqual(inbox.wait_pending(), [observation()])
        finally: timer.join()

    def test_stop_or_timeout_never_manufactures_result_data(self):
        self.assertEqual(self.inbox(cancelled=lambda: True, wait_seconds=1000).wait_pending(), [])
        self.assertEqual(self.inbox(wait_seconds=0).wait_pending(), [])

    def test_live_conditioning_is_applied_before_forward_and_eos_not_just_queued(self):
        fixture = ScriptedTokenLoop(lambda _step, _bias: 2)
        prompt = torch.tensor([[1, 259, 100, 260]])
        def condition(_bias, _step):
            return {"memoryBias": torch.ones(1, 4), "observationBindings": [{"observationId": "actual"}]}
        generated, _ = OmniDecoder.generate(fixture, prompt, memory_bias=torch.zeros(1, 4), max_new_tokens=4,
            top_k=1, conditioning_callback=condition)
        self.assertEqual(fixture.calls[0][1][0, 0].item(), 1)
        self.assertEqual(fixture.last_generation_tool_observations[0]["nativeDecodingUsed"], True)
        self.assertEqual(generated[0, -1].item(), 2)

    def test_natural_eos_can_resume_under_actual_tool_evidence_without_spending_a_fake_token(self):
        fixture = ScriptedTokenLoop(lambda index, bias: 2 if index == 0 or index > 1 else ord("B") + 3)
        prompt = torch.tensor([[1, 259, 100, 260]])
        def condition(_bias, _step): return {}
        waits = []
        def ready(_bias, _step):
            waits.append(True)
            return {"memoryBias": torch.ones(1, 4), "observationBindings": [{"observationId": "actual"}]} if len(waits) == 1 else {}
        condition.wait_for_evidence = ready
        generated, entropies = OmniDecoder.generate(fixture, prompt, memory_bias=torch.zeros(1, 4), max_new_tokens=2,
            top_k=1, conditioning_callback=condition)
        self.assertEqual(generated[0, prompt.shape[1]:].tolist(), [ord("B") + 3, 2])
        self.assertEqual(len(entropies), 2)
        self.assertEqual(fixture.last_generation_tool_observations[0]["tokenOffset"], 0)

    def test_explicit_native_stop_never_waits_for_pending_tools(self):
        fixture = ScriptedTokenLoop(lambda *_: 2)
        def condition(_bias, _step): return {}
        condition.wait_for_evidence = lambda *_: self.fail("native Stop waited")
        prompt = torch.tensor([[1, 259, 100, 260]])
        generated, _ = OmniDecoder.generate(fixture, prompt, max_new_tokens=4, top_k=1,
            activity_callback=lambda *_: {"stop": True}, conditioning_callback=condition)
        self.assertTrue(torch.equal(generated, prompt))


if __name__ == "__main__": unittest.main()
