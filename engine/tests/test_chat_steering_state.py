"""Pure completion/control fixtures; no brain/model is created or run."""
import sys
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.chat_steering import generation_completion, mark_interrupted_turn
from omni_core.brain import AdaptiveBrain
from worker import ChatSteeringState, RpcFault, Worker


class ChatSteeringTests(unittest.TestCase):
    def test_pre_neural_yield_does_not_construct_or_access_a_model(self):
        # An inert namespace intentionally has no decoder/parameters/memory.
        # The actual method must yield before any neural work can be touched.
        owner = SimpleNamespace(brain_id="brain", completed_chat_turns=[],
            config=SimpleNamespace(max_seq_len=256),
            _validated_chat_turn_id=AdaptiveBrain._validated_chat_turn_id)
        result = AdaptiveBrain.chat(owner, "original input", turn_id="old", steer_check=lambda: True)
        self.assertTrue(result["zeroTokenYield"])
        self.assertFalse(result["turnCommitted"])
        self.assertEqual(result["text"], "")
    def test_exact_emitted_prefix_has_no_trim_or_fabricated_ending(self):
        result = generation_completion([1, 2], lambda _ids: " actual prefix ", True)
        self.assertEqual(result, {"text": " actual prefix ", "steered": True, "nativeStopped": False, "zeroTokenYield": False})

    def test_private_candidate_and_empty_yield_never_manufacture_question_mark(self):
        self.assertEqual(generation_completion([1, 2], lambda _ids: "private answer", True, True)["text"], "")
        self.assertTrue(generation_completion([], lambda _ids: "", True)["zeroTokenYield"])

    def test_typed_disposition_preserves_exact_input_and_output(self):
        human, assistant, receipt = {"content": "original input"}, {"content": "actual prefix"}, {"turnId": "old"}
        mark_interrupted_turn(human, assistant, receipt)
        self.assertEqual(human["content"], "original input")
        self.assertEqual(assistant["content"], "actual prefix")
        self.assertEqual(receipt["generationEnd"], "steered")

    def test_native_stop_is_not_a_human_steer_and_has_no_fabricated_empty_answer(self):
        result = generation_completion([], lambda _ids: "", native_stopped=True)
        self.assertTrue(result["zeroTokenYield"])
        self.assertFalse(result["steered"])
        self.assertTrue(result["nativeStopped"])
        self.assertEqual(result["text"], "")

    def worker(self):
        worker = Worker.__new__(Worker)
        worker.worker_role = "neural"
        worker._steering_lock = threading.RLock()
        worker._chat_steering = {}
        worker._cooperative_cancel = threading.Event()
        return worker

    def steer(self, worker, **params):
        return worker.dispatch_control({"jsonrpc": "2.0", "id": "control", "method": "steer_chat",
            "params": {"brainId": "brain", "streamId": "old", "successorTurnId": "new", **params}})

    def test_control_can_reach_reserved_request_before_neural_dispatch_without_global_cancel(self):
        worker = self.worker()
        worker.reserve_chat_steering({"jsonrpc": "2.0", "id": "rpc-old", "method": "chat",
            "params": {"brainId": "brain", "streamId": "old"}})
        self.assertTrue(self.steer(worker)["result"]["warm"])
        worker._chat = lambda _params, _request, check: {"actualBoundaryStop": check()}
        result = worker.chat({"brainId": "brain", "streamId": "old"}, "rpc-old")
        self.assertTrue(result["actualBoundaryStop"])
        self.assertFalse(worker._cooperative_cancel.is_set())
        self.assertEqual(worker._chat_steering, {})

    def test_mismatched_ownership_does_not_interrupt_another_turn(self):
        worker = self.worker()
        session = ChatSteeringState("brain", "old")
        worker._chat_steering[("brain", "old")] = session
        for params in [{"brainId": "other"}, {"streamId": "other"}, {"successorTurnId": "old"}]:
            with self.assertRaises(RpcFault): self.steer(worker, **params)
        self.assertFalse(session.requested.is_set())

    def test_zero_yield_worker_path_does_not_reload_or_invent_committed_response(self):
        worker = self.worker()
        brain = SimpleNamespace(brain_id="brain", conversation=SimpleNamespace(summary=lambda: {}),
            resource_policy=SimpleNamespace(status=lambda **_: {"memoryPressure": False}, reserve_ram=lambda *_: nullcontext()),
            chat=lambda *_args, **_kwargs: {"steered": True, "zeroTokenYield": True, "turnCommitted": False})
        worker._get = lambda _params: brain
        worker._restore_committed_brain_after_failed_chat = lambda *_args, **_kwargs: self.fail("warm yield reloaded the brain")
        with self.assertRaises(RpcFault) as raised:
            worker._chat({"brainId": "brain", "input": "original input", "streamId": "old"}, "rpc-old", lambda: True)
        self.assertEqual(raised.exception.code, -32801)
        self.assertTrue(raised.exception.data["zeroTokenYield"])
        self.assertFalse(worker._cooperative_cancel.is_set())


if __name__ == "__main__": unittest.main()
