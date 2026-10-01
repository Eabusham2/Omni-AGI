"""Flags/session/dispatch fixtures only; no brain/model construction or forward."""
import hashlib
import json
import sys
import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import Future
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import ChatSteeringState, DeferredEventLog, InlineGeneration, RpcFault, Worker


class ApprovedInlineTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-ask-inline-")
        self.addCleanup(temporary.cleanup)
        self.engine = Path(temporary.name) / "engine"
        self.engine.mkdir()
        self.worker = Worker.__new__(Worker)
        self.worker.worker_role = "neural"
        self.worker._inline_lock = threading.RLock()
        self.worker._steering_lock = threading.RLock()
        self.worker._active_request_lock = threading.RLock()
        self.worker._cooperative_cancel = threading.Event()
        self.worker._inline_generations = {}
        self.worker._chat_steering = {}
        self.worker._inline_executor_closed = False
        self.worker.cancelled_jobs = set()
        self.notifications = []
        self.worker.notify = lambda *args, **kwargs: self.notifications.append((args, kwargs))
        self.policy = SimpleNamespace(status=lambda **_kwargs: {"memoryPressure": False}, reserve_ram=lambda *_args: nullcontext())
        self.brain = SimpleNamespace(brain_id="brain", engine_path=self.engine, resource_policy=self.policy,
            conversation=SimpleNamespace(summary=lambda: {}), config=SimpleNamespace(online_learning=True))
        self.worker.brains = {"brain": self.brain}
        self.worker._get = lambda _params: self.brain
        self.worker._restore_committed_brain_after_failed_chat = lambda *_args, **_kwargs: self.fail("fixture tried to construct/reload a brain")
        self.action_id = "a" * 32
        self.arguments = {"modality": "image", "conceptIds": ["fixture"]}
        self.action = {"kind": "imagine", "toolId": "modality.imagine", "action": "generate", "arguments": self.arguments}
        self.session = ChatSteeringState("brain", "turn", imagination_grant="ask")
        self.session.imagination_actions[self.action_id] = self.action
        self.worker._chat_steering[("brain", "turn")] = self.session
        self.record = InlineGeneration("brain", self.action_id, "turn", self.worker._inline_signature(self.arguments),
            self.engine / ".inline-imagination" / self.action_id, DeferredEventLog())
        self.worker._inline_generations[("brain", self.action_id)] = self.record
        self.execution_id = str(uuid.uuid4())

    def authorized_params(self, **argument_changes):
        arguments = {**self.arguments, "neuralActionId": self.action_id, "chatTurnId": "turn", **argument_changes}
        raw = json.dumps(arguments, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        directory = self.engine / "operational-tool-intents"
        directory.mkdir(exist_ok=True)
        path = directory / (self.execution_id + ".json")
        path.write_text(json.dumps({"format": "omni-authorized-tool-intent", "formatVersion": 1,
            "state": "authorized-before-side-effects", "id": self.execution_id, "brainId": "brain", "requestId": "turn",
            "toolId": "modality.imagine", "action": "generate", "argumentSha256": digest, "chatTurnId": "turn",
            "neuralActionId": self.action_id, "permission": "ask"}), encoding="utf-8")
        return {"brainId": "brain", "streamId": "turn", "neuralActionId": self.action_id, "executionId": self.execution_id,
            "requestId": "turn", "intentPath": str(path), "argumentsJson": raw, "argumentSha256": digest}

    def control(self, method, params):
        return self.worker.dispatch_control({"jsonrpc": "2.0", "id": "control", "method": method, "params": params})["result"]

    def test_authorization_is_flag_only_idempotent_and_cannot_do_snapshot_or_neural_work_on_reader(self):
        params = self.authorized_params()
        self.worker._start_inline_generation = lambda *_args: self.fail("reader performed a neural snapshot")
        self.worker._get = lambda *_args: self.fail("reader loaded a brain")
        first = self.control("authorize_inline_imagination", params)
        second = self.control("authorize_inline_imagination", params)
        self.assertEqual(first, second)
        self.assertTrue(first["accepted"])
        self.assertFalse(first["started"])
        self.assertEqual(self.session.approved_imagination, [self.action_id])
        self.assertIsNone(self.record.future)
        self.assertFalse(self.record.snapshot_started)
        self.assertFalse(self.worker._cooperative_cancel.is_set())

    def test_changed_operands_wrong_receipt_other_execution_and_off_grant_are_rejected_before_effects(self):
        with self.assertRaises(RpcFault): self.control("authorize_inline_imagination", self.authorized_params(conceptIds=["changed"]))
        params = self.authorized_params()
        with self.assertRaises(RpcFault): self.control("authorize_inline_imagination", {**params, "requestId": "sibling"})
        self.session.imagination_grant = "off"
        with self.assertRaises(RpcFault): self.control("authorize_inline_imagination", params)
        self.session.imagination_grant = "ask"
        self.control("authorize_inline_imagination", params)
        self.execution_id = str(uuid.uuid4())
        with self.assertRaises(RpcFault): self.control("authorize_inline_imagination", self.authorized_params())
        self.assertFalse(self.record.snapshot_started)

    def test_approval_without_the_actual_owned_pre_effect_receipt_cannot_bypass_ask(self):
        params = self.authorized_params()
        Path(params["intentPath"]).unlink()
        with self.assertRaises(RpcFault): self.control("authorize_inline_imagination", params)
        self.assertEqual(self.session.approved_imagination, [])

    def test_closed_steered_stopped_turns_do_not_begin_a_snapshot(self):
        params = self.authorized_params()
        self.session.requested.set()
        self.assertFalse(self.control("authorize_inline_imagination", params)["accepted"])
        self.session.requested.clear(); self.worker._cooperative_cancel.set()
        self.assertFalse(self.control("authorize_inline_imagination", params)["accepted"])
        self.worker._cooperative_cancel.clear(); self.session.generation_open = False
        self.assertFalse(self.control("authorize_inline_imagination", params)["accepted"])
        self.assertEqual(self.session.approved_imagination, [])

    def test_unstarted_ask_card_cancels_with_exact_cleanup_ack_without_cancelling_text(self):
        result = self.control("cancel_inline_generation", {"brainId": "brain", "streamId": "turn", "neuralActionId": self.action_id})
        self.assertTrue(result["acknowledged"])
        self.assertTrue(self.record.finished.is_set())
        self.assertFalse(self.worker._cooperative_cancel.is_set())
        with self.assertRaises(RpcFault): self.control("authorize_inline_imagination", self.authorized_params())

    def test_live_or_host_owned_job_has_no_five_minute_expiry_and_claim_never_waits_behind_decoder(self):
        self.record.created_at = time.monotonic() - 1000
        self.record.snapshot_started = True
        self.record.future = Future()
        self.record.future.set_running_or_notify_cancel()
        self.worker._cleanup_expired_inline_generations()
        self.assertIs(self.worker._inline_generations[("brain", self.action_id)], self.record)
        with self.assertRaises(RpcFault) as raised:
            self.worker._claim_inline_generation(self.brain, {**self.arguments, "neuralActionId": self.action_id}, "job")
        self.assertEqual(raised.exception.data, {"artifactPending": True})
        self.record.future.set_result({})
        self.record.finished.set()
        self.control("inline_generation_status", {"brainId": "brain", **self.arguments, "neuralActionId": self.action_id})
        self.worker._cleanup_expired_inline_generations()
        self.assertIs(self.worker._inline_generations[("brain", self.action_id)], self.record)

    def test_same_owned_ask_snapshot_starts_on_chat_safe_boundary_during_writing_and_committed_reply_returns_without_preview(self):
        starts = []
        def start(_brain, identifier, stream_id, _action, _preview, started):
            starts.append((identifier, stream_id)); self.record.snapshot_started = True
            self.record.future = Future()  # deliberately never ready: no decoder/preview/model runs
            started(self.record)
            return self.record
        self.worker._start_inline_generation = start
        def chat(_text, **kwargs):
            kwargs["stream_callback"]("action", {"actionId": self.action_id, "action": self.action})
            self.control("authorize_inline_imagination", self.authorized_params())
            self.assertEqual(starts, [])
            kwargs["chat_control_callback"]()
            self.assertEqual(starts, [(self.action_id, "turn")])
            kwargs["stream_callback"]("token", {"delta": "stub exact token"})
            return {"text": "stub exact token", "trace": {"id": "fixture-receipt"}, "turnCommitted": True}
        self.brain.chat = chat
        result = self.worker._chat({"brainId": "brain", "input": "actual human input", "streamId": "turn",
            "toolSchemas": [{"id": "modality.imagine", "actions": ["generate"], "grant": "ask"}]}, "request")
        self.assertEqual(result["text"], "stub exact token")
        self.assertFalse(self.record.future.done())
        self.assertFalse(self.record.first_preview.is_set())
        types = [entry[0][0] for entry in self.notifications]
        self.assertLess(types.index("inline-imagination-started"), types.index("chat-token"))
        self.assertEqual(types[-1], "brain-mutated")

    def test_serial_dispatch_wraps_loaded_operation_pressure_monitor_without_reader_neural_work(self):
        transitions = []
        @contextmanager
        def active(method):
            transitions.append(("enter", method))
            try: yield
            finally: transitions.append(("exit", method))
        self.policy.active_operation = active
        self.worker.methods = {"fixture-long-method": lambda _params, _request: {"actual": transitions[-1]}}
        self.worker._cleanup_expired_inline_generations = lambda: None
        response = self.worker.dispatch({"jsonrpc": "2.0", "id": "request", "method": "fixture-long-method", "params": {"brainId": "brain"}})
        self.assertEqual(response["result"], {"actual": ("enter", "fixture-long-method")})
        self.assertEqual(transitions, [("enter", "fixture-long-method"), ("exit", "fixture-long-method")])

    def test_success_route_rpc_gets_exact_cooperative_owner_callback_without_generic_failure_supervision(self):
        callbacks = []
        self.worker._active_request = ("learn_tool_route_outcome", "route-rpc")
        def learn(**kwargs):
            callbacks.append(kwargs["cancel_check"])
            self.assertFalse(kwargs["cancel_check"]())
            self.assertEqual(kwargs["outcome"], "success")
            return {"applied": False, "duplicate": True}
        self.brain.learn_tool_route_experience = learn
        self.brain.metrics = lambda: {}
        self.brain.summary = lambda: {}
        result = self.worker.learn_tool_route_outcome({"brainId": "brain", "eventId": str(uuid.uuid4()),
            "utterance": "actual human input", "toolId": "web.fetch", "action": "fetch", "outcome": "success", "arguments": {}}, "route-rpc")
        self.assertTrue(result["routeLearning"]["duplicate"])
        self.assertTrue(self.worker.request_cooperative_cancel())
        self.assertTrue(callbacks[0]())
        self.worker._active_request = ("chat", "successor")
        self.assertFalse(callbacks[0]())


if __name__ == "__main__": unittest.main()
