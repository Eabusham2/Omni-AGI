"""Worker control/session fixtures only: no AdaptiveBrain is constructed."""
import json
import sys
import tempfile
import threading
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worker import DeferredEventLog, InlineGeneration, RpcFault, Worker


class InlineControlTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.worker = Worker.__new__(Worker)
        self.worker.worker_role = "neural"
        self.worker._inline_lock = threading.RLock()
        self.worker._cooperative_cancel = threading.Event()
        self.worker._inline_generations = {}
        self.notifications = []
        self.worker.notify = lambda *args, **kwargs: self.notifications.append((args, kwargs))
        self.action_id = "a" * 32
        self.params = {"modality": "image", "conceptIds": ["fixture"]}
        staging = Path(self.temporary.name) / ".inline-imagination" / self.action_id
        staging.mkdir(parents=True)
        self.record = InlineGeneration("brain", self.action_id, "turn", self.worker._inline_signature(self.params), staging, DeferredEventLog(), snapshot_started=True)
        self.worker._inline_generations[("brain", self.action_id)] = self.record

    def control(self, **changes):
        params = {"brainId": "brain", "streamId": "turn", "neuralActionId": self.action_id, **changes}
        return self.worker.dispatch_control({"jsonrpc": "2.0", "id": "control", "method": "cancel_inline_generation", "params": params})

    def test_preparation_cancel_is_pending_until_actual_cleanup_and_never_cancels_chat(self):
        response = self.control()
        self.assertEqual(response["result"], {"requested": True, "acknowledged": False})
        self.assertTrue(self.record.cancelled)
        self.assertTrue(self.record.first_preview.is_set())
        self.assertFalse(self.worker._cooperative_cancel.is_set())
        self.assertTrue(self.record.staging_root.exists())
        self.record.finished.set()
        self.worker._acknowledge_inline_cancellation(self.record)
        self.assertFalse(self.record.staging_root.exists())
        self.assertEqual(self.notifications[0][1]["data"], {"acknowledged": True, "cleanupCompleted": True})
        self.assertEqual(self.notifications[0][1]["stream_id"], "turn")
        self.worker._acknowledge_inline_cancellation(self.record)
        self.assertEqual(len(self.notifications), 1)

    def test_running_future_is_not_acknowledged_or_abandoned_before_it_finishes(self):
        future = Future()
        future.set_running_or_notify_cancel()
        self.record.future = future
        self.assertFalse(self.control()["result"]["acknowledged"])
        self.assertFalse(future.done())
        future.set_result({})
        self.record.finished.set()
        self.assertTrue(self.control()["result"]["acknowledged"])
        self.assertFalse(self.worker._cooperative_cancel.is_set())

    def test_mismatched_brain_turn_or_action_never_changes_the_record(self):
        for changes in [{"brainId": "other"}, {"streamId": "other"}, {"neuralActionId": "b" * 32}]:
            with self.assertRaises(RpcFault):
                self.control(**changes)
        self.assertFalse(self.record.cancelled)
        self.assertFalse(self.record.first_preview.is_set())

    def test_cancelled_inline_record_cannot_fall_through_to_regeneration(self):
        self.control()
        with self.assertRaises(RpcFault) as raised:
            self.worker._claim_inline_generation(SimpleNamespace(brain_id="brain"), {**self.params, "neuralActionId": self.action_id}, "job")
        self.assertEqual(raised.exception.code, -32800)

    def test_cleanup_failure_is_not_falsely_acknowledged(self):
        self.record.finished.set()
        self.worker._remove_inline_root = lambda _record: None
        self.assertFalse(self.control()["result"]["acknowledged"])
        self.assertEqual(self.notifications, [])

    def test_publication_boundary_rejects_late_control_without_touching_artifact(self):
        self.record.publication_started = True
        with self.assertRaises(RpcFault):
            self.control()
        self.assertFalse(self.record.cancelled)
        self.assertTrue(self.record.staging_root.exists())

    def test_ownership_is_reserved_before_snapshot_preparation_and_can_cancel_without_a_model(self):
        self.worker._inline_executor_closed = False
        action_id = "c" * 32
        def cancel_preparation(record):
            response = self.worker.dispatch_control({"jsonrpc": "2.0", "id": "early", "method": "cancel_inline_generation",
                "params": {"brainId": "brain", "streamId": "second-turn", "neuralActionId": action_id}})
            self.assertFalse(response["result"]["acknowledged"])
        brain = SimpleNamespace(brain_id="brain", engine_path=Path(self.temporary.name) / "engine")
        record = self.worker._start_inline_generation(brain, action_id, "second-turn", {
            "kind": "imagine", "toolId": "modality.imagine", "action": "generate", "arguments": self.params
        }, lambda *_args: self.fail("cancelled preparation emitted a preview"), cancel_preparation)
        self.assertTrue(record.cancelled)
        self.assertTrue(record.finished.is_set())
        self.assertIsNone(record.future)
        self.assertFalse(record.staging_root.exists())
        self.assertFalse(self.worker._cooperative_cancel.is_set())


if __name__ == "__main__":
    unittest.main()
