"""Protocol-only regional writer routing; no model construction or training."""
import copy
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from omni_core.chat_tool_observation import ChatToolObservationInbox
from worker import Worker, ChatSteeringState


class WorkerInheritedLifecycleTests(unittest.TestCase):
    def test_borrowed_decode_has_root_stream_controls_but_keeps_native_input_and_policy(self):
        with tempfile.TemporaryDirectory(prefix="omni-inherited-wire-") as folder:
            worker = Worker(); events = []
            worker.notify = lambda kind, **fields: events.append((kind, fields))
            policy = SimpleNamespace(status=lambda **_: {}, reserve_ram=lambda *_args: nullcontext())
            source = SimpleNamespace(brain_id="parent", config=SimpleNamespace(online_learning=True), resource_policy=policy,
                current_context={"tokenCount": 3}, conversation=SimpleNamespace(summary=lambda: {"headSequence": 1, "headSha256": "a" * 64}))
            root = SimpleNamespace(brain_id="merged", engine_path=Path(folder), resource_policy=policy)
            inbox = ChatToolObservationInbox("parent", "original-turn", lambda _size: True, wait_seconds=0)
            manifest = {"minimumContextTokens": 3, "worker": {"brainId": "parent", "turnId": "original-turn", "lastStreamSequence": 7,
                "publishedText": "Prior ", "actions": [], "inbox": inbox.suspension_state(), "priorLedgerHead": None}}
            payload = {"context": {"clean": "Hello", "normalized_tools": []}}
            execution = SimpleNamespace(brain=source, root=root, original_turn_id="original-turn", root_turn_id="ancestral-turn",
                outward_event=lambda _kind, value: copy.deepcopy(value))
            def resume(saved, **callbacks):
                self.assertIs(saved, payload)
                self.assertIsNotNone(worker._chat_steering[("merged", "ancestral-turn")].observation_inbox)
                callbacks["chat_control_callback"]()
                callbacks["stream_callback"]("token", {"delta": "next", "outputTokens": 1})
                callbacks["stream_callback"]("phase", {"phase": "reply-complete-learning", "replyComplete": True,
                    "turnCommitted": False, "learning": True, "saving": True})
                return {"text": "Prior next", "turnCommitted": True, "trace": {"id": "root-trace"}, "actions": []}
            source.resume_chat = resume
            worker._chat_steering[("merged", "ancestral-turn")] = ChatSteeringState("merged", "ancestral-turn", claimed=True)
            try:
                with patch.object(worker, "_get", side_effect=AssertionError("unrelated root decoder selected")):
                    result = worker._chat({"brainId": "merged", "streamId": "ancestral-turn", "onlineLearning": False, "toolSchemas": []},
                        "request", resume_payload=payload, resume_manifest=manifest, execution_session=execution)
                self.assertEqual(result["text"], "Prior next")
                self.assertTrue(source.config.online_learning)
                ordered = [(kind, value["brain_id"], value.get("stream_id"), value.get("sequence")) for kind, value in events]
                self.assertEqual(ordered[:2], [("chat-token", "merged", "ancestral-turn", 8), ("chat-phase", "merged", "ancestral-turn", 9)])
                self.assertEqual(ordered[-1][0:2], ("brain-mutated", "merged"))
            finally:
                inbox.close(); worker._chat_steering.clear(); worker.shutdown({}, None)

    def test_cold_inherited_stop_routes_to_metadata_only_lease_without_loading(self):
        worker = Worker()
        try:
            with patch.object(worker, "_get", side_effect=AssertionError("Stop implicitly loaded model")), \
                    patch.object(worker, "_owner_lease", return_value=(None, True)), \
                    patch.object(worker, "_release_owner_lease") as release, \
                    patch("omni_core.connected_execution.stop_unloaded_connected_execution", return_value={"stopped": True}) as stop:
                result = worker.stop_inherited_chat({"brainId": "merged", "storagePath": "/tmp/omni-inherited-stop-fixture", "executionId": "a" * 64})
                self.assertTrue(result["stopped"]); stop.assert_called_once(); release.assert_called_once_with("merged")
        finally:
            worker.shutdown({}, None)

    def test_normal_cold_stop_also_holds_the_instance_lease_without_loading_or_qualifying_old_execution(self):
        worker = Worker(); identifier = "c" * 32
        store = SimpleNamespace(current=lambda expected: {"checkpointId": expected}, finish=lambda expected, **_kwargs: {"checkpointId": expected})
        try:
            with patch.object(worker, "_get", side_effect=AssertionError("Stop implicitly loaded model")), \
                    patch.object(worker, "_owner_lease", return_value=(None, True)) as lease, \
                    patch.object(worker, "_release_owner_lease") as release, \
                    patch.object(worker, "_execution_store", return_value=store), \
                    patch("omni_core.execution_compatibility.require_execution_compatibility", side_effect=AssertionError("archival Stop requires no restore seal")):
                result = worker.stop_suspended_chat({"brainId": "owner", "storagePath": "/tmp/omni-native-stop-fixture", "checkpointId": identifier, "turnId": "old-turn"})
                self.assertEqual(result["checkpointId"], identifier)
                self.assertTrue(result["inputReceiptRetained"]); lease.assert_called_once(); release.assert_called_once_with("owner")
        finally:
            worker.shutdown({}, None)


if __name__ == "__main__": unittest.main()
