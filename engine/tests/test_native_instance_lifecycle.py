"""Supported exact native suspension; no corpus or capability training.

The cold proof uses a private copy of an already initialized genuine native
origin. It is opt-in so ordinary CI does not run model creation/training.
"""
import hashlib
import gc
import os
import shutil
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core import AdaptiveBrain
from omni_core import OmniConfig
from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.chat_tool_observation import ChatToolObservationInbox
from omni_core.offload import ResourcePolicy
from worker import Worker, RpcFault


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeLifecycleAdmissionTests(unittest.TestCase):
    def test_selected_best_work_replay_resumes_without_repeating_private_search(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="omni-visible-cursor-") as folder, \
                patch.object(ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024), \
                patch.object(AdaptiveBrain, "_select_structured_actions", return_value=({"talk": 1., "ponder": 0.}, [])):
            brain = AdaptiveBrain("visible", Path(folder), OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1,
                n_layers=1, d_ff=16, max_seq_len=32, working_memory_slots=4, vsa_dim=16, router_neurons=4,
                modality_channels=2, vision_enabled=False, image_enabled=False, audio_enabled=False, video_enabled=False,
                online_learning=False, best_work_enabled=True, best_work_exploration_branches=2))
            store = ExecutionCheckpointStore(brain.engine_path, brain.brain_id, policy=brain.resource_policy)
            events, text, sequences = [], [], []
            inbox = ChatToolObservationInbox("visible", "selected-turn", lambda _size: True, wait_seconds=0)
            def emit(kind, value):
                sequences.append(kind)
                if kind == "token": text.append(value["delta"])
                events.append((kind, value))
            def suspend(context, cursor, continuation):
                saved = store.save(brain, context, cursor, phase=continuation["phase"],
                    worker_state={"brainId": "visible", "turnId": "selected-turn", "lastStreamSequence": len(sequences) - 1,
                        "publishedText": "".join(text), "actions": [], "inbox": inbox.suspension_state()},
                    journal=context["before_parameters"], continuation=continuation)
                return {"suspended": True, "resume": saved["frontier"]}
            reader = None
            try:
                paused = brain.chat("Hello", max_new_tokens=7, seed=719, turn_id="selected-turn", defer_slow_learning=True,
                    cancel_check=lambda: False, stream_callback=emit, suspension_check=lambda: bool(text), suspend_callback=suspend)
                self.assertTrue(paused["suspended"])
                saved, payload, reader = store.restore(brain, paused["resume"]["checkpointId"])
                self.assertEqual(saved["phase"], "selected-replay")
                with patch("omni_core.best_work.run_best_work", side_effect=AssertionError("selected private search repeated")):
                    result = brain.resume_chat(payload, stream_callback=emit, cancel_check=lambda: False)
                self.assertEqual("".join(text), result["text"])
                self.assertTrue(result["turnCommitted"])
                self.assertFalse(result["trace"]["single_pass_generation"])
            finally:
                if reader is not None: reader.close()
                inbox.close(); brain.close()

    def test_unloaded_read_and_stop_control_do_not_construct_or_load_a_brain(self):
        with tempfile.TemporaryDirectory(prefix="omni-lifecycle-metadata-") as folder:
            root = Path(folder); (root / "engine").mkdir()
            (root / "engine/brain.json").write_text("{}")
            worker = Worker()
            try:
                params = {"brainId": "metadata", "storagePath": str(root)}
                with patch("worker.AdaptiveBrain.load", side_effect=AssertionError("implicit neural load")):
                    status = worker.instance_status(params)
                    self.assertFalse(status["learnedRuntimeResident"])
                    self.assertEqual(status["minimumContextTokens"], 0)
                    with self.assertRaisesRegex(RpcFault, "Explicit Load"):
                        worker._get(params)
                    self.assertEqual(worker.request_chat_suspension({"brainId": "metadata", "streamId": "old"}), {"requested": False})
                self.assertEqual(worker.brains, {})
            finally:
                worker.shutdown({}, None)


@unittest.skipUnless(os.environ.get("OMNI_NATIVE_LIFECYCLE_FIXTURE"), "requires an explicit existing genuine native fixture; never initializes one")
class NativeLifecycleColdProof(unittest.TestCase):
    def test_native_worker_chat_close_reload_resume_and_stop_tombstone(self):
        torch.set_num_threads(1)
        source = Path(os.environ["OMNI_NATIVE_LIFECYCLE_FIXTURE"]).resolve()
        self.assertTrue((source / "engine/brain.json").is_file())
        source_hash = digest(source / "engine/brain.json")
        with tempfile.TemporaryDirectory(prefix="omni-native-lifecycle-proof-") as folder, \
                patch.object(ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024), \
                patch.object(AdaptiveBrain, "_select_structured_actions", return_value=({"talk": 1., "ponder": 0.}, [])):
            # This structural fixture fixes only action selection to Talk, so
            # output/RNG/cache checks cannot hang on an untrained action. All
            # neural decoding, experience changes, saves and reloads are real.
            reference_path, resumed_path = Path(folder) / "reference", Path(folder) / "resumed"
            shutil.copytree(source, reference_path)
            shutil.copytree(source, resumed_path)
            import json
            identifier = json.loads((source / "engine/brain.json").read_text())["brain_id"]
            params = {"brainId": identifier, "input": "Hello", "streamId": "same-native-turn",
                "seed": 719, "maxNewTokens": 7, "toolSchemas": [], "toolObservationWaitMs": 0,
                "onlineLearning": False}
            expected = None
            for root, should_suspend in ((reference_path, False), (resumed_path, True)):
                worker = Worker(); events = []; worker.notify = lambda kind, **kw: events.append((kind, kw))
                owned = {**params, "storagePath": str(root)}
                try:
                    worker.load(owned, "load")
                    brain = worker.brains[identifier]
                    before_count = brain.counters["experiences"]
                    before_inference = brain.counters["inference_count"]
                    before_origin = digest(root / "engine/origin/brain.json")
                    actual_notify = worker.notify
                    requested = []
                    def notify(kind, **kw):
                        actual_notify(kind, **kw)
                        if should_suspend and kind == "chat-token" and kw["data"]["delta"] and not requested:
                            requested.append(worker.dispatch_control({"jsonrpc": "2.0", "id": "pause-control",
                                "method": "suspend_chat", "params": {"brainId": identifier, "streamId": params["streamId"]}}))
                    worker.notify = notify
                    result = worker.chat(owned, "chat-request")
                    if not should_suspend:
                        expected = result
                        self.assertTrue(result["turnCommitted"])
                    else:
                        self.assertTrue(requested[0]["result"]["requested"])
                        self.assertTrue(result["suspended"])
                        checkpoint = result["resume"]["checkpointId"]
                        prefix = "".join(kw["data"]["delta"] for kind, kw in events if kind == "chat-token")
                        self.assertTrue(prefix)
                        self.assertEqual(brain.counters["inference_count"], before_inference)
                        self.assertEqual(brain.counters["experiences"], before_count + 1)
                        self.assertEqual(len([message for message in brain.messages if message.get("turn_id") == params["streamId"]]), 1)
                        prior_owner = weakref.ref(brain)
                        del brain
                        receipt = worker.suspend_instance({**owned, "expectedCheckpointId": checkpoint}, "unload")
                        self.assertTrue(receipt["unloaded"])
                        self.assertFalse(receipt["learnedRuntimeResident"])
                        self.assertNotIn(identifier, worker.brains)
                        self.assertNotIn(identifier, worker._brain_leases)
                        gc.collect()
                        self.assertIsNone(prior_owner(), "released worker still retains the learned native owner")
                        worker.shutdown({}, None)
                        worker = Worker(); worker.notify = lambda kind, **kw: events.append((kind, kw))
                        cold = worker.instance_status(owned)
                        self.assertEqual(cold["presentation"]["publishedText"], prefix)
                        self.assertEqual(cold["resume"]["checkpointId"], checkpoint)
                        self.assertFalse(cold["learnedRuntimeResident"])
                        with self.assertRaisesRegex(RpcFault, "Explicit Load"):
                            worker.resume_chat({"brainId": identifier, "storagePath": str(root), "checkpointId": checkpoint}, "resume-unloaded")
                        worker.load({"brainId": identifier, "storagePath": str(root)}, "load-again")
                        result = worker.resume_chat({"brainId": identifier, "storagePath": str(root),
                            "checkpointId": checkpoint, "streamId": params["streamId"]}, "resume-request")
                        self.assertTrue(result["turnCommitted"])
                        self.assertEqual(result["text"], expected["text"])
                        self.assertEqual(result["trace"]["generated_token_count"], expected["trace"]["generated_token_count"])
                        self.assertEqual("".join(kw["data"]["delta"] for kind, kw in events if kind == "chat-token"), result["text"])
                        sequences = [kw["sequence"] for kind, kw in events if kind.startswith("chat-")]
                        self.assertEqual(sequences, sorted(set(sequences)))
                        self.assertEqual(worker.brains[identifier].counters["experiences"], before_count + 1 + int(bool(result["text"].strip())))
                        self.assertEqual(worker.brains[identifier].counters["inference_count"], before_inference + 1)
                        self.assertIsNone(worker.instance_status(owned).get("resume"))
                        # Lost final ACK cannot emit the answer or an effect again.
                        event_count = len(events)
                        duplicate = worker.resume_chat({"brainId": identifier, "storagePath": str(root),
                            "checkpointId": checkpoint, "streamId": params["streamId"]}, "lost-ack")
                        self.assertTrue(duplicate["idempotentCompletion"])
                        self.assertEqual(duplicate["actions"], [])
                        self.assertEqual(len(events), event_count)
                        # A second real admitted turn is paused, then stopped
                        # while unloaded. Stop does not erase its human row or
                        # fast learning, and a tombstone cannot be resumed.
                        stop_events = []
                        def pause_next(kind, **kw):
                            events.append((kind, kw))
                            if kind == "chat-token" and kw["data"]["delta"] and not stop_events:
                                stop_events.append(worker.request_chat_suspension({"brainId": identifier, "streamId": "stop-native-turn"}))
                        worker.notify = pause_next
                        paused = worker.chat({**owned, "streamId": "stop-native-turn", "input": "Hi", "maxNewTokens": 4}, "stop-proof")
                        self.assertTrue(paused["suspended"])
                        stop_id = paused["resume"]["checkpointId"]
                        worker.suspend_instance({**owned, "expectedCheckpointId": stop_id}, "unload-stop")
                        stopped = worker.stop_suspended_chat({"brainId": identifier, "storagePath": str(root),
                            "checkpointId": stop_id, "turnId": "stop-native-turn"}, "explicit-stop")
                        self.assertTrue(stopped["stopped"])
                        self.assertTrue(stopped["inputReceiptRetained"])
                        self.assertFalse(worker.instance_status(owned)["learnedRuntimeResident"])
                        self.assertIsNone(worker.instance_status(owned).get("resume"))
                    self.assertEqual(digest(root / "engine/origin/brain.json"), before_origin)
                finally:
                    worker.shutdown({}, None)
            self.assertEqual(digest(source / "engine/brain.json"), source_hash)


if __name__ == "__main__": unittest.main()
