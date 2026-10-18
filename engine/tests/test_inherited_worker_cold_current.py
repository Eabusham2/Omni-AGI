"""Current sealed regional Worker continuation; no initialization/corpus run.

The supplied fixture must be the NEW owned capture made by
capture_connected_execution_fixture.py. Historical unsealed snapshots are
explicitly not accepted as if they contained current runtime evidence.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
from functools import wraps
import unittest
import weakref
from unittest.mock import patch

import torch
from omni_core.brain import AdaptiveBrain
from omni_core.connected_execution import _digest, resolve_execution_engine
from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.execution_compatibility import require_execution_compatibility
from worker import Worker


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): result.update(block)
    return result.hexdigest()


def two_minute_native_cap(method):
    @wraps(method)
    def bounded(self):
        if not hasattr(signal, "ITIMER_REAL"):
            raise unittest.SkipTest("this explicitly bounded developer fixture requires a POSIX wall watchdog")
        previous = signal.getsignal(signal.SIGALRM)
        def deadline(_signal, _frame):
            raise TimeoutError("The one current native Worker attempt exceeded its 120-second wall budget; saved source cursor remains authoritative.")
        signal.signal(signal.SIGALRM, deadline)
        signal.setitimer(signal.ITIMER_REAL, 120)
        try:
            return method(self)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
    return bounded


@unittest.skipUnless(os.environ.get("OMNI_CURRENT_INHERITED_WORKER_FIXTURE"), "requires a NEW owned genuine current-seal capture, not model initialization")
class CurrentInheritedWorkerColdProof(unittest.TestCase):
    @two_minute_native_cap
    def test_root_owned_resume_resuspend_cold_finish_and_no_duplicate_effects(self):
        torch.set_num_threads(1)
        started = time.perf_counter()
        def note(stage, **values):
            print(json.dumps({"stage": stage, "elapsedSeconds": round(time.perf_counter() - started, 3), **values}), flush=True)
        root = Path(os.environ["OMNI_CURRENT_INHERITED_WORKER_FIXTURE"]).resolve()
        metadata = json.loads((root / "engine/brain.json").read_text())
        identifier = metadata["brain_id"]
        entry = next(row for row in metadata["connected_executions"]["entries"] if row["status"] in {"queued", "suspended"})
        source = resolve_execution_engine(root / "engine", entry, resumed=bool(entry.get("resumeEnginePath")))
        store = ExecutionCheckpointStore(source, entry["parentId"])
        saved = store.current(entry.get("resumeCheckpointId", entry["sourceCheckpointId"]))
        require_execution_compatibility(saved.get("runtimeCompatibility"), "cpu")
        source_files = tuple(source / name for name in ("brain.json", "core.safetensors", "plasticity.safetensors"))
        immutable = {str(path): digest(path) for path in source_files}
        origin = root / "engine/origin/brain.json"
        if origin.exists(): immutable[str(origin)] = digest(origin)
        prefix = saved["presentation"]["publishedText"]
        events, pause = [], []
        worker = Worker()
        deadline_requested = threading.Event()
        params = {"brainId": identifier, "storagePath": str(root), "executionId": entry["id"],
            "streamId": entry["rootTurnId"], "toolSchemas": [], "toolObservationWaitMs": 0}
        def notify(kind, **fields):
            events.append((kind, fields))
            if kind == "chat-token" and fields["data"]["delta"] and not pause:
                pause.append(worker.request_chat_suspension({"brainId": identifier, "streamId": entry["rootTurnId"]}))
        worker.notify = notify
        def request_budget_boundary():
            deadline_requested.set()
            request = worker.request_chat_suspension({"brainId": identifier, "streamId": entry["rootTurnId"]})
            if not request.get("requested") and identifier not in worker.brains:
                worker._cooperative_cancel.set()  # Safe loader boundary, not a new Stop on saved work.
        deadline_timer = threading.Timer(max(0., 110. - (time.perf_counter() - started)), request_budget_boundary)
        deadline_timer.daemon = True; deadline_timer.start()
        with patch.object(AdaptiveBrain, "_select_structured_actions", return_value=({"talk": 1., "ponder": 0.}, [])):
            try:
                initial = worker.instance_status(params)
                self.assertFalse(initial["learnedRuntimeResident"])
                self.assertTrue(initial["resumeCompatibility"]["compatible"])
                note("current-native-frontier-verified", brainId=identifier, executionId=entry["id"], prefix=prefix)
                worker.load({"brainId": identifier, "storagePath": str(root)}, "explicit-load")
                note("first-explicit-load")
                brain = worker.brains[identifier]
                reference_path = root.parent / "capture-reference.json"
                if reference_path.exists():
                    reference = json.loads(reference_path.read_text())
                else:
                    # An independent saved-cursor computation precedes all
                    # actual Worker continuation. Reuse this resident owner,
                    # not another model Load or a posthoc observed response.
                    from .capture_connected_frontier_reference import reference_current_frontier
                    reference = reference_current_frontier(brain, entry["id"], progress=lambda stage, values: note(stage, **values), cancelled=deadline_requested.is_set)
                self.assertEqual(reference["format"], "omni-current-native-frontier-reference")
                self.assertEqual(reference["version"], 1)
                self.assertEqual(reference["checkpointId"], saved["checkpointId"])
                self.assertEqual(reference["manifestSha256"], _digest(saved))
                self.assertEqual(reference["runtimeCompatibilitySha256"], saved["runtimeCompatibility"]["sha256"])
                self.assertEqual(reference["publishedText"], prefix)
                self.assertTrue(reference["finalizationSkipped"]); self.assertTrue(reference["originalSourceUnchanged"])
                expected_text = reference["text"]
                self.assertIsInstance(expected_text, str); self.assertTrue(expected_text.startswith(prefix))
                note("independent-current-reference-verified")
                before_inference = brain.counters["inference_count"]
                before_human_count = sum(row["payload"].get("role") == "human" for row in brain.conversation.page(kinds=("message",), limit=1000))
                # There is one resident joint model; source owner borrowing may
                # not insert a separately loaded parent in the Worker registry.
                paused = worker.resume_inherited_chat(params, "inherited-owned-resume")
                self.assertTrue(pause[0]["requested"]); self.assertTrue(paused["suspended"])
                note("regional-worker-resuspended", checkpointId=paused["resume"]["checkpointId"])
                self.assertEqual(set(worker.brains), {identifier})
                self.assertEqual(brain.counters["inference_count"], before_inference)
                published = prefix + "".join(fields["data"]["delta"] for kind, fields in events if kind == "chat-token")
                frontier = worker.inherited_execution_presentation(params)
                self.assertEqual(frontier["presentation"]["publishedText"], published)
                self.assertEqual(frontier["frontier"]["turnId"], entry["rootTurnId"])
                previous_owner = weakref.ref(brain); del brain
                unloaded = worker.suspend_instance({"brainId": identifier, "storagePath": str(root)}, "owned-unload")
                self.assertTrue(unloaded["unloaded"]); self.assertFalse(unloaded["learnedRuntimeResident"])
                gc.collect(); self.assertIsNone(previous_owner())
                self.assertNotIn(identifier, worker._brain_leases)
                note("runtime-and-lease-released")
                worker.shutdown({}, None)
                worker = Worker(); worker.notify = lambda kind, **fields: events.append((kind, fields))
                self.assertFalse(worker.instance_status(params)["learnedRuntimeResident"])
                self.assertEqual(worker.inherited_execution_presentation(params)["presentation"]["publishedText"], published)
                worker.load({"brainId": identifier, "storagePath": str(root)}, "cold-explicit-load")
                note("cold-explicit-load")
                result = worker.resume_inherited_chat(params, "cold-same-frontier")
                note("root-turn-committed", text=result.get("text"), turnCommitted=result.get("turnCommitted"))
                self.assertTrue(result["turnCommitted"]); self.assertEqual(result["text"], expected_text)
                self.assertEqual(prefix + "".join(fields["data"]["delta"] for kind, fields in events if kind == "chat-token"), expected_text)
                self.assertEqual(worker.brains[identifier].counters["inference_count"], before_inference + 1)
                self.assertEqual(sum(row["payload"].get("role") == "human" for row in worker.brains[identifier].conversation.page(kinds=("message",), limit=1000)), before_human_count)
                self.assertEqual(worker.instance_status(params)["inheritedExecutions"]["pending"], 0)
                count = len(events); duplicate = worker.resume_inherited_chat(params, "lost-ack")
                self.assertTrue(duplicate["idempotentCompletion"]); self.assertEqual(duplicate["actions"], [])
                self.assertEqual(len(events), count)
                sequences = [fields["sequence"] for kind, fields in events if kind.startswith("chat-")]
                self.assertEqual(sequences, sorted(set(sequences)))
                for kind, fields in events:
                    if kind.startswith("chat-"):
                        self.assertEqual(fields["brain_id"], identifier); self.assertEqual(fields["stream_id"], entry["rootTurnId"])
                self.assertEqual({path: digest(path) for path in immutable}, immutable)
                note("current-worker-proof-complete", idempotent=True, originalSourceUnchanged=True)
            finally:
                deadline_timer.cancel()
                worker.shutdown({}, None)


if __name__ == "__main__": unittest.main()
