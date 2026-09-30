"""Pure codec ownership/control/lease fixtures: no brain, model or codec process."""
import json
import queue
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codec_gateway import CodecOwner, CodecRuntimeGateway, CodecRuntimeCancelled
from worker import Worker, RpcFault
from worker_transport import serve_worker_stdio


def configuration(path="/fixture/pinned/ffmpeg"):
    return {"executablePath": path, "artifactSha256": "a" * 64,
        "binarySha256": "b" * 64, "binarySizeBytes": 128, "target": "linux-x64"}


class CodecGatewayTests(unittest.TestCase):
    def gateway(self, external=None, verify=None):
        events = queue.Queue()
        gateway = CodecRuntimeGateway(lambda kind, owner, data: events.put((kind, owner, data)),
            environment={}, resolve_external=lambda _env: external,
            verify=verify or (lambda _params, **_kwargs: {"configured": True}))
        return gateway, events

    def thread_lease(self, gateway, owner, cancelled=lambda: False):
        effects, errors = [], []
        def run():
            try:
                with gateway.scope(owner, cancelled), gateway.lease(owner.brain_id, "decode-video") as path:
                    effects.append(path)
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=run)
        thread.start()
        return thread, effects, errors

    def test_first_actual_codec_challenge_can_be_resolved_without_neural_queue(self):
        gateway, events = self.gateway()
        owner = CodecOwner("rpc", "brain", "job")
        thread, effects, errors = self.thread_lease(gateway, owner)
        kind, _owner, data = events.get(timeout=1)
        self.assertEqual(kind, "codec-runtime-needed")
        self.assertEqual(effects, [])
        self.assertEqual(data["jobId"], "job")
        reply = {key: value for key, value in data.items() if key != "purpose"}
        gateway.resolve({**reply, "outcome": "ready", "configuration": configuration()})
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(effects, ["/fixture/pinned/ffmpeg"])
        self.assertEqual(errors, [])
        self.assertFalse(gateway.leases)

    def test_wrong_nonce_brain_job_turn_or_action_receipt_never_unblocks_effect(self):
        gateway, events = self.gateway()
        owner = CodecOwner("rpc", "brain", "job", "turn", "a" * 32)
        thread, effects, errors = self.thread_lease(gateway, owner)
        _kind, _owner, data = events.get(timeout=1)
        reply = {key: value for key, value in data.items() if key != "purpose"}
        for field in ("challengeId", "requestId", "brainId", "jobId", "streamId", "actionId"):
            with self.assertRaises(ValueError):
                gateway.resolve({**reply, field: "wrong", "outcome": "ready", "configuration": configuration()})
        self.assertEqual(effects, [])
        gateway.resolve({**reply, "outcome": "cancelled", "reason": "selected artifact cancelled"})
        thread.join(1)
        self.assertIsInstance(errors[0], CodecRuntimeCancelled)
        self.assertEqual(effects, [])

    def test_existing_external_runtime_does_not_request_or_install_on_text_or_media(self):
        gateway, events = self.gateway(external="/fixture/external/ffmpeg")
        with gateway.scope(CodecOwner("rpc", "brain"), lambda: False):
            self.assertTrue(events.empty()) # text/image work has no codec lease
            with gateway.lease("brain", "decode-audio") as path:
                self.assertEqual(path, "/fixture/external/ffmpeg")
        self.assertTrue(events.empty())

    def test_changed_selection_waits_only_for_actual_codec_leases_not_image_work(self):
        gateway, _events = self.gateway(external="/old/ffmpeg")
        acquired, release, configured = threading.Event(), threading.Event(), threading.Event()
        def old_codec():
            with gateway.scope(CodecOwner("old-rpc", "brain"), lambda: False), gateway.lease("brain", "decode-video"):
                acquired.set()
                release.wait(1)
        old = threading.Thread(target=old_codec)
        old.start()
        self.assertTrue(acquired.wait(1))
        selecting = threading.Thread(target=lambda: (gateway.configure(configuration("/new/ffmpeg")), configured.set()))
        selecting.start()
        self.assertFalse(configured.wait(.03))
        self.assertEqual(gateway.selected_path, "/old/ffmpeg")
        release.set()
        old.join(1)
        selecting.join(1)
        self.assertTrue(configured.is_set())
        with gateway.scope(CodecOwner("image-rpc", "brain"), lambda: False):
            gateway.configure(configuration("/next/ffmpeg"))
        self.assertEqual(gateway.selected_path, "/next/ffmpeg")

    def test_modified_pinned_binary_is_reverified_before_any_side_effect(self):
        calls = []
        def verify(_params, **_kwargs):
            calls.append(1)
            if len(calls) > 1:
                raise ValueError("changed pinned binary")
            return {"configured": True}
        gateway, _events = self.gateway(verify=verify)
        gateway.configure(configuration())
        thread, effects, errors = self.thread_lease(gateway, CodecOwner("rpc", "brain"))
        thread.join(1)
        self.assertEqual(effects, [])
        self.assertIn("changed pinned binary", str(errors[0]))
        self.assertFalse(gateway.leases)

    def test_cancellation_after_ready_receipt_cancels_pending_selection_without_changing_owner(self):
        entered, release = threading.Event(), threading.Event()
        def verify(_params, cancelled, **_kwargs):
            entered.set()
            release.wait(1)
            if cancelled(): raise CodecRuntimeCancelled("cancelled during verification")
            return {"configured": True}
        gateway, events = self.gateway(verify=verify)
        thread, effects, errors = self.thread_lease(gateway, CodecOwner("rpc", "brain", "job"))
        _kind, _owner, data = events.get(timeout=1)
        reply = {key: value for key, value in data.items() if key != "purpose"}
        gateway.resolve({**reply, "outcome": "ready", "configuration": configuration()})
        self.assertTrue(entered.wait(1))
        gateway.resolve({**reply, "outcome": "cancelled", "reason": "exact owner stopped"})
        release.set()
        thread.join(1)
        self.assertEqual(effects, [])
        self.assertIsInstance(errors[0], CodecRuntimeCancelled)
        self.assertNotIn("IMAGEIO_FFMPEG_EXE", gateway.environment)

    def test_busy_worker_reader_resolves_codec_and_typed_cancel_after_handler_unwinds(self):
        lines, messages = queue.Queue(), queue.Queue()
        gateway, events = self.gateway()
        worker = Worker.__new__(Worker)
        worker.worker_role, worker.running = "neural", True
        worker._codec_gateway = gateway
        worker._cooperative_cancel = threading.Event()
        worker.cancelled_jobs = set()
        worker._cleanup_expired_inline_generations = lambda: None
        worker._get = lambda _params: SimpleNamespace()
        unwound = threading.Event()
        def ingest(_params, _request_id):
            try:
                with gateway.lease("brain", "decode-video"):
                    self.fail("cancelled setup must never reach codec effect")
            finally:
                unwound.set() # stands in for actual handler transaction cleanup
        worker.methods = {"ingest": ingest}
        stream = SimpleNamespace(readline=lambda *_args: lines.get(timeout=2))
        transport = threading.Thread(target=serve_worker_stdio, args=(worker, stream, messages.put, lambda: 1024**3, RpcFault))
        transport.start()
        lines.put(json.dumps({"jsonrpc": "2.0", "id": "rpc", "method": "ingest", "params": {"brainId": "brain", "jobId": "job"}}).encode() + b"\n")
        _kind, _owner, data = events.get(timeout=1)
        reply = {key: value for key, value in data.items() if key != "purpose"}
        lines.put(json.dumps({"jsonrpc": "2.0", "id": "control", "method": "resolve_codec_runtime", "params": {
            **reply, "outcome": "cancelled", "reason": "owned Stop"}}).encode() + b"\n")
        answers = [messages.get(timeout=1), messages.get(timeout=1)]
        self.assertTrue(unwound.is_set())
        self.assertEqual(next(value for value in answers if value["id"] == "rpc")["error"]["data"], {"codecRuntimeCancelled": True, "safeBoundary": True})
        self.assertTrue(next(value for value in answers if value["id"] == "control")["result"]["acknowledged"])
        lines.put(b"")
        transport.join(1)
        self.assertFalse(transport.is_alive())

    def test_direct_modality_cooperative_control_never_selects_unrelated_ingestion(self):
        worker = Worker.__new__(Worker)
        worker._active_request_lock = threading.Lock()
        worker._cooperative_cancel = threading.Event()
        worker._active_request = ("generate_modality", "owned")
        self.assertTrue(worker.request_cooperative_cancel())
        self.assertTrue(worker._cooperative_cancel.is_set())
        worker._active_request = ("ingest", "unrelated")
        worker._cooperative_cancel.clear()
        self.assertFalse(worker.request_cooperative_cancel())
        self.assertFalse(worker._cooperative_cancel.is_set())

    def test_artifact_control_is_strictly_bound_without_signalling_another_request(self):
        worker = Worker.__new__(Worker)
        worker.worker_role = "neural"
        worker._active_request_lock = threading.Lock()
        worker._cooperative_cancel = threading.Event()
        owner = CodecOwner("rpc", "brain", "job", "turn", "a" * 32)
        worker._artifact_request_owner = owner
        worker._active_request = ("generate_modality", "rpc")
        for key in owner.fields():
            with self.assertRaises(RpcFault):
                worker.dispatch_control({"jsonrpc": "2.0", "id": "control", "method": "cancel_artifact_request", "params": {
                    **owner.fields(), key: "other"}})
            self.assertFalse(worker._cooperative_cancel.is_set())
        result = worker.dispatch_control({"jsonrpc": "2.0", "id": "control", "method": "cancel_artifact_request", "params": owner.fields()})
        self.assertTrue(result["result"]["requested"])
        self.assertTrue(worker._cooperative_cancel.is_set())


if __name__ == "__main__":
    unittest.main()
