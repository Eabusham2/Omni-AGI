"""Neural-free stdio/control concurrency fixtures; no brain or model is run."""
import json
import os
import queue
import sys
import threading
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))
from worker_transport import ProtocolLineReader, serve_worker_stdio


class Fault(Exception):
    def __init__(self, code, message):
        self.code, self.message, self.data = code, message, None


class Pipe:
    def __init__(self):
        self.lines = queue.Queue()

    def readline(self, *_args):
        return self.lines.get(timeout=3)

    def send(self, method, request_id):
        self.lines.put(json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": {}}).encode() + b"\n")


class TransportTests(unittest.TestCase):
    def test_unbuffered_reader_keeps_complete_frames_and_the_resource_envelope(self):
        reader, writer = os.pipe()
        try:
            os.write(writer, b"first\nsecond\nlonger-frame\n")
            os.close(writer)
            writer = None
            stream = ProtocolLineReader(reader)
            self.assertEqual(stream.readline(100), b"first\n")
            self.assertEqual(stream.readline(100), b"second\n")
            self.assertEqual(stream.readline(4), b"long")
            self.assertEqual(stream.readline(100), b"er-frame\n")
            self.assertEqual(stream.readline(100), b"")
        finally:
            os.close(reader)
            if writer is not None:
                os.close(writer)

    def test_inline_control_reaches_busy_dispatch_without_parallel_neural_work(self):
        started, release = threading.Event(), threading.Event()
        responses = queue.Queue()
        pipe = Pipe()

        class Session:
            running = True
            active = 0
            maximum_active = 0

            def dispatch(self, request):
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                if request["id"] == "reply":
                    started.set()
                    self.assert_release = release.wait(2)
                self.active -= 1
                return {"jsonrpc": "2.0", "id": request["id"], "result": "saved text"}

            def dispatch_control(self, request):
                return {"jsonrpc": "2.0", "id": request["id"], "result": {"requested": True}}

        session = Session()
        runner = threading.Thread(target=serve_worker_stdio, args=(session, pipe, responses.put, lambda: None, Fault))
        runner.start()
        try:
            pipe.send("chat", "reply")
            self.assertTrue(started.wait(1))
            pipe.send("cancel_inline_generation", "artifact-only")
            self.assertEqual(responses.get(timeout=1)["id"], "artifact-only")
            self.assertEqual(session.active, 1)
            pipe.send("chat", "next")
            pipe.lines.put(b"")
            release.set()
            self.assertEqual(responses.get(timeout=1)["id"], "reply")
            self.assertEqual(responses.get(timeout=1)["id"], "next")
            runner.join(1)
            self.assertFalse(runner.is_alive())
            self.assertEqual(session.maximum_active, 1)
        finally:
            release.set()
            pipe.lines.put(b"")
            runner.join(3)

    def test_native_sampling_metadata_controls_never_queue_behind_waiting_chat(self):
        started, release = threading.Event(), threading.Event()
        responses, pipe = queue.Queue(), Pipe()
        neural_threads, control_threads = [], []
        class Session:
            running = True
            def dispatch(self, request):
                neural_threads.append(threading.get_ident())
                started.set()
                release.wait(2)
                return {"jsonrpc": "2.0", "id": request["id"], "result": {"turnCommitted": True}}
            def dispatch_control(self, request):
                control_threads.append(threading.get_ident())
                return {"jsonrpc": "2.0", "id": request["id"], "result": {"state": "queued"}}
        session = Session()
        runner = threading.Thread(target=serve_worker_stdio, args=(session, pipe, responses.put, lambda: None, Fault))
        runner.start()
        try:
            pipe.send("chat", "waiting-turn")
            self.assertTrue(started.wait(1))
            for method in ("mcp_sampling_enqueue", "mcp_sampling_status", "mcp_sampling_cancel", "mcp_sampling_consume"):
                pipe.send(method, method)
                self.assertEqual(responses.get(timeout=1)["id"], method)
            self.assertEqual(len(neural_threads), 1)
            self.assertEqual(len(control_threads), 4)
            self.assertTrue(all(thread != neural_threads[0] for thread in control_threads))
            pipe.lines.put(b"")
            release.set()
            self.assertEqual(responses.get(timeout=1)["id"], "waiting-turn")
        finally:
            release.set(); pipe.lines.put(b""); runner.join(3)
        self.assertFalse(runner.is_alive())


if __name__ == "__main__":
    unittest.main()
