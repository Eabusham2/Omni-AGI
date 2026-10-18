"""NEW idle serial handoff using inert requests/controllers, no neural work."""
import json
import queue
import sys
import threading
import types
import unittest
import weakref
from contextlib import nullcontext

from test_operation_auto_boundaries import ROOT, methods

sys.path.insert(0, str(ROOT))
from worker_transport import serve_worker_stdio


class Fault(Exception):
    code, data = -32000, None
    def __init__(self, message): self.message = message


class Pipe:
    def __init__(self): self.queue = queue.Queue()
    def readline(self, *_args): return self.queue.get(timeout=3)
    def send(self, method, identifier):
        self.queue.put(json.dumps({"id": identifier, "method": method, "params": {}}).encode()+b"\n")


class WholeMediaTransportCases(unittest.TestCase):
    def test_actual_transport_sends_genuine_response_before_private_work_and_prioritizes_new_request(self):
        pipe, replies, order = Pipe(), queue.Queue(), []
        private_started, retired = threading.Event(), threading.Event()
        class Worker:
            running, pending, active, peak = True, False, 0, 0
            def dispatch(self, request):
                self.active += 1; self.peak = max(self.peak, self.active); order.append(request["id"])
                if request["id"] == "first": self.pending = True
                else: self.running = False
                self.active -= 1
                return {"id": request["id"], "result": "inert response"}
            def retire_private_native_work(self): retired.set()
            def step_private_native_work(self):
                if not self.pending: return False
                retired.clear(); private_started.set(); self.active += 1; self.peak = max(self.peak, self.active)
                order.append("private-after-response")
                if not retired.wait(2): raise RuntimeError("fixture new user request did not retire private work")
                self.active -= 1; self.pending = False
                return False
        worker = Worker()
        def send(response): order.append("sent-"+response["id"]); replies.put(response)
        runner = threading.Thread(target=serve_worker_stdio, args=(worker, pipe, send, lambda: None, Fault))
        runner.start()
        try:
            pipe.send("generate_modality", "first"); self.assertEqual(replies.get(timeout=2)["id"], "first")
            self.assertTrue(private_started.wait(2)); pipe.send("chat", "second")
            self.assertEqual(replies.get(timeout=2)["id"], "second")
            runner.join(2); self.assertFalse(runner.is_alive())
            self.assertEqual(order, ["first", "sent-first", "private-after-response", "second", "sent-second"])
            self.assertEqual(worker.peak, 1)
        finally:
            worker.running = False; retired.set(); pipe.queue.put(b""); runner.join(2)

    def test_actual_worker_retirement_is_flags_only_and_inline_work_prevents_private_steps(self):
        ns = methods("worker.py", "Worker", {"retire_private_native_work", "step_private_native_work"}, {"nullcontext": nullcontext, "sys": sys})
        calls = []
        class Controller:
            def retire_private_media(self, reason): calls.append(("flag", reason))
            def release_private_media(self): calls.append(("release", "serial"))
            def step_private_media(self): calls.append(("step", "serial")); return False
        controller = Controller(); record = types.SimpleNamespace(finished=threading.Event())
        worker = types.SimpleNamespace(_private_media_controllers=(weakref.ref(controller),), _inline_generations={"live": record}, _inline_lock=threading.RLock())
        ns["retire_private_native_work"](worker); self.assertEqual(calls, [("flag", "new-normal-worker-request")])
        calls.clear(); self.assertFalse(ns["step_private_native_work"](worker))
        self.assertEqual(calls, [("flag", "live-inline-generation-has-priority"), ("release", "serial")])
        record.finished.set(); calls.clear(); ns["step_private_native_work"](worker)
        self.assertEqual(calls, [("step", "serial")])
