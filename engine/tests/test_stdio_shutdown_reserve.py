"""One inert transport regression; no actual Worker, model, app or hardware."""
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path: sys.path.insert(0, str(ENGINE))
from worker_transport import ProtocolLineReader, serve_worker_stdio
from stdio_request_admission import CONTROL_WORKING_SET_BYTES


class Fault(Exception):
    def __init__(self, code, message): self.code, self.message, self.data = code, message, None


class Lease:
    def __init__(self, size, operation):
        self.size, self.operation, self.marks, self.releases = size, operation, 0, 0
    def mark_allocated(self, *args): self.marks += 1
    def release(self): self.releases += 1


def frame(method, identity, **params):
    return json.dumps({"jsonrpc": "2.0", "id": identity, "method": method, "params": params}).encode() + b"\n"


class ShutdownReserveCases(unittest.TestCase):
    def test_shutdown_uses_existing_reserve_once_after_health_when_ordinary_allocator_fails(self):
        for boundary in ("raw", "decode", "headroom"):
            with self.subTest(boundary=boundary):
                health_done, leases, normal, responses, reads = threading.Event(), [], [], [], [0]
                payloads = [frame("health", "health"), frame("chat", "pretend", text="shutdown")
                    + frame("shutdown", "nonempty", force=True)
                    + frame("chat", "oversized", input="x" * 8192)
                    + frame("shutdown", "shutdown") + frame("shutdown", "duplicate")]
                class AllocatorPaused(RuntimeError):
                    status = {"memoryPressure": True, "ramAdmissionVerified": True,
                        "systemRamBudgetBytes": 1024 * 1024, "privatePath": "OWNED_PRIVATE_PATH"}
                def reserve(size, operation):
                    if operation == "stdio OS read chunk" and reads[0] >= 1:
                        self.assertTrue(health_done.wait(2), "actual health receipt never completed")
                    if health_done.is_set() and (boundary == "raw" or operation == "stdio decoded request"):
                        raise AllocatorPaused("OWNED_PRIVATE_ALLOCATION_DETAIL")
                    lease = Lease(size, operation); leases.append(lease); return lease
                def read(_descriptor, count):
                    self.assertEqual(_descriptor, 42)
                    reads[0] += 1
                    if not payloads: return b""
                    value, payloads[0] = payloads[0][:count], payloads[0][count:]
                    if not payloads[0]: payloads.pop(0)
                    return value
                def headroom():
                    if boundary == "headroom" and reads[0] >= 1:
                        self.assertTrue(health_done.wait(2))
                        return 4
                    return 4 * 1024 * 1024
                fixture = self
                class Session:
                    running = True
                    def dispatch(self, request):
                        normal.append(request["id"])
                        if request["method"] == "health":
                            return {"jsonrpc": "2.0", "id": request["id"], "result": {"ready": True}}
                        fixture.assertEqual(request["method"], "shutdown")
                        fixture.assertEqual(request["params"], {})
                        fixture.assertEqual(leases[0].releases, 0, "queued shutdown lost its pre-admitted receipt")
                        self.running = False
                        return {"jsonrpc": "2.0", "id": request["id"], "result": {"stopping": True}}
                def send(response):
                    responses.append(response)
                    if response.get("id") == "health" and "result" in response: health_done.set()
                with patch("worker_transport.os.read", side_effect=read):
                    serve_worker_stdio(Session(), ProtocolLineReader(42), send, lambda: None, Fault,
                        reserve_request_memory=reserve, request_memory_headroom=headroom)
                self.assertEqual(normal, ["health", "shutdown"])
                self.assertEqual(responses[-1]["result"], {"stopping": True})
                rejected = [row for row in responses if "error" in row]
                self.assertEqual([row["id"] for row in rejected], ["pretend", "nonempty", "oversized"])
                self.assertTrue(all(row["error"]["data"]["frameRejected"] for row in rejected))
                self.assertTrue(all(row["error"]["data"]["workerRestartRequired"] is False for row in rejected))
                self.assertEqual((leases[0].size, leases[0].operation),
                    (CONTROL_WORKING_SET_BYTES, "stdio control/drain safety working set"))
                self.assertEqual(leases[0].marks, 1)
                self.assertTrue(all(lease.releases == 1 for lease in leases))
                self.assertNotIn("OWNED_PRIVATE_ALLOCATION_DETAIL", json.dumps(responses))


if __name__ == "__main__": unittest.main()
