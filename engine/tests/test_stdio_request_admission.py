"""Fake stdio and SQLite request ownership only; no worker/model/app run."""
import importlib.util
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path: sys.path.insert(0, str(ENGINE))
from worker_transport import ProtocolLineReader, serve_worker_stdio
from stdio_request_admission import (AdmittedRequest, RequestAllocationReceipt, RequestMemoryPressure,
    decoded_request_bound, request_id_hint, retain_request_allocation)

spec = importlib.util.spec_from_file_location("stdio_quota_fixture", ENGINE / "omni_core" / "shared_resource_ledger.py")
quota_module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = quota_module; spec.loader.exec_module(quota_module)


class Fault(Exception):
    def __init__(self, code, message): self.code, self.message, self.data = code, message, None


class Lease:
    def __init__(self, size, operation): self.size, self.operation, self.marked, self.released = size, operation, False, False
    def mark_allocated(self, *args): self.marked = True
    def release(self): self.released = True


class Session:
    running = True
    def __init__(self): self.normal, self.control, self.retained = [], [], None
    def dispatch(self, request):
        self.normal.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": "committed suffix"}
    def dispatch_control(self, request):
        self.control.append(request)
        self.retained = retain_request_allocation(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": "accepted"}


def frame(method, identity, **params):
    return json.dumps({"jsonrpc": "2.0", "id": identity, "method": method, "params": params}).encode() + b"\n"


class StdioAdmissionFixtures(unittest.TestCase):
    def test_decoded_bound_prices_many_small_objects_but_long_strings_proportionally(self):
        long_text = frame("chat", "x", input="a" * 100_000)
        tiny_objects = b"[" + b"{}," * 1000 + b"{}]"
        self.assertLess(decoded_request_bound(long_text), len(long_text) * 14)
        self.assertGreater(decoded_request_bound(tiny_objects), len(tiny_objects) * 60)

    def test_request_identity_hint_never_confuses_nested_id_or_incomplete_string(self):
        self.assertEqual(request_id_hint(b'{"jsonrpc":"2.0","id":"owned","params":'), "owned")
        self.assertIsNone(request_id_hint(b'{"params":{"id":"nested"'))
        self.assertIsNone(request_id_hint(b'{"id":"incomplete'))
        self.assertEqual(request_id_hint(b'{"params":{"id":"nested","objects":[{},{}]},"id":"root"}'), "root")

    def test_raw_reader_reserves_before_os_read_and_copy(self):
        leases, calls = [], []
        def reserve(size, operation):
            calls.append(operation); lease = Lease(size, operation); leases.append(lease); return lease
        stream = ProtocolLineReader(42); stream.reserve_memory = reserve
        def read(_descriptor, _size):
            self.assertEqual(calls[-1], "stdio OS read chunk")
            return b"first\nsecond\n"
        with patch("worker_transport.os.read", side_effect=read):
            self.assertEqual(stream.readline(100), b"first\n")
        self.assertIn("stdio complete raw frame", calls)
        RequestAllocationReceipt([stream.line_lease]).release()
        self.assertTrue(all(lease.marked for lease in leases))
        self.assertEqual(stream.readline(100), b"second\n")

    def test_oversized_frame_drains_only_itself_and_keeps_suffix_in_same_os_chunk(self):
        session, responses = Session(), []
        rejected = frame("chat", "large", input="x" * 400)
        accepted = frame("chat", "suffix")
        stream = ProtocolLineReader(42)
        with patch("worker_transport.os.read", side_effect=[rejected + accepted, b""]):
            serve_worker_stdio(session, stream, responses.append, lambda: 2048, Fault)
        self.assertEqual([value["id"] for value in session.normal], ["suffix"])
        self.assertEqual(responses[0]["id"], "large")
        self.assertTrue(responses[0]["error"]["data"]["frameRejected"])
        self.assertFalse(responses[0]["error"]["data"]["workerRestartRequired"])
        self.assertTrue(session.running)

    def test_decoder_admission_happens_before_jsonloads_and_preserves_valid_suffix(self):
        session, responses = Session(), []
        huge, small = frame("chat", "large", input="x" * 2000), frame("chat", "suffix")
        def reserve(size, operation):
            if operation == "stdio decoded request" and size > 20_000: raise RuntimeError("selected budget full")
            return Lease(size, operation)
        actual = json.loads
        decoded = []
        def loads(value): decoded.append(value); return actual(value)
        with patch("worker_transport.json.loads", side_effect=loads):
            serve_worker_stdio(session, io.BytesIO(huge + small), responses.append, lambda: None, Fault,
                reserve_request_memory=reserve, request_memory_headroom=lambda: 4 * 1024 * 1024)
        self.assertEqual(len(decoded), 1)
        self.assertNotIn("large", decoded[0])
        self.assertEqual(session.normal[0]["id"], "suffix")
        self.assertEqual(responses[0]["id"], "large")

    def test_live_control_inbox_holds_same_atomic_lease_until_close(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = quota_module.SharedResourceLedger(Path(folder) / "quota.sqlite3", process_alive=lambda _: True)
            ledger.register_owner("stdio", 65536); ledger.set_ram_ceiling(4 * 1024 * 1024)
            def reserve(size, operation):
                return ledger.reserve("stdio", "ram", size, ram_budget_bytes=4 * 1024 * 1024,
                    observed_ram_bytes=128 * 1024, verified=True)
            session, responses = Session(), []
            serve_worker_stdio(session, io.BytesIO(frame("observe_chat_action", "observation", observation={"text": "real result"})),
                responses.append, lambda: None, Fault, reserve_request_memory=reserve,
                request_memory_headroom=lambda: 3 * 1024 * 1024)
            self.assertEqual(responses[0]["id"], "observation")
            self.assertIsNotNone(session.retained)
            # Fresh measured usage cannot retire a STILL-OWNED decoded object.
            self.assertGreater(ledger.status(observed_ns=time.time_ns() + 1000, verified=True)["ramEscrowBytes"], 0)
            session.retained.release()
            self.assertEqual(ledger.status(observed_ns=time.time_ns() + 1000, verified=True)["ramEscrowBytes"], 0)
            serialized = json.dumps(session.control[0])
            self.assertNotIn("_request_allocation", serialized)

    def test_partial_raw_copy_denial_does_not_discard_next_complete_frame(self):
        session, responses, copies = Session(), [], [0]
        payload = frame("chat", "rejected") + frame("chat", "suffix")
        def reserve(size, operation):
            if operation == "stdio complete raw frame":
                copies[0] += 1
                if copies[0] == 1: raise RuntimeError("no selected budget for raw copy")
            return Lease(size, operation)
        stream = ProtocolLineReader(42)
        with patch("worker_transport.os.read", side_effect=[payload, b""]):
            serve_worker_stdio(session, stream, responses.append, lambda: None, Fault,
                reserve_request_memory=reserve, request_memory_headroom=lambda: 1024 * 1024)
        self.assertEqual([value["id"] for value in session.normal], ["suffix"])
        self.assertEqual(responses[0]["id"], "rejected")

    def test_parse_failure_keeps_suffix_and_does_not_reuse_a_released_decode_lease(self):
        session, responses, leases = Session(), [], []
        def reserve(size, operation):
            lease = Lease(size, operation); leases.append(lease); return lease
        serve_worker_stdio(session, io.BytesIO(b'{"broken":\n' + frame("chat", "suffix")), responses.append,
            lambda: None, Fault, reserve_request_memory=reserve, request_memory_headroom=lambda: 1024 * 1024)
        self.assertEqual(responses[0]["error"]["code"], -32700)
        self.assertEqual(session.normal[0]["id"], "suffix")
        self.assertTrue(all(lease.released for lease in leases))

    def test_private_receipt_is_idempotent_but_not_retired_while_a_live_owner_exists(self):
        lease = Lease(100, "decoded")
        receipt = RequestAllocationReceipt([lease])
        request = AdmittedRequest({"id": "owned"}, receipt)
        handle = retain_request_allocation(request)
        receipt.release()
        self.assertFalse(lease.released)
        handle.release(); handle.release()
        self.assertTrue(lease.released)
        self.assertTrue(lease.marked)


if __name__ == "__main__": unittest.main()
