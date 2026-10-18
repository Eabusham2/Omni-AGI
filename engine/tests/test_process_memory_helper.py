"""Constructor-free adapter protocol/range tests. No model or user process."""
import base64
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("process_memory_helper", Path(__file__).parents[1] / "process_memory_helper.py")
memory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(memory)

IDENTITY = {"platform": "linux", "pid": 123456, "creation": "synthetic-creation", "executable": "/fixture/owned",
            "executableIdentity": "fixture-inode"}


class FixtureTarget(memory.Target):
    def __init__(self, pid, access):
        super().__init__(pid, access)
        self.data, self.writes, self.reads, self.closed = bytearray(b"abcdefghijklmnop"), 0, 0, False
        self.partial = False
        self.protection = False
        self.changed = False

    def identity(self):
        return {**IDENTITY, "creation": "different" if self.changed else IDENTITY["creation"]}

    def regions(self, after):
        for start in (0x1000, 0x2000, 0x3000):
            yield memory.region(start, start + 16, True, not self.protection, False)

    def read(self, start, length):
        self.reads += 1
        return bytes(self.data[:length]), None

    def write(self, start, data):
        self.writes += 1
        count = 2 if self.partial else len(data)
        self.data[:count] = data[:count]
        return count, 299 if self.partial else None

    def close(self):
        self.closed = True


class ApprovalInput:
    def __init__(self, request, output, approve=True):
        self.req, self.output, self.approve, self.reads = request, output, approve, 0

    def readline(self, maximum):
        self.reads += 1
        if self.reads == 1:
            return (memory.canonical(self.req) + "\n").encode()
        ready = json.loads(self.output.getvalue().splitlines()[0])
        return (memory.canonical({"commit": ready["nonce"] if self.approve else "wrong", "requestSha256": ready["requestSha256"]}) + "\n").encode()


def req(operation="read", **changes):
    value = {"format": memory.FORMAT, "operationId": "fixture-op", "operation": operation, "pid": IDENTITY["pid"],
             "access": "read-write", "expectedIdentity": IDENTITY}
    if operation in {"read", "write"}:
        value.update(address="0x1000", length=4)
    if operation == "write":
        value.update(dataBase64=base64.b64encode(b"WXYZ").decode(), expectedBeforeSha256=memory.sha(b"abcd"))
    value.update(changes)
    return value


class ProcessMemoryHelperTests(unittest.TestCase):
    def run_request(self, request, target=None, approve=True):
        target = target or FixtureTarget(request["pid"], request["access"])
        output = io.BytesIO()
        memory.serve(ApprovalInput(request, output, approve), output, lambda pid, access: target)
        return target, [json.loads(line) for line in output.getvalue().splitlines()]

    def test_exact_read_requires_host_commit_and_hashes_actual_bytes(self):
        target, rows = self.run_request(req())
        self.assertEqual(rows[0]["type"], "ready")
        self.assertEqual(rows[-1]["state"], "complete")
        self.assertEqual(base64.b64decode(rows[-1]["dataBase64"]), b"abcd")
        self.assertEqual(rows[-1]["dataSha256"], memory.sha(b"abcd"))
        self.assertTrue(target.closed)
        self.assertEqual(target.writes, 0)

    def test_wrong_nonce_never_reads_or_writes(self):
        target, rows = self.run_request(req("write"), approve=False)
        self.assertEqual((target.reads, target.writes), (0, 0))
        self.assertEqual(rows[-1]["state"], "cancelled")
        self.assertFalse(rows[-1]["writeMayHaveOccurred"])

    def test_process_identity_change_never_gets_ready_authority(self):
        target = FixtureTarget(123456, "read-write"); target.changed = True
        _, rows = self.run_request(req(), target)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["state"], "identity-changed")
        self.assertEqual(target.reads, 0)

    def test_preimage_mismatch_does_not_write(self):
        target, rows = self.run_request(req("write", expectedBeforeSha256="0" * 64))
        self.assertEqual(rows[-1]["state"], "preimage-mismatch")
        self.assertEqual(target.writes, 0)

    def test_real_byte_array_write_count_readback_and_nonatomic_label(self):
        target, rows = self.run_request(req("write"))
        self.assertEqual(bytes(target.data[:4]), b"WXYZ")
        self.assertEqual(rows[-1]["transferredBytes"], 4)
        self.assertEqual(rows[-1]["afterSha256"], memory.sha(b"WXYZ"))
        self.assertFalse(rows[-1]["compareAtomic"])
        self.assertFalse(rows[-1]["retrySafe"])

    def test_partial_write_is_not_complete_or_safe_to_retry(self):
        target = FixtureTarget(123456, "read-write"); target.partial = True
        _, rows = self.run_request(req("write"), target)
        self.assertEqual(rows[-1]["state"], "partial")
        self.assertEqual(rows[-1]["transferredBytes"], 2)
        self.assertTrue(rows[-1]["writeMayHaveOccurred"])
        self.assertFalse(rows[-1]["retrySafe"])

    def test_os_protected_region_is_never_changed_to_enable_write(self):
        target = FixtureTarget(123456, "read-write"); target.protection = True
        _, rows = self.run_request(req("write"), target)
        self.assertEqual(rows[-1]["state"], "protected")
        self.assertEqual((target.reads, target.writes), (0, 0))

    def test_region_pages_are_bounded_without_process_enumeration(self):
        _, rows = self.run_request(req("regions", regionLimit=2, afterAddress="0x0"))
        self.assertEqual(len(rows[-1]["regions"]), 2)
        self.assertFalse(rows[-1]["endOfRegions"])
        self.assertEqual(rows[-1]["nextAddress"], "0x2010")

    def test_bad_access_count_and_float_address_rejected(self):
        for value in (req("write", access="read"), req(length=memory.MAX_BYTES + 1), req(address=4096), req(address="0xffffffffffffffff", length=2)):
            with self.assertRaises(memory.MemoryFailure):
                memory.request(value)

    def test_write_exception_keeps_uncertainty_after_effect_boundary(self):
        target = FixtureTarget(123456, "read-write")
        def broken(start, data):
            target.data[:2] = data[:2]
            raise OSError(5, "fixture transport interrupted")
        target.write = broken
        _, rows = self.run_request(req("write"), target)
        self.assertEqual(rows[-1]["state"], "write-uncertain")
        self.assertIsNone(rows[-1]["transferredBytes"])
        self.assertFalse(rows[-1]["retrySafe"])

    def test_helper_import_does_not_import_a_neural_runtime(self):
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("omni_core", sys.modules)

    def test_identity_change_after_read_withholds_bytes_but_retains_count_hash(self):
        target = FixtureTarget(123456, "read-write")
        def changed_read(start, length):
            target.changed = True
            return b"abcd", None
        target.read = changed_read
        _, rows = self.run_request(req(), target)
        self.assertEqual(rows[-1]["state"], "identity-changed")
        self.assertNotIn("dataBase64", rows[-1])
        self.assertEqual(rows[-1]["dataSha256"], memory.sha(b"abcd"))
        self.assertEqual(rows[-1]["transferredBytes"], 4)


if __name__ == "__main__":
    unittest.main()
