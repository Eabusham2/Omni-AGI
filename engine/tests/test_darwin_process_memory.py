"""Tiny numeric libproc fixtures; no brain construction, training, or devices."""

import ctypes
import errno
import os
import select
import subprocess
import sys
import unittest
from unittest.mock import patch

from omni_core import darwin_process_memory as native
from omni_core.managed_process_memory import (
    ManagedProcessMemorySampler, ProcessRow, ProcessTable, managed_family_sample,
)


def info(pid, parent, rss, started=(100, 5)):
    value = native.ProcTaskAllInfo()
    value.bsd.pid, value.bsd.parent_pid = pid, parent
    value.bsd.start_seconds, value.bsd.start_microseconds = started
    value.task.resident_size = rss
    return value


class FakeLibproc:
    def __init__(self, infos=None, children=None):
        self.infos = infos or {10: [info(10, 1, 100)], 11: [info(11, 10, 200)],
                               12: [info(12, 10, 300)], 13: [info(13, 11, 400)]}
        self.children = children or {10: [(11, 12)], 11: [(13,)], 12: [()], 13: [()]}
        self.info_calls, self.child_calls = {}, {}
        self.info_size = ctypes.sizeof(native.ProcTaskAllInfo)
        self.child_errno = 0
        self.child_bytes = None

    def proc_pidinfo(self, pid, flavor, arg, buffer, size):
        if flavor != native.PROC_PIDTASKALLINFO or arg != 0 or size != ctypes.sizeof(native.ProcTaskAllInfo):
            raise AssertionError("unexpected libproc info ABI")
        call = self.info_calls.get(pid, 0)
        self.info_calls[pid] = call + 1
        versions = self.infos[pid]
        value = versions[min(call, len(versions) - 1)]
        ctypes.memmove(buffer, ctypes.byref(value), size)
        return self.info_size

    def proc_listpids(self, flavor, parent, buffer, size):
        if flavor != native.PROC_PPID_ONLY:
            raise AssertionError("not a family-scoped process list")
        call = self.child_calls.get(parent, 0)
        self.child_calls[parent] = call + 1
        versions = self.children[parent]
        values = versions[min(call, len(versions) - 1)]
        for index, pid in enumerate(values[:len(buffer)]):
            buffer[index] = pid
        if self.child_errno:
            ctypes.set_errno(self.child_errno)
            return 0
        return self.child_bytes if self.child_bytes is not None else len(values) * ctypes.sizeof(ctypes.c_int)


class DarwinProcessMemoryTests(unittest.TestCase):
    def family(self, api, *, limit=32):
        with patch.object(native, "_api", return_value=api):
            return native.family_rows(10, max_process_rows=limit)

    def test_public_abi_layout(self):
        self.assertEqual(ctypes.sizeof(native.ProcBsdInfo), 136)
        self.assertEqual(native.ProcBsdInfo.pid.offset, 12)
        self.assertEqual(native.ProcBsdInfo.parent_pid.offset, 16)
        self.assertEqual(native.ProcBsdInfo.start_seconds.offset, 120)
        self.assertEqual(native.ProcBsdInfo.start_microseconds.offset, 128)
        self.assertEqual(ctypes.sizeof(native.ProcTaskInfo), 96)
        self.assertEqual(native.ProcTaskInfo.resident_size.offset, 8)
        self.assertEqual(ctypes.sizeof(native.ProcTaskAllInfo), 232)
        self.assertEqual(native.ProcTaskAllInfo.task.offset, 136)

    def test_exact_family_parent_proof_and_conservative_fresh_rss(self):
        api = FakeLibproc()
        api.infos[11].append(info(11, 10, 250))
        api.infos[12].append(info(12, 10, 275))
        rows = self.family(api)
        self.assertEqual(rows, {10: (1, 100), 11: (10, 250), 12: (10, 300), 13: (11, 400)})
        self.assertEqual(api.info_calls, {pid: 2 for pid in rows})
        self.assertEqual(api.child_calls, {pid: 2 for pid in rows})
        table = ProcessTable({pid: ProcessRow(pid, parent, rss) for pid, (parent, rss) in rows.items()})
        sample = managed_family_sample(table, 10, 13, "fixture")
        self.assertEqual(sample.rss_bytes, 1050)
        self.assertEqual(sample.worker_rss_bytes, 400)
        self.assertEqual(sample.owned_pids, (10, 11, 12, 13))
        with self.assertRaisesRegex(ValueError, "absent"):
            managed_family_sample(table, 10, 20, "fixture")

    def test_traced_original_parent_is_not_assumed_to_be_actual_parent(self):
        api = FakeLibproc()
        api.infos[11] = [info(11, 99, 200)]
        with self.assertRaisesRegex(ValueError, "parent changed"):
            self.family(api)

    def test_reused_pid_or_reparented_member_invalidates_sample(self):
        for changed in (info(11, 10, 200, (101, 5)), info(11, 99, 200)):
            api = FakeLibproc()
            api.infos[11].append(changed)
            with self.subTest(changed_parent=changed.bsd.parent_pid), \
                    self.assertRaisesRegex(ValueError, "identity changed"):
                self.family(api)

    def test_changed_root_identity_also_invalidates_sample(self):
        api = FakeLibproc()
        api.infos[10].append(info(10, 1, 100, (101, 5)))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.family(api)

    def test_child_appearance_or_exit_invalidates_sample(self):
        for changed in ((11,), (11, 12, 14)):
            api = FakeLibproc()
            api.children[10].append(changed)
            with self.subTest(children=changed), self.assertRaisesRegex(ValueError, "family changed"):
                self.family(api)

    def test_inaccessible_or_short_native_info_fails_closed(self):
        for size in (0, 16, ctypes.sizeof(native.ProcTaskAllInfo) + 1):
            api = FakeLibproc()
            api.info_size = size
            with self.subTest(size=size), self.assertRaises(OSError):
                self.family(api)

    def test_zero_child_result_with_errno_is_failure_not_empty_family(self):
        api = FakeLibproc()
        api.child_errno = errno.EPERM
        with self.assertRaises(OSError):
            self.family(api)
        ctypes.set_errno(errno.EPERM)
        self.assertEqual(len(self.family(FakeLibproc())), 4)

    def test_child_buffer_full_or_nonintegral_byte_count_fails_closed(self):
        for received in (3, 4 * (32 + 1), 4 * (32 + 2)):
            api = FakeLibproc()
            api.child_bytes = received
            with self.subTest(received=received), self.assertRaisesRegex(ValueError, "truncated or incompatible"):
                self.family(api)

    def test_row_limit_cycle_duplicate_and_bad_pid_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds its bound"):
            self.family(FakeLibproc(), limit=3)
        for children in ((11, 11), (10,), (0,), (-1,)):
            api = FakeLibproc()
            api.children[10] = [children]
            with self.subTest(children=children), self.assertRaises(ValueError):
                self.family(api)

    def test_identity_fields_are_validated(self):
        for bad in (info(99, 1, 100), info(10, 1, 100, (0, 1)),
                    info(10, 1, 100, (100, 1_000_000))):
            api = FakeLibproc()
            api.infos[10] = [bad]
            with self.assertRaisesRegex(ValueError, "identity is invalid"):
                self.family(api)

    def test_observation_past_deadline_is_never_verified(self):
        for times in ([0.0, 1.0], [0.0, 0.0, 1.1]):
            with patch.object(native.time, "monotonic", side_effect=times), \
                    self.assertRaisesRegex(OSError, "time bound"):
                self.family(FakeLibproc())

    def test_force_refresh_keeps_timestamp_and_does_not_fall_back_to_ps(self):
        api = FakeLibproc()
        sampler = ManagedProcessMemorySampler(worker_pid=11, owner_pid="10")
        with patch.object(native, "_api", return_value=api), \
                patch("omni_core.managed_process_memory.sys.platform", "darwin"), \
                patch("omni_core.managed_process_memory.os.name", "posix"), \
                patch("omni_core.managed_process_memory.time.time_ns", side_effect=(1000, 2000, 3000)), \
                patch("omni_core.managed_process_memory._posix_table", side_effect=AssertionError("ps used")), \
                patch("subprocess.Popen", side_effect=AssertionError("command launched")):
            first = sampler.sample(force=True)
            self.assertTrue(first.verified)
            self.assertEqual(first.sample_started_ns, 1000)
            cached = sampler.sample()
            self.assertTrue(cached.cached)
            self.assertEqual(cached.sample_started_ns, 1000)
            api.infos[11] = [info(11, 10, 500)]
            second = sampler.sample(force=True)
            self.assertTrue(second.verified)
            self.assertFalse(second.cached)
            self.assertEqual(second.sample_started_ns, 2000)
            self.assertEqual(second.rss_bytes, first.rss_bytes + 300)
            api.info_size = 0
            failure = sampler.sample(force=True)
            self.assertFalse(failure.verified)
            self.assertIsNone(failure.rss_bytes)
            self.assertEqual(failure.sample_started_ns, 3000)
            self.assertEqual(failure.owned_pids, ())
            self.assertIn("admission must pause", failure.error)

    @unittest.skipUnless(sys.platform == "darwin", "live Darwin libproc fixture")
    def test_live_child_family_excludes_parent_and_never_launches_a_probe(self):
        # This privately launched fixture parks a child and grandchild on
        # pipes. Sampling itself runs with process launch explicitly disabled.
        grandchild_code = "import os,sys; data=bytearray(3*1024*1024); print(os.getpid(), flush=True); sys.stdin.buffer.read()"
        child_code = (
            "import subprocess,sys\n"
            f"child=subprocess.Popen([sys.executable, '-c', {grandchild_code!r}], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)\n"
            "print(child.stdout.readline().strip(), flush=True)\n"
            "sys.stdin.buffer.read()\n"
            "child.stdin.close()\n"
            "child.wait(timeout=5)\n"
        )
        child = subprocess.Popen([sys.executable, "-c", child_code], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([child.stdout], [], [], 5)[0], "fixture did not start")
            grandchild_pid = int(child.stdout.readline().strip())
            sampler = ManagedProcessMemorySampler(worker_pid=grandchild_pid, owner_pid=str(child.pid))
            with patch("subprocess.Popen", side_effect=AssertionError("sampling launched a command")):
                sample = sampler.sample(force=True)
            self.assertTrue(sample.verified, sample.error)
            self.assertEqual(sample.owned_pids, tuple(sorted((child.pid, grandchild_pid))))
            self.assertEqual(sample.process_count, 2)
            self.assertNotIn(os.getpid(), sample.owned_pids)
            self.assertGreater(sample.worker_rss_bytes, 3 * 1024 * 1024)
            self.assertGreater(sample.rss_bytes, sample.worker_rss_bytes)
            self.assertGreater(sample.sample_started_ns, 0)
            self.assertLess(sample.sample_duration_ms, 1000)
        finally:
            child.stdin.close()
            child.wait(timeout=10)
            child.stdout.close()
            child.stderr.close()


if __name__ == "__main__":
    unittest.main()
