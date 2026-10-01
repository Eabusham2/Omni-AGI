"""Bounded managed-family RSS sampling, not OS-enforced memory isolation.

Only the trusted desktop launcher sets OMNI_MEMORY_OWNER_PID. Standalone CLI
defaults to its own PID and descendants, never the parent terminal/SSH shell.
RSS sums conservatively double-count pages shared by managed processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Tuple


MAX_PROCESS_ROWS = 32768
MAX_TABLE_BYTES = 2 * 1024 * 1024
SAMPLE_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    parent_pid: int
    rss_bytes: Optional[int]


@dataclass(frozen=True)
class ProcessTable:
    rows: Mapping[int, ProcessRow]
    rss_reader: Optional[Callable[[int], int]] = None


@dataclass(frozen=True)
class ManagedMemorySample:
    rss_bytes: Optional[int]
    worker_rss_bytes: Optional[int]
    root_pid: int
    process_count: int
    verified: bool
    scope: str
    cached: bool = False
    error: Optional[str] = None
    sample_duration_ms: float = 0.0
    sample_age_seconds: float = 0.0
    sample_started_ns: Optional[int] = None
    owned_pids: Tuple[int, ...] = ()


def parse_posix_process_table(payload: bytes) -> ProcessTable:
    if len(payload) > MAX_TABLE_BYTES:
        raise ValueError("process table exceeds bounded metadata allowance")
    rows: dict[int, ProcessRow] = {}
    for line in payload.decode("ascii").splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3 or any(not value.isdecimal() for value in fields):
            raise ValueError("invalid numeric process memory table")
        pid, parent, kib = map(int, fields)
        if pid < 1 or pid in rows or len(rows) >= MAX_PROCESS_ROWS:
            raise ValueError("invalid/oversized process memory inventory")
        rows[pid] = ProcessRow(pid, parent, kib * 1024)
    return ProcessTable(rows)


def managed_family_sample(table: ProcessTable, root_pid: int, worker_pid: int, scope: str) -> ManagedMemorySample:
    if len(table.rows) > MAX_PROCESS_ROWS or root_pid not in table.rows or worker_pid not in table.rows:
        raise ValueError("managed memory owner/worker is absent")
    ancestor, seen = worker_pid, set()
    while ancestor not in seen:
        if ancestor == root_pid:
            break
        seen.add(ancestor)
        row = table.rows.get(ancestor)
        if row is None:
            raise ValueError("managed memory root is not the worker ancestor")
        ancestor = row.parent_pid
    else:
        raise ValueError("process memory ancestor cycle")
    children: dict[int, list[int]] = {}
    for pid, row in table.rows.items():
        if pid != row.pid or pid < 1 or row.parent_pid < 0:
            raise ValueError("invalid process identity inventory")
        children.setdefault(row.parent_pid, []).append(pid)
    pending, owned, total, worker_rss = [root_pid], set(), 0, None
    while pending:
        pid = pending.pop()
        if pid in owned:
            raise ValueError("managed process descendant cycle")
        owned.add(pid)
        row = table.rows[pid]
        rss = row.rss_bytes
        if rss is None and table.rss_reader is not None:
            rss = table.rss_reader(pid)
        if type(rss) is not int or rss < 0:
            raise ValueError("managed process residency is unavailable")
        total += rss
        if pid == worker_pid:
            worker_rss = rss
        pending.extend(children.get(pid, ()))
    return ManagedMemorySample(total, worker_rss, root_pid, len(owned), True, scope,
        owned_pids=tuple(sorted(owned)))


def _posix_table() -> ProcessTable:
    # Read only pid/ppid/RSS: no command lines, paths, usernames or credentials.
    child = subprocess.Popen(["ps", "-axo", "pid=,ppid=,rss="], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    timer = threading.Timer(1.0, child.kill)  # Only this privately spawned probe.
    timer.daemon = True
    timer.start()
    try:
        assert child.stdout is not None
        payload = child.stdout.read(MAX_TABLE_BYTES + 1)
        if len(payload) > MAX_TABLE_BYTES:
            child.kill()
            raise ValueError("process table exceeds bounded metadata allowance")
        if child.wait(timeout=1.0) != 0:
            raise ValueError("managed process memory sample failed")
        return parse_posix_process_table(payload)
    finally:
        timer.cancel()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=1.0)
        if child.stdout is not None:
            child.stdout.close()


def _windows_table() -> ProcessTable:
    # Primary Toolhelp parent inventory + PSAPI working sets, no new package.
    # https://learn.microsoft.com/windows/win32/api/tlhelp32/ns-tlhelp32-processentry32w
    # https://learn.microsoft.com/windows/win32/api/psapi/nf-psapi-getprocessmemoryinfo
    import ctypes
    from ctypes import wintypes
    kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD),
            ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
            ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD), ("exe", wintypes.WCHAR * 260)]

    class Counters(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD),
            *[(name, ctypes.c_size_t) for name in ("peak", "resident", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")]]

    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    for name in ("Process32FirstW", "Process32NextW"):
        method = getattr(kernel, name)
        method.argtypes, method.restype = [wintypes.HANDLE, ctypes.POINTER(Entry)], wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot in (None, ctypes.c_void_p(-1).value):
        raise OSError("managed process snapshot failed")
    rows = {}
    try:
        entry = Entry(); entry.size = ctypes.sizeof(entry)
        more = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            if len(rows) >= MAX_PROCESS_ROWS:
                raise ValueError("process table exceeds bounded metadata allowance")
            if entry.pid:
                rows[int(entry.pid)] = ProcessRow(int(entry.pid), int(entry.parent), None)
            more = kernel.Process32NextW(snapshot, ctypes.byref(entry))
        if ctypes.get_last_error() not in (0, 18):  # ERROR_NO_MORE_FILES
            raise OSError("managed process inventory failed")
    finally:
        kernel.CloseHandle(snapshot)

    def rss(pid):
        handle = kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            raise OSError("managed process memory access failed")
        try:
            value = Counters(); value.size = ctypes.sizeof(value)
            if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(value), value.size):
                raise OSError("managed process memory sample failed")
            return int(value.resident)
        finally:
            kernel.CloseHandle(handle)
    return ProcessTable(rows, rss)


class ManagedProcessMemorySampler:
    def __init__(self, *, worker_pid: Optional[int] = None, owner_pid: Optional[str] = None,
                 provider: Optional[Callable[[], ProcessTable]] = None, now: Callable[[], float] = time.monotonic):
        self.worker_pid = os.getpid() if worker_pid is None else worker_pid
        configured = os.environ.get("OMNI_MEMORY_OWNER_PID") if owner_pid is None else owner_pid
        valid = configured is None or (configured.isdecimal() and len(configured) <= 10 and 1 <= int(configured) <= (1 << 31) - 1)
        self.root_pid = int(configured) if configured is not None and valid else self.worker_pid
        self.invalid_owner = not valid
        self.scope = "managed-app-process-family" if configured is not None else "standalone-process-family"
        self.provider, self.now = provider, now
        self._sample: Optional[ManagedMemorySample] = None
        self._at = float("-inf")
        self._lock = threading.Lock()

    def sample(self, *, force: bool = False) -> ManagedMemorySample:
        with self._lock:
            now = self.now()
            if not force and self._sample is not None and now - self._at < SAMPLE_INTERVAL_SECONDS:
                return ManagedMemorySample(**{**self._sample.__dict__, "cached": True, "sample_age_seconds": max(0.0, now - self._at)})
            try:
                sample_started_ns = time.time_ns()
                if self.invalid_owner:
                    raise ValueError("managed process memory owner is invalid")
                table = self.provider() if self.provider is not None else (_windows_table() if os.name == "nt" else _posix_table())
                sample = managed_family_sample(table, self.root_pid, self.worker_pid, self.scope)
            except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
                sample = ManagedMemorySample(None, None, self.root_pid, 0, False, self.scope,
                    error="managed process family residency unavailable; admission must pause")
            sample = ManagedMemorySample(**{**sample.__dict__, "sample_duration_ms": max(0.0, (self.now() - now) * 1000), "sample_started_ns": sample_started_ns})
            self._sample, self._at = sample, now
            return sample


_default_sampler: Optional[ManagedProcessMemorySampler] = None
_default_sampler_key = None
_default_lock = threading.Lock()


def default_managed_process_sampler() -> ManagedProcessMemorySampler:
    global _default_sampler, _default_sampler_key
    key = (os.getpid(), os.environ.get("OMNI_MEMORY_OWNER_PID"))
    with _default_lock:
        if key != _default_sampler_key:
            _default_sampler = ManagedProcessMemorySampler()
            _default_sampler_key = key
        return _default_sampler


def native_family_memory_containment(
    selected_bytes: int, *, root_pid: int, worker_pid: int,
    member_pids: Tuple[int, ...] = (),
    platform_name: Optional[str] = None, proc_root: Path = Path("/proc"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict:
    """Report an *existing* whole-family kernel bound, never install one.

    cgroup v2 can already constrain an app launched inside a container or a
    delegated slice. We only credit it when both the trusted owner and worker
    are in the same group and its hard memory.max is no higher than the selected
    budget and all sampled family members are in it. This is not RSS
    isolation, nor permission to write a shared system cgroup or set RLIMIT_AS
    (which would break large file-backed mappings).
    A missing/changed descendant membership fails this optional proof closed.
    """
    platform = sys.platform if platform_name is None else platform_name
    result = {"mechanism": "none", "existingNativeBound": False,
              "selectedCeilingNativeBounded": False, "hardRssIsolation": False,
              "scope": "unavailable-or-unverified"}
    if not platform.startswith("linux") or type(selected_bytes) is not int or selected_bytes < 0:
        return result

    def group(pid):
        lines = (proc_root / str(pid) / "cgroup").read_text("ascii").splitlines()
        matches = [line[3:] for line in lines if line.startswith("0::")]
        if len(matches) != 1 or not matches[0].startswith("/") or ".." in Path(matches[0]).parts:
            raise ValueError("unverified cgroup v2 membership")
        return matches[0]

    try:
        members = set(member_pids)
        if root_pid not in members or worker_pid not in members:
            return result
        root_group = group(root_pid)
        if any(group(pid) != root_group for pid in members):
            return result
        # A cgroup namespace often mounts the process's own group at the
        # visible root, while a host mount exposes its path below that root.
        candidates = (cgroup_root / root_group.lstrip("/") / "memory.max",)
        for candidate in candidates:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            raw = candidate.read_text("ascii").strip()
            if raw == "max":
                continue
            limit = int(raw)
            if limit <= 0:
                continue
            return {"mechanism": "existing-cgroup-v2-memory.max", "existingNativeBound": True,
                    "selectedCeilingNativeBounded": limit <= selected_bytes,
                    "hardRssIsolation": False, "scope": "sampled-managed-family-cgroup-v2",
                    "limitBytes": limit}
    except (OSError, ValueError):
        pass
    return result
