"""Fresh, bounded Darwin process-family RSS without launching a command.

The public libproc ABI is in libproc.h and sys/proc_info.h. Only the library
handle is cached. Unused BSD fields stay opaque: no process names, command
lines, paths, or credentials are decoded, persisted, or logged.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import lru_cache
import sys
import time


PROC_PPID_ONLY = 6
PROC_PIDTASKALLINFO = 2
MAX_SAMPLE_SECONDS = 1.0


class ProcBsdInfo(ctypes.Structure):
    # The opaque ranges include fixed-width fields and name arrays that this
    # numeric-only probe does not use. Apple defines MAXCOMLEN as 16 here.
    _fields_ = [
        ("_header", ctypes.c_uint32 * 3),
        ("pid", ctypes.c_uint32), ("parent_pid", ctypes.c_uint32),
        ("_unused", ctypes.c_ubyte * 100),
        ("start_seconds", ctypes.c_uint64),
        ("start_microseconds", ctypes.c_uint64),
    ]


class ProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("_virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("_unused", ctypes.c_ubyte * 80),
    ]


class ProcTaskAllInfo(ctypes.Structure):
    _fields_ = [("bsd", ProcBsdInfo), ("task", ProcTaskInfo)]


@dataclass(frozen=True)
class _ProcessInfo:
    pid: int
    parent_pid: int
    rss_bytes: int
    started: tuple[int, int]


@lru_cache(maxsize=1)
def _api():
    if sys.platform != "darwin":
        raise OSError("Darwin process APIs are unavailable")
    library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    library.proc_listpids.argtypes = [ctypes.c_uint32, ctypes.c_uint32,
                                      ctypes.c_void_p, ctypes.c_int]
    library.proc_listpids.restype = ctypes.c_int
    library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                                     ctypes.c_void_p, ctypes.c_int]
    library.proc_pidinfo.restype = ctypes.c_int
    return library


def _process_info(library, pid: int) -> _ProcessInfo:
    value = ProcTaskAllInfo()
    size = ctypes.sizeof(value)
    if library.proc_pidinfo(pid, PROC_PIDTASKALLINFO, 0, ctypes.byref(value), size) != size:
        raise OSError("Darwin managed process identity/residency is unavailable")
    bsd = value.bsd
    if (bsd.pid != pid or bsd.parent_pid > (1 << 31) - 1
            or bsd.start_seconds == 0 or bsd.start_microseconds >= 1_000_000):
        raise ValueError("Darwin process identity is invalid")
    return _ProcessInfo(pid, int(bsd.parent_pid), int(value.task.resident_size),
                        (int(bsd.start_seconds), int(bsd.start_microseconds)))


def _child_pids(library, pid: int, buffer) -> frozenset[int]:
    # proc_listpids returns BYTES, and its wrapper returns zero (not -1) on
    # syscall failure. Clear errno so an empty child list is distinguishable.
    ctypes.set_errno(0)
    size = ctypes.sizeof(buffer)
    received = library.proc_listpids(PROC_PPID_ONLY, pid, buffer, size)
    if received < 0 or (received == 0 and ctypes.get_errno()):
        raise OSError("Darwin managed child inventory is unavailable")
    width = ctypes.sizeof(ctypes.c_int)
    if received >= size or received % width:
        raise ValueError("Darwin process inventory is truncated or incompatible")
    count = received // width
    children = frozenset(int(buffer[index]) for index in range(count))
    if len(children) != count or any(child < 1 for child in children):
        raise ValueError("Darwin child identity inventory is invalid")
    return children


def family_rows(root_pid: int, *, max_process_rows: int) -> dict[int, tuple[int, int]]:
    """Return PID -> (actual PPID, RSS) after two fresh family observations.

    Neither pass is an atomic OS snapshot. A second parent/start-time/child
    check rejects observed exit, PID reuse, reparenting, and topology changes.
    The entire observation must finish within one second; this is a validity
    deadline checked around each synchronous OS call, not kernel-call
    cancellation or an OS-enforced allocation limit. The caller timestamps
    the observation before entry and still forces a new one for allocations.
    """
    if (type(root_pid) is not int or not 1 <= root_pid <= (1 << 31) - 1
            or type(max_process_rows) is not int or max_process_rows < 1):
        raise ValueError("Darwin managed process inventory bounds are invalid")
    deadline = time.monotonic() + MAX_SAMPLE_SECONDS

    def within_window():
        if time.monotonic() >= deadline:
            raise OSError("Darwin managed process observation exceeded its time bound")

    library = _api()
    # One extra slot makes a full/truncated buffer fail closed. Reuse this
    # sample-local numeric buffer for every parent; never retain an inventory.
    buffer = (ctypes.c_int * (max_process_rows + 1))()
    records: dict[int, _ProcessInfo] = {}
    children: dict[int, frozenset[int]] = {}
    discovered = {root_pid}
    pending: list[tuple[int, int | None]] = [(root_pid, None)]
    while pending:
        pid, expected_parent = pending.pop()
        within_window()
        info = _process_info(library, pid)
        within_window()
        # PROC_PPID_ONLY also matches the original parent of a traced child.
        # Validate the current PPID explicitly rather than assuming ownership.
        if expected_parent is not None and info.parent_pid != expected_parent:
            raise ValueError("Darwin managed process parent changed")
        records[pid] = info
        children[pid] = _child_pids(library, pid, buffer)
        within_window()
        for child in children[pid]:
            if child in discovered:
                raise ValueError("Darwin managed process descendant cycle or duplicate")
            discovered.add(child)
            if len(discovered) > max_process_rows:
                raise ValueError("Darwin managed process inventory exceeds its bound")
            pending.append((child, pid))

    result = {}
    for pid, previous in records.items():
        within_window()
        current = _process_info(library, pid)
        within_window()
        if (current.parent_pid, current.started) != (previous.parent_pid, previous.started):
            raise ValueError("Darwin managed process identity changed during observation")
        if _child_pids(library, pid, buffer) != children[pid]:
            raise ValueError("Darwin managed process family changed during observation")
        within_window()
        # Conservative within the observed interval; shared pages retain the
        # existing deliberate double counting across managed processes.
        result[pid] = (current.parent_pid, max(previous.rss_bytes, current.rss_bytes))
    return result
