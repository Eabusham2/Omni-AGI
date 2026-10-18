"""Read Darwin memory counters directly; never fork on a neural hot path.

ABI: XNU mach/vm_statistics.h and mach/host_info.h, HOST_VM_INFO64_REV0.
Only library/host-port/page-size handles are cached, not available memory.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
import os
import sys


class VMStatistics64Rev0(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in
                ("free_count", "active_count", "inactive_count", "wire_count")]
    _fields_ += [(name, ctypes.c_uint64) for name in
                 ("zero_fill_count", "reactivations", "pageins", "pageouts", "faults",
                  "cow_faults", "lookups", "hits", "purges")]
    _fields_ += [("purgeable_count", ctypes.c_uint32), ("speculative_count", ctypes.c_uint32)]


def available_bytes(total, page_size, stats, percent=None):
    if total <= 0 or page_size <= 0:
        raise ValueError("Darwin memory geometry is invalid")
    # Preserve the existing admission policy. Speculative pages are already
    # included in free_count, so must never be added again.
    available = (int(stats.free_count) + int(stats.inactive_count)
                 + int(stats.purgeable_count)) * page_size
    if type(percent) is int and 0 <= percent <= 100:
        available = max(available, total * percent // 100)
    return min(total, max(0, available))


@lru_cache(maxsize=1)
def _api():
    if sys.platform != "darwin":
        raise OSError("Darwin memory APIs are unavailable")
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    library.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t]
    library.sysctlbyname.restype = ctypes.c_int
    library.mach_host_self.argtypes = []
    library.mach_host_self.restype = ctypes.c_uint32
    library.host_statistics64.argtypes = [ctypes.c_uint32, ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    library.host_statistics64.restype = ctypes.c_int
    # One process-lifetime host send right. Do not leak one per measurement.
    host = library.mach_host_self()
    if not host:
        raise OSError("Darwin host port unavailable")
    return library, host, int(os.sysconf("SC_PAGE_SIZE"))


def _sysctl(library, name, value_type):
    value = value_type()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if library.sysctlbyname(name, ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "Darwin memory sysctl failed")
    if size.value != ctypes.sizeof(value):
        raise OSError("Darwin memory sysctl returned an incompatible size")
    return int(value.value)


def memory_reading():
    library, host, page_size = _api()
    total = _sysctl(library, b"hw.memsize", ctypes.c_uint64)
    stats = VMStatistics64Rev0()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32))
    if (library.host_statistics64(host, 4, ctypes.byref(stats), ctypes.byref(count)) != 0
            or count.value != ctypes.sizeof(stats) // ctypes.sizeof(ctypes.c_int32)):
        raise OSError("Darwin VM statistics unavailable or incompatible")
    try:
        percent = _sysctl(library, b"kern.memorystatus_level", ctypes.c_uint32)
    except OSError:
        percent = None
    return total, available_bytes(total, page_size, stats, percent)
