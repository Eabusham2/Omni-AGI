"""Durable, safe-tensor neural-state offload and resource governance.

The store deliberately does not use ``torch.save`` or pickle.  Replay is an
append-only-by-default SQLite journal and checkpoint generations point at
content-addressed safe-tensor blobs.  A checkpoint becomes authoritative only
when the caller places its pointer in ``brain.json``.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid
import tempfile
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union, overload

import torch

from .persistence import (
    atomic_save_tensors,
    atomic_write_bytes,
    atomic_write_json,
    load_tensors,
    read_json,
)
from .managed_process_memory import default_managed_process_sampler, native_family_memory_containment
from .shared_resource_ledger import SharedResourceLedger, SharedQuotaPause


GIB = 1024 ** 3
MIB = 1024 ** 2
MIN_TRAINING_WINDOW_TOKENS = 64
SEQUENTIAL_SCRATCH_CHUNK_BYTES = 8 * MIB


class NeuralStateResourcePause(RuntimeError):
    """A recoverable pause raised before a reserve watermark is crossed."""

    def __init__(self, message: str, status: Mapping[str, Any]):
        super().__init__(message)
        self.status = dict(status)


@dataclass(frozen=True)
class ResourceReading:
    total_memory_bytes: Optional[int]
    available_memory_bytes: Optional[int]
    process_memory_bytes: Optional[int]
    disk_total_bytes: int
    disk_free_bytes: int
    accelerator_total_memory_bytes: Optional[int] = None
    accelerator_free_memory_bytes: Optional[int] = None
    accelerator_allocated_memory_bytes: Optional[int] = None
    process_peak_memory_bytes: Optional[int] = None
    managed_process_memory_bytes: Optional[int] = None
    managed_worker_rss_bytes: Optional[int] = None
    managed_memory_verified: Optional[bool] = None
    managed_memory_scope: str = "single-process-provider"
    managed_memory_root_pid: Optional[int] = None
    managed_memory_process_count: int = 1
    managed_memory_cached: bool = False
    managed_memory_sample_duration_ms: float = 0.0
    managed_memory_sample_age_seconds: float = 0.0
    managed_memory_sample_started_ns: Optional[int] = None
    managed_process_pids: Tuple[int, ...] = ()


_MAC_FOOTPRINT_CACHE_SECONDS = 1.0
_mac_footprint_cache: Tuple[float, Optional[int], Optional[int]] = (
    0.0,
    None,
    None,
)
_mac_footprint_lock = threading.Lock()


def _linux_memory() -> Tuple[Optional[int], Optional[int]]:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None, None
    values: Dict[str, int] = {}
    try:
        for line in path.read_text("utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None, None
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if available is None:
        available = sum(
            values.get(key, 0)
            for key in ("MemFree", "Buffers", "Cached", "SReclaimable")
        )
    if total is None or available is None:
        return total, available
    # cgroup v2 first, then v1. /proc/meminfo often reports host RAM from
    # inside a container even when the neural worker has a much smaller limit.
    for limit_path, current_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            limit_text = limit_path.read_text("ascii").strip()
            if limit_text == "max":
                continue
            constrained = _apply_memory_limit(
                total,
                available,
                int(limit_text),
                int(current_path.read_text("ascii").strip()),
            )
            if constrained[0] < total:
                return constrained
        except (OSError, ValueError):
            continue
    return total, available


def _apply_memory_limit(
    total: int,
    available: int,
    limit: int,
    current: int,
) -> Tuple[int, int]:
    """Clamp a host reading to a process/container memory ceiling."""

    if limit <= 0 or current < 0 or limit >= total:
        return total, available
    return limit, max(0, min(available, limit - current))


def _mac_memory() -> Tuple[Optional[int], Optional[int]]:
    try:
        total = int(
            subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        )
        output = subprocess.check_output(
            ["vm_stat"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        first = output.splitlines()[0]
        page_size = int(first.split("page size of", 1)[1].split("bytes", 1)[0])
        pages: Dict[str, int] = {}
        for line in output.splitlines()[1:]:
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            pages[key.strip()] = int(raw.strip().rstrip("."))
        # vm_statistics reports speculative pages inside the free-page bucket
        # on current macOS; adding both would double count reclaimable RAM.
        available_pages = sum(
            pages.get(key, 0)
            for key in (
                "Pages free",
                "Pages inactive",
                "Pages purgeable",
            )
        )
        vm_available = available_pages * page_size
        # `vm_stat` is deliberately conservative and may omit clean pages the
        # kernel can reclaim without swapping.  Apple exposes that effective
        # pool through `memory_pressure -Q`; using the larger measured value
        # avoids rejecting a foundation that the same host can safely load.
        try:
            pressure_output = subprocess.check_output(
                ["memory_pressure", "-Q"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            pressure_available = _mac_memory_pressure_available(
                pressure_output, total
            )
        except (OSError, subprocess.SubprocessError):
            pressure_available = None
        return total, max(vm_available, int(pressure_available or 0))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None, None


def _mac_memory_pressure_available(
    text: str, total_memory_bytes: int
) -> Optional[int]:
    """Parse Apple's reclaimable-memory percentage without overcommitting."""

    import re

    match = re.search(
        r"System-wide memory free percentage:\s*(\d+(?:\.\d+)?)%",
        text,
        flags=re.IGNORECASE,
    )
    total = int(total_memory_bytes)
    if match is None or total <= 0:
        return None
    try:
        percent = float(match.group(1))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(percent) or percent < 0.0 or percent > 100.0:
        return None
    return min(total, max(0, math.floor(total * percent / 100.0)))


def _windows_memory() -> Tuple[Optional[int], Optional[int]]:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        return int(status.total_physical), int(status.available_physical)
    except (OSError, ValueError, AttributeError):
        return None, None


def _fallback_memory() -> Tuple[Optional[int], Optional[int]]:
    try:
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        return total_pages * page_size, available_pages * page_size
    except (OSError, ValueError, AttributeError):
        return None, None


def _parse_mac_process_footprint(
    text: str,
) -> Tuple[Optional[int], Optional[int]]:
    """Return Activity-Monitor-style current and peak physical footprint."""

    import re

    current_match = re.search(r"\bphys_footprint:\s*(\d+)\s+B\b", text)
    peak_match = re.search(r"\bphys_footprint_peak:\s*(\d+)\s+B\b", text)
    current = int(current_match.group(1)) if current_match else None
    peak = int(peak_match.group(1)) if peak_match else None
    if current is not None and peak is None:
        peak = current
    if current is not None and peak is not None:
        peak = max(current, peak)
    return current, peak


def _mac_process_rusage() -> Tuple[Optional[int], Optional[int]]:
    """Read macOS physical footprint through libproc without spawning a tool."""

    try:
        import ctypes

        names = (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
            "cpu_time_qos_default",
            "cpu_time_qos_maintenance",
            "cpu_time_qos_background",
            "cpu_time_qos_utility",
            "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive",
            "billed_system_time",
            "serviced_system_time",
            "logical_writes",
            "lifetime_max_phys_footprint",
            "instructions",
            "cycles",
            "billed_energy",
            "serviced_energy",
            "interval_max_phys_footprint",
            "runnable_time",
        )

        class RUsageInfoV4(ctypes.Structure):
            _fields_ = [
                ("uuid", ctypes.c_uint8 * 16),
                *((name, ctypes.c_uint64) for name in names),
            ]

        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        library.proc_pid_rusage.restype = ctypes.c_int
        usage = RUsageInfoV4()
        # RUSAGE_INFO_V4 includes both live phys_footprint and its lifetime max.
        if library.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) != 0:
            return None, None
        current = int(usage.phys_footprint)
        peak = max(current, int(usage.lifetime_max_phys_footprint))
        return (current or None), (peak or None)
    except (ImportError, OSError, ValueError, AttributeError):
        return None, None


def _mac_process_memory(*, fresh: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """Measure compressed-aware physical use without slowing neural steps."""

    global _mac_footprint_cache
    now = time.monotonic()
    with _mac_footprint_lock:
        cached_at, cached_current, cached_peak = _mac_footprint_cache
        if not fresh and now - cached_at < _MAC_FOOTPRINT_CACHE_SECONDS:
            return cached_current, cached_peak
        current, peak = _mac_process_rusage()
        if current is None:
            try:
                output = subprocess.check_output(
                    [
                        "/usr/bin/footprint",
                        "--pid",
                        str(os.getpid()),
                        "--noCategories",
                        "--format",
                        "bytes",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                current, peak = _parse_mac_process_footprint(output)
            except (OSError, ValueError, subprocess.SubprocessError):
                current, peak = None, None
        if current is None:
            try:
                rss_kib = int(
                    subprocess.check_output(
                        ["ps", "-o", "rss=", "-p", str(os.getpid())],
                        text=True,
                        stderr=subprocess.DEVNULL,
                        timeout=2,
                    ).strip()
                )
                current = rss_kib * 1024
            except (OSError, ValueError, subprocess.SubprocessError):
                current = None
        if peak is None:
            peak = max(int(cached_peak or 0), int(current or 0)) or None
        else:
            peak = max(int(cached_peak or 0), int(peak))
        _mac_footprint_cache = (now, current, peak)
        return current, peak


def _process_memory(*, fresh: bool = False) -> Tuple[Optional[int], Optional[int]]:
    try:
        if Path("/proc/self/statm").is_file():
            resident_pages = int(
                Path("/proc/self/statm").read_text("ascii").split()[1]
            )
            current = resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
            return current, None
        if os.name == "nt":
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return (
                    int(counters.working_set_size),
                    int(counters.peak_working_set_size),
                )
            return None, None
        if os.sys.platform == "darwin":
            return _mac_process_memory(fresh=fresh)
        # Portable fallback for platforms without a current-RSS API. This is
        # intentionally last because ru_maxrss is a peak, not live residency.
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak = value if os.sys.platform == "darwin" else value * 1024
        return None, peak
    except (
        ImportError,
        OSError,
        ValueError,
        AttributeError,
        IndexError,
        subprocess.SubprocessError,
    ):
        return None, None


def _accelerator_memory() -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Return best-effort accelerator totals without allocating a probe tensor.

    CUDA has a dedicated allocator. Apple silicon is unified memory, but MPS'
    recommended working set still gives the planner a useful second pressure
    boundary. DirectML does not expose a stable cross-vendor memory API, so it
    deliberately falls back to the physical-RAM watermark.
    """

    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            return int(total), int(free), int(allocated)
    except (RuntimeError, ValueError, AttributeError):
        pass
    try:
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and hasattr(torch, "mps")
        ):
            allocated = int(torch.mps.current_allocated_memory())
            driver = int(torch.mps.driver_allocated_memory())
            recommended = (
                int(torch.mps.recommended_max_memory())
                if hasattr(torch.mps, "recommended_max_memory")
                else None
            )
            free = (
                max(0, recommended - driver)
                if recommended is not None
                else None
            )
            return recommended, free, allocated
    except (RuntimeError, ValueError, AttributeError):
        pass
    return None, None, None


class _PolicyResourceLease:
    def __init__(self, lease, operation, status):
        self._lease, self.operation, self.status = lease, operation, status

    def _call(self, name, *args, **kwargs):
        try:
            return getattr(self._lease, name)(*args, **kwargs)
        except SharedQuotaPause as error:
            raise NeuralStateResourcePause(self.operation + ": " + str(error),
                {**self.status, "sharedQuota": error.status, "paused": True, "recoverable": True}) from error

    def bind_path(self, path):
        self._call("bind_path", path); return self

    def mark_allocated(self, actual_bytes=None):
        self._call("mark_allocated", actual_bytes); return self

    def commit(self, actual_allocated_bytes=None, **kwargs):
        self._call("commit", actual_allocated_bytes, **kwargs); return self

    def release(self): return self._call("release")
    def backing_closed(self): return self._call("backing_closed")
    def __enter__(self): return self
    def __exit__(self, kind, value, traceback): return self._call("__exit__", kind, value, traceback)


class ResourcePolicy:
    """Resolve adaptive RAM/disk reserves and make preflight decisions."""

    def __init__(
        self,
        probe_path: Path,
        *,
        ram_reserve_bytes: int = 0,
        disk_reserve_bytes: int = 0,
        system_ram_share_percent: float = 0.0,
        storage_bytes_per_second: int = 0,
        hardware_tier: str = "personal",
        reading_provider: Optional[Callable[[], ResourceReading]] = None,
        include_accelerator_memory: bool = True,
        shared_resource_owner_id: Optional[str] = None,
        shared_storage_pool_bytes: int = 0,
        shared_ledger: Optional[SharedResourceLedger] = None,
    ):
        self.probe_path = Path(probe_path)
        self.configured_ram_reserve = max(0, int(ram_reserve_bytes))
        self.configured_disk_reserve = max(0, int(disk_reserve_bytes))
        share = float(system_ram_share_percent)
        if share != 0.0 and not 30.0 <= share <= 100.0:
            raise ValueError("system RAM share must be auto (0) or 30-100 percent")
        self.configured_system_ram_share_percent = share
        self.storage_bytes_per_second = max(0, int(storage_bytes_per_second))
        if hardware_tier not in {"micro", "personal", "gpu", "workstation"}:
            raise ValueError("unsupported hardware tier")
        self.hardware_tier = hardware_tier
        self.reading_provider = reading_provider
        self.include_accelerator_memory = bool(include_accelerator_memory)
        self.shared_resource_owner_id = shared_resource_owner_id or str(self.probe_path.absolute())
        self.shared_storage_pool_bytes = max(0, int(shared_storage_pool_bytes))
        self._shared_ledger_instance = shared_ledger
        self._shared_owner_registered = False
        self._shared_ledger_lock = threading.RLock()
        self._active_ram_lock = threading.RLock()
        self._active_ram_watchers = 0
        self._active_ram_generation = 0
        self._active_ram_interval_seconds = 0.25
        self._active_ram_stop: Optional[threading.Event] = None
        self._active_ram_thread: Optional[threading.Thread] = None
        self._active_ram_status: Optional[Dict[str, Any]] = None
        self._active_ram_sampled_at = float("-inf")

    @property
    def shared_ledger(self) -> SharedResourceLedger:
        with self._shared_ledger_lock:
            if self._shared_ledger_instance is None:
                configured = os.environ.get("OMNI_SHARED_RESOURCE_LEDGER")
                path = Path(configured) if configured else Path(tempfile.gettempdir()) / ("omni-resource-ledger-%d" % os.getpid()) / "quota.sqlite3"
                self._shared_ledger_instance = SharedResourceLedger(path)
            if not self._shared_owner_registered:
                self._shared_ledger_instance.register_owner(self.shared_resource_owner_id, self.shared_storage_pool_bytes,
                    preserve_existing_pool=self.shared_storage_pool_bytes == 0)
                self._shared_owner_registered = True
            return self._shared_ledger_instance

    def configure_shared_resources(self, *, owner_id: str, storage_pool_bytes: int):
        with self._shared_ledger_lock:
            self.shared_resource_owner_id = owner_id
            self.shared_storage_pool_bytes = max(0, int(storage_pool_bytes))
            self.shared_ledger.register_owner(owner_id, self.shared_storage_pool_bytes)
            self._shared_owner_registered = True

    @staticmethod
    def _observed_memory_epoch(status):
        # Use the START of the known cached family sample, never retire RAM
        # escrow merely because a later caller read an old cached result.
        return status.get("managedMemorySampleStartedNs")

    def reserve_ram(self, estimated_bytes: int, operation: str = "native RAM allocation"):
        # Every new allocation receives a fresh family sample. A one-second
        # UI/status cache must not permit multiple independent over-admissions.
        status = self.status(estimated_ram_bytes=estimated_bytes, fresh_memory_sample=True)
        if status["memoryPressure"]:
            raise NeuralStateResourcePause(operation + " paused at selected RAM ceiling", status)
        try:
            lease = self.shared_ledger.reserve(self.shared_resource_owner_id, "ram", max(0, int(estimated_bytes)),
                ram_budget_bytes=status["systemRamBudgetBytes"], observed_ram_bytes=status["admissionResidentMemoryBytes"],
                observed_ns=self._observed_memory_epoch(status), verified=status["ramAdmissionVerified"])
            return _PolicyResourceLease(lease, operation, status)
        except SharedQuotaPause as error:
            raise NeuralStateResourcePause(operation + ": " + str(error), {**status, "sharedQuota": error.status}) from error

    def reserve_spill(self, estimated_bytes: int, operation: str = "native spill allocation", allocation_key=None):
        status = self.require_disk(estimated_bytes, operation)
        try:
            ledger = self.shared_ledger
            # Legacy standalone CLI has no main registry and zero means Auto,
            # not a zero-byte HDD pool. Resolve its physical allowance once;
            # never overwrite a trusted app registry or another larger owner.
            if not os.environ.get("OMNI_SHARED_RESOURCE_LEDGER") and self.shared_storage_pool_bytes == 0:
                quota = ledger.status()
                if quota["largestConfiguredPoolBytes"] == 0:
                    available = max(0, int(status["diskFreeBytes"]) - int(status["diskReserveBytes"]))
                    ledger.register_owner(self.shared_resource_owner_id, available)
            lease = ledger.reserve(self.shared_resource_owner_id, "spill", max(0, int(estimated_bytes)))
            return _PolicyResourceLease(lease, operation, status)
        except SharedQuotaPause as error:
            raise NeuralStateResourcePause(operation + ": " + str(error), {**status, "sharedQuota": error.status}) from error

    def reconcile_shared_owner(self, *, max_entries: int = 256, after_identity: str = ""):
        return self.shared_ledger.reconcile_owner(self.shared_resource_owner_id, max_entries=max_entries, after_identity=after_identity)

    def _record_active_ram_sample(self, status: Dict[str, Any], generation: Optional[int] = None) -> None:
        with self._active_ram_lock:
            if generation is None or generation == self._active_ram_generation:
                self._active_ram_status = status
                self._active_ram_sampled_at = time.monotonic()

    def _fresh_active_ram_status(self) -> Dict[str, Any]:
        try:
            return self.status(fresh_memory_sample=True)
        except Exception as error:
            # An unavailable sampler/ledger cannot be interpreted as zero
            # residency. The foreground owner receives a recoverable pause.
            return {"memoryPressure": True, "ramAdmissionVerified": False,
                "paused": True, "recoverable": True, "activeWatchdogError": str(error)}

    def _active_ram_watch(self, stop: threading.Event, generation: int) -> None:
        while not stop.is_set():
            status = self._fresh_active_ram_status()
            self._record_active_ram_sample(status, generation)
            with self._active_ram_lock:
                next_interval = self._active_ram_interval_seconds
            stop.wait(next_interval)

    @contextmanager
    def active_operation(self, operation: str, *, interval_seconds: float = 0.25):
        """Monitor an active operation; callers pause at their safe boundaries.

        A daemon only observes. It never kills a worker, mutates neural state,
        changes the selected geometry/cap, or raises from another thread.
        Concurrent chat/inline jobs share one monitor for this resource policy.
        """
        del operation
        interval = float(interval_seconds)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("active RAM monitor interval is invalid")
        interval = max(0.05, interval)
        with self._active_ram_lock:
            self._active_ram_watchers += 1
            if self._active_ram_watchers == 1:
                self._active_ram_generation += 1
                generation = self._active_ram_generation
                self._active_ram_interval_seconds = interval
                self._active_ram_status = None
                self._active_ram_sampled_at = float("-inf")
                stop = threading.Event()
                self._active_ram_stop = stop
                thread = threading.Thread(target=self._active_ram_watch,
                    args=(stop, generation), name="omni-active-ram-watch", daemon=True)
                self._active_ram_thread = thread
                try:
                    thread.start()
                except BaseException:
                    self._active_ram_watchers -= 1
                    self._active_ram_stop = None
                    self._active_ram_thread = None
                    raise
            else:
                self._active_ram_interval_seconds = min(self._active_ram_interval_seconds, interval)
        try:
            yield self
        finally:
            with self._active_ram_lock:
                self._active_ram_watchers -= 1
                if self._active_ram_watchers == 0:
                    if self._active_ram_stop is not None:
                        self._active_ram_stop.set()
                    self._active_ram_stop = None
                    self._active_ram_thread = None
                    self._active_ram_status = None
                    self._active_ram_sampled_at = float("-inf")

    def check_active_pressure(self, operation: str, *, reclaim: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        """At a safe boundary, reclaim once and pause if fresh pressure remains.

        The watcher makes active peaks visible without making each token pay
        for a full process inventory. Positive/stale samples are rechecked
        before pausing, so a closed application can let the work resume with
        exactly the same selected memory budget and neural geometry.
        """
        with self._active_ram_lock:
            active = self._active_ram_watchers > 0
            cached = self._active_ram_status
            age = time.monotonic() - self._active_ram_sampled_at
            interval = self._active_ram_interval_seconds
        if active and cached is not None and not cached.get("memoryPressure") and age < interval:
            return cached
        status = self._fresh_active_ram_status()
        if status["memoryPressure"] and reclaim is not None:
            reclaim()
            status = self._fresh_active_ram_status()
        self._record_active_ram_sample(status)
        if status["memoryPressure"]:
            raise NeuralStateResourcePause(operation + " paused at the active selected RAM ceiling",
                {**status, "paused": True, "recoverable": True, "activeWatchdog": active})
        return status

    def readings(self, *, fresh_memory_sample: bool = False) -> ResourceReading:
        if self.reading_provider is not None:
            return self.reading_provider()
        probe = self.probe_path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        disk = shutil.disk_usage(str(probe))
        if os.name == "nt":
            total, available = _windows_memory()
        elif os.sys.platform == "darwin":
            total, available = _mac_memory()
        elif os.sys.platform.startswith("linux"):
            total, available = _linux_memory()
        else:
            total, available = _fallback_memory()
        if total is None or available is None:
            fallback_total, fallback_available = _fallback_memory()
            total = total if total is not None else fallback_total
            available = available if available is not None else fallback_available
        accelerator_total, accelerator_free, accelerator_allocated = (
            _accelerator_memory()
            if self.include_accelerator_memory
            else (None, None, None)
        )
        process_memory, process_peak_memory = (
            _process_memory(fresh=True) if fresh_memory_sample else _process_memory()
        )
        managed = default_managed_process_sampler().sample(force=fresh_memory_sample)
        return ResourceReading(
            total_memory_bytes=total,
            available_memory_bytes=available,
            process_memory_bytes=process_memory,
            disk_total_bytes=int(disk.total),
            disk_free_bytes=int(disk.free),
            accelerator_total_memory_bytes=accelerator_total,
            accelerator_free_memory_bytes=accelerator_free,
            accelerator_allocated_memory_bytes=accelerator_allocated,
            process_peak_memory_bytes=process_peak_memory,
            managed_process_memory_bytes=managed.rss_bytes,
            managed_worker_rss_bytes=managed.worker_rss_bytes,
            managed_memory_verified=managed.verified,
            managed_memory_scope=managed.scope,
            managed_memory_root_pid=managed.root_pid,
            managed_memory_process_count=managed.process_count,
            managed_memory_cached=managed.cached,
            managed_memory_sample_duration_ms=managed.sample_duration_ms,
            managed_memory_sample_age_seconds=managed.sample_age_seconds,
            managed_memory_sample_started_ns=managed.sample_started_ns,
            managed_process_pids=managed.owned_pids,
        )

    @staticmethod
    def _adaptive_ram_reserve(total: Optional[int]) -> int:
        if total is None or total <= 0:
            return 384 * MIB
        if total <= 4 * GIB:
            return min(
                384 * MIB,
                max(128 * MIB, math.ceil(total * 0.09375)),
            )
        if total <= 8 * GIB:
            return 512 * MIB
        if total <= 16 * GIB:
            return 1 * GIB
        return min(2 * GIB, max(1 * GIB, math.ceil(total * 0.0625)))

    @staticmethod
    def _adaptive_disk_reserve(total: int) -> int:
        # The user repeated a twenty-GiB minimum for every local brain. The
        # small-device exception concerned RAM, not this disk watermark. Keep
        # an undersized or unmeasured volume paused instead of lowering it.
        del total
        return 20 * GIB

    @staticmethod
    def _adaptive_system_ram_share(
        hardware_tier: str, storage_bytes_per_second: int = 0
    ) -> float:
        """Dedicated Omni share of memory remaining after the OS reserve."""

        base = {
            "micro": 55.0,
            "personal": 65.0,
            "gpu": 75.0,
            "workstation": 80.0,
        }.get(hardware_tier, 65.0)
        storage_rate = max(0, int(storage_bytes_per_second))
        if 0 < storage_rate < 100 * MIB:
            base += 10.0
        elif 0 < storage_rate < 500 * MIB:
            base += 5.0
        return min(90.0, base)

    def _system_ram_envelope(
        self,
        reading: ResourceReading,
        ram_reserve: int,
        *,
        estimated_ram_bytes: int = 0,
    ) -> Tuple[int, int, float]:
        # This is persisted *capacity*, not a snapshot of whatever RAM happens
        # to be free while another application is open. Current availability
        # is measured separately by ``status`` and may make an operation wait,
        # but it must never silently shrink the configured cortex/context.
        safe_pool = max(
            0, int(reading.total_memory_bytes or 0) - ram_reserve
        )
        share = (
            self.configured_system_ram_share_percent
            or self._adaptive_system_ram_share(
                self.hardware_tier,
                self.storage_bytes_per_second,
            )
        )
        # Auto's recommendation is also a ceiling, not permission to raise
        # the selected share merely because the next allocation is larger.
        # Spill eligible hot data, choose smaller exhaustive windows, or pause.
        del estimated_ram_bytes
        return safe_pool, int(safe_pool * share / 100.0), float(share)

    @staticmethod
    def _admission_resident_memory(reading: ResourceReading) -> Tuple[Optional[int], str, bool]:
        current, peak = reading.process_memory_bytes, reading.process_peak_memory_bytes
        family, worker_rss = reading.managed_process_memory_bytes, reading.managed_worker_rss_bytes
        if type(current) is int and current >= 0:
            local, basis = current, "current-worker-residency"
        elif reading.managed_memory_verified is True and type(worker_rss) is int and worker_rss > 0:
            local, basis = worker_rss, "measured-family-worker-rss-fallback"
        elif type(peak) is int and peak >= 0:
            local, basis = peak, "conservative-worker-peak-fallback"
        else:
            return None, "unmeasured-worker-residency", False
        if type(family) is int and family >= 0 and reading.managed_memory_verified is True:
            # On macOS the primary worker footprint can exceed RSS (compressed
            # or driver-owned pages). Floor the local contribution at it.
            others = max(0, family - worker_rss) if type(worker_rss) is int and worker_rss >= 0 else family
            aggregate = max(family, others + local)
            return aggregate, "managed-family-rss-conservative-shared-pages-double-counted", basis in {"current-worker-residency", "measured-family-worker-rss-fallback"}
        if reading.managed_memory_verified is False:
            return local, basis + ";managed-family-unmeasured", False
        return local, basis, basis == "current-worker-residency"

    def status(
        self,
        *,
        estimated_write_bytes: int = 0,
        estimated_ram_bytes: int = 0,
        fresh_memory_sample: bool = False,
    ) -> Dict[str, Any]:
        reading = self.readings(fresh_memory_sample=fresh_memory_sample)
        ram_reserve = self.configured_ram_reserve or self._adaptive_ram_reserve(
            reading.total_memory_bytes
        )
        disk_reserve = max(
            self.configured_disk_reserve,
            self._adaptive_disk_reserve(reading.disk_total_bytes),
        )
        projected_disk = reading.disk_free_bytes - max(
            0, int(estimated_write_bytes)
        )
        available = reading.available_memory_bytes
        safe_ram_pool, system_ram_budget, system_ram_share = (
            self._system_ram_envelope(
                reading,
                ram_reserve,
                estimated_ram_bytes=estimated_ram_bytes,
            )
        )
        admission_resident, accounting_basis, accounting_verified = self._admission_resident_memory(reading)
        projected_process_memory = None if admission_resident is None else admission_resident + max(0, int(estimated_ram_bytes))
        available_safe_ram = max(0, int(available or 0) - ram_reserve)
        # Already-resident Omni pages count toward its committed envelope. A
        # free-RAM-only reading would report false pressure as the model fills
        # the very capacity reserved for it.
        current_omni_available = min(
            system_ram_budget,
            available_safe_ram + int(admission_resident or 0),
        )
        current_omni_shortfall = max(
            0, system_ram_budget - current_omni_available
        )
        unknown_ram = (not accounting_verified or reading.total_memory_bytes is None or available is None)
        quota = None
        if self._shared_ledger_instance is not None or os.environ.get("OMNI_SHARED_RESOURCE_LEDGER"):
            quota = self.shared_ledger.status(observed_ns=reading.managed_memory_sample_started_ns, verified=accounting_verified)
            if quota["globalRamCeilingBytes"] is not None:
                system_ram_budget = min(system_ram_budget, int(quota["globalRamCeilingBytes"]))
            if projected_process_memory is not None:
                projected_process_memory += int(quota["ramEscrowBytes"])
            current_omni_available = min(current_omni_available, system_ram_budget)
            current_omni_shortfall = max(0, system_ram_budget - current_omni_available)
        native_containment = native_family_memory_containment(
            system_ram_budget,
            root_pid=int(reading.managed_memory_root_pid or 0),
            worker_pid=os.getpid(), member_pids=reading.managed_process_pids,
        ) if self.reading_provider is None else {
            "mechanism": "injected-reading-unverified", "existingNativeBound": False,
            "selectedCeilingNativeBounded": False, "hardRssIsolation": False,
            "scope": "injected-reading"}
        memory_pressure = bool(
            unknown_ram
            or
            (
                available is not None
                and available - max(0, int(estimated_ram_bytes)) <= ram_reserve
            )
            or (projected_process_memory is not None and projected_process_memory > system_ram_budget)
        )
        disk_pressure = projected_disk <= disk_reserve
        return {
            "mode": "adaptive-reserve-watermarks",
            "totalMemoryBytes": reading.total_memory_bytes,
            "availableMemoryBytes": available,
            "processMemoryBytes": reading.process_memory_bytes,
            "processPeakMemoryBytes": reading.process_peak_memory_bytes,
            "admissionResidentMemoryBytes": admission_resident,
            "managedProcessMemoryBytes": reading.managed_process_memory_bytes,
            "managedMemoryRootPid": reading.managed_memory_root_pid,
            "managedMemoryProcessCount": reading.managed_memory_process_count,
            "managedMemoryScope": reading.managed_memory_scope,
            "managedMemorySampleCached": reading.managed_memory_cached,
            "managedMemorySampleDurationMs": reading.managed_memory_sample_duration_ms,
            "managedMemorySampleAgeSeconds": reading.managed_memory_sample_age_seconds,
            "managedMemorySampleStartedNs": reading.managed_memory_sample_started_ns,
            "managedMemoryCacheIntervalSeconds": 1.0,
            "memoryAccountingBasis": accounting_basis,
            "memoryAccountingVerified": accounting_verified,
            "ramAdmissionVerified": not unknown_ram,
            "ramCapMechanism": "cooperative-measured-family-plus-estimated-allocation-admission",
            "hardRssIsolation": False,
            "nativeMemoryContainment": native_containment,
            "crossProcessAtomicReservation": quota is not None,
            "sharedQuota": quota,
            "osPhysicalPagePinning": False,
            "acceleratorTotalMemoryBytes": (
                reading.accelerator_total_memory_bytes
            ),
            "acceleratorFreeMemoryBytes": (
                reading.accelerator_free_memory_bytes
            ),
            "acceleratorAllocatedMemoryBytes": (
                reading.accelerator_allocated_memory_bytes
            ),
            "ramReserveBytes": ram_reserve,
            "safeRamPoolBytes": safe_ram_pool,
            "availableSafeRamBytes": available_safe_ram,
            "systemRamBudgetBytes": system_ram_budget,
            "currentOmniAvailableBytes": current_omni_available,
            "currentOmniShortfallBytes": current_omni_shortfall,
            "systemRamSharePercent": system_ram_share,
            "hardwareTier": self.hardware_tier,
            "systemRamMode": (
                "manual"
                if self.configured_system_ram_share_percent
                else "auto"
            ),
            "projectedProcessMemoryBytes": projected_process_memory,
            "diskTotalBytes": reading.disk_total_bytes,
            "diskFreeBytes": reading.disk_free_bytes,
            "projectedDiskFreeBytes": projected_disk,
            "diskReserveBytes": disk_reserve,
            "mandatoryFreeDiskBytes": disk_reserve,
            "desktopRecommendedFreeDiskBytes": 20 * GIB,
            "estimatedWriteBytes": max(0, int(estimated_write_bytes)),
            "estimatedRamBytes": max(0, int(estimated_ram_bytes)),
            "storageBytesPerSecond": self.storage_bytes_per_second,
            "memoryPressure": memory_pressure,
            "diskPressure": disk_pressure,
            "paused": bool(memory_pressure or disk_pressure),
            "capacityPersistsAcrossPressure": True,
            "contextPagedToStorage": False,
            "waitForMemory": memory_pressure,
            "retryAfterSeconds": 5 if memory_pressure else 0,
            "userAction": (
                "RAM residency measurement is unavailable; retry under the selected ceiling."
                if unknown_ram else "Close memory-heavy applications, then retry; saved context "
                "capacity will not be reduced."
                if memory_pressure
                else None
            ),
        }

    def training_plan(
        self,
        *,
        max_window_tokens: int,
        requested_batch_size: int,
        requested_gradient_accumulation: int,
        effective_batch_target: Optional[int] = None,
        require_physical_batch_divisor: bool = False,
        trainable_parameter_bytes: int,
        activation_bytes_per_token: int,
        packed_update_scratch_bytes: int = 0,
        optimizer_state_resident: bool = False,
        resource_mode: str = "auto",
        manual_ram_budget_bytes: int = 0,
        manual_accelerator_budget_bytes: int = 0,
        manual_scratch_budget_bytes: int = 0,
        storage_bytes_per_second: int = 0,
        disk_state_offload: bool = True,
    ) -> Dict[str, Any]:
        """Choose a RAM-first, exhaustive streaming training shape.

        Reducing ``windowTokens`` never truncates a source: the tokenizer emits
        more windows until every byte has been visited. Scratch storage is not
        treated as virtual RAM. It is an emergency, restart-safe checkpoint
        tier with infrequent sequential writes, which avoids per-step storage
        wear and pathological slow-drive I/O.
        """

        if resource_mode not in {"auto", "manual"}:
            raise ValueError("training resource mode must be auto or manual")
        max_window_tokens = int(max_window_tokens)
        if max_window_tokens < 2:
            raise ValueError("training windows need at least two tokens for next-token labels")
        requested_batch_size = max(1, int(requested_batch_size))
        requested_gradient_accumulation = max(
            1, int(requested_gradient_accumulation)
        )
        resolved_effective_batch_target = max(
            1,
            int(
                effective_batch_target
                if effective_batch_target is not None
                else requested_batch_size
                * requested_gradient_accumulation
            ),
        )
        trainable_parameter_bytes = max(0, int(trainable_parameter_bytes))
        packed_update_scratch_bytes = max(0, int(packed_update_scratch_bytes))
        activation_bytes_per_token = max(
            1024, int(activation_bytes_per_token)
        )
        reading = self.readings()
        admission_resident, accounting_basis, accounting_verified = self._admission_resident_memory(reading)
        ram_reserve = self.configured_ram_reserve or self._adaptive_ram_reserve(
            reading.total_memory_bytes
        )
        optimizer_and_gradient_bytes = trainable_parameter_bytes * (
            1 if optimizer_state_resident else 3
        )
        # Packed synapses have no dense master/Adam moments. Their bounded
        # row-block decode and local-gradient scratch is charged once, not
        # multiplied as if it were another FP32 parameter array.
        learning_state_bytes = (
            optimizer_and_gradient_bytes + packed_update_scratch_bytes
        )
        # Two adjacent tokens are the actual mathematical minimum for shifted
        # labels. Sixty-four remains a preferred shape, never an allocation
        # floor that can override the measured live compute reservation.
        minimum_window = 2
        preferred_window = min(MIN_TRAINING_WINDOW_TOKENS, max_window_tokens)
        minimum_training_allocation = (
            learning_state_bytes
            + activation_bytes_per_token * minimum_window
        )
        safe_ram_pool, system_ram_budget, system_ram_share = (
            self._system_ram_envelope(
                reading,
                ram_reserve,
                estimated_ram_bytes=minimum_training_allocation,
            )
        )
        physical_ram_headroom = max(
            0,
            min(
                system_ram_budget - int(admission_resident or 0),
                int(reading.available_memory_bytes or 0) - ram_reserve,
            ),
        )
        if not accounting_verified or reading.total_memory_bytes is None or reading.available_memory_bytes is None:
            physical_ram_headroom = 0
        requested_ram_budget = max(0, int(manual_ram_budget_bytes))
        ram_budget = physical_ram_headroom
        warnings: list[str] = []
        if resource_mode == "manual" and requested_ram_budget:
            ram_budget = min(ram_budget, requested_ram_budget)
            if requested_ram_budget > physical_ram_headroom:
                warnings.append(
                    "Manual RAM budget was clamped to the live safe-memory watermark."
                )

        accelerator_free = reading.accelerator_free_memory_bytes
        accelerator_total = reading.accelerator_total_memory_bytes
        accelerator_reserve = (
            max(256 * MIB, min(2 * GIB, int(accelerator_total * 0.1)))
            if accelerator_total is not None
            else 0
        )
        accelerator_headroom = (
            max(0, int(accelerator_free) - accelerator_reserve)
            if accelerator_free is not None
            else None
        )
        requested_accelerator_budget = max(
            0, int(manual_accelerator_budget_bytes)
        )
        accelerator_budget = accelerator_headroom
        if accelerator_budget is not None and requested_accelerator_budget:
            accelerator_budget = min(
                accelerator_budget, requested_accelerator_budget
            )
            if requested_accelerator_budget > accelerator_headroom:
                warnings.append(
                    "Manual accelerator budget was clamped to the live safe-memory watermark."
                )

        # AdamW creates two moments and gradients in addition to the already
        # resident mutable parameters. The margin absorbs allocator
        # fragmentation, routing state, Python record objects, and OS jitter.
        def allocator_margin_for(budget: int) -> int:
            # The process/model baseline has already been measured and
            # subtracted above. Reserve a proportional part of a small
            # remaining partition instead of inventing a further 128-MiB
            # resident allocation that would reject otherwise valid windows.
            return max(
                1,
                math.ceil(max(0, budget) * 0.10),
                min(128 * MIB, math.ceil(max(0, budget) * 0.25)),
            )

        allocator_margin = allocator_margin_for(ram_budget)
        accelerator_margin = 0
        training_headroom = max(
            0,
            ram_budget - learning_state_bytes - allocator_margin,
        )
        if accelerator_budget is not None:
            accelerator_margin = allocator_margin_for(accelerator_budget)
            training_headroom = min(
                training_headroom,
                max(
                    0,
                    accelerator_budget
                    - learning_state_bytes
                    - accelerator_margin,
                ),
            )

        window_tokens = max_window_tokens
        affordable_window_tokens = training_headroom // activation_bytes_per_token
        bytes_per_full_sample = activation_bytes_per_token * window_tokens
        maximum_full_batch = training_headroom // max(1, bytes_per_full_sample)
        if maximum_full_batch < 1:
            window_tokens = min(
                max_window_tokens,
                max(
                    minimum_window,
                    affordable_window_tokens,
                ),
            )
        physical_batch_limit = min(
            requested_batch_size,
            max(
                1,
                training_headroom
                // max(1, activation_bytes_per_token * window_tokens),
            ),
        )
        if require_physical_batch_divisor:
            # Keep the optimizer's logical batch cardinality exact while Auto
            # spends more headroom on fewer physical microbatches. Restricting
            # the physical shape to a divisor makes accumulation integral and
            # lets an allocator fallback move 4 -> 2 -> 1 without silently
            # changing the number of records in each committed slow update.
            physical_batch = next(
                candidate
                for candidate in range(physical_batch_limit, 0, -1)
                if resolved_effective_batch_target % candidate == 0
            )
        else:
            physical_batch = physical_batch_limit
        minimum_step_bytes = (
            learning_state_bytes
            + allocator_margin
            + activation_bytes_per_token * minimum_window
        )
        pause_before_step = (
            ram_budget < minimum_step_bytes
            or (
                accelerator_budget is not None
                and accelerator_budget
                < learning_state_bytes
                + accelerator_margin
                + activation_bytes_per_token * minimum_window
            )
        )
        if window_tokens < max_window_tokens:
            warnings.append(
                "Training context was reduced to preserve memory; all source bytes still run through additional windows."
            )
        if physical_batch < requested_batch_size:
            warnings.append(
                "Physical batch size was reduced and replaced with RAM-only gradient accumulation."
            )

        effective_target = resolved_effective_batch_target
        accumulation = (
            effective_target // physical_batch
            if require_physical_batch_divisor
            else max(
                1,
                math.ceil(effective_target / max(1, physical_batch)),
            )
        )
        disk_reserve = max(
            20 * GIB,
            self.configured_disk_reserve,
            self._adaptive_disk_reserve(reading.disk_total_bytes),
        )
        disk_headroom = max(0, reading.disk_free_bytes - disk_reserve)
        requested_scratch_budget = max(0, int(manual_scratch_budget_bytes))
        scratch_budget = disk_headroom
        if resource_mode == "manual" and requested_scratch_budget:
            scratch_budget = min(scratch_budget, requested_scratch_budget)
            if requested_scratch_budget > disk_headroom:
                warnings.append(
                    "Manual scratch budget was clamped to the live safe-disk watermark."
                )
        emergency_checkpoint_bytes = (
            2 * trainable_parameter_bytes + 64 * MIB
        )
        storage_rate = max(0, int(storage_bytes_per_second))
        if storage_rate == 0:
            storage_class = "unknown-conservative"
            minimum_scratch_interval_seconds = 1800
        elif storage_rate < 100 * MIB:
            storage_class = "slow-storage"
            minimum_scratch_interval_seconds = 3600
        elif storage_rate < 500 * MIB:
            storage_class = "moderate-storage"
            minimum_scratch_interval_seconds = 900
        else:
            storage_class = "fast-storage"
            minimum_scratch_interval_seconds = 300
        scratch_available = bool(
            disk_state_offload
            and scratch_budget >= emergency_checkpoint_bytes
            and not self.status(
                estimated_write_bytes=emergency_checkpoint_bytes
            )["diskPressure"]
        )
        if pause_before_step:
            warnings.append(
                "Training must pause before the next step; scratch preserves progress but is never used as per-step virtual RAM."
            )

        return {
            "mode": resource_mode,
            "policy": "ram-first-adaptive-streaming",
            "physicalBatchRecords": int(physical_batch),
            "gradientAccumulation": int(accumulation),
            "effectiveBatchTarget": int(effective_target),
            "windowTokens": int(window_tokens),
            "requestedWindowTokens": int(max_window_tokens),
            "minimumWindowTokens": minimum_window,
            "preferredWindowTokens": preferred_window,
            "maximumAffordableWindowTokens": int(affordable_window_tokens),
            "admittedWindowTokens": 0 if pause_before_step else int(window_tokens),
            "windowOverlapTokens": 1,
            "labelTargetsCoveredOnce": True,
            "allSourceBytesVisited": True,
            "coverageConfirmed": False,
            "sourceCoverage": "exhaustive-overlapping-window-schedule-required",
            "pauseBeforeStep": bool(pause_before_step),
            "warnings": warnings,
            "memory": {
                "ramBudgetBytes": int(ram_budget),
                "ramHeadroomBytes": int(physical_ram_headroom),
                "safeRamPoolBytes": int(safe_ram_pool),
                "systemRamBudgetBytes": int(system_ram_budget),
                "systemRamSharePercent": float(system_ram_share),
                "systemRamMode": (
                    "manual"
                    if self.configured_system_ram_share_percent
                    else "auto"
                ),
                "ramReserveBytes": int(ram_reserve),
                "admissionResidentMemoryBytes": admission_resident,
                "memoryAccountingBasis": accounting_basis,
                "memoryAccountingVerified": accounting_verified,
                "acceleratorBudgetBytes": accelerator_budget,
                "acceleratorHeadroomBytes": accelerator_headroom,
                "acceleratorReserveBytes": int(accelerator_reserve),
                "optimizerAndGradientBytes": int(
                    optimizer_and_gradient_bytes
                ),
                "packedUpdateScratchBytes": int(packed_update_scratch_bytes),
                "activationBytesPerToken": int(activation_bytes_per_token),
                "allocatorMarginBytes": int(allocator_margin),
                "acceleratorAllocatorMarginBytes": int(accelerator_margin),
            },
            "scratch": {
                "available": scratch_available,
                "mode": "emergency-checkpoint-only",
                "usedAsVirtualRam": False,
                "activationPaging": False,
                "sequentialWrites": True,
                "sequentialChunkBytes": SEQUENTIAL_SCRATCH_CHUNK_BYTES,
                "budgetBytes": int(scratch_budget),
                "estimatedCheckpointBytes": int(
                    emergency_checkpoint_bytes
                ),
                "minimumWriteIntervalSeconds": int(
                    minimum_scratch_interval_seconds
                ),
                "storageBytesPerSecond": int(storage_rate),
                "storageClass": storage_class,
            },
        }

    def require_disk(self, estimated_write_bytes: int, operation: str) -> Dict[str, Any]:
        status = self.status(estimated_write_bytes=estimated_write_bytes)
        if status["diskPressure"]:
            raise NeuralStateResourcePause(
                "%s paused before crossing the configured disk reserve" % operation,
                status,
            )
        return status


class HotStateResidencyPlanner:
    """Continuously rank neural state for RAM residency under pressure.

    The planner does not impose a neural cardinality ceiling. It establishes
    eviction/spill order: active, frequently used, stable/rooted, and
    unfinished pathways stay hottest; replay, optimizer state, and cold
    scratch are already the first state moved to disk by :class:`AdaptiveBrain`.
    """

    def __init__(self) -> None:
        self.revision = 0
        self.hot_ids: frozenset[str] = frozenset()
        self.cold_ids: frozenset[str] = frozenset()
        self.became_hot_ids: frozenset[str] = frozenset()
        self.became_cold_ids: frozenset[str] = frozenset()
        self.page_in_candidate_ids: frozenset[str] = frozenset()
        self.page_out_candidate_ids: frozenset[str] = frozenset()
        self._access_heat: Dict[str, float] = {}
        self._observed_page_ins = 0
        self.last_status: Dict[str, Any] = {
            "revision": 0,
            "policy": "continuous-activity-access-retention-rooted-unfinished",
            "hotEntities": 0,
            "totalEntities": 0,
            "updatedContinuously": True,
            "dynamicTransitions": True,
            "physicalSubstratePaging": False,
        }

    def note_access(
        self, record_ids: Iterable[str], *, page_in: bool = False
    ) -> None:
        """Feed real accesses back into the next continuous ranking cycle."""

        observed = 0
        for value in record_ids:
            record_id = str(value)
            if not record_id:
                continue
            self._access_heat[record_id] = min(
                4.0, self._access_heat.get(record_id, 0.0) + 1.0
            )
            observed += 1
        if page_in:
            self._observed_page_ins += observed

    @staticmethod
    def _score(
        record: Mapping[str, Any],
        record_id: str,
        unfinished: frozenset[str],
        attention_active: frozenset[str],
        access_heat: float,
    ) -> float:
        firing = abs(
            float(
                record.get("activation", record.get("eligibility", 0.0))
                or 0.0
            )
            if record_id in attention_active
            else 0.0,
        )
        frequency = math.log1p(
            max(
                0.0,
                float(
                    record.get(
                        "uses",
                        record.get("exposures", record.get("rehearsals", 0.0)),
                    )
                    or 0.0
                ),
            )
        )
        stability = max(
            0.0,
            float(
                record.get(
                    "retention_score",
                    record.get(
                        "memory_strength",
                        record.get(
                            "stability", record.get("importance", 0.0)
                        ),
                    ),
                )
                or 0.0
            ),
        )
        rooted = bool(
            record.get("rooted")
            or record.get("root")
            or str(record.get("source", "")) in {"origin", "starter", "tool-curriculum"}
        )
        unfinished_boost = 1.0 if record_id in unfinished else 0.0
        return (
            4.0 * firing
            + 1.5 * frequency
            + 2.5 * stability
            + 2.0 * float(rooted)
            + 5.0 * unfinished_boost
            + 2.25 * max(0.0, float(access_heat))
        )

    def _update_paged(
        self,
        *,
        neurons: Mapping[str, Mapping[str, Any]],
        assemblies: Sequence[Mapping[str, Any]],
        synapses: Mapping[str, Mapping[str, Any]],
        unfinished_ids: Iterable[str],
        attention_active_ids: Iterable[str],
        attention_legacy_raw_active: bool,
        resident_budget: Optional[int],
        paged_assembly_ids: Iterable[str],
    ) -> Dict[str, Any]:
        """Rank only addressable live candidates, never the paged population."""

        previous_hot = self.hot_ids
        for record_id in tuple(self._access_heat):
            cooled = self._access_heat[record_id] * 0.72
            if cooled < 0.025:
                self._access_heat.pop(record_id, None)
            else:
                self._access_heat[record_id] = cooled
        unfinished = frozenset(str(value) for value in unfinished_ids if value)
        attention_active = frozenset(
            str(value) for value in attention_active_ids if value
        )
        paged = frozenset(str(value) for value in paged_assembly_ids if value)
        resident_items = getattr(synapses, "resident_items", None)
        resident_synapses = dict(
            resident_items() if callable(resident_items) else synapses.items()
        )
        accessed_ids = set(self._access_heat)
        tracked_ids = set(unfinished) | set(attention_active) | set(previous_hot) | accessed_ids
        # Every native assembly has a neuron row with the same ID. A cold
        # synapse is a distinct address; neither population needs an ID scan.
        neuron_count = int(neurons.status()["rowCount"])  # type: ignore[attr-defined]
        if len(assemblies) > neuron_count:
            raise ValueError("paged assembly count exceeds neuron count")
        synapse_count = len(synapses)
        total = neuron_count + synapse_count
        candidates = (
            tracked_ids | set(paged) | set(resident_synapses)
        )
        record_map: Dict[str, Dict[str, Any]] = {}
        for record_id in candidates:
            record: Dict[str, Any] = {}
            neuron = neurons.get(record_id)
            if neuron is not None:
                record.update(neuron)
            assembly = assemblies.get_by_id(record_id)  # type: ignore[attr-defined]
            if assembly is not None:
                record.update(assembly)
            resident = resident_synapses.get(record_id)
            if resident is not None:
                if neuron is not None:
                    raise ValueError("paged neuron and synapse IDs overlap")
                record.update(resident)
            if record or (record_id in tracked_ids and record_id in synapses):
                record_map[record_id] = record
        if attention_legacy_raw_active:
            attention_active = frozenset(
                set(attention_active)
                | {
                    record_id for record_id, record in record_map.items()
                    if float(record.get("activation", 0.0) or 0.0) > 0.0
                    or abs(float(record.get("eligibility", 0.0) or 0.0)) > 0.0
                }
            )
        protected_ids = {
            record_id for record_id, record in record_map.items()
            if record_id in unfinished
            or bool(record.get("rooted") or record.get("root"))
            or str(record.get("source", "")) in {"origin", "starter", "tool-curriculum"}
            or record_id in attention_active
        }
        budget = len(record_map) if resident_budget is None else max(
            0, min(total, int(resident_budget))
        )
        budget = max(budget, len(protected_ids))
        if budget >= len(record_map):
            hot_ids = set(record_map)
        else:
            hot_ids = set(protected_ids)
            hot_ids.update(
                record_id for _, record_id in heapq.nlargest(
                    max(0, budget - len(hot_ids)),
                    (
                        (
                            self._score(record, record_id, unfinished, attention_active,
                                        self._access_heat.get(record_id, 0.0)),
                            record_id,
                        )
                        for record_id, record in record_map.items()
                        if record_id not in hot_ids
                    ),
                )
            )
        self.hot_ids = frozenset(hot_ids)
        # The full cold ID set is intentionally not materialized. Counts are
        # exact for the selected hot set; transitions/candidates are exact for
        # tracked active, accessed, page-addressed, and resident records.
        self.cold_ids = frozenset(record_map.keys() - hot_ids)
        self.became_hot_ids = self.hot_ids.difference(previous_hot)
        self.became_cold_ids = previous_hot.difference(self.hot_ids)
        self.page_in_candidate_ids = self.hot_ids.intersection(paged)
        self.page_out_candidate_ids = (
            frozenset() if attention_legacy_raw_active
            else self.cold_ids.difference(paged)
        )
        self.revision += 1
        self.last_status = {
            "revision": self.revision,
            "policy": "continuous-activity-access-retention-rooted-unfinished",
            "updatedContinuously": True,
            "dynamicTransitions": True,
            "totalEntities": total,
            "hotEntities": len(self.hot_ids),
            "coldEntities": total - len(self.hot_ids),
            "becameHotEntities": len(self.became_hot_ids),
            "becameColdEntities": len(self.became_cold_ids),
            "protectedHotEntities": len(protected_ids),
            "activeCortexPriorityStable": not attention_legacy_raw_active,
            "unfinishedEntities": len(unfinished),
            "hotUnfinishedEntities": len(self.hot_ids.intersection(unfinished)),
            "recentlyAccessedEntities": len(self._access_heat),
            "pageInCandidates": len(self.page_in_candidate_ids),
            "pageOutCandidates": len(self.page_out_candidate_ids),
            "observedPageIns": self._observed_page_ins,
            "physicalSubstratePaging": True,
            "classificationOnly": False,
            "persistedColdSynapses": (
                max(0, int(getattr(synapses, "persisted_cold_count", 0)))
                if callable(resident_items) else 0
            ),
            "coldIdSetComplete": False,
            "candidateScope": "active-accessed-resident-page-addressed",
            "protectedEntityScope": "tracked-candidates",
            "legacyRawActivationUnscanned": bool(attention_legacy_raw_active),
            "priority": ["currently firing", "recently read or paged in", "frequently used",
                         "continuous retention score or rooted", "unfinished activity"],
            "spillOrder": ["cold scratch trail", "replay batches", "optimizer moments",
                           "inactive working patterns"],
            "noCardinalityLimit": True,
        }
        return dict(self.last_status)

    def update(
        self,
        *,
        neurons: Mapping[str, Mapping[str, Any]],
        assemblies: Sequence[Mapping[str, Any]],
        synapses: Mapping[str, Mapping[str, Any]],
        unfinished_ids: Iterable[str] = (),
        attention_active_ids: Iterable[str] = (),
        attention_legacy_raw_active: bool = False,
        resident_budget: Optional[int] = None,
        paged_assembly_ids: Iterable[str] = (),
    ) -> Dict[str, Any]:
        if (
            callable(getattr(neurons, "status", None))
            and callable(getattr(neurons, "iter_pages", None))
            and callable(getattr(assemblies, "get_by_id", None))
        ):
            return self._update_paged(
                neurons=neurons,
                assemblies=assemblies,
                synapses=synapses,
                unfinished_ids=unfinished_ids,
                attention_active_ids=attention_active_ids,
                attention_legacy_raw_active=attention_legacy_raw_active,
                resident_budget=resident_budget,
                paged_assembly_ids=paged_assembly_ids,
            )
        previous_hot = self.hot_ids
        resident_items = getattr(synapses, "resident_items", None)
        physically_paged = callable(resident_items)
        synapse_items = (
            list(resident_items())
            if physically_paged
            else list(synapses.items())
        )
        persisted_cold_synapses = (
            max(0, int(getattr(synapses, "persisted_cold_count", 0)))
            if physically_paged
            else 0
        )
        for record_id in tuple(self._access_heat):
            cooled = self._access_heat[record_id] * 0.72
            if cooled < 0.025:
                self._access_heat.pop(record_id, None)
            else:
                self._access_heat[record_id] = cooled
        unfinished = frozenset(str(value) for value in unfinished_ids if value)
        attention_active = frozenset(
            str(value) for value in attention_active_ids if value
        )
        if attention_legacy_raw_active:
            attention_active = frozenset(
                list(attention_active)
                + [
                    str(key)
                    for key, value in neurons.items()
                    if float(value.get("activation", 0.0) or 0.0) > 0.0
                ]
                + [
                    str(key)
                    for key, value in synapse_items
                    if abs(float(value.get("eligibility", 0.0) or 0.0)) > 0.0
                ]
            )
        # An assembly and its neural node intentionally share an ID. Merge
        # their complementary live fields so residency totals and budgets are
        # about addressable entities rather than duplicate metadata records.
        record_map: Dict[str, Dict[str, Any]] = {}
        for key, value in neurons.items():
            record_map.setdefault(str(key), {}).update(dict(value))
        for value in assemblies:
            if value.get("id"):
                record_map.setdefault(str(value.get("id")), {}).update(
                    dict(value)
                )
        for key, value in synapse_items:
            record_map.setdefault(str(key), {}).update(dict(value))
        records: list[Tuple[str, Mapping[str, Any]]] = list(record_map.items())
        total = len(records) + persisted_cold_synapses
        budget = total if resident_budget is None else max(0, min(total, int(resident_budget)))
        protected_ids = {
            record_id
            for record_id, record in records
            if record_id in unfinished
            or bool(record.get("rooted") or record.get("root"))
            or str(record.get("source", ""))
            in {"origin", "starter", "tool-curriculum"}
            or (
                record_id in attention_active
                and abs(float(
                record.get("activation", record.get("eligibility", 0.0))
                or 0.0
            ))
                > 0.0
            )
        }
        # Active/rooted/unfinished neural state is not an eligible cold page.
        # Under severe pressure this protected floor can exceed the nominal
        # metadata budget; the runtime then waits instead of destabilizing the
        # live cortex by changing which active pathways are resident.
        budget = max(budget, len(protected_ids))
        if budget >= total:
            hot_ids = {record_id for record_id, _ in records}
        elif budget == 0:
            hot_ids = set()
        else:
            hot_ids = set(protected_ids)
            hot_ids.update(
                record_id
                for _, record_id in heapq.nlargest(
                    max(0, budget - len(hot_ids)),
                    (
                        (
                            self._score(
                                record,
                                record_id,
                                unfinished,
                                attention_active,
                                self._access_heat.get(record_id, 0.0),
                            ),
                            record_id,
                        )
                        for record_id, record in records
                        if record_id not in hot_ids
                    ),
                )
            )
        hot_unfinished = len(hot_ids.intersection(unfinished))
        self.hot_ids = frozenset(hot_ids)
        all_ids = frozenset(record_id for record_id, _ in records)
        self.cold_ids = all_ids.difference(self.hot_ids)
        self.became_hot_ids = self.hot_ids.difference(previous_hot)
        self.became_cold_ids = previous_hot.difference(self.hot_ids)
        paged = frozenset(str(value) for value in paged_assembly_ids if value)
        self.page_in_candidate_ids = self.hot_ids.intersection(paged)
        self.page_out_candidate_ids = self.cold_ids.difference(paged)
        self.revision += 1
        self.last_status = {
            "revision": self.revision,
            "policy": "continuous-activity-access-retention-rooted-unfinished",
            "updatedContinuously": True,
            "dynamicTransitions": True,
            "totalEntities": total,
            "hotEntities": len(hot_ids),
            "coldEntities": total - len(hot_ids),
            "becameHotEntities": len(self.became_hot_ids),
            "becameColdEntities": len(self.became_cold_ids),
            "protectedHotEntities": len(protected_ids),
            "activeCortexPriorityStable": True,
            "unfinishedEntities": len(unfinished),
            "hotUnfinishedEntities": hot_unfinished,
            "recentlyAccessedEntities": len(self._access_heat),
            "pageInCandidates": len(self.page_in_candidate_ids),
            "pageOutCandidates": len(self.page_out_candidate_ids),
            "observedPageIns": self._observed_page_ins,
            "physicalSubstratePaging": physically_paged,
            "classificationOnly": not physically_paged,
            "persistedColdSynapses": persisted_cold_synapses,
            "priority": [
                "currently firing",
                "recently read or paged in",
                "frequently used",
                "continuous retention score or rooted",
                "unfinished activity",
            ],
            "spillOrder": [
                "cold scratch trail",
                "replay batches",
                "optimizer moments",
                "inactive working patterns",
            ],
            "noCardinalityLimit": True,
        }
        return dict(self.last_status)


_DTYPES = {
    str(value): value
    for value in (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
}


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(_tensor_bytes(contiguous))
    return digest.hexdigest()


class DurableReplayBuffer(Sequence[torch.Tensor]):
    """A transactional, unbounded-on-disk latent replay sequence."""

    def __init__(
        self, path: Path, policy: ResourcePolicy, *, read_only: bool = False
    ):
        self.path = Path(path)
        self.policy = policy
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file() or self.path.is_symlink():
                raise ValueError("read-only latent replay is not a regular file")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.quarantined_shared_blobs: list[str] = []
        self.last_checkpoint_error: Optional[str] = None
        self.detached_shared_inode = (
            False if self.read_only else self._detach_shared_database()
        )
        if not self.read_only:
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS replay (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL NOT NULL,
                        sha256 TEXT NOT NULL,
                        dtype TEXT NOT NULL,
                        shape_json TEXT NOT NULL,
                        payload BLOB NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS replay_sha ON replay(sha256)"
                )

    def _require_writable(self) -> None:
        if self.read_only:
            raise RuntimeError("read-only latent replay cannot be mutated")

    def _detach_shared_database(self) -> bool:
        """Break unsafe hard links before SQLite can mutate another identity.

        Historical desktop forks linked the mutable replay database through the
        immutable blob store. SQLite writes in place, so one fork's rollback
        could truncate every sibling sharing that inode. A read-only SQLite
        backup includes committed WAL state and produces a private, self-
        contained database before this instance opens it for schema or writes.
        """

        if not self.path.is_file() or self.path.stat().st_nlink <= 1:
            return False
        shared_stat = self.path.stat()
        temporary = self.path.with_name(
            ".%s.%s.detach" % (self.path.name, uuid.uuid4().hex)
        )
        source_uri = "file:%s?mode=ro" % self.path.resolve().as_posix()
        try:
            with sqlite3.connect(source_uri, uri=True, timeout=30.0) as source:
                with sqlite3.connect(str(temporary), timeout=30.0) as destination:
                    source.backup(destination)
                    integrity = destination.execute("PRAGMA quick_check").fetchone()
                    if not integrity or integrity[0] != "ok":
                        raise ValueError(
                            "detached latent replay database failed integrity check"
                        )
                    destination.execute("PRAGMA journal_mode=DELETE")
            os.replace(str(temporary), str(self.path))
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.path) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
            self._quarantine_corrupt_shared_blobs(
                int(shared_stat.st_dev), int(shared_stat.st_ino)
            )
            return True
        finally:
            if temporary.exists():
                temporary.unlink()

    def _quarantine_corrupt_shared_blobs(
        self, device: int, inode: int
    ) -> None:
        """Remove hash-mismatched former replay inodes from the blob namespace."""

        if len(self.path.parents) < 4:
            return
        blob_root = self.path.parents[3] / ".blobs"
        if not blob_root.is_dir():
            return
        quarantine = blob_root / ".quarantine"
        for candidate in blob_root.iterdir():
            if candidate.is_symlink() or not candidate.is_file():
                continue
            status = candidate.stat()
            if int(status.st_dev) != device or int(status.st_ino) != inode:
                continue
            actual = _file_sha256(candidate)
            if candidate.name == actual:
                continue
            quarantine.mkdir(parents=True, exist_ok=True)
            destination = quarantine / (
                "%s.%s.corrupt" % (candidate.name, actual)
            )
            if destination.exists():
                candidate.unlink()
            else:
                os.replace(str(candidate), str(destination))
            self.quarantined_shared_blobs.append(str(destination))

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(
                "file:%s?mode=ro" % self.path.resolve().as_posix(),
                uri=True,
                timeout=30.0,
                isolation_level=None,
            )
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            return connection
        connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection


    def append_many(self, tensors: Iterable[torch.Tensor]) -> Tuple[int, ...]:
        """Atomically append selected replay vectors in their original order.

        The reserve estimate sums the same conservative per-row allowance as
        ``append`` so a batch never gets a weaker disk preflight. The caller
        chooses the batch boundary; replay itself has no cardinality cap.
        """

        self._require_writable()
        rows = []
        estimated_write_bytes = 0
        for tensor in tensors:
            value = tensor.detach().cpu().contiguous().reshape(-1)
            payload = _tensor_bytes(value)
            # SQLite/WAL and index overhead are intentionally estimated
            # conservatively so the reserve is never consumed by journal growth.
            estimated_write_bytes += max(64 * 1024, len(payload) * 3)
            rows.append(
                (
                    time.time(),
                    _tensor_sha256(value),
                    str(value.dtype),
                    json.dumps(list(value.shape), separators=(",", ":")),
                    sqlite3.Binary(payload),
                )
            )
        if not rows:
            return ()
        self.policy.require_disk(estimated_write_bytes, "latent replay spill")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                sequences = []
                for row in rows:
                    cursor = connection.execute(
                        """
                        INSERT INTO replay
                        (created_at, sha256, dtype, shape_json, payload)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        row,
                    )
                    sequences.append(int(cursor.lastrowid))
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error:
                        # Preserve the original insert/commit failure.
                        pass
                raise
            # A successful synchronous=FULL COMMIT is durable in the WAL.
            # Checkpointing copies it into the main database, but a failure
            # here cannot roll back the committed rows. Returning an error
            # would invite a retry that duplicates the whole batch.
            self.last_checkpoint_error = None
            try:
                checkpoint = connection.execute(
                    "PRAGMA wal_checkpoint(FULL)"
                ).fetchone()
                if checkpoint is not None and int(checkpoint[0]) != 0:
                    self.last_checkpoint_error = (
                        "latent replay WAL checkpoint incomplete (busy)"
                    )
            except sqlite3.Error as error:
                self.last_checkpoint_error = str(error)
        return tuple(sequences)

    def append(self, tensor: torch.Tensor) -> int:
        return self.append_many((tensor,))[0]

    def extend(self, values: Iterable[torch.Tensor]) -> None:
        for value in values:
            self.append(value)

    def __len__(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM replay").fetchone()
        return int(row[0] if row else 0)

    @staticmethod
    def _decode(row: Sequence[Any]) -> torch.Tensor:
        _sequence, expected_sha, dtype_name, shape_json, payload = row
        dtype = _DTYPES.get(str(dtype_name))
        if dtype is None:
            raise ValueError("replay tensor dtype is unsupported")
        try:
            shape = tuple(int(value) for value in json.loads(str(shape_json)))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("replay tensor shape is invalid") from error
        raw = bytearray(payload)
        value = torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)
        if _tensor_sha256(value) != str(expected_sha):
            raise ValueError("replay tensor checksum mismatch")
        return value

    def verified_rows(
        self, *, high_water_id: Optional[int] = None
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Return rows only after validating SQLite and every tensor payload."""

        with self._connect() as connection:
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise ValueError("latent replay database failed integrity check")
            if high_water_id is None:
                rows = connection.execute(
                    """
                    SELECT sequence, created_at, sha256, dtype, shape_json, payload
                    FROM replay ORDER BY sequence
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT sequence, created_at, sha256, dtype, shape_json, payload
                    FROM replay WHERE sequence <= ? ORDER BY sequence
                    """,
                    (max(0, int(high_water_id)),),
                ).fetchall()
        verified = []
        for row in rows:
            self._decode((row[0], row[2], row[3], row[4], row[5]))
            verified.append(tuple(row))
        return tuple(verified)

    @staticmethod
    def checkpoint_for_rows(rows: Sequence[Sequence[Any]]) -> Dict[str, Any]:
        digest = hashlib.sha256()
        high_water = 0
        for row in rows:
            high_water = DurableReplayBuffer._update_checkpoint_digest(digest, row)
        return DurableReplayBuffer._checkpoint_summary(len(rows), high_water, digest)

    @staticmethod
    def _update_checkpoint_digest(digest: Any, row: Sequence[Any]) -> int:
        sequence = int(row[0])
        checksum = str(row[2])
        digest.update(str(sequence).encode("ascii"))
        digest.update(b"\0")
        digest.update(checksum.encode("ascii"))
        digest.update(b"\n")
        return sequence

    @staticmethod
    def _checkpoint_summary(
        count: int, high_water: int, digest: Any
    ) -> Dict[str, Any]:
        return {
            "format": "omni-replay-sqlite",
            "formatVersion": 1,
            "path": "replay.sqlite3",
            "count": count,
            "highWaterId": high_water,
            "contentSha256": digest.hexdigest(),
            "transactional": True,
            "silentEviction": False,
        }

    def _verified_checkpoint_scan(
        self, *, high_water_id: Optional[int] = None
    ) -> Tuple[Dict[str, Any], int]:
        """Validate all durable rows in one snapshot without retaining the corpus.

        A checkpoint covers the committed prefix, but rows after its high-water
        mark must also be decoded so corruption in pending replay is not hidden.
        The SELECT cursor holds one SQLite read snapshot while bounded batches
        are processed; no decoded tensors or row batches escape the loop.
        """

        digest = hashlib.sha256()
        committed_count = 0
        durable_count = 0
        high_water = 0
        with self._connect() as connection:
            connection.execute("BEGIN")
            integrity = connection.execute("PRAGMA quick_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise ValueError("latent replay database failed integrity check")
            cursor = connection.execute(
                """
                SELECT sequence, created_at, sha256, dtype, shape_json, payload
                FROM replay ORDER BY sequence
                """
            )
            while True:
                rows = cursor.fetchmany(32)
                if not rows:
                    break
                for row in rows:
                    self._decode((row[0], row[2], row[3], row[4], row[5]))
                    durable_count += 1
                    if high_water_id is None or int(row[0]) <= high_water_id:
                        high_water = self._update_checkpoint_digest(digest, row)
                        committed_count += 1
        return self._checkpoint_summary(committed_count, high_water, digest), durable_count

    def restore_verified_rows(
        self, rows: Sequence[Sequence[Any]]
    ) -> int:
        """Append authenticated missing rows without replacing existing data."""

        self._require_writable()
        if not rows:
            return 0
        current = self.verified_rows()
        current_by_id = {int(row[0]): row for row in current}
        additions = []
        for raw in rows:
            row = tuple(raw)
            self._decode((row[0], row[2], row[3], row[4], row[5]))
            sequence = int(row[0])
            existing = current_by_id.get(sequence)
            if existing is not None:
                if tuple(existing[1:]) != tuple(row[1:]):
                    raise ValueError("authenticated replay row conflicts with live data")
                continue
            additions.append(row)
        if not additions:
            return 0
        maximum = max(current_by_id, default=0)
        if [int(row[0]) for row in additions] != sorted(
            int(row[0]) for row in additions
        ) or int(additions[0][0]) <= maximum:
            raise ValueError("authenticated replay recovery is not an append-only suffix")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executemany(
                    """
                    INSERT INTO replay
                    (sequence, created_at, sha256, dtype, shape_json, payload)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    additions,
                )
                connection.execute("COMMIT")
                connection.execute("PRAGMA wal_checkpoint(FULL)")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return len(additions)

    def replace_with_verified_rows(
        self, rows: Sequence[Sequence[Any]]
    ) -> None:
        """Atomically replace replay with one fully verified snapshot."""

        self._require_writable()
        normalized = []
        previous_sequence = 0
        for raw in rows:
            row = tuple(raw)
            if len(row) != 6:
                raise ValueError("authenticated replay row is invalid")
            sequence = int(row[0])
            if sequence <= previous_sequence:
                raise ValueError("authenticated replay rows are not ordered")
            self._decode((row[0], row[2], row[3], row[4], row[5]))
            normalized.append(row)
            previous_sequence = sequence
        temporary = self.path.with_name(
            ".%s.%s.recovery" % (self.path.name, uuid.uuid4().hex)
        )
        try:
            replacement = DurableReplayBuffer(temporary, self.policy)
            with replacement._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.executemany(
                        """
                        INSERT INTO replay
                        (sequence, created_at, sha256, dtype, shape_json, payload)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        normalized,
                    )
                    connection.execute("COMMIT")
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
            replacement.verified_rows()
            os.replace(str(temporary), str(self.path))
            for suffix in ("-wal", "-shm"):
                sidecar = Path(str(self.path) + suffix)
                if sidecar.exists():
                    sidecar.unlink()
        finally:
            for candidate in (
                temporary,
                Path(str(temporary) + "-wal"),
                Path(str(temporary) + "-shm"),
            ):
                if candidate.exists():
                    candidate.unlink()

    def _row_at(self, index: int) -> Sequence[Any]:
        length = len(self)
        normalized = index + length if index < 0 else index
        if normalized < 0 or normalized >= length:
            raise IndexError("replay index out of range")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sequence, sha256, dtype, shape_json, payload
                FROM replay ORDER BY sequence LIMIT 1 OFFSET ?
                """,
                (normalized,),
            ).fetchone()
        if row is None:
            raise IndexError("replay index out of range")
        return row

    @overload
    def __getitem__(self, index: int) -> torch.Tensor:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[torch.Tensor]:
        ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[position] for position in range(start, stop, step)]
        return self._decode(self._row_at(int(index)))

    def __iter__(self) -> Iterator[torch.Tensor]:
        last_seen = 0
        batch_size = 256
        while True:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT sequence, sha256, dtype, shape_json, payload
                    FROM replay WHERE sequence > ? ORDER BY sequence LIMIT ?
                    """,
                    (last_seen, batch_size),
                ).fetchall()
            if not rows:
                return
            for row in rows:
                value = self._decode(row)
                last_seen = int(row[0])
                yield value

    def truncate(self, length: int) -> int:
        """Explicitly roll back entries beyond a candidate transaction mark."""

        self._require_writable()
        length = max(0, int(length))
        with self._connect() as connection:
            keep = connection.execute(
                "SELECT sequence FROM replay ORDER BY sequence LIMIT 1 OFFSET ?",
                (length,),
            ).fetchone()
            if keep is None:
                return 0
            before = int(
                connection.execute("SELECT COUNT(*) FROM replay").fetchone()[0]
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "DELETE FROM replay WHERE sequence >= ?", (int(keep[0]),)
                )
                connection.execute("COMMIT")
                connection.execute("PRAGMA wal_checkpoint(FULL)")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return before - length

    def checkpoint(self) -> Dict[str, Any]:
        checkpoint, _durable_count = self._verified_checkpoint_scan()
        return checkpoint

    def verify_checkpoint(self, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
        expected_count = int(checkpoint.get("count", -1))
        high_water = int(checkpoint.get("highWaterId", -1))
        actual, durable_count = self._verified_checkpoint_scan(
            high_water_id=high_water
        )
        if (
            checkpoint.get("format") != "omni-replay-sqlite"
            or int(checkpoint.get("formatVersion", 0)) != 1
            or actual["count"] != expected_count
            or actual["highWaterId"] != high_water
            or actual["contentSha256"]
            != str(checkpoint.get("contentSha256", ""))
        ):
            raise ValueError("latent replay checkpoint checksum mismatch")
        return {
            "committedExamples": int(actual["count"]),
            "durableExamples": durable_count,
            "pendingExamples": max(0, durable_count - int(actual["count"])),
            "contentSha256": str(actual["contentSha256"]),
        }

    def status(self) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(MAX(sequence), 0) FROM replay"
            ).fetchone()
        return {
            "format": "omni-replay-sqlite",
            "formatVersion": 1,
            "count": int(row[0] if row else 0),
            "highWaterId": int(row[1] if row else 0),
            "bytes": self.path.stat().st_size if self.path.is_file() else 0,
            "storage": "disk",
            "transactional": True,
            "silentEviction": False,
        }


class PagedWorkingMemory:
    """Restart-safe cold working-pattern spill with a bounded RAM hot cache.

    This is an addressable paged sequence rather than dense attention. Values
    retain explicit dtype, shape, and checksum; no pickle payload is accepted.
    """

    def __init__(self, path: Path, policy: ResourcePolicy):
        self.path = Path(path)
        self.policy = policy
        self._metrics_lock = threading.Lock()
        self._reads = 0
        self._read_misses = 0
        self._page_ins = 0
        self._page_in_misses = 0
        self._page_outs = 0
        self._bytes_read = 0
        self._last_read_at: Optional[float] = None
        self._last_page_in_at: Optional[float] = None
        self._last_page_out_at: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS working_pages (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    page_id TEXT NOT NULL UNIQUE,
                    assembly_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    dtype TEXT NOT NULL,
                    shape_json TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS working_pages_assembly
                ON working_pages(assembly_id)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS working_pages_updated
                ON working_pages(updated_at)
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _decode_page_row(
        row: Sequence[Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        (
            sequence,
            page_id,
            assembly_id,
            metadata_json,
            expected_sha,
            dtype_name,
            shape_json,
            payload,
            updated_at,
        ) = row
        dtype = _DTYPES.get(str(dtype_name))
        if dtype is None:
            raise ValueError("working-memory page tensor dtype is unsupported")
        try:
            shape = tuple(int(value) for value in json.loads(str(shape_json)))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("working-memory page tensor shape is invalid") from error
        if not shape or any(value < 0 for value in shape):
            raise ValueError("working-memory page tensor shape is invalid")
        try:
            metadata_value = json.loads(str(metadata_json))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("working-memory page metadata is invalid") from error
        if not isinstance(metadata_value, Mapping):
            raise ValueError("working-memory page metadata is invalid")
        raw = bytearray(payload)
        try:
            value = torch.frombuffer(raw, dtype=dtype).clone().reshape(shape)
        except (RuntimeError, ValueError) as error:
            raise ValueError("working-memory page tensor payload is invalid") from error
        if _tensor_sha256(value) != str(expected_sha):
            raise ValueError("working-memory page tensor checksum mismatch")
        metadata = dict(metadata_value)
        recorded_assembly_id = str(metadata.get("assemblyId", ""))
        if recorded_assembly_id and recorded_assembly_id != str(assembly_id):
            raise ValueError("working-memory page assembly metadata mismatch")
        metadata["assemblyId"] = str(assembly_id)
        metadata["pageId"] = str(page_id)
        metadata["pageSequence"] = int(sequence)
        metadata["pagedUpdatedAt"] = float(updated_at)
        return value, metadata

    @staticmethod
    def _cold_priority(
        *,
        assembly_id: str,
        metadata: Mapping[str, Any],
        updated_at: float,
        active_ids: frozenset[str],
        unfinished_ids: frozenset[str],
        now: float,
    ) -> float:
        try:
            salience = max(
                0.0, min(1.0, float(metadata.get("salience", 0.0) or 0.0))
            )
            retention = max(
                0.0,
                min(
                    1.0,
                    float(
                        metadata.get(
                            "retentionScore", metadata.get("strength", salience)
                        )
                        or 0.0
                    ),
                ),
            )
            rehearsals = max(0, int(metadata.get("rehearsals", 0)))
        except (TypeError, ValueError):
            salience = 0.0
            retention = 0.0
            rehearsals = 0
        recency = math.exp(-max(0.0, now - float(updated_at)) / 3600.0)
        return (
            5.0 * float(assembly_id in active_ids)
            + 5.5 * float(assembly_id in unfinished_ids)
            + 2.0 * salience
            + 2.5 * retention
            + 1.2 * math.log1p(rehearsals)
            + recency
        )

    def _note_io(
        self,
        *,
        read: bool = False,
        read_miss: bool = False,
        page_in: bool = False,
        page_in_miss: bool = False,
        page_out: bool = False,
        bytes_read: int = 0,
    ) -> None:
        now = time.time()
        with self._metrics_lock:
            if read:
                self._reads += 1
                self._last_read_at = now
            if read_miss:
                self._read_misses += 1
            if page_in:
                self._page_ins += 1
                self._last_page_in_at = now
            if page_in_miss:
                self._page_in_misses += 1
            if page_out:
                self._page_outs += 1
                self._last_page_out_at = now
            self._bytes_read += max(0, int(bytes_read))

    def append(
        self,
        tensor: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> str:
        value = tensor.detach().cpu().contiguous().reshape(-1)
        payload = _tensor_bytes(value)
        self.policy.require_disk(
            max(64 * 1024, len(payload) * 3), "working-memory page spill"
        )
        page_id = uuid.uuid4().hex
        metadata_json = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":")
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO working_pages
                    (page_id, assembly_id, metadata_json, sha256, dtype,
                     shape_json, payload, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        page_id,
                        str(metadata.get("assemblyId", "")),
                        metadata_json,
                        _tensor_sha256(value),
                        str(value.dtype),
                        json.dumps(list(value.shape), separators=(",", ":")),
                        sqlite3.Binary(payload),
                        time.time(),
                    ),
                )
                connection.execute("COMMIT")
                connection.execute("PRAGMA wal_checkpoint(FULL)")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        self._note_io(page_out=True)
        return page_id

    def page_ids(self) -> Tuple[str, ...]:
        """Return stable page addresses without reading tensor payloads."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT page_id FROM working_pages ORDER BY sequence"
            ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def assembly_ids(self) -> Tuple[str, ...]:
        """Return cold assembly addresses for residency planning."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT assembly_id FROM working_pages ORDER BY sequence"
            ).fetchall()
        return tuple(str(row[0]) for row in rows if str(row[0]))

    def read(
        self, page_id: str, *, touch: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Read and verify one cold page without removing it from storage."""

        page_key = str(page_id)
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sequence, page_id, assembly_id, metadata_json, sha256,
                       dtype, shape_json, payload, updated_at
                FROM working_pages WHERE page_id = ?
                """,
                (page_key,),
            ).fetchone()
            if row is None:
                self._note_io(read_miss=True)
                raise KeyError("working-memory page not found: %s" % page_key)
            value, metadata = self._decode_page_row(row)
            if touch:
                connection.execute(
                    "UPDATE working_pages SET updated_at = ? WHERE page_id = ?",
                    (time.time(), page_key),
                )
        self._note_io(read=True, bytes_read=value.numel() * value.element_size())
        return value, metadata

    def page_in(self, page_id: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Atomically verify and move one addressed page from disk to RAM.

        The database row is deleted only after dtype, shape, metadata, and
        checksum validation succeeds. A corrupt or interrupted read therefore
        leaves the cold page available for diagnosis/recovery.
        """

        page_key = str(page_id)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    """
                    SELECT sequence, page_id, assembly_id, metadata_json,
                           sha256, dtype, shape_json, payload, updated_at
                    FROM working_pages WHERE page_id = ?
                    """,
                    (page_key,),
                ).fetchone()
                if row is None:
                    connection.execute("ROLLBACK")
                    self._note_io(page_in_miss=True)
                    raise KeyError(
                        "working-memory page not found: %s" % page_key
                    )
                value, metadata = self._decode_page_row(row)
                connection.execute(
                    "DELETE FROM working_pages WHERE page_id = ?", (page_key,)
                )
                connection.execute("COMMIT")
                connection.execute("PRAGMA wal_checkpoint(FULL)")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        self._note_io(
            page_in=True, bytes_read=value.numel() * value.element_size()
        )
        return value, metadata

    def _rank_hot_page_ids(
        self,
        *,
        hot_assembly_ids: Iterable[str],
        unfinished_ids: Iterable[str] = (),
        limit: int = 1,
    ) -> List[str]:
        """Select cold addresses without reading or removing their tensors."""

        active = frozenset(str(value) for value in hot_assembly_ids if value)
        unfinished = frozenset(str(value) for value in unfinished_ids if value)
        candidates = active.union(unfinished)
        maximum = max(0, int(limit))
        if not candidates or maximum == 0:
            return []
        now = time.time()
        ranked: List[Tuple[float, int, str]] = []
        # SQLite builds before 3.32 may allow only 999 bound variables. The
        # chunk size is a query parameter budget, not a candidate/recall cap.
        identifiers = sorted(candidates)
        bind_chunk = 900
        with self._connect() as connection:
            for offset in range(0, len(identifiers), bind_chunk):
                chunk = identifiers[offset : offset + bind_chunk]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    "SELECT sequence, page_id, assembly_id, metadata_json, "
                    "updated_at FROM working_pages WHERE assembly_id IN ("
                    + placeholders + ")",
                    chunk,
                )
                for sequence, page_id, assembly_id, metadata_json, updated_at in rows:
                    assembly_key = str(assembly_id)
                    try:
                        metadata_value = json.loads(str(metadata_json))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        metadata_value = {}
                    metadata = (
                        metadata_value
                        if isinstance(metadata_value, Mapping)
                        else {}
                    )
                    priority = self._cold_priority(
                        assembly_id=assembly_key,
                        metadata=metadata,
                        updated_at=float(updated_at),
                        active_ids=active,
                        unfinished_ids=unfinished,
                        now=now,
                    )
                    ranked.append((priority, int(sequence), str(page_id)))
        return [page_id for _priority, _sequence, page_id in sorted(
            ranked, reverse=True
        )[:maximum]]

    def peek_hot(
        self,
        *,
        hot_assembly_ids: Iterable[str],
        unfinished_ids: Iterable[str] = (),
        limit: int = 1,
    ) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Verify likely hot pages without consuming their cold addresses."""

        selected = self._rank_hot_page_ids(
            hot_assembly_ids=hot_assembly_ids,
            unfinished_ids=unfinished_ids,
            limit=limit,
        )
        previewed: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
        for page_id in selected:
            try:
                previewed.append(self.read(page_id, touch=False))
            except KeyError:
                continue
        return previewed

    def page_in_hot(
        self,
        *,
        hot_assembly_ids: Iterable[str],
        unfinished_ids: Iterable[str] = (),
        limit: int = 1,
    ) -> List[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Move the hottest currently requested cold patterns back into RAM."""

        selected = self._rank_hot_page_ids(
            hot_assembly_ids=hot_assembly_ids,
            unfinished_ids=unfinished_ids,
            limit=limit,
        )
        restored: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
        for page_id in selected:
            try:
                restored.append(self.page_in(page_id))
            except KeyError:
                # Another worker may have won the transactional page-in.
                continue
        return restored

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM working_pages"
            ).fetchone()
        return int(row[0] if row else 0)

    @staticmethod
    def empty_checkpoint() -> Dict[str, Any]:
        return {
            "format": "omni-working-memory-pages",
            "formatVersion": 1,
            "count": 0,
            "highWaterId": 0,
            "contentSha256": hashlib.sha256(b"").hexdigest(),
            "temporary": True,
            "runtimeReadable": True,
            "learningReadable": True,
            "pageInSupported": True,
        }

    def clear(self) -> int:
        """Atomically discard only the temporary cold attention trail."""

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM working_pages"
                    ).fetchone()[0]
                )
                connection.execute("DELETE FROM working_pages")
                connection.execute("COMMIT")
                connection.execute("PRAGMA wal_checkpoint(FULL)")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return count

    def trim_to(self, maximum: int) -> int:
        """Remove the dynamically coldest pages beyond the physical window."""

        maximum = max(0, int(maximum))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, page_id, assembly_id, metadata_json,
                       updated_at FROM working_pages
                """
            ).fetchall()
            count = len(rows)
            remove = max(0, count - maximum)
            if remove:
                now = time.time()
                ranked: List[Tuple[float, int]] = []
                for sequence, _page_id, assembly_id, metadata_json, updated_at in rows:
                    try:
                        metadata_value = json.loads(str(metadata_json))
                    except (TypeError, ValueError, json.JSONDecodeError):
                        metadata_value = {}
                    metadata = (
                        metadata_value
                        if isinstance(metadata_value, Mapping)
                        else {}
                    )
                    ranked.append(
                        (
                            self._cold_priority(
                                assembly_id=str(assembly_id),
                                metadata=metadata,
                                updated_at=float(updated_at),
                                active_ids=frozenset(),
                                unfinished_ids=frozenset(),
                                now=now,
                            ),
                            int(sequence),
                        )
                    )
                coldest = [
                    sequence
                    for _priority, sequence in sorted(ranked)[:remove]
                ]
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.executemany(
                        "DELETE FROM working_pages WHERE sequence = ?",
                        ((sequence,) for sequence in coldest),
                    )
                    connection.execute("COMMIT")
                    connection.execute("PRAGMA wal_checkpoint(FULL)")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        return remove

    def checkpoint(self) -> Dict[str, Any]:
        digest = hashlib.sha256()
        count = 0
        high_water = 0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, page_id, sha256
                FROM working_pages ORDER BY sequence
                """
            )
            for sequence, page_id, checksum in rows:
                count += 1
                high_water = int(sequence)
                digest.update(str(sequence).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(page_id).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(checksum).encode("ascii"))
                digest.update(b"\n")
        return {
            "format": "omni-working-memory-pages",
            "formatVersion": 1,
            "count": count,
            "highWaterId": high_water,
            "contentSha256": digest.hexdigest(),
            "temporary": True,
            "runtimeReadable": True,
            "learningReadable": True,
            "pageInSupported": True,
        }

    def recover_checkpoint(self, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
        """Remove pages written ahead of committed neural metadata.

        Cold pages are temporary workspace state with an explicit verified
        page-in API. If an interrupted trim removed a previously committed
        page, the safest recovery is still to clear this transient generation
        rather than expose a mixed one.
        """

        if (
            checkpoint.get("format") != "omni-working-memory-pages"
            or int(checkpoint.get("formatVersion", 0)) != 1
        ):
            raise ValueError("working-memory page checkpoint is invalid")
        high_water = int(checkpoint.get("highWaterId", -1))
        expected_count = int(checkpoint.get("count", -1))
        if high_water < 0 or expected_count < 0:
            raise ValueError("working-memory page checkpoint is invalid")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                pending = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM working_pages WHERE sequence > ?",
                        (high_water,),
                    ).fetchone()[0]
                )
                connection.execute(
                    "DELETE FROM working_pages WHERE sequence > ?", (high_water,)
                )
                connection.execute("COMMIT")
                connection.execute("PRAGMA wal_checkpoint(FULL)")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        current = self.checkpoint()
        reset = (
            int(current["count"]) != expected_count
            or str(current["contentSha256"])
            != str(checkpoint.get("contentSha256", ""))
        )
        if reset:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    connection.execute("DELETE FROM working_pages")
                    connection.execute("COMMIT")
                    connection.execute("PRAGMA wal_checkpoint(FULL)")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        return {
            "committedPages": 0 if reset else expected_count,
            "rolledBackPages": pending,
            "scratchReset": reset,
            "runtimeReadable": True,
            "learningReadable": True,
            "pageInSupported": True,
        }

    def status(self) -> Dict[str, Any]:
        with self._metrics_lock:
            metrics = {
                "reads": self._reads,
                "readMisses": self._read_misses,
                "pageIns": self._page_ins,
                "pageInMisses": self._page_in_misses,
                "pageOuts": self._page_outs,
                "bytesRead": self._bytes_read,
                "lastReadAt": self._last_read_at,
                "lastPageInAt": self._last_page_in_at,
                "lastPageOutAt": self._last_page_out_at,
            }
        return {
            "format": "sqlite-typed-tensor-pages",
            "count": self.count(),
            "path": str(self.path),
            "pickle": False,
            "restartSafe": True,
            "denseAttention": False,
            "runtimeReadable": True,
            "learningReadable": True,
            "pageInSupported": True,
            "dynamicHotCold": True,
            **metrics,
        }


def _encode_safe_tree(value: Any, tensors: Dict[str, torch.Tensor]) -> Any:
    if isinstance(value, torch.Tensor):
        name = "tensor.%08d" % len(tensors)
        tensors[name] = value.detach().cpu().contiguous()
        return {"type": "tensor", "name": name}
    if value is None or isinstance(value, (bool, int, str)):
        return {"type": "scalar", "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer state contains a non-finite scalar")
        return {"type": "scalar", "value": value}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode_safe_tree(v, tensors) for v in value]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode_safe_tree(v, tensors) for v in value]}
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "items": [
                [_encode_safe_tree(key, tensors), _encode_safe_tree(item, tensors)]
                for key, item in value.items()
            ],
        }
    raise TypeError("optimizer state contains unsupported type %s" % type(value).__name__)


def _decode_safe_tree(value: Mapping[str, Any], tensors: Mapping[str, torch.Tensor]) -> Any:
    kind = value.get("type")
    if kind == "tensor":
        name = str(value.get("name", ""))
        if name not in tensors:
            raise ValueError("optimizer state references a missing tensor")
        return tensors[name].detach().cpu().clone()
    if kind == "scalar":
        return value.get("value")
    if kind in {"list", "tuple"}:
        items = [_decode_safe_tree(item, tensors) for item in value.get("items", [])]
        return tuple(items) if kind == "tuple" else items
    if kind == "dict":
        return {
            _decode_safe_tree(pair[0], tensors): _decode_safe_tree(pair[1], tensors)
            for pair in value.get("items", [])
        }
    raise ValueError("optimizer state structure is invalid")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class MutableStateStore:
    """Content-addressed checkpoint and pressure-scratch generations."""

    def __init__(self, path: Path, brain_id: str, policy: ResourcePolicy):
        self.path = Path(path)
        self.brain_id = str(brain_id)
        self.policy = policy
        self.path.mkdir(parents=True, exist_ok=True)
        self.last_recovery: Dict[str, Any] = {
            "recovered": False,
            "reason": "new runtime",
        }
        self.last_gc: Dict[str, Any] = {
            "completed": False,
            "reason": "no committed checkpoint yet",
            "generationsRetained": 0,
            "generationsRemoved": 0,
            "blobsRemoved": 0,
            "bytesReclaimed": 0,
        }

    @staticmethod
    def _tensor_bytes_estimate(tensors: Mapping[str, torch.Tensor]) -> int:
        return sum(
            int(value.numel()) * int(value.element_size()) + 256
            for value in tensors.values()
        ) + 4096

    def _write_tensor_blob(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        role: str,
    ) -> Dict[str, Any]:
        blobs = self.path / "blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        temporary = blobs / (".%s.%s.tmp" % (role, uuid.uuid4().hex))
        try:
            atomic_save_tensors(
                temporary,
                tensors,
                metadata={
                    "format": "omni-mutable-state-blob",
                    "role": role,
                    "brain_id": self.brain_id,
                },
            )
            checksum = _file_sha256(temporary)
            destination = blobs / (checksum + ".safetensors")
            if destination.exists():
                if _file_sha256(destination) != checksum:
                    raise ValueError("content-addressed state blob conflicts")
                temporary.unlink()
            else:
                os.replace(str(temporary), str(destination))
            return {
                "path": "blobs/%s.safetensors" % checksum,
                "sha256": checksum,
                "bytes": destination.stat().st_size,
                "tensorCount": len(tensors),
            }
        finally:
            if temporary.exists():
                temporary.unlink()

    def stage_generation(
        self,
        *,
        core: Mapping[str, torch.Tensor],
        plasticity: Mapping[str, torch.Tensor],
        optimizer_state: Mapping[str, Any],
        replay: DurableReplayBuffer,
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        optimizer_tensors: Dict[str, torch.Tensor] = {}
        optimizer_structure = _encode_safe_tree(optimizer_state, optimizer_tensors)
        # One immutable recovery blob plus one compatibility/runtime copy is
        # budgeted. Filesystems may transparently clone blocks, but correctness
        # never depends on hard links that could corrupt both copies at once.
        estimate = 2 * (
            self._tensor_bytes_estimate(core)
            + self._tensor_bytes_estimate(plasticity)
            + self._tensor_bytes_estimate(optimizer_tensors)
        )
        self.policy.require_disk(estimate, "neural-state checkpoint")
        roles = {
            "core": self._write_tensor_blob(core, role="core"),
            "plasticity": self._write_tensor_blob(
                plasticity, role="plasticity"
            ),
            "optimizer": self._write_tensor_blob(
                optimizer_tensors, role="optimizer"
            ),
        }
        body: Dict[str, Any] = {
            "format": "omni-mutable-state",
            "formatVersion": 1,
            "brainId": self.brain_id,
            "createdAt": time.time(),
            "roles": roles,
            "optimizerStructure": optimizer_structure,
            "replay": replay.checkpoint(),
            "metadata": dict(metadata),
            "activationState": [
                name
                for name in plasticity
                if name.startswith("state.") or name.startswith("router.")
            ],
            "safeTensorOnly": True,
            "transactional": True,
        }
        content_sha = _canonical_sha(body)
        generation = {**body, "contentSha256": content_sha}
        relative = "generations/%s/manifest.json" % content_sha
        manifest_path = self.path / relative
        atomic_write_json(manifest_path, generation)
        manifest_sha = _file_sha256(manifest_path)
        return {
            "format": "omni-mutable-state",
            "formatVersion": 1,
            "activeGeneration": content_sha,
            "contentSha256": content_sha,
            "generationManifest": relative,
            "generationManifestSha256": manifest_sha,
            "replayCount": int(generation["replay"]["count"]),
        }

    def _generation(self, pointer: Mapping[str, Any]) -> Dict[str, Any]:
        generation_id = str(pointer.get("activeGeneration", ""))
        relative = str(pointer.get("generationManifest", ""))
        if (
            len(generation_id) != 64
            or any(value not in "0123456789abcdef" for value in generation_id)
            or relative != "generations/%s/manifest.json" % generation_id
        ):
            raise ValueError("mutable-state generation pointer is invalid")
        path = (self.path / relative).resolve()
        try:
            path.relative_to(self.path.resolve())
        except ValueError as error:
            raise ValueError("mutable-state generation escapes its store") from error
        if _file_sha256(path) != str(pointer.get("generationManifestSha256", "")):
            raise ValueError("mutable-state generation manifest checksum mismatch")
        generation = read_json(path)
        body = {key: value for key, value in generation.items() if key != "contentSha256"}
        content_sha = _canonical_sha(body)
        if (
            generation.get("format") != "omni-mutable-state"
            or int(generation.get("formatVersion", 0)) != 1
            # Fork/duplicate changes the live identity while intentionally
            # retaining the immutable parent's safe state generation.
            or not str(generation.get("brainId", ""))
            or content_sha != generation_id
            or content_sha != str(generation.get("contentSha256", ""))
            or content_sha != str(pointer.get("contentSha256", ""))
        ):
            raise ValueError("mutable-state generation content checksum mismatch")
        for role in ("core", "plasticity", "optimizer"):
            spec = generation.get("roles", {}).get(role)
            if not isinstance(spec, Mapping):
                raise ValueError("mutable-state generation role is missing")
            checksum = str(spec.get("sha256", ""))
            blob = self.path / "blobs" / (checksum + ".safetensors")
            if (
                str(spec.get("path", ""))
                != "blobs/%s.safetensors" % checksum
                or not blob.is_file()
                or blob.stat().st_size != int(spec.get("bytes", -1))
                or _file_sha256(blob) != checksum
            ):
                raise ValueError("mutable-state %s blob checksum mismatch" % role)
        return generation

    @staticmethod
    def _materialize_blob(source: Path, destination: Path) -> None:
        if destination.is_file() and _file_sha256(destination) == _file_sha256(source):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            shutil.copy2(str(source), str(temporary))
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                temporary.unlink()

    def materialize(
        self, pointer: Mapping[str, Any], engine_path: Path
    ) -> Dict[str, Any]:
        generation = self._generation(pointer)
        for role, filename in (
            ("core", "core.safetensors"),
            ("plasticity", "plasticity.safetensors"),
        ):
            spec = generation["roles"][role]
            source = self.path / str(spec["path"])
            self._materialize_blob(source, Path(engine_path) / filename)
        optimizer_spec = generation["roles"]["optimizer"]
        optimizer_tensors = load_tensors(
            self.path / str(optimizer_spec["path"]), device="cpu"
        )
        optimizer_state = _decode_safe_tree(
            generation["optimizerStructure"], optimizer_tensors
        )
        return {
            "generation": generation,
            "optimizerState": optimizer_state,
        }

    def prior_core_generations(
        self, pointer: Mapping[str, Any], *, bounded: bool = False,
    ) -> Sequence[Tuple[str, Mapping[str, torch.Tensor]]]:
        """Return newest-first, fully verified core recovery generations.

        The collector retains the immediately previous committed generation
        specifically as a recovery point.  It is intentionally not trusted by
        directory name alone: each manifest, content hash, role path, blob
        length, and blob checksum is revalidated through :meth:`_generation`
        before any tensor is exposed to checkpoint repair.
        ``bounded=True`` keeps only a verified lazy inventory and reads the
        affected control tensor on demand, rather than loading a full core map.
        """

        active = self._generation(pointer)
        active_id = str(pointer.get("activeGeneration", ""))
        active_created = float(active.get("createdAt", float("inf")))
        candidates: list[Tuple[float, str, Mapping[str, torch.Tensor]]] = []
        generations = self.path / "generations"
        if not generations.is_dir():
            return ()
        for candidate in generations.iterdir():
            generation_id = candidate.name
            manifest_path = candidate / "manifest.json"
            if (
                generation_id == active_id
                or candidate.is_symlink()
                or not candidate.is_dir()
                or len(generation_id) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in generation_id
                )
                or not manifest_path.is_file()
            ):
                continue
            try:
                candidate_pointer = self.generation_pointer(generation_id)
                generation = self._generation(candidate_pointer)
                created = float(generation.get("createdAt", 0.0))
                if not math.isfinite(created) or created > active_created:
                    continue
                core_spec = generation["roles"]["core"]
                if bounded:
                    from .bounded_tensor_io import LazyTensorMapping
                    core = LazyTensorMapping(self.path / str(core_spec["path"]))
                else:
                    core = load_tensors(
                        self.path / str(core_spec["path"]), device="cpu"
                    )
                candidates.append((created, generation_id, core))
            except (OSError, ValueError, KeyError, TypeError):
                # A broken non-authoritative recovery point must never prevent
                # the authoritative active generation from loading.
                continue
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return tuple((generation_id, core) for _, generation_id, core in candidates)

    def generation_pointer(self, generation_id: str) -> Dict[str, Any]:
        """Reconstruct and verify a pointer to a retained generation."""

        generation_id = str(generation_id)
        if (
            len(generation_id) != 64
            or any(
                character not in "0123456789abcdef"
                for character in generation_id
            )
        ):
            raise ValueError("mutable-state generation id is invalid")
        manifest_path = (
            self.path / "generations" / generation_id / "manifest.json"
        )
        pointer = {
            "format": "omni-mutable-state",
            "formatVersion": 1,
            "activeGeneration": generation_id,
            "contentSha256": generation_id,
            "generationManifest": (
                "generations/%s/manifest.json" % generation_id
            ),
            "generationManifestSha256": _file_sha256(manifest_path),
        }
        generation = self._generation(pointer)
        pointer["replayCount"] = int(generation["replay"]["count"])
        return pointer

    def publish(self, pointer: Mapping[str, Any]) -> None:
        # Validation before publication prevents a dangling active pointer.
        self._generation(pointer)
        atomic_write_json(self.path / "manifest.json", dict(pointer))

    @staticmethod
    def _generation_references(
        root: Path, pointer: Mapping[str, Any]
    ) -> Tuple[str, set[str]]:
        """Read one already-committed generation without rehashing large blobs.

        Publication performs the expensive tensor checksum verification.  The
        collector repeats the canonical manifest checks, then uses only exact
        content-addressed paths.  If any pointer is malformed, collection is
        abandoned before deleting anything.
        """

        generation_id = str(pointer.get("activeGeneration", ""))
        relative = str(pointer.get("generationManifest", ""))
        if (
            len(generation_id) != 64
            or any(value not in "0123456789abcdef" for value in generation_id)
            or relative != "generations/%s/manifest.json" % generation_id
        ):
            raise ValueError("mutable-state generation pointer is invalid")
        manifest_path = root / relative
        generation = read_json(manifest_path)
        body = {
            key: value
            for key, value in generation.items()
            if key != "contentSha256"
        }
        if (
            generation.get("format") != "omni-mutable-state"
            or int(generation.get("formatVersion", 0)) != 1
            or _canonical_sha(body) != generation_id
            or str(generation.get("contentSha256", "")) != generation_id
            or str(pointer.get("contentSha256", "")) != generation_id
        ):
            raise ValueError("mutable-state generation content checksum mismatch")
        references: set[str] = set()
        for role in ("core", "plasticity", "optimizer"):
            spec = generation.get("roles", {}).get(role)
            if not isinstance(spec, Mapping):
                raise ValueError("mutable-state generation role is missing")
            checksum = str(spec.get("sha256", ""))
            expected = "blobs/%s.safetensors" % checksum
            if (
                len(checksum) != 64
                or any(value not in "0123456789abcdef" for value in checksum)
                or str(spec.get("path", "")) != expected
            ):
                raise ValueError("mutable-state blob reference is invalid")
            references.add(expected)
        return generation_id, references

    def _scratch_references(self) -> Tuple[Optional[str], set[str]]:
        pointer_path = self.path / "scratch.json"
        if not pointer_path.is_file():
            return None, set()
        pointer = read_json(pointer_path)
        content_sha = str(pointer.get("contentSha256", ""))
        relative = str(pointer.get("manifest", ""))
        if (
            len(content_sha) != 64
            or any(value not in "0123456789abcdef" for value in content_sha)
            or relative != "scratch/%s.json" % content_sha
        ):
            raise ValueError("pressure-scratch pointer is invalid")
        path = self.path / relative
        if _file_sha256(path) != str(pointer.get("manifestSha256", "")):
            raise ValueError("pressure-scratch manifest checksum mismatch")
        value = read_json(path)
        body = {
            key: item for key, item in value.items() if key != "contentSha256"
        }
        if (
            value.get("format") != "omni-pressure-scratch"
            or _canonical_sha(body) != content_sha
        ):
            raise ValueError("pressure-scratch content checksum mismatch")
        references: set[str] = set()
        for role in ("optimizer", "activations"):
            spec = value.get(role)
            if not isinstance(spec, Mapping):
                raise ValueError("pressure-scratch blob reference is missing")
            checksum = str(spec.get("sha256", ""))
            expected = "blobs/%s.safetensors" % checksum
            if (
                len(checksum) != 64
                or any(character not in "0123456789abcdef" for character in checksum)
                or str(spec.get("path", "")) != expected
            ):
                raise ValueError("pressure-scratch blob reference is invalid")
            references.add(expected)
        return content_sha, references

    def prune_unreferenced(
        self, pointers: Sequence[Optional[Mapping[str, Any]]]
    ) -> Dict[str, Any]:
        """Keep the active and immediately previous recovery generations.

        Long-running learning used to retain every immutable checkpoint blob,
        even after both authoritative pointers had advanced.  That is not
        neural memory; it is unreachable recovery garbage.  Collection runs
        only after the new ``brain.json`` and store pointer are committed and
        never follows symlinks or removes an unrecognized filename.
        """

        try:
            retained_generations: set[str] = set()
            retained_blobs: set[str] = set()
            for pointer in pointers:
                if not isinstance(pointer, Mapping) or not pointer:
                    continue
                generation_id, references = self._generation_references(
                    self.path, pointer
                )
                retained_generations.add(generation_id)
                retained_blobs.update(references)
            if not retained_generations:
                raise ValueError("no committed mutable-state generation to retain")
            retained_scratch, scratch_blobs = self._scratch_references()
            retained_blobs.update(scratch_blobs)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.last_gc = {
                "completed": False,
                "reason": str(error),
                "generationsRetained": 0,
                "generationsRemoved": 0,
                "blobsRemoved": 0,
                "bytesReclaimed": 0,
            }
            return dict(self.last_gc)

        generations_removed = 0
        blobs_removed = 0
        bytes_reclaimed = 0
        generations = self.path / "generations"
        if generations.is_dir():
            for candidate in generations.iterdir():
                if (
                    candidate.name in retained_generations
                    or len(candidate.name) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in candidate.name
                    )
                    or candidate.is_symlink()
                    or not candidate.is_dir()
                ):
                    continue
                bytes_reclaimed += sum(
                    item.stat().st_size
                    for item in candidate.rglob("*")
                    if item.is_file() and not item.is_symlink()
                )
                shutil.rmtree(candidate)
                generations_removed += 1

        blobs = self.path / "blobs"
        if blobs.is_dir():
            for candidate in blobs.iterdir():
                relative = "blobs/" + candidate.name
                stem = candidate.stem
                if (
                    relative in retained_blobs
                    or candidate.is_symlink()
                    or not candidate.is_file()
                    or candidate.suffix != ".safetensors"
                    or len(stem) != 64
                    or any(character not in "0123456789abcdef" for character in stem)
                ):
                    continue
                size = candidate.stat().st_size
                candidate.unlink()
                blobs_removed += 1
                bytes_reclaimed += size

        scratch = self.path / "scratch"
        if scratch.is_dir():
            for candidate in scratch.iterdir():
                if (
                    candidate.is_symlink()
                    or not candidate.is_file()
                    or candidate.suffix != ".json"
                    or len(candidate.stem) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in candidate.stem
                    )
                    or candidate.stem == retained_scratch
                ):
                    continue
                size = candidate.stat().st_size
                candidate.unlink()
                bytes_reclaimed += size

        self.last_gc = {
            "completed": True,
            "reason": "unreferenced recovery generations reclaimed",
            "generationsRetained": len(retained_generations),
            "generationsRemoved": generations_removed,
            "blobsRemoved": blobs_removed,
            "bytesReclaimed": bytes_reclaimed,
        }
        return dict(self.last_gc)

    def recover(
        self,
        pointer: Mapping[str, Any],
        engine_path: Path,
        replay: DurableReplayBuffer,
    ) -> Dict[str, Any]:
        active_pointer_corrupt = False
        active_path = self.path / "manifest.json"
        if active_path.is_file():
            try:
                active = read_json(active_path)
                self._generation(active)
                active_pointer_corrupt = active != dict(pointer)
            except (OSError, ValueError, KeyError, TypeError):
                active_pointer_corrupt = True
        result = self.materialize(pointer, engine_path)
        replay_status = replay.verify_checkpoint(result["generation"]["replay"])
        pending_examples = int(replay_status.get("pendingExamples", 0))
        if pending_examples:
            # Rows above the generation high-water mark belong to an
            # interrupted, uncommitted neural batch. Leaving them readable
            # would silently duplicate replay learning after the record cursor
            # retries that suffix.
            replay.truncate(int(replay_status["committedExamples"]))
            verified = replay.verify_checkpoint(result["generation"]["replay"])
            replay_status = {
                **verified,
                "rolledBackExamples": pending_examples,
                "pendingExamples": 0,
            }
        if active_pointer_corrupt or not active_path.is_file():
            self.publish(pointer)
        self.last_recovery = {
            "recovered": bool(active_pointer_corrupt),
            "reason": (
                "repaired corrupt or incomplete active pointer from brain metadata"
                if active_pointer_corrupt
                else "active generation verified"
            ),
            "activeGeneration": pointer.get("activeGeneration"),
            "identityReboundFrom": (
                result["generation"].get("brainId")
                if result["generation"].get("brainId") != self.brain_id
                else None
            ),
            "replay": replay_status,
        }
        return {**result, "pointer": dict(pointer), "replay": replay_status}

    def save_pressure_scratch(
        self,
        *,
        optimizer_state: Mapping[str, Any],
        activations: Mapping[str, torch.Tensor],
        metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        optimizer_tensors: Dict[str, torch.Tensor] = {}
        structure = _encode_safe_tree(optimizer_state, optimizer_tensors)
        estimate = self._tensor_bytes_estimate(optimizer_tensors) + self._tensor_bytes_estimate(activations)
        self.policy.require_disk(estimate, "memory-pressure offload")
        body = {
            "format": "omni-pressure-scratch",
            "formatVersion": 1,
            "brainId": self.brain_id,
            "createdAt": time.time(),
            "optimizer": self._write_tensor_blob(optimizer_tensors, role="optimizer-scratch"),
            "optimizerStructure": structure,
            "activations": self._write_tensor_blob(activations, role="activation-scratch"),
            "metadata": dict(metadata),
        }
        content_sha = _canonical_sha(body)
        value = {**body, "contentSha256": content_sha}
        path = self.path / "scratch" / (content_sha + ".json")
        atomic_write_json(path, value)
        pointer = {
            "format": "omni-pressure-scratch",
            "formatVersion": 1,
            "contentSha256": content_sha,
            "manifest": "scratch/%s.json" % content_sha,
            "manifestSha256": _file_sha256(path),
        }
        atomic_write_json(self.path / "scratch.json", pointer)
        return pointer

    def load_pressure_optimizer(self, pointer: Mapping[str, Any]) -> Dict[str, Any]:
        content_sha = str(pointer.get("contentSha256", ""))
        path = self.path / "scratch" / (content_sha + ".json")
        if (
            str(pointer.get("manifest", "")) != "scratch/%s.json" % content_sha
            or _file_sha256(path) != str(pointer.get("manifestSha256", ""))
        ):
            raise ValueError("pressure scratch manifest checksum mismatch")
        value = read_json(path)
        body = {key: item for key, item in value.items() if key != "contentSha256"}
        if (
            value.get("format") != "omni-pressure-scratch"
            or _canonical_sha(body) != content_sha
            or str(value.get("brainId", "")) != self.brain_id
        ):
            raise ValueError("pressure scratch content checksum mismatch")
        spec = value["optimizer"]
        blob = self.path / str(spec["path"])
        if _file_sha256(blob) != str(spec["sha256"]):
            raise ValueError("pressure optimizer blob checksum mismatch")
        tensors = load_tensors(blob, device="cpu")
        return _decode_safe_tree(value["optimizerStructure"], tensors)


def copy_mutable_state_snapshot(
    source_engine: Path,
    destination_engine: Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Copy exactly the state generation and replay rows declared by metadata."""

    source_engine = Path(source_engine)
    destination_engine = Path(destination_engine)
    state = dict(metadata or read_json(source_engine / "brain.json"))
    pointer = state.get("mutable_state")
    if not isinstance(pointer, Mapping):
        return None
    brain_id = str(state.get("brain_id", ""))
    source_policy = ResourcePolicy(source_engine)
    source_store = MutableStateStore(source_engine / "state", brain_id, source_policy)
    generation = source_store._generation(pointer)
    source_replay_path = source_engine / "state" / "replay.sqlite3"
    if not source_replay_path.is_file():
        raise ValueError("mutable-state snapshot is missing latent replay")
    source_replay = DurableReplayBuffer(
        source_replay_path, source_policy, read_only=True
    )
    source_replay.verify_checkpoint(generation["replay"])

    destination_engine.mkdir(parents=True, exist_ok=True)
    destination_store_path = destination_engine / "state"
    staged_store = destination_engine / (".state-%s.next" % uuid.uuid4().hex)
    backup_store = destination_engine / (".state-%s.previous" % uuid.uuid4().hex)
    destination_policy = ResourcePolicy(destination_engine)
    estimated_bytes = source_replay_path.stat().st_size + sum(
        int(spec.get("bytes", 0))
        for spec in generation["roles"].values()
        if isinstance(spec, Mapping)
    )
    destination_policy.require_disk(
        max(64 * 1024, estimated_bytes * 2),
        "mutable-state copy-on-write snapshot",
    )
    staged_store.mkdir(parents=False, exist_ok=False)
    previous_moved = False
    try:
        for spec in generation["roles"].values():
            source = source_engine / "state" / str(spec["path"])
            destination = staged_store / str(spec["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(
                destination.name + "." + uuid.uuid4().hex + ".tmp"
            )
            try:
                shutil.copy2(str(source), str(temporary))
                if (
                    _file_sha256(temporary) != str(spec["sha256"])
                    or temporary.stat().st_size != int(spec["bytes"])
                ):
                    raise ValueError("mutable-state COW blob checksum mismatch")
                os.replace(str(temporary), str(destination))
            finally:
                if temporary.exists():
                    temporary.unlink()
        generation_source = source_engine / "state" / str(
            pointer["generationManifest"]
        )
        generation_destination = staged_store / str(
            pointer["generationManifest"]
        )
        atomic_write_bytes(generation_destination, generation_source.read_bytes())

        destination_replay_path = staged_store / "replay.sqlite3"
        destination_replay = DurableReplayBuffer(
            destination_replay_path, destination_policy
        )
        high_water = int(generation["replay"]["highWaterId"])
        with closing(source_replay._connect()) as source_connection:
            rows = source_connection.execute(
                """
                SELECT sequence, created_at, sha256, dtype, shape_json, payload
                FROM replay WHERE sequence <= ? ORDER BY sequence
                """,
                (high_water,),
            )
            with closing(destination_replay._connect()) as destination_connection:
                destination_connection.execute("BEGIN IMMEDIATE")
                try:
                    for row in rows:
                        destination_connection.execute(
                            """
                            INSERT INTO replay
                            (sequence, created_at, sha256, dtype, shape_json, payload)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            row,
                        )
                    destination_connection.execute("COMMIT")
                    destination_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except BaseException:
                    destination_connection.execute("ROLLBACK")
                    raise
        destination_replay.verify_checkpoint(generation["replay"])
        staged_state_store = MutableStateStore(
            staged_store, brain_id, destination_policy
        )
        staged_state_store.publish(pointer)
        # No protocol pointer is exposed until the complete directory is ready.
        # Directory replacement also prevents an interrupted restore from
        # deleting the previously committed replay database.
        if destination_store_path.exists():
            os.replace(str(destination_store_path), str(backup_store))
            previous_moved = True
        try:
            os.replace(str(staged_store), str(destination_store_path))
        except BaseException:
            if previous_moved and backup_store.exists():
                os.replace(str(backup_store), str(destination_store_path))
                previous_moved = False
            raise
        if previous_moved:
            previous_moved = False
            try:
                shutil.rmtree(backup_store)
            except OSError:
                # The new authoritative state is already committed. A stale
                # app-owned backup is recoverable garbage, not a reason to
                # report that the COW promotion failed.
                pass
    finally:
        if staged_store.exists():
            shutil.rmtree(staged_store)
        if previous_moved and backup_store.exists() and not destination_store_path.exists():
            os.replace(str(backup_store), str(destination_store_path))
    return dict(pointer)
