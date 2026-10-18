"""Observed hardware and small, explicitly admitted copy probes; never a model."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import statistics
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path


TOPOLOGY_FORMAT = "omni-device-resource-topology"
PROBE_MAX_BYTES = 4 * 1024 * 1024


def _iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _count(call):
    try:
        result = call()
        return int(result) if not isinstance(result, bool) and int(result) >= 0 else None
    except (AttributeError, RuntimeError, ValueError, TypeError, OSError):
        return None


def _truth(call):
    try:
        return bool(call())
    except (AttributeError, RuntimeError, ValueError, TypeError, OSError):
        return False


def _device(identifier, label, kind, backend, group, total, available, allocated, reserved,
            unified, *, execution=None, native=False, reason=None):
    return {"id": identifier, "label": label, "kind": kind, "backend": backend,
            "memoryGroupId": group, "totalBytes": total, "availableBytes": available,
            "allocatedBytes": allocated, "reservedBytes": reserved, "unifiedMemory": unified,
            "nativeComputeSupported": bool(native),
            **({"executionDevice": execution} if execution is not None else {}),
            **({"unavailableReason": reason} if reason else {})}


def collect_device_topology(*, policy=None, storage_paths=(), measure=False, probe_budget_bytes=PROBE_MAX_BYTES * 4,
                            cancelled=lambda: False, torch_module=None, host_reading=None,
                            native_support=None, clock_ns=time.perf_counter_ns):
    if type(measure) is not bool or type(probe_budget_bytes) is not int or probe_budget_bytes < 0:
        raise ValueError("hardware profiling requires an explicit nonnegative probe budget")
    if torch_module is None:
        import torch as torch_module
    if host_reading is None:
        if policy is None:
            from .offload import ResourcePolicy
            policy = ResourcePolicy(Path.cwd(), include_accelerator_memory=False)
        host_reading = policy.readings(fresh_memory_sample=True)
    total = host_reading.total_memory_bytes
    available = host_reading.available_memory_bytes
    if native_support is None:
        from .packed_native_kernel import native_packed_kernel_available
        native_support = {"cpu": _truth(native_packed_kernel_available), "cuda": False,
                          "mps": hasattr(getattr(torch_module, "mps", None), "compile_shader")}
    devices = [_device("ram:system", "System RAM", "ram", "cpu", "ram:system", total, available,
                       host_reading.process_memory_bytes, None, False, execution="cpu", native=native_support.get("cpu", False))]
    warnings = []
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and _truth(cuda.is_available):
        for index in range(_count(cuda.device_count) or 0):
            if cancelled():
                raise InterruptedError("device enumeration cancelled")
            try:
                properties = cuda.get_device_properties(index)
                native_id = str(getattr(properties, "uuid", "") or getattr(properties, "pci_bus_id", "") or index)
                if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,240}", native_id) is None:
                    native_id = hashlib.sha256(native_id.encode()).hexdigest()
                identifier = "cuda:" + native_id
                try:
                    free, capacity = (int(value) for value in cuda.mem_get_info(index))
                except (AttributeError, RuntimeError, ValueError):
                    free, capacity = None, _count(lambda: properties.total_memory)
                supported = bool(native_support.get("cuda", False))
                devices.append(_device(identifier, str(getattr(properties, "name", "CUDA %d" % index)), "gpu", "cuda", identifier,
                    capacity, free, _count(lambda: cuda.memory_allocated(index)), _count(lambda: cuda.memory_reserved(index)),
                    False, execution="cuda:%d" % index, native=supported,
                    reason=None if supported else "native-packed-compute-kernel-unavailable"))
            except (AttributeError, RuntimeError, ValueError, OSError) as error:
                warnings.append("CUDA device %d could not be inspected: %s" % (index, error))
    mps = getattr(torch_module, "mps", None)
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps is not None and mps_backend is not None and _truth(mps_backend.is_available):
        unified = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
        recommended = _count(lambda: mps.recommended_max_memory())
        driver = _count(lambda: mps.driver_allocated_memory())
        free = max(0, recommended - driver) if recommended is not None and driver is not None else None
        if unified and available is not None:
            free = min(free, available) if free is not None else available
        entry = _device("mps:0", "Apple Metal", "gpu", "mps", "ram:system" if unified else "mps:0",
            total if unified else None, free, _count(lambda: mps.current_allocated_memory()), driver,
            unified, execution="mps", native=native_support.get("mps", False),
            reason=None if native_support.get("mps", False) else "native-packed-compute-kernel-unavailable")
        entry["recommendedLimitBytes"] = recommended
        devices.append(entry)
    for backend in ("xpu", "npu"):
        module = getattr(torch_module, backend, None)
        if module is None or not _truth(lambda: module.is_available()):
            continue
        for index in range(_count(lambda: module.device_count()) or 0):
            identifier = "%s:%d" % (backend, index)
            try:
                label = str(module.get_device_name(index))
            except (AttributeError, RuntimeError, ValueError):
                label = identifier
            devices.append(_device(identifier, label, "npu" if backend == "npu" else "gpu", backend, identifier,
                None, None, _count(lambda: module.memory_allocated(index)), _count(lambda: module.memory_reserved(index)),
                None, execution=identifier, reason="native-packed-backend-and-memory-capacity-unverified"))
    seen_paths = set()
    for supplied in storage_paths:
        path = Path(supplied)
        if not path.is_absolute() or path.is_symlink() or not path.is_dir():
            raise ValueError("hardware probes require an existing selected pool directory")
        path = path.resolve()
        if str(path) in seen_paths:
            continue
        seen_paths.add(str(path))
        stat = path.stat()
        usage = shutil.disk_usage(path)
        identifier = "storage:" + hashlib.sha256(str(path).encode()).hexdigest()[:24]
        entry = _device(identifier, str(path), "storage", "filesystem", "volume:%d" % stat.st_dev,
                        int(usage.total), int(usage.free), None, None, False)
        entry.update(poolDirectory=str(path), filesystemDeviceId=str(stat.st_dev), readOnly=not os.access(path, os.W_OK))
        devices.append(entry)
    links = []
    memory_devices = [item for item in devices if "executionDevice" in item]
    for source in memory_devices:
        for destination in memory_devices:
            if source is destination:
                continue
            peer = None
            if source["backend"] == destination["backend"] == "cuda":
                try:
                    peer = bool(cuda.can_device_access_peer(int(source["executionDevice"].split(":")[-1]), int(destination["executionDevice"].split(":")[-1])))
                except (AttributeError, RuntimeError, ValueError):
                    pass
            links.append({"from": source["id"], "to": destination["id"], "peerAccess": peer})
    result = {"format": TOPOLOGY_FORMAT, "version": 1, "measuredAt": _iso(), "platform": platform.system().lower(),
              "architecture": platform.machine(), "devices": devices, "links": links,
              "measurementRequested": measure, "neuralModelConstructed": False, "warnings": warnings}
    if measure:
        _measure(result, torch_module, policy, probe_budget_bytes, cancelled, clock_ns)
    return result


def _measurement(samples, byte_count, scope):
    middle = int(statistics.median(samples))
    if middle <= 0:
        raise RuntimeError("hardware probe clock did not advance")
    return {"bytesPerSecond": max(1, byte_count * 1_000_000_000 // middle), "sampleBytes": byte_count,
            "runs": len(samples), "medianNanoseconds": middle, "measuredAt": _iso(), "scope": scope}


def _sync(torch, entry):
    backend = entry["backend"]
    if backend == "cuda":
        torch.cuda.synchronize(entry["executionDevice"])
    elif backend == "mps":
        torch.mps.synchronize()
    elif backend in {"xpu", "npu"}:
        getattr(torch, backend).synchronize()


def _measure(topology, torch, policy, budget, cancelled, clock_ns):
    # Small cache-scale copies give placement evidence, not neural throughput.
    count = min(PROBE_MAX_BYTES, budget // 8)
    if count < 4096:
        topology["warnings"].append("Device measurements await at least 32 KiB of admitted probe memory.")
        return
    by_id = {entry["id"]: entry for entry in topology["devices"]}

    def boundary():
        if cancelled():
            raise InterruptedError("hardware measurement cancelled at a copy boundary")

    def check(entries, bytes_each):
        boundary()
        for entry in entries:
            free = entry["availableBytes"]
            capacity = entry["totalBytes"] or entry.get("recommendedLimitBytes")
            # Enumeration describes identity/capacity. Another application can
            # allocate between probes, so each new GPU copy observes allocator
            # headroom again instead of spending the initial snapshot twice.
            if entry["backend"] == "cuda":
                free, capacity = torch.cuda.mem_get_info(entry["executionDevice"])
            elif entry["backend"] == "mps":
                capacity = torch.mps.recommended_max_memory()
                free = max(0, capacity - torch.mps.driver_allocated_memory())
            if type(free) is not int or type(capacity) is not int or free < 0 or capacity < 1:
                raise RuntimeError("device probe awaits measured free memory")
            reserve = max(1, int(capacity or 0) // 10)
            if free is None or free < bytes_each * 4 + reserve:
                raise RuntimeError("device probe awaits measured free memory")

    def lease():
        return policy.reserve_ram(count * 8, "bounded hardware copy measurement") if policy is not None else nullcontext()

    def copy_probe(source, destination, scope):
        check((source, destination), count)
        with lease() as held:
            left = torch.empty(count, dtype=torch.uint8, device=source["executionDevice"])
            right = torch.empty(count, dtype=torch.uint8, device=destination["executionDevice"])
            if held is not None and hasattr(held, "mark_allocated"):
                held.mark_allocated(count * 2)
            try:
                left.fill_(7)
                _sync(torch, source)
                right.copy_(left)
                _sync(torch, destination)
                samples = []
                for _ in range(3):
                    boundary()
                    start = clock_ns()
                    right.copy_(left)
                    _sync(torch, destination)
                    samples.append(clock_ns() - start)
                return _measurement(samples, count, scope)
            finally:
                del left, right
    for entry in topology["devices"]:
        boundary()
        try:
            if entry["backend"] in {"cpu", "cuda", "mps"}:
                entry["measurement"] = copy_probe(entry, entry, "memory-copy")
            elif entry["kind"] == "storage" and not entry["readOnly"]:
                if entry["availableBytes"] is None or entry["availableBytes"] < count * 4:
                    raise RuntimeError("storage probe awaits measured free space")
                with lease():
                    payload = bytes([73]) * count
                    path = Path(entry["poolDirectory"]) / (".omni-device-probe-%s.tmp" % uuid.uuid4().hex)
                    handle = None
                    try:
                        usage = shutil.disk_usage(path.parent)
                        reserve = (max(int(policy.configured_disk_reserve), policy._adaptive_disk_reserve(usage.total))
                                   if policy is not None else 20 * 1024 ** 3)
                        if usage.free - count < reserve:
                            raise RuntimeError("selected-drive probe would cross its disk reserve")
                        handle = path.open("xb", buffering=0)
                        os.chmod(path, 0o600)
                        samples = []
                        for _ in range(3):
                            boundary()
                            handle.seek(0)
                            start = clock_ns()
                            written = handle.write(payload)
                            if written != count:
                                raise RuntimeError("storage probe was not fully written")
                            handle.flush(); os.fsync(handle.fileno())
                            samples.append(clock_ns() - start)
                        entry["measurement"] = _measurement(samples, count, "durable-file-write")
                    finally:
                        if handle is not None:
                            handle.close()
                            path.unlink(missing_ok=True)
        except InterruptedError:
            raise
        except (RuntimeError, OSError, ValueError, AttributeError, TypeError) as error:
            topology["warnings"].append("%s measurement unavailable: %s" % (entry["label"], error))
    for link in topology["links"]:
        source, destination = by_id[link["from"]], by_id[link["to"]]
        if source["backend"] not in {"cpu", "cuda", "mps"} or destination["backend"] not in {"cpu", "cuda", "mps"}:
            link["unavailableReason"] = "transfer-backend-unavailable"
            continue
        try:
            link["measurement"] = copy_probe(source, destination, "device-transfer")
        except InterruptedError:
            raise
        except (RuntimeError, OSError, ValueError, AttributeError, TypeError) as error:
            link["unavailableReason"] = str(error)
