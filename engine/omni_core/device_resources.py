"""Per-device selections and allocator admission for the native runtime.

RAM retains ResourcePolicy's measured family ceiling. CUDA/MPS add their
allocator-specific process limits; neither is advertised as global OS RSS
isolation. One physical pool may have several compute interfaces.
"""

from __future__ import annotations

import copy
import math
import re
import threading
import time
from pathlib import Path

from .device_topology import collect_device_topology
from .regional_execution import canonical_device, validate_regional_execution_settings


DEVICE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,255}\Z")


def validate_device_resource_settings(value):
    if not isinstance(value, dict) or value.get("version") != 1 or type(value.get("version")) is not int:
        raise ValueError("unsupported device resource settings")
    fields = {"version", "allocations", "allowStorageSpill", "autoIncreaseStorage", "storageDirectories"}
    if set(value) - (fields | {"regionalExecution"}) or not fields.issubset(value):
        raise ValueError("device resource settings contain unsupported fields")
    if not isinstance(value["allocations"], list) or not isinstance(value["storageDirectories"], list):
        raise ValueError("device allocations and storage directories must be lists")
    if type(value["allowStorageSpill"]) is not bool or type(value["autoIncreaseStorage"]) is not bool:
        raise ValueError("device resource switches must be booleans")
    seen = set()
    for entry in value["allocations"]:
        if not isinstance(entry, dict) or set(entry) != {"deviceId", "enabled", "mode", "percent", "autoIncrease"}:
            raise ValueError("device allocation must contain its exact settings")
        identifier = entry["deviceId"]
        if not isinstance(identifier, str) or not DEVICE_ID.fullmatch(identifier) or identifier in seen:
            raise ValueError("device allocation identity is invalid or duplicated")
        seen.add(identifier)
        if (type(entry["enabled"]) is not bool or type(entry["autoIncrease"]) is not bool
                or type(entry["mode"]) is not str or entry["mode"] not in {"auto", "manual"}):
            raise ValueError("device allocation policy is invalid")
        percent = entry["percent"]
        minimum = 30 if identifier == "ram:system" else 1
        if type(percent) not in (int, float) or not math.isfinite(percent) or not minimum <= percent <= 100:
            raise ValueError("device allocation percentage is invalid")
    for path in value["storageDirectories"]:
        if not isinstance(path, str) or not path or "\0" in path:
            raise ValueError("selected storage directory is invalid")
    if len(set(value["storageDirectories"])) != len(value["storageDirectories"]):
        raise ValueError("selected storage directories must be unique")
    if "regionalExecution" in value:
        regional = validate_regional_execution_settings(value["regionalExecution"])
        if regional["enabled"] and any(not any(item["deviceId"] == route["deviceId"] and item["enabled"]
                                              for item in value["allocations"]) for route in regional["routes"]):
            raise ValueError("each regional route requires an explicitly enabled device allocation")
    return copy.deepcopy(value)


def device_resource_budgets(topology, settings, *, system_ram_budget_bytes, disk_reserve_bytes):
    """Physical ceilings; current free bytes affect admission, not saved shape."""
    settings = validate_device_resource_settings(settings)
    selections = {entry["deviceId"]: entry for entry in settings["allocations"]}
    budgets = []
    for device in topology["devices"]:
        selection = selections.get(device["id"], {"enabled": device["kind"] in {"ram", "storage"} or device["nativeComputeSupported"],
            "mode": "auto", "percent": 90, "autoIncrease": False})
        capacity = device["totalBytes"]
        reserve = 0
        if device["kind"] == "ram":
            ceiling = min(int(capacity or 0), max(0, int(system_ram_budget_bytes)))
            reserve = max(0, int(capacity or 0) - ceiling)
            if selection["mode"] == "manual":
                ceiling = min(ceiling, int(int(capacity or 0) * selection["percent"] / 100))
        elif device["kind"] == "storage":
            reserve = max(0, int(disk_reserve_bytes))
            capacity = device["availableBytes"]
            ceiling = int(max(0, int(capacity or 0) - reserve) * selection["percent"] / 100)
        else:
            capacity = capacity if capacity is not None else device.get("recommendedLimitBytes")
            reserve = max(0, int(capacity or 0) // 10)
            ceiling = int(max(0, int(capacity or 0) - reserve) * selection["percent"] / 100)
            if device["memoryGroupId"] == "ram:system":
                ceiling = min(ceiling, max(0, int(system_ram_budget_bytes)))
            if device.get("recommendedLimitBytes") is not None:
                ceiling = min(ceiling, device["recommendedLimitBytes"])
        enabled = bool(selection["enabled"]) and not (device["kind"] == "storage" and (not settings["allowStorageSpill"] or device.get("readOnly")))
        budgets.append({"deviceId": device["id"], "memoryGroupId": device["memoryGroupId"], "kind": device["kind"],
            "enabled": enabled, "capacityBytes": capacity, "ceilingBytes": ceiling if enabled else 0,
            "currentlyAvailableBytes": device["availableBytes"], "reserveBytes": reserve,
            "selection": selection["mode"], "autoIncrease": bool(selection["autoIncrease"]),
            **({"reason": "capacity-unavailable"} if capacity is None else {})})
    return budgets


class DeviceResourceAdmissionError(RuntimeError):
    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


class DeviceResourceManager:
    def __init__(self, settings, *, policy, torch_module=None, topology_provider=None):
        self.settings = validate_device_resource_settings(settings)
        self.policy = policy
        if torch_module is None:
            import torch as torch_module
        self.torch = torch_module
        self.topology_provider = topology_provider
        self._lock = threading.RLock()
        self._topology = None
        self._sampled_at = float("-inf")
        self._limits = {}
        self._allocator_limits = {}
        self._drive_pool = None
        self.regional_execution_provider = None

    def allocate_spill_file(self, byte_count, operation, suffix, *, cancelled=None):
        from .drive_pool import DrivePoolManager
        with self._lock:
            if self._drive_pool is None:
                self._drive_pool = DrivePoolManager(self.policy, self.settings)
        return self._drive_pool.allocate(byte_count, operation, suffix, cancelled=cancelled)

    def topology(self, *, fresh=False):
        with self._lock:
            if fresh or self._topology is None or time.monotonic() - self._sampled_at >= 1.0:
                self._topology = (self.topology_provider() if self.topology_provider is not None else collect_device_topology(
                    policy=self.policy, storage_paths=self.settings["storageDirectories"], torch_module=self.torch))
                self._sampled_at = time.monotonic()
            return self._topology

    def budgets(self, host_status, *, fresh=False, observed_topology=None):
        topology = self.topology(fresh=fresh) if observed_topology is None else observed_topology
        proposed = device_resource_budgets(topology, self.settings,
            system_ram_budget_bytes=int(host_status["systemRamBudgetBytes"]), disk_reserve_bytes=int(host_status["diskReserveBytes"]))
        with self._lock:
            for budget in proposed:
                identifier = budget["deviceId"]
                selected = self._limits.get(identifier)
                proposed_limit = budget["ceilingBytes"]
                allow_increase = budget["autoIncrease"] and (budget["kind"] != "storage" or self.settings["autoIncreaseStorage"])
                if selected is None or allow_increase and proposed_limit > selected:
                    self._limits[identifier] = proposed_limit
                # A temporarily occupied volume does not shrink its selected
                # quota. Every write still checks its actual remaining space.
                budget["ceilingBytes"] = (self._limits[identifier] if budget["kind"] == "storage"
                    else min(self._limits[identifier], proposed_limit)) if budget["enabled"] else 0
        return proposed

    def _entry(self, execution_device, host_status, *, fresh=False):
        name = canonical_device(execution_device)
        if name == "cuda":
            name = "cuda:%d" % self.torch.cuda.current_device()
        topology = self.topology()
        device = next((entry for entry in topology["devices"] if entry.get("executionDevice") == name), None)
        if device is None:
            raise DeviceResourceAdmissionError("The selected execution device is unavailable", {"executionDevice": name})
        # Mutable readings belong to this observation, not the identity cache.
        # A failed fresh probe must not poison or silently reuse another call's
        # successful allocation observation.
        device = dict(device)
        if fresh and device["backend"] == "cuda":
            try:
                free, total = self.torch.cuda.mem_get_info(name)
                device.update(totalBytes=int(total), availableBytes=int(free), allocatedBytes=int(self.torch.cuda.memory_allocated(name)),
                              reservedBytes=int(self.torch.cuda.memory_reserved(name)))
            except (AttributeError, RuntimeError, ValueError):
                device["availableBytes"] = None
        elif fresh and device["backend"] == "mps":
            try:
                recommended = int(self.torch.mps.recommended_max_memory())
                driver = int(self.torch.mps.driver_allocated_memory())
                device.update(recommendedLimitBytes=recommended, availableBytes=max(0, recommended-driver),
                              allocatedBytes=int(self.torch.mps.current_allocated_memory()), reservedBytes=driver)
            except (AttributeError, RuntimeError, ValueError):
                device["availableBytes"] = None
        observed = {**topology, "devices": [device if entry["id"] == device["id"] else entry for entry in topology["devices"]]}
        budget = next(item for item in self.budgets(host_status, observed_topology=observed) if item["deviceId"] == device["id"])
        return device, budget

    def device_budget(self, execution_device, host_status, *, fresh=False):
        device, budget = self._entry(execution_device, host_status, fresh=fresh)
        used = max(int(device.get("allocatedBytes") or 0), int(device.get("reservedBytes") or 0))
        available = device["availableBytes"]
        limit = budget["ceilingBytes"]
        if available is not None:
            limit = min(limit, max(0, int(available) - budget["reserveBytes"]) + used)
        measured = available is not None
        if device["memoryGroupId"] == "ram:system":
            # MPS and CPU do not own two independent memory pools. Account for
            # renderer/worker/other model residency as well as Metal's allocator,
            # using the same fresh family observation as host allocation checks.
            resident = host_status.get("admissionResidentMemoryBytes")
            safe = host_status.get("availableSafeRamBytes")
            measured = (measured and host_status.get("ramAdmissionVerified") is True
                        and type(resident) is int and type(safe) is int)
            if measured:
                if device["backend"] == "cpu":
                    used = max(used, resident)
                room = max(0, min(int(host_status["systemRamBudgetBytes"]) - resident, safe))
                limit = min(limit, used + room)
            else:
                limit = 0
        return {**budget, "executionDevice": device["executionDevice"], "allocatedBytes": used,
            "liveCeilingBytes": max(0, limit), "remainingBytes": max(0, limit - used),
            "nativeComputeSupported": device["nativeComputeSupported"], "backend": device["backend"],
            "sharedWithRam": device["memoryGroupId"] == "ram:system", "capacityMeasured": measured}

    def require_device(self, execution_device, byte_count, operation, host_status):
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("device reservation requires a nonnegative byte count")
        status = self.device_budget(execution_device, host_status, fresh=True)
        if not status["enabled"] or not status["capacityMeasured"] or byte_count > status["remainingBytes"]:
            raise DeviceResourceAdmissionError(operation + " awaits the selected device budget", status)
        if status["backend"] in {"cuda", "mps"}:
            self.apply_allocator_limit(execution_device, host_status, fresh=True)
        return status

    def apply_allocator_limit(self, execution_device, host_status, *, fresh=False):
        device, budget = self._entry(execution_device, host_status, fresh=fresh)
        name = device["executionDevice"]
        ceiling = budget["ceilingBytes"]
        denominator = device.get("recommendedLimitBytes") if device["backend"] == "mps" else device["totalBytes"]
        if ceiling <= 0 or denominator is None or denominator <= 0:
            raise DeviceResourceAdmissionError("Device allocation is disabled or its capacity is unknown", budget)
        used = max(int(device.get("allocatedBytes") or 0), int(device.get("reservedBytes") or 0))
        if used > ceiling:
            raise DeviceResourceAdmissionError("Existing device allocations exceed the selected ceiling; unload or increase the allocation", {**budget, "allocatedBytes": used})
        fraction = min(1.0, ceiling / denominator)
        # On MPS zero disables the limit. It must never represent a disabled device.
        if not 0.0 < fraction <= 1.0:
            raise DeviceResourceAdmissionError("The selected allocator fraction is invalid", budget)
        with self._lock:
            # Torch's ceiling is process-global. Another loaded instance can
            # have applied its own choice since our previous operation; a
            # per-manager cache is not proof that our limit remains installed.
            try:
                if device["backend"] == "cuda":
                    self.torch.cuda.set_per_process_memory_fraction(fraction, name)
                elif device["backend"] == "mps":
                    self.torch.mps.set_per_process_memory_fraction(fraction)
                else:
                    return {"applied": False, "reason": "allocator-limit-api-unavailable"}
            except (AttributeError, RuntimeError, ValueError, OSError) as error:
                raise DeviceResourceAdmissionError("The selected device allocator limit could not be applied", {
                    **budget, "executionDevice": name, "allocatorLimitApplied": False,
                }) from error
            self._allocator_limits[name] = fraction
        return {"applied": True, "executionDevice": name, "fraction": fraction,
            "ceilingBytes": ceiling, "scope": "pytorch-process-allocator", "osPhysicalReservation": False}

    def status(self, host_status):
        topology = self.topology()
        budgets = self.budgets(host_status)
        groups = {}
        for budget in budgets:
            if budget["enabled"]:
                groups[budget["memoryGroupId"]] = max(groups.get(budget["memoryGroupId"], 0), budget["ceilingBytes"])
        regional = self.regional_execution_provider() if callable(self.regional_execution_provider) else None
        regional_active = bool(regional and regional["active"])
        regional_gpus = {item["executionDevice"] for item in regional["routes"]
                         if item["executionDevice"].startswith("cuda:")} if regional_active else set()
        return {"version": 1, "budgets": budgets, "physicalGroupCeilings": groups,
            "allocatorLimits": dict(self._allocator_limits), "topologyMeasuredAt": topology["measuredAt"],
            "unifiedMemoryCountedOnce": True, "multiGpuComputeScheduling": len(regional_gpus) > 1,
            "executionPlacement": "manual-decoder-regions-with-admitted-activation-roundtrips" if regional_active else "current-operation-device-with-per-device-admission",
            "regionalExecution": regional,
            "spillPlacement": {"active": self._drive_pool is not None,
                "scope": "authorized-ephemeral-native-and-attention-pages",
                "checkpointRelocation": False,
                "physicalGroups": self.policy.shared_ledger.drive_usage(self.policy.shared_resource_owner_id)
                    if self._drive_pool is not None else []}}
