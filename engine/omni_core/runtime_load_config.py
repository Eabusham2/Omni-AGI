"""Reviewed resource-only overrides before a native cold-load allocation.

Offline RAM/spill changes must govern construction, not only a later save.
Learned geometry, sampling, training, origin and capability state are never
taken from this public resource dictionary.
"""
from dataclasses import replace
from typing import Mapping

from .config import safe_rounded_storage_bytes_per_second


REVIEWED_RUNTIME_KEYS = frozenset({"name", "contextWindowTokens", "memoryOffloadBytes", "storagePoolBytes", "contextOffloadBudgetBytes",
    "memoryResidentItems", "memoryOffloadSlowdownPercent", "storageBytesPerSecond", "workingMemoryMode",
    "extendedWorkingMemory", "systemRamMode", "systemRamSharePercent", "deviceResources"})


def apply_runtime_load_config(config, raw, *, minimum_context_tokens=0):
    if raw is None:
        return config
    if not isinstance(raw, Mapping):
        raise ValueError("reviewed Load resource settings must be an object")
    changes = {}
    for key, target, minimum in (
        ("contextWindowTokens", "max_seq_len", max(8, minimum_context_tokens)),
        ("storagePoolBytes", "storage_pool_bytes", 0),
        ("memoryOffloadBytes", "memory_offload_bytes", 0),
        ("contextOffloadBudgetBytes", "working_attention_scratch_budget_bytes", 0),
        ("memoryResidentItems", "memory_resident_items", 1),
    ):
        if key in raw:
            value = raw[key]
            if type(value) is not int or not minimum <= value <= (1 << 53) - 1:
                raise ValueError("reviewed Load " + key + " is invalid or below retained context occupancy")
            changes[target] = value
    if "systemRamMode" in raw or "systemRamSharePercent" in raw:
        mode = raw.get("systemRamMode", "manual" if "systemRamSharePercent" in raw else "auto")
        value = raw.get("systemRamSharePercent", config.system_ram_share_percent) if mode == "manual" else 0.
        if mode not in {"auto", "manual"} or type(value) not in (int, float) or mode == "manual" and not 30 <= value <= 100:
            raise ValueError("reviewed Load RAM policy is invalid")
        changes["system_ram_share_percent"] = float(value)
    if "workingMemoryMode" in raw:
        if raw["workingMemoryMode"] not in {"auto", "extended", "manual"}:
            raise ValueError("reviewed Load working-memory policy is invalid")
        changes["working_memory_mode"] = raw["workingMemoryMode"]
    if "extendedWorkingMemory" in raw:
        if type(raw["extendedWorkingMemory"]) is not bool:
            raise ValueError("reviewed Load extended working memory must be boolean")
        changes["extended_working_memory"] = raw["extendedWorkingMemory"]
    if "storageBytesPerSecond" in raw:
        changes["storage_bytes_per_second"] = safe_rounded_storage_bytes_per_second(raw["storageBytesPerSecond"])
    if "deviceResources" in raw:
        from .device_resources import validate_device_resource_settings
        selected = validate_device_resource_settings(raw["deviceResources"])
        changes["device_resource_settings"] = selected
        ram = next((value for value in selected["allocations"] if value["deviceId"] == "ram:system"), None)
        if ram is not None:
            changes["system_ram_share_percent"] = float(ram["percent"]) if ram["mode"] == "manual" else 0.
    updated = replace(config, **changes)
    updated.validate()
    return updated
