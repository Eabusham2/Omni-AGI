"""Non-neural retained-context storage eligibility and physical-pool accounting.

Capacity observations authorize a bounded storage interface, not native packed
arithmetic or a speed claim. Live allocations still require the owner's fresh
DeviceResourceManager admission. Unknown copy rates sort conservatively last.
"""
from __future__ import annotations

import math
import re


def _count(value):
    return type(value) is int and value >= 0


def _rate(value):
    number = value.get("bytesPerSecond") if isinstance(value, dict) else None
    return float(number) if type(number) in (int, float) and math.isfinite(number) and number > 0 else None


def retained_context_pools(topology, budgets, *, manual=False, minimums=None, page_bytes=1):
    """Distinct enabled GPU activity partitions; RAM/unified aliases add zero.

    The existing core/activity/transfer split is 55/30/15. Applying the same
    activity share to each distinct GPU ceiling prevents double credit against
    its packed owners and transfer reservation. Free bytes gate *allocations*,
    never shrink a selected context's configured capacity.
    """
    devices = topology.get("devices", [])
    ram_groups = {item.get("memoryGroupId") for item in devices if item.get("kind") == "ram"} | {"ram:system"}
    selected = {item.get("deviceId"): item for item in budgets}
    links = {(item.get("from"), item.get("to")): _rate(item.get("measurement"))
             for item in topology.get("links", [])}
    groups = {}
    for device in devices:
        budget = selected.get(device.get("id"), {})
        backend, execution = device.get("backend"), device.get("executionDevice")
        group = device.get("memoryGroupId")
        supported = ((backend == "cuda" and isinstance(execution, str) and re.fullmatch(r"cuda:[0-9]+", execution))
                     or (backend == "mps" and execution == "mps"))
        if (not supported or device.get("kind") != "gpu" or not isinstance(group, str)
                or group in ram_groups or device.get("unifiedMemory") is not False
                or budget.get("enabled") is not True or budget.get("memoryGroupId") != group
                or not _count(device.get("totalBytes")) or not device["totalBytes"]
                or not _count(device.get("availableBytes")) or not _count(budget.get("ceilingBytes"))):
            continue
        ceiling = min(device["totalBytes"], budget["ceilingBytes"])
        local, upload, download = (_rate(device.get("measurement")), links.get(("ram:system", device["id"])),
                                   links.get((device["id"], "ram:system")))
        rate = 1 / (1 / local + 1 / upload + 1 / download) if local and upload and download else None
        pool = {"deviceId": device["id"], "executionDevice": execution, "memoryGroupId": group,
                "budgetBytes": ceiling * 30 // 100, "selectedCeilingBytes": ceiling,
                "effectiveCopyBytesPerSecond": rate, "copyTimingMeasured": rate is not None}
        if manual:
            minimum = (minimums or {}).get(execution)
            if isinstance(minimum, dict) and set(minimum) == {"packedOwnerBytes", "controlBytes", "transferBytes"} and all(
                    _count(value) for value in minimum.values()):
                pool.update(budgetBytes=max(0, ceiling - sum(minimum.values())), partitionMode="manual-admitted-minimums",
                    minimumPackedOwnerBytes=minimum["packedOwnerBytes"], controlReserveBytes=minimum["controlBytes"],
                    transferReserveBytes=minimum["transferBytes"])
            else:
                pool["partitionMode"] = "manual-conservative-missing-minimum"
        if type(page_bytes) is not int or page_bytes < 1:
            raise ValueError("retained context requires an exact positive page granularity")
        pool["budgetBytes"] = pool["budgetBytes"] // page_bytes * page_bytes
        if page_bytes != 1:
            pool["pageBytes"] = page_bytes
        previous = groups.get(group)
        if previous is None or (pool["budgetBytes"], rate or 0, execution) > (
                previous["budgetBytes"], previous["effectiveCopyBytesPerSecond"] or 0, previous["executionDevice"]):
            groups[group] = pool
    return sorted(groups.values(), key=lambda item: (-(item["effectiveCopyBytesPerSecond"] or 0), item["deviceId"]))


def context_snapshot_bytes(tokens, *, kv_bytes_per_token, index_bytes_per_token, page_tokens, layers):
    """Full paused K/V + token/index allowance + bounded safe-tensor headers.

    ExecutionStateWriter reserves 64 KiB per tensor page. This is a space
    promise, not a claim that each final header occupies those physical bytes.
    """
    return (tokens * (kv_bytes_per_token + index_bytes_per_token)
            + ((tokens + page_tokens - 1) // page_tokens) * 2 * layers * 65536)


def context_capacity(*, ram_working_bytes, device_context_bytes, kv_bytes_per_token,
                     index_bytes_per_token, model_limit_tokens, spill_capacity_bytes,
                     paged_kv_bytes_per_token, checkpoint_capacity_bytes, page_tokens, layers,
                     shared_storage_capacity_bytes=None):
    """Conservative pager-backed limits with a real host metadata floor."""
    metadata = ram_working_bytes // max(1, index_bytes_per_token)
    resident = min(model_limit_tokens, metadata,
                   (ram_working_bytes + device_context_bytes) // max(1, kv_bytes_per_token + index_bytes_per_token))
    shared = checkpoint_capacity_bytes if shared_storage_capacity_bytes is None else shared_storage_capacity_bytes
    low, high = 0, min(model_limit_tokens, metadata,
        checkpoint_capacity_bytes // max(1, kv_bytes_per_token + index_bytes_per_token))
    while low < high:
        middle = (low + high + 1) // 2
        snapshot = context_snapshot_bytes(middle, kv_bytes_per_token=kv_bytes_per_token,
                                          index_bytes_per_token=index_bytes_per_token, page_tokens=page_tokens, layers=layers)
        cold = context_cold_spill_bytes(middle, ram_working_bytes=ram_working_bytes, device_context_bytes=device_context_bytes,
            kv_bytes_per_token=kv_bytes_per_token, index_bytes_per_token=index_bytes_per_token,
            paged_kv_bytes_per_token=paged_kv_bytes_per_token, page_tokens=page_tokens)
        if snapshot <= checkpoint_capacity_bytes and cold <= spill_capacity_bytes and snapshot + cold <= shared:
            low = middle
        else:
            high = middle - 1
    return {"residentMaximumTokens": min(resident, low), "storageMaximumTokens": low if spill_capacity_bytes else 0,
            "maximumTokens": low, "metadataMaximumTokens": metadata}


def context_cold_spill_bytes(tokens, *, ram_working_bytes, device_context_bytes, kv_bytes_per_token,
                            index_bytes_per_token, paged_kv_bytes_per_token, page_tokens):
    """Only cold K/V, rounded conservatively to one page per attention owner.

    Metadata is charged to host first. The final group covers up to one tail
    page per K/V layer even when hot/cold boundaries differ between layers.
    """
    hot = max(0, ram_working_bytes - tokens * index_bytes_per_token) + device_context_bytes
    cold = max(0, tokens * kv_bytes_per_token - hot)
    groups = (cold + kv_bytes_per_token * page_tokens - 1) // max(1, kv_bytes_per_token * page_tokens)
    return groups * page_tokens * paged_kv_bytes_per_token
