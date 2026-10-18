"""Manual, process-local decoder-region dispatch; no model/hardware on import.

Preferences are portable hardware metadata, never an armed lease. The caller
owns the current brain, exact module registry, topology and resource callbacks.
No weights, rewards, source text, or alternate neural model live in this helper.
"""
from __future__ import annotations

import copy
import contextvars
import functools
import hashlib
import json
import math
import re
import threading
import time
import uuid
import weakref
from collections import deque
from contextlib import contextmanager


_BLOCK_CONSUMERS = weakref.WeakKeyDictionary()
_BIND_LOCK = threading.RLock()
_ACTIVE_REGIONAL = contextvars.ContextVar("omni_manual_regional_execution", default=None)
_TRANSIENT_REGIONAL = contextvars.ContextVar("omni_measured_cached_region_operation", default=None)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,255}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def canonical_device(value):
    value = str(value)
    return "mps" if value == "mps:0" else value


def current_regional_target(policy):
    value = _ACTIVE_REGIONAL.get()
    if value is None:
        return None
    consumer, epoch, target = value
    if not consumer.active or consumer._lease["epoch"] != epoch or consumer.policy_identity() is not policy:
        raise RuntimeError("regional device scope is stale for this resource owner")
    return target


@contextmanager
def transient_regional_scope(consumer):
    """Trusted same-owner operation only; never a saved/manual route lease."""
    token = _TRANSIENT_REGIONAL.set(consumer)
    try:
        yield
    finally:
        _TRANSIENT_REGIONAL.reset(token)


def transient_regional_limits(policy):
    consumer = _TRANSIENT_REGIONAL.get()
    return None if consumer is None else consumer.packed_limits(policy)


@contextmanager
def regional_direct_scope(consumer, target):
    """Nested actual region target; preserve its complete-operation owner."""
    if not consumer.active or not isinstance(consumer._lease.get("epoch"), str):
        raise RuntimeError("regional target belongs to a retired operation")
    token = _ACTIVE_REGIONAL.set((consumer, consumer._lease["epoch"], canonical_device(target)))
    try:
        yield
    finally:
        _ACTIVE_REGIONAL.reset(token)


def validate_regional_execution_settings(value):
    if (not isinstance(value, dict) or set(value) != {"version", "enabled", "mode", "routes"}
            or type(value["version"]) is not int or value["version"] != 1
            or type(value["enabled"]) is not bool or value["mode"] != "manual"
            or not isinstance(value["routes"], list)):
        raise ValueError("regional execution requires exact reviewed manual settings")
    seen = set()
    for route in value["routes"]:
        if (not isinstance(route, dict) or set(route) - {"regionId", "deviceId", "executionMode", "acknowledgePackedCpuFallback"}
                or not {"regionId", "deviceId", "executionMode"}.issubset(route)
                or not isinstance(route["regionId"], str) or not route["regionId"] or "\0" in route["regionId"]
                or route["regionId"] in seen or not isinstance(route["deviceId"], str) or not _ID.fullmatch(route["deviceId"])
                or route["executionMode"] not in {"native", "packed-cpu-tile-fallback"}):
            raise ValueError("regional route identity/mode is invalid or duplicated")
        seen.add(route["regionId"])
        if route["executionMode"] == "packed-cpu-tile-fallback":
            if route.get("acknowledgePackedCpuFallback") is not True:
                raise ValueError("regional CUDA fallback requires its explicit acknowledgment")
        elif "acknowledgePackedCpuFallback" in route:
            raise ValueError("native regional routes cannot carry a fallback acknowledgment")
    return copy.deepcopy(value)


def topology_identity(topology):
    """Stable interface/route identity; changing free bytes are admitted live."""
    if not isinstance(topology, dict) or topology.get("version") != 1 or not isinstance(topology.get("devices"), list):
        raise ValueError("regional execution needs an observed device topology")
    devices = []
    for item in topology["devices"]:
        if not isinstance(item, dict):
            raise ValueError("regional topology device must be an object")
        if item.get("kind") == "storage":
            continue
        if (not isinstance(item.get("id"), str) or not _ID.fullmatch(item["id"])
                or not isinstance(item.get("executionDevice"), str)
                or not isinstance(item.get("memoryGroupId"), str)
                or type(item.get("nativeComputeSupported")) is not bool):
            raise ValueError("regional topology interface is incomplete")
        devices.append({key: item.get(key) for key in (
            "id", "kind", "backend", "memoryGroupId", "totalBytes", "recommendedLimitBytes",
            "unifiedMemory", "nativeComputeSupported", "unavailableReason")}
            | {"executionDevice": canonical_device(item["executionDevice"])})
    ids = {item["id"] for item in devices}
    if len(ids) != len(devices):
        raise ValueError("regional topology contains duplicate interfaces")
    if not isinstance(topology.get("links"), list) or any(not isinstance(item, dict) for item in topology["links"]):
        raise ValueError("regional topology links must be observed objects")
    links = [{key: item.get(key) for key in ("from", "to", "peerAccess", "unavailableReason")}
             for item in topology["links"] if item.get("from") in ids and item.get("to") in ids]
    if len({(item["from"], item["to"]) for item in links}) != len(links):
        raise ValueError("regional topology contains duplicate transfer links")
    return {"devices": sorted(devices, key=lambda item: item["id"]),
            "links": sorted(links, key=lambda item: (item["from"], item["to"]))}


def regional_block_call(method):
    """Actual DecoderBlock entry seam; absent leases keep the original call."""
    @functools.wraps(method)
    def wrapped(module, hidden, *args, **kwargs):
        transient = _TRANSIENT_REGIONAL.get()
        if transient is not None:
            return transient.execute(module, method, hidden, args, kwargs)
        reference = _BLOCK_CONSUMERS.get(module)
        consumer = None if reference is None else reference()
        if consumer is None or not consumer.active:
            return method(module, hidden, *args, **kwargs)
        return consumer.execute(module, method, hidden, args, kwargs)
    return wrapped


def bind_regional_blocks(consumer, regions):
    """Bind real registered objects; deepcopy/imported modules acquire no lease."""
    consumer.register(regions)
    with _BIND_LOCK:
        for reference, _descriptor in consumer.regions.values():
            module = reference()
            previous = _BLOCK_CONSUMERS.get(module)
            if previous is not None and previous() not in (None, consumer):
                raise ValueError("decoder block is already owned by a different regional consumer")
            _BLOCK_CONSUMERS[module] = weakref.ref(consumer)


class RegionalExecutionConsumer:
    """Serialized real call/transfer consumer with injected, owner-bound effects.

    Production callbacks perform actual Torch transfers and fresh admissions.
    Tests can use inert tensors/blocks without constructing any neural model.
    """
    def __init__(self, *, owner_id, settings, policy_identity, policy_sha, topology,
                 quiescent, describe, tensor_bytes, tensor_device, transfer,
                 reserve_transfer, admit_compute, training, cancelled=lambda: False, validate_route=lambda _route: None):
        self.owner_id = owner_id
        self.settings = settings
        self.policy_identity, self.policy_sha, self.topology = policy_identity, policy_sha, topology
        self.quiescent, self.describe = quiescent, describe
        self.tensor_bytes, self.tensor_device, self.transfer = tensor_bytes, tensor_device, transfer
        self.reserve_transfer, self.admit_compute, self.training = reserve_transfer, admit_compute, training
        self.cancelled = cancelled
        self.validate_route = validate_route
        self.regions = {}
        self._by_object = {}
        self._lease = None
        self._lock = threading.RLock()
        self._active_calls = 0
        self._closed = False
        self._reason = "manual-review-required-after-load"
        self._dispatches = self._transfers = self._transfer_bytes = 0
        self._training_skips = 0
        self._estimated_transfer_seconds = 0.0
        self._unknown_transfer_costs = 0
        self._trace = deque(maxlen=64)

    @property
    def active(self):
        return self._lease is not None and not self._closed

    def register(self, regions):
        replacement, by_object = {}, {}
        for path, module in regions:
            if path in replacement or id(module) in by_object:
                raise ValueError("regional registry contains duplicate paths/objects")
            descriptor = {"regionId": path, **self.describe(module), "runtimeOwnerId": str(id(module))}
            replacement[path] = (weakref.ref(module), descriptor)
            by_object[id(module)] = path
        signature = canonical_sha([row for _ref, row in sorted(replacement.values(), key=lambda pair: pair[1]["regionId"])])
        with self._lock:
            if self.active and signature != self.registry_sha:
                # Retire only at a safe owner boundary, never clear live context.
                self.retire("registered-region-geometry-changed")
            self.regions, self._by_object, self.registry_sha = replacement, by_object, signature

    def prepare(self, *, start=0, count=64, topology=None):
        if type(start) is not int or start < 0 or type(count) is not int or not 1 <= count <= 256:
            raise ValueError("regional descriptor page requires start>=0 and count in1..256")
        current = self.topology()
        if topology is not None and topology_identity(topology) != topology_identity(current):
            raise ValueError("main regional topology observation is stale")
        topology = current if topology is None else topology
        keys = sorted(self.regions)
        return {"format": "omni-manual-regional-execution-review", "version": 1,
            "ownerId": self.owner_id, "policySha256": self.policy_sha(),
            "topologyIdentitySha256": canonical_sha(topology_identity(topology)),
            "registrySha256": self.registry_sha, "topology": copy.deepcopy(topology),
            "settings": copy.deepcopy(self.settings()), "start": start, "total": len(keys),
            "regions": [{key: value for key, value in self.regions[path][1].items() if key != "runtimeOwnerId"}
                        for path in keys[start:start + count]], "status": self.status()}

    def activate(self, reviewed):
        if (not isinstance(reviewed, dict) or set(reviewed) != {"policySha256", "topologyIdentitySha256", "registrySha256", "topology"}
                or any(not isinstance(reviewed.get(key), str) or not _SHA.fullmatch(reviewed[key])
                       for key in ("policySha256", "topologyIdentitySha256", "registrySha256"))):
            raise ValueError("regional activation requires the exact main-reviewed source identities")
        with self._lock:
            self.quiescent()
            if self._closed or self._active_calls:
                raise RuntimeError("regional activation awaits the current owner boundary")
            settings = validate_regional_execution_settings(self.settings())
            if not settings["enabled"] or not settings["routes"]:
                raise ValueError("enable reviewed manual regional preferences first")
            current_identity = topology_identity(self.topology())
            identity_sha = canonical_sha(current_identity)
            if (reviewed["policySha256"] != self.policy_sha() or reviewed["registrySha256"] != self.registry_sha
                    or reviewed["topologyIdentitySha256"] != identity_sha
                    or canonical_sha(topology_identity(reviewed["topology"])) != identity_sha):
                raise ValueError("regional review is stale for the current owner/topology/budget policy")
            devices = {item["id"]: item for item in current_identity["devices"]}
            routes = {}
            for route in settings["routes"]:
                if route["regionId"] not in self.regions:
                    raise ValueError("regional route is not an exact current registered decoder block")
                device = devices.get(route["deviceId"])
                if device is None or device["backend"] not in {"cpu", "mps", "cuda"}:
                    raise ValueError("regional backend is unavailable; XPU/NPU are not implemented")
                if route["executionMode"] == "native":
                    if device["backend"] not in {"cpu", "mps"} or not device["nativeComputeSupported"]:
                        raise ValueError("regional direct packed backend is unsupported")
                elif device["backend"] != "cuda" or route.get("acknowledgePackedCpuFallback") is not True:
                    raise ValueError("regional CPU-tile fallback is explicit CUDA-only authority")
                routes[route["regionId"]] = {**route, "executionDevice": device["executionDevice"],
                    "memoryGroupId": device["memoryGroupId"], "nativeComputeSupported": device["nativeComputeSupported"]}
                self.validate_route(routes[route["regionId"]])
            self._lease = {"epoch": uuid.uuid4().hex, "policyObject": self.policy_identity(),
                "policySha256": reviewed["policySha256"], "topologyIdentitySha256": identity_sha,
                "registrySha256": self.registry_sha, "routes": routes,
                "topology": copy.deepcopy(reviewed["topology"])}
            self._reason = "active-manual-decoder-regions"
            return self.status()

    def retire(self, reason="explicitly-deactivated", *, expected_epoch=None, closing=False):
        with self._lock:
            if expected_epoch is not None and (not self.active or expected_epoch != self._lease["epoch"]):
                raise ValueError("regional deactivation epoch is stale")
            if self.active and not closing:
                self.quiescent()
            if self._active_calls:
                raise RuntimeError("regional execution still has an active call")
            self._lease = None
            self._reason = reason
            return self.status()

    def _validate_live(self, module, path):
        lease = self._lease
        if lease is None or self._closed:
            raise RuntimeError("regional execution lease is inactive")
        if lease.get("invalidated"):
            raise RuntimeError("regional execution lease was invalidated; review current devices/settings")
        if (lease["policyObject"] is not self.policy_identity() or lease["policySha256"] != self.policy_sha()
                or lease["registrySha256"] != self.registry_sha
                or lease["topologyIdentitySha256"] != canonical_sha(topology_identity(self.topology()))):
            lease["invalidated"] = True
            self._reason = "lease-invalidated-owner-topology-or-budget-policy-changed"
            raise RuntimeError("regional execution lease is stale; review current devices/settings")
        reference, before = self.regions[path]
        if reference() is not module or {**self.describe(module), "regionId": path, "runtimeOwnerId": str(id(module))} != before:
            lease["invalidated"] = True
            self._reason = "lease-invalidated-owner-or-geometry-changed"
            raise RuntimeError("regional execution owner/geometry changed; review again")
        if self.cancelled():
            raise InterruptedError("regional execution cancelled before dispatch/transfer")
        return lease

    def _move(self, value, destination, label):
        before = canonical_device(self.tensor_device(value))
        destination = canonical_device(destination)
        if before == destination:
            return value
        size = self.tensor_bytes(value)
        with self.reserve_transfer(destination, size, label) as reservation:
            if self.cancelled():
                raise InterruptedError("regional transfer cancelled before allocation")
            result = self.transfer(value, destination)
            if canonical_device(self.tensor_device(result)) != destination or self.tensor_bytes(result) != size:
                raise RuntimeError("regional transfer did not preserve the exact tensor geometry/bytes")
            if reservation is not None:
                reservation.mark_allocated(size)
            self._transfers += 1
            self._transfer_bytes += size
            # The real path returns to its caller; do not substitute a direct
            # peer-GPU rate for these actual two-direction transfers.
            topology = self._lease["topology"]
            identifiers = {canonical_device(item.get("executionDevice", "")): item["id"]
                           for item in topology["devices"] if "executionDevice" in item}
            link = next((item for item in topology.get("links", [])
                         if item.get("from") == identifiers.get(before) and item.get("to") == identifiers.get(destination)), None)
            rate = (link or {}).get("measurement", {}).get("bytesPerSecond")
            if type(rate) in (int, float) and math.isfinite(rate) and rate > 0:
                self._estimated_transfer_seconds += size / rate
            else:
                self._unknown_transfer_costs += 1
            return result

    def execute(self, module, method, hidden, args, kwargs):
        with self._lock:
            path = self._by_object.get(id(module))
            if path is None or not self.active:
                return method(module, hidden, *args, **kwargs)
            if self.training(module):
                self._training_skips += 1
                self._reason = "regional-training-inapplicable-original-single-device-path"
                return method(module, hidden, *args, **kwargs)
            lease = self._validate_live(module, path)
            route = lease["routes"].get(path)
            if route is None:
                return method(module, hidden, *args, **kwargs)
            target, home = route["executionDevice"], canonical_device(self.tensor_device(hidden))
            start, transferred = time.monotonic(), self._transfer_bytes
            estimated_before, unknown_before = self._estimated_transfer_seconds, self._unknown_transfer_costs
            self._active_calls += 1
            completed = False
            scope = _ACTIVE_REGIONAL.set((self, lease["epoch"], target))
            try:
                self.admit_compute(module, hidden, target, route)
                regional_hidden = self._move(hidden, target, "regional decoder input transfer")
                self._validate_live(module, path)
                result = method(module, regional_hidden, *args, **kwargs)
                output = result[0] if isinstance(result, tuple) else result
                self._validate_live(module, path)
                output = self._move(output, home, "regional decoder output transfer")
                self._dispatches += 1
                self._reason = "active-manual-decoder-regions"
                completed = True
                return (output, *result[1:]) if isinstance(result, tuple) else output
            finally:
                _ACTIVE_REGIONAL.reset(scope)
                self._active_calls -= 1
                self._trace.append({"regionId": path, "executionDevice": target, "callerDevice": home,
                    "executionMode": route["executionMode"], "completed": completed,
                    "actualTransferBytes": self._transfer_bytes - transferred,
                    "estimatedTransferSeconds": (self._estimated_transfer_seconds - estimated_before)
                        if self._unknown_transfer_costs == unknown_before else None,
                    "unknownTransferCosts": self._unknown_transfer_costs - unknown_before,
                    "elapsedSeconds": max(0.0, time.monotonic() - start)})

    def devices(self):
        return {} if not self.active else {item["executionDevice"]: item["memoryGroupId"] for item in self._lease["routes"].values()}

    def status(self):
        return {"format": "omni-manual-regional-execution-status", "version": 1, "ownerId": self.owner_id,
            "active": self.active, "epoch": self._lease["epoch"] if self.active else None, "reason": self._reason,
            "leaseValid": self.active and not self._lease.get("invalidated", False),
            "registeredRegions": len(self.regions), "activeCalls": self._active_calls,
            "routes": list(self._lease["routes"].values()) if self.active else [],
            "completedRegionDispatches": self._dispatches, "actualActivationTransfers": self._transfers,
            "actualActivationTransferBytes": self._transfer_bytes, "trainingInapplicableCalls": self._training_skips,
            "estimatedKnownTransferSeconds": self._estimated_transfer_seconds,
            "unknownTransferCosts": self._unknown_transfer_costs,
            "trainingSupported": False, "automaticDefault": False, "contextGpuCapacityCredited": False,
            "wholePathPerformanceMeasured": False, "trace": list(self._trace)}

    def close(self):
        self.retire("owner-closed", closing=True)
        self._closed = True
        with _BIND_LOCK:
            for reference, _descriptor in self.regions.values():
                module = reference()
                current = _BLOCK_CONSUMERS.get(module) if module is not None else None
                if current is not None and current() is self:
                    _BLOCK_CONSUMERS.pop(module, None)


def regional_group_admission(usage, limits, target, required):
    """Pure device+physical-group hot-cache bounds consumed by the real pager."""
    if target not in limits or type(required) is not int or required < 0:
        raise ValueError("regional packed admission requires a current target/byte count")
    group = limits[target]["memoryGroupId"]
    per_device = max(0, int(limits[target]["hotBytes"]))
    group_limit = max((max(0, int(item["hotBytes"])) for item in limits.values() if item["memoryGroupId"] == group), default=0)
    group_used = sum(max(0, int(count)) for name, count in usage.items()
                     if name in limits and limits[name]["memoryGroupId"] == group)
    return max(0, int(usage.get(target, 0))) + required <= per_device and group_used + required <= group_limit
