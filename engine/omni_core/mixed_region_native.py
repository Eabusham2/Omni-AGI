"""Actual sequential decoder-region placement inside complete Auto operations.

Only an already captured read-only closure enters this adapter. Imports do not
construct a brain, access a device, probe hardware or start qualification.
"""
from __future__ import annotations

import json
from contextlib import ExitStack

from .mixed_region_schedule import MIXED_MODE, MixedRegionCandidatePool, checked_mixed_candidate
from .whole_path_execution import comparison_key, digest


def _host(ops): return getattr(ops, "host", ops)


class NativeMixedRegionPool:
    def __init__(self, ops, operation, review, current):
        from .model import DecoderBlock
        self.ops, self.operation, self.current = ops, operation, current
        self.stack, self.rows, self.by_object, self.closed = ExitStack(), [], {}, False
        try:
            # Actual closure traversal already belongs to its captured owner.
            # Retain only region metadata and references, never tensor copies.
            root = self.stack.enter_context(ops.brain.resource_policy.reserve_ram(
                65536 + len(review["candidates"]) * 8192, "mixed complete-operation candidate metadata"))
            root.mark_allocated(65536 + len(review["candidates"]) * 8192)
            for identifier, module in operation["closure"]["modules"].items():
                if type(module) is not DecoderBlock: continue
                path = operation["closure"]["names"][identifier]
                size = len(path.encode("utf-8")) * 8 + 8192
                lease = self.stack.enter_context(ops.brain.resource_policy.reserve_ram(size, "actual mixed decoder region registry"))
                row = {"regionId": path, "ownerObject": str(id(module)), "descriptor": ops.brain._regional_describe(module)}
                ordinal = len(self.rows); self.rows.append(row); self.by_object[id(module)] = (ordinal, module, row)
                lease.mark_allocated(size)
            if not self.rows: raise ValueError("captured operation has no actual decoder-block regions")
            self.registry_sha = digest(self.rows)
            self.pool = MixedRegionCandidatePool(review["currentCandidate"], review["candidates"], self.registry_sha, len(self.rows))
            # Metadata only, under a real reservation before construction.
            # Avoid re-enumerating every span for every complete receipt.
            bound = len(self.pool) * (8192 + len(json.dumps(self.pool.bases)) * 8)
            index_lease = self.stack.enter_context(ops.brain.resource_policy.reserve_ram(bound, "mixed schedule exact descriptor index"))
            self.by_sha = {row["planSha256"]: row for row in self.pool}
            if len(self.by_sha) != len(self.pool): raise ValueError("mixed pool descriptor identities collided")
            index_lease.mark_allocated(bound)
            self.identity = operation["identity"]
            self.host = _host(ops)
            registry = getattr(self.host, "_mixed_region_pools", None)
            if registry is None: registry = self.host._mixed_region_pools = {}
            self.key = comparison_key(self.identity)
            if self.key in registry: raise RuntimeError("another operation owns this mixed schedule authority")
            registry[self.key] = self
        except BaseException: self.stack.close(); raise
    @property
    def has_mixed(self): return self.pool.has_mixed
    def __len__(self): return len(self.pool)
    def __iter__(self):
        if self.closed: raise RuntimeError("mixed candidate pool retired")
        yield from self.by_sha.values()
    def resolve(self, plan_sha):
        if self.closed: raise RuntimeError("mixed candidate authority retired")
        value = self.by_sha.get(plan_sha)
        if value is None: raise ValueError("mixed selection was not issued by this actual operation")
        return value
    def require(self, candidate):
        self.current()
        actual = self.resolve(candidate["planSha256"])
        if actual != candidate: raise ValueError("mixed route differs from this operation's issued schedule")
        checked_mixed_candidate(candidate, bases=self.pool.bases, registry_sha=self.registry_sha, region_count=len(self.rows))
        return candidate
    def close(self):
        if self.closed: return
        self.closed = True
        registry = getattr(self.host, "_mixed_region_pools", {})
        if registry.get(self.key) is self: registry.pop(self.key)
        self.by_object.clear(); self.by_sha.clear(); self.rows.clear(); self.stack.close()


def prepare_mixed_operation_candidates(ops, operation, review, *, current):
    """Homogeneous fallback remains available when extra metadata cannot fit."""
    if len(review["candidates"]) < 2: return review["candidates"]
    pool = None
    try:
        pool = NativeMixedRegionPool(ops, operation, review, current)
        operation["mixedRegionPool"] = pool
        operation["stack"].callback(pool.close)
        return pool
    except Exception as error:
        if pool is not None: pool.close()
        operation["mixedScheduleUnavailable"] = type(error).__name__
        return review["candidates"]


def admit_mixed_candidate(ops, candidate):
    value = checked_mixed_candidate(candidate)
    schedule = value["regionSchedule"]
    # Base admissions read actual enabled device ceilings; no sums or rate
    # estimates manufacture a physical group or enlarge its allocation.
    outer = ops.admit_candidate(schedule["outerCandidate"])
    changed = []
    for row in schedule["segments"]:
        changed.append(ops.admit_candidate(row["candidate"]))
    return {**outer, "selection": "auto" if all(row["selection"] == "auto" for row in changed) else "manual"}


def admit_mixed_primitive(host, identity, primitive, reviewed):
    """The fixed-memory selector consumes live operation custody, not JSON trust."""
    pool = getattr(host, "_mixed_region_pools", {}).get(comparison_key(identity))
    if pool is None or pool.closed or pool.identity != identity: raise RuntimeError("mixed selection has no current operation owner")
    candidate = pool.resolve(primitive["planSha256"])
    if candidate.get("backendMode") != MIXED_MODE or {key: candidate[key] for key in primitive} != primitive:
        raise RuntimeError("mixed selection differs from its actual issued primitive")
    checked_mixed_candidate(candidate, bases=reviewed["candidates"], registry_sha=pool.registry_sha, region_count=len(pool.rows))
    # The caller fences native state before/after the complete source stream;
    # this per-row check adds exact issued custody and fresh device admissions
    # without hashing the entire brain again for every measurement row.
    pool.ops.admit_candidate(candidate)
    return True


class MixedDecoderBlockConsumer:
    def __init__(self, ops, candidate, operation, stopped):
        self.ops, self.candidate, self.stopped = ops, candidate, stopped
        self.pool = operation.get("mixedRegionPool")
        if self.pool is None or self.pool.ops is not ops: raise RuntimeError("mixed block route lacks its current captured registry")
        self.pool.require(candidate)
    def execute(self, target_consumer, module, method, hidden, args, kwargs):
        import torch
        from .regional_execution import canonical_device, regional_direct_scope
        from .native_compute_profile import native_projection_tensor_budget
        from .working_attention_paging import PagedTensorSequence
        if self.pool.closed or self.stopped.is_set(): raise InterruptedError("mixed decoder operation stopped")
        row = self.pool.by_object.get(id(module))
        if row is None or row[1] is not module: raise RuntimeError("mixed execution encountered a different block owner")
        ordinal, _module, declared = row
        descriptor = self.ops.brain._regional_describe(module)
        if descriptor != declared["descriptor"] or module.training or torch.is_grad_enabled():
            raise RuntimeError("mixed decoder block geometry/training owner changed")
        route = next((segment["candidate"] for segment in self.candidate["regionSchedule"]["segments"]
                      if segment["first"] <= ordinal < segment["first"] + segment["count"]), None)
        if route is None: return method(module, hidden, *args, **kwargs)
        if method.__name__ == "forward_cached":
            cache = kwargs.get("cache", args[0] if args else None)
            if cache is not None and (not isinstance(cache.key, PagedTensorSequence) or not isinstance(cache.value, PagedTensorSequence)):
                raise RuntimeError("mixed cached regions require the existing private paged KV frontier")
        target, home = route["executionDevice"], canonical_device(hidden.device)
        self.ops.admit_candidate(route)
        brain, pager = self.ops.brain, self.ops.brain.core_pager
        controls, hints = [], []
        size = 8192
        for child in module.modules():
            if child in pager._owners:
                hints.append((child, "_native_compute_device" in vars(child), getattr(child, "_native_compute_device", None)))
            else:
                if any(isinstance(value, torch.Tensor) for value in child._parameters.values()):
                    raise RuntimeError("mixed region has an unreviewed floating learned owner")
                buffers = dict(child._buffers); controls.append((child, buffers))
                size += sum(value.numel() * value.element_size() * 2 for value in buffers.values() if isinstance(value, torch.Tensor))
        with brain.resource_policy.reserve_ram(size, "mixed block controls and transfer lifetime"), regional_direct_scope(target_consumer, target):
            try:
                for child, _present, _value in hints: child._native_compute_device = torch.device(target)
                for child, buffers in controls:
                    for name, value in buffers.items():
                        if isinstance(value, torch.Tensor) and name != "_next_page": child._buffers[name] = self.ops.move_activity(value, target)
                width, ff = descriptor["dimensions"], descriptor["feedForward"]
                count = max(1, hidden.numel() // width)
                needed = max(native_projection_tensor_budget(count, a, b)["forwardTensorReserveBytes"]
                             for a, b in ((width, width * 3), (width, ff * 2), (ff, width)))
                brain.working_attention_pager.admit_compute(needed + hidden.numel() * hidden.element_size() * 4 + 16,
                    "mixed decoder region complete transient compute", device=torch.device(target))
                moved = self.ops.move_activity(hidden, target)
                result = method(module, moved, *args, **kwargs)
                if self.stopped.is_set(): raise InterruptedError("mixed decoder block stopped after computation")
                value = result[0] if isinstance(result, tuple) else result
                value = self.ops.move_activity(value, home)
                return (value, *result[1:]) if isinstance(result, tuple) else value
            finally:
                for child, buffers in controls:
                    child._buffers.clear(); child._buffers.update(buffers)
                for child, present, value in hints:
                    if present: child._native_compute_device = value
                    else: child.__dict__.pop("_native_compute_device", None)
                # The containing complete operation restores actual packed
                # residency/caches after this body and proves its exact state.
