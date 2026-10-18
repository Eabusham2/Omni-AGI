"""Current-owner adapter for fixed memory effects and aggregate selection."""
from __future__ import annotations

import copy
import hashlib
import json
import weakref

from .whole_path_execution import digest


def _tensor_frontier(value):
    if value is None: return None
    try:
        return [str(id(value)), int(value._version), str(value.device), str(value.dtype), list(value.shape)]
    except (AttributeError, RuntimeError) as error:
        raise ValueError("actual memory activity has no provable version frontier") from error


class NativeFixedMemoryOps:
    def __init__(self, brain, host):
        self._owner, self.host = weakref.ref(brain), host

    @property
    def brain(self):
        brain = self._owner()
        if brain is None: raise RuntimeError("aggregate memory owner was released")
        return brain

    def _controller(self):
        value = getattr(self.brain, "_native_operation_auto", None)
        if value is None or value.ops is not self.host: raise RuntimeError("aggregate owner controller changed")
        return value

    def armed(self):
        try:
            controller = self._controller()
            with controller._lock:
                return (not controller._closed and not controller._cancel.is_set()
                    and controller._phase in {"awaiting-operation", "qualifying", "active"})
        except RuntimeError: return False

    def _regions(self):
        root = self.brain.decoder
        root_regions = getattr(root, "regions", ())
        if not root_regions: return
        # Depth-first iterators avoid an all-regions pending list. Distinct
        # calls borrowing the same decoder are allowed; cycles in the current
        # ancestry are not. Metadata admission grows with actual depth.
        pending, seen, leases, admitted = [(id(root), iter(root_regions))], {id(root)}, [], 0
        try:
            while pending:
                if len(pending)*512 > admitted:
                    size = max(16384, admitted)
                    leases.append(self.reserve_metadata(size)); admitted += size
                key, iterator = pending[-1]
                try: region = next(iterator)
                except StopIteration:
                    pending.pop(); seen.remove(key); continue
                yield region
                child = region.decoder
                if id(child) in seen: raise ValueError("cyclic actual memory owner")
                seen.add(id(child)); pending.append((id(child), iter(getattr(child, "regions", ()))))
        finally:
            for lease in leases: lease.release()

    def owns(self, value):
        if value is self.brain or value is self.brain.router: return True
        return any(value is region or value is region.router for region in self._regions())

    def binding(self):
        brain, controller = self.brain, self._controller()
        with controller._lock:
            if controller._review is None: raise RuntimeError("aggregate path requires the current explicit review")
            reviewed = copy.deepcopy(controller._review["binding"])
        return {"ownerSessionId": controller.owner_session, "ownerObject": str(id(brain)),
            "policyObject": str(id(brain.resource_policy)),
            "policySha256": digest({"resource": brain._regional_policy_sha(), "configuration": brain.config.to_dict()}),
            "budgetSha256": digest(self.host._budgets()),
            "reviewedPolicySha256": reviewed["policySha256"], "reviewedBudgetSha256": reviewed["budgetSha256"],
            "topologySha256": reviewed["topologySha256"], "registrySha256": reviewed["registrySha256"],
            "homeDevice": str(brain.device), "contextCapacity": int(brain.config.max_seq_len),
            "workingCapacity": int(brain.config.working_memory_slots),
            "allowStorageSpill": bool(brain.resource_policy.device_resources.settings["allowStorageSpill"])}

    def frontier(self):
        brain = self.brain
        def memory(value):
            return [str(id(value)), int(getattr(value, "state_revision", 0)),
                *[str(id(getattr(value, field, None))) for field in ("neurons", "synapses", "assemblies", "neuron_vectors", "assembly_vectors")]]
        regions, count = hashlib.sha256(), 0
        for region in self._regions():
            raw = json.dumps([str(id(region)), region.identity.region_id, str(id(region._recall)), str(id(region._learn_memory)),
                _tensor_frontier(region.liquid_state), _tensor_frontier(region.router.population.membrane),
                _tensor_frontier(region.router.synapses.plasticity_events)], separators=(",", ":")).encode()
            regions.update(len(raw).to_bytes(8, "little")); regions.update(raw); count += 1
        # These are current runtime ownership/cursor fences, not a checksum
        # claiming to contain all learned facts or a cloned graph image.
        return {"memory": memory(brain.memory), "regions": {"count": count, "ownerFrontierSha256": regions.hexdigest()},
            "working": [str(id(brain.working_memory)), len(brain.working_memory), str(id(brain.workspace_items)), len(brain.workspace_items)],
            "afterimages": [int(brain.memory_lifecycle.cycle), str(id(brain.memory_lifecycle.afterimage_vectors)),
                len(brain.memory_lifecycle.afterimage_vectors), str(id(brain.memory_lifecycle.active_focus))],
            "recentTokens": [str(id(brain.recent_token_context)), len(brain.recent_token_context)],
            "liquid": _tensor_frontier(brain.liquid_state), "membrane": _tensor_frontier(brain.router.population.membrane),
            "plasticity": _tensor_frontier(brain.router.synapses.plasticity_events),
            "experienceCount": int(brain.counters.get("experiences", 0)),
            "replayOwner": str(id(brain.replay)), "pagedWorkingOwner": str(id(brain.paged_working_memory))}

    def reserve_metadata(self, size):
        return self.brain.resource_policy.reserve_ram(size, "actual fixed-memory aggregate placement metadata")

    def admit_effect(self, stage):
        # The actual memory allocation/paging gates remain inside the body.
        # No derived-plan estimate expands this selected family RAM ceiling.
        self.brain.resource_policy.check_active_pressure("actual " + stage,
            reclaim=self.brain._maintain_neural_state_resources)

    def _synchronize_home(self):
        device = str(self.brain.device)
        if device != "cpu": self.host.synchronize([device])

    def complete_effect(self, _stage): self._synchronize_home()
    def drain_prior_effect(self, _stage): self._synchronize_home()

    def resource_observation(self):
        controller = self._controller()
        with controller._lock:
            reviewed = copy.deepcopy(controller._review)
        return self.host.observation_resources(reviewed["currentCandidate"], reviewed["binding"]["budgetSha256"])

    def admit_selection(self, identity, candidate):
        controller, brain = self._controller(), self.brain
        with controller._lock: reviewed = copy.deepcopy(controller._review)
        if not self.armed() or identity["ownerId"] != brain.brain_id or identity["ownerSessionId"] != controller.owner_session:
            raise RuntimeError("aggregate selection belongs to another or stopped actual owner")
        binding = self.binding()
        if (binding["policySha256"] != identity["policySha256"] or binding["budgetSha256"] != identity["budgetSha256"]
                or binding["topologySha256"] != identity["topologySha256"]
                or binding["reviewedPolicySha256"] != identity["policySha256"]
                or binding["reviewedBudgetSha256"] != identity["budgetSha256"]):
            raise RuntimeError("aggregate placement policy or selected budget changed")
        if len(brain.recent_token_context) > brain.config.max_seq_len:
            raise RuntimeError("unreconciled current context exceeds selected capacity")
        exact = next((row for row in reviewed["candidates"]
            if controller.primitive(row) == candidate), None)
        if exact is None:
            from .mixed_region_native import admit_mixed_primitive
            if candidate.get("backendMode") != "native-mixed-sequential":
                raise RuntimeError("aggregate compute route was absent from the actual review")
            admit_mixed_primitive(self.host, identity, candidate, reviewed)
        else: self.host.admit_candidate(exact)
        brain.resource_policy.check_active_pressure("actual memory plus complete compute selection")

    def record(self, payload):
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.brain.resource_policy.require_disk(len(raw)*2+65536, "owned aggregate memory/compute receipt")
        lease = self.reserve_metadata(len(raw)*4)
        try:
            self.brain.events.append("native-whole-path", payload, job_id="owned-path-" + payload["requestSha256"][:32])
        finally: lease.release()
