"""Exact native capability transaction with RAM-first packed rollback.

Only compact native bytes spill. Control tensors, counter key sets and mixed
training modes remain explicit; no floating learned model is constructed.
"""
from pathlib import Path

import torch

from .optimizers import PackedMutationSnapshot
from .persistence import tensor_checksum


class CapabilityRollbackSnapshot:
    def __init__(self, brain, heads, *, label="capability retention transaction"):
        self.brain, self.heads, self.label = brain, dict(heads), label
        self.packed = None
        self.leases, self.disk_leases = [], []
        self.closed = False
        self.reserved_packed_bytes = 0
        self.reserve = getattr(getattr(brain, "resource_policy", None), "reserve_ram", None)
        self.modes = [(module, module.training) for module in brain.decoder.modules()]
        modules = {id(module): module for head in self.heads.values() for module in head.modules()}
        self.events = [(module, int(module._pending_stability_events)) for module in modules.values()
                       if hasattr(module, "_pending_stability_events")]
        packed_ids = set()
        for module in modules.values():
            getter = getattr(module, "authoritative_packed_tensors", None)
            if callable(getter):
                packed_ids.update(id(value) for value in getter())
                packed_ids.update(id(value) for name in ("_row_stability", "_bias_row_stability")
                                  if isinstance((value := getattr(module, name, None)), torch.Tensor))
            packed_ids.update(id(value) for name, value in module.named_buffers(recurse=False)
                              if value.dtype == torch.uint8 and name not in getattr(module, "_non_persistent_buffers_set", ()))
        controls, seen = [], set()
        for module in modules.values():
            for name, value in module.named_buffers(recurse=False):
                if name in getattr(module, "_non_persistent_buffers_set", ()) or id(value) in packed_ids or id(value) in seen:
                    continue
                seen.add(id(value))
                controls.append((module, name, value))
        owner_ids = {id(value) for head in self.heads.values() for value in head.parameters()}
        named = brain._named_slow_parameters() if owner_ids else {}
        names = {name for name, value in named.items() if id(value) in owner_ids}
        external = [(source, name, value) for source in (getattr(brain, "slow_anchors", {}), getattr(brain, "slow_importance", {}))
                    for name, value in source.items() if name in names]
        control_bytes = sum(value.numel() * value.element_size() for _, _, value in [*controls, *external])
        self.controls, self.external = [], []
        try:
            self._lease(control_bytes + len(modules) * 1024 + 8192, label + " controls")
            self.controls = [(module, name, value.detach().cpu().clone()) for module, name, value in controls]
            self.external = [(source, name, value.detach().clone()) for source, name, value in external]
            self.counters = dict(brain.counters)
            policy = getattr(brain, "resource_policy", None)
            directory = getattr(brain, "_live_paging_cache_directory", None)
            allowed = bool(getattr(getattr(brain, "config", None), "disk_state_offload", False))
            disk = Path(directory) / "capability-rollback" if allowed and directory is not None and callable(getattr(policy, "require_disk", None)) else None
            self.packed = PackedMutationSnapshot.capture(self.heads.values(), reserve=self._reserve_packed,
                disk_directory=disk, disk_reserve=self._reserve_disk, disk_chunk_bytes=64 * 1024,
                disk_allocate=getattr(policy, "allocate_spill_file", None) if disk is not None else None,
                include_all_uint8=True)
            if self.packed._disk_snapshot is not None:
                for lease in self.disk_leases:
                    lease.bind_path(self.packed._disk_snapshot.path)
                    lease.commit(path=self.packed._disk_snapshot.path)
            self.before_checksum = tensor_checksum((value for head in self.heads.values() for value in head.state_dict().values()), chunk_bytes=64 * 1024)
        except BaseException:
            self.close()
            raise

    def _lease(self, size, label):
        lease = self.reserve(int(size), label) if callable(self.reserve) else None
        if lease is not None:
            self.leases.append(lease)

    def _reserve_packed(self, size):
        # Keep the admitted peak until the snapshot is closed; restoration
        # under pressure must not ask for an identical second allocation.
        if size > self.reserved_packed_bytes:
            self._lease(size - self.reserved_packed_bytes, self.label + " packed rollback")
            self.reserved_packed_bytes = size

    def _reserve_disk(self, size, label):
        policy = self.brain.resource_policy
        policy.require_disk(size, label)
        reserve = getattr(policy, "reserve_spill", None)
        if callable(reserve):
            self.disk_leases.append(reserve(size, label))

    @property
    def storage_mode(self):
        return self.packed.storage_mode

    @torch.no_grad()
    def restore(self):
        if self.closed:
            raise RuntimeError("capability rollback snapshot is closed")
        self.packed.restore()
        for module, name, original in self.controls:
            target = module._buffers.get(name)
            if not isinstance(target, torch.Tensor) or target.shape != original.shape or target.dtype != original.dtype:
                raise RuntimeError("capability control topology changed during rollback")
            target.copy_(original)
        for module, count in self.events:
            module._pending_stability_events = count
        self.brain.counters.clear()
        self.brain.counters.update(self.counters)
        for source, name, original in self.external:
            source[name] = original.clone()

    def restore_modes(self):
        original = {id(module): training for module, training in self.modes}
        for head in self.heads.values():
            head.train(original[id(head)])
        for module, training in self.modes:
            module.training = training

    def close(self):
        if self.closed:
            return
        self.closed = True
        failure = None
        try:
            if self.packed is not None:
                self.packed.close()
        except BaseException as error:
            failure = error
        finally:
            self.controls.clear()
            self.external.clear()
            for lease in reversed([*self.leases, *self.disk_leases]):
                try:
                    lease.release()
                except BaseException as error:
                    failure = failure or error
            self.leases.clear()
            self.disk_leases.clear()
        if failure is not None:
            raise failure
