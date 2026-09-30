"""Resource-admitted CPU skeleton/state copy of an existing neural region.

No model constructor, full brain clone, original-device deepcopy, or floating
master for packed weights. Packed buffers use admitted RAM first, then private
mapped CPU pages only when the shared remaining envelope cannot fit an owner.
"""
import copy
import types
from pathlib import Path

import torch
from torch import nn
from .native_core_paging import NativeCorePager
from .offload import NeuralStateResourcePause
from .slow_state_snapshot import admit_snapshot_metadata


class IsolatedModuleSnapshot:
    def __init__(self, sources, *, directory, policy, cancelled=lambda: False, chunk_bytes=1024 * 1024):
        self.policy, self.cancelled = policy, cancelled
        self.chunk_bytes = int(chunk_bytes)
        self.roots, self.memo = {}, {}
        self.closed = False
        modules, tensors, mappable = {}, {}, set()
        visited = set()
        def collect(value):
            if isinstance(value, torch.Tensor): tensors[id(value)] = value; return
            if isinstance(value, nn.Module) or id(value) in visited: return
            visited.add(id(value))
            if isinstance(value, dict):
                for item in value.values(): collect(item)
            elif isinstance(value, (list, tuple)):
                for item in value: collect(item)
        for source in sources.values():
            for module in source.modules():
                modules[id(module)] = module
                if getattr(module, "_online_transaction", None) is not None:
                    raise RuntimeError("cannot snapshot a region during an online mutation")
                for tensor in (*module._parameters.values(), *module._buffers.values()):
                    if isinstance(tensor, torch.Tensor):
                        tensors[id(tensor)] = tensor
                for name, tensor in module._buffers.items():
                    if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.uint8 and name not in module._non_persistent_buffers_set:
                        mappable.add(id(tensor))
                for name, value in module.__dict__.items():
                    if name not in {"_native_core_pager", "native_core_pager"}: collect(value)
        packed = sum(value.numel() for identity, value in tensors.items() if identity in mappable)
        nonweight = sum(value.numel() * value.element_size() for identity, value in tensors.items() if identity not in mappable)
        self.minimum_resident_bytes = nonweight + len(modules) * 4096 + 131072
        self.packed_bytes = packed
        metadata_bytes = admit_snapshot_metadata(policy, {str(identity): {name: value for name, value in module.__dict__.items()
            if name not in {"_native_core_pager", "native_core_pager", "_modules"}} for identity, module in modules.items()}, "isolated module metadata")
        self.minimum_resident_bytes = nonweight + max(metadata_bytes, len(modules) * 4096 + 131072)
        self.admit(self.minimum_resident_bytes + self.chunk_bytes * 3, "isolated modality nonweight state and transfer scratch")
        low, high = 0, packed
        while low < high:
            candidate = (low + high + 1) // 2
            if policy.status(estimated_ram_bytes=self.minimum_resident_bytes + self.chunk_bytes * 3 + candidate).get("memoryPressure"):
                high = candidate - 1
            else: low = candidate
        self.admitted_packed_ram_bytes = low
        if packed > low:
            policy.require_disk(packed - low + len(tensors) * 4096 + 131072, "isolated same-brain cold packed modality state")
        self.pager = NativeCorePager(Path(directory), cpu_hot_bytes=low, accelerator_hot_bytes=0,
            reserve_disk=policy.require_disk, chunk_bytes=self.chunk_bytes,
            reserve_admission=lambda size, _device: self.admit(size, "isolated modality transfer"))
        try:
            source_pagers = {getattr(module, "_native_core_pager", None) for module in modules.values()} - {None}
            for pager in source_pagers: pager.flush()
            for name, source in sources.items(): self.roots[name] = self.clone_module(source)
            self.pager.bind_names((name + ".", root) for name, root in self.roots.items())
        except BaseException:
            self.close()
            raise

    def admit(self, size, stage):
        if self.cancelled(): raise InterruptedError("isolated modality snapshot cancelled")
        status = self.policy.status(estimated_ram_bytes=int(size))
        if status.get("memoryPressure"):
            raise NeuralStateResourcePause("%s requires an admitted resident minimum" % stage,
                {**status, "stage": stage, "recoverable": True, "paused": True,
                 "sameBrain": True, "minimumResidentBytes": int(size), "packedSnapshotBytes": self.packed_bytes,
                 "snapshotMode": "bounded-cpu-mapped-packed-region", "neuralWorkerRestartRequired": False})

    def tensor(self, source, *, owner=None, name=None, persistent=False, parameter=False):
        cached = self.memo.get(id(source))
        if cached is not None: return cached
        if not source.is_contiguous():
            raise RuntimeError("isolated neural state must expose contiguous tensor owners")
        count = source.numel() * source.element_size()
        self.admit(min(count, self.chunk_bytes) * 3 + 4096, "isolated modality bounded copy")
        if owner is not None and persistent and source.dtype == torch.uint8:
            # Other neural work shares this process envelope. Recheck at each
            # owner rather than treating the earlier RAM-first estimate as a
            # reserved independent budget if headroom changed during copying.
            if self.policy.status(estimated_ram_bytes=count + self.chunk_bytes * 3 + 4096).get("memoryPressure"):
                self.pager.cpu_hot_bytes = min(self.pager.cpu_hot_bytes, self.pager._heap_bytes())
            target = self.pager.allocate(owner, name, tuple(source.shape))
        else:
            self.admit(count + min(count, self.chunk_bytes) * 2 + 4096, "isolated modality unavoidable nonweight destination")
            target = torch.empty(tuple(source.shape), dtype=source.dtype, device="cpu")
        flat_source, flat_target = source.detach().reshape(-1), target.reshape(-1)
        step = max(1, self.chunk_bytes // source.element_size())
        for start in range(0, source.numel(), step):
            if self.cancelled(): raise InterruptedError("isolated modality snapshot cancelled at byte boundary")
            block = flat_source[start:start + step].to(device="cpu")
            flat_target[start:start + block.numel()].copy_(block)
            from .native_core_paging import release_native_tensor_chunk
            release_native_tensor_chunk(source, start * source.element_size(), block.numel() * source.element_size())
            release_native_tensor_chunk(target, start * target.element_size(), block.numel() * target.element_size())
        if parameter: target = nn.Parameter(target, requires_grad=False)
        elif source.requires_grad: target.requires_grad_(True)
        self.memo[id(source)] = target
        return target

    def value(self, value):
        if isinstance(value, nn.Module): return self.clone_module(value)
        if isinstance(value, torch.Tensor): return self.tensor(value)
        if isinstance(value, dict): return type(value)((key, self.value(item)) for key, item in value.items())
        if isinstance(value, list): return [self.value(item) for item in value]
        if isinstance(value, tuple): return tuple(self.value(item) for item in value)
        if isinstance(value, torch.device): return torch.device("cpu")
        if callable(value): return value
        if hasattr(value, "__dict__") and any(isinstance(item, torch.Tensor) for item in vars(value).values()):
            raise RuntimeError("isolated modality has unregistered tensor-bearing control state")
        return copy.deepcopy(value)

    def clone_module(self, source):
        cached = self.memo.get(id(source))
        if cached is not None: return cached
        clone = object.__new__(type(source)) # exact existing architecture, no constructor/initialization
        self.memo[id(source)] = clone
        clone.__dict__ = {}
        for key, value in source.__dict__.items():
            if key in {"_native_core_pager", "native_core_pager"}: clone.__dict__[key] = self.pager
            elif key == "_native_compute_device": clone.__dict__[key] = torch.device("cpu")
            elif key in {"_validated_device", "_online_pager_scope"}: clone.__dict__[key] = None
            elif key == "_parameters":
                clone.__dict__[key] = {name: self.tensor(item, parameter=True) if item is not None else None for name, item in value.items()}
            elif key == "_buffers":
                nonpersistent = source._non_persistent_buffers_set
                clone.__dict__[key] = {name: self.tensor(item, owner=clone, name=name, persistent=name not in nonpersistent) if item is not None else None for name, item in value.items()}
            elif isinstance(value, types.MethodType) and value.__self__ is source:
                clone.__dict__[key] = types.MethodType(value.__func__, clone)
            else: clone.__dict__[key] = self.value(value)
        for key in ("_packed_validated_version", "_bias_validated_version", "_scale_validated_version", "_row_stability_validated_version", "_bias_row_stability_validated_version"):
            if key in clone.__dict__: clone.__dict__[key] = -1
        self.pager.attach(clone, loaded=True)
        clone.eval()
        return clone

    def close(self):
        if self.closed: return
        self.closed = True
        for value in list(self.memo.values()):
            if isinstance(value, nn.Module):
                value.__dict__.get("_parameters", {}).clear()
                value.__dict__.get("_buffers", {}).clear()
                for name, item in list(value.__dict__.items()):
                    if name not in {"_parameters", "_buffers", "_modules", "training"}:
                        value.__dict__[name] = None
        self.memo.clear()
        self.roots.clear()
        if hasattr(self, "pager"): self.pager.close()
