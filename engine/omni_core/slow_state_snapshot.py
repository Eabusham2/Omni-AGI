"""Bounded private integrity state for slow mutations, not a brain clone."""
import copy
import functools
import tempfile
from contextvars import ContextVar
from pathlib import Path
from collections.abc import Mapping

import torch
from .bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from .offload import NeuralStateResourcePause
from .parameter_diagnostics import active_diagnostic_scopes, packed_diagnostic_restore, ParameterDeltaJournal

_LIFETIME = ContextVar("slow_integrity_snapshot_lifetime", default=None)

def snapshot_metadata_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen: return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor): return 512 # payload stays referenced until bounded file write
    if isinstance(value, str): return len(value) * 4 + 128
    if isinstance(value, Mapping): return 256 + sum(128 + snapshot_metadata_bytes(key, seen) + snapshot_metadata_bytes(item, seen) for key, item in value.items())
    if isinstance(value, (list, tuple)): return 128 + sum(64 + snapshot_metadata_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"): return 256 + snapshot_metadata_bytes(vars(value), seen)
    return 128

def admit_snapshot_metadata(policy, value, stage):
    size = snapshot_metadata_bytes(value) * 3 + 131072
    status = policy.status(estimated_ram_bytes=size)
    if status.get("memoryPressure"):
        raise NeuralStateResourcePause("%s awaits admitted metadata RAM" % stage,
            {**status, "stage": stage, "recoverable": True, "paused": True, "estimatedMetadataBytes": size})
    return size

def slow_snapshot_lifetime(method):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        snapshots = []
        token = _LIFETIME.set(snapshots)
        try: return method(*args, **kwargs)
        finally:
            try:
                for snapshot in reversed(snapshots): snapshot.close()
            finally: _LIFETIME.reset(token)
    return wrapped


class SlowStateSnapshot(dict):
    def __init__(self, modules, controls, metadata, *, directory, policy):
        admit_snapshot_metadata(policy, (controls, metadata), "slow snapshot metadata/control structure")
        super().__init__(copy.deepcopy(metadata))
        self.policy, self.closed = policy, False
        self._diagnostic_scopes = active_diagnostic_scopes()
        self._controls, self._module_keys, tensors, identities = {}, {}, {}, {}
        def tensor_key(value):
            if not value.is_contiguous(): raise ValueError("slow integrity state must expose contiguous owners")
            identity = (value.device, value.data_ptr(), tuple(value.shape), value.dtype)
            if identity not in identities:
                identities[identity] = "tensor.%08d" % len(tensors)
                tensors[identities[identity]] = value.detach()
            return identities[identity]
        def encode(value):
            if isinstance(value, torch.Tensor): return ("tensor", tensor_key(value))
            if isinstance(value, dict): return ("dict", [(encode(key), encode(item)) for key, item in value.items()])
            if isinstance(value, (list, tuple)): return ("tuple" if isinstance(value, tuple) else "list", [encode(item) for item in value])
            if value is None or isinstance(value, (str, int, float, bool)): return ("scalar", value)
            raise TypeError("slow snapshot contains unsupported control type")
        for name, module in modules.items():
            learned_storage = set()
            for child in module.modules():
                getter = getattr(child, "authoritative_packed_tensors", None)
                if callable(getter):
                    learned_storage.update((value.device, value.data_ptr(), tuple(value.shape), value.dtype) for value in getter())
            self._module_keys[name] = {key: "@journal" if (value.device, value.data_ptr(), tuple(value.shape), value.dtype) in learned_storage
                else tensor_key(value) for key, value in module.state_dict().items()}
        self._controls = {key: encode(value) for key, value in controls.items()}
        byte_count = sum(value.numel() * value.element_size() for value in tensors.values())
        self._io_bytes = 8 * 1024 * 1024 + len(tensors) * 4096 + 131072
        self.admit(self._io_bytes, "slow integrity snapshot transfer/header")
        policy.require_disk(byte_count + len(tensors) * 4096 + 131072, "slow mutation nonweight/control integrity state")
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="slow-integrity-", dir=str(directory)))
        self.path = self.directory / "state.safetensors"
        reserve_spill = getattr(policy, "reserve_spill", None)
        self._state_quota = None
        try:
            self._state_quota = reserve_spill(byte_count + len(tensors) * 4096 + 131072, "slow mutation integrity state") if callable(reserve_spill) else None
            atomic_save_tensors_bounded(self.path, tensors, metadata={"format": "omni-slow-integrity-snapshot"}, spill_lease=self._state_quota)
            self.reader = BoundedTensorFile(self.path)
        except BaseException:
            self.close()
            raise
        try:
            self.journal = ParameterDeltaJournal(tuple(modules.items()), directory=Path(directory) / "changed-bytes",
                reserve_ram=lambda size: self.admit(size, "slow first-original byte rollback journal"),
                reserve_disk=policy.require_disk, include_state=True, resource_policy=policy)
        except BaseException:
            self.close()
            raise
        for key in controls: dict.__setitem__(self, key, None)
        dict.__setitem__(self, "modules", None)
        dict.__setitem__(self, "_bounded_state", self)
        lifetime = _LIFETIME.get()
        if lifetime is not None: lifetime.append(self)

    def admit(self, size, stage):
        status = self.policy.status(estimated_ram_bytes=int(size))
        if status.get("memoryPressure"):
            raise NeuralStateResourcePause("%s awaits bounded physical RAM" % stage,
                {**status, "stage": stage, "recoverable": True, "paused": True, "snapshotBytes": int(size)})

    def __getitem__(self, key):
        if key not in self._controls: return dict.__getitem__(self, key)
        encoded = self._controls[key]
        def sizes(value):
            kind, payload = value
            if kind == "tensor": return self.reader.specs[payload].byte_count
            if kind == "dict": return sum(sizes(left) + sizes(right) for left, right in payload)
            if kind in {"list", "tuple"}: return sum(sizes(item) for item in payload)
            return 0
        self.admit(sizes(encoded) * 3 + self._io_bytes, "slow rollback adopted control state")
        def decode(value):
            kind, payload = value
            if kind == "tensor": return self.reader.tensor(payload)
            if kind == "dict": return {decode(left): decode(right) for left, right in payload}
            if kind in {"list", "tuple"}:
                result = [decode(item) for item in payload]
                return tuple(result) if kind == "tuple" else result
            return payload
        return decode(encoded)

    def get(self, key, default=None):
        return self[key] if key in self else default

    def restore_modules(self, modules):
        self.admit(self._io_bytes, "slow rollback bounded module transfer")
        self.journal.restore()
        for name, module in modules.items():
            current = module.state_dict()
            if set(current) != set(self._module_keys[name]): raise RuntimeError("slow rollback module topology did not restore exactly")
            children = dict(module.named_modules())
            for key, target in current.items():
                source_key = self._module_keys[name][key]
                if source_key == "@journal": continue
                spec = self.reader.specs[source_key]
                if target.dtype != spec.dtype or tuple(target.shape) != spec.shape: raise RuntimeError("slow rollback state shape/dtype changed")
                path, _, buffer_name = key.rpartition(".")
                owner = children[path]
                actual = owner._buffers.get(buffer_name, target)
                offset, flat = 0, target.reshape(-1)
                for source in self.reader.chunks(source_key):
                    if target.dtype == torch.uint8:
                        with packed_diagnostic_restore(owner, actual, source, start=offset, captured_scopes=self._diagnostic_scopes):
                            flat[offset:offset + source.numel()].copy_(source)
                    else: flat[offset:offset + source.numel()].copy_(source)
                    from .native_core_paging import release_native_tensor_chunk
                    release_native_tensor_chunk(target, offset * target.element_size(), source.numel() * target.element_size())
                    offset += source.numel()

    def close(self):
        if self.closed: return
        self.closed = True
        if hasattr(self, "journal"): self.journal.close()
        if hasattr(self, "path"): self.path.unlink(missing_ok=True)
        if getattr(self, "_state_quota", None) is not None: self._state_quota.release()
        if hasattr(self, "directory") and self.directory.exists(): self.directory.rmdir()
        self._controls.clear()
        self._module_keys.clear()
        dict.clear(self)
