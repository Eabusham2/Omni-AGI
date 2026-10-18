"""Closed-type, bounded safe-tensor encoding for native execution activity.

No pickle, dynamic imports from metadata, executable payloads, model weights or
source/history reconstruction. A cache is restored from its exact saved bytes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import weakref
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from .bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from .generation_cursor import NativeGenerationCursor
from .working_attention_paging import PagedTensorSequence, SavedActivityTensor


class SuspendedRecallView(Sequence):
    """Read-only structural recall rows for this paused invocation only."""
    def __init__(self, path, count):
        connection = sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA cache_size=-256")
        self.state = SimpleNamespace(connection=connection, recalled_count=count)
        if connection.execute("SELECT COUNT(*) FROM recalled").fetchone()[0] != count:
            connection.close(); raise ValueError("suspended recall coverage changed")
    def __len__(self):
        return self.state.recalled_count
    def __iter__(self):
        for row in self.state.connection.execute("SELECT payload FROM recalled ORDER BY position"):
            yield json.loads(row[0])
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        index = index if index >= 0 else len(self) + index
        row = self.state.connection.execute("SELECT payload FROM recalled WHERE position=?", (index,)).fetchone()
        if row is None:
            raise IndexError("suspended recall position is out of range")
        return json.loads(row[0])
    def close(self):
        self.state.connection.close()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


class ExecutionStateWriter:
    def __init__(self, directory, *, policy=None, cancelled=None):
        self.directory = Path(directory)
        self.policy, self.cancelled = policy, cancelled
        self.files, self._tensor_refs = {}, {}

    def _check(self):
        if self.cancelled is not None and self.cancelled():
            raise InterruptedError("execution snapshot paused before its next exact value")

    def tensor(self, value):
        self._check()
        if not value.is_contiguous() or value.layout != torch.strided:
            raise ValueError("execution activity must be contiguous before snapshot admission")
        key = (str(value.device), str(value.dtype), value.data_ptr(), tuple(value.shape), tuple(value.stride()), int(value._version))
        prior = self._tensor_refs.get(key)
        if prior is not None and prior[0]() is not None:
            return {"kind": "tensor", "file": prior[1]}
        name = "t-%d.safetensors" % len(self.files)
        path = self.directory / name
        size = value.numel() * value.element_size()
        reserve = getattr(self.policy, "reserve_spill", None)
        lease = reserve(size + 65536, "exact execution activity snapshot") if callable(reserve) else None
        if lease is None and self.policy is not None:
            self.policy.require_disk(size + 65536, "exact execution activity snapshot")
        try:
            ram = getattr(self.policy, "reserve_ram", None)
            with ram(min(size, 1024 * 1024) * 2 + 65536, "execution snapshot transfer") if callable(ram) else nullcontext():
                atomic_save_tensors_bounded(path, {"activity": value}, chunk_bytes=1024 * 1024,
                    cancelled=self.cancelled, spill_lease=lease)
            self.files[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size,
                "device": str(value.device), "shape": list(value.shape), "dtype": str(value.dtype)}
            self._tensor_refs[key] = (weakref.ref(value), name)
            return {"kind": "tensor", "file": name}
        finally:
            if lease is not None:
                lease.release()  # Existing immutable snapshot bytes stay charged.

    def encode(self, value):
        self._check()
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                raise ValueError("execution metadata must be finite")
            return value
        if isinstance(value, torch.Tensor):
            return self.tensor(value)
        from .paged_recurrent_spreading import PagedRecalledAssemblies
        if isinstance(value, (PagedRecalledAssemblies, SuspendedRecallView)):
            # Only final readout rows are needed, not a duplicated neural
            # graph or a rerun of spreading activation on changed state.
            source = value.state.connection
            count, payload = source.execute("SELECT COUNT(*),COALESCE(SUM(LENGTH(payload)),0) FROM recalled").fetchone()
            estimate = 65536 + payload * 4 + count * 512
            name = "recall-%d.sqlite3" % len(self.files); path = self.directory / name
            reserve = getattr(self.policy, "reserve_spill", None)
            lease = reserve(estimate, "paused recall readout") if callable(reserve) else None
            if lease is None and self.policy is not None:
                self.policy.require_disk(estimate, "paused recall readout")
            if lease is not None:
                lease.bind_path(path)
            connection = sqlite3.connect(path)
            try:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA cache_size=-256")
                connection.execute("CREATE TABLE recalled(position INTEGER PRIMARY KEY,assembly_id TEXT NOT NULL UNIQUE,payload BLOB NOT NULL)")
                for row in source.execute("SELECT position,assembly_id,payload FROM recalled ORDER BY position"):
                    self._check(); connection.execute("INSERT INTO recalled VALUES(?,?,?)", row)
                connection.commit()
            finally:
                connection.close()
                if lease is not None:
                    lease.commit(path=path); lease.release()
            self.files[name] = {"sha256": file_digest(path), "bytes": path.stat().st_size}
            return {"kind": "recall-view", "file": name, "count": count}
        if isinstance(value, PagedTensorSequence):
            return {"kind": "paged-sequence", "shape": list(value.shape), "axis": value.axis,
                "ends": list(value.ends), "pageTokens": value._page_tokens,
                "pages": [self.tensor(value.pager.read_page(page)) for page in value.pages]}
        if isinstance(value, SavedActivityTensor):
            return {"kind": "saved-activity", "shape": list(value.shape), "device": str(value.device),
                "pages": [self.tensor(value.pager.read_page(page)) for page in value.pages]}
        from .model import _DecoderInferenceCache, _AttentionInferenceCache, _WorkspaceInferenceChunk
        from .connected_regions import _ConnectedInferenceCache
        fields = None
        if isinstance(value, NativeGenerationCursor):
            kind, fields = "generation", vars(value)
        elif isinstance(value, _DecoderInferenceCache):
            kind, fields = "decoder-cache", {name: getattr(value, name) for name in
                ("length", "workspace", "attention", "hidden_sum", "memory_hidden", "neural_state")}
        elif isinstance(value, _ConnectedInferenceCache):
            kind, fields = "connected-cache", {name: getattr(value, name) for name in
                ("length", "workspace", "attention", "temporal", "hidden_sum", "nested")}
        elif isinstance(value, _AttentionInferenceCache):
            kind, fields = "attention-cache", {"key": value.key, "value": value.value}
        elif isinstance(value, _WorkspaceInferenceChunk):
            kind, fields = "workspace-cache", {"queries": value.queries, "mass": value.mass, "weighted_values": value.weighted_values}
        if fields is not None:
            return {"kind": kind, "fields": self.encode(fields)}
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("execution dictionaries require string keys")
            return {"kind": "map", "entries": [[key, self.encode(item)] for key, item in value.items()]}
        if type(value) in (list, tuple, set):
            return {"kind": "tuple" if type(value) is tuple else "set" if type(value) is set else "list",
                "items": [self.encode(item) for item in value]}
        raise ValueError("unsupported native execution-state type: " + type(value).__name__)


class ExecutionStateReader:
    def __init__(self, directory, files, *, pager, device, policy=None):
        self.directory, self.files = Path(directory), files
        self.pager, self.device, self.policy = pager, torch.device(device), policy
        self._values, self._owners = {}, []

    def tensor(self, name, *, page=False):
        if not isinstance(name, str) or not name.startswith("t-") or not name.endswith(".safetensors") or not name[2:-12].isdigit():
            raise ValueError("invalid execution tensor reference")
        if not page and name in self._values:
            return self._values[name]
        descriptor = self.files.get(name)
        path = self.directory / name
        if (not isinstance(descriptor, dict) or path.is_symlink() or not path.is_file()
                or path.stat().st_size != descriptor.get("bytes") or file_digest(path) != descriptor.get("sha256")):
            raise ValueError("execution tensor checksum/ownership failed")
        source = BoundedTensorFile(path)
        if set(source.specs) != {"activity"}:
            raise ValueError("execution payload contains non-activity tensors")
        spec = source.specs["activity"]
        if list(spec.shape) != descriptor.get("shape") or str(spec.dtype) != descriptor.get("dtype"):
            raise ValueError("execution tensor geometry changed")
        recorded = descriptor.get("device")
        if recorded != "cpu" and recorded != str(self.device):
            raise ValueError("exact execution resume requires its recorded compute backend")
        target = torch.device("cpu") if page or recorded == "cpu" else self.device
        reserve = getattr(self.policy, "reserve_ram", None)
        reservation = reserve(spec.byte_count + min(spec.byte_count, 1024 * 1024), "restore exact activity") if callable(reserve) else nullcontext()
        with reservation as held:
            if target.type != "cpu" and self.policy is not None:
                self.policy.require_device(target, spec.byte_count, "restore exact accelerator activity")
            value = torch.empty(spec.shape, dtype=spec.dtype, device=target)
            source.copy_into("activity", value)
            if held is not None:
                held.mark_allocated(spec.byte_count)
        if not page:
            self._values[name] = value
        return value

    def decode(self, value):
        if value is None or type(value) in (str, bool, int, float):
            if type(value) is float and not math.isfinite(value):
                raise ValueError("execution metadata must be finite")
            return value
        if not isinstance(value, dict):
            raise ValueError("invalid execution state tree")
        kind = value.get("kind")
        if kind == "recall-view":
            name = value.get("file")
            if not isinstance(name, str) or not name.startswith("recall-") or not name.endswith(".sqlite3") or not name[7:-8].isdigit():
                raise ValueError("invalid paused recall path")
            path = self.directory / name; descriptor = self.files.get(name)
            if not isinstance(descriptor, dict) or path.is_symlink() or not path.is_file() or path.stat().st_size != descriptor.get("bytes") or file_digest(path) != descriptor.get("sha256"):
                raise ValueError("paused recall checksum failed")
            result = SuspendedRecallView(path, value["count"])
            self._owners.append(result)
            return result
        if kind == "tensor":
            return self.tensor(value["file"])
        if kind in {"map", "list", "tuple", "set"}:
            if kind == "map":
                entries = value.get("entries")
                if not isinstance(entries, list) or any(not isinstance(item, list) or len(item) != 2 or type(item[0]) is not str for item in entries):
                    raise ValueError("invalid execution mapping")
                result = {key: self.decode(item) for key, item in entries}
                if len(result) != len(entries):
                    raise ValueError("duplicated execution mapping key")
                return result
            items = [self.decode(item) for item in value["items"]]
            return tuple(items) if kind == "tuple" else set(items) if kind == "set" else items
        if kind in {"paged-sequence", "saved-activity"}:
            if self.pager is None:
                raise ValueError("paged execution activity requires its admitted pager")
            pages = value.get("pages")
            if not isinstance(pages, list) or not pages:
                raise ValueError("saved activity has no pages")
            first = self.tensor(pages[0]["file"], page=True)
            if kind == "paged-sequence":
                result = PagedTensorSequence(self.pager, tuple(value["shape"]), first.dtype, value["axis"])
                self._owners.append(result)
                page_tokens = value.get("pageTokens")
                if page_tokens is not None and (type(page_tokens) is not int or page_tokens < 1):
                    raise ValueError("paged execution activity has invalid tail geometry")
                result._page_tokens = page_tokens
                for item in pages:
                    tensor = first if item is pages[0] else self.tensor(item["file"], page=True)
                    expected = list(value["shape"]); expected.pop(result.axis)
                    if tensor.dtype != result.dtype or list(tensor.shape[1:]) != expected:
                        raise ValueError("paged execution activity geometry changed")
                    result.pages.append(self.pager.save_page(tensor, retained_context=True))
                    result._shape[result.axis] += tensor.shape[0]
                    result.ends.append(result.length)
                if list(result.shape) != value["shape"] or result.ends != value["ends"]:
                    raise ValueError("paged execution activity has changed coverage")
            else:
                result = SavedActivityTensor.__new__(SavedActivityTensor)
                result.pager, result.shape, result.dtype = self.pager, tuple(value["shape"]), first.dtype
                if value["device"] not in {"cpu", str(self.device)}:
                    raise ValueError("saved activity requires its recorded backend")
                result.device, result.numel = torch.device(value["device"]), math.prod(result.shape)
                result.element_size, result.pages, result._restore_image, result._closed = first.element_size(), [], None, False
                self._owners.append(result)
                total = 0
                for item in pages:
                    tensor = first if item is pages[0] else self.tensor(item["file"], page=True)
                    if tensor.dtype != result.dtype:
                        raise ValueError("saved execution activity dtype changed")
                    total += tensor.numel(); result.pages.append(self.pager.save_page(tensor))
                if total != result.numel:
                    raise ValueError("saved execution activity has incomplete coverage")
            return result
        fields = self.decode(value.get("fields"))
        if not isinstance(fields, dict):
            raise ValueError("invalid execution cache fields")
        from .model import _DecoderInferenceCache, _AttentionInferenceCache, _WorkspaceInferenceChunk
        from .connected_regions import _ConnectedInferenceCache
        types = {"generation": NativeGenerationCursor, "decoder-cache": _DecoderInferenceCache,
                 "connected-cache": _ConnectedInferenceCache, "attention-cache": _AttentionInferenceCache,
                 "workspace-cache": _WorkspaceInferenceChunk}
        if kind not in types:
            raise ValueError("unknown execution-state kind")
        if kind in {"decoder-cache", "connected-cache"}:
            fields["signature"] = None  # Lifecycle binds to the verified reloaded owners.
        result = types[kind](**fields)
        if hasattr(result, "close"):
            self._owners.append(result)
        return result

    def release(self):
        self._owners.clear(); self._values.clear()

    def close(self):
        for owner in reversed(self._owners):
            owner.close()
        self.release()
