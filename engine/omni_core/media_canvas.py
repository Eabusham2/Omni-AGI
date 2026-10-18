"""Owned out-of-core media activity, not learned weights or an output lookup.

Small media retains its existing device-tensor implementation. This file-backed
canvas holds the SAME overlap-add numerators/weights for admitted large output.
Only bounded native regions and codec rows become resident; storage allocation
uses the existing selected-drive/shared quota protocol.
"""
from __future__ import annotations

import functools
import hashlib
import itertools
import json
import math
import os
import stat
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import torch
from .media_randomness import media_transfer

_LIFETIME = ContextVar("native_media_activity_lifetime", default=None)
IO_BYTES = 256 * 1024


def _track(value):
    owners = _LIFETIME.get()
    if owners is None: raise RuntimeError("media storage needs its owned operation lifetime")
    owners.append(value)
    return value


def media_storage_lifetime(method):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        owners = []
        token = _LIFETIME.set(owners)
        try: return method(*args, **kwargs)
        finally:
            cleanup_error = None
            try:
                for owner in reversed(owners):
                    try: owner.close()
                    except BaseException as error:
                        if cleanup_error is None: cleanup_error = error
            finally: _LIFETIME.reset(token)
            if cleanup_error is not None: raise cleanup_error
    return wrapped


def active_media_lifetime(): return _LIFETIME.get() is not None


class _Grid:
    def __init__(self, shape, dtype, *, directory, policy, cancelled):
        self.shape, self.dtype, self.policy, self.cancelled = tuple(int(value) for value in shape), dtype, policy, cancelled
        if not self.shape or any(value < 1 for value in self.shape): raise ValueError("media canvas dimensions must be positive")
        self.element = torch.empty((), dtype=dtype).element_size()
        self.size = math.prod(self.shape) * self.element
        if self.size > (1 << 63) - 1: raise ValueError("media canvas exceeds the platform file offset range")
        self.handle = self.lease = self.path = None
        self.closed, self._lock = False, threading.RLock()
        try:
            self.check()
            placed = policy.allocate_spill_file(self.size, "native media overlap activity", ".activity", cancelled=self.check)
            if placed is not None:
                self.lease, self.path = placed, Path(placed.path)
                descriptor = placed.open(os.O_RDWR)
            else:
                directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
                self.path = directory / (uuid.uuid4().hex + ".activity")
                self.lease = policy.reserve_spill(self.size, "native media overlap activity")
                self.lease.bind_path(self.path)
                descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            self.handle = os.fdopen(descriptor, "r+b", buffering=0)
            opened = os.fstat(self.handle.fileno()); self.identity = (opened.st_dev, opened.st_ino)
            self.handle.truncate(self.size)
            self.lease.commit(path=self.path, promised_bytes=self.size, retain_open_backing=True)
        except BaseException:
            self.close(); raise

    def check(self):
        if self.closed: raise RuntimeError("media activity canvas is closed")
        if self.cancelled is not None and self.cancelled(): raise InterruptedError("media canvas stopped at an I/O boundary")

    def _verify(self):
        self.check()
        current, opened = self.path.lstat(), os.fstat(self.handle.fileno())
        if (not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self.identity
                or (opened.st_dev, opened.st_ino) != self.identity or current.st_size != self.size):
            raise RuntimeError("owned media activity backing was replaced")

    def _rows(self, slices):
        if len(slices) != len(self.shape): raise ValueError("media region rank changed")
        bounds = []
        for selected, capacity in zip(slices, self.shape):
            if not isinstance(selected, slice) or selected.step not in (None, 1): raise ValueError("media regions need unit-stride slices")
            first, end = 0 if selected.start is None else selected.start, capacity if selected.stop is None else selected.stop
            if type(first) is not int or type(end) is not int or not 0 <= first < end <= capacity: raise ValueError("media region exceeds its owned shape")
            bounds.append((first, end))
        shape = tuple(end-first for first, end in bounds)
        strides = [math.prod(self.shape[index+1:]) for index in range(len(self.shape))]
        for coordinates in itertools.product(*(range(first, end) for first, end in bounds[:-1])):
            offset = sum(value * stride for value, stride in zip(coordinates, strides)) + bounds[-1][0]
            yield offset * self.element, shape[-1], shape

    def read(self, slices):
        rows = self._rows(slices); first = next(rows)
        shape = first[2]
        result = torch.empty(shape, dtype=self.dtype, device="cpu")
        flat, offset = result.reshape(-1), 0
        with self._lock:
            self._verify()
            for position, count, _shape in itertools.chain((first,), rows):
                self.check(); self.handle.seek(position)
                raw = bytearray(self.handle.read(count * self.element))
                if len(raw) != count * self.element: raise RuntimeError("media canvas row was truncated")
                flat[offset:offset+count].copy_(torch.frombuffer(raw, dtype=self.dtype))
                offset += count
        return result

    def write(self, slices, value):
        with self._lock:
            self._verify(); rows = self._rows(slices); first = next(rows)
            flat, offset = media_transfer(value.detach(), "cpu").to(dtype=self.dtype).contiguous().reshape(-1), 0
            for position, count, shape in itertools.chain((first,), rows):
                if math.prod(shape) != value.numel(): raise ValueError("media region payload does not match its exact slices")
                self.check(); self.handle.seek(position)
                raw = flat[offset:offset+count].view(torch.uint8).numpy().tobytes()
                written = self.handle.write(raw)
                if written != len(raw): raise RuntimeError("media canvas row write was incomplete")
                offset += count
            self.lease.refresh_physical()

    def close(self):
        if self.closed: return
        self.closed = True
        if self.handle is not None: self.handle.close(); self.handle = None
        if self.path is not None:
            try:
                current = self.path.lstat()
                if stat.S_ISREG(current.st_mode) and (current.st_dev, current.st_ino) == getattr(self, "identity", None): self.path.unlink()
            except FileNotFoundError: pass
        if self.lease is not None: self.lease.backing_closed(); self.lease = None


class MediaCanvas:
    def __init__(self, shape, *, modality, dtype, directory, policy, cancelled=None):
        self.shape, self.modality, self.dtype, self.policy = tuple(shape), modality, dtype, policy
        self.output = self.weights = None
        self.closed = False
        try:
            self.output = _Grid(shape, dtype, directory=directory, policy=policy, cancelled=cancelled)
            weight_shape = (1, shape[-1]) if modality == "audio" else (1, 1, *shape[2:])
            self.weights = _Grid(weight_shape, dtype, directory=directory, policy=policy, cancelled=cancelled)
            _track(self)
        except BaseException: self.close(); raise

    def add(self, region, value, weight):
        shape = tuple((self.shape[index] if item.stop is None else item.stop) - (item.start or 0) for index, item in enumerate(region))
        count = math.prod(shape)
        with self.policy.reserve_ram(count * self.output.element * 8 + IO_BYTES, "bounded media overlap-add native region"):
            old = self.output.read(region)
            weight = media_transfer(weight.detach(), "cpu").to(dtype=self.dtype)
            weighted = media_transfer(value.detach(), "cpu").to(dtype=self.dtype) * weight
            self.output.write(region, old + weighted)
            weight_region = (slice(0, 1), region[-1]) if self.modality == "audio" else (slice(0, 1), slice(0, 1), *region[2:])
            old_weight = self.weights.read(weight_region)
            self.weights.write(weight_region, old_weight + weight)

    def read(self, region):
        weight_region = (slice(0, 1), region[-1]) if self.modality == "audio" else (slice(0, 1), slice(0, 1), *region[2:])
        return (self.output.read(region) / self.weights.read(weight_region).clamp_min(1e-8)).clamp(-1.0, 1.0)

    def view(self, *, end=None): return MediaCanvasView(self, end=end)

    def covered_fraction(self):
        covered, total = 0, math.prod(self.weights.shape)
        width = self.weights.shape[-1]; fragment = max(1, IO_BYTES // self.output.element)
        with self.policy.reserve_ram(IO_BYTES * 3, "media covered-activity scan"):
            for prefix in itertools.product(*(range(value) for value in self.weights.shape[:-1])):
                for start in range(0, width, fragment):
                    region = tuple(slice(value, value+1) for value in prefix) + (slice(start, min(width, start+fragment)),)
                    covered += int((self.weights.read(region) > 0).sum().item())
        return covered / float(total)

    def close(self):
        if self.closed: return
        self.closed = True
        try:
            if self.output is not None: self.output.close()
        finally:
            if self.weights is not None: self.weights.close()


class MediaCanvasView:
    def __init__(self, canvas, *, end=None):
        self.canvas, self.dtype = canvas, canvas.dtype
        shape = list(canvas.shape)
        axis = 1 if canvas.modality == "audio" else 2
        if end is not None:
            if type(end) is not int or not 1 <= end <= shape[axis]: raise ValueError("media prefix exceeds its actual activity")
            shape[axis] = end
        self.shape, self.ndim = tuple(shape), len(shape)
    def numel(self): return math.prod(self.shape)

    def _rgb_rows(self, frame=0):
        width, height = self.shape[-1], self.shape[-2]
        pixels = max(1, IO_BYTES // 64)
        with self.canvas.policy.reserve_ram(IO_BYTES * 6, "streamed RGB activity row"):
            for row in range(height):
                yield None  # row boundary, not a fabricated pixel
                for start in range(0, width, pixels):
                    region = (slice(0, 1), slice(0, 3))
                    if self.canvas.modality == "video": region += (slice(frame, frame+1),)
                    region += (slice(row, row+1), slice(start, min(width, start+pixels)))
                    value = self.canvas.read(region).reshape(3, -1)
                    value = ((value.float() + 1.0) * 127.5).to(torch.uint8).transpose(0, 1).contiguous()
                    yield value.numpy().tobytes()

    def pcm_chunks(self):
        fragment = max(1, IO_BYTES // 32)
        with self.canvas.policy.reserve_ram(IO_BYTES * 6, "streamed PCM activity window"):
            for batch in range(self.shape[0]):
                for start in range(0, self.shape[-1], fragment):
                    value = self.canvas.read((slice(batch, batch+1), slice(start, min(self.shape[-1], start+fragment))))
                    integers = (value.float().reshape(-1) * 32767.0).to(torch.int16)
                    raw = integers.view(torch.uint8).numpy().tobytes()
                    if os.sys.byteorder != "little": raw = b"".join(raw[index:index+2][::-1] for index in range(0, len(raw), 2))
                    yield raw

    def neural_sha256(self):
        digest = hashlib.sha256(json.dumps([list(self.shape), str(self.dtype)], separators=(",", ":")).encode())
        width, fragment = self.shape[-1], max(1, IO_BYTES // self.canvas.output.element)
        with self.canvas.policy.reserve_ram(IO_BYTES * 6, "streamed exact media activity proof"):
            for prefix in itertools.product(*(range(value) for value in self.shape[:-1])):
                for start in range(0, width, fragment):
                    region = tuple(slice(value, value+1) for value in prefix) + (slice(start, min(width, start+fragment)),)
                    values = self.canvas.read(region).contiguous()
                    if not bool(torch.isfinite(values).all()): raise RuntimeError("native media contains nonfinite activity")
                    digest.update(values.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()


def create_media_canvas(brain, plan, idea, cancelled=None):
    """Actual caller factory: memory-first lease, selected-drive spill if enabled."""
    if not active_media_lifetime(): return None
    if getattr(plan, "canvas_storage", "device") != "file": return None
    if not brain.config.disk_state_offload: raise RuntimeError("media activity spill is disabled for this instance")
    shape = ((idea.shape[0], plan.total_samples) if plan.modality == "audio" else
             (idea.shape[0], 3, plan.total_frames, plan.height, plan.width) if plan.modality == "video" else
             (idea.shape[0], 3, plan.height, plan.width))
    return MediaCanvas(shape, modality=plan.modality, dtype=idea.dtype,
        directory=brain._live_paging_cache_directory / "media-activity", policy=brain.resource_policy, cancelled=cancelled)
