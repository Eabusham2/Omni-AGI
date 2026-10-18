"""Home-device randomness for the same native media computation.

An absent scope preserves the original generator/device path. A scoped route
draws noise and samples codec logits on the ORIGINAL home, then transfers live
activity. Equal seeds on different backends are never treated as equal RNG.
Only hashes/control metadata are retained, never sampled output as a lookup.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.nn import functional as F

_SCOPE = ContextVar("native_media_home_randomness", default=None)


class _Generator:
    def __init__(self, raw, scope, ordinal): self.raw, self.scope, self.ordinal = raw, scope, ordinal
    @property
    def device(self): return self.raw.device


class MediaRandomScope:
    def __init__(self, home, *, transfer=None, cancelled=lambda: False):
        self.home, self.transfer, self.cancelled = torch.device(home), transfer, cancelled
        self.digest, self.generators, self.draws = hashlib.sha256(), 0, 0
    def check(self):
        if self.cancelled(): raise InterruptedError("media home RNG stopped")
    def _record(self, value):
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.digest.update(len(raw).to_bytes(8, "little")); self.digest.update(raw)
    def generator(self, seed):
        self.check(); value = torch.Generator(device=self.home); value.manual_seed(int(seed))
        ordinal = self.generators; self.generators += 1
        self._record(["generator", ordinal, int(seed), str(self.home)])
        return _Generator(value, self, ordinal)
    def move(self, value, target):
        self.check(); target = torch.device(target)
        if value.device == target: return value
        if self.transfer is None: raise RuntimeError("media cross-device RNG needs its admitted actual-transfer callback")
        return self.transfer(value, target)
    def observe(self, generator, kind, shape, dtype):
        self.check()
        if not isinstance(generator, _Generator) or generator.scope is not self: raise RuntimeError("media randomness encountered an undeclared generator")
        state = generator.raw.get_state()
        self._record([kind, generator.ordinal, list(shape), str(dtype), hashlib.sha256(state.numpy().tobytes()).hexdigest()])
        self.draws += 1
    def summary(self): return {"home": str(self.home), "draws": self.draws, "generators": self.generators, "sha256": self.digest.hexdigest()}
    @contextmanager
    def activate(self):
        token = _SCOPE.set(self)
        try: yield self
        finally: _SCOPE.reset(token)


def media_generator(device, seed):
    scope = _SCOPE.get()
    if scope is not None: return scope.generator(seed)
    return torch.Generator(device=device).manual_seed(int(seed))


def media_randn(*shape, generator, device, dtype):
    scope = _SCOPE.get()
    if scope is None: return torch.randn(*shape, generator=generator, device=device, dtype=dtype)
    if not isinstance(generator, _Generator) or generator.scope is not scope: raise RuntimeError("media noise generator belongs to another operation")
    value = torch.randn(*shape, generator=generator.raw, device=generator.device, dtype=dtype)
    scope.observe(generator, "normal", value.shape, value.dtype)
    return scope.move(value, device)


def media_codec_sample(logits, generator):
    scope = _SCOPE.get()
    if scope is not None and (not isinstance(generator, _Generator) or generator.scope is not scope):
        raise RuntimeError("media codec generator belongs to another operation")
    values = logits if scope is None else scope.move(logits, generator.device)
    batch, length, choices = values.shape
    probabilities = F.softmax(values.float(), dim=-1).reshape(batch * length, choices)
    raw = generator if scope is None else generator.raw
    result = torch.multinomial(probabilities, 1, generator=raw).reshape(batch, length)
    if scope is None: return result
    scope.observe(generator, "codec-multinomial", result.shape, result.dtype)
    return scope.move(result, logits.device)


def media_transfer(value, device):
    scope = _SCOPE.get()
    return value.to(device=device, non_blocking=False) if scope is None else scope.move(value, device)
