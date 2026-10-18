"""Private same-home learner RNG, with one explicit genuine-state commit.

Ordinary unscoped learning uses the unchanged torch RNG calls. A qualified
learner wave draws on its original home using independent generators. Trials
never reset/advance the process RNG; the genuine wave advances it only once.
"""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import hashlib

import torch

_RANDOM = ContextVar("native_whole_learner_randomness", default=None)
_EXECUTION = ContextVar("native_whole_learner_execution", default=None)


def training_execution_device(default):
    current = _EXECUTION.get()
    return torch.device(default if current is None else current["device"])


def training_private():
    current = _EXECUTION.get()
    return current is not None and current["private"]


def training_wave_active(): return _EXECUTION.get() is not None


@contextmanager
def training_execution(device, *, private, batches=None):
    token = _EXECUTION.set({"device": torch.device(device), "private": bool(private), "batches": batches})
    try: yield
    finally: _EXECUTION.reset(token)


def training_prepared_batches():
    current = _EXECUTION.get()
    return None if current is None else current["batches"]


def current_rng_state(home):
    home = torch.device(home)
    if home.type == "cpu": return torch.get_rng_state()
    if home.type == "cuda": return torch.cuda.get_rng_state(home)
    if home.type == "mps": return torch.mps.get_rng_state()
    raise ValueError("learner RNG home has no reviewed state protocol")


def publish_rng_state(home, state):
    home = torch.device(home)
    if home.type == "cpu": torch.set_rng_state(state)
    elif home.type == "cuda": torch.cuda.set_rng_state(state, home)
    elif home.type == "mps": torch.mps.set_rng_state(state)
    else: raise ValueError("learner RNG home has no reviewed state protocol")


class TrainingRandomScope:
    def __init__(self, home, state, *, transfer, cancelled, reserve_conversion=None):
        self.home, self.before = torch.device(home), state
        self.transfer, self.cancelled = transfer, cancelled
        self.reserve_conversion = reserve_conversion or (lambda _size: nullcontext())
        self.raw = torch.Generator(device=self.home)
        self.raw.set_state(state)
        self.seeded, self.draws = {}, 0
    def check(self):
        if self.cancelled(): raise InterruptedError("private learner RNG stopped")
    def generator(self, original):
        if original is None: return self.raw
        # A seed cannot translate an already-advanced backend RNG state.
        # This local consumer has no reviewed cross-rank generator protocol.
        if torch.device(original.device) != self.home:
            raise ValueError("whole learner encountered an unowned non-home generator")
        key = id(original)
        if key not in self.seeded:
            before = original.get_state().clone()
            private = torch.Generator(device=self.home); private.set_state(before)
            self.seeded[key] = (original, private, before)
        return self.seeded[key][1]
    def draw(self, shape, *, dtype, device, normal=False, generator=None):
        self.check(); self.draws += 1
        method = torch.randn if normal else torch.rand
        value = method(shape, dtype=dtype, device=self.home, generator=self.generator(generator))
        return self.transfer(value, torch.device(device))
    def summary(self):
        return {"home": str(self.home), "draws": self.draws,
            "stateSha256": hashlib.sha256(self.raw.get_state().numpy().tobytes()).hexdigest(),
            "seeded": [[value[0].initial_seed(), hashlib.sha256(value[1].get_state().numpy().tobytes()).hexdigest()]
                for value in self.seeded.values()]}
    def commit(self):
        # A private trial must not undo unrelated draws by another owner.
        if not torch.equal(current_rng_state(self.home), self.before):
            raise RuntimeError("original learner RNG changed before genuine publication")
        if any(not torch.equal(original.get_state(), before) for original, _private, before in self.seeded.values()):
            raise RuntimeError("explicit learner generator changed before genuine publication")
        publish_rng_state(self.home, self.raw.get_state())
        for original, private, _before in self.seeded.values(): original.set_state(private.get_state())
    @contextmanager
    def activate(self):
        token = _RANDOM.set(self)
        try: yield self
        finally: _RANDOM.reset(token)


def learning_rand(shape, *, device, dtype=None, generator=None):
    scope = _RANDOM.get()
    if scope is None: return torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return scope.draw(shape, dtype=dtype or torch.get_default_dtype(), device=device, generator=generator)


def learning_randn_like(value):
    scope = _RANDOM.get()
    return torch.randn_like(value) if scope is None else scope.draw(value.shape, dtype=value.dtype, device=value.device, normal=True)


def learning_transfer(value, device, *, dtype=None):
    scope = _RANDOM.get()
    if scope is None: return value.to(device=device, dtype=dtype, non_blocking=False)
    if dtype is None or value.dtype == dtype: return scope.transfer(value, torch.device(device))
    size = value.numel()*torch.empty((), dtype=dtype).element_size()
    with scope.reserve_conversion(size) as held:
        converted = value.to(dtype=dtype)
        if held is not None: held.mark_allocated(size)
        return scope.transfer(converted, torch.device(device))


class PrivateWaveSnapshot:
    """A failed private namespace is discarded, never restored into originals."""
    def close(self): pass
    def restore(self): raise RuntimeError("private learner has no original rollback authority")
