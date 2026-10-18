"""Current inspection's entered high-water escrow and cancellation boundary."""
from contextlib import contextmanager
from contextvars import ContextVar

_CURRENT = ContextVar("owned_inspection_resources", default=(None, None))


@contextmanager
def inspection_resources(admit, cancelled=None):
    token = _CURRENT.set((admit, cancelled))
    try:
        yield
    finally:
        _CURRENT.reset(token)


def current_inspection_resources():
    return _CURRENT.get()
