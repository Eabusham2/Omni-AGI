"""Scoped packed derivative deferral; no model import or weight master."""

from contextlib import contextmanager
from contextvars import ContextVar


_SINK = ContextVar("omni_packed_derivative_sink", default=None)
_OWNER = ContextVar("omni_packed_derivative_owner", default=None)


def current_packed_row_owner():
    return _OWNER.get()


def packed_derivative_collective_active():
    """Collective loss gradients are already globally normalized."""
    return _SINK.get() is not None


@contextmanager
def packed_row_owner(module):
    token = _OWNER.set(module)
    try:
        yield
    finally:
        _OWNER.reset(token)


@contextmanager
def packed_derivative_sink(sink):
    token = _SINK.set(sink)
    try:
        yield
    finally:
        _SINK.reset(token)


def defer_packed_rows(packed, width, row_start, gradient, rate, scale,
                      row_stability=None, stability_strength=0.0):
    """Return None normally, or stage a derivative without local mutation.

    Collectives are deliberately NOT called inside backward: ranks can visit
    projections in different orders, and empty tail ranks visit none.
    """
    sink = _SINK.get()
    if sink is None:
        return None
    owner = _OWNER.get()
    if owner is None:
        raise RuntimeError("distributed packed derivative has no registered owner")
    return sink.stage(owner, packed, int(width), int(row_start), gradient,
                      float(rate), scale, row_stability, float(stability_strength))
