"""Actual CPU recurrent transfer counters, off by default, no neural imports."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_RECURRENT_TRANSFER = ContextVar("native_operation_recurrent_transfer_observer", default=None)


@contextmanager
def recurrent_transfer_observation(callback):
    token = _RECURRENT_TRANSFER.set(callback)
    try: yield
    finally: _RECURRENT_TRANSFER.reset(token)


def observed_cpu_recurrence(method):
    @wraps(method)
    def wrapped(owner, inputs):
        callback = _RECURRENT_TRANSFER.get()
        if callback is None or inputs.device.type == "cpu": return method(owner, inputs)
        # Current STDPSynapses.recurrent actually transfers this whole live
        # input to CPU and its completed post-neuron result back to input home.
        value = method(owner, inputs)
        callback(int(inputs.numel() * inputs.element_size() + value.numel() * value.element_size()))
        return value
    return wrapped
