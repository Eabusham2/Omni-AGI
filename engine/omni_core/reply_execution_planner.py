"""Actual request-owned reply dispatch, never private replay of a conversation.

No neural imports or work on import. The thirteen complete native consumers
remain responsible for measured equality, placement, transfer and admission.
This executor chooses the semantic-compatible consumer at the real frontier,
pins non-isolatable action/Ponder work to its owner, and fences world effects.
"""
from __future__ import annotations

import functools
import inspect
import threading
from contextlib import contextmanager
from contextvars import ContextVar

from .operation_auto_registry import PRIVATE_OPERATION
from .whole_path_scheduler import (current_owned_path, owned_phase,
    request_qualification_gate, world_effect_barrier)

_REPLY = ContextVar("omni_actual_reply_executor", default=None)
_EVENT_TYPE = threading.Event
_CALLBACKS = ("stream_callback", "cancel_check", "steer_check", "tool_observation_provider",
    "chat_control_callback", "suspension_check", "suspend_callback")
_STAGES = frozenset({"routing-forward", "action-readout", "structured-action-arguments",
    "voluntary-ponder", "ponder-self-report", "branch-action-arguments", "reply-finalization"})


class _WorldCallback:
    """Keep callable-object protocols; do not turn inboxes into plain lambdas."""
    def __init__(self, callback): self.callback = callback
    def __call__(self, *args, **kwargs):
        with world_effect_barrier(): return self.callback(*args, **kwargs)
    def __getattr__(self, name):
        value = getattr(self.callback, name)
        return _WorldCallback(value) if callable(value) else value


def world_callback(callback):
    return callback if callback is None or isinstance(callback, _WorldCallback) else _WorldCallback(callback)


class ReplyControlFlags:
    """Exact existing reader-lane Events, scoped to one Worker turn/request."""
    def __init__(self, brain, path, turn_id, sessions, key, session, cancellation):
        self.brain, self.path, self.turn_id = brain, path, turn_id
        self.sessions, self.key, self.session = sessions, key, session
        self.events = (cancellation, session.requested, session.suspension_requested)
        self.active = True

    def current(self):
        return (self.active and current_owned_path() is self.path
            and self.sessions.get(self.key) is self.session
            and self.session.brain_id == self.brain.brain_id
            and self.session.stream_id == self.turn_id
            and self.session.requested is self.events[1]
            and self.session.suspension_requested is self.events[2]
            and getattr(self.brain, "_native_reply_private_flags", None) is self
            and all(type(flag) is _EVENT_TYPE for flag in self.events))

    def is_set(self):
        return not self.current() or any(flag.is_set() for flag in self.events)


def bind_reply_controls(brain, request_id, turn_id, *, sessions, key, session, cancellation):
    """Worker-only binding; no new Event, clear/reset or inherited authority."""
    path = current_owned_path()
    if (path is None or path.operation != "chat" or path.request_id != str(request_id or "")
            or path.scheduler.ops.brain is not brain or session is None
            or sessions.get(key) is not session or key != (brain.brain_id, turn_id)
            or session.brain_id != brain.brain_id or session.stream_id != turn_id
            or any(type(flag) is not _EVENT_TYPE for flag in
                (cancellation, session.requested, session.suspension_requested))):
        return None
    value = ReplyControlFlags(brain, path, turn_id, sessions, key, session, cancellation)
    brain._native_reply_private_flags = value
    return value


def retire_reply_controls(brain, value):
    if value is None: return
    value.active = False
    if getattr(brain, "_native_reply_private_flags", None) is value:
        del brain._native_reply_private_flags


class ReplyExecutionPlanner:
    def __init__(self, brain, path):
        self.brain, self.path, self.closed = brain, path, False
        self.decoder = brain.decoder
        self.controller = getattr(brain, "_native_operation_auto", None)
        self.flags = getattr(brain, "_native_reply_private_flags", None)
        self.steps, self.stages, self.disabled = 0, {}, None

    def current(self):
        try:
            return (not self.closed and current_owned_path() is self.path and not self.path.failed
                and not self.path.disabled and self.path.scheduler.ops.brain is self.brain
                and self.brain.decoder is self.decoder
                and getattr(self.brain, "_native_operation_auto", None) is self.controller
                and self.path.scheduler.ops.armed()
                and self.path.scheduler.ops.binding() == self.path.binding)
        except (AttributeError, KeyError, RuntimeError, TypeError): return False

    def private_allowed(self):
        # Only concrete request-local Events may be read in private trials.
        # An arbitrary cancel/control callback can tick tools, persist receipts
        # or publish results; it must not be called speculatively.
        return (not self.disabled and self.current() and type(self.flags) is ReplyControlFlags
            and self.flags.current() and not self.flags.is_set())

    def private_stopped(self):
        # Called inside actual private kernel loops: fields and trusted Events
        # only. Existing native current() fences verify policy/frontiers around
        # each trial; do not poll resource status or opaque callbacks here.
        return bool(self.closed or self.disabled or current_owned_path() is not self.path
            or self.brain.decoder is not self.decoder
            or getattr(self.brain, "_native_operation_auto", None) is not self.controller
            or type(self.flags) is not ReplyControlFlags or self.flags.is_set())

    @contextmanager
    def stage(self, name, *, pinned=True):
        if name not in _STAGES:
            # An unknown schedule is ordinary, never new qualification authority.
            self.disabled = "unknown-reply-stage"
            with world_effect_barrier(): yield
            return
        if PRIVATE_OPERATION.get():
            raise RuntimeError("private operation cannot replay reply orchestration")
        if not self.current():
            self.disabled = "reply-owner-or-policy-retired"
            with world_effect_barrier(): yield
            return
        started = self.path.scheduler.clock()
        failed = False
        try:
            if pinned:
                # Actions, arguments, router/Ponder and final learning cannot
                # currently migrate as one isolated unit. Admit/account for
                # them on their actual owner, and execute their effects once.
                with owned_phase(self.brain, "reply-" + name), world_effect_barrier(): yield
            else:
                # Complete routing forward has its own exact native consumer;
                # do not nest it inside a fixed-prefix timing or double count it.
                self.path.scheduler.ops.admit_effect("reply-" + name)
                yield
        except BaseException:
            failed = True
            raise
        finally:
            row = self.stages.setdefault(name, {"calls": 0, "failed": 0, "nanoseconds": "0", "placement": "actual-owner" if pinned else "complete-native-consumer"})
            row["calls"] += 1
            row["failed"] += failed
            row["nanoseconds"] = str(int(row["nanoseconds"]) + max(0, self.path.scheduler.clock() - started))

    def sample(self, decoder, *, use_cache, resumed, allowed, window, cue, cache,
               cached, prefix, one_shot, ordinary, ordinary_after_boundary, kwargs):
        self.steps += 1
        # The caller's explicit full-forward arithmetic is never exchanged for
        # cached reductions. Best Work private branches and unknown/resumed
        # schedules keep their existing ordinary route.
        if (resumed or decoder is not self.brain.decoder
                or self.disabled or not self.private_allowed()):
            with world_effect_barrier(): return ordinary()
        if not use_cache: return ordinary()
        if not allowed:
            with world_effect_barrier(): return ordinary()
        args = {**kwargs, "flags_cancelled": self.private_stopped,
            "fallback": ordinary, "fallback_after_boundary": ordinary_after_boundary}
        ops = self.controller.ops
        # Admission/equality/device choice still consumes measured complete
        # operation receipts in these existing consumers, not primitive timings.
        try:
            retained = cache is not None and ops.cached_operations.valid_frontier(window, cue, cache)
            refill = not retained and ops.prefill_operations.valid_prefix(window, cue, cache)
        except (AttributeError, KeyError, TypeError, RuntimeError, ValueError):
            with world_effect_barrier(): return ordinary()
        if retained:
            return cached(decoder, allowed=True, **args)
        if refill:
            return prefix(decoder, allowed=True, **args)
        # A manually armed one-shot path remains available only under its own
        # unchanged exact review. No Auto pool or inference is invented here.
        return one_shot(decoder, allowed=allowed, **args)


def current_reply_planner(owner=None):
    value = _REPLY.get()
    if value is None or value.closed: return None
    return value if owner is None or owner is value.brain or owner is value.brain.decoder else None


@contextmanager
def reply_stage(owner, name, *, pinned=True):
    value = current_reply_planner(owner)
    if value is None:
        yield
    else:
        with value.stage(name, pinned=pinned): yield


def owned_reply(method):
    signature = inspect.signature(method)
    @functools.wraps(method)
    def wrapped(brain, *args, **kwargs):
        if PRIVATE_OPERATION.get():
            raise RuntimeError("private operation cannot execute a chat request")
        bound = signature.bind(brain, *args, **kwargs)
        for name in _CALLBACKS:
            if name in bound.arguments: bound.arguments[name] = world_callback(bound.arguments[name])
        path = current_owned_path()
        planner = None
        if path is not None and path.operation == "chat":
            try:
                if path.scheduler.ops.brain is brain: planner = ReplyExecutionPlanner(brain, path)
            except (AttributeError, RuntimeError): pass
        token = _REPLY.set(planner)
        try:
            if planner is None: return method(*bound.args, **bound.kwargs)
            with request_qualification_gate(planner.private_allowed):
                return method(*bound.args, **bound.kwargs)
        finally:
            _REPLY.reset(token)
            if planner is not None:
                planner.closed = True
                path.reply = {"tokenFrontiersDispatched": planner.steps, "stages": planner.stages,
                    "metadataIssue": planner.disabled,
                    "placementBasis": "existing-exact-complete-operation-selection-and-pinned-owner-phases",
                    "externalCallbacksPrivatelyReplayed": False, "wholeReplySpeedupMeasured": False}
    return wrapped


def dispatch_reply_sample(decoder, *, use_cache, resumed, allowed, window, cue, cache,
                          cached, prefix, one_shot, ordinary, ordinary_after_boundary, **kwargs):
    """Choose a real compatible consumer; no saved output or tool is replayed."""
    if PRIVATE_OPERATION.get():
        raise RuntimeError("private operation cannot replay a generation frontier")
    shared = dict(window=window, cue=cue, cache=cache, **kwargs)
    planner = current_reply_planner()
    if planner is not None:
        return planner.sample(decoder, use_cache=use_cache, resumed=resumed, allowed=allowed,
            window=window, cue=cue, cache=cache, cached=cached, prefix=prefix, one_shot=one_shot,
            ordinary=ordinary, ordinary_after_boundary=ordinary_after_boundary, kwargs=shared)
    if not use_cache: return ordinary()
    # Preserve standalone/manual behavior exactly without granting a reply plan.
    return prefix(decoder, allowed=allowed and not resumed, **shared,
        fallback=lambda: cached(decoder, allowed=allowed and not resumed, **shared,
            fallback=lambda: one_shot(decoder, allowed=allowed, **shared,
                fallback=ordinary, fallback_after_boundary=ordinary_after_boundary),
            fallback_after_boundary=ordinary_after_boundary),
        fallback_after_boundary=ordinary_after_boundary)
