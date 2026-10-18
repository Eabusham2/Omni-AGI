"""Actual prepared-media boundary and private, typed effect routing.

An absent/unarmed controller executes the original prepared producer once.
Renderer/server data cannot install a scope or choose a private output path.
No neural work, filesystem access or model initialization occurs on import.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

_SCOPE = ContextVar("native_whole_media_artifact_scope", default=None)


@contextmanager
def media_artifact_scope(scope):
    token = _SCOPE.set(scope)
    try: yield scope
    finally: _SCOPE.reset(token)


def measured_media_artifact(brain, prepared, producer, preview, directory, cancelled, started_at):
    current = _SCOPE.get()
    if current is not None: raise RuntimeError("nested whole-media publication boundary is unsupported")
    controller = getattr(brain, "_native_operation_auto", None)
    if controller is None or getattr(controller, "consume_media", None) is None:
        return producer(prepared["idea"], preview, directory, cancelled, started_at)
    return controller.consume_media(brain, prepared, producer, preview, directory, cancelled, started_at)


def media_artifact_directory(brain, owned):
    scope = _SCOPE.get()
    if scope is not None and scope.brain() is brain and scope.private:
        return scope.artifact_directory()
    from spontaneous_output import checked_output_directory
    return checked_output_directory(brain.engine_path, owned)


def observe_media_preview(brain, progress, neural, mime_type, payload, details):
    scope = _SCOPE.get()
    if scope is not None and scope.brain() is brain: scope.preview(progress, neural, mime_type, payload, details)


def observe_media_output(brain, role, neural):
    scope = _SCOPE.get()
    if scope is not None and scope.brain() is brain: scope.output(role, neural)


def media_resource_pause(brain, value):
    scope = _SCOPE.get()
    if scope is not None and scope.brain() is brain and scope.private:
        scope.resource_paused = True
    else:
        brain.resource_pause = value


@contextmanager
def media_payload_write(path, byte_count):
    scope = _SCOPE.get()
    if scope is None or not scope.private:
        yield
    else:
        with scope.payload_write(path, byte_count): yield
