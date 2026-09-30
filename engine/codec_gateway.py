"""Correlated, first-use codec setup outside the serial neural RPC queue.

No brain/model imports, installer commands or renderer-provided executable paths.
The stdio reader only deposits a scoped receipt; the owner reverifies/leases it.
"""
import os
import copy
import re
import shutil
import threading
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from video_runtime import configure_verified_video_runtime, VideoRuntimeConfigurationCancelled


class CodecRuntimeCancelled(RuntimeError):
    pass


class CodecRuntimeUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class CodecOwner:
    request_id: str
    brain_id: str
    job_id: str = ""
    stream_id: str = ""
    action_id: str = ""

    def fields(self):
        return {"requestId": self.request_id, "brainId": self.brain_id,
            "jobId": self.job_id, "streamId": self.stream_id, "actionId": self.action_id}


_SCOPE = ContextVar("codec_runtime_owner", default=None)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
CODEC_PURPOSES = frozenset({"decode-live-audio", "decode-audio", "decode-video", "inspect-video-audio", "encode-video"})


def existing_external_executable(environment):
    explicit = environment.get("IMAGEIO_FFMPEG_EXE", "")
    candidate = explicit or shutil.which("ffmpeg", path=environment.get("PATH", ""))
    if not candidate:
        return None
    path = Path(candidate).resolve(strict=True)
    if not path.is_file() or (os.name != "nt" and not os.access(path, os.X_OK)):
        raise CodecRuntimeUnavailable("The selected external FFmpeg is not an accessible executable")
    return str(path)


class CodecRuntimeGateway:
    def __init__(self, notify, *, environment=None, resolve_external=None, verify=None):
        self.environment = os.environ if environment is None else environment
        self.notify = notify
        self.resolve_external = resolve_external or existing_external_executable
        self.verify = verify or configure_verified_video_runtime
        self.condition = threading.Condition(threading.RLock())
        self.challenges = {}
        self.leases = {}
        self.selected_path = ""
        self.selected_identity = None
        self.selected_configuration = None

    @contextmanager
    def scope(self, owner, cancelled):
        token = _SCOPE.set((self, owner, cancelled))
        try:
            yield
        finally:
            _SCOPE.reset(token)

    def current_owner(self):
        scope = _SCOPE.get()
        return scope[1] if scope is not None and scope[0] is self else None

    def resolve(self, params):
        required = {"challengeId", "requestId", "brainId", "jobId", "streamId", "actionId", "outcome"}
        outcome = params.get("outcome")
        allowed = required | ({"configuration"} if outcome == "ready" else {"reason"})
        if set(params) != allowed or outcome not in {"ready", "external", "failed", "cancelled"}:
            raise ValueError("invalid codec control receipt fields")
        challenge_id = params.get("challengeId")
        with self.condition:
            challenge = self.challenges.get(challenge_id)
            if challenge is None or any(params.get(key) != value for key, value in challenge["owner"].fields().items()):
                raise ValueError("codec receipt does not own this live challenge")
            if challenge["receipt"] is not None and outcome != "cancelled":
                raise ValueError("codec challenge already has a receipt")
            if outcome == "ready":
                configuration = params.get("configuration")
                if not isinstance(configuration, dict) or set(configuration) != {"executablePath", "artifactSha256", "binarySha256", "binarySizeBytes", "target"}:
                    raise ValueError("codec receipt lacks a pinned runtime configuration")
            elif not isinstance(params.get("reason"), str) or len(params["reason"]) > 4000:
                raise ValueError("codec receipt reason is invalid")
            challenge["receipt"] = copy.deepcopy(params)
            if outcome == "cancelled":
                challenge["cancelled"] = True
            self.condition.notify_all()
        return {"acknowledged": True, "challengeId": challenge_id}

    def configure(self, params, cancelled=lambda: False):
        # Verify into a private environment first. No global selector changes
        # until every actual old codec lease (not image decoder) is released.
        scratch = dict(self.environment)
        try:
            result = self.verify(params, environment=scratch, cancelled=cancelled)
        except VideoRuntimeConfigurationCancelled as error:
            raise CodecRuntimeCancelled(str(error)) from error
        identity = (params["executablePath"], params["binarySha256"])
        with self.condition:
            while self.leases and identity != self.selected_identity:
                if cancelled():
                    raise CodecRuntimeCancelled("codec selection was cancelled while awaiting active codec leases")
                self.condition.wait(0.05)
            if cancelled():
                raise CodecRuntimeCancelled("codec selection was cancelled before publication")
            self.selected_path = params["executablePath"]
            self.selected_identity = identity
            self.selected_configuration = copy.deepcopy(params)
            self.environment["IMAGEIO_FFMPEG_EXE"] = self.selected_path
            self.condition.notify_all()
        return result

    @contextmanager
    def lease(self, brain_id, purpose):
        scope = _SCOPE.get()
        if scope is None or scope[0] is not self:
            raise CodecRuntimeUnavailable("codec use has no trusted worker ownership scope")
        _gateway, owner, cancelled = scope
        effective_cancelled = cancelled
        if owner.brain_id != brain_id or not _ID.fullmatch(owner.request_id) or not _ID.fullmatch(brain_id) or purpose not in CODEC_PURPOSES or \
                any(not isinstance(value, str) or (value and not _ID.fullmatch(value)) for value in owner.fields().values()):
            raise CodecRuntimeUnavailable("codec use has invalid request/brain/purpose ownership")
        if cancelled():
            raise CodecRuntimeCancelled("codec use was cancelled before setup")
        with self.condition:
            if not self.selected_path:
                external = self.resolve_external(self.environment)
                if external:
                    self.selected_path, self.selected_identity = external, (external, "external")
            path = self.selected_path
        if not path:
            challenge_id = uuid.uuid4().hex
            challenge = {"owner": owner, "receipt": None, "cancelled": False}
            with self.condition:
                self.challenges[challenge_id] = challenge
            fields = {"challengeId": challenge_id, **owner.fields()}
            self.notify("codec-runtime-needed", owner, {**fields, "purpose": purpose})
            is_cancelled = lambda: cancelled() or challenge["cancelled"]
            effective_cancelled = is_cancelled
            try:
                with self.condition:
                    while challenge["receipt"] is None:
                        if is_cancelled():
                            raise CodecRuntimeCancelled("owned codec setup was cancelled")
                        self.condition.wait(0.05)
                    receipt = challenge["receipt"]
                if is_cancelled() or receipt["outcome"] == "cancelled":
                    raise CodecRuntimeCancelled("owned codec setup was cancelled")
                if receipt["outcome"] == "ready":
                    self.configure(receipt["configuration"], is_cancelled)
                elif receipt["outcome"] == "external":
                    external = self.resolve_external(self.environment)
                    if not external:
                        raise CodecRuntimeUnavailable("main found external FFmpeg but this worker has no accessible external executable")
                    with self.condition:
                        if not self.selected_path:
                            self.selected_path, self.selected_identity = external, (external, "external")
                else:
                    raise CodecRuntimeUnavailable(receipt["reason"] or "owned codec setup did not complete")
            finally:
                with self.condition:
                    self.challenges.pop(challenge_id, None)
                self.notify("codec-runtime-released", owner, fields)
        with self.condition:
            if effective_cancelled():
                raise CodecRuntimeCancelled("codec use was cancelled before its side effect")
            path = self.selected_path
            self.leases[path] = self.leases.get(path, 0) + 1
        try:
            if self.selected_configuration is not None:
                # Reverify under the actual lease immediately before execution;
                # a modified cached binary is never selected on an old receipt.
                try:
                    self.verify(self.selected_configuration, environment=dict(self.environment), cancelled=effective_cancelled)
                except VideoRuntimeConfigurationCancelled as error:
                    raise CodecRuntimeCancelled(str(error)) from error
            yield path
        finally:
            with self.condition:
                self.leases[path] -= 1
                if not self.leases[path]:
                    del self.leases[path]
                self.condition.notify_all()


@contextmanager
def codec_executable(brain_id, purpose):
    scope = _SCOPE.get()
    if scope is not None:
        with scope[0].lease(brain_id, purpose) as executable:
            yield executable
    else:
        # Standalone research callers retain their existing configured runtime;
        # automatic setup authority only exists in a correlated desktop scope.
        import imageio_ffmpeg
        yield imageio_ffmpeg.get_ffmpeg_exe()
