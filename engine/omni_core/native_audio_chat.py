"""Typed same-brain audio conversation input. No transcript, caption or prompt.

The attachment is resolved by trusted main code. Native verifies its immutable
bytes before complete sensory traversal; ordinary admission is the only learner.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .mcp_sampling_media import AUDIO_MIMES, _audio_windows
from .parameter_diagnostics import native_parameter_read_only
from .working_attention_paging import WorkingAttentionCancelled

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_CHUNK = 65536


def _id(value):
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= (1 << 53) - 1


def audio_identity_sha256(value):
    body = {key: value[key] for key in ("format", "version", "attachment", "capture")}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def validate_native_audio_input(value, *, brain_id=None, turn_id=None):
    keys = {"format", "version", "attachment", "capture", "inputSha256"}
    if not isinstance(value, dict) or set(value) != keys or value.get("format") != "omni-native-audio-input" or type(value.get("version")) is not int or value["version"] != 1:
        raise ValueError("native audio requires its exact trusted typed identity")
    attachment, capture = value["attachment"], value["capture"]
    if (not isinstance(attachment, dict) or set(attachment) != {"brainId", "turnId", "attachmentId", "contentSha256", "bytes", "mimeType", "kind", "relativePath", "displayName", "manifestSha256"}
            or not isinstance(capture, dict) or set(capture) != {"sessionId", "utteranceId", "sequence", "capturedAt", "permissionRevision"}):
        raise ValueError("native audio attachment or capture identity is malformed")
    if not all(_id(attachment[name]) for name in ("brainId", "turnId", "attachmentId")) or not all(_id(capture[name]) for name in ("sessionId", "utteranceId")):
        raise ValueError("native audio source identity is invalid")
    if any(not isinstance(attachment[name], str) or not _SHA.fullmatch(attachment[name]) for name in ("contentSha256", "manifestSha256")):
        raise ValueError("native audio source checksum is invalid")
    if (not _integer(attachment["bytes"], 1) or attachment["kind"] != "audio" or attachment["mimeType"] not in AUDIO_MIMES
            or attachment["relativePath"] != "attachments/blobs/" + attachment["contentSha256"] + ".data"
            or not isinstance(attachment["displayName"], str) or "\0" in attachment["displayName"]
            or not _integer(capture["sequence"]) or not _integer(capture["permissionRevision"]) or not isinstance(capture["capturedAt"], str)):
        raise ValueError("native audio source codec, bounds or session identity is invalid")
    from datetime import datetime
    try:
        datetime.fromisoformat(capture["capturedAt"].replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("native audio capture timestamp is invalid") from error
    if brain_id is not None and attachment["brainId"] != brain_id or turn_id is not None and attachment["turnId"] != turn_id:
        raise ValueError("native audio input belongs to another brain or turn")
    if value["inputSha256"] != audio_identity_sha256(value):
        raise ValueError("native audio input lost its typed identity checksum")
    return copy.deepcopy(value)


def validate_native_audio_receipt(value, *, input_sha256=None):
    keys = {"input", "assemblyId", "sourceBytes", "sourceSha256", "decodedWindows", "modelSamples", "modelSampleRate", "wholeInputTraversed", "nativeSensory", "transcriptUsed"}
    if not isinstance(value, dict) or set(value) - keys - {"sourceSampleFrames", "sourceSampleRate"} or not keys <= set(value):
        raise ValueError("native audio receipt is malformed")
    identity = validate_native_audio_input(value["input"])
    if (not _id(value["assemblyId"]) or value["sourceBytes"] != identity["attachment"]["bytes"] or value["sourceSha256"] != identity["attachment"]["contentSha256"]
            or any(not _integer(value[name], 1) for name in ("decodedWindows", "modelSamples", "modelSampleRate"))
            or any(name in value and not _integer(value[name], 1) for name in ("sourceSampleFrames", "sourceSampleRate"))
            or value["wholeInputTraversed"] is not True or value["nativeSensory"] is not True or value["transcriptUsed"] is not False
            or input_sha256 is not None and identity["inputSha256"] != input_sha256):
        raise ValueError("native audio receipt lost its full sensory coverage or exact input binding")
    return copy.deepcopy(value)


def observed_input_sha256(message):
    receipt = message.get("native_audio_input", message.get("nativeAudioInput"))
    if receipt is not None:
        if message.get("content") != "":
            raise ValueError("native audio must not contain a fabricated transcript")
        return validate_native_audio_receipt(receipt)["input"]["inputSha256"]
    return hashlib.sha256(str(message.get("content", "")).encode("utf-8")).hexdigest()


class OwnedAudioSource:
    """Verified existing attachment; never unlink or copy the user's raw data."""
    def __init__(self, brain, identity, cancelled):
        self.block = SimpleNamespace(mime_type=identity["attachment"]["mimeType"], decoded_bytes=identity["attachment"]["bytes"])
        root = Path(brain.storage_path)
        relative = Path(identity["attachment"]["relativePath"])
        self.path, self.handle = root / relative, None
        self.sha256 = identity["attachment"]["contentSha256"]
        try:
            for part in (root, root / "attachments", root / "attachments" / "blobs", self.path):
                if part.is_symlink():
                    raise ValueError("native audio source contains a symbolic link")
            descriptor = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            self.handle = os.fdopen(descriptor, "rb")
            opened = os.fstat(self.handle.fileno())
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != self.block.decoded_bytes:
                raise ValueError("native audio source has a different declared size/type")
            self.identity = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            digest = hashlib.sha256()
            with brain.resource_policy.reserve_ram(_CHUNK * 2 + 4096, "native audio immutable source verification"):
                while True:
                    if cancelled():
                        raise InterruptedError("native audio source verification cancelled")
                    piece = self.handle.read(_CHUNK)
                    if not piece:
                        break
                    digest.update(piece)
            if digest.hexdigest() != self.sha256:
                raise ValueError("native audio attachment bytes changed")
            self.handle.seek(0)
            self.verify_path()
        except BaseException:
            self.close()
            raise

    def verify_path(self):
        opened, path = os.fstat(self.handle.fileno()), self.path.lstat()
        if not stat.S_ISREG(path.st_mode) or any((value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) != self.identity for value in (opened, path)):
            raise ValueError("native audio source identity changed during reading")
        return self.path

    def ensure_path(self):
        return self.verify_path()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


@contextmanager
def encode_native_audio(brain, identity, *, cancelled=lambda: False):
    """Read every actual sample into one bounded shared-space sensory cue."""
    identity = validate_native_audio_input(identity, brain_id=brain.brain_id)
    if not brain.config.audio_enabled:
        raise ValueError("this brain's native audio region is disabled")
    dimensions = brain.config.vsa_dim
    receipt = {"sourceBytes": identity["attachment"]["bytes"], "sourceSha256": identity["attachment"]["contentSha256"],
               "decodedWindows": 0, "wholeInputTraversed": False, "nativeSensory": False, "transcriptUsed": False}
    source = OwnedAudioSource(brain, identity, cancelled)
    modes = [(module, module.training) for module in brain.modalities.modules()]
    pager = getattr(brain, "working_attention_pager", None)
    def check():
        if cancelled():
            raise InterruptedError("native audio input was cancelled before admission")
        pressure = getattr(brain.resource_policy, "check_active_pressure", None)
        if callable(pressure):
            pressure("native audio chat input")
    try:
        with brain.resource_policy.reserve_ram(dimensions * 48 + 131072, "native audio shared-space accumulator"):
            total = torch.zeros(dimensions, dtype=torch.float32)
            correction = torch.zeros_like(total)
            brain.modalities.eval()
            with torch.no_grad(), native_parameter_read_only(), pager.activate(cancelled) if pager is not None else nullcontext():
                windows = _audio_windows(brain, source, receipt, cancelled)
                try:
                    for index, (tensor, elapsed) in enumerate(windows):
                        check()
                        scratch = int(brain.config.audio_samples) * 128 + int(brain.config.idea_dim) * 64 + dimensions * 64 + 131072
                        with brain.resource_policy.reserve_ram(scratch, "native audio perception forward"):
                            embedding = brain.modalities.perception_embedding("audio", tensor)
                            if embedding.shape != (1, brain.config.idea_dim) or not bool(torch.isfinite(embedding).all()):
                                raise ValueError("native audio perception returned invalid shared activity")
                            vector = brain._sensory_substrate_vector(embedding, "audio", source.sha256)
                            position = brain.memory.space.symbol("native-audio-window:%d" % index)
                            bound = brain.memory.space.permute(brain.memory.space.bind(vector, position), index + 1)
                            increment = bound * max(1e-6, float(elapsed)) - correction
                            updated = total + increment
                            correction = (updated - total) - increment
                            total = updated
                finally:
                    windows.close()
                check()
                source.verify_path()
                if receipt["decodedWindows"] < 1 or receipt.get("modelSamples", 0) < 1 or not bool(torch.isfinite(total).all()) or float(total.norm()) <= 1e-8:
                    raise ValueError("native audio produced no fully traversed sensory activity")
                receipt.update(wholeInputTraversed=True, nativeSensory=True)
                cue = F.normalize(total.reshape(1, -1), dim=-1)[0]
            for module, mode in modes:
                module.training = mode
            yield cue, receipt
    except WorkingAttentionCancelled as error:
        if cancelled():
            raise InterruptedError("native audio read stopped at its safe boundary") from error
        raise
    finally:
        for module, mode in modes:
            module.training = mode
        source.close()


def audio_preview_experience(brain, cue, recalled, *, assembly_id="", settling=None, novelty=None):
    # Use the ordinary recurrent feature calculation, but remove its text-only
    # inspection placeholders; no fabricated text labels enter learned memory.
    preview = brain._preview_chat_experience("", cue, recalled)
    preview.update(idea_id=assembly_id, assembly_id=assembly_id, concept_ids=[], neuron_ids=[], labels=[], memory_settling=settling or {})
    if novelty is not None:
        preview["novelty"] = float(novelty)
    return preview


def admit_native_audio_experience(brain, identity, cue, decoded, recalled):
    admitted = brain._admit_sensory_embedding(cue, kind="audio-conversation", source_name="microphone",
        fingerprint="native-audio:" + identity["inputSha256"], substrate_space=True, experience_source="conversation-audio", importance=0.7)
    if getattr(brain, "_connected_regions", None) is not None:
        brain.decoder.observe_experience(brain._idea_model_vector(admitted["vector"]).detach(),
            event_key="native-audio:" + identity["inputSha256"], importance=0.7)
    receipt = validate_native_audio_receipt({"input": identity, "assemblyId": admitted["assemblyId"], **decoded}, input_sha256=identity["inputSha256"])
    preview = audio_preview_experience(brain, admitted["vector"], recalled, assembly_id=admitted["assemblyId"],
        settling=admitted["memorySettling"], novelty=admitted["novelty"])
    return preview, receipt


def accepted_audio_cue(brain, accepted):
    receipt = validate_native_audio_receipt(accepted.get("nativeAudioInput"), input_sha256=accepted["inputSha256"])
    assembly_id = receipt["assemblyId"]
    if assembly_id not in brain.memory.neuron_vectors:
        raise RuntimeError("accepted native audio lost its actual packed sensory assembly")
    return brain.memory.neuron_vectors[assembly_id], receipt
