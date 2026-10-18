"""Private, admitted MCP sensory input; no ingestion, captions or learning."""
from __future__ import annotations

import base64
import hashlib
import io
import os
import queue
import re
import stat
import subprocess
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .offload import NeuralStateResourcePause
from .speech_training import StreamingSpeechResampler


IMAGE_MIMES = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP", "image/gif": "GIF"}
AUDIO_MIMES = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/flac": "flac",
               "audio/x-flac": "flac", "audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/mp3": "mp3",
               "audio/aac": "aac", "audio/mp4": "mov", "audio/x-m4a": "mov"}
_BASE64 = re.compile(r"[A-Za-z0-9+/]*={0,2}\Z")
_CHUNK = 64 * 1024  # Physical copy window, not an input-size limit.


@dataclass(frozen=True)
class NativeSamplingMedia:
    message_index: int
    block_index: int
    kind: str
    mime_type: str
    data: str
    decoded_bytes: int


def media_block(value, message_index, block_index):
    if (not isinstance(value, dict) or set(value) != {"type", "data", "mimeType"}
            or not isinstance(value.get("type"), str) or value["type"] not in {"image", "audio"}):
        raise ValueError("native sampling supports typed inline text, image or audio blocks only")
    kind, mime, data = value["type"], value["mimeType"], value["data"]
    allowed = IMAGE_MIMES if kind == "image" else AUDIO_MIMES
    if not isinstance(mime, str) or mime.lower() not in allowed:
        raise ValueError("unsupported native MCP %s MIME type" % kind)
    if (not isinstance(data, str) or not data or len(data) % 4 or _BASE64.fullmatch(data) is None
            or base64.b64encode(base64.b64decode(data[-4:], validate=True)).decode("ascii") != data[-4:]):
        raise ValueError("native sampling media must be canonical inline base64")
    padding = 2 if data.endswith("==") else 1 if data.endswith("=") else 0
    return NativeSamplingMedia(message_index, block_index, kind, mime, data, len(data) // 4 * 3 - padding)


def _check(brain, cancelled):
    if cancelled():
        raise InterruptedError("native MCP media input was interrupted")
    pressure = getattr(brain.resource_policy, "check_active_pressure", None)
    if callable(pressure):
        pressure("private MCP media decoding")


class PrivateMediaSource:
    """RAM first, with session-owned authorized spill when actually necessary."""
    def __init__(self, brain, block, cancelled):
        self.brain, self.block, self.cancelled = brain, block, cancelled
        self.handle = self.path = self.ram = self.disk = self.copy_ram = None
        self.file_identity = None
        self.sha256 = ""
        try:
            try:
                self.ram = brain.resource_policy.reserve_ram(block.decoded_bytes * 2 + _CHUNK * 2,
                                                            "private MCP source RAM buffer")
                self.handle = io.BytesIO()
            except NeuralStateResourcePause:
                self._new_disk_source()
            digest, written = hashlib.sha256(), 0
            for offset in range(0, len(block.data), _CHUNK):
                _check(brain, cancelled)
                piece = base64.b64decode(block.data[offset:offset + _CHUNK], validate=True)
                digest.update(piece)
                if self.handle.write(piece) != len(piece):
                    raise OSError("private MCP source had a short write")
                written += len(piece)
            if written != block.decoded_bytes:
                raise ValueError("native MCP source bytes do not match reviewed base64")
            self.sha256 = digest.hexdigest()
            self.handle.flush(); self.handle.seek(0)
            if self.disk is not None:
                self.disk.commit(path=self.path)
        except BaseException:
            self.close()
            raise

    def _new_disk_source(self):
        _check(self.brain, self.cancelled)
        policy = self.brain.resource_policy
        self.copy_ram = policy.reserve_ram(_CHUNK * 3 + 8192, "private MCP source spill copy window")
        self.disk = policy.allocate_spill_file(self.block.decoded_bytes + 4096,
            "private reviewed MCP media source", ".mcp-media", cancelled=lambda: _check(self.brain, self.cancelled))
        if self.disk is not None:
            self.path = Path(self.disk.path)
            self.handle = os.fdopen(self.disk.open(os.O_RDWR), "w+b")
        else:
            self.disk = policy.reserve_spill(self.block.decoded_bytes + 4096, "private reviewed MCP media source")
            directory = Path(self.brain._live_paging_cache_directory) / "mcp-media"
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / (uuid.uuid4().hex + ".mcp-media")
            self.disk.bind_path(self.path)
            descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            self.handle = os.fdopen(descriptor, "w+b")
        opened = os.fstat(self.handle.fileno())
        self.file_identity = (opened.st_dev, opened.st_ino)

    def verify_path(self):
        """The codec may reopen only our still-identical regular source."""
        if self.path is None or self.file_identity is None:
            raise RuntimeError("private MCP codec source is not owned")
        value = self.path.lstat()
        opened = os.fstat(self.handle.fileno())
        if (not stat.S_ISREG(value.st_mode) or (value.st_dev, value.st_ino) != self.file_identity
                or (opened.st_dev, opened.st_ino) != self.file_identity
                or value.st_size != self.block.decoded_bytes):
            raise RuntimeError("private MCP codec source identity changed")
        return self.path

    def ensure_path(self):
        """A seekable codec source, never a server-supplied pathname/URL."""
        if self.path is not None:
            return self.verify_path()
        previous, self.handle = self.handle, None
        try:
            self._new_disk_source()
            previous.seek(0)
            while True:
                _check(self.brain, self.cancelled)
                piece = previous.read(_CHUNK)
                if not piece:
                    break
                if self.handle.write(piece) != len(piece):
                    raise OSError("private MCP codec source had a short write")
            self.handle.flush(); self.handle.seek(0)
            self.disk.commit(path=self.path)
        finally:
            previous.close()
            if self.ram is not None:
                self.ram.release(); self.ram = None
        return self.verify_path()

    def close(self):
        try:
            if self.handle is not None:
                self.handle.close(); self.handle = None
        finally:
            try:
                if self.path is not None and self.file_identity is not None:
                    try:
                        value = self.path.lstat()
                    except FileNotFoundError:
                        pass
                    else:
                        if not stat.S_ISREG(value.st_mode) or (value.st_dev, value.st_ino) != self.file_identity:
                            raise RuntimeError("private MCP source was replaced; replacement was not deleted")
                        self.path.unlink()
            finally:
                for name in ("ram", "copy_ram", "disk"):
                    lease = getattr(self, name)
                    if lease is not None:
                        lease.release(); setattr(self, name, None)


@contextmanager
def private_media_source(brain, block, cancelled):
    source = PrivateMediaSource(brain, block, cancelled)
    try:
        yield source
    finally:
        source.close()


def _signature(source, expected):
    source.handle.seek(0); header = source.handle.read(32); source.handle.seek(0)
    image = (expected == "PNG" and header.startswith(b"\x89PNG\r\n\x1a\n") or
        expected == "JPEG" and header.startswith(b"\xff\xd8\xff") or
        expected == "WEBP" and header[:4] == b"RIFF" and header[8:12] == b"WEBP" or
        expected == "GIF" and header[:6] in {b"GIF87a", b"GIF89a"})
    audio = (expected == "wav" and header[:4] in {b"RIFF", b"RIFX", b"RF64"} and header[8:12] == b"WAVE" or
        expected == "flac" and header.startswith(b"fLaC") or expected == "ogg" and header.startswith(b"OggS") or
        expected == "mp3" and (header.startswith(b"ID3") or len(header) > 1 and header[0] == 255 and header[1] & 0xE0 == 0xE0 and header[1] & 6 != 0) or
        expected == "aac" and len(header) > 1 and header[0] == 255 and header[1] & 0xF6 == 0xF0 or
        expected == "mov" and header[4:8] == b"ftyp")
    if not image and not audio:
        raise ValueError("native MCP media MIME type does not match its container signature")
    if expected == "wav" and header[:4] == b"RIFF" and int.from_bytes(header[4:8], "little") + 8 > source.block.decoded_bytes:
        raise ValueError("native MCP WAV is truncated before its declared container end")


def _image_windows(brain, source, receipt, cancelled):
    from PIL import Image, ImageOps
    import numpy as np

    expected = IMAGE_MIMES[source.block.mime_type.lower()]
    _signature(source, expected)
    with Image.open(source.handle, formats=[expected]) as image:
        if image.format != expected or any(type(size) is not int or size < 1 for size in image.size):
            raise ValueError("native MCP image has an invalid declared size/format")
        width, height = image.size
        # Before pixel decode/conversion: current source frame, composition,
        # transpose/resize intermediates and final native-sized float tensor.
        scratch = width * height * 64 + brain.config.image_size ** 2 * 128 + _CHUNK
        with brain.resource_policy.reserve_ram(scratch, "private MCP image decoded pixels and native frame"):
            index = 0
            while True:
                _check(brain, cancelled)
                try:
                    image.seek(index)
                except EOFError:
                    break
                if image.size != (width, height):
                    raise ValueError("native MCP animation changed its admitted canvas size")
                frame = ImageOps.exif_transpose(image).convert("RGB").resize((brain.config.image_size, brain.config.image_size))
                values = np.asarray(frame, dtype="float32").copy()
                tensor = torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).to(brain.device) / 127.5 - 1.
                duration = image.info.get("duration")
                elapsed = max(1e-6, float(duration) / 1000.) if isinstance(duration, (int, float)) and duration > 0 else 1.
                receipt["decodedWindows"] += 1; receipt["sourceFrames"] += 1
                yield tensor, elapsed
                index += 1
            if index < 1:
                raise ValueError("native MCP image has no decodable frames")


def _single_ogg_stream(source, brain, cancelled):
    source.handle.seek(0)
    serial = None
    while True:
        _check(brain, cancelled)
        header = source.handle.read(27)
        if not header:
            break
        if len(header) != 27 or header[:4] != b"OggS" or header[4] != 0:
            raise ValueError("native MCP Ogg container is truncated or unsupported")
        current = int.from_bytes(header[14:18], "little")
        if serial is not None and serial != current:
            raise ValueError("multiple or chained Ogg streams are unsupported by native MCP sampling")
        serial = current
        table = source.handle.read(header[26])
        if len(table) != header[26] or len(source.handle.read(sum(table))) != sum(table):
            raise ValueError("native MCP Ogg page is truncated")
    source.handle.seek(0)


def _audio_windows(brain, source, receipt, cancelled):
    expected = AUDIO_MIMES[source.block.mime_type.lower()]
    _signature(source, expected)
    if expected == "ogg":
        _single_ogg_stream(source, brain, cancelled)
    try:
        import soundfile
    except ImportError:
        soundfile = None
    decoded = None
    if soundfile is not None:
        try:
            decoded = soundfile.SoundFile(source.handle, mode="r")
        except (OSError, RuntimeError):
            source.handle.seek(0)
    if decoded is None:
        yield from _ffmpeg_audio_windows(brain, source, receipt, cancelled, expected)
        return
    with decoded:
        rate, channels, declared = int(decoded.samplerate), int(decoded.channels), int(decoded.frames)
        if rate < 1 or channels < 1 or declared < 0:
            raise ValueError("native MCP audio metadata is invalid")
        receipt.update(sourceSampleRate=rate, sourceSampleFrames=0, modelSampleRate=16000, modelSamples=0)
        resampler = StreamingSpeechResampler(rate, 16000, brain.config.audio_samples)
        pending = torch.empty(0, dtype=torch.float32)
        def model_windows(piece):
            nonlocal pending
            pending = torch.cat((pending, piece))
            while pending.numel() >= brain.config.audio_samples:
                values, pending = pending[:brain.config.audio_samples], pending[brain.config.audio_samples:]
                receipt["decodedWindows"] += 1; receipt["modelSamples"] += values.numel()
                yield values.reshape(1, 1, -1).to(brain.device), values.numel() / 16000.
        # One decoded multichannel source block and the signal-processing
        # interpolation buffers. It scales with actual native window geometry.
        with brain.resource_policy.reserve_ram(brain.config.audio_samples * (channels * 16 + 160) + _CHUNK,
                                               "private MCP audio source/resampling window"):
            while True:
                _check(brain, cancelled)
                block = decoded.read(frames=brain.config.audio_samples, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                receipt["sourceSampleFrames"] += int(block.shape[0])
                values = torch.from_numpy(block).mean(dim=1)
                if not bool(torch.isfinite(values).all()):
                    raise ValueError("native MCP decoded audio is nonfinite")
                for piece in resampler.push(values):
                    yield from model_windows(piece)
            if not receipt["sourceSampleFrames"] or receipt["sourceSampleFrames"] != declared:
                raise ValueError("native MCP audio ended before its declared sample coverage")
            for piece in resampler.push(torch.empty(0), final=True):
                yield from model_windows(piece)
            if pending.numel():
                actual = pending.numel()
                receipt["decodedWindows"] += 1; receipt["modelSamples"] += actual
                yield F.pad(pending, (0, brain.config.audio_samples - actual)).reshape(1, 1, -1).to(brain.device), actual / 16000.


def _ffmpeg_audio_windows(brain, source, receipt, cancelled, expected):
    path = source.ensure_path()
    receipt.update(modelSampleRate=16000, modelSamples=0)
    size = brain.config.audio_samples * 4
    # Includes a bounded child decoder allowance; actual family pressure is
    # still monitored while waiting for its I/O and kills the private child.
    budget = 16 * 1024 ** 2 + source.block.decoded_bytes * 4 + size * 64
    inspect = getattr(brain.resource_policy, "status", None)
    if callable(inspect):
        measured = inspect(fresh_memory_sample=True)
        available = measured.get("availableSafeRamBytes")
        selected = measured.get("systemRamBudgetBytes")
        resident = measured.get("admissionResidentMemoryBytes")
        if all(type(value) is int for value in (available, selected, resident)):
            remaining = min(available, max(0, selected - resident - int((measured.get("sharedQuota") or {}).get("ramEscrowBytes", 0))))
            # Per-allocation decoder allowance follows actual remaining
            # capacity; it is not a fixed codec/file-size cap. Keep a quarter
            # for neural window processing and control/drain operations.
            budget = max(budget, remaining * 3 // 4)
    with brain.resource_policy.reserve_ram(budget, "private MCP codec process and PCM window"), \
            brain._video_runtime_executable("decode-audio") as executable:
        arguments = [executable, "-nostdin", "-hide_banner", "-v", "error", "-xerror", "-max_alloc", str(budget),
            "-protocol_whitelist", "file,pipe", "-format_whitelist", expected, "-f", expected]
        if expected == "mov":
            arguments.extend(["-enable_drefs", "0", "-use_absolute_path", "0"])
        arguments.extend(["-i", str(path), "-map", "0:a", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "16000", "-f", "f32le", "pipe:1"])
        process = subprocess.Popen(arguments, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        stopped, blocks = threading.Event(), queue.Queue(maxsize=1)
        def offer(piece):
            while not stopped.is_set():
                try:
                    blocks.put(piece, timeout=.05)
                    return
                except queue.Full:
                    pass
        def reader():
            try:
                while not stopped.is_set():
                    piece = process.stdout.read(size)
                    offer(piece)
                    if not piece:
                        break
            except BaseException as error:
                if not stopped.is_set():
                    offer(error)
        thread = threading.Thread(target=reader, name="omni-private-mcp-codec-reader", daemon=True)
        thread.start()
        pending = bytearray()
        try:
            while True:
                _check(brain, cancelled)
                try:
                    piece = blocks.get(timeout=.05)
                except queue.Empty:
                    continue
                if isinstance(piece, BaseException):
                    raise RuntimeError("private MCP audio codec read failed") from piece
                if piece:
                    pending.extend(piece)
                if len(pending) >= size or not piece and pending:
                    if len(pending) % 4:
                        raise ValueError("private MCP audio codec returned a partial float sample")
                    values = torch.frombuffer(bytearray(pending), dtype=torch.float32).clone()
                    if not bool(torch.isfinite(values).all()):
                        raise ValueError("private MCP audio codec returned nonfinite samples")
                    actual = values.numel()
                    receipt["decodedWindows"] += 1; receipt["modelSamples"] += actual
                    yield F.pad(values, (0, brain.config.audio_samples - actual)).reshape(1, 1, -1).to(brain.device), actual / 16000.
                    pending.clear()
                if not piece:
                    break
            while process.poll() is None:
                _check(brain, cancelled)
                stopped.wait(.02)
            if process.returncode != 0 or receipt["modelSamples"] == 0:
                raise ValueError("native MCP audio codec could not traverse this container (including unsupported multiple audio tracks)")
            source.verify_path()
        finally:
            stopped.set()
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=2)
            thread.join(2)
            if process.stdout is not None and not thread.is_alive():
                process.stdout.close()
            if thread.is_alive():
                raise RuntimeError("private MCP codec reader did not release after process exit")


def sensory_windows(brain, block, cancelled, receipt):
    """Yield every admitted sensory window; only EOF marks full traversal."""
    receipt.update(messageIndex=block.message_index, blockIndex=block.block_index, kind=block.kind,
        mimeType=block.mime_type, sourceBytes=block.decoded_bytes, decodedWindows=0,
        wholeInputTraversed=False, nativeSensory=False)
    with private_media_source(brain, block, cancelled) as source:
        receipt["sourceSha256"] = source.sha256
        if block.kind == "image":
            receipt["sourceFrames"] = 0
            yield from _image_windows(brain, source, receipt, cancelled)
        else:
            yield from _audio_windows(brain, source, receipt, cancelled)
        _check(brain, cancelled)
        receipt["wholeInputTraversed"] = True
