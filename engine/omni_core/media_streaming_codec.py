"""Bounded owned media encoding and file payloads for actual native output.

No codec/model is run on import. File payloads are internal capabilities with
opened-file identity/checksum verification, not renderer/server path authority.
"""
from __future__ import annotations

import binascii
import hashlib
import itertools
import json
import os
import stat
import struct
import subprocess
import threading
import uuid
import zlib
from pathlib import Path

import torch

from .media_canvas import IO_BYTES, MediaCanvasView, _track
from .media_randomness import media_transfer


class EncodedMediaFile:
    def __init__(self, directory, policy, expected_bytes, *, cancelled=lambda: False):
        self.policy, self.cancelled, self.path, self.lease, self.handle = policy, cancelled, None, None, None
        self.byte_count, self.sha256, self.closed, self.complete = 0, None, False, False
        self._digest = hashlib.sha256()
        try:
            self.check()
            placed = policy.allocate_spill_file(expected_bytes, "native encoded media activity", ".activity", cancelled=self.check)
            if placed is not None:
                self.lease, self.path = placed, Path(placed.path); descriptor = placed.open(os.O_RDWR)
            else:
                directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
                self.path = directory / (uuid.uuid4().hex + ".activity")
                self.lease = policy.reserve_spill(expected_bytes, "native encoded media activity")
                self.lease.bind_path(self.path)
                descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            self.handle = os.fdopen(descriptor, "r+b", buffering=0)
            value = os.fstat(self.handle.fileno()); self.identity = (value.st_dev, value.st_ino)
            self.promised = expected_bytes
            self.lease.commit(path=self.path, promised_bytes=expected_bytes, retain_open_backing=True)
            _track(self)
        except BaseException: self.close(); raise
    def check(self):
        if self.closed: raise RuntimeError("owned media payload is closed")
        if self.cancelled(): raise InterruptedError("media encoding stopped at an I/O boundary")
    def verify(self):
        self.check(); value, opened = self.path.lstat(), os.fstat(self.handle.fileno())
        if (not stat.S_ISREG(value.st_mode) or (value.st_dev, value.st_ino) != self.identity
                or (opened.st_dev, opened.st_ino) != self.identity or value.st_size != self.byte_count):
            raise RuntimeError("owned media payload backing changed")
    def write(self, payload):
        self.check()
        if self.complete: raise RuntimeError("encoded media payload is already sealed")
        if not isinstance(payload, bytes): raise TypeError("media encoders write bounded byte chunks")
        self.policy.require_disk(len(payload) + 4096, "native media encoder write")
        if self.byte_count + len(payload) > self.promised: raise RuntimeError("media encoder exceeded its declared storage bound")
        if self.handle.write(payload) != len(payload): raise RuntimeError("media encoder write was incomplete")
        self._digest.update(payload); self.byte_count += len(payload)
    def finish(self):
        self.check(); self.handle.flush(); os.fsync(self.handle.fileno())
        self.sha256, self.complete = self._digest.hexdigest(), True
        self.lease.commit(path=self.path, retain_open_backing=True)
        self.verify(); return self
    def finish_external(self):
        self.check(); current, opened = self.path.lstat(), os.fstat(self.handle.fileno())
        if (not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != self.identity
                or (opened.st_dev, opened.st_ino) != self.identity or current.st_size > self.promised):
            raise RuntimeError("native codec replaced or exceeded its owned output")
        self.byte_count, self._digest = current.st_size, hashlib.sha256()
        with self.policy.reserve_ram(IO_BYTES * 3, "streamed native codec output proof"):
            self.handle.seek(0)
            for block in iter(lambda: self.handle.read(IO_BYTES), b""):
                self.check(); self._digest.update(block)
        return self.finish()
    def __len__(self): return self.byte_count
    def inline_bytes(self, limit):
        if not self.complete: raise RuntimeError("media payload is not sealed")
        if self.byte_count > limit: return None
        self.verify()
        with self.policy.reserve_ram(self.byte_count * 2 + 4096, "bounded media inline fallback"):
            self.handle.seek(0); value = self.handle.read(self.byte_count)
            if len(value) != self.byte_count or hashlib.sha256(value).hexdigest() != self.sha256:
                raise RuntimeError("media inline payload checksum changed")
            return value
    def copy_to(self, destination):
        if not self.complete: raise RuntimeError("media payload is not sealed")
        self.verify(); destination = Path(destination)
        self.policy.require_disk(self.byte_count + 65536, "native media immutable artifact copy")
        temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
        try:
            with self.policy.reserve_ram(IO_BYTES * 3, "streamed native media artifact copy"), temporary.open("xb") as target:
                self.handle.seek(0); digest, count = hashlib.sha256(), 0
                while True:
                    self.check(); block = self.handle.read(IO_BYTES)
                    if not block: break
                    digest.update(block); target.write(block); count += len(block)
                target.flush(); os.fsync(target.fileno())
                if count != self.byte_count or digest.hexdigest() != self.sha256: raise RuntimeError("media artifact copy lost exact bytes")
            self.verify()
            if destination.exists():
                if destination.is_symlink() or destination.stat().st_size != self.byte_count: raise RuntimeError("media artifact collision")
                with destination.open("rb") as existing:
                    digest = hashlib.sha256()
                    for block in iter(lambda: existing.read(IO_BYTES), b""): digest.update(block)
                if digest.hexdigest() != self.sha256: raise RuntimeError("media artifact collision")
            else: os.replace(temporary, destination)
            return destination
        finally: temporary.unlink(missing_ok=True)
    def close(self):
        if self.closed: return
        self.closed = True
        if self.handle is not None: self.handle.close(); self.handle = None
        if self.path is not None:
            try:
                value = self.path.lstat()
                if stat.S_ISREG(value.st_mode) and (value.st_dev, value.st_ino) == getattr(self, "identity", None): self.path.unlink()
            except FileNotFoundError: pass
        if self.lease is not None: self.lease.backing_closed(); self.lease = None


class TensorMediaView:
    def __init__(self, value, modality):
        if not isinstance(value, torch.Tensor): raise TypeError("native media output must be a tensor or owned canvas view")
        self.value, self.modality, self.shape, self.ndim, self.dtype = value, modality, tuple(value.shape), value.ndim, value.dtype
    def numel(self): return self.value.numel()
    def neural_sha256(self):
        value, fragment = self.value.detach(), max(1, IO_BYTES // self.value.element_size())
        version = int(value._version)
        proof = hashlib.sha256(json.dumps([list(self.shape), str(self.dtype)], separators=(",", ":")).encode())
        for prefix in itertools.product(*(range(size) for size in self.shape[:-1])):
            for start in range(0, self.shape[-1], fragment):
                selected = value[prefix + (slice(start, min(self.shape[-1], start+fragment)),)]
                selected = media_transfer(selected, "cpu").contiguous()
                if not bool(torch.isfinite(selected).all()): raise RuntimeError("native media contains nonfinite activity")
                proof.update(selected.view(torch.uint8).numpy().tobytes())
        if int(value._version) != version: raise RuntimeError("native media output changed during its exact proof")
        return proof.hexdigest()
    def _rgb_rows(self, frame=0):
        value = self.value.detach()
        if self.modality == "video" and value.ndim == 5: value = value[0]
        elif self.modality == "image" and value.ndim == 4: value = value[0]
        width, height, pixels = value.shape[-1], value.shape[-2], max(1, IO_BYTES // 64)
        for row in range(height):
            yield None
            for start in range(0, width, pixels):
                selected = value[:, frame, row, start:start+pixels] if self.modality == "video" else value[:, row, start:start+pixels]
                selected = media_transfer(selected, "cpu").float().clamp(-1, 1)
                yield ((selected + 1.0) * 127.5).to(torch.uint8).transpose(0, 1).contiguous().numpy().tobytes()
    def pcm_chunks(self):
        value = self.value.detach(); width, fragment = value.shape[-1], max(1, IO_BYTES // 32)
        prefixes = __import__("itertools").product(*(range(size) for size in value.shape[:-1]))
        for prefix in prefixes:
            for start in range(0, width, fragment):
                selected = media_transfer(value[prefix + (slice(start, min(width, start+fragment)),)], "cpu").float().clamp(-1, 1)
                raw = (selected * 32767.0).to(torch.int16).contiguous().view(torch.uint8).numpy().tobytes()
                if os.sys.byteorder != "little": raw = b"".join(raw[index:index+2][::-1] for index in range(0, len(raw), 2))
                yield raw


def media_view(value, modality): return value if isinstance(value, MediaCanvasView) else TensorMediaView(value, modality)


def _chunk(writer, kind, payload):
    writer.write(struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", binascii.crc32(kind + payload) & 0xffffffff))


def _compressed_rows(view, frame=0):
    compressor = zlib.compressobj(level=6)
    for value in view._rgb_rows(frame):
        compressed = compressor.compress(b"\0" if value is None else value)
        if compressed: yield compressed
    final = compressor.flush()
    if final: yield final


def _zlib_bound(count): return count + (count >> 12) + (count >> 14) + (count >> 25) + 64


def encode_png_file(value, *, directory, policy, cancelled=lambda: False, fps=None):
    view = media_view(value, "image" if fps is None else "video")
    width, height = int(view.shape[-1]), int(view.shape[-2])
    frames = 1 if fps is None else int(view.shape[-3])
    if not 1 <= width <= 0x7fffffff or not 1 <= height <= 0x7fffffff or not 1 <= frames <= 0xffffffff:
        raise ValueError("PNG dimensions/frame count exceed the container integer fields")
    if fps is not None and (type(fps) is not int or not 1 <= fps <= 65535): raise ValueError("APNG frame timebase is invalid")
    compressed = _zlib_bound((width*3+1)*height)
    bound = 4096 + frames * (compressed + ((compressed + IO_BYTES-1)//IO_BYTES) * 32 + 128)
    writer = EncodedMediaFile(directory, policy, bound, cancelled=cancelled)
    with policy.reserve_ram(IO_BYTES * 8 + 131072, "streamed PNG/APNG codec window"):
        writer.write(b"\x89PNG\r\n\x1a\n")
        _chunk(writer, b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        if fps is not None: _chunk(writer, b"acTL", struct.pack(">II", frames, 0))
        sequence = 0
        for frame in range(frames):
            writer.check()
            if fps is not None:
                _chunk(writer, b"fcTL", struct.pack(">IIIIIHHBB", sequence, width, height, 0, 0, 1, fps, 0, 0)); sequence += 1
            for payload in _compressed_rows(view, frame):
                for start in range(0, len(payload), IO_BYTES):
                    part = payload[start:start+IO_BYTES]
                    if frame == 0: _chunk(writer, b"IDAT", part)
                    else:
                        if sequence > 0xffffffff: raise ValueError("APNG sequence exceeds the container integer fields")
                        _chunk(writer, b"fdAT", struct.pack(">I", sequence) + part); sequence += 1
        _chunk(writer, b"IEND", b"")
    return writer.finish()


def encode_wav_file(value, *, directory, policy, sample_rate=16000, cancelled=lambda: False):
    if type(sample_rate) is not int or not 1 <= sample_rate <= 0x7fffffff: raise ValueError("PCM rate exceeds the WAV byte-rate field")
    view = media_view(value, "audio"); count = int(view.numel()); data_size = count * 2
    rf64 = data_size + 36 > 0xffffffff
    writer = EncodedMediaFile(directory, policy, data_size + 128, cancelled=cancelled)
    writer.write((b"RF64" + struct.pack("<I", 0xffffffff) if rf64 else b"RIFF" + struct.pack("<I", data_size + 36)) + b"WAVE")
    if rf64: writer.write(b"ds64" + struct.pack("<IQQQI", 28, data_size + 72, data_size, count, 0))
    writer.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate*2, 2, 16))
    writer.write(b"data" + struct.pack("<I", 0xffffffff if rf64 else data_size))
    with policy.reserve_ram(IO_BYTES * 8, "streamed WAV codec window"):
        for payload in view.pcm_chunks(): writer.write(payload)
    if writer.byte_count != data_size + (80 if rf64 else 44): raise RuntimeError("PCM output did not cover its exact sample timeline")
    return writer.finish()


def encode_mp4_file(value, *, directory, policy, executable, fps=8, audio=None, sample_rate=16000, cancelled=lambda: False):
    """Same host codec, streamed RGB/PCM input and cancellable owned output.

    No elapsed-time cutoff truncates a valid large movie. The watcher enforces
    actual cancellation/resource/output-identity bounds, not a quality preset.
    Exact codec determinism is separately qualified; failure retains the
    ordinary APNG fallback at the existing caller.
    """
    view = media_view(value, "video")
    width, height, frames = int(view.shape[-1]), int(view.shape[-2]), int(view.shape[-3])
    if type(fps) is not int or not 1 <= fps <= 65535: raise ValueError("MP4 frame timebase is invalid")
    audio_file = None
    if audio is not None: audio_file = encode_wav_file(audio, directory=directory, policy=policy, sample_rate=sample_rate, cancelled=cancelled)
    bound = max(1024*1024, width*height*frames*6 + frames*4096 + (0 if audio_file is None else len(audio_file)*2) + 131072)
    writer = EncodedMediaFile(directory, policy, bound, cancelled=cancelled)
    process = None; finished, errors = threading.Event(), []
    with policy.reserve_ram(IO_BYTES * 12, "streamed MP4 codec input/error window"):
        command = [str(executable), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", "%dx%d" % (width, height),
                   "-r", str(fps), "-i", "-"]
        if audio_file is not None:
            audio_file.verify(); command += ["-i", str(audio_file.path), "-map", "0:v:0", "-map", "1:a:0"]
        else: command += ["-an"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if audio_file is not None: command += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
        command += ["-f", "mp4", str(writer.path)]
        def terminate():
            if process is not None and process.poll() is None:
                try: process.terminate()
                except ProcessLookupError: pass
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try: process.kill()
                    except ProcessLookupError: pass
                    process.wait()
        diagnostics, diagnostic_count = [b""], [0]
        def drain_stderr():
            # A pipe applies OS backpressure; retain only a small tail in RAM.
            # No unlimited TemporaryFile can fill the user's disk between
            # cancellation/resource watcher observations.
            try:
                while True:
                    block = process.stderr.read(16384)
                    if not block: return
                    diagnostic_count[0] += len(block); diagnostics[0] = (diagnostics[0]+block)[-16384:]
                    if diagnostic_count[0] > IO_BYTES*4:
                        raise RuntimeError("native codec exceeded its admitted diagnostic stream")
            except BaseException as error:
                if not finished.is_set(): errors.append(error); terminate()
        def watch():
            while not finished.wait(.05):
                try:
                    writer.check(); current = writer.path.lstat()
                    if ((current.st_dev, current.st_ino) != writer.identity or not stat.S_ISREG(current.st_mode)
                            or current.st_size > bound):
                        raise RuntimeError("native codec exceeded or replaced an admitted output/error window")
                    policy.require_disk(0, "streamed native codec live reserve")
                    writer.lease.refresh_physical()
                except BaseException as error:
                    errors.append(error); terminate(); return
        watcher = diagnostic_reader = None
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                bufsize=0, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            diagnostic_reader = threading.Thread(target=drain_stderr, name="omni-media-codec-diagnostics", daemon=True); diagnostic_reader.start()
            watcher = threading.Thread(target=watch, name="omni-media-codec-reserve", daemon=True); watcher.start()
            for frame in range(frames):
                for payload in view._rgb_rows(frame):
                    if payload is None: continue
                    writer.check(); offset = 0
                    while offset < len(payload):
                        if errors: raise errors[0]
                        written = process.stdin.write(payload[offset:])
                        if not written: raise RuntimeError("native codec stopped accepting its exact input stream")
                        offset += written
            process.stdin.close(); process.wait()
            diagnostic_reader.join()
            if errors: raise errors[0]
            if process.returncode:
                raise RuntimeError("native MP4 codec failed: " + diagnostics[0].decode("utf-8", errors="replace"))
            result = writer.finish_external()
            writer.handle.seek(0); header = writer.handle.read(12)
            if len(header) != 12 or header[4:8] != b"ftyp": raise RuntimeError("native codec did not produce an MP4 container")
            if audio_file is not None:
                writer.handle.seek(0); tail, found = b"", False
                while True:
                    writer.check(); block = writer.handle.read(IO_BYTES)
                    if not block: break
                    value = tail + block
                    if b"soun" in value: found = True; break
                    tail = value[-3:]
                if not found: raise RuntimeError("native MP4 codec did not preserve its neural audio track")
                writer.verify()
            return result
        except BaseException:
            terminate()
            if errors: raise errors[0]
            raise
        finally:
            finished.set()
            if watcher is not None: watcher.join()
            if process is not None and process.stdin is not None and not process.stdin.closed: process.stdin.close()
            if process is not None and process.stderr is not None: process.stderr.close()
            if diagnostic_reader is not None: diagnostic_reader.join()
            if errors: raise errors[0]


def payload_sha256(payload): return payload.sha256 if isinstance(payload, EncodedMediaFile) else hashlib.sha256(payload).hexdigest()


def existing_media_sha256(path):
    path = Path(path); before = path.lstat()
    if not stat.S_ISREG(before.st_mode): raise RuntimeError("media artifact must remain a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    key = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    try:
        opened = os.fstat(descriptor)
        if key(opened) != key(before): raise RuntimeError("media artifact changed before checksum")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, IO_BYTES)
            if not block: break
            digest.update(block)
        if key(os.fstat(descriptor)) != key(opened) or key(path.lstat()) != key(opened): raise RuntimeError("media artifact changed during checksum")
        return digest.hexdigest()
    finally: os.close(descriptor)


def write_media_payload(path, payload):
    from .media_artifact_operation import media_payload_write
    with media_payload_write(path, len(payload)):
        if isinstance(payload, EncodedMediaFile): return payload.copy_to(path)
        Path(path).write_bytes(payload); return Path(path)
