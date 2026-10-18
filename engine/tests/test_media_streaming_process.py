"""NEW codec process protocol using a fake Popen, never real FFmpeg."""
import io
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from test_media_canvas_streaming import CANVAS, CODEC, Policy


class FakeInput:
    def __init__(self, process): self.process, self.closed = process, False
    def write(self, payload):
        if self.process.on_input: self.process.on_input(self.process)
        count = min(3, len(payload)); self.process.inputs.extend(payload[:count]); return count
    def close(self): self.closed = True


class FakeProcess:
    def __init__(self, command, *, on_input=None, replace=False, diagnostic=b"", **kwargs):
        self.command, self.output, self.on_input, self.replace = command, Path(command[-1]), on_input, replace
        self.stdin, self.stderr = FakeInput(self), io.BytesIO(diagnostic)
        self.inputs, self.returncode, self.terminated = bytearray(), None, 0
        self.stdin_mode, self.stderr_mode = kwargs["stdin"], kwargs["stderr"]
    def poll(self): return self.returncode
    def terminate(self): self.terminated += 1; self.returncode = -15
    def kill(self): self.returncode = -9
    def wait(self, timeout=None):
        if self.returncode is None:
            payload = b"\0\0\0\x18ftypisom" + (b"soun" if "1:a:0" in self.command else b"") + bytes(self.inputs)
            if self.replace:
                temporary = self.output.with_suffix(".replacement"); temporary.write_bytes(payload); os.replace(temporary, self.output)
            else: self.output.write_bytes(payload)
            self.returncode = 0
        return self.returncode


class MediaStreamingProcessCases(unittest.TestCase):
    def test_actual_mp4_source_streams_exact_rgb_and_pcm_without_full_serialized_input(self):
        processes = []
        def popen(command, **kwargs):
            value = FakeProcess(command, **kwargs); processes.append(value); return value
        with tempfile.TemporaryDirectory() as root, patch.object(CODEC.subprocess, "Popen", popen):
            policy = Policy()
            def run():
                video, audio = torch.zeros((1, 3, 2, 2, 3)), torch.linspace(-.5, .5, 6).reshape(1, 6)
                output = CODEC.encode_mp4_file(video, audio=audio, directory=root, policy=policy, executable="not-executed", fps=8, sample_rate=16)
                self.assertEqual(bytes(processes[0].inputs), bytes([127])*36)
                self.assertEqual(output.inline_bytes(65536)[4:8], b"ftyp")
                command = processes[0].command
                self.assertIn("aac", command); self.assertEqual(command[-3:-1], ["-f", "mp4"])
                self.assertEqual(processes[0].stderr_mode, CODEC.subprocess.PIPE)
            CANVAS.media_storage_lifetime(run)()
            self.assertEqual(list(Path(root).iterdir()), []); self.assertTrue(all(lease.closed for lease in policy.leases))

    def test_codec_cancellation_terminates_owned_process_and_releases_activity(self):
        cancelled, processes = threading.Event(), []
        def popen(command, **kwargs):
            value = FakeProcess(command, on_input=lambda _process: cancelled.set(), **kwargs); processes.append(value); return value
        with tempfile.TemporaryDirectory() as root, patch.object(CODEC.subprocess, "Popen", popen):
            policy = Policy()
            with self.assertRaises(InterruptedError):
                CANVAS.media_storage_lifetime(lambda: CODEC.encode_mp4_file(torch.zeros((1, 3, 2, 2, 3)),
                    directory=root, policy=policy, executable="not-executed", cancelled=cancelled.is_set))()
            self.assertEqual(processes[0].terminated, 1); self.assertEqual(list(Path(root).iterdir()), [])

    def test_replaced_codec_output_is_rejected_without_unlinking_the_replacement(self):
        processes = []
        def popen(command, **kwargs):
            value = FakeProcess(command, replace=True, **kwargs); processes.append(value); return value
        with tempfile.TemporaryDirectory() as root, patch.object(CODEC.subprocess, "Popen", popen):
            policy = Policy()
            with self.assertRaisesRegex(RuntimeError, "replaced or exceeded"):
                CANVAS.media_storage_lifetime(lambda: CODEC.encode_mp4_file(torch.zeros((1, 3, 2, 2, 3)),
                    directory=root, policy=policy, executable="not-executed"))()
            self.assertTrue(processes[0].output.exists())

    def test_codec_diagnostics_are_bounded_memory_not_an_unlimited_disk_log(self):
        processes = []
        def popen(command, **kwargs):
            value = FakeProcess(command, diagnostic=b"failure\n"*((CODEC.IO_BYTES*4//8)+1), **kwargs); processes.append(value); return value
        with tempfile.TemporaryDirectory() as root, patch.object(CODEC.subprocess, "Popen", popen):
            policy = Policy()
            with self.assertRaisesRegex(RuntimeError, "diagnostic stream"):
                CANVAS.media_storage_lifetime(lambda: CODEC.encode_mp4_file(torch.zeros((1, 3, 2, 2, 3)),
                    directory=root, policy=policy, executable="not-executed"))()
            self.assertEqual(list(Path(root).iterdir()), []); self.assertTrue(all(lease.closed for lease in policy.leases))
