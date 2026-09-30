"""RAM-first bounded learner input waves; never a whole-record window list.

Only transient input IDs/activity are queued. A resource-admitted spill is
removed on close and never enters a native checkpoint or cursor authority.
"""

import os
import hashlib
import sqlite3
import tempfile
from pathlib import Path

import torch
from .text_spool import DatasetResourcePause


class PreparedWindowWave:
    def __init__(self, *, physical_batch, ram_budget, directory, reserve, allow_spill=True):
        if physical_batch < 1 or ram_budget < 0:
            raise ValueError("prepared window wave budget is invalid")
        self.physical_batch = int(physical_batch)
        self.ram_budget = int(ram_budget)
        self.reserve = reserve
        self.allow_spill = allow_spill
        self.directory = Path(directory)
        self.rows = []
        self.ram_bytes = 0
        self.database = None
        self.path = None
        self.window_count = 0
        self.label_count = 0

    def _write(self, values):
        ids, cue, noise = values
        blobs = [value.detach().cpu().contiguous().numpy().tobytes() for value in values]
        self.reserve(disk_bytes=sum(map(len, blobs)) * 2 + 4096)
        digest = hashlib.sha256(b"".join(blobs)).hexdigest()
        self.database.execute("INSERT INTO windows VALUES (?,?,?,?,?)", (self.window_count, *blobs, digest))

    def append(self, ids, cue, noise):
        values = (ids.detach().cpu().to(torch.int64).contiguous(),
            cue.detach().cpu().to(torch.float32).contiguous(), noise.detach().cpu().to(torch.float32).contiguous())
        amount = sum(value.numel() * value.element_size() for value in values) + 512
        if self.database is None and self.ram_bytes + amount <= self.ram_budget:
            self.reserve(ram_bytes=amount)
            self.rows.append(values)
            self.ram_bytes += amount
        else:
            if self.database is None:
                if not self.allow_spill:
                    raise DatasetResourcePause("requested learner wave exceeds admitted input RAM and input spill is disabled", {
                        "requestedInputRamBytes": self.ram_bytes + amount, "admittedInputRamBytes": self.ram_budget,
                        "requestedWaveHonored": False, "resumeAtCommittedWaveBoundary": True})
                self.directory.mkdir(parents=True, exist_ok=True)
                descriptor, path = tempfile.mkstemp(prefix="input-wave-", suffix=".sqlite3", dir=self.directory)
                os.close(descriptor)
                self.path = Path(path)
                self.database = sqlite3.connect(str(self.path))
                self.database.execute("PRAGMA journal_mode=OFF")
                self.database.execute("CREATE TABLE windows (ordinal INTEGER PRIMARY KEY, ids BLOB, cue BLOB, noise BLOB, sha256 TEXT)")
                saved = self.window_count
                for index, old in enumerate(self.rows):
                    self.window_count = index
                    self._write(old)
                self.window_count = saved
                self.rows.clear()
                self.ram_bytes = 0
            self._write(values)
        self.window_count += 1
        self.label_count += max(0, int(values[0].numel()) - 1)

    def __bool__(self):
        return bool(self.window_count)

    def batches(self):
        if self.database is None:
            for offset in range(0, len(self.rows), self.physical_batch):
                yield self.rows[offset:offset + self.physical_batch]
            return
        self.database.commit()
        batch = []
        for ids, cue, noise, claimed in self.database.execute("SELECT ids,cue,noise,sha256 FROM windows ORDER BY ordinal"):
            if hashlib.sha256(ids + cue + noise).hexdigest() != claimed:
                raise RuntimeError("prepared input-wave activity was corrupted before neural consumption")
            self.reserve(ram_bytes=(len(ids) + len(cue) + len(noise)) * 2)
            batch.append((torch.frombuffer(bytearray(ids), dtype=torch.int64),
                torch.frombuffer(bytearray(cue), dtype=torch.float32),
                torch.frombuffer(bytearray(noise), dtype=torch.float32)))
            if len(batch) == self.physical_batch:
                yield batch
                batch = []
        if batch:
            yield batch

    def close(self):
        self.rows.clear()
        if self.database is not None:
            self.database.close()
            self.database = None
        if self.path is not None:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            self.path = None
