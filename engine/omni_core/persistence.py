"""Atomic JSON and safe-tensor persistence helpers."""

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import torch
from safetensors.torch import load_file, save_file


def atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("%s does not contain a JSON object" % path)
    return value


def atomic_save_tensors(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Dict[str, str] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    os.unlink(temporary_name)
    safe = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in tensors.items()
    }
    if not safe:
        # safetensors permits an empty mapping; keeping the real file is useful
        # to exporters and makes the layout invariant.
        safe = {}
    try:
        save_file(safe, temporary_name, metadata=metadata or {})
        os.replace(temporary_name, str(path))
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_tensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return load_file(str(path), device=device)


def tensor_checksum(tensors: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def snapshot_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("brain.json", "core.safetensors", "plasticity.safetensors"):
        source_file = source / filename
        if not source_file.exists():
            raise FileNotFoundError(str(source_file))
        shutil.copy2(str(source_file), str(destination / filename))


class EventLog:
    """Append-only SQLite operational event journal.

    WAL mode makes completed events durable across a worker crash.  Database
    triggers reject accidental updates/deletes so traces remain auditable.
    """

    def __init__(self, path: Path, brain_id: str):
        self.path = Path(path)
        self.brain_id = brain_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(self.path), timeout=10.0, isolation_level=None
        )
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                kind TEXT NOT NULL,
                brain_id TEXT NOT NULL,
                job_id TEXT,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS events_no_update
            BEFORE UPDATE ON events
            BEGIN SELECT RAISE(ABORT, 'events are append-only'); END
            """
        )
        self.connection.execute(
            """
            CREATE TRIGGER IF NOT EXISTS events_no_delete
            BEFORE DELETE ON events
            BEGIN SELECT RAISE(ABORT, 'events are append-only'); END
            """
        )

    def append(
        self,
        kind: str,
        payload: Dict[str, Any],
        job_id: str = None,
    ) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        event_id = uuid.uuid4().hex
        checksum = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self.connection.execute(
            """
            INSERT INTO events
            (event_id, created_at, kind, brain_id, job_id, payload_json, payload_sha256)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                time.time(),
                str(kind),
                self.brain_id,
                job_id,
                serialized,
                checksum,
            ),
        )
        return event_id

    def recent(self, limit: int = 100) -> list:
        cursor = self.connection.execute(
            """
            SELECT sequence, event_id, created_at, kind, job_id, payload_json,
                   payload_sha256
            FROM events ORDER BY sequence DESC LIMIT ?
            """,
            (max(1, min(int(limit), 10000)),),
        )
        return [
            {
                "sequence": row[0],
                "eventId": row[1],
                "createdAt": row[2],
                "kind": row[3],
                "jobId": row[4],
                "payload": json.loads(row[5]),
                "sha256": row[6],
            }
            for row in cursor.fetchall()
        ]

    def integrity(self) -> str:
        row = self.connection.execute("PRAGMA quick_check").fetchone()
        return str(row[0] if row else "unknown")

    def close(self) -> None:
        self.connection.close()
