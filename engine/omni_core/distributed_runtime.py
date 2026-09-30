"""Deterministic, restart-safe primitives for distributed ground-up training.

This module intentionally has no dependency on :class:`AdaptiveBrain`.  It is
safe to import from the desktop worker merely to report launch state, and the
training entry point can use it before allocating a model or CUDA context.

The on-disk protocol is deliberately small:

* a canonical, source-free ``DatasetManifest`` identifies every valid record;
* rank cursors advance only at a globally committed optimizer-step boundary;
* rank zero publishes a content-addressed checkpoint descriptor last; and
* the descriptor contains the exact native ``brain.json`` bytes that name the
  matching immutable mutable-state/substrate generations.

If a process dies between the native neural save and publication, recovery
restores those saved ``brain.json`` bytes before loading the brain.  The native
state store then materializes the referenced generation and truncates any
uncommitted replay suffix.  Consequently retrying a wave cannot silently skip
or duplicate a record.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import sqlite3
import stat
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from .datasets import DatasetCoverage, DatasetRecord, iter_dataset_records
from .persistence import atomic_write_bytes, atomic_write_json, read_json
from .record_window_wave import validate_record_window_cursor
from .text_spool import bounded_json_sha256
from .distributed_seal import validate_distributed_training_seal

try:  # ``resource`` is POSIX-only; importing this module must work on Windows.
    import resource as _resource
except ImportError:  # pragma: no cover - exercised by the Windows CI runner.
    _resource = None


DATASET_MANIFEST_FORMAT = "omni-distributed-dataset-manifest"
DATASET_MANIFEST_VERSION = 2
CHECKPOINT_FORMAT = "omni-distributed-training-checkpoint"
CHECKPOINT_VERSION = 2
STATUS_FORMAT = "omni-distributed-training-status"
STATUS_VERSION = 1


class DistributedRunLease:
    """One rank-zero writer owns an external run folder until it is quiescent."""

    def __init__(self, path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("distributed run lease cannot follow a symbolic link")
        self.descriptor = os.open(str(path), os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            identity = os.fstat(self.descriptor)
            if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                raise ValueError("distributed run lease must have one regular-file owner")
            if os.name == "nt":
                import msvcrt
                if identity.st_size == 0:
                    os.write(self.descriptor, b"0")
                os.lseek(self.descriptor, 0, os.SEEK_SET)
                msvcrt.locking(self.descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.descriptor)
            self.descriptor = None
            raise

    def close(self):
        if self.descriptor is None:
            return
        # Closing releases the OS lock. Never unlink a lock inode while a
        # concurrent opener could still hold it and create a second owner.
        os.close(self.descriptor)
        self.descriptor = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_peak_rss_bytes() -> int:
    """Return a best-effort process peak without a mandatory dependency."""

    if _resource is not None:
        peak = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
        # getrusage reports KiB on Linux and bytes on macOS/BSD.
        return max(0, peak if platform.system().lower() == "darwin" else peak * 1024)
    try:
        import psutil  # type: ignore

        process = psutil.Process(os.getpid())
        info = process.memory_info()
        value = getattr(info, "peak_wset", getattr(info, "rss", 0))
        return max(0, int(value))
    except (ImportError, OSError, RuntimeError, ValueError):
        # Memory telemetry is diagnostic. Unknown is represented as zero and
        # never interpreted as permission to allocate more memory.
        return 0


def _safe_int_environment(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return int(default)
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError("%s must be an integer" % name) from error
    return value


@dataclass(frozen=True)
class DistributedLaunchConfig:
    """The torchrun identity visible to both worker and training CLI."""

    world_size: int
    rank: int
    local_rank: int
    local_world_size: int
    master_addr: str
    master_port: int
    torchrun: bool

    @classmethod
    def from_environment(cls) -> "DistributedLaunchConfig":
        world_size = _safe_int_environment("WORLD_SIZE", 1)
        rank = _safe_int_environment("RANK", 0)
        local_rank = _safe_int_environment("LOCAL_RANK", 0)
        local_world_size = _safe_int_environment(
            "LOCAL_WORLD_SIZE", world_size
        )
        master_port = _safe_int_environment("MASTER_PORT", 0)
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("RANK must be within WORLD_SIZE")
        if local_rank < 0 or local_rank >= max(1, local_world_size):
            raise ValueError("LOCAL_RANK must be within LOCAL_WORLD_SIZE")
        if master_port < 0 or master_port > 65535:
            raise ValueError("MASTER_PORT is invalid")
        return cls(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            local_world_size=max(1, local_world_size),
            master_addr=os.environ.get("MASTER_ADDR", "").strip(),
            master_port=master_port,
            torchrun=(
                world_size > 1
                or "TORCHELASTIC_RUN_ID" in os.environ
                or "LOCAL_RANK" in os.environ
            ),
        )

    def as_status(self) -> Dict[str, Any]:
        return {
            "detected": self.torchrun,
            "worldSize": self.world_size,
            "rank": self.rank,
            "localRank": self.local_rank,
            "localWorldSize": self.local_world_size,
            "masterAddress": self.master_addr or None,
            "masterPort": self.master_port or None,
        }


@dataclass(frozen=True)
class DistributedContext:
    """Initialized process-group and device selection."""

    launch: DistributedLaunchConfig
    device: torch.device
    backend: str
    initialized_here: bool

    @property
    def world_size(self) -> int:
        return self.launch.world_size

    @property
    def rank(self) -> int:
        return self.launch.rank

    @property
    def local_rank(self) -> int:
        return self.launch.local_rank

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()

    def status(self) -> Dict[str, Any]:
        return {
            **self.launch.as_status(),
            "initialized": bool(dist.is_initialized()),
            "backend": self.backend,
            "device": str(self.device),
            "host": socket.gethostname(),
            "platform": platform.system().lower(),
        }


def initialize_distributed(
    *,
    requested_device: str = "auto",
    timeout_seconds: int = 300,
) -> DistributedContext:
    """Initialize torchrun when present, otherwise choose one local device.

    CUDA is bound to ``LOCAL_RANK`` before NCCL starts.  Linux uses NCCL for
    CUDA; Windows uses Gloo because the official Windows wheels do not ship a
    production NCCL backend.  MPS is deliberately single-process.  A request
    for MPS under a multi-process launch falls back to CPU/Gloo so two ranks
    never contend for the one Metal device.
    """

    launch = DistributedLaunchConfig.from_environment()
    requested = str(requested_device or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda", "mps"} and not requested.startswith(
        "cuda:"
    ):
        raise ValueError("distributed device must be auto, cpu, mps, or cuda[:index]")

    cuda_requested = requested == "cuda" or requested.startswith("cuda:")
    cuda_selected = torch.cuda.is_available() and requested not in {"cpu", "mps"}
    if cuda_requested and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    if cuda_selected:
        if launch.local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                "LOCAL_RANK %d has no matching CUDA device" % launch.local_rank
            )
        torch.cuda.set_device(launch.local_rank)
        device = torch.device("cuda", launch.local_rank)
    elif (
        requested in {"auto", "mps"}
        and launch.world_size == 1
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        if requested == "mps" and launch.world_size == 1:
            raise RuntimeError("MPS was requested but is unavailable")
        device = torch.device("cpu")

    backend = "single"
    initialized_here = False
    if launch.world_size > 1:
        backend = (
            "nccl"
            if device.type == "cuda"
            and platform.system().lower() != "windows"
            and dist.is_nccl_available()
            else "gloo"
        )
        if not dist.is_initialized():
            from datetime import timedelta

            dist.init_process_group(
                backend=backend,
                init_method="env://",
                rank=launch.rank,
                world_size=launch.world_size,
                timeout=timedelta(seconds=max(30, int(timeout_seconds))),
            )
            initialized_here = True
    return DistributedContext(
        launch=launch,
        device=device,
        backend=backend,
        initialized_here=initialized_here,
    )


@dataclass(frozen=True)
class DatasetManifestEntry:
    ordinal: int
    record_id: str
    name: str
    kind: str
    bytes_read: int
    content_sha256: str
    provenance_sha256: str

    @staticmethod
    def from_record(ordinal: int, record: DatasetRecord) -> "DatasetManifestEntry":
        content_hash = str(record.content_sha256 or "").strip().lower()
        if not content_hash:
            payload = getattr(record, "text_payload", None)
            content_hash = payload.sha256 if payload is not None else hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        provenance_hash = bounded_json_sha256(record.provenance)
        identity = _canonical_sha256(
            {
                "ordinal": int(ordinal),
                "name": str(record.name),
                "kind": str(record.kind),
                "bytesRead": int(record.bytes_read),
                "contentSha256": content_hash,
                "provenanceSha256": provenance_hash,
            }
        )
        return DatasetManifestEntry(
            ordinal=int(ordinal),
            record_id=identity,
            name=str(record.name),
            kind=str(record.kind),
            bytes_read=max(0, int(record.bytes_read)),
            content_sha256=content_hash,
            provenance_sha256=provenance_hash,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "recordId": self.record_id,
            "name": self.name,
            "kind": self.kind,
            "bytesRead": self.bytes_read,
            "contentSha256": self.content_sha256,
            "provenanceSha256": self.provenance_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DatasetManifestEntry":
        entry = cls(
            ordinal=int(value["ordinal"]),
            record_id=str(value["recordId"]),
            name=str(value["name"]),
            kind=str(value["kind"]),
            bytes_read=int(value["bytesRead"]),
            content_sha256=str(value["contentSha256"]),
            provenance_sha256=str(value["provenanceSha256"]),
        )
        if entry.ordinal < 0 or entry.bytes_read < 0:
            raise ValueError("dataset manifest entry has a negative counter")
        if any(
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in (
                entry.record_id,
                entry.content_sha256,
                entry.provenance_sha256,
            )
        ):
            raise ValueError("dataset manifest entry has an invalid checksum")
        return entry


class _ManifestEntrySequence(Sequence[DatasetManifestEntry]):
    """Lazy ordinal index backed by the manifest's read-only SQLite file."""

    def __init__(self, path: Path, count: int):
        self.path = Path(path).resolve()
        self.count = max(0, int(count))

    def __len__(self) -> int:
        return self.count

    @staticmethod
    def _entry(row: Sequence[Any]) -> DatasetManifestEntry:
        return DatasetManifestEntry.from_dict(
            {
                "ordinal": row[0],
                "recordId": row[1],
                "name": row[2],
                "kind": row[3],
                "bytesRead": row[4],
                "contentSha256": row[5],
                "provenanceSha256": row[6],
            }
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.execute("PRAGMA query_only=ON")
        return connection

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.count)
            if step != 1:
                return [self[position] for position in range(start, stop, step)]
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT ordinal, record_id, name, kind, bytes_read, "
                    "content_sha256, provenance_sha256 FROM records "
                    "WHERE ordinal >= ? AND ordinal < ? ORDER BY ordinal",
                    (start, stop),
                )
                return [self._entry(row) for row in rows]
        position = int(index)
        if position < 0:
            position += self.count
        if position < 0 or position >= self.count:
            raise IndexError("dataset manifest ordinal is out of range")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT ordinal, record_id, name, kind, bytes_read, "
                "content_sha256, provenance_sha256 FROM records WHERE ordinal = ?",
                (position,),
            ).fetchone()
        if row is None:
            raise ValueError("dataset manifest ordinal index has a gap")
        return self._entry(row)

    def iter_from(self, start: int = 0) -> Iterator[DatasetManifestEntry]:
        connection = self._connect()
        try:
            cursor = connection.execute(
                "SELECT ordinal, record_id, name, kind, bytes_read, "
                "content_sha256, provenance_sha256 FROM records "
                "WHERE ordinal >= ? ORDER BY ordinal",
                (max(0, int(start)),),
            )
            while True:
                rows = cursor.fetchmany(1024)
                if not rows:
                    break
                for row in rows:
                    yield self._entry(row)
        finally:
            connection.close()

    def __iter__(self) -> Iterator[DatasetManifestEntry]:
        return self.iter_from(0)


@dataclass(frozen=True)
class DatasetManifest:
    source: str
    requested_kind: str
    entries: _ManifestEntrySequence
    coverage: Mapping[str, Any]
    content_sha256: str
    record_chain_sha256: str
    index_path: Path
    temporary_index: bool = False

    @staticmethod
    def _body(
        *,
        source: str,
        requested_kind: str,
        record_count: int,
        record_chain_sha256: str,
        coverage: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "format": DATASET_MANIFEST_FORMAT,
            "formatVersion": DATASET_MANIFEST_VERSION,
            "source": source,
            "requestedKind": requested_kind,
            "validRecordCount": int(record_count),
            "recordIndex": {
                "format": "sqlite-ordinal-index",
                "formatVersion": 1,
                "recordChainSha256": record_chain_sha256,
            },
            "coverage": dict(coverage),
            "recordAssignment": "global-ordinal-mod-world-size-v1",
            "rawSourceStored": False,
            "rawTokenIdsStored": False,
            "boundedMemoryBuild": True,
        }

    @classmethod
    def build(
        cls,
        path: Path,
        requested_kind: str = "",
        *,
        database_path: Optional[Path] = None,
        resource_admission=None,
    ) -> "DatasetManifest":
        source = str(Path(path).resolve())
        if database_path is None:
            descriptor, name = tempfile.mkstemp(
                prefix="omni-dataset-manifest-", suffix=".sqlite3"
            )
            os.close(descriptor)
            index_path = Path(name).resolve()
            temporary_index = True
        else:
            destination = Path(database_path).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            index_path = destination.parent / (
                ".%s-%s.next" % (destination.name, uuid.uuid4().hex)
            )
            temporary_index = False
        try:
            connection = sqlite3.connect(str(index_path))
            try:
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.executescript(
                    "CREATE TABLE records ("
                    "ordinal INTEGER PRIMARY KEY, record_id TEXT NOT NULL UNIQUE, "
                    "name TEXT NOT NULL, kind TEXT NOT NULL, bytes_read INTEGER NOT NULL, "
                    "content_sha256 TEXT NOT NULL, provenance_sha256 TEXT NOT NULL);"
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
                )
                coverage = DatasetCoverage()
                chain = hashlib.sha256(b"omni-dataset-record-chain-v1\n")
                batch: List[Tuple[Any, ...]] = []
                count = 0
                for ordinal, record in enumerate(
                    iter_dataset_records(
                        Path(source),
                        requested_kind=requested_kind,
                        coverage=coverage,
                        _resource_admission=resource_admission,
                    )
                ):
                    entry = DatasetManifestEntry.from_record(ordinal, record)
                    chain.update(_canonical_json(entry.to_dict()))
                    chain.update(b"\n")
                    batch.append(
                        (
                            entry.ordinal,
                            entry.record_id,
                            entry.name,
                            entry.kind,
                            entry.bytes_read,
                            entry.content_sha256,
                            entry.provenance_sha256,
                        )
                    )
                    count += 1
                    if len(batch) >= 1024:
                        if resource_admission is not None:
                            resource_admission("distributed manifest ordinal index", 0, 4096 * len(batch))
                        connection.executemany(
                            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", batch
                        )
                        connection.commit()
                        batch.clear()
                if batch:
                    if resource_admission is not None:
                        resource_admission("distributed manifest ordinal index", 0, 4096 * len(batch))
                    connection.executemany(
                        "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", batch
                    )
                coverage_value = coverage.as_dict()
                if not bool(coverage_value.get("complete", False)):
                    raise RuntimeError(
                        "dataset traversal did not produce complete coverage"
                    )
                record_chain = chain.hexdigest()
                body = cls._body(
                    source=source,
                    requested_kind=str(requested_kind or ""),
                    record_count=count,
                    record_chain_sha256=record_chain,
                    coverage=coverage_value,
                )
                content_sha = _canonical_sha256(body)
                metadata = {**body, "contentSha256": content_sha}
                connection.execute(
                    "INSERT INTO metadata VALUES ('manifest', ?)",
                    (_canonical_json(metadata).decode("utf-8"),),
                )
                connection.commit()
            finally:
                connection.close()
            if database_path is not None:
                destination = Path(database_path).resolve()
                os.replace(str(index_path), str(destination))
                index_path = destination
            return cls(
                source=source,
                requested_kind=str(requested_kind or ""),
                entries=_ManifestEntrySequence(index_path, count),
                coverage=coverage_value,
                content_sha256=content_sha,
                record_chain_sha256=record_chain,
                index_path=index_path,
                temporary_index=temporary_index,
            )
        except BaseException:
            if index_path.exists():
                index_path.unlink()
            raise

    def to_dict(self) -> Dict[str, Any]:
        body = self._body(
            source=self.source,
            requested_kind=self.requested_kind,
            record_count=len(self.entries),
            record_chain_sha256=self.record_chain_sha256,
            coverage=self.coverage,
        )
        return {**body, "contentSha256": self.content_sha256}

    def write(self, path: Path) -> None:
        pointer_path = Path(path).resolve()
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        database_path = pointer_path.with_suffix(".sqlite3")
        if self.index_path != database_path:
            temporary = database_path.parent / (
                ".%s-%s.next" % (database_path.name, uuid.uuid4().hex)
            )
            try:
                shutil.copy2(str(self.index_path), str(temporary))
                os.replace(str(temporary), str(database_path))
            finally:
                if temporary.exists():
                    temporary.unlink()
        payload = self.to_dict()
        pointer_body = {
            **payload,
            "indexFile": {
                "path": database_path.name,
                "sha256": _file_sha256(database_path),
                "bytes": database_path.stat().st_size,
            },
        }
        pointer = {
            **pointer_body,
            "pointerSha256": _canonical_sha256(pointer_body),
        }
        atomic_write_json(pointer_path, pointer)

    @classmethod
    def read(cls, path: Path) -> "DatasetManifest":
        pointer_path = Path(path).resolve()
        value = read_json(pointer_path)
        pointer_body = {
            key: item for key, item in value.items() if key != "pointerSha256"
        }
        if _canonical_sha256(pointer_body) != str(value.get("pointerSha256", "")):
            raise ValueError("distributed dataset manifest pointer checksum mismatch")
        index_spec = value.get("indexFile")
        if not isinstance(index_spec, Mapping):
            raise ValueError("distributed dataset manifest index is missing")
        relative = str(index_spec.get("path", ""))
        if not relative or Path(relative).name != relative:
            raise ValueError("distributed dataset manifest index path is unsafe")
        index_path = (pointer_path.parent / relative).resolve()
        if (
            not index_path.is_file()
            or index_path.stat().st_size != int(index_spec.get("bytes", -1))
            or _file_sha256(index_path) != str(index_spec.get("sha256", ""))
        ):
            raise ValueError("distributed dataset manifest index checksum mismatch")
        connection = sqlite3.connect(str(index_path))
        try:
            connection.execute("PRAGMA query_only=ON")
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='manifest'"
            ).fetchone()
            if row is None:
                raise ValueError("distributed dataset manifest metadata is missing")
            metadata = json.loads(str(row[0]))
            count_row = connection.execute(
                "SELECT COUNT(*), MIN(ordinal), MAX(ordinal) FROM records"
            ).fetchone()
        finally:
            connection.close()
        count = int(metadata.get("validRecordCount", -1))
        body = {
            key: item for key, item in metadata.items() if key != "contentSha256"
        }
        claimed = str(metadata.get("contentSha256", ""))
        if (
            value.get("format") != DATASET_MANIFEST_FORMAT
            or int(value.get("formatVersion", 0)) != DATASET_MANIFEST_VERSION
            or metadata != {
                key: item for key, item in value.items()
                if key not in {"indexFile", "pointerSha256"}
            }
            or _canonical_sha256(body) != claimed
            or count_row is None
            or int(count_row[0]) != count
            or (count > 0 and (int(count_row[1]) != 0 or int(count_row[2]) != count - 1))
        ):
            raise ValueError("distributed dataset manifest checksum mismatch")
        coverage = metadata.get("coverage")
        record_index = metadata.get("recordIndex")
        if (
            not isinstance(coverage, Mapping)
            or not bool(coverage.get("complete", False))
            or not isinstance(record_index, Mapping)
        ):
            raise ValueError("dataset manifest coverage/index is invalid")
        return cls(
            source=str(metadata["source"]),
            requested_kind=str(metadata.get("requestedKind", "")),
            entries=_ManifestEntrySequence(index_path, count),
            coverage=dict(coverage),
            content_sha256=claimed,
            record_chain_sha256=str(record_index["recordChainSha256"]),
            index_path=index_path,
            temporary_index=False,
        )

    def verify_current_source(self, resource_admission=None) -> None:
        current = DatasetManifest.build(Path(self.source), self.requested_kind, resource_admission=resource_admission)
        try:
            if current.content_sha256 != self.content_sha256:
                raise ValueError(
                    "dataset changed after its distributed manifest was committed"
                )
        finally:
            current.close()

    def close(self) -> None:
        if self.temporary_index:
            try:
                self.index_path.unlink()
            except FileNotFoundError:
                pass

    def iter_verified_records(
        self,
        *,
        rank: int,
        world_size: int,
        start_ordinal: int = 0,
        stop_ordinal: Optional[int] = None,
        resource_admission=None,
    ) -> Iterator[tuple[DatasetManifestEntry, DatasetRecord]]:
        """Stream and verify this rank's exact global-ordinal shard."""

        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("invalid distributed dataset shard")
        start = max(0, int(start_ordinal))
        stop = len(self.entries) if stop_ordinal is None else min(
            len(self.entries), max(start, int(stop_ordinal))
        )
        coverage = DatasetCoverage()
        expected_iterator = self.entries.iter_from(0)
        visited = 0
        for ordinal, record in enumerate(
            iter_dataset_records(
                Path(self.source),
                requested_kind=self.requested_kind,
                coverage=coverage,
                _resource_admission=resource_admission,
            )
        ):
            try:
                expected = next(expected_iterator)
            except StopIteration as error:
                raise ValueError(
                    "dataset gained records after manifest creation"
                ) from error
            actual = DatasetManifestEntry.from_record(ordinal, record)
            if actual != expected:
                raise ValueError(
                    "dataset record %d changed after manifest creation" % ordinal
                )
            visited += 1
            if ordinal < start:
                continue
            if ordinal >= stop:
                break
            if ordinal % world_size == rank:
                yield expected, record
        if stop == len(self.entries) and visited != len(self.entries):
            raise ValueError("dataset lost records after manifest creation")

    def owned_record_count(self, rank: int, world_size: int) -> int:
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("invalid distributed dataset shard")
        if rank >= len(self.entries):
            return 0
        return 1 + (len(self.entries) - 1 - rank) // world_size


@dataclass(frozen=True)
class RankCursor:
    rank: int
    world_size: int
    epoch: int
    next_global_ordinal: int
    owned_records_completed: int
    optimizer_steps_completed: int
    manifest_sha256: str
    record_window: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        value = {
            "rank": self.rank,
            "worldSize": self.world_size,
            "epoch": self.epoch,
            "nextGlobalOrdinal": self.next_global_ordinal,
            "ownedRecordsCompleted": self.owned_records_completed,
            "optimizerStepsCompleted": self.optimizer_steps_completed,
            "manifestSha256": self.manifest_sha256,
        }
        if self.record_window is not None:
            value["recordWindow"] = validate_record_window_cursor(self.record_window)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankCursor":
        cursor = cls(
            rank=int(value["rank"]),
            world_size=int(value["worldSize"]),
            epoch=int(value["epoch"]),
            next_global_ordinal=int(value["nextGlobalOrdinal"]),
            owned_records_completed=int(value["ownedRecordsCompleted"]),
            optimizer_steps_completed=int(value["optimizerStepsCompleted"]),
            manifest_sha256=str(value["manifestSha256"]),
            record_window=validate_record_window_cursor(value["recordWindow"]) if "recordWindow" in value else None,
        )
        if (
            cursor.rank < 0
            or cursor.rank >= cursor.world_size
            or cursor.world_size < 1
            or cursor.epoch < 0
            or cursor.next_global_ordinal < 0
            or cursor.owned_records_completed < 0
            or cursor.optimizer_steps_completed < 0
        ):
            raise ValueError("distributed rank cursor is invalid")
        if cursor.record_window is not None and (
            cursor.record_window["ordinal"] % cursor.world_size != cursor.rank
            or cursor.record_window["phase"] == "complete"
            or not cursor.next_global_ordinal <= cursor.record_window["ordinal"] < cursor.next_global_ordinal + cursor.world_size
        ):
            raise ValueError("distributed active record does not belong to its exact rank group")
        return cursor


def initial_rank_cursors(
    *, world_size: int, manifest_sha256: str
) -> List[RankCursor]:
    return [
        RankCursor(
            rank=rank,
            world_size=world_size,
            epoch=0,
            next_global_ordinal=0,
            owned_records_completed=0,
            optimizer_steps_completed=0,
            manifest_sha256=manifest_sha256,
        )
        for rank in range(world_size)
    ]


@dataclass(frozen=True)
class ResourceReading:
    rank: int
    host: str
    device: str
    measured_at: str
    disk_total_bytes: int
    disk_free_bytes: int
    disk_reserve_bytes: int
    available_memory_bytes: Optional[int]
    ram_reserve_bytes: int
    process_peak_rss_bytes: int
    accelerator_allocated_bytes: int
    accelerator_reserved_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "host": self.host,
            "device": self.device,
            "measuredAt": self.measured_at,
            "diskTotalBytes": self.disk_total_bytes,
            "diskFreeBytes": self.disk_free_bytes,
            "diskReserveBytes": self.disk_reserve_bytes,
            "availableMemoryBytes": self.available_memory_bytes,
            "ramReserveBytes": self.ram_reserve_bytes,
            "processPeakRssBytes": self.process_peak_rss_bytes,
            "acceleratorAllocatedBytes": self.accelerator_allocated_bytes,
            "acceleratorReservedBytes": self.accelerator_reserved_bytes,
        }


def sample_resources(
    *, rank: int, device: torch.device, path: Path, policy_status: Mapping[str, Any]
) -> ResourceReading:
    disk = shutil.disk_usage(Path(path))
    peak_rss = _process_peak_rss_bytes()
    allocated = 0
    reserved = 0
    if device.type == "cuda" and torch.cuda.is_available():
        allocated = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
    return ResourceReading(
        rank=int(rank),
        host=socket.gethostname(),
        device=str(device),
        measured_at=_utc_now(),
        disk_total_bytes=int(policy_status.get("diskTotalBytes", disk.total)),
        disk_free_bytes=int(policy_status.get("diskFreeBytes", disk.free)),
        disk_reserve_bytes=int(policy_status.get("diskReserveBytes", 0)),
        available_memory_bytes=(
            int(policy_status["availableMemoryBytes"])
            if isinstance(policy_status.get("availableMemoryBytes"), int)
            else None
        ),
        ram_reserve_bytes=int(policy_status.get("ramReserveBytes", 0)),
        process_peak_rss_bytes=max(0, peak_rss),
        accelerator_allocated_bytes=allocated,
        accelerator_reserved_bytes=reserved,
    )


def aggregate_resource_readings(
    readings: Sequence[ResourceReading],
) -> Dict[str, Any]:
    if not readings:
        raise ValueError("resource telemetry requires at least one rank")
    ordered = sorted(readings, key=lambda item: item.rank)
    if [item.rank for item in ordered] != list(range(len(ordered))):
        raise ValueError("resource telemetry ranks are incomplete")
    available = [
        int(item.available_memory_bytes)
        for item in ordered
        if item.available_memory_bytes is not None
    ]
    return {
        "measuredAt": _utc_now(),
        "rankCount": len(ordered),
        "hosts": sorted({item.host for item in ordered}),
        "minimumDiskFreeBytes": min(item.disk_free_bytes for item in ordered),
        "maximumDiskReserveBytes": max(
            item.disk_reserve_bytes for item in ordered
        ),
        "minimumDiskAboveReserveBytes": min(
            item.disk_free_bytes - item.disk_reserve_bytes for item in ordered
        ),
        "minimumAvailableMemoryBytes": min(available) if available else None,
        "totalProcessPeakRssBytes": sum(
            item.process_peak_rss_bytes for item in ordered
        ),
        "totalAcceleratorAllocatedBytes": sum(
            item.accelerator_allocated_bytes for item in ordered
        ),
        "totalAcceleratorReservedBytes": sum(
            item.accelerator_reserved_bytes for item in ordered
        ),
        "diskPressure": any(
            item.disk_free_bytes <= item.disk_reserve_bytes for item in ordered
        ),
        "perRank": [item.to_dict() for item in ordered],
    }


class DistributedRunStore:
    """Content-addressed rank-zero publication and recovery metadata."""

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        self.manifest_path = self.path / "dataset-manifest.json"
        self.checkpoints_path = self.path / "checkpoints"
        self.active_path = self.path / "active-checkpoint.json"
        self.status_path = self.path / "distributed-status.json"
        self.telemetry_path = self.path / "telemetry.jsonl"
        self.cancel_path = self.path / "cancel.requested"
        self.failures_path = self.path / "failures"
        self.ranks_path = self.path / "ranks"
        self.template_path = self.path / "template"

    def initialize(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        self.failures_path.mkdir(parents=True, exist_ok=True)
        self.ranks_path.mkdir(parents=True, exist_ok=True)

    def acquire_run_lease(self):
        self.path.mkdir(parents=True, exist_ok=True)
        return DistributedRunLease(self.path / ".run-owner.lock")

    def write_status(self, **fields: Any) -> Dict[str, Any]:
        prior: Dict[str, Any] = {}
        if self.status_path.is_file():
            try:
                prior = read_json(self.status_path)
            except (OSError, ValueError, TypeError):
                prior = {}
        value = {
            "format": STATUS_FORMAT,
            "formatVersion": STATUS_VERSION,
            "updatedAt": _utc_now(),
            **prior,
            **fields,
        }
        # updatedAt belongs to this write, even when a caller copied an old
        # status object into fields.
        value["updatedAt"] = _utc_now()
        atomic_write_json(self.status_path, value)
        return value

    def request_cancel(self, reason: str = "user requested cancellation") -> None:
        atomic_write_json(
            self.cancel_path,
            {"requestedAt": _utc_now(), "reason": str(reason)},
        )

    def cancel_requested(self) -> bool:
        return self.cancel_path.is_file()

    def clear_cancel(self) -> None:
        try:
            self.cancel_path.unlink()
        except FileNotFoundError:
            pass

    def record_failure(self, rank: int, error: BaseException) -> None:
        self.failures_path.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            self.failures_path / ("rank-%05d.json" % int(rank)),
            {
                "rank": int(rank),
                "failedAt": _utc_now(),
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def recent_failures(self) -> List[Dict[str, Any]]:
        values: List[Dict[str, Any]] = []
        if not self.failures_path.is_dir():
            return values
        for path in sorted(self.failures_path.glob("rank-*.json")):
            try:
                values.append(read_json(path))
            except (OSError, ValueError, TypeError):
                continue
        return values

    def append_telemetry(self, value: Mapping[str, Any]) -> None:
        payload = _canonical_json(dict(value)) + b"\n"
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            str(self.telemetry_path), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600
        )
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def publish_checkpoint(
        self,
        *,
        brain_json_path: Path,
        manifest: DatasetManifest,
        cursors: Sequence[RankCursor],
        epochs_requested: int,
        global_optimizer_steps: int,
        dynamic_high_water: int,
        strategy: str,
        telemetry: Mapping[str, Any],
        capability_rehearsal: Optional[Mapping[str, Any]] = None,
        media_training: Optional[Mapping[str, Any]] = None,
        native_brain_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Publish the cursor only after the matching brain save exists."""

        brain_bytes = Path(brain_json_path).read_bytes()
        try:
            brain_value = json.loads(brain_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("native brain checkpoint metadata is invalid") from error
        if not isinstance(brain_value, dict):
            raise ValueError("native brain checkpoint metadata is invalid")
        ordered = sorted(cursors, key=lambda item: item.rank)
        if [cursor.rank for cursor in ordered] != list(range(len(ordered))):
            raise ValueError("checkpoint rank cursors are incomplete")
        if any(cursor.world_size != len(ordered) for cursor in ordered):
            raise ValueError("checkpoint rank cursor world sizes diverged")
        if any(cursor.manifest_sha256 != manifest.content_sha256 for cursor in ordered):
            raise ValueError("checkpoint cursor manifest hash diverged")
        dynamic_value = max(0, int(dynamic_high_water))
        body = {
            "format": CHECKPOINT_FORMAT,
            "formatVersion": CHECKPOINT_VERSION,
            "createdAt": _utc_now(),
            "manifestSha256": manifest.content_sha256,
            "worldSize": len(ordered),
            "epochsRequested": int(epochs_requested),
            "globalOptimizerSteps": int(global_optimizer_steps),
            "strategy": str(strategy),
            "brainJsonSha256": hashlib.sha256(brain_bytes).hexdigest(),
            "nativeMutableGeneration": (
                brain_value.get("mutable_state", {}) or {}
            ).get("activeGeneration"),
            "nativeSubstrateGeneration": (
                brain_value.get("substrate", {}) or {}
            ).get("persistence", {}).get("activeGeneration"),
            "rankCursors": [cursor.to_dict() for cursor in ordered],
            "dynamicHighWater": dynamic_value,
            "telemetry": dict(telemetry),
            "capabilityRehearsal": (
                dict(capability_rehearsal)
                if isinstance(capability_rehearsal, Mapping)
                else None
            ),
            "mediaTraining": (
                dict(media_training)
                if isinstance(media_training, Mapping)
                else None
            ),
            "rawSourceStored": False,
            "rawTokenIdsStored": False,
            "transactional": True,
        }
        if native_brain_path is not None:
            native_source = Path(native_brain_path).resolve()
            if (native_source / "engine" / "brain.json").resolve() != Path(brain_json_path).resolve():
                raise ValueError("native distributed snapshot is not the cursor's exact neural save")
            body["nativeBrainDirectory"] = "native"
            seal = validate_distributed_training_seal(brain_value.get("distributed_training_seal"))
            if seal is None or seal["manifestSha256"] != manifest.content_sha256 or seal["rankCursors"] != body["rankCursors"] or seal["recordsPerEpoch"] != len(manifest.entries) or seal["epochsRequested"] != epochs_requested or seal["committedRecordStop"] != dynamic_value or seal["globalOptimizerSteps"] != global_optimizer_steps:
                raise ValueError("distributed native save is not independently sealed to these exact cursors")
            for cursor in ordered:
                if cursor.record_window is not None:
                    window = cursor.record_window
                    expected = manifest.entries[window["ordinal"]]
                    if expected.record_id != window["recordId"] or expected.content_sha256 != window["contentSha256"]:
                        raise ValueError("distributed native cursor does not bind its exact manifest record")
            body["distributedSealSha256"] = seal["contentSha256"]
        content_sha = _canonical_sha256(body)
        checkpoint = {**body, "contentSha256": content_sha}
        temporary = self.checkpoints_path / (".%s.next" % uuid.uuid4().hex)
        destination = self.checkpoints_path / content_sha
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            if native_brain_path is not None:
                # Copy all referenced immutable generations before the only
                # publication pointer moves. An old pointer never depends on
                # mutable checkpoint-brain contents or a just-pruned suffix.
                shutil.copytree(native_source, temporary / "native")
            atomic_write_bytes(temporary / "brain.json", brain_bytes)
            ranks_dir = temporary / "ranks"
            ranks_dir.mkdir()
            for cursor in ordered:
                atomic_write_json(
                    ranks_dir / ("rank-%05d.json" % cursor.rank),
                    cursor.to_dict(),
                )
            atomic_write_json(temporary / "checkpoint.json", checkpoint)
            if destination.exists():
                existing = self._read_checkpoint_directory(destination)
                if existing.get("contentSha256") != content_sha:
                    raise ValueError("distributed checkpoint identity conflict")
                shutil.rmtree(temporary)
            else:
                os.replace(str(temporary), str(destination))
            pointer = {
                "format": CHECKPOINT_FORMAT,
                "formatVersion": CHECKPOINT_VERSION,
                "activeGeneration": content_sha,
                "checkpoint": "checkpoints/%s/checkpoint.json" % content_sha,
                "checkpointSha256": _file_sha256(
                    destination / "checkpoint.json"
                ),
            }
            # This is the only publication point. An old pointer always names
            # an entirely complete old brain/cursor pair.
            atomic_write_json(self.active_path, pointer)
            return checkpoint
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def _read_checkpoint_directory(self, path: Path) -> Dict[str, Any]:
        checkpoint_path = Path(path) / "checkpoint.json"
        value = read_json(checkpoint_path)
        body = {key: item for key, item in value.items() if key != "contentSha256"}
        if (
            value.get("format") != CHECKPOINT_FORMAT
            or int(value.get("formatVersion", 0)) != CHECKPOINT_VERSION
            or _canonical_sha256(body) != str(value.get("contentSha256", ""))
        ):
            raise ValueError("distributed checkpoint checksum mismatch")
        brain_path = Path(path) / "brain.json"
        if _file_sha256(brain_path) != str(value.get("brainJsonSha256", "")):
            raise ValueError("distributed checkpoint brain metadata checksum mismatch")
        if "nativeBrainDirectory" in value:
            if value["nativeBrainDirectory"] != "native" or _file_sha256(Path(path) / "native" / "engine" / "brain.json") != str(value["brainJsonSha256"]):
                raise ValueError("distributed immutable native snapshot identity is invalid")
            native_metadata = read_json(Path(path) / "native" / "engine" / "brain.json")
            seal = validate_distributed_training_seal(native_metadata.get("distributed_training_seal"))
            if seal is None or seal["contentSha256"] != value.get("distributedSealSha256") or seal["manifestSha256"] != value["manifestSha256"] or seal["rankCursors"] != value["rankCursors"] or seal["worldSize"] != value["worldSize"] or seal["epochsRequested"] != value["epochsRequested"] or seal["committedRecordStop"] != value["dynamicHighWater"] or seal["globalOptimizerSteps"] != value["globalOptimizerSteps"]:
                raise ValueError("external distributed cursor differs from the committed native seal")
        raw_cursors = value.get("rankCursors")
        if not isinstance(raw_cursors, list):
            raise ValueError("distributed checkpoint cursors are invalid")
        cursors = [RankCursor.from_dict(item) for item in raw_cursors]
        if [cursor.rank for cursor in cursors] != list(range(len(cursors))):
            raise ValueError("distributed checkpoint rank cursor order is invalid")
        for cursor in cursors:
            path_value = Path(path) / "ranks" / ("rank-%05d.json" % cursor.rank)
            if RankCursor.from_dict(read_json(path_value)) != cursor:
                raise ValueError("per-rank cursor does not match checkpoint")
        return value

    def load_active_checkpoint(
        self, *, manifest_sha256: str, world_size: int
    ) -> Optional[Dict[str, Any]]:
        if not self.active_path.is_file():
            return None
        pointer = read_json(self.active_path)
        generation = str(pointer.get("activeGeneration", ""))
        if (
            pointer.get("format") != CHECKPOINT_FORMAT
            or int(pointer.get("formatVersion", 0)) != CHECKPOINT_VERSION
            or len(generation) != 64
            or str(pointer.get("checkpoint", ""))
            != "checkpoints/%s/checkpoint.json" % generation
        ):
            raise ValueError("distributed checkpoint pointer is invalid")
        directory = self.checkpoints_path / generation
        checkpoint_path = directory / "checkpoint.json"
        if _file_sha256(checkpoint_path) != str(pointer.get("checkpointSha256", "")):
            raise ValueError("distributed checkpoint pointer checksum mismatch")
        checkpoint = self._read_checkpoint_directory(directory)
        if checkpoint.get("contentSha256") != generation:
            raise ValueError("distributed checkpoint generation identity mismatch")
        if str(checkpoint.get("manifestSha256", "")) != manifest_sha256:
            raise ValueError("distributed checkpoint belongs to another dataset manifest")
        if int(checkpoint.get("worldSize", 0)) != int(world_size):
            raise ValueError(
                "distributed checkpoint WORLD_SIZE changed; resume with the original rank count"
            )
        return checkpoint

    def published_native_path(self, checkpoint: Mapping[str, Any]) -> Path:
        if checkpoint.get("nativeBrainDirectory") != "native":
            raise ValueError("distributed cursor is not bound to a complete immutable native snapshot")
        generation = str(checkpoint.get("contentSha256", ""))
        if len(generation) != 64 or any(value not in "0123456789abcdef" for value in generation):
            raise ValueError("distributed native generation identity is invalid")
        path = self.checkpoints_path / generation / "native"
        if _file_sha256(path / "engine" / "brain.json") != checkpoint.get("brainJsonSha256"):
            raise ValueError("distributed native publication metadata changed")
        return path

    def restore_published_brain_json(
        self, checkpoint: Mapping[str, Any], destination: Path
    ) -> None:
        generation = str(checkpoint["contentSha256"])
        source = self.checkpoints_path / generation / "brain.json"
        if _file_sha256(source) != str(checkpoint["brainJsonSha256"]):
            raise ValueError("published brain metadata failed recovery verification")
        atomic_write_bytes(Path(destination), source.read_bytes())

    def prune_checkpoints(self, keep: int = 2) -> Dict[str, int]:
        """Remove only old external descriptors, never native neural blobs."""

        keep = max(1, int(keep))
        active = None
        if self.active_path.is_file():
            active = str(read_json(self.active_path).get("activeGeneration", ""))
        candidates: List[tuple[float, Path]] = []
        if self.checkpoints_path.is_dir():
            for path in self.checkpoints_path.iterdir():
                if path.is_symlink() or not path.is_dir() or path.name == active:
                    continue
                try:
                    checkpoint = self._read_checkpoint_directory(path)
                    timestamp = datetime.fromisoformat(
                        str(checkpoint["createdAt"])
                    ).timestamp()
                except (OSError, ValueError, KeyError, TypeError):
                    continue
                candidates.append((timestamp, path))
        candidates.sort(reverse=True, key=lambda item: (item[0], item[1].name))
        removed = 0
        for _timestamp, path in candidates[max(0, keep - 1) :]:
            shutil.rmtree(path)
            removed += 1
        return {"retained": min(len(candidates), max(0, keep - 1)) + int(bool(active)), "removed": removed}


def read_distributed_status(run_path: Path) -> Dict[str, Any]:
    """Read a rank-zero status file for the worker/app without torch init."""

    path = Path(run_path).resolve() / "distributed-status.json"
    if not path.is_file():
        return {
            "format": STATUS_FORMAT,
            "formatVersion": STATUS_VERSION,
            "state": "not-started",
            "environment": DistributedLaunchConfig.from_environment().as_status(),
        }
    value = read_json(path)
    if (
        value.get("format") != STATUS_FORMAT
        or int(value.get("formatVersion", 0)) != STATUS_VERSION
    ):
        raise ValueError("distributed training status format is invalid")
    return value


__all__ = [
    "CHECKPOINT_FORMAT",
    "DATASET_MANIFEST_FORMAT",
    "DatasetManifest",
    "DatasetManifestEntry",
    "DistributedContext",
    "DistributedLaunchConfig",
    "DistributedRunStore",
    "RankCursor",
    "ResourceReading",
    "aggregate_resource_readings",
    "initial_rank_cursors",
    "initialize_distributed",
    "read_distributed_status",
    "sample_resources",
]
