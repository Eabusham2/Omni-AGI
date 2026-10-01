"""Atomic JSON and safe-tensor persistence helpers."""

import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

import torch
from safetensors.torch import load_file

from .bounded_tensor_io import atomic_save_tensors_bounded


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


def atomic_write_bytes(path: Path, value: bytes) -> None:
    """Atomically replace a file with exact bytes and a durable directory entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
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
    atomic_save_tensors_bounded(path, tensors, metadata=metadata)


def load_tensors(path: Path, device: str = "cpu") -> Dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    return load_file(str(path), device=device)


@dataclass(frozen=True)
class BoundedChecksumSource:
    shape: tuple
    dtype: torch.dtype
    chunks: Any


def tensor_checksum(tensors: Iterable[torch.Tensor], *, chunk_bytes: int = 4 * 1024 * 1024, reserve=None, on_chunk=None) -> str:
    if int(chunk_bytes) < 1:
        raise ValueError("checksum chunk size must be positive")
    def blocks(value, elements):
        if value.numel() <= elements:
            yield value.contiguous().reshape(-1)
        elif value.is_contiguous():
            flat = value.reshape(-1)
            for start in range(0, flat.numel(), elements):
                yield flat[start:start + elements]
        else:
            # Preserve historical logical C-order bytes without materializing
            # an entire strided GPU owner merely to hash it.
            tail = math.prod(value.shape[1:])
            if tail <= elements:
                rows = max(1, elements // max(1, tail))
                for start in range(0, value.shape[0], rows):
                    yield value[start:start + rows].contiguous().reshape(-1)
            else:
                for row in value:
                    yield from blocks(row, elements)
    digest = hashlib.sha256()
    for tensor in tensors:
        if isinstance(tensor, BoundedChecksumSource):
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            count = 0
            for part in tensor.chunks():
                if part.dtype != tensor.dtype:
                    raise ValueError("bounded checksum source dtype changed")
                if reserve is not None:
                    reserve(part.numel() * part.element_size() * 3 + 4096)
                count += part.numel()
                if count > math.prod(tensor.shape):
                    raise ValueError("bounded checksum source exceeded its shape")
                block = part.detach().cpu().contiguous().reshape(-1)
                digest.update(memoryview(block.view(torch.uint8).numpy()))
                if on_chunk is not None:
                    on_chunk(block.numel() * block.element_size())
            if count != math.prod(tensor.shape):
                raise ValueError("bounded checksum source has incomplete coverage")
            continue
        value = tensor.detach()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        step = max(1, int(chunk_bytes) // value.element_size())
        if reserve is not None:
            reserve(min(step, value.numel()) * value.element_size() * 3 + 4096)
        position = 0
        for source in blocks(value, step):
            block = source.to(device="cpu").contiguous()
            if block.numel() and block.stride(-1) != 1:
                # A one-element strided view is "contiguous" to PyTorch but
                # cannot be reinterpreted as bytes until its stride is one.
                compact = torch.empty((block.numel(),), dtype=block.dtype, device="cpu")
                compact.copy_(block)
                block = compact
            digest.update(memoryview(block.view(torch.uint8).numpy()))
            if on_chunk is not None:
                on_chunk(block.numel() * block.element_size())
            if value.is_contiguous():
                from .native_core_paging import release_native_tensor_chunk
                release_native_tensor_chunk(value, position * value.element_size(), block.numel() * value.element_size())
                from .router_state_paging import release_router_tensor_chunk
                release_router_tensor_chunk(value, position * value.element_size(), block.numel() * value.element_size())
            position += block.numel()
    return digest.hexdigest()


def _safe_relative(root: Path, relative: str) -> Path:
    if (
        not relative
        or relative.startswith(("/", "\\"))
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in relative.split("/"))
    ):
        raise ValueError("substrate snapshot contains an unsafe relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("substrate snapshot path escapes its store") from error
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def copy_substrate_snapshot(
    source_engine: Path,
    destination_engine: Path,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Copy one declared immutable substrate generation into another engine.

    Existing content-addressed blobs are reused. The active pointer is replaced
    only after the generation manifest and every checksummed blob are present.
    """

    state = metadata or read_json(Path(source_engine) / "brain.json")
    substrate = state.get("substrate")
    if not isinstance(substrate, dict):
        return None
    pointer = substrate.get("persistence")
    if not isinstance(pointer, dict):
        # Backward-safe stable-v1 checkpoints kept substrate vectors in the
        # monolithic plasticity file and need no auxiliary snapshot.
        return None
    if (
        pointer.get("format") != "omni-substrate-shards"
        or int(pointer.get("formatVersion", 0)) not in {1, 2, 3}
    ):
        raise ValueError("substrate snapshot pointer is incompatible")
    source_store = Path(source_engine) / "substrate"
    destination_store = Path(destination_engine) / "substrate"
    generation_relative = str(pointer.get("generationManifest", ""))
    active_generation = str(pointer.get("activeGeneration", ""))
    if (
        len(active_generation) != 64
        or any(
            character not in "0123456789abcdef"
            for character in active_generation
        )
        or generation_relative
        != "generations/%s/manifest.json" % active_generation
    ):
        raise ValueError("substrate snapshot generation identity is invalid")
    generation_source = _safe_relative(source_store, generation_relative)
    generation_bytes = generation_source.read_bytes()
    expected_generation_sha = str(
        pointer.get("generationManifestSha256", "")
    )
    if _file_sha256(generation_source) != expected_generation_sha:
        raise ValueError("substrate generation manifest checksum mismatch")
    generation = json.loads(generation_bytes.decode("utf-8"))
    if not isinstance(generation, dict):
        raise ValueError("substrate generation manifest is invalid")
    content_body = {
        key: value for key, value in generation.items() if key != "contentSha256"
    }
    content_sha = hashlib.sha256(
        json.dumps(
            content_body,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if (
        generation.get("format") != "omni-substrate-shards"
        or int(generation.get("formatVersion", 0))
        != int(pointer.get("formatVersion", 0))
        or (
            int(generation.get("formatVersion", 0)) == 3
            and generation.get("schema") != "neural-substrate-2"
        )
        or content_sha != str(generation.get("contentSha256", ""))
        or content_sha != active_generation
        or content_sha != str(pointer.get("contentSha256", ""))
    ):
        raise ValueError("substrate generation content checksum mismatch")

    declared: Dict[str, Dict[str, Any]] = {}
    for shard in generation.get("shards", []):
        if not isinstance(shard, dict):
            raise ValueError("substrate generation contains an invalid shard")
        for key in ("records", "tensors"):
            spec = shard.get(key)
            if spec is None:
                continue
            if not isinstance(spec, dict):
                raise ValueError("substrate generation contains an invalid blob")
            relative = str(spec.get("path", ""))
            suffix = ".json" if key == "records" else ".safetensors"
            if relative != "blobs/%s%s" % (
                str(spec.get("sha256", "")),
                suffix,
            ):
                raise ValueError("substrate generation blob identity is invalid")
            prior = declared.get(relative)
            if prior is not None and prior != spec:
                raise ValueError("substrate generation has conflicting blob records")
            declared[relative] = spec

    for relative, spec in sorted(declared.items()):
        source = _safe_relative(source_store, relative)
        if (
            _file_sha256(source) != str(spec.get("sha256", ""))
            or source.stat().st_size != int(spec.get("bytes", -1))
        ):
            raise ValueError("substrate snapshot blob checksum mismatch")
        destination = _safe_relative(destination_store, relative)
        if destination.exists():
            if (
                _file_sha256(destination) != str(spec.get("sha256", ""))
                or destination.stat().st_size != int(spec.get("bytes", -1))
            ):
                raise ValueError("existing substrate blob conflicts with snapshot")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            destination.name + "." + uuid.uuid4().hex + ".tmp"
        )
        try:
            shutil.copy2(str(source), str(temporary))
            os.replace(str(temporary), str(destination))
        finally:
            if temporary.exists():
                temporary.unlink()

    generation_destination = _safe_relative(
        destination_store, generation_relative
    )
    if generation_destination.exists():
        if _file_sha256(generation_destination) != expected_generation_sha:
            raise ValueError("existing substrate generation conflicts with snapshot")
    else:
        atomic_write_bytes(generation_destination, generation_bytes)
    atomic_write_json(destination_store / "manifest.json", dict(pointer))
    return dict(pointer)


def snapshot_required_bytes(
    source: Path,
    *,
    include_packed_ternary: bool = False,
) -> int:
    """Return the declared physical bytes copied by :func:`snapshot_files`.

    Only the active substrate/mutable generations named by brain.json are
    counted. Old unreferenced generations are deliberately excluded so a
    reserve check cannot block on bytes the operation will not write.
    """

    source = Path(source).resolve()
    metadata = read_json(source / "brain.json")
    total = sum(
        (source / filename).stat().st_size
        for filename in ("brain.json", "core.safetensors", "plasticity.safetensors")
    )
    substrate = metadata.get("substrate")
    pointer = (
        substrate.get("persistence")
        if isinstance(substrate, dict)
        else None
    )
    if isinstance(pointer, dict):
        store = source / "substrate"
        generation_path = _safe_relative(
            store, str(pointer.get("generationManifest", ""))
        )
        generation_bytes = generation_path.read_bytes()
        generation = json.loads(generation_bytes.decode("utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("substrate generation manifest is invalid")
        total += len(generation_bytes)
        total += len(
            json.dumps(
                pointer,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for shard in generation.get("shards", []):
            if not isinstance(shard, dict):
                raise ValueError("substrate generation contains an invalid shard")
            for role in ("records", "tensors"):
                spec = shard.get(role)
                if isinstance(spec, dict):
                    total += max(0, int(spec.get("bytes", 0)))
    mutable_pointer = metadata.get("mutable_state")
    if isinstance(mutable_pointer, dict):
        store = source / "state"
        generation_path = _safe_relative(
            store, str(mutable_pointer.get("generationManifest", ""))
        )
        generation_bytes = generation_path.read_bytes()
        generation = json.loads(generation_bytes.decode("utf-8"))
        if not isinstance(generation, dict):
            raise ValueError("mutable-state generation manifest is invalid")
        total += len(generation_bytes)
        total += len(
            json.dumps(
                mutable_pointer,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        roles = generation.get("roles", {})
        if not isinstance(roles, dict):
            raise ValueError("mutable-state generation roles are invalid")
        total += sum(
            max(0, int(spec.get("bytes", 0)))
            for spec in roles.values()
            if isinstance(spec, dict)
        )
        replay = store / "replay.sqlite3"
        if replay.is_file():
            total += replay.stat().st_size
    if include_packed_ternary:
        packed = source / "packed-ternary"
        if packed.is_dir():
            total += sum(
                path.stat().st_size
                for path in packed.rglob("*")
                if path.is_file() and not path.is_symlink()
            )
    conversation = source / "conversation.sqlite3"
    if conversation.is_file() and not conversation.is_symlink():
        total += conversation.stat().st_size
    from .registered_geometry_holdouts import registered_geometry_snapshot_files
    total += sum(path.stat().st_size for path in registered_geometry_snapshot_files(source).values())
    return max(0, int(total))


def snapshot_files(source: Path, destination: Path, *, resource_policy=None) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for filename in ("brain.json", "core.safetensors", "plasticity.safetensors"):
        source_file = source / filename
        if not source_file.exists():
            raise FileNotFoundError(str(source_file))
        shutil.copy2(str(source_file), str(destination / filename))
    copy_substrate_snapshot(source, destination)
    # Imported lazily because the offload store itself builds on the atomic
    # persistence primitives in this module.
    from .offload import copy_mutable_state_snapshot

    copy_mutable_state_snapshot(source, destination)
    from .registered_geometry_holdouts import registered_geometry_snapshot_files
    for relative, path in registered_geometry_snapshot_files(source).items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if resource_policy is None:
            shutil.copy2(path, target)
        else:
            with resource_policy.reserve_spill(((path.stat().st_size + 65535) // 65536) * 65536,
                "protected geometry snapshot data") as lease:
                lease.bind_path(target)
                shutil.copy2(path, target)
                lease.commit(path=target)
    # Structural argument views referenced by saved action/trace records are
    # operational state, not learned vectors or text context. Preserve their
    # checked immutable bytes so snapshot history never gains broken refs.
    from .concept_id_views import IdView
    views = source / "state" / "concept-id-views"
    if views.is_dir():
        if views.is_symlink():
            raise ValueError("saved concept ID views must not be a symlink")
        for path in views.iterdir():
            if path.name.startswith("."):
                continue
            if path.is_symlink() or not path.is_file() or not re.fullmatch(r"[a-f0-9]{64}\.jsonl", path.name):
                raise ValueError("saved concept ID view file is unsafe")
            with path.open("rb") as handle:
                header = json.loads(handle.readline(4096))
            descriptor = {**header, "path": "state/concept-id-views/" + path.name,
                          "sha256": path.stem, "bytes": path.stat().st_size}
            IdView(source, descriptor, brain_id=header["brainId"], turn_id=header["turnId"])
            target = destination / descriptor["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            IdView(destination, descriptor, brain_id=header["brainId"], turn_id=header["turnId"])
    conversation = source / "conversation.sqlite3"
    if conversation.is_file() and not conversation.is_symlink():
        shutil.copy2(str(conversation), str(destination / "conversation.sqlite3"))


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
