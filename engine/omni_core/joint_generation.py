"""Publish one recoverable neural/paged-store ingestion boundary.

The caller must serialize neural saves, paged-store writes, and publication. It
supplies an existing immutable ``MutableStateStore`` generation and a callback
that makes a WAL-consistent SQLite backup (normally
``sqlite3.Connection.backup``). This module references and revalidates the
content-addressed neural blobs without copying them. It never copies an active
SQLite database file directly and never persists source text, absolute paths,
or credentials.

The full SQLite backup is a bounded correctness prototype: repeating it every
checkpoint writes O(current database size) and can become O(N^2) over a large
ingestion. A page-level/content-addressed SQLite snapshot protocol is needed
before claiming low-wear, unbounded training.

An optional process-local verification cache avoids rereading unchanged blobs
while staging. It uses file identity metadata as a change detector, not as
cryptographic proof against silent corruption or an adversarial filesystem.
Recovery and an explicit scrub always stream-hash every referenced blob.

Only a reference atomically committed inside ``brain.json`` is authoritative.
This module stages and fsyncs immutable artifacts but never publishes a second
pointer or reorders ``save()``. An interrupted staging directory or a finished
but unreferenced generation is ignored on recovery; neither is permission to
replay an ingestion cursor. An optional SQLite snapshot is a derived cache and
must be reconciled to the committed substrate generation before use.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import struct
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

from .ingestion_schedule_v3 import (
    EMPTY_RECORD_PREFIX_SHA256,
    source_parser_manifest_sha256,
)


_FORMAT = "omni-joint-checkpoint-generation"
_REFERENCE_FORMAT = "omni-joint-checkpoint-reference"
_VERSION = 2
_NEURAL_ROLES = ("core", "plasticity", "optimizer")
_SQLITE_NAME = "packed-vector-index.sqlite3"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GENERATION_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_COUNT = (1 << 63) - 1
_MAX_JSON_BYTES = 64 * 1024
_MAX_TENSOR_HEADER_BYTES = 64 * 1024 * 1024
_TENSOR_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
    "U32": 4, "I32": 4, "F32": 4,
    "U64": 8, "I64": 8, "F64": 8,
}


class JointGenerationError(ValueError):
    """The active joint checkpoint is incomplete or fails validation."""


class SQLiteBackupWriter(Protocol):
    """Write and close a consistent SQLite backup at ``destination``."""

    def __call__(self, destination: Path) -> None: ...


@dataclass(frozen=True)
class JointGeneration:
    generation_id: str
    manifest: Dict[str, Any]
    directory: Path
    neural_generation_id: str
    neural_files: Dict[str, Path]
    sqlite_snapshot: Optional[Path]


class VerifiedArtifactCache:
    """Process-local hash evidence for repeated immutable content-addressed refs.

    Reuse requires the exact path, size, device, inode, mtime, and ctime to
    match a file that was stream-hashed earlier in this process. This is a
    performance hint only; strict recovery must not use cached metadata.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, Tuple[Tuple[int, ...], str, int]] = {}
        self._reused: set[str] = set()
        self._touched: set[str] = set()
        self.hashed_files = 0
        self.reused_files = 0

    def begin_validation(self) -> None:
        self._touched.clear()
        self._reused.clear()

    def retain_current_refs(self) -> None:
        self._entries = {
            key: value for key, value in self._entries.items()
            if key in self._touched
        }
        self._touched.clear()
        self._reused.clear()

    @staticmethod
    def _key(path: Path) -> str:
        return str(path.absolute())

    @staticmethod
    def _identity(path: Path) -> Tuple[int, ...]:
        info = os.stat(path, follow_symlinks=False)
        return (
            info.st_dev, info.st_ino, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns,
        )

    def was_reused(self, path: Path) -> bool:
        return self._key(path) in self._reused


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as error:
        raise JointGenerationError("joint checkpoint is not finite JSON") from error


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise JointGenerationError("%s must be a lowercase SHA-256" % label)
    return value


def _count(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_COUNT:
        raise JointGenerationError("%s count is invalid" % label)
    return value


def _fields(value: Any, expected: set[str], label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise JointGenerationError("%s fields are invalid" % label)
    return dict(value)


def _validate_cursor_coverage(
    cursor_value: Any, coverage_value: Any, source_content_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cursor_fields = {"committedRecords", "recordPrefixSha256"}
    if isinstance(cursor_value, Mapping) and "activeRecordWindowSha256" in cursor_value:
        cursor_fields.add("activeRecordWindowSha256")
    cursor = _fields(
        cursor_value, cursor_fields, "cursor",
    )
    _count(cursor["committedRecords"], "committed records")
    _hash(cursor["recordPrefixSha256"], "record prefix")
    if "activeRecordWindowSha256" in cursor:
        _hash(cursor["activeRecordWindowSha256"], "active record window")
    if (
        cursor["committedRecords"] == 0
        and cursor["recordPrefixSha256"] != EMPTY_RECORD_PREFIX_SHA256
    ):
        raise JointGenerationError("empty cursor record prefix is invalid")
    coverage = _fields(coverage_value, {
        "visitedRecords", "processedRecords", "rejectedRecords",
        "processedBytes", "expectedRecords", "sourceStreamExhausted",
        "sourceContentReverifiedSha256",
    }, "coverage")
    for name in (
        "visitedRecords", "processedRecords", "rejectedRecords",
        "processedBytes",
    ):
        _count(coverage[name], name)
    if coverage["expectedRecords"] is not None:
        _count(coverage["expectedRecords"], "expected records")
    if type(coverage["sourceStreamExhausted"]) is not bool:
        raise JointGenerationError("source exhaustion flag is invalid")
    if (
        coverage["visitedRecords"]
        != coverage["processedRecords"] + coverage["rejectedRecords"]
        or cursor["committedRecords"] > coverage["visitedRecords"]
        or (
            coverage["expectedRecords"] is not None
            and coverage["visitedRecords"] > coverage["expectedRecords"]
        )
    ):
        raise JointGenerationError("record coverage is incomplete")
    if coverage["sourceStreamExhausted"]:
        if "activeRecordWindowSha256" in cursor:
            raise JointGenerationError("exhausted source still has an active record window")
        if (
            coverage["expectedRecords"] is not None
            and coverage["visitedRecords"] != coverage["expectedRecords"]
        ):
            raise JointGenerationError("final record coverage is incomplete")
        if coverage["sourceContentReverifiedSha256"] != source_content_sha256:
            raise JointGenerationError("final source hash was not reverified")
    elif coverage["sourceContentReverifiedSha256"] is not None:
        raise JointGenerationError("source hash was marked final prematurely")
    return cursor, coverage


def _regular_file(path: Path, label: str) -> int:
    if path.is_symlink():
        raise JointGenerationError("%s must not be a symlink" % label)
    try:
        mode = path.stat()
    except OSError as error:
        raise JointGenerationError("%s is missing" % label) from error
    if not stat.S_ISREG(mode.st_mode):
        raise JointGenerationError("%s is not a regular file" % label)
    return mode.st_size


def _file_digest(
    path: Path, label: str, *, cache: Optional[VerifiedArtifactCache] = None,
    allow_reuse: bool = False,
) -> Tuple[str, int]:
    size = _regular_file(path, label)
    key = cache._key(path) if cache is not None else ""
    identity = cache._identity(path) if cache is not None else None
    if cache is not None:
        cache._touched.add(key)
        cache._reused.discard(key)
        existing = cache._entries.get(key)
        if allow_reuse and existing is not None and existing[0] == identity:
            cache._reused.add(key)
            cache.reused_files += 1
            return existing[1], existing[2]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != size or (
        cache is not None and cache._identity(path) != identity
    ):
        raise JointGenerationError("%s changed while hashing" % label)
    result = digest.hexdigest()
    if cache is not None:
        cache._entries[key] = (identity, result, size)
        cache.hashed_files += 1
    return result, size


def _state_path(root: Path, relative: str, label: str) -> Path:
    """Resolve only the two fixed state-store layouts, never arbitrary paths."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise JointGenerationError("neural state-store root is missing")
    parts = relative.split("/")
    if (
        len(parts) not in (2, 3)
        or any(part in ("", ".", "..") or "\\" in part for part in parts)
        or parts[0] not in ("blobs", "generations")
    ):
        raise JointGenerationError("%s path is unsafe" % label)
    path = root.joinpath(*parts)
    parent = path.parent
    while parent != root:
        if parent.is_symlink():
            raise JointGenerationError("%s parent is a symlink" % label)
        parent = parent.parent
    if path.is_symlink():
        raise JointGenerationError("%s is a symlink" % label)
    return path


def _neural_state_from_pointer(
    neural_store_root: Path, pointer: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(pointer, Mapping):
        raise JointGenerationError("neural generation pointer is invalid")
    generation_id = pointer.get("activeGeneration")
    if (
        pointer.get("format") != "omni-mutable-state"
        or type(pointer.get("formatVersion")) is not int
        or pointer["formatVersion"] != 1
        or not isinstance(generation_id, str)
        or _HASH.fullmatch(generation_id) is None
        or pointer.get("contentSha256") != generation_id
        or pointer.get("generationManifest")
        != "generations/%s/manifest.json" % generation_id
    ):
        raise JointGenerationError("neural generation pointer is invalid")
    manifest_sha256 = _hash(
        pointer.get("generationManifestSha256"), "neural manifest",
    )
    relative = "generations/%s/manifest.json" % generation_id
    path = _state_path(neural_store_root, relative, "neural manifest")
    manifest = _read_canonical_json(path, "neural manifest", maximum=None)
    roles = manifest.get("roles")
    if not isinstance(roles, Mapping):
        raise JointGenerationError("neural manifest roles are missing")
    summary_roles: Dict[str, Dict[str, Any]] = {}
    for role in _NEURAL_ROLES:
        spec = roles.get(role)
        if not isinstance(spec, Mapping):
            raise JointGenerationError("%s neural blob is missing" % role)
        summary_roles[role] = {
            "path": spec.get("path"), "sha256": spec.get("sha256"),
            "bytes": spec.get("bytes"),
        }
    summary = {
        "generationId": generation_id,
        "generationManifest": relative,
        "generationManifestSha256": manifest_sha256,
        "blobs": summary_roles,
    }
    return summary


def _verify_neural_state(
    neural_store_root: Path, value: Any, *, durable: bool = False,
    cache: Optional[VerifiedArtifactCache] = None,
    allow_reuse: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Path]]:
    state = _fields(value, {
        "generationId", "generationManifest", "generationManifestSha256",
        "blobs",
    }, "neural state")
    generation_id = _hash(state["generationId"], "neural generation")
    expected_relative = "generations/%s/manifest.json" % generation_id
    if state["generationManifest"] != expected_relative:
        raise JointGenerationError("neural generation manifest path is invalid")
    manifest_sha256 = _hash(
        state["generationManifestSha256"], "neural manifest",
    )
    blob_specs = _fields(state["blobs"], set(_NEURAL_ROLES), "neural blobs")
    manifest_path = _state_path(
        neural_store_root, expected_relative, "neural manifest",
    )
    actual_manifest_sha, _ = _file_digest(manifest_path, "neural manifest")
    if actual_manifest_sha != manifest_sha256:
        raise JointGenerationError("neural manifest checksum mismatch")
    manifest = _read_canonical_json(
        manifest_path, "neural manifest", maximum=None,
    )
    body = {key: item for key, item in manifest.items()
            if key != "contentSha256"}
    if (
        manifest.get("format") != "omni-mutable-state"
        or type(manifest.get("formatVersion")) is not int
        or manifest["formatVersion"] != 1
        or manifest.get("contentSha256") != generation_id
        or _sha256(_canonical(body)) != generation_id
    ):
        raise JointGenerationError("neural generation content checksum mismatch")
    manifest_roles = manifest.get("roles")
    if not isinstance(manifest_roles, Mapping):
        raise JointGenerationError("neural manifest roles are missing")
    neural_files: Dict[str, Path] = {}
    for role in _NEURAL_ROLES:
        spec = _fields(
            blob_specs[role], {"path", "sha256", "bytes"}, role + " blob",
        )
        digest = _hash(spec["sha256"], role + " blob")
        size = _count(spec["bytes"], role + " blob bytes", minimum=1)
        relative = "blobs/%s.safetensors" % digest
        if spec["path"] != relative:
            raise JointGenerationError("%s blob path is invalid" % role)
        original = manifest_roles.get(role)
        if not isinstance(original, Mapping) or any(
            original.get(field) != spec[field]
            for field in ("path", "sha256", "bytes")
        ):
            raise JointGenerationError("%s blob differs from neural manifest" % role)
        path = _state_path(neural_store_root, relative, role + " blob")
        actual_digest, actual_size = _file_digest(
            path, role + " blob", cache=cache, allow_reuse=allow_reuse,
        )
        if actual_digest != digest or actual_size != size:
            raise JointGenerationError("%s neural blob checksum mismatch" % role)
        _validate_safetensors(path)
        if durable and not (cache is not None and cache.was_reused(path)):
            _fsync_file(path)
        blob_specs[role] = spec
        neural_files[role] = path
    if durable:
        _fsync_file(manifest_path)
        _fsync_directory(Path(neural_store_root) / "blobs")
        _fsync_directory(manifest_path.parent)
        _fsync_directory(Path(neural_store_root) / "generations")
        _fsync_directory(Path(neural_store_root))
    state["blobs"] = blob_specs
    return state, neural_files


def _substrate_state_from_pointer(
    substrate_root: Path, pointer: Mapping[str, Any],
) -> Dict[str, Any]:
    if not isinstance(pointer, Mapping):
        raise JointGenerationError("substrate generation pointer is invalid")
    generation_id = pointer.get("activeGeneration")
    if (
        pointer.get("format") != "omni-substrate-shards"
        or type(pointer.get("formatVersion")) is not int
        or pointer["formatVersion"] != 3
        or not isinstance(generation_id, str)
        or _HASH.fullmatch(generation_id) is None
        or pointer.get("contentSha256") != generation_id
        or pointer.get("generationManifest")
        != "generations/%s/manifest.json" % generation_id
    ):
        raise JointGenerationError("substrate generation pointer is invalid")
    state = {
        "generationId": generation_id,
        "generationManifest": pointer["generationManifest"],
        "generationManifestSha256": pointer.get("generationManifestSha256"),
        "counts": pointer.get("counts"),
    }
    return state


def _verify_substrate_state(
    substrate_root: Path, value: Any, *, durable: bool = False,
    cache: Optional[VerifiedArtifactCache] = None,
    allow_reuse: bool = False,
) -> Dict[str, Any]:
    state = _fields(value, {
        "generationId", "generationManifest", "generationManifestSha256",
        "counts",
    }, "substrate state")
    generation_id = _hash(state["generationId"], "substrate generation")
    relative = "generations/%s/manifest.json" % generation_id
    if state["generationManifest"] != relative:
        raise JointGenerationError("substrate manifest path is invalid")
    manifest_sha = _hash(
        state["generationManifestSha256"], "substrate manifest",
    )
    counts = _fields(
        state["counts"], {"neurons", "assemblies", "synapses"},
        "substrate counts",
    )
    for kind in counts:
        _count(counts[kind], kind)
    manifest_path = _state_path(substrate_root, relative, "substrate manifest")
    actual_sha, _ = _file_digest(manifest_path, "substrate manifest")
    if actual_sha != manifest_sha:
        raise JointGenerationError("substrate manifest checksum mismatch")
    manifest = _read_canonical_json(
        manifest_path, "substrate manifest", maximum=None,
    )
    body = {key: item for key, item in manifest.items()
            if key != "contentSha256"}
    if (
        manifest.get("format") != "omni-substrate-shards"
        or type(manifest.get("formatVersion")) is not int
        or manifest["formatVersion"] != 3
        or manifest.get("contentSha256") != generation_id
        or _sha256(_canonical(body)) != generation_id
        or manifest.get("counts") != counts
    ):
        raise JointGenerationError("substrate generation content checksum mismatch")
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise JointGenerationError("substrate shard manifest is invalid")
    observed_counts = {kind: 0 for kind in counts}
    for shard in shards:
        if not isinstance(shard, Mapping) or shard.get("kind") not in counts:
            raise JointGenerationError("substrate shard is invalid")
        kind = shard["kind"]
        observed_counts[kind] += _count(shard.get("count"), "shard")
        for field, suffix in (("records", ".json"), ("tensors", ".safetensors")):
            spec = shard.get(field)
            if spec is None and field == "tensors":
                continue
            if not isinstance(spec, Mapping):
                raise JointGenerationError("substrate shard blob is missing")
            digest = _hash(spec.get("sha256"), "substrate shard blob")
            size = _count(spec.get("bytes"), "substrate shard bytes", minimum=1)
            blob_relative = "blobs/%s%s" % (digest, suffix)
            if spec.get("path") != blob_relative:
                raise JointGenerationError("substrate shard blob path is invalid")
            blob_path = _state_path(
                substrate_root, blob_relative, "substrate shard blob",
            )
            actual_digest, actual_size = _file_digest(
                blob_path, "substrate shard blob", cache=cache,
                allow_reuse=allow_reuse,
            )
            if actual_digest != digest or actual_size != size:
                raise JointGenerationError("substrate shard blob checksum mismatch")
            if field == "tensors":
                _validate_safetensors(blob_path)
            if durable and not (
                cache is not None and cache.was_reused(blob_path)
            ):
                _fsync_file(blob_path)
    if observed_counts != counts:
        raise JointGenerationError("substrate shard counts do not match")
    if durable:
        _fsync_file(manifest_path)
        _fsync_directory(Path(substrate_root) / "blobs")
        _fsync_directory(manifest_path.parent)
        _fsync_directory(Path(substrate_root) / "generations")
        _fsync_directory(Path(substrate_root))
    state["counts"] = counts
    return state


def _validate_safetensors(path: Path) -> None:
    size = _regular_file(path, "neural safe-tensor file")
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise JointGenerationError("safe-tensor header is missing")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 1 <= header_size <= _MAX_TENSOR_HEADER_BYTES:
            raise JointGenerationError("safe-tensor header length is invalid")
        if header_size > size - 8:
            raise JointGenerationError("safe-tensor header is truncated")
        try:
            header = json.loads(handle.read(header_size))
        except (ValueError, UnicodeError) as error:
            raise JointGenerationError("safe-tensor header is invalid") from error
    if not isinstance(header, dict):
        raise JointGenerationError("safe-tensor header is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise JointGenerationError("safe-tensor metadata is invalid")
    spans = []
    for name, raw in header.items():
        if not isinstance(name, str) or not name:
            raise JointGenerationError("safe-tensor tensor name is invalid")
        spec = _fields(raw, {"dtype", "shape", "data_offsets"}, "tensor")
        width = _TENSOR_DTYPE_BYTES.get(spec["dtype"])
        shape = spec["shape"]
        offsets = spec["data_offsets"]
        if (
            width is None or not isinstance(shape, list)
            or len(shape) > 64 or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or any(type(offset) is not int or offset < 0 for offset in offsets)
        ):
            raise JointGenerationError("safe-tensor tensor descriptor is invalid")
        start, end = offsets
        elements = 1
        for dimension in shape:
            elements *= dimension
            if elements > size:
                raise JointGenerationError("safe-tensor tensor shape exceeds file")
        if end < start or end - start != elements * width:
            raise JointGenerationError("safe-tensor tensor byte span is invalid")
        spans.append((start, end))
    position = 0
    for start, end in sorted(spans):
        if start != position:
            raise JointGenerationError("safe-tensor tensor spans are not contiguous")
        position = end
    if position != size - 8 - header_size:
        raise JointGenerationError("safe-tensor file length is invalid")


def _validate_sqlite_snapshot(path: Path) -> None:
    if _regular_file(path, "SQLite snapshot") < 100:
        raise JointGenerationError("SQLite snapshot is too small")
    for suffix in ("-wal", "-shm", "-journal"):
        if path.with_name(path.name + suffix).exists():
            raise JointGenerationError("SQLite snapshot has an uncommitted sidecar")
    with path.open("rb") as handle:
        if handle.read(16) != b"SQLite format 3\x00":
            raise JointGenerationError("SQLite snapshot header is invalid")
    try:
        # Immutable read prevents a validation read from creating sidecars.
        with closing(sqlite3.connect(
            path.absolute().as_uri() + "?mode=ro&immutable=1", uri=True,
        )) as db:
            result = db.execute("PRAGMA integrity_check").fetchall()
            if result != [("ok",)]:
                raise JointGenerationError("SQLite snapshot integrity check failed")
    except sqlite3.DatabaseError as error:
        raise JointGenerationError("SQLite snapshot is corrupt") from error


def _validate_manifest(value: Any, generation_id: str) -> Dict[str, Any]:
    manifest = _fields(value, {
        "format", "formatVersion", "generationId", "neuralState",
        "neuralGenerationSha256", "substrateState",
        "substrateGenerationSha256", "sqliteSnapshot", "checkpointSequence",
        "previousManifestSha256",
        "sourceManifestSha256", "parserManifestSha256",
        "sourceParserManifestSha256", "sourceContentSha256", "cursor",
        "coverage",
    }, "generation manifest")
    if (
        manifest["format"] != _FORMAT
        or type(manifest["formatVersion"]) is not int
        or manifest["formatVersion"] != _VERSION
        or manifest["generationId"] != generation_id
    ):
        raise JointGenerationError("joint generation identity is invalid")
    _count(manifest["checkpointSequence"], "checkpoint sequence", minimum=1)
    if manifest["previousManifestSha256"] is not None:
        _hash(manifest["previousManifestSha256"], "previous joint manifest")
    for name in (
        "sourceManifestSha256", "parserManifestSha256", "sourceContentSha256",
        "sourceParserManifestSha256", "neuralGenerationSha256",
        "substrateGenerationSha256",
    ):
        _hash(manifest[name], name)
    if manifest["sourceParserManifestSha256"] != source_parser_manifest_sha256(
        manifest["sourceManifestSha256"], manifest["parserManifestSha256"],
    ):
        raise JointGenerationError("source/parser manifest binding is invalid")
    neural = _fields(manifest["neuralState"], {
        "generationId", "generationManifest", "generationManifestSha256",
        "blobs",
    }, "neural state")
    substrate = _fields(manifest["substrateState"], {
        "generationId", "generationManifest", "generationManifestSha256",
        "counts",
    }, "substrate state")
    if manifest["neuralGenerationSha256"] != neural["generationId"]:
        raise JointGenerationError("neural generation binding is invalid")
    if manifest["substrateGenerationSha256"] != substrate["generationId"]:
        raise JointGenerationError("substrate generation binding is invalid")
    snapshot = manifest["sqliteSnapshot"]
    if snapshot is not None:
        snapshot = _fields(
            snapshot, {"file", "sha256", "bytes"}, "SQLite snapshot",
        )
        if snapshot["file"] != _SQLITE_NAME:
            raise JointGenerationError("SQLite snapshot filename is invalid")
        _hash(snapshot["sha256"], "SQLite snapshot checksum")
        _count(snapshot["bytes"], "SQLite snapshot bytes", minimum=1)
    cursor, coverage = _validate_cursor_coverage(
        manifest["cursor"], manifest["coverage"],
        manifest["sourceContentSha256"],
    )
    manifest["neuralState"] = neural
    manifest["substrateState"] = substrate
    manifest["sqliteSnapshot"] = snapshot
    manifest["cursor"] = cursor
    manifest["coverage"] = coverage
    if len(_canonical(manifest)) > _MAX_JSON_BYTES:
        raise JointGenerationError("joint manifest is too large")
    return manifest


def _read_canonical_json(
    path: Path, label: str, *, maximum: Optional[int] = _MAX_JSON_BYTES,
) -> Dict[str, Any]:
    size = _regular_file(path, label)
    if size < 1 or (maximum is not None and size > maximum):
        raise JointGenerationError("%s size is invalid" % label)
    try:
        body = path.read_bytes()
        value = json.loads(body)
    except (OSError, ValueError, UnicodeError) as error:
        raise JointGenerationError("%s is invalid JSON" % label) from error
    if not isinstance(value, dict) or _canonical(value) != body:
        raise JointGenerationError("%s is not canonical JSON" % label)
    return value


def _validate_generation(
    directory: Path, generation_id: str, manifest_sha256: str,
    *, source_manifest_sha256: str, parser_manifest_sha256: str,
    source_content_sha256: str, neural_store_root: Path,
    substrate_store_root: Path, durable: bool = False,
    cache: Optional[VerifiedArtifactCache] = None,
    allow_reuse: bool = False,
) -> JointGeneration:
    if (
        directory.parent.is_symlink() or directory.is_symlink()
        or not directory.is_dir()
    ):
        raise JointGenerationError("joint generation directory is missing")
    manifest_path = directory / "manifest.json"
    actual_sha, _ = _file_digest(manifest_path, "joint manifest")
    if actual_sha != manifest_sha256:
        raise JointGenerationError("joint manifest checksum mismatch")
    manifest = _validate_manifest(
        _read_canonical_json(manifest_path, "joint manifest"), generation_id,
    )
    for field, expected in (
        ("sourceManifestSha256", source_manifest_sha256),
        ("parserManifestSha256", parser_manifest_sha256),
        ("sourceContentSha256", source_content_sha256),
    ):
        if manifest[field] != _hash(expected, field):
            raise JointGenerationError("%s does not match observed source" % field)
    neural_state, neural_files = _verify_neural_state(
        neural_store_root, manifest["neuralState"], durable=durable,
        cache=cache, allow_reuse=allow_reuse,
    )
    _verify_substrate_state(
        substrate_store_root, manifest["substrateState"], durable=durable,
        cache=cache, allow_reuse=allow_reuse,
    )
    spec = manifest["sqliteSnapshot"]
    snapshot: Optional[Path] = None
    expected_files = {"manifest.json"}
    if spec is not None:
        snapshot = directory / _SQLITE_NAME
        digest, size = _file_digest(snapshot, "SQLite snapshot")
        if digest != spec["sha256"] or size != spec["bytes"]:
            raise JointGenerationError("SQLite snapshot checksum mismatch")
        _validate_sqlite_snapshot(snapshot)
        expected_files.add(_SQLITE_NAME)
    if set(path.name for path in directory.iterdir()) != expected_files:
        raise JointGenerationError("joint generation contains unexpected files")
    return JointGeneration(
        generation_id=generation_id, manifest=manifest, directory=directory,
        neural_generation_id=neural_state["generationId"],
        neural_files=neural_files, sqlite_snapshot=snapshot,
    )


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        # Windows does not expose POSIX directory fsync. File data remains
        # fsynced, but power-loss durability of the rename is platform-limited.
        if os.name != "nt":
            raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _validated_reference(value: Mapping[str, Any]) -> Dict[str, Any]:
    reference = _fields(value, {
        "format", "formatVersion", "generationId", "relativeManifest",
        "sha256",
    }, "joint reference")
    generation_id = reference["generationId"]
    if (
        reference["format"] != _REFERENCE_FORMAT
        or type(reference["formatVersion"]) is not int
        or reference["formatVersion"] != _VERSION
        or not isinstance(generation_id, str)
        or _GENERATION_ID.fullmatch(generation_id) is None
        or reference["relativeManifest"]
        != "generations/%s/manifest.json" % generation_id
    ):
        raise JointGenerationError("joint reference identity is invalid")
    _hash(reference["sha256"], "joint manifest")
    return reference


def _previous_manifest_digest(
    root: Path, reference_value: Mapping[str, Any],
) -> str:
    reference = _validated_reference(reference_value)
    generations = root / "generations"
    if generations.is_symlink():
        raise JointGenerationError("joint generations directory is a symlink")
    path = root / reference["relativeManifest"]
    if path.parent.is_symlink():
        raise JointGenerationError("previous joint generation is a symlink")
    actual_sha, _ = _file_digest(path, "previous joint manifest")
    if actual_sha != reference["sha256"]:
        raise JointGenerationError("previous joint manifest checksum mismatch")
    _validate_manifest(
        _read_canonical_json(path, "previous joint manifest"),
        reference["generationId"],
    )
    return reference["sha256"]


def stage_joint_generation(
    root: Path, *, neural_store_root: Path,
    neural_pointer: Mapping[str, Any], substrate_store_root: Path,
    substrate_pointer: Mapping[str, Any],
    sqlite_backup: Optional[SQLiteBackupWriter] = None,
    source_manifest_sha256: str,
    parser_manifest_sha256: str, source_content_sha256: str,
    checkpoint_sequence: int, cursor: Mapping[str, Any],
    coverage: Mapping[str, Any],
    previous_reference: Optional[Mapping[str, Any]] = None,
    verified_cache: Optional[VerifiedArtifactCache] = None,
) -> Dict[str, Any]:
    """Stage and verify one immutable generation for a later brain.json commit.

    An optional SQLite cache callback must finish and close its destination
    before returning. It must use a transactional SQLite backup, not copy a
    live database file. Omitting it keeps only the shard-backed authority; the
    caller must rebuild or reconcile its derived index before using it.
    The returned reference is not committed until the caller includes it in
    an atomic ``brain.json`` replacement. A failure or crash before that point
    cannot advance the authoritative ingestion cursor.

    ``previous_reference`` must be the already committed brain.json reference.
    A cache may skip byte hashing only for unchanged files verified earlier in
    this process. Omitting the cache performs full hashing during staging.
    """

    if sqlite_backup is not None and not callable(sqlite_backup):
        raise JointGenerationError("SQLite backup callback is invalid")
    _hash(source_manifest_sha256, "source manifest")
    _hash(parser_manifest_sha256, "parser manifest")
    _hash(source_content_sha256, "source content")
    _count(checkpoint_sequence, "checkpoint sequence", minimum=1)
    safe_cursor, safe_coverage = _validate_cursor_coverage(
        cursor, coverage, source_content_sha256,
    )
    neural_state = _neural_state_from_pointer(
        neural_store_root, neural_pointer,
    )
    substrate_state = _substrate_state_from_pointer(
        substrate_store_root, substrate_pointer,
    )
    root = Path(root)
    if root.exists() and root.is_symlink():
        raise JointGenerationError("joint checkpoint root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    previous_sha = (
        _previous_manifest_digest(root, previous_reference)
        if previous_reference is not None else None
    )
    stages = root / "staging"
    generations = root / "generations"
    if stages.is_symlink() or generations.is_symlink():
        raise JointGenerationError("joint store directory is a symlink")
    stages.mkdir(exist_ok=True)
    generations.mkdir(exist_ok=True)
    _fsync_directory(root)
    generation_id = uuid.uuid4().hex
    stage = stages / (generation_id + ".stage")
    stage.mkdir(mode=0o700)
    _fsync_directory(stages)
    snapshot_spec: Optional[Dict[str, Any]] = None
    if sqlite_backup is not None:
        snapshot_path = stage / _SQLITE_NAME
        sqlite_backup(snapshot_path)
        _validate_sqlite_snapshot(snapshot_path)
        # A callback can fsync its own backup, but publication does not trust
        # it to do so. The SQLite connection must be closed by this point.
        with snapshot_path.open("rb") as handle:
            os.fsync(handle.fileno())
        snapshot_sha, snapshot_bytes = _file_digest(
            snapshot_path, "SQLite snapshot",
        )
        snapshot_spec = {
            "file": _SQLITE_NAME, "sha256": snapshot_sha,
            "bytes": snapshot_bytes,
        }
    manifest = {
        "format": _FORMAT, "formatVersion": _VERSION,
        "generationId": generation_id,
        "neuralState": neural_state,
        "neuralGenerationSha256": neural_state["generationId"],
        "substrateState": substrate_state,
        "substrateGenerationSha256": substrate_state["generationId"],
        "sqliteSnapshot": snapshot_spec,
        "checkpointSequence": checkpoint_sequence,
        "previousManifestSha256": previous_sha,
        "sourceManifestSha256": source_manifest_sha256,
        "parserManifestSha256": parser_manifest_sha256,
        "sourceParserManifestSha256": source_parser_manifest_sha256(
            source_manifest_sha256, parser_manifest_sha256,
        ),
        "sourceContentSha256": source_content_sha256,
        "cursor": safe_cursor, "coverage": safe_coverage,
    }
    manifest = _validate_manifest(manifest, generation_id)
    manifest_bytes = _canonical(manifest)
    manifest_sha256 = _sha256(manifest_bytes)
    _write_fsynced(stage / "manifest.json", manifest_bytes)
    _fsync_directory(stage)
    if verified_cache is not None:
        verified_cache.begin_validation()
    _validate_generation(
        stage, generation_id, manifest_sha256,
        source_manifest_sha256=source_manifest_sha256,
        parser_manifest_sha256=parser_manifest_sha256,
        source_content_sha256=source_content_sha256,
        neural_store_root=neural_store_root,
        substrate_store_root=substrate_store_root,
        durable=True,
        cache=verified_cache,
        allow_reuse=verified_cache is not None,
    )
    if verified_cache is not None:
        verified_cache.retain_current_refs()
    destination = generations / generation_id
    if destination.exists():
        raise JointGenerationError("joint generation identity already exists")
    os.rename(stage, destination)
    _fsync_directory(generations)
    _fsync_directory(stages)
    return {
        "format": _REFERENCE_FORMAT,
        "formatVersion": _VERSION,
        "generationId": generation_id,
        "relativeManifest": "generations/%s/manifest.json" % generation_id,
        "sha256": manifest_sha256,
    }


def recover_joint_generation(
    root: Path, committed_reference: Optional[Mapping[str, Any]], *,
    neural_store_root: Path, substrate_store_root: Path,
    source_manifest_sha256: str,
    parser_manifest_sha256: str, source_content_sha256: str,
    verified_cache: Optional[VerifiedArtifactCache] = None,
) -> Optional[JointGeneration]:
    """Validate only a brain.json-committed reference; ignore all orphans.

    ``None`` means brain.json contains no joint reference. A present but bad
    reference raises rather than silently resetting an ingestion cursor to
    zero. Source/parser/content hashes must be freshly observed by the caller.
    """

    root = Path(root)
    if root.is_symlink():
        raise JointGenerationError("joint checkpoint root must not be a symlink")
    if committed_reference is None:
        return None
    reference = _validated_reference(committed_reference)
    generation_id = reference["generationId"]
    if verified_cache is not None:
        verified_cache.begin_validation()
    generation = _validate_generation(
        root / "generations" / generation_id,
        generation_id, reference["sha256"],
        source_manifest_sha256=source_manifest_sha256,
        parser_manifest_sha256=parser_manifest_sha256,
        source_content_sha256=source_content_sha256,
        neural_store_root=neural_store_root,
        substrate_store_root=substrate_store_root,
        cache=verified_cache,
        allow_reuse=False,
    )
    if verified_cache is not None:
        verified_cache.retain_current_refs()
    return generation


def scrub_joint_generation(
    root: Path, committed_reference: Optional[Mapping[str, Any]], *,
    neural_store_root: Path, substrate_store_root: Path,
    source_manifest_sha256: str, parser_manifest_sha256: str,
    source_content_sha256: str,
    verified_cache: Optional[VerifiedArtifactCache] = None,
) -> Optional[JointGeneration]:
    """Explicit full-hash scrub of the brain.json-bound joint generation.

    The cache, if supplied, is refreshed from bytes; metadata never shortcuts
    this verification.
    """

    return recover_joint_generation(
        root, committed_reference,
        neural_store_root=neural_store_root,
        substrate_store_root=substrate_store_root,
        source_manifest_sha256=source_manifest_sha256,
        parser_manifest_sha256=parser_manifest_sha256,
        source_content_sha256=source_content_sha256,
        verified_cache=verified_cache,
    )


def orphan_stage_paths(root: Path) -> Tuple[Path, ...]:
    """List ignored stage directories for an explicit maintenance decision.

    This intentionally does not delete them or promote their cursors. The
    caller should coordinate cleanup with any live publisher before removal.
    """

    stages = Path(root) / "staging"
    if not stages.is_dir() or stages.is_symlink():
        return ()
    return tuple(sorted(
        path for path in stages.iterdir()
        if path.name.endswith(".stage") and path.is_dir()
        and not path.is_symlink()
    ))
