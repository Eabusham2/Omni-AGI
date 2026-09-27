"""Atomically move one resident substrate's vectors and assemblies to pages.

The shared SQLite file is a *working cache*, never a new brain checkpoint.
Only the caller's later v3 shard/brain save can make its learned state durable.
One fixed staging path and one fixed live path prevent repeated failed attempts
from accumulating arbitrary orphan directories. A stale path from a prior
process fails closed; recovery must reconcile it with the committed brain.
This helper moves packed rows and assemblies, not resident neuron metadata;
that requires the separate paged-neuron mapping and an atomic attach path.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import json
import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from .packed_vsa_vectors import PackedTernaryVectorView, PackedTernaryVectors
from .paged_assembly_index import PagedAssemblyIndex, _record_payload
from .paged_assembly_vector_view import PagedAssemblyVectorView
from .paged_assembly_view import PagedAssemblyView
from .paged_packed_vectors import PagedPackedVectors
from .vsa import NeuralSubstrate, SubstrateResourcePause


_STAGING_NAME = ".live-paged-staging"
_LIVE_NAME = "live-paged"
_DB_NAME = "working.sqlite3"
_RECEIPT_NAME = "receipt.json"
_OWNER_MARKER = "owner.json"
_OWNER_FORMAT = "omni-live-paged-cache-owner"
_VECTOR_ROWS = 512
_ASSEMBLY_ROWS = 128
_ASSEMBLY_BYTES = 8 * 1024 * 1024


def _reserve(
    substrate: NeuralSubstrate,
    callback: Optional[Callable[[int, str], Any]],
    size: int,
    operation: str,
) -> None:
    estimate = max(1, int(size))
    if callback is not None:
        allowed = callback(estimate, operation)
    elif substrate.growth_guard is not None:
        allowed = substrate.growth_guard(estimate)
    else:
        return
    if allowed is False:
        raise SubstrateResourcePause("live paging migration reached disk reserve")


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _clean_owned_staging(root: Path, stage: Path) -> None:
    # Only this exact private child, created by the current invocation, may be
    # removed. An existing stage is never adopted or cleaned automatically.
    if stage.parent != root or stage.name != _STAGING_NAME or stage.is_symlink():
        raise ValueError("unsafe live paging staging cleanup target")
    if stage.is_dir():
        shutil.rmtree(stage)


def _flush_assembly_window(
    index: PagedAssemblyIndex,
    window: list[Mapping[str, Any]],
) -> None:
    if not window:
        return
    with index.batch(max_rows=_ASSEMBLY_ROWS, max_payload_bytes=_ASSEMBLY_BYTES) as batch:
        for record in window:
            if not batch.upsert(record):
                raise ValueError("live paging migration found a duplicate assembly")
    window.clear()


def migrate_live_substrate_to_paged(
    substrate: NeuralSubstrate,
    cache_directory: Path,
    *,
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
    memory_reserve: Optional[Callable[[int, str], Any]] = None,
    page_size: int = 128,
) -> dict[str, Any]:
    """Stage and verify a shared vector/assembly cache, then switch atomically.

    Requires resident ``PackedTernaryVectors`` and a resident assembly list.
    The caller must serialize live neural writes for this operation. A
    new migration names ``cache_directory/live-paged/working.sqlite3``.
    An already-paged shared substrate is a no-op, including one recovered
    into a separately verified cache. A stale live
    or staging path is never treated as authority or silently overwritten.
    """

    if not isinstance(substrate, NeuralSubstrate):
        raise TypeError("live paging migration requires a neural substrate")
    if type(page_size) is not int or not 1 <= page_size <= 4096:
        raise ValueError("paged assembly page size is invalid")
    raw_root = Path(cache_directory)
    if raw_root.is_symlink():
        raise ValueError("live paging cache directory must not be a symlink")
    root = raw_root.resolve()
    stage = root / _STAGING_NAME
    live = root / _LIVE_NAME
    live_database = live / _DB_NAME
    if live.is_symlink():
        raise ValueError("live paging cache must not be a symlink")
    if (
        isinstance(substrate.assemblies, PagedAssemblyView)
        and isinstance(substrate.neuron_vectors, PagedPackedVectors)
        and substrate.assemblies.index._vectors is substrate.neuron_vectors
    ):
        actual_database = substrate.assemblies.index.path.resolve()
        owned = actual_database == live_database
        if owned:
            try:
                receipt = json.loads((live / _RECEIPT_NAME).read_text("utf-8"))
            except (OSError, UnicodeError, ValueError) as error:
                raise ValueError("live paging cache has no owned receipt") from error
            if (
                not isinstance(receipt, dict)
                or receipt.get("format") != "omni-live-paged-working-cache"
                or receipt.get("formatVersion") != 1
                or receipt.get("authoritative") is not False
                or receipt.get("sqliteFile") != _DB_NAME
                or type(receipt.get("neuronCount")) is not int
                or type(receipt.get("assemblyCount")) is not int
                or not 0 <= receipt["neuronCount"] <= len(substrate.neurons)
                or not 0 <= receipt["assemblyCount"] <= len(substrate.assemblies)
            ):
                raise ValueError("live paging cache receipt disagrees with live state")
        substrate._validate_packed_vector_identity()
        return {
            "path": str(actual_database),
            "neuronCount": len(substrate.neurons),
            "assemblyCount": len(substrate.assemblies),
            "alreadyPaged": True,
            "ownedCache": owned,
        }
    if (
        not isinstance(substrate.neuron_vectors, PackedTernaryVectors)
        or not isinstance(substrate.assemblies, list)
        or not isinstance(substrate.assembly_vectors, PackedTernaryVectorView)
    ):
        raise ValueError("live paging migration needs one resident packed substrate")
    substrate._validate_packed_vector_identity()
    if stage.exists() or stage.is_symlink() or live.exists() or live.is_symlink():
        raise FileExistsError("live paging cache has an unverified prior path")
    source_vectors = substrate.neuron_vectors
    source_assemblies = substrate.assemblies
    source_view = substrate.assembly_vectors
    vector_count = len(source_vectors)
    assembly_count = len(source_assemblies)
    state_revision = substrate.state_revision
    _reserve(substrate, disk_reserve, 65536, "live paging staging directory")
    root.mkdir(parents=True, exist_ok=True)
    _fsync(root.parent)
    stage.mkdir(exist_ok=False)
    _fsync(root)
    published = False
    try:
        marker = NeuralSubstrate._canonical_json({
            "format": _OWNER_FORMAT,
            "formatVersion": 1,
            "authoritative": False,
            "sqliteFile": _DB_NAME,
        })
        _reserve(substrate, disk_reserve, len(marker) + 4096, "live paging owner marker")
        with (stage / _OWNER_MARKER).open("xb") as handle:
            handle.write(marker)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync(stage)
        staged_database = stage / _DB_NAME
        def reserve_disk(size: int, operation: str) -> None:
            _reserve(substrate, disk_reserve, size, operation)

        index = PagedAssemblyIndex(
            staged_database,
            dimensions=substrate.space.dimensions,
            seed=substrate.space.seed,
            zero_deadband=source_vectors.zero_deadband,
            disk_reserve=reserve_disk,
            memory_reserve=memory_reserve,
        )
        assert isinstance(index._vectors, PagedPackedVectors)
        ids: list[str] = []
        imported = 0

        def import_vectors() -> None:
            nonlocal imported
            if not ids:
                return
            metadata, tensors = source_vectors.export_state(keys=ids)
            imported += index._vectors.import_state(metadata, tensors)
            ids.clear()

        for identifier in source_vectors:
            ids.append(identifier)
            if len(ids) == _VECTOR_ROWS:
                import_vectors()
        import_vectors()
        if imported != vector_count or len(index._vectors) != vector_count:
            raise ValueError("live paging migration missed packed neuron rows")

        window: list[Mapping[str, Any]] = []
        window_bytes = 0
        migrated_assemblies = 0
        for record in source_assemblies:
            identifier, fingerprint, payload, _digest = _record_payload(record)
            if identifier not in index._vectors:
                raise ValueError("assembly lacks its shared packed neuron row")
            charge = len(payload) + len(identifier) + len(fingerprint)
            if charge > _ASSEMBLY_BYTES:
                raise SubstrateResourcePause("assembly record exceeds migration window")
            if window and (
                len(window) >= _ASSEMBLY_ROWS or window_bytes + charge > _ASSEMBLY_BYTES
            ):
                _flush_assembly_window(index, window)
                window_bytes = 0
            window.append(record)
            window_bytes += charge
            migrated_assemblies += 1
        _flush_assembly_window(index, window)
        if migrated_assemblies != assembly_count or index.count() != assembly_count:
            raise ValueError("live paging migration missed assembly metadata")
        if (
            substrate.state_revision != state_revision
            or substrate.neuron_vectors is not source_vectors
            or substrate.assemblies is not source_assemblies
            or len(source_vectors) != vector_count
            or len(source_assemblies) != assembly_count
        ):
            raise ValueError("live substrate changed during paging migration")

        # Compare exact canonical row bytes/counters and structural metadata
        # before switching authority. This reads one row at a time.
        for identifier in source_vectors:
            if (
                index._vectors.packed_row(identifier) != source_vectors.packed_row(identifier)
                or index._vectors.update_count(identifier) != source_vectors.update_count(identifier)
            ):
                raise ValueError("live paging migration changed a packed vector")
        for record in source_assemblies:
            stored = index.get_by_id(str(record.get("id", "")))
            if stored != record:
                raise ValueError("live paging migration changed assembly metadata")

        with sqlite3.connect(staged_database) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        _fsync(staged_database)
        receipt = NeuralSubstrate._canonical_json({
            "format": "omni-live-paged-working-cache",
            "formatVersion": 1,
            "authoritative": False,
            "sqliteFile": _DB_NAME,
            "neuronCount": vector_count,
            "assemblyCount": assembly_count,
        })
        _reserve(substrate, disk_reserve, len(receipt) + 4096, "live paging receipt")
        with (stage / _RECEIPT_NAME).open("xb") as handle:
            handle.write(receipt)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync(stage)
        if live.exists() or live.is_symlink():
            raise FileExistsError("live paging cache was created concurrently")
        _reserve(substrate, disk_reserve, 8192, "live paging cache publication")
        os.rename(stage, live)
        published = True
        _fsync(root)
        final_index = PagedAssemblyIndex(
            live_database,
            dimensions=substrate.space.dimensions,
            seed=substrate.space.seed,
            zero_deadband=source_vectors.zero_deadband,
            disk_reserve=reserve_disk,
            memory_reserve=memory_reserve,
        )
        final_vectors = final_index._vectors
        if final_vectors is None:
            raise ValueError("live paging migration lost packed vector authority")
        final_assemblies = PagedAssemblyView(final_index, page_size=page_size)
        final_view = PagedAssemblyVectorView(final_index, final_vectors)
        substrate.neuron_vectors = final_vectors
        substrate.assemblies = final_assemblies
        substrate.assembly_vectors = final_view
        try:
            substrate._validate_packed_vector_identity()
        except BaseException:
            substrate.neuron_vectors = source_vectors
            substrate.assemblies = source_assemblies
            substrate.assembly_vectors = source_view
            raise
        return {
            "path": str(live_database),
            "neuronCount": vector_count,
            "assemblyCount": assembly_count,
            "alreadyPaged": False,
            "ownedCache": True,
        }
    except BaseException:
        if published and live.is_dir() and not stage.exists():
            os.rename(live, stage)
            _fsync(root)
        _clean_owned_staging(root, stage)
        raise


def prune_abandoned_live_paging_cache(
    cache_directory: Path,
    engine_path: Path,
    *,
    verified_rebuild: Any,
    target: str,
    no_consumers: Callable[[Path], bool],
) -> dict[str, Any]:
    """Remove one exact old derived path after verified committed recovery.

    ``verified_rebuild`` must be the receipt returned by
    ``rebuild_committed_paged_cache`` for the *current* ``brain.json``. The
    host must affirm that no thread/process still uses ``target``. This
    operation never scans broadly or adopts a stale cache as neural state.
    Choose ``target='staging'`` or ``target='live'`` explicitly.
    """

    from .committed_paged_cache import RebuiltPagedCache

    if not isinstance(verified_rebuild, RebuiltPagedCache):
        raise TypeError("cache pruning requires a verified paged rebuild receipt")
    if target not in {"staging", "live"}:
        raise ValueError("cache pruning target must be staging or live")
    if not callable(no_consumers):
        raise TypeError("cache pruning requires a no-consumers callback")
    raw_root = Path(cache_directory)
    if raw_root.is_symlink():
        raise ValueError("live paging cache directory must not be a symlink")
    root = raw_root.resolve()
    candidate = root / (_STAGING_NAME if target == "staging" else _LIVE_NAME)
    if not candidate.exists() and not candidate.is_symlink():
        return {"removed": False, "path": str(candidate)}
    if candidate.is_symlink() or not candidate.is_dir() or candidate.parent != root:
        raise ValueError("cache pruning target is not an owned directory")
    marker_path = candidate / _OWNER_MARKER
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError("cache pruning target has no owned marker")
    try:
        marker = json.loads(marker_path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("cache pruning owner marker is invalid") from error
    if marker != {
        "format": _OWNER_FORMAT,
        "formatVersion": 1,
        "authoritative": False,
        "sqliteFile": _DB_NAME,
    }:
        raise ValueError("cache pruning owner marker is invalid")
    if target == "live":
        receipt_path = candidate / _RECEIPT_NAME
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ValueError("live cache has no completed receipt")
    engine = Path(engine_path).resolve()
    brain_bytes = (engine / "brain.json").read_bytes()
    if hashlib.sha256(brain_bytes).hexdigest() != verified_rebuild.brain_sha256:
        raise ValueError("verified cache rebuild does not match current brain")
    try:
        brain = json.loads(brain_bytes)
    except (UnicodeError, ValueError) as error:
        raise ValueError("committed brain pointer is invalid") from error
    substrate = brain.get("substrate") if isinstance(brain, dict) else None
    pointer = substrate.get("persistence") if isinstance(substrate, dict) else None
    if not isinstance(pointer, dict):
        raise ValueError("committed brain has no substrate pointer")
    generation, _references = NeuralSubstrate._generation_references(
        engine / "substrate", pointer
    )
    if (
        generation != verified_rebuild.generation_sha256
        or pointer.get("generationManifestSha256")
        != verified_rebuild.generation_manifest_sha256
    ):
        raise ValueError("verified cache rebuild does not match committed generation")
    rebuilt_path = Path(verified_rebuild.path).resolve()
    if not rebuilt_path.is_file() or rebuilt_path == candidate / _DB_NAME:
        raise ValueError("verified replacement cache is missing or is the target")
    rebuild_receipt_path = rebuilt_path.parent / _RECEIPT_NAME
    if rebuild_receipt_path.is_symlink() or not rebuild_receipt_path.is_file():
        raise ValueError("verified replacement cache has no receipt")
    try:
        rebuilt_receipt = json.loads(rebuild_receipt_path.read_text("utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("verified replacement receipt is invalid") from error
    if (
        rebuilt_receipt.get("format") != "omni-derived-paged-cache"
        or rebuilt_receipt.get("formatVersion") != 1
        or rebuilt_receipt.get("generationSha256") != generation
        or rebuilt_receipt.get("brainSha256") != verified_rebuild.brain_sha256
        or rebuilt_receipt.get("sqliteFile") != _DB_NAME
    ):
        raise ValueError("verified replacement receipt disagrees with brain")
    if no_consumers(candidate) is not True:
        raise RuntimeError("live paging cache still has consumers")
    shutil.rmtree(candidate)
    _fsync(root)
    return {"removed": True, "path": str(candidate)}
