"""Bounded v3 checkpoint writer for paged assembly metadata and packed rows.

This checkpoint path publishes exact neuron, assembly, and synapse shards (at
most 512 records each). A separate bounded synapse plan handles changed dirty
groups; silently reusing their old descriptors would lose learning. The caller
must serialize working-state writes and then commit the returned pointer in
``brain.json``. The substrate root pointer is only a convenience pointer,
never recovery authority.

Clean generation-bound shared caches publish only changed complete stable
groups, reusing untouched descriptors. Initial migration, stale journal
recovery, and a global neuron decay epoch retain explicit full/group rewrite
costs. The v3 descriptor manifest still costs O(number of shards).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import sqlite3
import re
import subprocess
import sys
from collections import namedtuple
from contextlib import closing
from functools import lru_cache
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

import torch

from .packed_vsa_vectors import PackedTernaryVectorView, PackedTernaryVectors
from .paged_assembly_index import _record_payload
from .paged_assembly_vector_view import PagedAssemblyVectorView
from .paged_assembly_view import PagedAssemblyView
from .paged_neuron_metadata import PagedNeuronMetadata
from .paged_packed_vectors import PagedPackedVectors
from .paged_dirty_journal import (
    DirtyJournalResourcePause, DirtyShardPlan, rebuild_generation_journal, source_stamp,
)
from .authenticated_paged_cache import active_blob_session, cache_session
from .paged_synapse_endpoints import SynapseEndpointIndex, SynapseEndpointPlan
from .persistence import atomic_save_tensors
from .substrate_inspection import neuron_shard_inspection
from .vsa import (
    NeuralSubstrate,
    SubstrateResourcePause,
    _SUBSTRATE_STORE_FORMAT,
    _SUBSTRATE_STORE_VERSION,
)


_BUCKETS = "0123456789abcdef"
_NEURON_FIELDS = frozenset({
    "id", "neuron_id", "label", "region", "activation", "importance",
    "uncertainty", "exposures", "created_at", "last_activated_at", "aliases",
    "memory_strength", "retention_score", "activity_score", "plasticity_score",
    "reinforcement_drive", "unfinished_score", "unfinished", "settling_signals",
    "last_settled_at",
})
_MAX_BLOB_BYTES = 64 * 1024 * 1024
_MAX_BUFFER_BYTES = 8 * 1024 * 1024
_PROOF_STATS = {"hits": 0, "misses": 0}
_ProofCacheInfo = namedtuple("ProofCacheInfo", "hits misses maxsize currsize")


def _canonical(value: Any) -> bytes:
    return NeuralSubstrate._canonical_json(value)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _thaw(value: Any) -> Any:
    """Turn a paged view's recursively read-only JSON into detached JSON."""

    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verified_identity_sha(
    path: str, device: int, inode: int, size: int, mtime_ns: int, ctime_ns: int
) -> str:
    """Hash immutable content once per process/file identity, then reuse proof.

    This is only a save-time optimization. Recovery and explicit scrub must
    still stream-hash the referenced shards; stat identity is not a defense
    against a malicious filesystem that forges all change timestamps.
    """

    target = Path(path)
    identity = (device, inode, size, mtime_ns, ctime_ns)
    session = active_blob_session()
    known = session.lookup_blob(target, identity) if session is not None else None
    if known is not None:
        _PROOF_STATS["hits"] += 1
        return known
    _PROOF_STATS["misses"] += 1
    observed = NeuralSubstrate._file_sha256(target)
    after = target.stat()
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("substrate blob changed during cryptographic proof read")
    if session is not None:
        session.remember_blob(target, observed, identity)
    return observed


def _proof_cache_info() -> tuple:
    # Compatibility diagnostics: proofs are disk-backed, not an entry-capped LRU.
    return _ProofCacheInfo(_PROOF_STATS["hits"], _PROOF_STATS["misses"], None, 0)


def _clear_proof_stats() -> None:
    _PROOF_STATS.update(hits=0, misses=0)


_verified_identity_sha.cache_info = _proof_cache_info
_verified_identity_sha.cache_clear = _clear_proof_stats


@lru_cache(maxsize=64)
def _strong_local_change_times(device: int, directory: str, platform: str) -> bool:
    """Conservative allowlist; Windows birth times/network/unknown FS rehash."""

    try:
        if platform == "darwin":
            result = subprocess.run(
                ["/sbin/mount"], check=True, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, timeout=5, text=True,
            )
            if len(result.stdout) > 256 * 1024:
                return False
            mounts = []
            for line in result.stdout.splitlines():
                match = re.fullmatch(r".+ on (.+) \(([^)]+)\)", line)
                if match is not None:
                    mount_path, raw = match.groups()
                    flags = raw.split(", ")
                    if directory == mount_path or directory.startswith(mount_path.rstrip("/") + "/"):
                        mounts.append((len(mount_path), flags))
            if not mounts:
                return False
            flags = max(mounts, key=lambda item: item[0])[1]
            # HFS-family/coarse timestamps are deliberately not proof sources.
            return flags[0] == "apfs" and "local" in flags
        # No tested reliable Linux inode change-cookie contract is available
        # here. A filesystem type alone does not prove timestamp granularity.
        # It therefore uses actual hashes, just like unknown/network/Windows.
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return False


def _identity_proof_supported(path: Path, details: os.stat_result) -> bool:
    return (
        sys.platform == "darwin"
        and details.st_ctime_ns > 0 and details.st_mtime_ns > 0
        and details.st_ctime_ns % 1_000_000_000 != 0
        and _strong_local_change_times(details.st_dev, str(path.parent.resolve()), sys.platform)
    )


def remember_verified_blob(
    path: Path, checksum: str, *, before: Optional[os.stat_result] = None,
) -> None:
    """Reuse an already checked read/publication, not another whole blob read."""

    details = path.stat()
    identity = lambda item: (
        item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns
    )
    if before is not None and identity(before) != identity(details):
        raise ValueError("substrate blob changed during checked read")
    if not _identity_proof_supported(path, details):
        return
    session = active_blob_session()
    if session is not None:
        session.remember_blob(path, checksum, identity(details))


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
        raise SubstrateResourcePause(
            "paged substrate checkpoint paused at the host disk reserve"
        )


def _verified_file(path: Path, checksum: str, size: int) -> None:
    if path.is_symlink():
        raise ValueError("substrate blob must not be a symlink")
    details = path.stat()
    if not stat.S_ISREG(details.st_mode) or details.st_size != size:
        raise ValueError("substrate blob size or type mismatch")
    identity = (
        details.st_dev, details.st_ino, details.st_size,
        details.st_mtime_ns, details.st_ctime_ns,
    )
    observed = (
        _verified_identity_sha(str(path), *identity)
        if _identity_proof_supported(path, details)
        else NeuralSubstrate._file_sha256(path)
    )
    after = path.stat()
    if identity != (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns,
    ):
        raise ValueError("substrate blob changed while being verified")
    if observed != checksum:
        raise ValueError("substrate blob checksum mismatch")


def _verified_spec(root: Path, spec: Mapping[str, Any], suffix: str) -> None:
    checksum = spec.get("sha256")
    size = spec.get("bytes")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(char not in _BUCKETS for char in checksum)
        or type(size) is not int
        or size < 0
        or spec.get("path") != "blobs/%s%s" % (checksum, suffix)
    ):
        raise ValueError("substrate blob reference is invalid")
    _verified_file(NeuralSubstrate._safe_store_path(root, spec["path"]), checksum, size)


def _link_immutable(temporary: Path, destination: Path, checksum: str, size: int) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        _verified_file(destination, checksum, size)
    else:
        # Remove the owned temporary link FIRST. That link-count mutation
        # changes inode change-time; a proof from before it would immediately
        # become stale and cause a full unchanged-blob rehash on the next save.
        temporary.unlink()
        _fsync_directory(destination.parent)
        # Prove the published inode with actual bytes before storing a proof;
        # no unchecked link/publication timestamp becomes content authority.
        _verified_file(destination, checksum, size)


def _write_immutable_bytes(
    substrate: NeuralSubstrate,
    root: Path,
    relative: str,
    payload: bytes,
    callback: Optional[Callable[[int, str], Any]],
    *,
    max_blob_bytes: Optional[int] = None,
) -> None:
    if max_blob_bytes is not None and len(payload) > max_blob_bytes:
        raise SubstrateResourcePause("paged substrate shard exceeds byte window")
    destination = NeuralSubstrate._safe_store_path(root, relative)
    checksum = _sha(payload)
    if destination.exists() or destination.is_symlink():
        _verified_file(destination, checksum, len(payload))
        return
    _reserve(substrate, callback, len(payload) * 2 + 4096, "substrate immutable blob")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(destination.parent.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".substrate-", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _link_immutable(temporary, destination, checksum, len(payload))
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_blob(
    substrate: NeuralSubstrate,
    root: Path,
    value: dict[str, Any],
    callback: Optional[Callable[[int, str], Any]],
) -> dict[str, Any]:
    payload = _canonical(value)
    checksum = _sha(payload)
    relative = "blobs/%s.json" % checksum
    _write_immutable_bytes(
        substrate, root, relative, payload, callback, max_blob_bytes=_MAX_BLOB_BYTES
    )
    return {"path": relative, "sha256": checksum, "bytes": len(payload)}


def _tensor_state_checksum(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        row = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(row.shape)).encode("ascii"))
        digest.update(str(row.dtype).encode("ascii"))
        digest.update(row.numpy().tobytes())
    return digest.hexdigest()


def _write_tensor_blob(
    substrate: NeuralSubstrate,
    root: Path,
    tensors: Mapping[str, torch.Tensor],
    reusable: Optional[Mapping[str, Any]],
    callback: Optional[Callable[[int, str], Any]],
) -> dict[str, Any]:
    state_checksum = _tensor_state_checksum(tensors)
    if isinstance(reusable, Mapping) and reusable.get("stateSha256") == state_checksum:
        _verified_spec(root, reusable, ".safetensors")
        return dict(reusable)
    estimate = 65536 + sum(value.numel() * value.element_size() for value in tensors.values())
    if estimate > _MAX_BLOB_BYTES:
        raise SubstrateResourcePause("paged substrate tensor shard exceeds byte window")
    _reserve(substrate, callback, estimate * 2, "substrate tensor shard")
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".substrate-", suffix=".tmp", dir=blobs)
    os.close(descriptor)
    temporary = Path(name)
    try:
        atomic_save_tensors(
            temporary,
            tensors,
            metadata={
                "format": _SUBSTRATE_STORE_FORMAT,
                "formatVersion": str(_SUBSTRATE_STORE_VERSION),
            },
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        size = temporary.stat().st_size
        if size > _MAX_BLOB_BYTES:
            raise SubstrateResourcePause("paged substrate tensor shard exceeds byte window")
        checksum = NeuralSubstrate._file_sha256(temporary)
        relative = "blobs/%s.safetensors" % checksum
        destination = NeuralSubstrate._safe_store_path(root, relative)
        _link_immutable(temporary, destination, checksum, size)
        return {
            "path": relative,
            "sha256": checksum,
            "bytes": size,
            "stateSha256": state_checksum,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _prior_generation(root: Path, pointer: Mapping[str, Any]) -> dict[str, Any]:
    generation_id, _references = NeuralSubstrate._generation_references(
        root, dict(pointer)
    )
    path = NeuralSubstrate._safe_store_path(
        root, "generations/%s/manifest.json" % generation_id
    )
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict) or value.get("formatVersion") != 3:
        raise ValueError("prior substrate generation is not v3")
    return value


def _neuron_record(record_id: str, record: Mapping[str, Any]) -> dict[str, Any]:
    record = _thaw(record)
    if (
        not isinstance(record_id, str)
        or not record_id
        or not isinstance(record, Mapping)
        or record.get("id") != record_id
        or set(record) - _NEURON_FIELDS
    ):
        raise ValueError("neuron metadata is invalid or contains non-structural text")
    label = record.get("label")
    if label is not None and (
        not isinstance(label, str)
        or len(label) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in label)
    ):
        raise ValueError("neuron label is not bounded structural metadata")
    if record.get("neuron_id", record_id) != record_id:
        raise ValueError("neuron metadata identity differs from its mapping key")
    region = record.get("region")
    if region is not None and (
        not isinstance(region, str) or len(region) > 64
        or any(ord(char) < 32 or ord(char) == 127 for char in region)
    ):
        raise ValueError("neuron region is not bounded structural metadata")
    aliases = record.get("aliases", [])
    if (
        not isinstance(aliases, list) or len(aliases) > 32
        or any(
            not isinstance(alias, str) or len(alias) > 256
            or any(ord(char) < 32 or ord(char) == 127 for char in alias)
            for alias in aliases
        )
    ):
        raise ValueError("neuron aliases are not bounded structural metadata")
    signals = record.get("settling_signals")
    if signals is not None and (
        not isinstance(signals, dict) or len(signals) > 64
        or any(
            not isinstance(name, str) or len(name) > 64
            or not isinstance(value, (int, float)) or isinstance(value, bool)
            for name, value in signals.items()
        )
    ):
        raise ValueError("neuron settling signals are not bounded numeric metadata")
    settled_at = record.get("last_settled_at")
    if settled_at is not None and (
        not isinstance(settled_at, str) or len(settled_at) > 64
        or any(ord(char) < 32 or ord(char) == 127 for char in settled_at)
    ):
        raise ValueError("neuron settling timestamp is invalid")
    return json.loads(_canonical(dict(record)))


def _replace_pointer(
    substrate: NeuralSubstrate,
    path: Path,
    payload: bytes,
    callback: Optional[Callable[[int, str], Any]],
) -> None:
    _reserve(substrate, callback, 2 * len(payload) + 4096, "substrate pointer")
    descriptor, name = tempfile.mkstemp(prefix=".manifest-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _check_paged_authority(substrate: NeuralSubstrate) -> None:
    """Check identities/counts now; shard iteration verifies every emitted row.

    Calling the resident ``_validate_packed_vector_identity`` here would do
    an extra full-neuron and full-assembly scan before the writer's own exact
    streamed validation. That is not a cheap precondition on a paged brain.
    """

    vectors = substrate.neuron_vectors
    assembly_vectors = substrate.assembly_vectors
    assemblies = substrate.assemblies
    if (
        getattr(vectors, "packed_authoritative", False) is not True
        or getattr(vectors, "dimensions", None) != substrate.space.dimensions
        or not callable(getattr(vectors, "export_state", None))
        or not isinstance(assembly_vectors, (PackedTernaryVectorView, PagedAssemblyVectorView))
        or assembly_vectors.backing is not vectors
        or len(vectors) != len(substrate.neurons)
        or len(assembly_vectors) != len(assemblies)
    ):
        raise ValueError("paged substrate has inconsistent packed authority or counts")
    if isinstance(assembly_vectors, PagedAssemblyVectorView) and assembly_vectors.index is not assemblies.index:
        raise ValueError("paged assembly vector view has a different index")
    if assemblies.index._vectors is not None:
        status = assemblies.index.status()
        if status["packedVectorRows"] != status["count"]:
            raise ValueError("paged assembly membership lacks shared packed neuron rows")


def _shared_paged_path(substrate: NeuralSubstrate) -> Optional[Path]:
    if not (
        isinstance(substrate.neurons, PagedNeuronMetadata)
        and isinstance(substrate.neuron_vectors, PagedPackedVectors)
        and isinstance(substrate.assemblies, PagedAssemblyView)
        and substrate.assemblies.index._vectors is substrate.neuron_vectors
    ):
        return None
    path = substrate.assemblies.index.path.resolve()
    if (
        substrate.neurons.path.resolve() != path
        or substrate.neuron_vectors.path.resolve() != path
    ):
        return None
    return path


def _emit_record_group(
    substrate: NeuralSubstrate, store: Path, kind: str, bucket: str, part: int,
    group: list[tuple[str, dict[str, Any]]], reusable: Optional[Mapping[str, Any]],
    callback: Optional[Callable[[int, str], Any]],
) -> dict[str, Any]:
    ordered = sorted(group, key=lambda item: item[0])
    ids = [identifier for identifier, _record in ordered]
    if not ids or ids != sorted(set(ids)) or len(ids) > 512:
        raise ValueError("paged shard IDs or bounded membership are invalid")
    records = [record for _identifier, record in ordered]
    payload: dict[str, Any] = {
        "kind": kind, "ids": ids, "records": records, "vectorIds": ids,
    }
    tensors = None
    if kind == "neurons":
        vector_metadata, tensors = substrate.neuron_vectors.export_state(keys=ids)
        if vector_metadata.get("ids") != ids or set(tensors) != {
            "packed_rows", "update_counters_le"
        }:
            raise ValueError("paged vector shard export differs from neuron IDs")
        PackedTernaryVectors.from_state(vector_metadata, tensors)
        payload["packedVectorState"] = vector_metadata
    else:
        payload["vectorStorage"] = "shared-neuron-packed"
    return {
        "kind": kind, "bucket": bucket, "part": part, "count": len(ids),
        "records": _write_json_blob(substrate, store, payload, callback),
        "tensors": (
            _write_tensor_blob(substrate, store, tensors, reusable, callback)
            if tensors is not None else None
        ),
        **({"inspection": neuron_shard_inspection(records)} if kind == "neurons" else {}),
    }


def _incremental_record_descriptors(
    substrate: NeuralSubstrate, store: Path, prior: Mapping[str, Any],
    plan: DirtyShardPlan, callback: Optional[Callable[[int, str], Any]],
) -> list[dict[str, Any]]:
    descriptors = {
        (item["kind"], item["bucket"], item["part"]): dict(item)
        for item in prior["shards"] if item["kind"] in {"neurons", "assemblies"}
    }
    changed: set[tuple[str, str, int]] = set()
    for group in plan.groups():
        key = (group.kind, group.bucket, group.part)
        changed.add(key)
        reusable = descriptors.get(key, {}).get("tensors")
        if not group.record_ids:
            descriptors.pop(key, None)
            continue
        records: list[tuple[str, dict[str, Any]]] = []
        charge = 0
        for identifier in group.record_ids:
            if group.kind == "neurons":
                record = _neuron_record(identifier, substrate.neurons[identifier])
            else:
                found = substrate.assemblies.index.get_by_id_with_ordinal(identifier)
                if found is None:
                    raise ValueError("dirty assembly group is missing a live record")
                ordinal, raw = found
                if identifier not in substrate.neurons or identifier not in substrate.neuron_vectors:
                    raise ValueError("paged assembly lacks its shared neuron row")
                _identifier, _fingerprint, payload, _digest = _record_payload(raw)
                record = json.loads(payload)
                record["__persistence_ordinal"] = ordinal
            charge += len(_canonical(record)) + len(identifier) + 128
            if charge > _MAX_BUFFER_BYTES:
                # Stable membership must not silently drop or repartition IDs.
                raise SubstrateResourcePause("changed paged group exceeds byte window")
            records.append((identifier, record))
        descriptors[key] = _emit_record_group(
            substrate, store, group.kind, group.bucket, group.part, records,
            reusable, callback,
        )
    for key, descriptor in descriptors.items():
        if key not in changed:
            _verified_spec(store, descriptor["records"], ".json")
            if descriptor.get("tensors") is not None:
                _verified_spec(store, descriptor["tensors"], ".safetensors")
    plan.assert_unchanged()
    return list(descriptors.values())


def commit_paged_substrate_generation(
    substrate: NeuralSubstrate, store_root: Path, brain_json_path: Path,
    *, disk_reserve: Optional[Callable[[int, str], Any]] = None,
) -> dict[str, Any]:
    """Finalize a writer's cache ONLY after the real atomic brain commit.

    Production calls this immediately after replacing brain.json. Recovery
    never adopts this pending Python object or a convenience root pointer;
    it rebuilds a fresh cache from the independently verified brain pointer.
    """

    pending = getattr(substrate, "_pending_paged_checkpoint", None)
    if pending is None:
        return {"incremental": False, "reason": "no shared paged checkpoint"}
    path, pointer, stamp, plan, state_revision, endpoint_plan, synapses, synapse_revision, persistence_revision = pending
    brain_path = Path(brain_json_path)
    if brain_path.is_symlink() or not brain_path.is_file():
        raise ValueError("paged journal needs the real committed brain.json")
    brain_bytes = brain_path.read_bytes()
    brain = json.loads(brain_bytes)
    committed = brain.get("substrate", {}).get("persistence") if isinstance(brain, dict) else None
    if committed != pointer or substrate.persistence_manifest != pointer:
        raise ValueError("brain.json has not committed the pending paged generation")

    def verify() -> bool:
        return (
            brain_path.read_bytes() == brain_bytes
            and substrate.state_revision == state_revision
            and _shared_paged_path(substrate) == path
            and substrate.synapses is synapses
            and getattr(synapses, "graph_revision", None) == synapse_revision
            and getattr(synapses, "persistence_revision", None) == persistence_revision
        )

    def reserve(size: int, operation: str) -> None:
        _reserve(substrate, disk_reserve, size, operation)

    if plan is not None:
        plan.rebase(
            pointer, verify_brain_commit=verify, disk_reserve=reserve,
            additional_rebase=(
                lambda connection: endpoint_plan.rebase_in_connection(connection, pointer)
            ) if endpoint_plan is not None else None,
        )
    else:
        rebuild_generation_journal(
            path, store_root, pointer, expected_stamp=stamp,
            verify_brain_commit=verify, disk_reserve=reserve,
            additional_rebase=(
                lambda connection: endpoint_plan.rebase_in_connection(connection, pointer)
            ) if endpoint_plan is not None else None,
        )
    del substrate._pending_paged_checkpoint
    session = cache_session(substrate.assemblies.index)
    return {
        "incremental": plan is not None, "generationSha256": pointer["activeGeneration"],
        "synapseHotGroupsReindexed": endpoint_plan.reindexed_groups if endpoint_plan is not None else None,
        "endpointLookupSelective": endpoint_plan.selective if endpoint_plan is not None else False,
        "fileProofStorage": "process-authenticated-sqlite",
        "fileProofHits": session.proof_hits, "fileProofMisses": session.proof_misses,
        "fileProofWritePauses": session.proof_write_pauses,
        "fileProofSqliteWindowBytes": session.sqlite_window_bytes,
    }


def _write_paged_substrate_generation(
    substrate: NeuralSubstrate,
    root: Path,
    *,
    records_per_shard: int = 512,
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
) -> dict[str, Any]:
    """Publish exact v3 shards from a paged assembly view and packed rows.

    Record buffers contain at most 512 rows per hash bucket. The manifest's
    descriptor list scales with shard count, never record count. Unchanged
    synapse shards are reused; changed groups are persisted exactly.
    A ready shared-store journal reads only changed complete stable groups.
    Unbound/old caches retain the exact full-scan fallback until their first
    brain.json commit creates a safe journal. Global decay rewrites every
    neuron group without treating logically changed rows as unchanged.
    Returned pointer is suitable for ``NeuralSubstrate.load_sharded`` and for
    the caller's later atomic ``brain.json`` commit.
    """

    if not isinstance(substrate, NeuralSubstrate):
        raise TypeError("paged writer requires a neural substrate")
    if not isinstance(substrate.assemblies, PagedAssemblyView):
        raise ValueError("paged writer requires a paged assembly view")
    if type(records_per_shard) is not int or not 1 <= records_per_shard <= 512:
        raise ValueError("paged writer shard size must be between 1 and 512")
    indexed_vectors = getattr(substrate.assemblies.index, "_vectors", None)
    if indexed_vectors is not None and indexed_vectors is not substrate.neuron_vectors:
        raise ValueError("paged assembly index must share the neuron vector store")
    _check_paged_authority(substrate)
    if Path(root).is_symlink():
        raise ValueError("substrate store must not be a symlink")
    store = Path(root).resolve()
    store_was_missing = not store.exists()
    if store_was_missing:
        _reserve(substrate, disk_reserve, 16384, "substrate shard directories")
    for directory in (store / "blobs", store / "generations"):
        if directory.is_symlink():
            raise ValueError("substrate shard directory must not be a symlink")
        if not directory.exists():
            _reserve(substrate, disk_reserve, 8192, "substrate shard directory")
    (store / "blobs").mkdir(parents=True, exist_ok=True)
    (store / "generations").mkdir(parents=True, exist_ok=True)
    if store_was_missing:
        _fsync_directory(store.parent)
    _fsync_directory(store)
    prior_pointer = substrate.persistence_manifest
    prior = _prior_generation(store, prior_pointer) if prior_pointer else None
    shared_path = _shared_paged_path(substrate)
    initial_stamp = None
    dirty_plan = None
    if shared_path is not None:
        with closing(sqlite3.connect(shared_path)) as connection:
            initial_stamp = source_stamp(connection)
        if prior_pointer is not None:
            try:
                candidate = DirtyShardPlan(
                    shared_path, prior_pointer,
                    disk_reserve=lambda size, operation: _reserve(
                        substrate, disk_reserve, size, operation
                    ),
                )
                if candidate.records_per_shard == records_per_shard:
                    dirty_plan = candidate
            except DirtyJournalResourcePause as error:
                raise SubstrateResourcePause(str(error)) from error
            except ValueError:
                # Never use absent/stale/building membership to skip rows.
                # A successful full brain commit will reconstruct the journal.
                pass
    from .bounded_synapse_persistence import (
        BoundedSynapseShardPlan,
        PagedHotNodeIds,
    )

    endpoint_plan = None
    if shared_path is not None:
        session = active_blob_session()
        assert session is not None
        endpoint_index = getattr(substrate.assemblies.index, "_synapse_endpoint_index", None)
        if not isinstance(endpoint_index, SynapseEndpointIndex) or endpoint_index.session is not session:
            endpoint_index = SynapseEndpointIndex(session)
            substrate.assemblies.index._synapse_endpoint_index = endpoint_index
        prior_synapses = [item for item in prior["shards"] if item["kind"] == "synapses"] if prior else []
        endpoint_plan = SynapseEndpointPlan(
            endpoint_index, store, prior_pointer, prior_synapses,
            incremental_membership=dirty_plan is not None,
            forward_manifest=getattr(substrate.synapses, "_forward_index_manifest", None),
        )

    synapse_plan = BoundedSynapseShardPlan(
        substrate,
        store,
        records_per_shard=records_per_shard,
        write_json_blob=lambda payload: _write_json_blob(
            substrate, store, payload, disk_reserve
        ),
        write_tensor_blob=lambda tensors, reusable: _write_tensor_blob(
            substrate, store, tensors, reusable, disk_reserve
        ),
        disk_reserve=disk_reserve,
        verify_reusable=lambda spec, suffix: _verified_spec(store, spec, suffix),
        endpoint_plan=endpoint_plan,
    )
    previous_tensors = {
        (item["kind"], item["bucket"], item["part"]): item.get("tensors")
        for item in prior.get("shards", [])
        if isinstance(item, dict) and item.get("kind") == "neurons"
    } if prior is not None else {}
    initial_state_revision = substrate.state_revision
    initial_index = substrate.assemblies.index.status()
    neuron_status = getattr(substrate.neurons, "status", None)
    initial_neurons = neuron_status() if callable(neuron_status) else None
    vector_status = getattr(substrate.neuron_vectors, "status", None)
    initial_vectors = vector_status() if callable(vector_status) else None
    shards: list[dict[str, Any]] = []
    observed = {"neurons": 0, "assemblies": 0}
    if dirty_plan is not None:
        assert prior is not None
        shards = _incremental_record_descriptors(
            substrate, store, prior, dirty_plan, disk_reserve,
        )
        for item in shards:
            observed[item["kind"]] += int(item["count"])

    for kind in (() if dirty_plan is not None else ("neurons", "assemblies")):
        buffers: dict[str, list[tuple[str, dict[str, Any]]]] = {
            bucket: [] for bucket in _BUCKETS
        }
        buffer_bytes = {bucket: 0 for bucket in _BUCKETS}
        parts = {bucket: 0 for bucket in _BUCKETS}

        def flush(bucket: str) -> None:
            group = buffers[bucket]
            if not group:
                return
            ordered = sorted(group, key=lambda item: item[0])
            ids = [record_id for record_id, _record in ordered]
            if ids != sorted(set(ids)):
                raise ValueError("paged shard has duplicate identifiers")
            records = [record for _record_id, record in ordered]
            payload: dict[str, Any] = {
                "kind": kind,
                "ids": ids,
                "records": records,
                "vectorIds": ids,
            }
            tensors = None
            if kind == "neurons":
                vector_metadata, tensors = substrate.neuron_vectors.export_state(keys=ids)
                if vector_metadata.get("ids") != ids or set(tensors) != {
                    "packed_rows", "update_counters_le"
                }:
                    raise ValueError("paged vector shard export differs from neuron IDs")
                PackedTernaryVectors.from_state(vector_metadata, tensors)
                payload["packedVectorState"] = vector_metadata
            else:
                payload["vectorStorage"] = "shared-neuron-packed"
            record_spec = _write_json_blob(substrate, store, payload, disk_reserve)
            tensor_spec = (
                _write_tensor_blob(
                    substrate, store, tensors,
                    previous_tensors.get((kind, bucket, parts[bucket])),
                    disk_reserve,
                ) if tensors is not None else None
            )
            shards.append({
                "kind": kind,
                "bucket": bucket,
                "part": parts[bucket],
                "count": len(ordered),
                "records": record_spec,
                "tensors": tensor_spec,
                **({"inspection": neuron_shard_inspection(records)} if kind == "neurons" else {}),
            })
            observed[kind] += len(ordered)
            parts[bucket] += 1
            buffers[bucket] = []
            buffer_bytes[bucket] = 0

        if kind == "neurons":
            values = (
                (str(record_id), _neuron_record(str(record_id), record))
                for record_id, record in substrate.neurons.items()
            )
        else:
            def assembly_values():
                for ordinal, raw in enumerate(substrate.assemblies):
                    identifier, _fingerprint, payload, _digest = _record_payload(_thaw(raw))
                    if identifier not in substrate.neurons or identifier not in substrate.neuron_vectors:
                        raise ValueError("paged assembly lacks its shared neuron row")
                    record = json.loads(payload)
                    record["__persistence_ordinal"] = ordinal
                    yield identifier, record

            values = assembly_values()
        for record_id, record in values:
            bucket = NeuralSubstrate._bucket(kind, record_id)
            charge = len(_canonical(record)) + len(record_id) + 128
            if charge > _MAX_BUFFER_BYTES:
                raise SubstrateResourcePause("paged substrate record exceeds byte window")
            if buffers[bucket] and buffer_bytes[bucket] + charge > _MAX_BUFFER_BYTES:
                flush(bucket)
            group = buffers[bucket]
            group.append((record_id, record))
            buffer_bytes[bucket] += charge
            if len(group) == records_per_shard:
                flush(bucket)
        for bucket in _BUCKETS:
            flush(bucket)

    for descriptor in synapse_plan.iter_descriptors():
        shards.append(descriptor)

    counts = {
        "neurons": len(substrate.neurons),
        "assemblies": len(substrate.assemblies),
        "synapses": len(substrate.synapses),
    }
    if observed != {key: counts[key] for key in observed}:
        raise ValueError("paged source changed while writing substrate shards")
    if substrate.state_revision != initial_state_revision:
        raise ValueError("substrate state changed while writing shards")
    current_index = substrate.assemblies.index.status()
    if any(
        current_index[key] != initial_index[key]
        for key in ("storeId", "count", "indexRevision")
    ):
        raise ValueError("paged assembly metadata changed while writing shards")
    current_neurons = neuron_status() if initial_neurons is not None else None
    if initial_neurons is not None and any(
        current_neurons.get(key) != initial_neurons.get(key)
        for key in ("storeId", "revision", "rowCount", "decayEpoch")
    ):
        raise ValueError("paged neuron metadata changed while writing shards")
    current_vectors = vector_status() if initial_vectors is not None else None
    if initial_vectors is not None and any(
        current_vectors.get(key) != initial_vectors.get(key)
        for key in ("storeId", "revision", "rowCount")
    ):
        raise ValueError("packed vector rows changed while writing shards")
    if shared_path is not None:
        with closing(sqlite3.connect(shared_path)) as connection:
            if source_stamp(connection) != initial_stamp:
                raise ValueError("paged source revisions/decay changed while writing shards")
    if dirty_plan is not None:
        dirty_plan.assert_unchanged()

    body = {
        "format": _SUBSTRATE_STORE_FORMAT,
        "formatVersion": _SUBSTRATE_STORE_VERSION,
        "schema": substrate.SCHEMA,
        "dimensions": substrate.space.dimensions,
        "seed": substrate.space.seed,
        "growthEvents": substrate.growth_events,
        "growthPauses": substrate.growth_pauses,
        "stateRevision": substrate.state_revision,
        "recordsPerShard": records_per_shard,
        "counts": counts,
        "shards": sorted(shards, key=lambda item: (
            str(item["kind"]), str(item["bucket"]), int(item["part"])
        )),
    }
    content_checksum = _sha(_canonical(body))
    generation = {**body, "contentSha256": content_checksum}
    generation_payload = _canonical(generation)
    generation_relative = "generations/%s/manifest.json" % content_checksum
    _write_immutable_bytes(substrate, store, generation_relative, generation_payload, disk_reserve)
    synapse_plan.publish_index(
        content_checksum,
        _sha(generation_payload),
        PagedHotNodeIds(substrate.assemblies),
    )
    pointer = {
        "format": _SUBSTRATE_STORE_FORMAT,
        "formatVersion": _SUBSTRATE_STORE_VERSION,
        "activeGeneration": content_checksum,
        "generationManifest": generation_relative,
        "generationManifestSha256": _sha(generation_payload),
        "counts": dict(counts),
        "shardCount": len(shards),
        "contentSha256": content_checksum,
    }
    synapse_plan.prepare_commit(pointer)
    if endpoint_plan is not None:
        endpoint_plan.prepare_commit(pointer, synapse_plan.descriptors, synapse_plan._index_manifest)
    if dirty_plan is not None:
        dirty_plan.assert_unchanged()
    _replace_pointer(substrate, store / "manifest.json", _canonical(pointer), disk_reserve)
    synapse_plan.commit(pointer)
    substrate.persistence_manifest = dict(pointer)
    if shared_path is not None:
        substrate._pending_paged_checkpoint = (
            shared_path, dict(pointer), initial_stamp, dirty_plan, initial_state_revision,
            endpoint_plan, substrate.synapses, getattr(substrate.synapses, "graph_revision", None),
            getattr(substrate.synapses, "persistence_revision", None),
        )
    return dict(pointer)


def write_paged_substrate_generation(
    substrate: NeuralSubstrate, root: Path, *, records_per_shard: int = 512,
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
) -> dict[str, Any]:
    """Publish within one bounded, process-authenticated derived-cache scope."""

    if not isinstance(substrate, NeuralSubstrate):
        raise TypeError("paged writer requires a neural substrate")
    if not isinstance(substrate.assemblies, PagedAssemblyView):
        raise ValueError("paged writer requires a paged assembly view")
    session = cache_session(
        substrate.assemblies.index,
        disk_reserve=lambda size, operation: _reserve(substrate, disk_reserve, size, operation),
    )
    with session.blob_scope():
        return _write_paged_substrate_generation(
            substrate, root, records_per_shard=records_per_shard, disk_reserve=disk_reserve,
        )


def forget_paged_blob_proof(substrate: NeuralSubstrate, path: Path) -> None:
    """Best-effort GC hook; this never deletes learned state or blob files."""

    if isinstance(substrate.assemblies, PagedAssemblyView):
        session = getattr(substrate.assemblies.index, "_authenticated_cache_session", None)
        if session is not None:
            session.forget_blob(path)
