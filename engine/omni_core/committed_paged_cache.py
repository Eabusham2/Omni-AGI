"""Rebuild a derived paged working cache from one committed v3 substrate.

``brain.json`` is the commit pointer. The substrate's own ``manifest.json``
may be ahead after an interrupted save, and an existing SQLite working cache
may contain uncommitted rows or a source cursor. Neither is read or adopted.
This coordinator verifies one bounded shard at a time, creates a fresh SQLite
database in a private sibling directory, and publishes that directory only
after the whole generation and both cache bindings have been checked.

The result is a *derived cache*, never a replacement brain checkpoint. The
caller may atomically switch its own cache handle to ``result.path`` after the
function returns; an old cache remains untouched on success and failure.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import quote

import torch
from safetensors.torch import load as load_safetensors

from .packed_vsa_vectors import PackedTernaryVectors
from .paged_assembly_index import PagedAssemblyIndex
from .paged_assembly_vector_view import PagedAssemblyVectorView
from .paged_assembly_view import PagedAssemblyView
from .paged_neuron_metadata import PagedNeuronMetadata
from .paged_packed_vectors import MAX_EXPORT_ROWS, PagedPackedVectors
from .vsa import (
    LazyPersistedSynapses,
    NeuralSubstrate,
    _load_forward_index,
    _synapse_id_matches_endpoints,
    _unpack_persisted_synapse_weights,
)


DEFAULT_MAX_BLOB_BYTES = 64 * 1024 * 1024
_CACHE_RECEIPT_FORMAT = "omni-derived-paged-cache"
_CACHE_DIRECTORY_PATTERN = re.compile(r"paged-([0-9a-f]{16})-([0-9a-f]{32})")


class CommittedCacheResourcePause(RuntimeError):
    """A configured bounded-shard read reserve refused reconstruction."""


class _IndexedAssemblyIDs:
    """Exact paged membership for the lazy synapse forward index."""

    def __init__(self, index: PagedAssemblyIndex) -> None:
        self.index = index

    def __contains__(self, identifier: object) -> bool:
        if not isinstance(identifier, str):
            return False
        with self.index._transaction() as connection:
            return connection.execute(
                "SELECT 1 FROM assembly_records WHERE assembly_id=?",
                (identifier,),
            ).fetchone() is not None

    def contains_many(self, identifiers: Any) -> set[str]:
        values = [str(value) for value in identifiers if str(value)]
        if len(values) > 4096:
            raise ValueError("assembly membership request exceeds bounded page")
        found: set[str] = set()
        with self.index._transaction() as connection:
            for start in range(0, len(values), 256):
                window = values[start : start + 256]
                if window:
                    placeholders = ",".join("?" for _ in window)
                    found.update(row[0] for row in connection.execute(
                        "SELECT assembly_id FROM assembly_records "
                        "WHERE assembly_id IN (%s)" % placeholders,
                        window,
                    ))
        return found

    def iter_sorted_ids(self):
        status = self.index.status()
        after = ""
        seen = 0
        while True:
            with self.index._transaction() as connection:
                if (
                    self.index._store_id(connection) != status["storeId"]
                    or self.index._index_revision(connection) != status["indexRevision"]
                ):
                    raise ValueError("assembly membership generation drift")
                rows = connection.execute(
                    "SELECT assembly_id FROM assembly_records "
                    "WHERE assembly_id>? ORDER BY assembly_id LIMIT 256",
                    (after,),
                ).fetchall()
            if not rows:
                if seen != status["count"]:
                    raise ValueError("assembly membership coverage mismatch")
                return
            for (identifier,) in rows:
                if not isinstance(identifier, str) or identifier <= after:
                    raise ValueError("assembly membership ordering is invalid")
                after = identifier
                seen += 1
                yield identifier

    def __iter__(self):
        return self.iter_sorted_ids()


@dataclass(frozen=True)
class RebuiltPagedCache:
    path: Path
    generation_sha256: str
    generation_manifest_sha256: str
    brain_sha256: str
    neurons: int
    assemblies: int
    synapses: int


def _write_cache_receipt(
    directory: Path,
    generation_sha256: str,
    brain_sha256: str,
    disk_reserve: Optional[Callable[[int, str], Any]],
) -> None:
    """Mark only a completely verified staging directory as helper-owned."""

    receipt = NeuralSubstrate._canonical_json({
        "format": _CACHE_RECEIPT_FORMAT,
        "formatVersion": 1,
        "generationSha256": generation_sha256,
        "brainSha256": brain_sha256,
        "sqliteFile": "working.sqlite3",
    })
    _reserve(disk_reserve, len(receipt) + 4096, "derived cache receipt")
    with (directory / "receipt.json").open("xb") as handle:
        handle.write(receipt)
        handle.flush()
        os.fsync(handle.fileno())


def _required_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError("%s is invalid" % label)
    return value


def _reserve(
    callback: Optional[Callable[[int, str], Any]], size: int, label: str
) -> None:
    if callback is not None and callback(max(1, size), label) is False:
        raise CommittedCacheResourcePause("%s paused at resource reserve" % label)


def _read_blob(
    store: Path,
    spec: Mapping[str, Any],
    suffix: str,
    *,
    max_blob_bytes: int,
    memory_reserve: Optional[Callable[[int, str], Any]],
) -> bytes:
    checksum = spec.get("sha256")
    if (
        not isinstance(checksum, str)
        or len(checksum) != 64
        or any(char not in "0123456789abcdef" for char in checksum)
        or spec.get("path") != "blobs/%s%s" % (checksum, suffix)
    ):
        raise ValueError("substrate shard blob identity is invalid")
    size = _required_int(spec.get("bytes"), "substrate shard blob size")
    if size > max_blob_bytes:
        raise CommittedCacheResourcePause("substrate shard exceeds bounded read window")
    _reserve(memory_reserve, 3 * size + 4096, "committed substrate shard read")
    path = NeuralSubstrate._safe_store_path(store, spec["path"])
    if path.stat().st_size != size:
        raise ValueError("substrate shard blob size mismatch")
    with path.open("rb") as handle:
        payload = handle.read(size + 1)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("substrate shard blob checksum mismatch")
    return payload


def _read_record_shard(
    store: Path,
    descriptor: Mapping[str, Any],
    *,
    max_blob_bytes: int,
    memory_reserve: Optional[Callable[[int, str], Any]],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    kind = descriptor["kind"]
    spec = descriptor.get("records")
    if not isinstance(spec, Mapping):
        raise ValueError("substrate record shard is invalid")
    blob = _read_blob(
        store, spec, ".json", max_blob_bytes=max_blob_bytes,
        memory_reserve=memory_reserve,
    )
    try:
        payload = json.loads(blob.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("substrate record shard JSON is invalid") from error
    fields = {
        "neurons": {"kind", "ids", "records", "vectorIds", "packedVectorState"},
        "assemblies": {"kind", "ids", "records", "vectorIds", "vectorStorage"},
        "synapses": {"kind", "ids", "records"},
    }[kind]
    if not isinstance(payload, dict) or set(payload) != fields or payload.get("kind") != kind:
        raise ValueError("substrate record shard structure is invalid")
    ids, records = payload.get("ids"), payload.get("records")
    if (
        not isinstance(ids, list)
        or not isinstance(records, list)
        or len(ids) != descriptor["count"]
        or len(records) != descriptor["count"]
        or any(not isinstance(value, str) or not value for value in ids)
        or any(not isinstance(value, dict) for value in records)
        or ids != sorted(set(ids))
        or ids != [record.get("id") for record in records]
        or any(
            NeuralSubstrate._bucket(kind, identifier) != descriptor["bucket"]
            for identifier in ids
        )
    ):
        raise ValueError("substrate shard identifiers or count do not match")
    return ids, records, payload


def _read_tensor_shard(
    store: Path,
    descriptor: Mapping[str, Any],
    *,
    max_blob_bytes: int,
    memory_reserve: Optional[Callable[[int, str], Any]],
) -> dict[str, torch.Tensor]:
    spec = descriptor.get("tensors")
    if not isinstance(spec, Mapping):
        raise ValueError("substrate tensor shard is missing")
    blob = _read_blob(
        store, spec, ".safetensors", max_blob_bytes=max_blob_bytes,
        memory_reserve=memory_reserve,
    )
    try:
        return load_safetensors(blob)
    except (ValueError, RuntimeError) as error:
        raise ValueError("substrate tensor shard is invalid") from error


def _committed_generation(
    engine_path: Path, *, max_shard_rows: int
) -> tuple[bytes, Path, dict[str, Any], dict[str, Any], dict[str, int]]:
    brain_bytes = (engine_path / "brain.json").read_bytes()
    try:
        brain = json.loads(brain_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("committed brain pointer is invalid") from error
    substrate = brain.get("substrate") if isinstance(brain, dict) else None
    pointer = substrate.get("persistence") if isinstance(substrate, dict) else None
    if (
        not isinstance(pointer, dict)
        or pointer.get("format") != "omni-substrate-shards"
        or pointer.get("formatVersion") != 3
        or substrate.get("schema") != NeuralSubstrate.SCHEMA
    ):
        raise ValueError("brain has no committed v3 substrate pointer")
    store = engine_path / "substrate"
    generation_id, _references = NeuralSubstrate._generation_references(store, pointer)
    generation_path = NeuralSubstrate._safe_store_path(
        store, pointer["generationManifest"]
    )
    generation_bytes = generation_path.read_bytes()
    if hashlib.sha256(generation_bytes).hexdigest() != pointer.get("generationManifestSha256"):
        raise ValueError("substrate generation manifest checksum mismatch")
    generation = json.loads(generation_bytes.decode("utf-8"))
    if (
        not isinstance(generation, dict)
        or generation.get("schema") != NeuralSubstrate.SCHEMA
        or generation.get("formatVersion") != 3
        or generation.get("contentSha256") != generation_id
        or type(generation.get("dimensions")) is not int
        or generation["dimensions"] < 1
        or type(generation.get("seed")) is not int
        or generation["dimensions"] != substrate.get("dimensions")
        or generation["seed"] != substrate.get("seed")
    ):
        raise ValueError("substrate generation disagrees with committed brain")
    records_per_shard = _required_int(
        generation.get("recordsPerShard"), "substrate shard row window", minimum=1
    )
    if records_per_shard > max_shard_rows:
        raise CommittedCacheResourcePause("substrate shard exceeds bounded row window")
    counts = generation.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"neurons", "assemblies", "synapses"}:
        raise ValueError("substrate generation counts are invalid")
    expected = {
        kind: _required_int(counts[kind], "substrate %s count" % kind)
        for kind in ("neurons", "assemblies", "synapses")
    }
    shards = generation.get("shards")
    if (
        not isinstance(shards, list)
        or pointer.get("counts") != expected
        or pointer.get("shardCount") != len(shards)
    ):
        raise ValueError("substrate generation shard count disagrees with pointer")
    seen_descriptors: set[tuple[str, str, int]] = set()
    described = {kind: 0 for kind in expected}
    for shard in shards:
        if not isinstance(shard, dict):
            raise ValueError("substrate shard descriptor is invalid")
        kind, bucket, part = shard.get("kind"), shard.get("bucket"), shard.get("part")
        count = _required_int(shard.get("count"), "substrate shard count", minimum=1)
        if (
            kind not in described
            or not isinstance(bucket, str)
            or len(bucket) != 1
            or bucket not in "0123456789abcdef"
            or type(part) is not int
            or part < 0
            or count > records_per_shard
            or (kind, bucket, part) in seen_descriptors
            or (kind == "assemblies" and shard.get("tensors") is not None)
            or (kind != "assemblies" and not isinstance(shard.get("tensors"), dict))
        ):
            raise ValueError("substrate shard descriptor placement is invalid")
        seen_descriptors.add((kind, bucket, part))
        described[kind] += count
    if described != expected:
        raise ValueError("substrate generation record counts disagree with shards")
    return brain_bytes, store, pointer, generation, expected


def rebuild_committed_paged_cache(
    engine_path: Path,
    cache_directory: Path,
    *,
    resource_policy: Optional[Any] = None,
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
    memory_reserve: Optional[Callable[[int, str], Any]] = None,
    max_shard_rows: int = MAX_EXPORT_ROWS,
    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
) -> RebuiltPagedCache:
    """Stream a verified committed generation into a new, bound SQLite cache.

    The returned path is unique and has no source checkpoint. This function
    never opens or replaces an existing cache and never advances ingestion.
    ``resource_policy.require_disk`` (or ``disk_reserve``) is passed through
    to the paged stores; ``memory_reserve`` guards each bounded shard read.
    """

    if resource_policy is not None and disk_reserve is not None:
        raise ValueError("provide either resource_policy or disk_reserve")
    if type(max_shard_rows) is not int or not 1 <= max_shard_rows <= MAX_EXPORT_ROWS:
        raise ValueError("cache rebuild row window is invalid")
    if type(max_blob_bytes) is not int or max_blob_bytes < 1:
        raise ValueError("cache rebuild blob window is invalid")
    engine_path = Path(engine_path).resolve()
    cache_directory = Path(cache_directory).resolve()
    brain_bytes, store, pointer, generation, expected = _committed_generation(
        engine_path, max_shard_rows=max_shard_rows
    )
    descriptors = generation["shards"]
    neuron_shards = [item for item in descriptors if item["kind"] == "neurons"]
    assembly_shards = [item for item in descriptors if item["kind"] == "assemblies"]
    synapse_shards = [item for item in descriptors if item["kind"] == "synapses"]

    # The first verified neuron shard supplies the persisted deadband. It is
    # the only shard held across SQLite initialization, never all vectors.
    first: Optional[tuple[list[str], list[dict[str, Any]], dict[str, Any], dict[str, torch.Tensor]]] = None
    deadband = 0.25
    if neuron_shards:
        ids, records, payload = _read_record_shard(
            store, neuron_shards[0], max_blob_bytes=max_blob_bytes,
            memory_reserve=memory_reserve,
        )
        tensors = _read_tensor_shard(
            store, neuron_shards[0], max_blob_bytes=max_blob_bytes,
            memory_reserve=memory_reserve,
        )
        if payload["vectorIds"] != ids or set(tensors) != {
            "packed_rows", "update_counters_le"
        }:
            raise ValueError("neuron shard lacks exact packed vector state")
        verified = PackedTernaryVectors.from_state(payload["packedVectorState"], tensors)
        if list(verified) != ids or verified.seed != generation["seed"] or verified.dimensions != generation["dimensions"]:
            raise ValueError("neuron packed vector IDs or parameters disagree")
        deadband = verified.zero_deadband
        first = ids, records, payload, tensors

    cache_directory.mkdir(parents=True, exist_ok=True)
    options = {
        "resource_policy": resource_policy,
        "disk_reserve": disk_reserve,
        "memory_reserve": memory_reserve,
    }
    working_disk_reserve = (
        resource_policy.require_disk if resource_policy is not None else disk_reserve
    )
    with tempfile.TemporaryDirectory(prefix=".paged-cache-rebuild-", dir=cache_directory) as temporary:
        staged_directory = Path(temporary)
        staged_path = staged_directory / "working.sqlite3"
        index = PagedAssemblyIndex(
            staged_path, dimensions=generation["dimensions"],
            seed=generation["seed"], zero_deadband=deadband, **options,
        )
        vectors = index._vectors
        assert isinstance(vectors, PagedPackedVectors)
        observed = {kind: 0 for kind in expected}
        for number, descriptor in enumerate(neuron_shards):
            if number == 0:
                assert first is not None
                ids, _records, payload, tensors = first
            else:
                ids, _records, payload = _read_record_shard(
                    store, descriptor, max_blob_bytes=max_blob_bytes,
                    memory_reserve=memory_reserve,
                )
                tensors = _read_tensor_shard(
                    store, descriptor, max_blob_bytes=max_blob_bytes,
                    memory_reserve=memory_reserve,
                )
            if payload["vectorIds"] != ids or set(tensors) != {
                "packed_rows", "update_counters_le"
            }:
                raise ValueError("neuron shard lacks exact packed vector state")
            observed["neurons"] += vectors.import_state(payload["packedVectorState"], tensors)
            # Drop each shard page before fetching the next one.
            first = None
        verification_path = staged_directory / "verification.sqlite3"
        with sqlite3.connect(verification_path) as connection:
            connection.execute(
                "CREATE TABLE rebuild_assembly_rows ("
                "ordinal INTEGER PRIMARY KEY, record_json BLOB NOT NULL)"
            )
            connection.execute(
                "CREATE TABLE rebuild_synapse_ids (id TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            for descriptor in assembly_shards:
                _reserve(
                    working_disk_reserve,
                    4 * descriptor["records"]["bytes"] + 64 * 1024,
                    "assembly rebuild staging",
                )
                ids, records, payload = _read_record_shard(
                    store, descriptor, max_blob_bytes=max_blob_bytes,
                    memory_reserve=memory_reserve,
                )
                if payload["vectorIds"] != ids or payload["vectorStorage"] != "shared-neuron-packed":
                    raise ValueError("assembly shard does not alias neuron vector rows")
                for record in records:
                    ordinal = _required_int(
                        record.get("__persistence_ordinal"), "assembly persistence ordinal"
                    )
                    clean = {key: value for key, value in record.items() if key != "__persistence_ordinal"}
                    connection.execute(
                        "INSERT INTO rebuild_assembly_rows(ordinal,record_json) VALUES (?,?)",
                        (ordinal, NeuralSubstrate._canonical_json(clean)),
                    )
                observed["assemblies"] += len(records)
            for descriptor in synapse_shards:
                _reserve(
                    working_disk_reserve,
                    1024 * descriptor["count"] + 64 * 1024,
                    "synapse ID verification staging",
                )
                ids, records, _payload = _read_record_shard(
                    store, descriptor, max_blob_bytes=max_blob_bytes,
                    memory_reserve=memory_reserve,
                )
                tensors = _read_tensor_shard(
                    store, descriptor, max_blob_bytes=max_blob_bytes,
                    memory_reserve=memory_reserve,
                )
                if set(tensors) != {
                    "packed_effective_weight", "eligibility", "plasticity",
                    "uses", "stability", "last_updated_at",
                }:
                    raise ValueError("synapse shard tensor fields are invalid")
                effective = _unpack_persisted_synapse_weights(tensors, len(ids), 3)
                if (
                    any(tensor.numel() != len(ids) for name, tensor in tensors.items()
                        if name != "packed_effective_weight")
                    or any(not bool(torch.isfinite(tensor).all()) for name, tensor in tensors.items()
                           if name not in {"packed_effective_weight", "uses"})
                    or not bool(((effective >= -1) & (effective <= 1)).all())
                ):
                    raise ValueError("synapse shard tensor count or value is invalid")
                for identifier, record in zip(ids, records):
                    if not _synapse_id_matches_endpoints(
                        identifier, record.get("source_id"), record.get("target_id")
                    ):
                        raise ValueError("synapse shard endpoints do not match ID")
                    connection.execute(
                        "INSERT INTO rebuild_synapse_ids(id) VALUES (?)", (identifier,)
                    )
                observed["synapses"] += len(ids)
            connection.commit()

        if observed != expected:
            raise ValueError("substrate shard generation count mismatch")
        # Keyset pages preserve original assembly order without an all-row list.
        after = -1
        next_ordinal = 0
        while True:
            with sqlite3.connect(verification_path) as connection:
                page = connection.execute(
                    "SELECT ordinal,record_json FROM rebuild_assembly_rows "
                    "WHERE ordinal>? ORDER BY ordinal LIMIT 64", (after,),
                ).fetchall()
            if not page:
                break
            with index.batch(max_rows=len(page), max_payload_bytes=64 * 1024 * 1024) as batch:
                for ordinal, raw in page:
                    if ordinal != next_ordinal:
                        raise ValueError("assembly persistence ordinals are not contiguous")
                    record = json.loads(raw)
                    if record.get("id") not in vectors:
                        raise ValueError("assembly shard has no shared neuron vector")
                    batch.upsert(record)
                    next_ordinal += 1
                    after = ordinal
        if next_ordinal != expected["assemblies"]:
            raise ValueError("assembly persistence ordinal count mismatch")

        with sqlite3.connect(verification_path) as connection:
            if connection.execute("SELECT COUNT(*) FROM rebuild_synapse_ids").fetchone()[0] != expected["synapses"]:
                raise ValueError("synapse shard global ID count mismatch")
        verification_path.unlink()
        with sqlite3.connect(staged_path) as connection:
            if connection.execute("SELECT COUNT(*) FROM progress_checkpoints").fetchone()[0] != 0:
                raise ValueError("rebuilt cache unexpectedly contains an ingestion cursor")
        status = index.status()
        if (
            len(vectors) != expected["neurons"]
            or status["count"] != expected["assemblies"]
            or status["packedVectorRows"] != expected["assemblies"]
        ):
            raise ValueError("rebuilt cache does not cover committed substrate")
        # Detect a concurrent save before binding this derived cache.
        if (engine_path / "brain.json").read_bytes() != brain_bytes:
            raise ValueError("committed brain pointer changed during cache rebuild")
        generation_id = pointer["activeGeneration"]
        index.bind_committed_generation(
            generation_id,
            expected_index_revision=status["indexRevision"],
            expected_vector_revision=status["vectorRevision"],
        )
        index.discard_or_reconcile_uncommitted(generation_id)
        with sqlite3.connect(staged_path) as connection:
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("rebuilt SQLite cache failed integrity check")
            busy, _log, _checkpointed = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if busy:
                raise ValueError("rebuilt SQLite cache could not checkpoint WAL")
        if (engine_path / "brain.json").read_bytes() != brain_bytes:
            raise ValueError("committed brain pointer changed before cache publication")
        _write_cache_receipt(
            staged_directory, generation_id, hashlib.sha256(brain_bytes).hexdigest(),
            working_disk_reserve,
        )
        final_directory = cache_directory / (
            "paged-%s-%s" % (generation_id[:16], uuid.uuid4().hex)
        )
        os.rename(staged_directory, final_directory)
        return RebuiltPagedCache(
            path=final_directory / "working.sqlite3",
            generation_sha256=generation_id,
            generation_manifest_sha256=pointer["generationManifestSha256"],
            brain_sha256=hashlib.sha256(brain_bytes).hexdigest(),
            neurons=expected["neurons"],
            assemblies=expected["assemblies"],
            synapses=expected["synapses"],
        )


class PreparedPagedCache:
    """An empty shared SQLite cache owned by one committed cold-load attempt.

    Use as a context manager. A failed or unfinished load removes only its
    private staging directory; no preexisting cache is ever opened.
    """

    def __init__(
        self,
        engine_path: Path,
        cache_directory: Path,
        *,
        resource_policy: Optional[Any] = None,
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
        memory_reserve: Optional[Callable[[int, str], Any]] = None,
        max_shard_rows: int = MAX_EXPORT_ROWS,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        if resource_policy is not None and disk_reserve is not None:
            raise ValueError("provide either resource_policy or disk_reserve")
        if type(max_shard_rows) is not int or not 1 <= max_shard_rows <= MAX_EXPORT_ROWS:
            raise ValueError("cache rebuild row window is invalid")
        if type(max_blob_bytes) is not int or max_blob_bytes < 1:
            raise ValueError("cache rebuild blob window is invalid")
        self.engine_path = Path(engine_path).resolve()
        self.cache_directory = Path(cache_directory).resolve()
        self.brain_bytes, self.store, self.pointer, self.generation, self.expected = (
            _committed_generation(self.engine_path, max_shard_rows=max_shard_rows)
        )
        self.memory_reserve = memory_reserve
        self.disk_reserve = (
            resource_policy.require_disk if resource_policy is not None
            else disk_reserve
        )
        self.max_blob_bytes = max_blob_bytes
        self._active = False
        self._published = False
        self._closed = False
        deadband = 0.25
        first_neuron = next(
            (item for item in self.generation["shards"] if item["kind"] == "neurons"),
            None,
        )
        if first_neuron is not None:
            ids, _records, payload = _read_record_shard(
                self.store, first_neuron, max_blob_bytes=max_blob_bytes,
                memory_reserve=memory_reserve,
            )
            packed_meta = payload["packedVectorState"]
            if (
                payload["vectorIds"] != ids
                or not isinstance(packed_meta, dict)
                or packed_meta.get("ids") != ids
                or packed_meta.get("dimensions") != self.generation["dimensions"]
                or packed_meta.get("seed") != self.generation["seed"]
            ):
                raise ValueError("neuron packed vector parameters disagree")
            from .packed_vsa_vectors import _valid_deadband
            deadband = _valid_deadband(packed_meta.get("zeroDeadband"))
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix=".paged-cache-rebuild-", dir=self.cache_directory
        )
        self.staged_directory = Path(self._temporary.name)
        self.path = self.staged_directory / "working.sqlite3"
        try:
            self.index = PagedAssemblyIndex(
                self.path,
                dimensions=self.generation["dimensions"],
                seed=self.generation["seed"],
                zero_deadband=deadband,
                resource_policy=resource_policy,
                disk_reserve=disk_reserve,
                memory_reserve=memory_reserve,
            )
        except BaseException:
            self._temporary.cleanup()
            raise
        self.vectors = self.index._vectors
        assert isinstance(self.vectors, PagedPackedVectors)
        self.neurons = PagedNeuronMetadata(
            self.path,
            resource_policy=resource_policy,
            disk_reserve=disk_reserve,
            memory_reserve=memory_reserve,
        )

    def __enter__(self) -> "PreparedPagedCache":
        if self._active or self._published or self._closed:
            raise RuntimeError("prepared cache is already active or published")
        self._active = True
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> bool:
        self._active = False
        self._closed = True
        self._temporary.cleanup()
        return False


def prepare_committed_paged_cache(
    engine_path: Path,
    cache_directory: Path,
    *,
    resource_policy: Optional[Any] = None,
    disk_reserve: Optional[Callable[[int, str], Any]] = None,
    memory_reserve: Optional[Callable[[int, str], Any]] = None,
    max_shard_rows: int = MAX_EXPORT_ROWS,
    max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
) -> PreparedPagedCache:
    """Prepare a fresh DB for ``load_sharded(..., paged_vectors=...)``.

    Example::

        with prepare_committed_paged_cache(engine, cache_dir) as prepared:
            loaded = NeuralSubstrate.load_sharded(
                engine / "substrate", metadata["substrate"],
                paged_vectors=prepared.vectors,
            )
            result = finish_verified_index_from_loaded_vectors(prepared, loaded)

    ``result.path`` is published only after all bounded checks pass.
    """

    return PreparedPagedCache(
        engine_path, cache_directory,
        resource_policy=resource_policy, disk_reserve=disk_reserve,
        memory_reserve=memory_reserve,
        max_shard_rows=max_shard_rows, max_blob_bytes=max_blob_bytes,
    )


def finish_verified_index_from_loaded_vectors(
    prepared: PreparedPagedCache,
    loaded_substrate: NeuralSubstrate,
) -> RebuiltPagedCache:
    """Finish the SAME fresh DB after the v3 loader imported verified rows.

    This requires the exact substrate returned by ``load_sharded`` with this
    cache's empty vector store. It does not import or copy neuron vectors a
    second time. Assembly/synapse shards are checked in bounded pages, every
    assembly ID must reference an existing neuron row, and no cursor is copied
    or advanced. The old cache and brain pointer are not modified.
    """

    if not isinstance(prepared, PreparedPagedCache) or not prepared._active or prepared._published:
        raise RuntimeError("prepared cache is not active")
    deferred = bool(getattr(loaded_substrate, "_paged_load_incomplete", False))
    if (
        not isinstance(loaded_substrate, NeuralSubstrate)
        or loaded_substrate.neuron_vectors is not prepared.vectors
        or (deferred and loaded_substrate.neurons is not prepared.neurons)
        or loaded_substrate.persistence_manifest != prepared.pointer
        or len(loaded_substrate.neurons) != prepared.expected["neurons"]
        or len(loaded_substrate.assemblies) != (
            0 if deferred else prepared.expected["assemblies"]
        )
        or len(loaded_substrate.synapses) != (
            0 if deferred else prepared.expected["synapses"]
        )
        or len(prepared.vectors) != prepared.expected["neurons"]
        or prepared.vectors.committed_binding()["generationSha256"] is not None
        or prepared.index.status()["count"] != 0
    ):
        raise ValueError("loaded substrate is not this fresh verified vector import")
    store = prepared.store
    index = prepared.index
    vectors = prepared.vectors
    expected = prepared.expected
    staged_path = prepared.path
    assembly_shards = [
        shard for shard in prepared.generation["shards"]
        if shard["kind"] == "assemblies"
    ]
    synapse_shards = [
        shard for shard in prepared.generation["shards"]
        if shard["kind"] == "synapses"
    ]
    observed_assemblies = 0
    observed_synapses = 0
    verification_path = prepared.staged_directory / "verification.sqlite3"
    with sqlite3.connect(verification_path) as connection:
        connection.execute(
            "CREATE TABLE rebuild_assembly_rows ("
            "ordinal INTEGER PRIMARY KEY, record_json BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE rebuild_synapse_ids (id TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        for descriptor in assembly_shards:
            _reserve(
                prepared.disk_reserve,
                4 * descriptor["records"]["bytes"] + 64 * 1024,
                "assembly rebuild staging",
            )
            ids, records, payload = _read_record_shard(
                store, descriptor, max_blob_bytes=prepared.max_blob_bytes,
                memory_reserve=prepared.memory_reserve,
            )
            if payload["vectorIds"] != ids or payload["vectorStorage"] != "shared-neuron-packed":
                raise ValueError("assembly shard does not alias neuron vector rows")
            for record in records:
                ordinal = _required_int(
                    record.get("__persistence_ordinal"), "assembly persistence ordinal"
                )
                clean = {
                    key: value for key, value in record.items()
                    if key != "__persistence_ordinal"
                }
                connection.execute(
                    "INSERT INTO rebuild_assembly_rows(ordinal,record_json) VALUES (?,?)",
                    (ordinal, NeuralSubstrate._canonical_json(clean)),
                )
            observed_assemblies += len(records)
        for descriptor in synapse_shards:
            _reserve(
                prepared.disk_reserve,
                1024 * descriptor["count"] + 64 * 1024,
                "synapse ID verification staging",
            )
            ids, records, _payload = _read_record_shard(
                store, descriptor, max_blob_bytes=prepared.max_blob_bytes,
                memory_reserve=prepared.memory_reserve,
            )
            tensors = _read_tensor_shard(
                store, descriptor, max_blob_bytes=prepared.max_blob_bytes,
                memory_reserve=prepared.memory_reserve,
            )
            if set(tensors) != {
                "packed_effective_weight", "eligibility", "plasticity",
                "uses", "stability", "last_updated_at",
            }:
                raise ValueError("synapse shard tensor fields are invalid")
            effective = _unpack_persisted_synapse_weights(tensors, len(ids), 3)
            if (
                any(tensor.numel() != len(ids) for name, tensor in tensors.items()
                    if name != "packed_effective_weight")
                or any(not bool(torch.isfinite(tensor).all()) for name, tensor in tensors.items()
                       if name not in {"packed_effective_weight", "uses"})
                or not bool(((effective >= -1) & (effective <= 1)).all())
            ):
                raise ValueError("synapse shard tensor count or value is invalid")
            for identifier, record in zip(ids, records):
                if not _synapse_id_matches_endpoints(
                    identifier, record.get("source_id"), record.get("target_id")
                ):
                    raise ValueError("synapse shard endpoints do not match ID")
                connection.execute(
                    "INSERT INTO rebuild_synapse_ids(id) VALUES (?)", (identifier,)
                )
            observed_synapses += len(ids)
        connection.commit()
    if observed_assemblies != expected["assemblies"] or observed_synapses != expected["synapses"]:
        raise ValueError("substrate shard generation count mismatch")

    # Reconstruct insertion order through disk-backed ordinals, not an
    # all-assembly Python list. Upserts never include a source checkpoint.
    after = -1
    ordinal_expected = 0
    while True:
        with sqlite3.connect(verification_path) as connection:
            page = connection.execute(
                "SELECT ordinal,record_json FROM rebuild_assembly_rows "
                "WHERE ordinal>? ORDER BY ordinal LIMIT 64", (after,),
            ).fetchall()
        if not page:
            break
        page_payload_bytes = sum(len(raw) for _ordinal, raw in page)
        if 4 * page_payload_bytes + 128 * len(page) <= 60 * 1024 * 1024:
            with index.batch(max_rows=len(page), max_payload_bytes=64 * 1024 * 1024) as batch:
                for ordinal, raw in page:
                    if ordinal != ordinal_expected:
                        raise ValueError("assembly persistence ordinals are not contiguous")
                    record = json.loads(raw)
                    if record.get("id") not in vectors:
                        raise ValueError("assembly shard has no shared neuron vector")
                    batch.upsert(record)
                    ordinal_expected += 1
                    after = ordinal
        else:
            for ordinal, raw in page:
                if ordinal != ordinal_expected:
                    raise ValueError("assembly persistence ordinals are not contiguous")
                record = json.loads(raw)
                if record.get("id") not in vectors:
                    raise ValueError("assembly shard has no shared neuron vector")
                index.upsert(record)
                ordinal_expected += 1
                after = ordinal
    if ordinal_expected != expected["assemblies"]:
        raise ValueError("assembly persistence ordinal count mismatch")
    with sqlite3.connect(verification_path) as connection:
        if connection.execute("SELECT COUNT(*) FROM rebuild_synapse_ids").fetchone()[0] != expected["synapses"]:
            raise ValueError("synapse shard global ID count mismatch")
    verification_path.unlink()
    with sqlite3.connect(staged_path) as connection:
        if connection.execute("SELECT COUNT(*) FROM progress_checkpoints").fetchone()[0] != 0:
            raise ValueError("rebuilt cache unexpectedly contains an ingestion cursor")
    status = index.status()
    if (
        status["count"] != expected["assemblies"]
        or status["packedVectorRows"] != expected["assemblies"]
        or len(vectors) != expected["neurons"]
        or (
            deferred
            and prepared.neurons.status()["rowCount"] != expected["neurons"]
        )
    ):
        raise ValueError("rebuilt cache does not cover committed substrate")
    if (prepared.engine_path / "brain.json").read_bytes() != prepared.brain_bytes:
        raise ValueError("committed brain pointer changed during cache rebuild")
    generation_id = prepared.pointer["activeGeneration"]
    index.bind_committed_generation(
        generation_id,
        expected_index_revision=status["indexRevision"],
        expected_vector_revision=status["vectorRevision"],
    )
    if deferred:
        prepared.neurons.bind_committed_generation(generation_id)
    index.discard_or_reconcile_uncommitted(generation_id)
    if deferred:
        membership = _IndexedAssemblyIDs(index)
        forward_index = _load_forward_index(
            store,
            generation=generation_id,
            generation_manifest_sha256=prepared.pointer["generationManifestSha256"],
            synapse_count=expected["synapses"],
            records_per_shard=int(prepared.generation.get("recordsPerShard", 0)),
            hot_node_ids=membership,
            descriptors=synapse_shards,
        )
        if forward_index is None:
            raise ValueError("paged cold-load requires a verified forward index")
        loaded_substrate.assemblies = PagedAssemblyView(index)
        loaded_substrate.assembly_vectors = PagedAssemblyVectorView(index, vectors)
        loaded_substrate.invalidate_assembly_index()
        loaded_substrate.synapses = LazyPersistedSynapses(
            store,
            synapse_shards,
            expected["synapses"],
            int(prepared.generation.get("recordsPerShard", 0)),
            membership,
            forward_index,
            store_version=3,
        )
        loaded_substrate._validate_packed_vector_identity()
        loaded_substrate._paged_load_incomplete = False
        del loaded_substrate._deferred_lazy_synapse_shards
    with sqlite3.connect(staged_path) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("rebuilt SQLite cache failed integrity check")
        busy, _log, _checkpointed = connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        if busy:
            raise ValueError("rebuilt SQLite cache could not checkpoint WAL")
    if (prepared.engine_path / "brain.json").read_bytes() != prepared.brain_bytes:
        raise ValueError("committed brain pointer changed before cache publication")
    _write_cache_receipt(
        prepared.staged_directory, generation_id,
        hashlib.sha256(prepared.brain_bytes).hexdigest(), prepared.disk_reserve,
    )
    final_directory = prepared.cache_directory / (
        "paged-%s-%s" % (generation_id[:16], uuid.uuid4().hex)
    )
    os.rename(prepared.staged_directory, final_directory)
    published_path = final_directory / "working.sqlite3"
    # Both stores open short-lived connections by path. Retarget the same
    # objects used by the loaded substrate after the atomic directory move.
    prepared.path = published_path
    index.path = published_path
    vectors.path = published_path
    prepared.neurons.path = published_path
    prepared._published = True
    return RebuiltPagedCache(
        path=published_path,
        generation_sha256=generation_id,
        generation_manifest_sha256=prepared.pointer["generationManifestSha256"],
        brain_sha256=hashlib.sha256(prepared.brain_bytes).hexdigest(),
        neurons=expected["neurons"],
        assemblies=expected["assemblies"],
        synapses=expected["synapses"],
    )


def _cache_receipt(directory: Path) -> Optional[dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        return None
    marker = directory / "receipt.json"
    if marker.is_symlink() or not marker.is_file():
        return None
    try:
        blob = marker.read_bytes()
        value = json.loads(blob.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {
            "format", "formatVersion", "generationSha256",
            "brainSha256", "sqliteFile",
        }
        or value.get("format") != _CACHE_RECEIPT_FORMAT
        or value.get("formatVersion") != 1
        or value.get("sqliteFile") != "working.sqlite3"
        or NeuralSubstrate._canonical_json(value) != blob
    ):
        return None
    generation = value.get("generationSha256")
    brain_sha = value.get("brainSha256")
    match = _CACHE_DIRECTORY_PATTERN.fullmatch(directory.name)
    if (
        match is None
        or not isinstance(generation, str)
        or not isinstance(brain_sha, str)
        or len(generation) != 64
        or len(brain_sha) != 64
        or any(char not in "0123456789abcdef" for char in generation + brain_sha)
        or match.group(1) != generation[:16]
    ):
        return None
    database = directory / "working.sqlite3"
    if database.is_symlink() or not database.is_file():
        return None
    return value


def _cache_has_clean_binding(database: Path, generation: str) -> bool:
    uri = "file:%s?mode=ro" % quote(str(database), safe="/")
    try:
        with closing(sqlite3.connect(uri, uri=True, timeout=5.0)) as connection:
            connection.execute("PRAGMA query_only=ON")
            index_meta = dict(connection.execute(
                "SELECT key,value FROM index_metadata WHERE key IN "
                "('index_revision','committed_generation_sha256',"
                "'committed_index_revision','committed_vector_revision')"
            ))
            vector_meta = dict(connection.execute(
                "SELECT key,value FROM paged_vector_meta WHERE key IN "
                "('revision','committed_generation_sha256','committed_revision')"
            ))
    except (OSError, sqlite3.DatabaseError):
        return False
    return (
        index_meta.get("committed_generation_sha256") == generation
        and vector_meta.get("committed_generation_sha256") == generation
        and index_meta.get("committed_index_revision") == index_meta.get("index_revision")
        and index_meta.get("committed_vector_revision") == vector_meta.get("revision")
        and vector_meta.get("committed_revision") == vector_meta.get("revision")
    )


def prune_derived_caches(
    cache_directory: Path,
    retain_path: Path,
    committed_generation: str,
) -> dict[str, Any]:
    """Remove only superseded helper-created caches after a verified switch.

    The caller must first switch all consumers to ``retain_path`` and quiesce
    handles to older caches. This never visits substrate shards, brain
    checkpoints, training data, in-progress staging dirs, or unknown files.
    A missing/dirty retained cache fails closed without deleting anything.
    """

    cache_directory = Path(cache_directory).resolve(strict=True)
    raw_retain = Path(retain_path)
    if raw_retain.is_symlink():
        raise ValueError("retained derived cache path must not be a symlink")
    retained = raw_retain.resolve(strict=True)
    retained_directory = retained.parent
    if (
        retained.name != "working.sqlite3"
        or retained_directory.parent != cache_directory
        or not isinstance(committed_generation, str)
        or len(committed_generation) != 64
        or any(char not in "0123456789abcdef" for char in committed_generation)
    ):
        raise ValueError("retained derived cache target is invalid")
    receipt = _cache_receipt(retained_directory)
    if (
        receipt is None
        or receipt["generationSha256"] != committed_generation
        or not _cache_has_clean_binding(retained, committed_generation)
    ):
        raise ValueError("retained derived cache is not verified and clean")
    allowed_files = {
        "working.sqlite3", "working.sqlite3-wal", "working.sqlite3-shm",
        "receipt.json",
    }
    removed: list[str] = []
    reclaimed = 0
    for candidate in cache_directory.iterdir():
        if candidate == retained_directory:
            continue
        prior = _cache_receipt(candidate)
        if prior is None:
            continue
        children = list(candidate.iterdir())
        if (
            not children
            or any(
                child.name not in allowed_files
                or child.is_symlink()
                or not child.is_file()
                for child in children
            )
        ):
            continue
        size = sum(child.stat().st_size for child in children)
        for child in children:
            child.unlink()
        candidate.rmdir()
        removed.append(str(candidate))
        reclaimed += size
    return {
        "retainedPath": str(retained),
        "retainedGeneration": committed_generation,
        "removedPaths": removed,
        "bytesReclaimed": reclaimed,
    }
