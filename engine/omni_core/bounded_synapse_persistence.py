"""Bounded, exact v3 persistence plan for dynamic ternary synapses.

This module owns no authoritative pointer. A caller writes neuron/assembly
shards, consumes :meth:`BoundedSynapseShardPlan.iter_descriptors`, publishes
the generation and forward index, and calls :meth:`commit` only after the
generation pointer is durable. Large resident maps are externally sorted in
an expendable SQLite scratch file; lazy maps rewrite just changed shard pages.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import torch

from .substrate_inspection import synapse_shard_inspection
from .paged_assembly_view import PagedAssemblyView
from .vsa import (
    LazyPersistedSynapses,
    NeuralSubstrate,
    SubstrateResourcePause,
    _SYNAPSE_TENSOR_FIELDS,
    _SUBSTRATE_STORE_VERSION,
    _forward_index_entry,
    _forward_hot_ids_sha256,
    _synapse_id_matches_endpoints,
    _write_forward_index,
    _pack_ternary_levels,
)


_BUCKETS = "0123456789abcdef"
_MAX_SCRATCH_BATCH = 1024
_ALLOWED_SYNAPSE_FIELDS = frozenset({
    "id", "source_id", "target_id", "kind", *_SYNAPSE_TENSOR_FIELDS,
})


class PagedHotNodeIds:
    """Exact assembly-ID membership with no in-memory all-assembly set."""

    def __init__(self, view: PagedAssemblyView) -> None:
        if not isinstance(view, PagedAssemblyView):
            raise TypeError("paged hot IDs require a paged assembly view")
        self.view = view

    def __contains__(self, identifier: object) -> bool:
        if not isinstance(identifier, str) or not identifier:
            return False
        return identifier in self.contains_many((identifier,))

    def contains_many(self, identifiers: Sequence[str] | set[str]) -> set[str]:
        """Check one bounded shard of endpoints in one indexed read session."""

        pending = sorted({value for value in identifiers if isinstance(value, str) and value})
        found: set[str] = set()
        with self.view.index._transaction() as connection:
            for offset in range(0, len(pending), 800):
                chunk = pending[offset:offset + 800]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                found.update(
                    str(row[0]) for row in connection.execute(
                        f"SELECT assembly_id FROM assembly_records WHERE assembly_id IN ({placeholders})",
                        chunk,
                    )
                )
        return found

    def iter_sorted_ids(self) -> Iterator[str]:
        with self.view.index._transaction() as connection:
            cursor = connection.execute(
                "SELECT assembly_id FROM assembly_records ORDER BY assembly_id"
            )
            for row in cursor:
                yield str(row[0])

    def __iter__(self) -> Iterator[str]:
        return self.iter_sorted_ids()

    def ids_sha256(self) -> str:
        return self.view.index.ids_sha256()

    @property
    def ids_checksum_algorithm(self) -> str:
        return self.view.index.ids_checksum_algorithm


class BoundedSynapseShardPlan:
    """Stream v3 descriptors with no all-changed-groups materialization.

    ``write_json_blob`` and ``write_tensor_blob`` must use the same immutable,
    checksummed blob publisher as the containing substrate generation.
    ``disk_reserve`` is checked before scratch batches and the forward index.
    The caller must serialize neural writes for the plan's lifetime.
    """

    def __init__(
        self,
        substrate: NeuralSubstrate,
        root: Path,
        *,
        records_per_shard: int,
        write_json_blob: Callable[[dict[str, Any]], dict[str, Any]],
        write_tensor_blob: Callable[
            [Mapping[str, torch.Tensor], Optional[Mapping[str, Any]]],
            dict[str, Any],
        ],
        disk_reserve: Optional[Callable[[int, str], Any]] = None,
        verify_reusable: Optional[
            Callable[[Mapping[str, Any], str], None]
        ] = None,
        endpoint_plan: Optional[Any] = None,
    ) -> None:
        if type(records_per_shard) is not int or not 1 <= records_per_shard <= 512:
            raise ValueError("synapse shard size must be between 1 and 512")
        self.substrate = substrate
        self.root = Path(root).resolve()
        self.records_per_shard = records_per_shard
        self.write_json_blob = write_json_blob
        self.write_tensor_blob = write_tensor_blob
        self.disk_reserve = disk_reserve
        self.verify_reusable = verify_reusable
        self.endpoint_plan = endpoint_plan
        self.descriptors: list[dict[str, Any]] = []
        self.forward_entries: list[dict[str, Any]] = []
        self._index_manifest: Optional[dict[str, Any]] = None
        self._next_lazy: Optional[LazyPersistedSynapses] = None
        self._prepared_generation: Optional[str] = None
        self._started = False
        self._finished = False
        self._committed = False
        self._initial_count = len(substrate.synapses)
        self._initial_revision = getattr(substrate.synapses, "graph_revision", None)
        self._initial_persistence_revision = getattr(substrate.synapses, "persistence_revision", None)
        self._initial_state_revision = substrate.state_revision
        self._lazy = (
            substrate.synapses
            if isinstance(substrate.synapses, LazyPersistedSynapses)
            else None
        )
        if self._lazy is not None:
            self._lazy.validate_dirty()
            if self._lazy.records_per_shard != records_per_shard:
                raise ValueError("lazy synapse shard size must be preserved")
            if self._lazy.store_version != _SUBSTRATE_STORE_VERSION:
                raise ValueError("legacy synapse shards cannot enter the v3 writer")
        self._hot_ids: Sequence[str] | PagedHotNodeIds = (
            PagedHotNodeIds(substrate.assemblies)
            if isinstance(substrate.assemblies, PagedAssemblyView)
            else [str(item["id"]) for item in substrate.assemblies if item.get("id")]
        )
        self._hot_ids_sha256 = _forward_hot_ids_sha256(self._hot_ids)
        self._reindex_unchanged = bool(
            self._lazy is not None
            and (
                self._lazy._forward_index_manifest is None
                or self._lazy._forward_index_manifest.get("hotNodeIdsChecksumAlgorithm")
                != getattr(self._hot_ids, "ids_checksum_algorithm", "sha256-sorted-ids-v1")
                or self._lazy._forward_index_manifest.get("hotNodeIdsSha256")
                != self._hot_ids_sha256
            )
        )
        if endpoint_plan is not None and endpoint_plan.full_forward_rebuild:
            self._reindex_unchanged = True

    def _reserve(self, size: int, operation: str) -> None:
        if self.disk_reserve is None:
            if self.substrate.growth_guard is not None and not self.substrate.growth_guard(size):
                raise SubstrateResourcePause("synapse checkpoint reached resource reserve")
            return
        if self.disk_reserve(size, operation) is False:
            raise SubstrateResourcePause("synapse checkpoint reached disk reserve")

    def _check_unchanged(self) -> None:
        if (
            len(self.substrate.synapses) != self._initial_count
            or getattr(self.substrate.synapses, "graph_revision", None)
            != self._initial_revision
            or self.substrate.state_revision != self._initial_state_revision
            or getattr(self.substrate.synapses, "persistence_revision", None)
            != self._initial_persistence_revision
        ):
            raise ValueError("synapses changed while writing checkpoint shards")

    def _scratch(self) -> tuple[sqlite3.Connection, Path]:
        self._reserve(8192, "synapse sort scratch initialization")
        directory = self.root / "staging"
        if directory.is_symlink():
            raise ValueError("synapse checkpoint staging path must not be a symlink")
        directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".synapse-plan-", suffix=".sqlite3", dir=directory)
        os.close(descriptor)
        path = Path(name)
        try:
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-1024")
            connection.execute("PRAGMA mmap_size=0")
            connection.execute(
                "CREATE TABLE ordered_synapses ("
                "bucket TEXT NOT NULL, id TEXT PRIMARY KEY, payload BLOB)"
            )
            connection.execute(
                "CREATE INDEX ordered_synapses_bucket_id "
                "ON ordered_synapses(bucket, id)"
            )
            return connection, path
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _clean_scratch(connection: sqlite3.Connection, path: Path) -> None:
        connection.close()
        path.unlink(missing_ok=True)
        # Scratch is not recovery authority. Directory syncing is best effort
        # because Windows cannot fsync directory handles on some filesystems.
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except OSError:
            pass

    def _stage_rows(self, connection: sqlite3.Connection) -> None:
        lazy = self._lazy
        rows: list[tuple[str, str, Optional[bytes]]] = []
        staged_bytes = 0
        source = (
            ((record_id, lazy._dirty[record_id]) for record_id in lazy._new_ids)
            if lazy is not None
            else self.substrate.synapses.items()
        )
        connection.execute("BEGIN")
        for raw_id, raw_record in source:
            record_id = str(raw_id)
            if not isinstance(raw_id, str) or not isinstance(raw_record, Mapping):
                raise ValueError("synapse checkpoint record is invalid")
            if raw_record.get("id") != record_id:
                raise ValueError("synapse mapping key differs from record id")
            payload = (
                None
                if lazy is not None
                else NeuralSubstrate._canonical_json(dict(raw_record))
            )
            rows.append((NeuralSubstrate._bucket("synapses", record_id), record_id, payload))
            staged_bytes += len(record_id.encode("utf-8")) + (len(payload) if payload else 0) + 128
            if len(rows) >= _MAX_SCRATCH_BATCH:
                self._reserve(staged_bytes * 3 + 4096, "synapse sort scratch")
                connection.executemany(
                    "INSERT INTO ordered_synapses(bucket,id,payload) VALUES(?,?,?)", rows
                )
                rows.clear()
                staged_bytes = 0
        if rows:
            self._reserve(staged_bytes * 3 + 4096, "synapse sort scratch")
            connection.executemany(
                "INSERT INTO ordered_synapses(bucket,id,payload) VALUES(?,?,?)", rows
            )
        connection.commit()

    def _region_lookup(self, group: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, str]:
        endpoints = {
            str(record.get(field, ""))
            for _record_id, record in group
            for field in ("source_id", "target_id")
        }
        return {
            identifier: str(self.substrate.neurons.get(identifier, {}).get("region", "semantic"))
            for identifier in endpoints
        }

    def _hot_lookup(self, group: Sequence[tuple[str, Mapping[str, Any]]]) -> set[str]:
        assemblies = self.substrate.assemblies
        get_by_id = getattr(assemblies, "get_by_id", None)
        endpoints = {
            str(record.get(field, ""))
            for _record_id, record in group
            for field in ("source_id", "target_id")
        }
        if callable(get_by_id):
            return PagedHotNodeIds(assemblies).contains_many(endpoints)
        # The resident assembly list path is used only for a fresh small brain.
        # Build its membership once per group rather than retaining an extra
        # corpus-sized structure for the entire checkpoint.
        return {
            identifier for identifier in endpoints
            if any(item.get("id") == identifier for item in assemblies)
        }

    def _verified_reusable(self, descriptor: Mapping[str, Any]) -> dict[str, Any]:
        if self._lazy is None or self._lazy.root != self.root:
            raise ValueError("synapse shard reuse needs the same verified store")
        for field, suffix in (("records", ".json"), ("tensors", ".safetensors")):
            spec = descriptor.get(field)
            if not isinstance(spec, Mapping):
                raise ValueError("synapse shard reference is incomplete")
            checksum = spec.get("sha256")
            size = spec.get("bytes")
            if (
                not isinstance(checksum, str)
                or len(checksum) != 64
                or any(character not in _BUCKETS for character in checksum)
                or type(size) is not int
                or size < 0
                or spec.get("path") != f"blobs/{checksum}{suffix}"
            ):
                raise ValueError("synapse shard reference is invalid")
            if self.verify_reusable is not None:
                self.verify_reusable(spec, suffix)
                continue
            path = NeuralSubstrate._safe_store_path(self.root, str(spec["path"]))
            raw_path = self.root / str(spec["path"])
            if raw_path.is_symlink() or path.stat().st_size != size:
                raise ValueError("synapse shard size or type mismatch")
            if NeuralSubstrate._file_sha256(path) != checksum:
                raise ValueError("synapse shard checksum mismatch")
        return dict(descriptor)

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"synapse {name} is not numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"synapse {name} is not finite")
        return converted

    def _write_group(
        self,
        bucket: str,
        part: int,
        group: list[tuple[str, Mapping[str, Any]]],
        reusable: Optional[Mapping[str, Any]],
    ) -> dict[str, Any]:
        if not 1 <= len(group) <= self.records_per_shard:
            raise ValueError("synapse shard count is out of bounds")
        group.sort(key=lambda item: item[0])
        identifiers = [record_id for record_id, _record in group]
        if identifiers != sorted(set(identifiers)):
            raise ValueError("synapse shard identifiers are duplicated")
        structures: list[dict[str, Any]] = []
        weights: list[int] = []
        floats: dict[str, list[float]] = {
            field: [] for field in ("eligibility", "plasticity", "stability", "last_updated_at")
        }
        uses: list[int] = []
        for record_id, record in group:
            source = record.get("source_id")
            target = record.get("target_id")
            if (
                not isinstance(source, str)
                or not isinstance(target, str)
                or record.get("id") != record_id
                or NeuralSubstrate._bucket("synapses", record_id) != bucket
                or not _synapse_id_matches_endpoints(record_id, source, target)
                or set(record) - _ALLOWED_SYNAPSE_FIELDS
                or not isinstance(record.get("kind", ""), str)
                or len(record.get("kind", "")) > 96
            ):
                raise ValueError("synapse record has invalid identity or non-structural text")
            weights.append(NeuralSubstrate.exact_effective_weight(record.get("effective_weight", 0)))
            count = record.get("uses", 0)
            if type(count) is not int or count < 0:
                raise ValueError("synapse use count is invalid")
            uses.append(count)
            for field, default in (("eligibility", 0.0), ("plasticity", 1.0), ("stability", 0.0), ("last_updated_at", 0.0)):
                floats[field].append(self._finite_number(record.get(field, default), field))
            structures.append({
                key: value for key, value in record.items()
                if key not in _SYNAPSE_TENSOR_FIELDS
            })
        record_spec = self.write_json_blob({
            "kind": "synapses", "ids": identifiers, "records": structures,
        })
        tensors = {
            "packed_effective_weight": torch.tensor(
                list(_pack_ternary_levels(weights)), dtype=torch.uint8
            ),
            "eligibility": torch.tensor(floats["eligibility"], dtype=torch.float64),
            "plasticity": torch.tensor(floats["plasticity"], dtype=torch.float64),
            "uses": torch.tensor(uses, dtype=torch.int64),
            "stability": torch.tensor(floats["stability"], dtype=torch.float64),
            "last_updated_at": torch.tensor(floats["last_updated_at"], dtype=torch.float64),
        }
        tensor_spec = self.write_tensor_blob(
            tensors,
            reusable.get("tensors") if isinstance(reusable, Mapping) else None,
        )
        descriptor = {
            "kind": "synapses", "bucket": bucket, "part": part,
            "count": len(group), "records": record_spec, "tensors": tensor_spec,
            "inspection": synapse_shard_inspection(
                [record for _record_id, record in group], self._region_lookup(group)
            ),
        }
        self._append_forward_entry(_forward_index_entry(descriptor, group, self._hot_lookup(group)))
        if self.endpoint_plan is not None:
            self.endpoint_plan.stage_group(descriptor, group)
        return descriptor

    def _iter_resident(self, connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
        for bucket in _BUCKETS:
            part = 0
            group: list[tuple[str, Mapping[str, Any]]] = []
            rows = connection.execute(
                "SELECT id,payload FROM ordered_synapses WHERE bucket=? ORDER BY id", (bucket,)
            )
            for record_id, payload in rows:
                if not isinstance(payload, bytes):
                    raise ValueError("resident synapse scratch payload is missing")
                value = json.loads(payload)
                if not isinstance(value, dict):
                    raise ValueError("resident synapse scratch record is invalid")
                group.append((record_id, value))
                if len(group) == self.records_per_shard:
                    yield self._write_group(bucket, part, group, None)
                    group = []
                    part += 1
            if group:
                yield self._write_group(bucket, part, group, None)

    def _iter_lazy(self, connection: sqlite3.Connection) -> Iterator[dict[str, Any]]:
        lazy = self._lazy
        assert lazy is not None
        changed = {
            location[0] for location in lazy._dirty_locations.values()
        }.union(location[0] for location in lazy._deleted_locations.values())
        ordered = iter(lazy._ordered_keys)
        current = next(ordered, None)
        for bucket in _BUCKETS:
            last: Optional[tuple[str, int]] = None
            while current is not None and current[0] == bucket:
                if last is not None:
                    if last in changed:
                        group = lazy._group_records(last)
                        if group:
                            yield self._write_group(bucket, last[1], group, lazy._descriptors[last])
                        elif self.endpoint_plan is not None:
                            self.endpoint_plan.stage_deleted(last)
                    else:
                        yield self._emit_reused(lazy, last)
                last = current
                current = next(ordered, None)
            additions_cursor = connection.execute(
                "SELECT id FROM ordered_synapses WHERE bucket=? ORDER BY id", (bucket,)
            )
            additions = (str(row[0]) for row in additions_cursor)
            pending = next(additions, None)
            if last is not None:
                if last in changed or pending is not None:
                    group = lazy._group_records(last)
                    while pending is not None and len(group) < self.records_per_shard:
                        group.append((pending, dict(lazy._dirty[pending])))
                        pending = next(additions, None)
                    if group:
                        yield self._write_group(bucket, last[1], group, lazy._descriptors[last])
                    elif self.endpoint_plan is not None:
                        self.endpoint_plan.stage_deleted(last)
                else:
                    yield self._emit_reused(lazy, last)
            part = last[1] + 1 if last is not None else 0
            while pending is not None:
                group = []
                while pending is not None and len(group) < self.records_per_shard:
                    group.append((pending, dict(lazy._dirty[pending])))
                    pending = next(additions, None)
                yield self._write_group(bucket, part, group, None)
                part += 1
        if current is not None:
            raise ValueError("lazy synapse shard bucket traversal diverged")

    def _emit_reused(
        self, lazy: LazyPersistedSynapses, key: tuple[str, int]
    ) -> dict[str, Any]:
        descriptor = self._verified_reusable(lazy._descriptors[key])
        if self._reindex_unchanged and (
            self.endpoint_plan is None or self.endpoint_plan.requires_reindex(key)
        ):
            # A newly formed assembly can become an endpoint of an old edge.
            # Keep immutable shard bytes but rebuild its hot-location index
            # from one verified bounded page instead of trusting stale IDs.
            group = lazy._group_records(key)
            self._append_forward_entry(_forward_index_entry(descriptor, group, self._hot_lookup(group)))
            if self.endpoint_plan is not None:
                self.endpoint_plan.reindexed_groups += 1
        else:
            self.forward_entries.append(lazy._persisted_index_entry(key))
        return descriptor

    def _append_forward_entry(self, entry: Mapping[str, Any]) -> None:
        from .paged_forward_index import publish_entries, ReusedForwardEntry
        def guard(size: int) -> bool:
            self._reserve(size, "bounded derived forward group")
            return True
        descriptor = publish_entries(self.root, (entry,), guard)[0]
        self.forward_entries.append(ReusedForwardEntry(self.root, descriptor))

    def iter_descriptors(self) -> Iterator[dict[str, Any]]:
        if self._started:
            raise RuntimeError("synapse shard plan can only be consumed once")
        self._started = True
        connection, scratch = self._scratch()
        emitted = 0
        try:
            self._stage_rows(connection)
            source = self._iter_lazy(connection) if self._lazy is not None else self._iter_resident(connection)
            for descriptor in source:
                emitted += int(descriptor["count"])
                self.descriptors.append(descriptor)
                yield descriptor
            self._check_unchanged()
            if emitted != self._initial_count:
                raise ValueError("bounded synapse shard count diverges from live graph")
            if len(self.forward_entries) != len(self.descriptors):
                raise ValueError("synapse forward-index shard coverage diverges")
            self._finished = True
        finally:
            self._clean_scratch(connection, scratch)

    def publish_index(
        self,
        generation_id: str,
        generation_sha256: str,
        hot_node_ids: Sequence[str] | PagedHotNodeIds,
    ) -> dict[str, Any]:
        if not self._finished:
            raise RuntimeError("synapse shards must finish before index publication")
        self._check_unchanged()
        if _forward_hot_ids_sha256(hot_node_ids) != self._hot_ids_sha256:
            raise ValueError("assembly hot-ID membership changed during synapse checkpoint")
        def guard(size: int) -> bool:
            self._reserve(size, "synapse forward index")
            return True
        manifest = _write_forward_index(
            self.root,
            generation=generation_id,
            generation_manifest_sha256=generation_sha256,
            synapse_count=self._initial_count,
            records_per_shard=self.records_per_shard,
            hot_node_ids=hot_node_ids,
            entries=self.forward_entries,
            growth_guard=guard,
            verified_rebuild=bool(self.endpoint_plan is not None and self.endpoint_plan.full_forward_rebuild),
        )
        self._index_manifest = manifest
        replacement = LazyPersistedSynapses(
            self.root,
            self.descriptors,
            self._initial_count,
            self.records_per_shard,
            hot_node_ids,
            manifest,
        )
        replacement.graph_revision = (
            self._lazy.graph_revision
            if self._lazy is not None
            else getattr(self.substrate.synapses, "graph_revision", 0)
        )
        replacement.persistence_revision = getattr(self.substrate.synapses, "persistence_revision", 0)
        self._next_lazy = replacement
        return dict(manifest)

    def prepare_commit(self, pointer: Mapping[str, Any]) -> None:
        """Perform all fallible validation before the authoritative pointer."""

        if self._committed or self._index_manifest is None:
            raise RuntimeError("synapse generation has not been staged")
        if pointer.get("activeGeneration") != self._index_manifest.get("sourceGeneration"):
            raise ValueError("synapse index and committed generation differ")
        self._check_unchanged()
        if self._next_lazy is None:
            raise RuntimeError("verified lazy synapse replacement is missing")
        self._prepared_generation = str(pointer["activeGeneration"])

    def commit(self, pointer: Mapping[str, Any]) -> None:
        """Adopt the prepared state after the pointer is durable."""

        if (
            self._committed
            or self._prepared_generation is None
            or pointer.get("activeGeneration") != self._prepared_generation
            or self._next_lazy is None
        ):
            raise RuntimeError("synapse generation was not prepared for commit")
        previous = self.substrate.synapses
        self.substrate.synapses = self._next_lazy
        cached = getattr(self.substrate, "_paged_recall_graph", None)
        if cached is not None and cached[0] == (id(previous), id(self.substrate.neurons)):
            graph = cached[1]
            if graph.refresh(self._next_lazy, self.substrate.neurons):
                self._next_lazy._recall_graph_observer = graph.queue_change
                self.substrate._paged_recall_graph = ((id(self._next_lazy), id(self.substrate.neurons)), graph)
            else:
                self.substrate._paged_recall_graph = None
        self._committed = True
