"""Bounded immutable group blobs for the disposable forward topology cache.

The authoritative neural checkpoint remains v3. Derived cache v4 stores one
bounded complete structural group per content-addressed file and a small
descriptor manifest. Unchanged groups reuse descriptors without materializing
all forward edges/hot locators or rewriting a corpus-sized JSON body.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .authenticated_paged_cache import canonical


def group_payload(root: Path, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    from .paged_substrate_writer import _verified_file, remember_verified_blob
    spec = descriptor["entry"]
    checksum = spec["sha256"]
    relative = "forward-index/blobs/%s.json" % checksum
    if spec.get("path") != relative or type(spec.get("bytes")) is not int or spec["bytes"] < 0:
        raise ValueError("derived forward group path/size is invalid")
    path = root / relative
    _verified_file(path, checksum, spec["bytes"])
    before = path.stat()
    payload = path.read_bytes()
    if len(payload) != spec["bytes"] or hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError("derived forward group changed during read")
    value = json.loads(payload)
    if not isinstance(value, dict) or canonical(value) != payload:
        raise ValueError("derived forward group is not canonical")
    for key in ("bucket", "part", "count", "recordsSha256", "tensorsSha256", "firstId", "lastId", "synapticUses"):
        if value.get(key) != descriptor.get(key):
            raise ValueError("derived forward group descriptor binding is stale")
    if len(value.get("forwardRecords", ())) != descriptor.get("forwardCount"):
        raise ValueError("derived forward group count is invalid")
    remember_verified_blob(path, checksum, before=before)
    return value


class ReusedForwardEntry(Mapping[str, Any]):
    """Descriptor reference which decodes a group only when explicitly read."""

    def __init__(self, root: Path, descriptor: Mapping[str, Any]) -> None:
        self.root = root
        self.descriptor = dict(descriptor)

    def _value(self):
        return group_payload(self.root, self.descriptor)

    def __getitem__(self, key):
        if key in self.descriptor and key != "entry":
            return self.descriptor[key]
        return self._value()[key]

    def __iter__(self):
        return iter(self._value())

    def __len__(self):
        return len(self._value())


def iter_entries(root: Path, manifest: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    if manifest.get("formatVersion") == 4:
        for descriptor in manifest["shards"]:
            yield group_payload(root, descriptor)
    else:
        yield from manifest["shards"]


def publish_entries(root: Path, entries, guard) -> list[dict[str, Any]]:
    from .paged_substrate_writer import _verified_file, _write_immutable_bytes
    result = []
    owner = SimpleNamespace(growth_guard=guard)
    seen = set()
    for entry in entries:
        if isinstance(entry, ReusedForwardEntry):
            descriptor = dict(entry.descriptor)
            spec = descriptor["entry"]
            _verified_file(root / spec["path"], spec["sha256"], spec["bytes"])
        else:
            value = dict(entry)
            payload = canonical(value)
            sha = hashlib.sha256(payload).hexdigest()
            relative = "forward-index/blobs/%s.json" % sha
            _write_immutable_bytes(owner, root, relative, payload, None, max_blob_bytes=64 * 1024 * 1024)
            descriptor = {key: value[key] for key in
                          ("bucket", "part", "count", "recordsSha256", "tensorsSha256", "firstId", "lastId", "synapticUses")}
            descriptor.update(forwardCount=len(value["forwardRecords"]),
                              packedBytes=len(base64.b64decode(value["packedEffectiveWeights"], validate=True)),
                              entry={"path": relative, "sha256": sha, "bytes": len(payload)})
        key = descriptor["bucket"], descriptor["part"]
        if key in seen:
            raise ValueError("derived forward group is repeated")
        seen.add(key)
        result.append(descriptor)
    return sorted(result, key=lambda value: (value["bucket"], value["part"]))


class BlobForwardTopology(Mapping[tuple[str, int], tuple[list[tuple], bytes]]):
    """At most one immutable group is decoded at a time, never the full graph."""

    def __init__(self, root: Path, descriptors: Sequence[Mapping[str, Any]]) -> None:
        self.root = root
        self.descriptors = {(row["bucket"], row["part"]): dict(row) for row in descriptors}

    def __getitem__(self, key):
        value = group_payload(self.root, self.descriptors[key])
        structures = value["forwardRecords"]
        packed = base64.b64decode(value["packedEffectiveWeights"], validate=True)
        from .vsa import _pack_ternary_levels, _unpack_ternary_level, _synapse_id_matches_endpoints
        if not isinstance(structures, list) or any(
            not isinstance(row, list) or len(row) != 4 or type(row[0]) is not int
            or not 0 <= row[0] < value["count"] or not all(isinstance(item, str) and item for item in row[1:])
            or not _synapse_id_matches_endpoints(row[1], row[2], row[3]) for row in structures
        ) or [row[0] for row in structures] != sorted({row[0] for row in structures}):
            raise ValueError("paged forward contribution structure is invalid")
        levels = [_unpack_ternary_level(packed, index) for index in range(len(structures))]
        if any(level == 0 for level in levels) or _pack_ternary_levels(levels) != packed:
            raise ValueError("paged forward packed sign is invalid")
        return [tuple(row) for row in structures], packed

    def __iter__(self):
        return iter(self.descriptors)

    def __len__(self):
        return len(self.descriptors)

    def reused_entry(self, key):
        return ReusedForwardEntry(self.root, self.descriptors[key])


class _IncidentLocations(Sequence):
    def __init__(self, owner: "EndpointHotLocations", node_id: str) -> None:
        self.owner, self.node_id = owner, node_id

    def __iter__(self):
        for key in self.owner.incident_keys(self.node_id):
            entry = group_payload(self.owner.root, self.owner.topology.descriptors[key])
            for node, identifier, index in entry["hotLocations"]:
                if node == self.node_id:
                    yield identifier, key, index

    def __len__(self):
        return sum(1 for _item in self)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return list(self)[index]
        if index < 0:
            index += len(self)
        for position, value in enumerate(self):
            if position == index:
                return value
        raise IndexError("hot location is out of range")


class EndpointHotLocations(Mapping[str, Sequence]):
    """Exact hot node→incident groups backed by the complete endpoint index.

    When no verified paged endpoint owner exists, a bounded streamed group
    scan is retained as the compatibility fallback. Production paged stores
    use the authenticated complete endpoint lookup, including zero weights.
    """

    def __init__(self, root: Path, topology: BlobForwardTopology, hot_ids, manifest) -> None:
        self.root, self.topology, self.hot_ids, self.manifest = root, topology, hot_ids, manifest
        view = getattr(hot_ids, "view", None)
        self.index_owner = getattr(view, "index", getattr(hot_ids, "index", None))

    def incident_keys(self, node_id):
        if node_id not in self.hot_ids:
            return
        if self.index_owner is None:
            for key in self.topology:
                entry = group_payload(self.root, self.topology.descriptors[key])
                if any(row[0] == node_id for row in entry["hotLocations"]):
                    yield key
            return
        from .paged_synapse_endpoints import SynapseEndpointIndex
        endpoint_index = getattr(self.index_owner, "_synapse_endpoint_index", None)
        if endpoint_index is None:
            with self.index_owner._transaction() as connection:
                has_endpoint_schema = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='synapse_endpoint_meta'"
                ).fetchone() is not None
            if not has_endpoint_schema:
                # Compatibility producers without this optional derived
                # index still perform an exact bounded full group traversal.
                # A present malformed/stale index is never treated as empty.
                for key in self.topology:
                    entry = group_payload(self.root, self.topology.descriptors[key])
                    if any(row[0] == node_id for row in entry["hotLocations"]):
                        yield key
                return
            from .authenticated_paged_cache import cache_session
            endpoint_index = SynapseEndpointIndex(cache_session(self.index_owner))
            self.index_owner._synapse_endpoint_index = endpoint_index
        with endpoint_index.session.transaction() as connection:
            meta = endpoint_index._meta(connection)
            if (meta["generation"] != self.manifest["sourceGeneration"]
                or meta["manifestSha256"] != self.manifest["sourceGenerationManifestSha256"]
                or meta.get("verifiedForwardSha256") != self.manifest["contentSha256"]):
                raise ValueError("hot endpoint query index is stale for this generation")
        for key in endpoint_index.incident_groups(node_id):
            if key not in self.topology.descriptors:
                raise ValueError("hot endpoint query names an unknown group")
            yield key

    def __getitem__(self, key):
        if key not in self.hot_ids:
            raise KeyError(key)
        return _IncidentLocations(self, key)

    def __iter__(self):
        return iter(self.hot_ids)

    def __len__(self):
        if self.index_owner is not None:
            return int(self.index_owner.status()["count"])
        return sum(1 for _identifier in self.hot_ids)

    def for_group(self, key):
        return group_payload(self.root, self.topology.descriptors[key])["hotLocations"]
