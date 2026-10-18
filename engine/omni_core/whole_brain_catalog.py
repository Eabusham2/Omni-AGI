"""Actual registered-owner topology, persisted without constructing a model.

Object identity is used ONLY in a temporary SQLite grouping index. Portable
addresses use canonical registered aliases and authenticated parent lineage.
The catalog describes ownership/real bridge registration, not semantic neurons
or measured activity. Parameter/contact populations stay in their paged stores.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Mapping

SHARD_BYTES = 64 * 1024
FORMAT = "omni-whole-brain-catalog"
_PACKED = frozenset(("_packed_forward_weight", "_packed_forward_bias", "_packed_weights", "_packed_sign"))


class InspectionScratch:
    """Entered high-water escrow; streamed pages do not accumulate old leases."""
    def __init__(self, reserve):
        self.reserve, self.high_water, self.stack = reserve, 0, ExitStack()

    def admit(self, byte_count, label):
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("whole-brain scratch amount is invalid")
        if byte_count > self.high_water:
            self.stack.enter_context(self.reserve(byte_count - self.high_water, label))
            self.high_water = byte_count

    def __enter__(self):
        return self

    def __exit__(self, *error):
        return self.stack.__exit__(*error)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _text(value, label):
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError("whole-brain %s is invalid" % label)
    return value


def check_cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("whole-brain inspection cancelled")


def binding_from_metadata(metadata):
    brain_id = _text(metadata.get("brain_id"), "brain identity")
    core = metadata.get("mutable_state") or {}
    substrate = (metadata.get("substrate") or {}).get("persistence") or {}
    if not _sha(core.get("activeGeneration")) or not _sha(substrate.get("activeGeneration")):
        raise ValueError("whole-brain catalog needs actual committed cortical/substrate generations")
    connected = metadata.get("connected_regions")
    graph_states = metadata.get("connected_graph_states")
    for value in (connected, graph_states):
        if value is not None and (not isinstance(value, Mapping) or not _sha(value.get("sha256")) or
                digest({k: v for k, v in value.items() if k != "sha256"}) != value["sha256"]):
            raise ValueError("whole-brain connected descriptor checksum differs")
    overlay = metadata.get("attention_overlay") or {}
    return {"brainId": brain_id, "coreGeneration": core["activeGeneration"],
            "substrateGeneration": substrate["activeGeneration"], "attentionOverlaySha256": digest(overlay),
            "connectedDescriptorSha256": connected.get("sha256") if connected else None,
            "connectedGraphStatesSha256": graph_states.get("sha256") if graph_states else None}


def read_owned_json(root, relative, *, admit, cancelled=None, expected_sha=None):
    """Admit bytes before decoding; bind the opened FD and selected owned path."""
    if not isinstance(relative, str) or "\\" in relative or any(p in ("", ".", "..") for p in relative.split("/")):
        raise ValueError("unsafe whole-brain catalog path")
    root = Path(root).resolve()
    path = root.joinpath(*relative.split("/"))
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
        raise ValueError("whole-brain catalog source is not a regular owned file")
    path.resolve().relative_to(root)
    admit(before.st_size * 8 + SHARD_BYTES, "whole-brain admitted JSON metadata")
    check_cancel(cancelled)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    try:
        if identity(os.fstat(fd)) != identity(before):
            raise ValueError("whole-brain source changed before opening")
        chunks, total = [], 0
        while total < before.st_size:
            check_cancel(cancelled)
            chunk = os.read(fd, min(SHARD_BYTES, before.st_size - total))
            if not chunk:
                raise ValueError("whole-brain source shortened while reading")
            chunks.append(chunk)
            total += len(chunk)
        if os.read(fd, 1):
            raise ValueError("whole-brain source grew beyond admitted metadata")
        encoded = b"".join(chunks)
        path.resolve().relative_to(root)
        if identity(os.fstat(fd)) != identity(before) or identity(path.lstat()) != identity(before):
            raise ValueError("whole-brain source changed while reading")
        actual = hashlib.sha256(encoded).hexdigest()
        if expected_sha is not None and actual != expected_sha:
            raise ValueError("whole-brain catalog checksum differs")
        value = json.loads(encoded)
        if not isinstance(value, dict):
            raise ValueError("whole-brain metadata is not an object")
        return value, actual
    finally:
        os.close(fd)


def _blob(root, body, *, admit, reserve_disk, cancelled):
    encoded = canonical(body)
    admit(len(encoded) * 3 + SHARD_BYTES, "whole-brain catalog shard")
    reserve_disk(len(encoded), "whole-brain immutable catalog shard")
    check_cancel(cancelled)
    sha = hashlib.sha256(encoded).hexdigest()
    name = "whole-brain-catalog/blobs/%s.json" % sha
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".shard-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            read_owned_json(root, name, admit=admit, cancelled=cancelled, expected_sha=sha)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha


def _address(kind, brain_id, value):
    return kind + "-" + digest([kind, brain_id, value])


def _region_identity(module, parent, brain_id, native_path=()):
    identity = getattr(module, "identity", None)
    if identity is None:
        return None
    region = getattr(identity, "region_id", None)
    source = getattr(identity, "parent_id", None)
    generation = getattr(identity, "generation_sha256", None)
    if not isinstance(region, str) or not isinstance(source, str) or not _sha(generation):
        raise ValueError("registered parent region has invalid actual lineage")
    value = {"regionId": region, "parentId": source, "generationSha256": generation}
    return {"kind": "region", "id": _address("region", brain_id, [parent, value]), "parentRegionId": parent,
            "label": region, "regionKind": "connected-parent", "lineage": value,
            "nativeRegionPath": [*native_path, region],
            "evidence": "actual-NativeRegion-identity"}


def publish_whole_brain_catalog(brain, metadata, engine_path, *, admit, reserve_disk, cancelled=None, tensor_registry=None):
    """Save hook AFTER staged source pointers, BEFORE atomic brain.json publication.

    Caller owns the native writer and lifetime RAM/disk admission. Unreferenced
    immutable shards after interruption are not a learned-state publication.
    """
    binding = binding_from_metadata(metadata)
    brain_id = binding["brainId"]
    if getattr(brain, "brain_id", None) != brain_id:
        raise ValueError("whole-brain catalog current owner differs")
    roots = brain._ternary_export_roots()
    if not isinstance(roots, Mapping) or set(roots) != {"decoder", "memory_bridge", "idea_adapter", "router", "liquid", "modalities"}:
        raise ValueError("whole-brain actual trainable-role roots differ")
    engine_path = Path(engine_path)
    directory = engine_path / "whole-brain-catalog"
    directory.mkdir(parents=True, exist_ok=True)
    admit(512 * 1024, "whole-brain disk-index/one-shard scratch")
    reserve_disk(64 * 1024, "whole-brain temporary alias index")
    fd, temporary = tempfile.mkstemp(prefix=".aliases-", suffix=".sqlite3", dir=directory)
    os.close(fd)
    database = sqlite3.connect(temporary)
    from .whole_brain_catalog_index import CatalogIndexWriter
    location_index = None
    counts = {"regions": 0, "owners": 0, "aliases": 0, "memberships": 0, "bridges": 0, "inspections": 0}
    try:
        location_index = CatalogIndexWriter(engine_path, admit, reserve_disk, cancelled)
        database.executescript("""PRAGMA journal_mode=DELETE; PRAGMA cache_size=-128; PRAGMA temp_store=FILE;
          CREATE TABLE aliases(alias TEXT PRIMARY KEY, object_key TEXT NOT NULL, body TEXT NOT NULL);
          CREATE INDEX alias_objects ON aliases(object_key,alias);
          CREATE TABLE memberships(object_key TEXT,region_id TEXT,PRIMARY KEY(object_key,region_id));
          CREATE TABLE regions(id TEXT PRIMARY KEY,body TEXT NOT NULL);
          CREATE TABLE inspections(alias TEXT,role TEXT,tensor_key TEXT,field TEXT,PRIMARY KEY(alias,role,tensor_key));
          CREATE TABLE bridges(object_key TEXT,source TEXT,target TEXT,exchange TEXT,PRIMARY KEY(object_key,source,target,exchange));""")

        def charge(value):
            check_cancel(cancelled)
            reserve_disk(len(canonical(value)) * 4 + 512, "whole-brain alias-index row")

        def region(value):
            charge(value)
            database.execute("INSERT OR IGNORE INTO regions VALUES(?,?)", (value["id"], canonical(value).decode()))

        root_id = _address("region", brain_id, ["whole-brain"])
        region({"kind": "region", "id": root_id, "parentRegionId": None, "label": "Whole brain", "regionKind": "whole-brain",
                "lineage": {"brainId": brain_id}, "evidence": "actual-brain-owner"})
        for role, root in sorted(roots.items()):
            role_id = _address("region", brain_id, ["registered-role", role])
            region({"kind": "region", "id": role_id, "parentRegionId": root_id, "label": role,
                    "regionKind": "registered-role", "lineage": {"registeredRoot": role}, "evidence": "actual-trainable-role-reference"})
            # Explicit stack preserves every registered alias, including shared
            # objects, without a whole-model tensor list or recursion ceiling.
            stack = [(role, root, (role_id,), frozenset(), ())]
            frontier_bytes = len(role) + 128
            while stack:
                alias, module, memberships, ancestors, native_path = stack.pop()
                frontier_bytes -= len(alias) + 128 + len(ancestors) * 16
                check_cancel(cancelled)
                key = str(id(module))  # temporary equality grouping ONLY
                if key in ancestors:
                    raise ValueError("cyclic registered module graph is unsupported")
                native_region = _region_identity(module, memberships[-1], brain_id, native_path)
                if native_region:
                    region(native_region)
                    memberships = (*memberships, native_region["id"])
                    native_path = tuple(native_region["nativeRegionPath"])
                buffers = getattr(module, "_buffers", {})
                packed = []
                for name in sorted(_PACKED.intersection(buffers)):
                    value = buffers[name]
                    if value is not None:
                        packed.append({"field": name, "shape": [int(n) for n in value.shape]})
                value = {"registeredAlias": alias, "className": type(module).__name__, "packedFields": packed}
                charge(value)
                database.execute("INSERT INTO aliases VALUES(?,?,?)", (alias, key, canonical(value).decode()))
                for field in packed:
                    tensor_key = alias + "." + field["field"]
                    for saved_role, registry in (tensor_registry or {}).items():
                        if saved_role not in ("core", "plasticity"):
                            raise ValueError("whole-brain save role registry is invalid")
                        if tensor_key in registry:
                            tensor = registry[tensor_key]
                            if tuple(int(n) for n in tensor.shape) != tuple(field["shape"]):
                                raise ValueError("actual saved tensor differs from registered packed owner")
                            database.execute("INSERT INTO inspections VALUES(?,?,?,?)", (alias, saved_role, tensor_key,
                                             "bias" if field["field"] == "_packed_forward_bias" else "weight"))
                for member in memberships:
                    database.execute("INSERT OR IGNORE INTO memberships VALUES(?,?)", (key, member))
                # Only the actual ConnectedOmniDecoder references and exact
                # RegionExchange registration establish regional bridge ends.
                children = getattr(module, "regions", None)
                if children is not None and hasattr(module, "workspace_exchange") and hasattr(module, "cortical_exchange"):
                    actual_regions = [_region_identity(child, memberships[-1], brain_id, native_path) for child in children]
                    if any(item is None for item in actual_regions):
                        raise ValueError("connected exchange has an unidentified actual parent")
                    for exchange_name in ("workspace_exchange", "cortical_exchange"):
                        exchange = getattr(module, exchange_name)
                        bridges = getattr(getattr(exchange, "bridges", None), "_modules", None)
                        widths = getattr(exchange, "widths", ())
                        if not isinstance(bridges, Mapping) or len(widths) != len(actual_regions):
                            raise ValueError("actual regional exchange registration is incomplete")
                        for target in range(len(actual_regions)):
                            for source in range(len(actual_regions)):
                                if source == target:
                                    continue
                                bridge = bridges.get("r%d_to_r%d" % (source, target))
                                if bridge is None:
                                    raise ValueError("actual registered regional bridge is missing")
                                row = (str(id(bridge)), actual_regions[source]["id"], actual_regions[target]["id"], exchange_name)
                                charge(row)
                                database.execute("INSERT OR IGNORE INTO bridges VALUES(?,?,?,?)", row)
                children = getattr(module, "_modules", {})
                if not isinstance(children, Mapping):
                    raise ValueError("actual registered module children are invalid")
                for name, child in children.items():
                    if child is not None:
                        _text(name, "registered child")
                        if "." in name:
                            raise ValueError("registered child cannot contain a path separator")
                        child_alias = alias + "." + name
                        frontier_bytes += len(child_alias) + 128 + (len(ancestors) + 1) * 16
                        admit(512 * 1024 + frontier_bytes, "whole-brain registered traversal frontier")
                        stack.append((child_alias, child, memberships, ancestors | {key}, native_path))
        database.commit()
        database.execute("CREATE TABLE owners AS SELECT object_key,MIN(alias) AS alias FROM aliases GROUP BY object_key")
        database.execute("CREATE UNIQUE INDEX owner_objects ON owners(object_key)")
        tail = None
        shards = records = 0
        page, page_bytes = [], 0

        def append(value):
            nonlocal tail, shards, records, page, page_bytes
            encoded = canonical(value)
            admit(512 * 1024 + len(encoded) * 4, "whole-brain catalog record")
            if page and page_bytes + len(encoded) > SHARD_BYTES:
                tail = _blob(engine_path, {"format": FORMAT, "version": 1, "previousSha256": tail, "records": page},
                             admit=admit, reserve_disk=reserve_disk, cancelled=cancelled)
                location_index.add(tail, page)
                shards += 1
                page, page_bytes = [], 0
            page.append(value)
            page_bytes += len(encoded)
            records += 1

        # Canonical aliases are selected with a disk index. Nothing that names
        # a process object or address enters the immutable published rows.
        owner_id = lambda alias: _address("owner", brain_id, alias)
        for key, alias, body in database.execute("SELECT o.object_key,o.alias,a.body FROM owners o JOIN aliases a ON a.alias=o.alias ORDER BY o.alias"):
            value = json.loads(body)
            append({"kind": "owner", "id": owner_id(alias), "canonicalAlias": alias, "className": value["className"],
                    "packedFields": value["packedFields"], "evidence": "actual-registered-object"})
            counts["owners"] += 1
        for alias, canonical_alias in database.execute("SELECT a.alias,o.alias FROM aliases a JOIN owners o USING(object_key) ORDER BY a.alias"):
            append({"kind": "alias", "id": _address("alias", brain_id, alias), "alias": alias, "ownerId": owner_id(canonical_alias),
                    "evidence": "same-actual-registered-object"})
            counts["aliases"] += 1
        for alias, role, tensor_key, field, canonical_alias in database.execute("SELECT i.alias,i.role,i.tensor_key,i.field,o.alias FROM inspections i JOIN aliases a ON a.alias=i.alias JOIN owners o USING(object_key) ORDER BY i.tensor_key,i.role"):
            append({"kind": "inspection", "id": _address("inspection", brain_id, [role, tensor_key]),
                    "ownerId": owner_id(canonical_alias), "role": role, "moduleId": tensor_key, "tensorKey": tensor_key,
                    "field": field, "evidence": "actual-save-role-and-registered-buffer-key"})
            counts["inspections"] += 1
        for member, alias in database.execute("SELECT m.region_id,o.alias FROM memberships m JOIN owners o USING(object_key) ORDER BY m.region_id,o.alias"):
            append({"kind": "membership", "id": _address("membership", brain_id, [member, alias]), "regionId": member,
                    "ownerId": owner_id(alias), "evidence": "actual-registered-containment-not-semantic-neuron-overlap"})
            counts["memberships"] += 1
        for key, source, target, exchange in database.execute("SELECT object_key,source,target,exchange FROM bridges ORDER BY source,target,exchange"):
            row = database.execute("SELECT alias FROM owners WHERE object_key=?", (key,)).fetchone()
            if not row or not row[0]:
                raise ValueError("regional bridge has no actual registered packed owner")
            owner = owner_id(row[0])
            append({"kind": "bridge", "id": _address("bridge", brain_id, [owner, source, target, exchange]), "sourceRegionId": source,
                    "targetRegionId": target, "ownerId": owner, "exchange": exchange,
                    "evidence": "actual-RegionExchange-registration", "learnedNonzeroObserved": False, "firingObserved": False})
            counts["bridges"] += 1
        for (body,) in database.execute("SELECT body FROM regions ORDER BY id"):
            append(json.loads(body))
            counts["regions"] += 1
        if page:
            tail = _blob(engine_path, {"format": FORMAT, "version": 1, "previousSha256": tail, "records": page},
                         admit=admit, reserve_disk=reserve_disk, cancelled=cancelled)
            location_index.add(tail, page)
            shards += 1
        index_descriptor = location_index.finish()
        return {"format": FORMAT, "version": 1, "brainId": brain_id, "binding": binding, "bindingSha256": digest(binding),
                "tailSha256": tail, "shards": shards, "records": records, "counts": counts,
                "index": index_descriptor,
                "semanticAxisMapping": False, "activityObserved": False}
    finally:
        if location_index is not None:
            location_index.close()
        database.close()
        Path(temporary).unlink(missing_ok=True)
        Path(temporary + "-journal").unlink(missing_ok=True)


def validate_catalog_pointer(pointer, metadata=None):
    binding = binding_from_metadata(metadata) if metadata is not None else pointer.get("binding") if isinstance(pointer, Mapping) else None
    if (not isinstance(binding, Mapping) or set(binding) != {"brainId", "coreGeneration", "substrateGeneration", "attentionOverlaySha256", "connectedDescriptorSha256", "connectedGraphStatesSha256"}
            or not isinstance(binding.get("brainId"), str) or not binding["brainId"]
            or any(not _sha(binding.get(name)) for name in ("coreGeneration", "substrateGeneration", "attentionOverlaySha256"))
            or any(binding.get(name) is not None and not _sha(binding[name]) for name in ("connectedDescriptorSha256", "connectedGraphStatesSha256"))):
        raise ValueError("whole-brain catalog binding shape differs")
    if not isinstance(pointer, Mapping) or pointer.get("format") != FORMAT or pointer.get("version") != 1 or pointer.get("brainId") != binding["brainId"]:
        raise ValueError("whole-brain catalog owner/format differs")
    if pointer.get("binding") != binding or pointer.get("bindingSha256") != digest(binding) or not _sha(pointer.get("tailSha256")):
        raise ValueError("whole-brain catalog was not published with these source generations")
    counts = pointer.get("counts")
    if (not isinstance(counts, Mapping) or set(counts) != {"regions", "owners", "aliases", "memberships", "bridges", "inspections"}
            or any(type(value) is not int or value < 0 for value in counts.values()) or sum(counts.values()) != pointer.get("records")
            or type(pointer.get("shards")) is not int or pointer["shards"] < 1
            or pointer.get("semanticAxisMapping") is not False or pointer.get("activityObserved") is not False):
        raise ValueError("whole-brain catalog declared counts/evidence differ")
    index = pointer.get("index")
    if (not isinstance(index, Mapping) or not _sha(index.get("sha256")) or index.get("path") != "whole-brain-catalog/index/%s.sqlite3" % index["sha256"]
            or type(index.get("bytes")) is not int or index["bytes"] < 1):
        raise ValueError("whole-brain catalog index descriptor differs")
    return dict(pointer)


def iter_catalog_files(engine_path, metadata, *, admit, cancelled=None):
    """Exact current snapshot/archive closure; never glob historical catalogs."""
    pointer = metadata.get("whole_brain_catalog")
    if pointer is None:
        return
    pointer = validate_catalog_pointer(pointer, metadata)
    yield from _iter_pointer_files(engine_path, pointer, admit=admit, cancelled=cancelled)


def _iter_pointer_files(engine_path, pointer, *, admit, cancelled=None):
    pointer = validate_catalog_pointer(pointer)
    from .whole_brain_catalog_index import catalog_index
    with catalog_index(engine_path, pointer["index"], admit=admit, cancelled=cancelled):
        yield pointer["index"]["path"], pointer["index"]["bytes"], pointer["index"]["sha256"]
    sha, remaining, rows = pointer["tailSha256"], pointer["shards"], 0
    while sha is not None:
        check_cancel(cancelled)
        if not remaining or not _sha(sha):
            raise ValueError("whole-brain catalog chain/count differs")
        name = "whole-brain-catalog/blobs/%s.json" % sha
        body, _ = read_owned_json(engine_path, name, admit=admit, cancelled=cancelled, expected_sha=sha)
        if body.get("format") != FORMAT or body.get("version") != 1 or not isinstance(body.get("records"), list):
            raise ValueError("whole-brain catalog shard format differs")
        rows += len(body["records"])
        yield name, Path(engine_path).joinpath(*name.split("/")).stat().st_size, sha
        sha, remaining = body.get("previousSha256"), remaining - 1
    if remaining or rows != pointer["records"]:
        raise ValueError("whole-brain catalog closure lost records")


def validate_catalog_lineage_entry(value):
    if not isinstance(value, Mapping):
        raise ValueError("whole-brain catalog lineage entry is invalid")
    pointer = validate_catalog_pointer(value.get("catalogPointer"))
    for name in ("entryKeySha256", "entrySha256", "sourceMetadataSha256", "catalogBindingSha256", "coreGeneration", "substrateGeneration"):
        if not _sha(value.get(name)):
            raise ValueError("whole-brain catalog lineage digest differs")
    if (pointer["brainId"] != value.get("sourceBrainId") or pointer["bindingSha256"] != value["catalogBindingSha256"]
            or pointer["binding"]["coreGeneration"] != value["coreGeneration"] or pointer["binding"]["substrateGeneration"] != value["substrateGeneration"]):
        raise ValueError("whole-brain catalog lineage source binding differs")
    key = digest({"sourceBrainId": value["sourceBrainId"], "sourceMetadataSha256": value["sourceMetadataSha256"],
                  "catalogBindingSha256": value["catalogBindingSha256"], "tailSha256": pointer["tailSha256"], "indexSha256": pointer["index"]["sha256"]})
    body = {name: value.get(name) for name in ("entryKeySha256", "sourceBrainId", "sourceMetadataSha256", "catalogBindingSha256", "coreGeneration", "substrateGeneration", "closureRoot", "catalogPointer")}
    if value["entryKeySha256"] != key or value.get("closureRoot") != "whole-brain-catalog/lineage/sources/" + key or value["entrySha256"] != digest(body):
        raise ValueError("whole-brain catalog lineage entry checksum differs")
    return dict(value)


def iter_catalog_lineage_files(engine_path, metadata, *, admit, cancelled=None):
    """Retained original source catalogs only; never current-owner topology."""
    pointer = metadata.get("whole_brain_catalog_lineage")
    if pointer is None:
        return
    if not isinstance(pointer, Mapping):
        raise ValueError("whole-brain catalog lineage pointer is invalid")
    body = {name: pointer.get(name) for name in ("format", "version", "tailSha256", "shards", "entries")}
    if (body["format"] != "omni-whole-brain-catalog-lineage" or body["version"] != 1 or not _sha(body["tailSha256"])
            or type(body["shards"]) is not int or body["shards"] < 1 or type(body["entries"]) is not int or body["entries"] < 1
            or pointer.get("manifestSha256") != digest(body)):
        raise ValueError("whole-brain catalog lineage pointer differs")
    sha, remaining, seen = body["tailSha256"], body["shards"], {}
    while sha is not None:
        if not remaining or not _sha(sha):
            raise ValueError("whole-brain catalog lineage chain exceeds its declaration")
        name = "whole-brain-catalog/lineage/blobs/%s.json" % sha
        shard, _ = read_owned_json(engine_path, name, admit=admit, cancelled=cancelled, expected_sha=sha)
        if shard.get("format") != "omni-whole-brain-catalog-lineage-shard" or shard.get("version") != 1 or not isinstance(shard.get("entries"), list):
            raise ValueError("whole-brain catalog lineage shard differs")
        yield name, Path(engine_path).joinpath(*name.split("/")).stat().st_size, sha
        for value in shard["entries"]:
            entry = validate_catalog_lineage_entry(value)
            key = entry["entryKeySha256"]
            prior = seen.get(key)
            if prior is not None and prior != entry["entrySha256"]:
                raise ValueError("conflicting retained whole-brain source catalogs")
            if prior is not None:
                continue
            admit(512 * 1024 + (len(seen) + 1) * 256, "whole-brain retained lineage key proofs")
            seen[key] = entry["entrySha256"]
            source_root = Path(engine_path).joinpath(*entry["closureRoot"].split("/"))
            source_root.resolve().relative_to(Path(engine_path).resolve())
            for relative, size, checksum in _iter_pointer_files(source_root, entry["catalogPointer"], admit=admit, cancelled=cancelled):
                yield entry["closureRoot"] + "/" + relative, size, checksum
        sha, remaining = shard.get("previousSha256"), remaining - 1
    if remaining or len(seen) != body["entries"]:
        raise ValueError("whole-brain catalog lineage closure lost entries")
