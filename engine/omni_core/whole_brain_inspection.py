"""One generation-fenced graph interface over actual paged native sources."""
from __future__ import annotations

import base64
import json
import hashlib
import hmac
import os
from pathlib import Path

from .committed_substrate_source import VerifiedCommittedSubstrateSource
from .inspection_resources import inspection_resources
from .whole_brain_catalog import (FORMAT, _sha, canonical, digest, check_cancel,
                                 read_owned_json, validate_catalog_pointer)
_CURSOR_KEY = os.urandom(32)  # process-local paging only; never portable authority


def _cursor(value):
    try:
        envelope = json.loads(base64.urlsafe_b64decode(value))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid whole-brain cursor") from error
    if not isinstance(envelope, dict) or not isinstance(envelope.get("body"), dict) or not isinstance(envelope.get("mac"), str):
        raise ValueError("invalid whole-brain cursor")
    decoded = envelope["body"]
    if not hmac.compare_digest(envelope["mac"], hmac.new(_CURSOR_KEY, canonical(decoded), hashlib.sha256).hexdigest()):
        raise ValueError("whole-brain cursor is not a current owned traversal")
    return decoded


def _encode(value):
    return base64.urlsafe_b64encode(canonical({"body": value, "mac": hmac.new(_CURSOR_KEY, canonical(value), hashlib.sha256).hexdigest()})).decode("ascii")


def _catalog_shard(engine, sha, admit, cancelled):
    if not _sha(sha):
        raise ValueError("invalid whole-brain shard identity")
    body, _ = read_owned_json(engine, "whole-brain-catalog/blobs/%s.json" % sha,
                             admit=admit, cancelled=cancelled, expected_sha=sha)
    if body.get("format") != FORMAT or body.get("version") != 1 or not isinstance(body.get("records"), list):
        raise ValueError("invalid whole-brain shard")
    previous = body.get("previousSha256")
    if previous is not None and not _sha(previous):
        raise ValueError("invalid whole-brain previous shard")
    return body


def _find_region(engine, pointer, region_id, admit, cancelled):
    from .whole_brain_catalog_index import catalog_index
    with catalog_index(engine, pointer["index"], admit=admit, cancelled=cancelled) as index:
        record = index.get(region_id)
        if record is None or record.get("kind") != "region":
            raise ValueError("selected whole-brain region is not in the current catalog")
        return record


def _regional_source(engine, metadata, record, assert_current):
    path = record.get("nativeRegionPath")
    if not isinstance(path, list) or not path or any(not isinstance(part, str) or not part or "/" in part or "\\" in part or part in (".", "..") for part in path):
        raise ValueError("selected region has no verified substrate path")
    states = metadata.get("connected_graph_states")
    base = Path(engine).parent
    selected = None
    for index, part in enumerate(path):
        if not isinstance(states, dict) or states.get("sha256") != digest({k: v for k, v in states.items() if k != "sha256"}):
            raise ValueError("regional graph-state descriptor checksum differs")
        selected = states.get("regions", {}).get(part)
        expected = "engine/connected-state/" + part
        if not isinstance(selected, dict) or selected.get("enginePath") != expected:
            raise ValueError("regional graph source is not the exact owned path")
        selected_engine = base / expected
        selected_engine.resolve().relative_to(Path(engine).resolve())
        if index != len(path) - 1:
            if selected.get("nestedStateRoot") != expected + "/nested":
                raise ValueError("nested regional graph-state root differs")
            base, states = base / selected["nestedStateRoot"], selected.get("nestedGraphStates")
    source_brain = record.get("lineage", {}).get("parentId")
    if not isinstance(source_brain, str) or not source_brain:
        raise ValueError("regional graph lineage has no actual parent owner")
    trusted = {"brain_id": source_brain, "substrate": selected["substrate"], "attention_overlay": selected.get("attentionOverlay", {})}
    return VerifiedCommittedSubstrateSource(selected_engine, source_brain, trusted, assert_current)


def _stream_json_size(value):
    """Exact canonical JSON byte size with bounded individual string copies."""
    if isinstance(value, str):
        return 2 + sum(len(canonical(value[start:start + 1024])) - 2 for start in range(0, len(value), 1024))
    if isinstance(value, dict):
        return 2 + max(0, len(value) - 1) + sum(_stream_json_size(key) + 1 + _stream_json_size(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return 2 + max(0, len(value) - 1) + sum(_stream_json_size(item) for item in value)
    return len(canonical(value))


def _actual_sparse_endpoints(source, engine, brain_id, *, descriptor=None, admit, cancelled=None, factories=None):
    """Resolve only the returned page's real sparse endpoints, never placeholders.

    The ordinary reader is unchanged. Added child-assembly anchors keep their
    own relationships paged rather than recursively expanding the whole graph.
    """
    if source.get("entity") not in ("synapses", "assemblies") or not (source.get("synapses") or source.get("assemblies")):
        return source, []
    admit(512 * 1024, "whole-brain bounded sparse size/traversal scratch")
    if factories is None:
        from .substrate_inspection import PersistedSubstrateView, _inspect_neuron, _inspect_assembly
        from .paged_inspection_queries import PagedInspectionIndex
        factories = (PersistedSubstrateView.open, PagedInspectionIndex, _inspect_neuron, _inspect_assembly)
    open_view, make_index, inspect_neuron, inspect_assembly = factories
    view = open_view(engine, brain_id, _committed_source=descriptor)
    if source.get("revision") != view.revision:
        raise ValueError("sparse endpoint anchors differ from the selected source revision")
    index = make_index(view)
    request_count = len(source.get("assemblies", [])) + 2 * len(source.get("synapses", []))
    request_count += sum(len(row.get("neuronIds", [])) + len(row.get("childAssemblyIds", [])) for row in source.get("assemblies", []))
    held = _stream_json_size(source) * 8 + request_count * 256 + 512 * 1024
    admit(held, "whole-brain selected sparse page and endpoint requests")
    neurons = {row["id"]: row for row in source.get("neurons", [])}
    assemblies = {row["id"]: row for row in source.get("assemblies", [])}
    primary_assemblies = set(assemblies)
    queue = [("assemblies", identifier) for identifier in primary_assemblies]
    queue.extend(("neurons", row[field]) for row in source.get("synapses", []) for field in ("sourceId", "targetId"))
    for row in source.get("assemblies", []):
        queue.extend(("neurons", identifier) for identifier in row.get("neuronIds", []))
        queue.extend(("assemblies", identifier) for identifier in row.get("childAssemblyIds", []))
    visited, aliases = set(), []
    with index.connection() as connection:
        while queue:
            check_cancel(cancelled)
            kind, identifier = queue.pop()
            if (kind, identifier) in visited:
                continue
            visited.add((kind, identifier))
            size = connection.execute("SELECT length(payload) FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            if size is None:
                raise ValueError("selected sparse relationship has no actual %s endpoint: %s" % (kind, identifier))
            admit(held + int(size[0]) * 8, "whole-brain exact indexed endpoint metadata")
            encoded = connection.execute("SELECT payload FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            if encoded is None or len(encoded[0]) != int(size[0]):
                raise ValueError("indexed sparse endpoint changed while reading")
            raw = json.loads(encoded[0])
            if raw.get("id", raw.get("neuron_id")) != identifier:
                raise ValueError("indexed sparse endpoint identity differs")
            held += int(size[0]) * 8 + 256
            if held > (16 * 1024 * 1024) * 8:
                # This is a response/metadata viewport envelope, never a model
                # cardinality limit or silently omitted relationship.
                raise ValueError("sparse endpoint viewport exceeds transport scratch; narrow the page or inspect connected synapses")
            if kind == "neurons":
                if identifier not in neurons:
                    neurons[identifier] = inspect_neuron(view, raw)
            else:
                if identifier not in assemblies:
                    anchor = inspect_assembly(view, raw)
                    anchor = {**anchor, "neuronIds": [], "childAssemblyIds": [], "relationshipsPaged": bool(anchor.get("neuronCount") or anchor.get("childAssemblyCount"))}
                    assemblies[identifier] = anchor
                backing = raw.get("assembly_neuron_id")
                if backing is not None:
                    if backing != identifier:
                        raise ValueError("assembly backing-neuron alias differs from the native shared-row contract")
                    aliases.append({"assemblyId": identifier, "neuronId": backing, "evidence": "actual-assembly_neuron_id"})
                    queue.append(("neurons", backing))
    for identifier, assembly in assemblies.items():
        if identifier in neurons:
            assembly.update(activation=neurons[identifier]["activation"], activationObserved=True)
    result = {**source, "neurons": list(neurons.values()), "assemblies": list(assemblies.values())}
    if _stream_json_size(result) > 16 * 1024 * 1024:
        raise ValueError("sparse endpoint response exceeds the inspected transport viewport; narrow the page")
    return result, aliases


def _selected_cortex_owner_anchors(engine, pointer, selected_module, *, admit, cancelled=None, page_size=64):
    """Exact saved-key join, not a guess from an exported module-name prefix."""
    if pointer is None or selected_module is None:
        return [], None
    from .whole_brain_catalog import _address
    from .whole_brain_catalog_index import catalog_index
    role, tensor_key = selected_module.get("role"), selected_module.get("id")
    if role not in ("core", "plasticity") or not isinstance(tensor_key, str):
        raise ValueError("selected cortical matrix has no exact source role/key")
    binding_id = _address("inspection", pointer["brainId"], [role, tensor_key])
    with catalog_index(engine, pointer["index"], admit=admit, cancelled=cancelled) as index:
        binding = index.get(binding_id)
        if binding is None:
            return [], "selected cortical field has no current registered-owner inspection binding"
        if (binding.get("kind") != "inspection" or binding.get("role") != role
                or binding.get("moduleId") != tensor_key or binding.get("tensorKey") != tensor_key):
            raise ValueError("selected cortical field differs from its actual saved registry binding")
        owner = index.get(binding["ownerId"])
        if owner is None or owner.get("kind") != "owner":
            raise ValueError("selected cortical field has no actual registered owner")
        anchors, known = [binding, owner], {binding["id"], owner["id"]}
        limit = max(1, min(256, page_size))
        rows = index.database.execute("SELECT id,shard,ordinal FROM records WHERE kind='membership' AND owner_id=? ORDER BY id LIMIT ?",
                                      (owner["id"], limit + 1))
        more = False
        for number, location in enumerate(rows):
            check_cancel(cancelled)
            if number == limit:
                more = True
                break
            membership = index.row(location)
            if membership.get("kind") != "membership" or membership.get("ownerId") != owner["id"]:
                raise ValueError("selected owner membership differs from the actual catalog")
            anchors.append(membership); known.add(membership["id"])
            region_id = membership["regionId"]
            while region_id and region_id not in known:
                region = index.get(region_id)
                if region is None or region.get("kind") != "region":
                    raise ValueError("selected owner membership has no actual region endpoint")
                anchors.append(region); known.add(region_id)
                region_id = region.get("parentRegionId")
        return anchors, "additional registered memberships remain available in paged owner details" if more else None


def query_whole_brain(engine, brain_id, query=None, *, admit, cancelled=None, cortex_query=None, substrate_query=None, _sparse_anchor_factories=None):
    """Loaded or cold read; never _get/Load. Root owns RPC/storage paths."""
    engine = Path(engine)
    metadata, metadata_sha = read_owned_json(engine, "brain.json", admit=admit, cancelled=cancelled)
    if metadata.get("brain_id") != brain_id:
        raise ValueError("whole-brain source belongs to another identity")
    from .whole_brain_catalog import binding_from_metadata
    binding = binding_from_metadata(metadata)
    pointer = metadata.get("whole_brain_catalog")
    if pointer is not None:
        pointer = validate_catalog_pointer(pointer, metadata)
    snapshot = {**binding, "metadataSha256": metadata_sha, "catalogSha256": digest(pointer) if pointer else None}
    raw = dict(query or {})
    if set(raw) - {"entity", "cursor", "pageSize", "regionId", "ownerId", "catalogFacet", "cortex", "substrate"}:
        raise ValueError("whole-brain query contains unsupported source authority")
    entity, size = raw.get("entity", "catalog"), raw.get("pageSize", 64)
    if entity not in ("catalog", "cortex", "substrate") or type(size) is not int or not 1 <= size <= 256:
        raise ValueError("invalid whole-brain source facet/page size")
    region_id, owner_id = raw.get("regionId", ""), raw.get("ownerId", "")
    if not isinstance(region_id, str) or not isinstance(owner_id, str):
        raise ValueError("invalid whole-brain region identity")
    selected = _find_region(engine, pointer, region_id, admit, cancelled) if region_id and pointer else None
    if region_id and selected is None:
        raise ValueError("historical source has no registry for regional selection")
    region_path = selected.get("nativeRegionPath", []) if selected else []
    scope = {"id": region_id if region_path and entity == "substrate" else "root", "sourceBrainId": selected["lineage"]["parentId"] if region_path and entity == "substrate" else brain_id,
             "nativeRegionPath": region_path}
    unavailable = ["semantic-neuron/router-channel/decoder-axis mapping is not recorded"]
    if pointer is None:
        unavailable.append("historical checkpoint has no registered-owner catalog; no registry was fabricated or loaded")
    response = {"format": "omni-whole-brain-inspection", "version": 1, "brainId": brain_id, "snapshot": snapshot,
                "entity": entity, "scope": scope, "catalog": {"available": pointer is not None,
                "reason": "actual registered owner catalog" if pointer else "missing-historical-owner-registry",
                "records": pointer["records"] if pointer else 0, "shards": pointer["shards"] if pointer else 0},
                "records": [], "anchors": [], "sparseAliases": [], "cortex": None, "substrate": None, "hasMore": False, "nextCursor": None,
                "evidence": {"topology": "registered-ownership-and-explicit-regional-bridges", "semanticAxisMapping": False,
                "tokenNeuronEquivalence": False, "liveActivityJoined": False, "snapshotConsistent": True, "unavailable": unavailable}}
    request_binding = digest({k: v for k, v in raw.items() if k != "cursor"})
    cursor = _cursor(raw["cursor"]) if raw.get("cursor") else None
    if cursor and (cursor.get("snapshot") != snapshot or cursor.get("requestBinding") != request_binding):
        raise ValueError("whole-brain committed generation/query changed; refresh instead of joining snapshots")

    def assert_current():
        check_cancel(cancelled)
        _, current_sha = read_owned_json(engine, "brain.json", admit=admit, cancelled=cancelled)
        if current_sha != metadata_sha:
            raise ValueError("whole-brain committed source changed during inspection")

    selected_owner = None
    if entity == "catalog" and pointer:
        from .whole_brain_catalog_index import catalog_index
        facet = raw.get("catalogFacet", "owner" if owner_id else "members" if region_id else "overview")
        if selected and selected.get("regionKind") == "whole-brain" and facet == "members": facet = "all"
        after = tuple(cursor["after"]) if cursor else (-1, "")
        if len(after) != 2 or type(after[0]) is not int or not isinstance(after[1], str):
            raise ValueError("invalid whole-brain indexed cursor")
        with catalog_index(engine, pointer["index"], admit=admit, cancelled=cancelled) as index:
            if owner_id:
                selected_owner = index.get(owner_id)
                if selected_owner is None or selected_owner.get("kind") != "owner": raise ValueError("selected registered owner is absent")
            response["records"], response["anchors"], after, more = index.page(facet, region_id=region_id, owner_id=owner_id, after=after, count=size)
        if more:
            response["hasMore"] = True
            response["nextCursor"] = _encode({"snapshot": snapshot, "requestBinding": request_binding, "after": after})
    elif entity in ("cortex", "substrate"):
        delegated = dict(raw.get(entity) or {})
        if "cursor" in delegated:
            raise ValueError("whole-brain traversal owns the underlying source cursor")
        delegated["pageSize"] = size
        if cursor:
            delegated["cursor"] = cursor["sourceCursor"]
        assert_current()
        metadata_held = _stream_json_size(metadata) * 8
        source_admit = lambda size, label: admit(metadata_held + size, label)
        if entity == "cortex":
            if region_path:
                unavailable.append("regional cortical owners use the joint root tensor inventory; exact owner inspection bindings select their fields")
                scope["nativeRegionPath"] = []
            if cortex_query is None:
                from .cortical_inspection import query_committed_cortex
                cortex_query = query_committed_cortex
            source = cortex_query(engine, brain_id, delegated)
            anchors, reason = _selected_cortex_owner_anchors(engine, pointer, source.get("selectedModule"),
                admit=admit, cancelled=cancelled, page_size=size)
            response["anchors"].extend(anchors)
            if reason:
                unavailable.append(reason)
        else:
            if substrate_query is None:
                from .substrate_inspection import query_persisted_substrate
                substrate_query = query_persisted_substrate
            if region_path:
                descriptor = _regional_source(engine, metadata, selected, assert_current)
                with inspection_resources(source_admit, cancelled):
                    source = substrate_query(descriptor.engine_directory, descriptor.brain_id, delegated, _committed_source=descriptor)
                    descriptor.resolve(descriptor.engine_directory, descriptor.brain_id)
                    source, response["sparseAliases"] = _actual_sparse_endpoints(source, descriptor.engine_directory, descriptor.brain_id,
                        descriptor=descriptor, admit=source_admit, cancelled=cancelled, factories=_sparse_anchor_factories)
            else:
                with inspection_resources(source_admit, cancelled):
                    source = substrate_query(engine, brain_id, delegated)
                    source, response["sparseAliases"] = _actual_sparse_endpoints(source, engine, brain_id,
                        admit=source_admit, cancelled=cancelled, factories=_sparse_anchor_factories)
        response[entity] = source
        response["hasMore"] = source["hasMore"]
        if source["hasMore"]:
            response["nextCursor"] = _encode({"snapshot": snapshot, "requestBinding": request_binding, "sourceCursor": source["nextCursor"]})
    # The main-process validator must be able to bind a delegated owner/scope
    # to the exact immutable catalog record on every page.  These are endpoint
    # proofs, not extra neural records and do not consume the requested page.
    known = {record.get("id") for record in response["records"] + response["anchors"] if isinstance(record, dict)}
    for anchor in (selected, selected_owner):
        if anchor is not None and anchor.get("id") not in known:
            response["anchors"].append(anchor)
            known.add(anchor.get("id"))
    assert_current()
    return response
