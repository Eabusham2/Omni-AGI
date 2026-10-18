"""Verified, bounded assembly of complete parents into connected native regions.

The host first seals and snapshots parents under its multi-instance write
lease. This module only reads those private snapshots. It never opens a live
parent brain or writes its source directory. New and loaded regional state is
registered under the normal native decoder/pager/checkpoint collectors.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from .bounded_tensor_io import BoundedTensorFile, load_module_bounded
from .committed_paged_cache import (
    finish_verified_index_from_loaded_vectors, prepare_committed_paged_cache,
)
from .config import OmniConfig
from .connected_regions import (
    ConnectedModalityHub, ConnectedOmniDecoder, NativeRegion, RegionIdentity,
)
from .persistence import atomic_write_json, read_json, tensor_checksum, snapshot_files
from .vsa import NeuralSubstrate
from .spiking import STDPSynapses


FORMAT = "omni-connected-native-regions"
FORMAT_VERSION = 1
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}\Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _file_hash(path: Path, *, cancelled=None, reserve=None) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("a merged checkpoint must be a regular private snapshot")
    chunk = 1024 * 1024
    if reserve is not None:
        reserve(min(chunk, path.stat().st_size), "parent checksum buffer")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if cancelled is not None and cancelled():
                raise InterruptedError("full merge stopped at a checksum boundary")
            value = handle.read(chunk)
            if not value:
                break
            digest.update(value)
    return digest.hexdigest()


def _inside(root: Path, relative: str) -> Path:
    parts = relative.split("/")
    if any(not part or part in {".", ".."} or "\\" in part for part in parts):
        raise ValueError("connected region snapshot path is unsafe")
    path = root.joinpath(*parts)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("connected region snapshots cannot follow symbolic links")
    if root.resolve() not in path.resolve().parents:
        raise ValueError("connected region snapshot escaped the new instance")
    return path


def resolve_connected_input(source_engine: Path, declaration: Mapping[str, Any], *, verify=True) -> Path:
    """Resolve an unchanged descriptor in a live instance or native snapshot.

    A snapshot keeps dependencies below its own engine directory. This is a
    location alias, not a changed descriptor or a newly trusted neural state:
    the exact original metadata/core/plasticity hashes are checked afterward.
    """
    source_engine = Path(source_engine).resolve()
    logical = declaration.get("enginePath")
    if not isinstance(logical, str):
        raise ValueError("connected input path is invalid")
    candidates = [
        _inside(source_engine / "connected-inputs", logical),
        _inside(source_engine.parent, logical),
    ]
    # An existing immutable origin may predate self-contained snapshots. Its
    # live sibling can supply the same exact dependency hashes for copying.
    sibling = source_engine.parent / "brain.json"
    if sibling.is_file() and not sibling.is_symlink() and (source_engine / "brain.json").is_file():
        current = read_json(sibling)
        own = read_json(source_engine / "brain.json")
        if current.get("connected_regions") == own.get("connected_regions"):
            candidates.extend((
                _inside(source_engine.parent / "connected-inputs", logical),
                _inside(source_engine.parent.parent, logical),
            ))
    for candidate in candidates:
        if (candidate / "brain.json").is_file():
            if verify:
                for filename, field in (("brain.json", "metadataSha256"), ("core.safetensors", "coreSha256"), ("plasticity.safetensors", "plasticitySha256")):
                    if _file_hash(candidate / filename) != declaration.get(field):
                        raise ValueError("connected input alias changed original bytes")
            return candidate
    raise FileNotFoundError("connected native snapshot is missing a declared parent input")


def _connected_input_sources(source_engine: Path, *, verify=True):
    metadata = read_json(Path(source_engine) / "brain.json")
    raw = metadata.get("connected_regions")
    if raw is None:
        return
    descriptor = validate_connected_descriptor(raw)
    for declaration in descriptor["regions"]:
        yield declaration, resolve_connected_input(source_engine, declaration, verify=verify)


def connected_input_snapshot_required_bytes(source_engine: Path) -> int:
    from .persistence import snapshot_required_bytes
    total = 0
    for _declaration, parent in _connected_input_sources(source_engine, verify=False):
        total += snapshot_required_bytes(parent, include_packed_ternary=True)
        if (parent / "origin" / "brain.json").is_file():
            total += snapshot_required_bytes(parent / "origin", include_packed_ternary=True)
    return total


def copy_connected_inputs(source_engine: Path, destination_engine: Path, *, resource_policy=None):
    """Copy a self-contained immutable native input closure inside a snapshot."""
    from .persistence import snapshot_files
    from .ternary_packing import verify_ternary_shards
    def copy_packed(source, destination):
        if not (source / "manifest.json").is_file():
            return
        verify_ternary_shards(source, retain_names=())
        for path in source.rglob("*"):
            if path.is_symlink():
                raise ValueError("native packed dependency contains a symbolic link")
            if not path.is_file():
                continue
            target = _inside(destination, path.relative_to(source).as_posix())
            if resource_policy is not None:
                resource_policy.require_disk(path.stat().st_size, "native connected dependency")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _file_hash(target) != _file_hash(path):
                    raise ValueError("native packed snapshot dependency conflicts")
            else:
                shutil.copy2(path, target)
    for declaration, parent in _connected_input_sources(source_engine):
        target = _inside(Path(destination_engine) / "connected-inputs", declaration["enginePath"])
        if target.exists():
            for filename, field in (("brain.json", "metadataSha256"), ("core.safetensors", "coreSha256"), ("plasticity.safetensors", "plasticitySha256")):
                if _file_hash(target / filename) != declaration[field]:
                    raise ValueError("connected snapshot dependency conflicts")
        else:
            snapshot_files(parent, target, resource_policy=resource_policy)
        copy_packed(parent / "packed-ternary", target / "packed-ternary")
        origin = parent / "origin"
        if (origin / "brain.json").is_file():
            if not (target / "origin").exists():
                snapshot_files(origin, target / "origin", resource_policy=resource_policy)
            copy_packed(origin / "packed-ternary", target / "origin" / "packed-ternary")


@dataclass(frozen=True)
class ParentRegionSnapshot:
    parent_id: str
    region_id: str
    engine_relative_path: str
    generation_sha256: str
    neural_sha256: str
    config: Mapping[str, Any]
    core_sha256: str
    plasticity_sha256: str
    metadata_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"parentId": self.parent_id, "regionId": self.region_id,
                "enginePath": self.engine_relative_path,
                "generationSha256": self.generation_sha256,
                "neuralSha256": self.neural_sha256, "config": dict(self.config),
                "coreSha256": self.core_sha256,
                "plasticitySha256": self.plasticity_sha256,
                "metadataSha256": self.metadata_sha256}


def inspect_parent_snapshot(
    instance_root: Path, *, parent_id: str, engine_relative_path: str,
    generation_sha256: str, cancelled=None, reserve=None,
) -> ParentRegionSnapshot:
    """Bind a parent to exact saved tensors, graph generation and controls."""
    if not _ID.fullmatch(parent_id) or not _HASH.fullmatch(generation_sha256):
        raise ValueError("parent identity or generation digest is invalid")
    engine = _inside(Path(instance_root), engine_relative_path)
    metadata_path = engine / "brain.json"
    metadata = read_json(metadata_path)
    config = metadata.get("config")
    if not isinstance(config, Mapping) or config.get("origin_kind") != "ground-up" or any(
        name in config for name in ("foundation_model_id", "foundationModelId")
    ) or metadata.get("neural_sequence_memory") is not None:
        raise ValueError("full merge accepts complete native OmniCortex state only")
    if metadata.get("brain_id") != parent_id:
        raise ValueError("parent snapshot identity does not match its recorded provenance")
    OmniConfig.from_dict(dict(config)).validate()
    substrate = metadata.get("substrate")
    persistence = substrate.get("persistence") if isinstance(substrate, Mapping) else None
    if not isinstance(persistence, Mapping) or persistence.get("formatVersion") != 3:
        raise ValueError("full merge requires the authoritative packed neural graph")
    core = engine / "core.safetensors"
    plastic = engine / "plasticity.safetensors"
    core_reader, plastic_reader = BoundedTensorFile(core), BoundedTensorFile(plastic)
    if any(not name.startswith(("decoder.", "memory_bridge.", "idea_adapter.", "liquid.", "modalities.")) for name in core_reader.specs):
        raise ValueError("parent checkpoint includes a foreign core owner")
    if any(name.startswith("sequence_memory.") or name == "router.synapses.weights" for name in plastic_reader.specs):
        raise ValueError("parent checkpoint includes obsolete or floating learned memory")
    core_hash = _file_hash(core, cancelled=cancelled, reserve=reserve)
    plastic_hash = _file_hash(plastic, cancelled=cancelled, reserve=reserve)
    metadata_hash = _file_hash(metadata_path, cancelled=cancelled, reserve=reserve)
    # Common ancestry is insufficient proof of equality: descendants may
    # have changed. Only exact neural bytes, graph pointer, semantic config,
    # and transient control state can share one current region.
    semantic_config = {k: v for k, v in config.items() if k != "name"}
    runtime_identity = {name: metadata.get(name) for name in (
        "mutable_state", "paged_working_memory", "workspace_items",
        "memory_lifecycle", "recent_token_context", "recent_token_activity",
        "current_context", "attention_overlay", "fresh_attention_boundary",
        "pending_chat_slow_learning", "ingestion_checkpoints",
        "ingestion_joint_generation", "connected_regions", "connected_graph_states",
        "connected_layout", "connected_runtime_adoption", "connected_pending_ingestions",
    )}
    from .connected_execution import suspended_execution_identity, validated_connected_executions
    execution = suspended_execution_identity(engine, parent_id)
    if execution is not None:
        runtime_identity["native_execution"] = execution
    inherited = validated_connected_executions(metadata.get("connected_executions"))
    if inherited["entries"]:
        runtime_identity["connected_executions"] = inherited
    neural_hash = hashlib.sha256(_canonical({"core": core_hash,
        "plasticity": plastic_hash, "substrate": substrate,
        "config": semantic_config, "experts": metadata.get("expert_count", 0),
        "runtimeState": runtime_identity})).hexdigest()
    return ParentRegionSnapshot(parent_id, "r" + neural_hash[:32], engine_relative_path,
        generation_sha256, neural_hash, dict(config), core_hash, plastic_hash, metadata_hash)


def connected_descriptor(parents: Sequence[ParentRegionSnapshot]) -> dict[str, Any]:
    if len(parents) < 2 or len({p.parent_id for p in parents}) != len(parents):
        raise ValueError("full merge requires at least two distinct parents")
    by_neural: dict[str, ParentRegionSnapshot] = {}
    aliases = []
    for parent in parents:
        previous = by_neural.setdefault(parent.neural_sha256, parent)
        aliases.append({"parentId": parent.parent_id, "regionId": previous.region_id,
                        "generationSha256": parent.generation_sha256})
    regions = [parent.to_dict() for parent in by_neural.values()]
    # Identical parents remain distinct historical identities but have one
    # exact neural owner, not duplicate learned weights or provenance loss.
    body = {"format": FORMAT, "formatVersion": FORMAT_VERSION,
            "regions": regions, "parents": aliases,
            "deduplication": "identical-complete-neural-state-only"}
    return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_connected_descriptor(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or value.get("format") != FORMAT or value.get("formatVersion") != FORMAT_VERSION:
        raise ValueError("unsupported connected-region descriptor")
    clean = dict(value)
    claimed = clean.pop("sha256", None)
    if claimed != hashlib.sha256(_canonical(clean)).hexdigest():
        raise ValueError("connected-region descriptor checksum is invalid")
    regions, parents = clean.get("regions"), clean.get("parents")
    if not isinstance(regions, list) or not regions or not isinstance(parents, list) or len(parents) < 2:
        raise ValueError("connected-region descriptor has no complete parent set")
    region_ids = set()
    for region in regions:
        if not isinstance(region, Mapping) or not _ID.fullmatch(str(region.get("regionId", ""))) or not _ID.fullmatch(str(region.get("parentId", ""))):
            raise ValueError("connected region identity is invalid")
        if region["regionId"] in region_ids:
            raise ValueError("connected region identity is duplicated")
        region_ids.add(region["regionId"])
        for name in ("generationSha256", "neuralSha256", "coreSha256", "plasticitySha256", "metadataSha256"):
            if not _HASH.fullmatch(str(region.get(name, ""))):
                raise ValueError("connected region digest is invalid")
        OmniConfig.from_dict(dict(region.get("config", {}))).validate()
        if not isinstance(region.get("enginePath"), str):
            raise ValueError("connected region engine path is invalid")
    parent_ids = set()
    for parent in parents:
        if not isinstance(parent, Mapping) or not _ID.fullmatch(str(parent.get("parentId", ""))) or parent["parentId"] in parent_ids or parent.get("regionId") not in region_ids or not _HASH.fullmatch(str(parent.get("generationSha256", ""))):
            raise ValueError("connected parent provenance is invalid")
        parent_ids.add(parent["parentId"])
    return {**clean, "sha256": claimed}


def merged_config(descriptor: Mapping[str, Any], *, settings: OmniConfig) -> OmniConfig:
    value = validate_connected_descriptor(descriptor)
    configs = [OmniConfig.from_dict(dict(r["config"])) for r in value["regions"]]
    if len({c.vocab_size for c in configs}) != 1:
        raise ValueError("parent byte token vocabularies are incompatible")
    width = sum(c.d_model for c in configs)
    # The root descriptor records the connected graph separately; a plain
    # single-stack architecture manifest must not pretend to count its bytes.
    return replace(settings, d_model=width, idea_dim=width, vocab_size=configs[0].vocab_size,
        n_heads=1 if width % 2 == 0 else settings.n_heads,
        n_layers=max(c.n_layers for c in configs), d_ff=sum(c.d_ff for c in configs),
        vsa_dim=sum(c.vsa_dim for c in configs), router_neurons=sum(c.router_neurons for c in configs),
        native_architecture=None, image_size=max(c.image_size for c in configs),
        audio_samples=max(c.audio_samples for c in configs), video_frames=max(c.video_frames for c in configs))


@dataclass
class LoadedConnectedRegions:
    descriptor: dict[str, Any]
    decoder: ConnectedOmniDecoder
    modalities: ConnectedModalityHub
    memories: dict[str, NeuralSubstrate]
    snapshot_metadata: dict[str, dict[str, Any]]
    # Storage objects own bounded maps/SQLite state. Root close must call this
    # after releasing module state, under the same worker lifecycle lease.
    index_owners: list[Any]
    graph_sources: dict[str, Path]
    attention_epoch: int = 0
    nested: Optional[dict[str, Any]] = None

    def graph_pointers(self) -> dict[str, Any]:
        return {key: memory.persistence_manifest for key, memory in self.memories.items()}

    def iter_memories(self, prefix=""):
        """Authoritative memory owners, including arbitrarily nested merges."""
        for region_id, memory in self.memories.items():
            key = prefix + region_id
            yield key, memory
            nested = (self.nested or {}).get(region_id)
            if nested is not None:
                yield from nested.iter_memories(key + "/")

    def pointer_snapshot(self):
        return {"regions": copy.deepcopy(self.graph_pointers()),
                "nested": {key: value.pointer_snapshot() for key, value in (self.nested or {}).items()}}

    def restore_pointers(self, snapshot):
        if set(snapshot["regions"]) != set(self.memories) or set(snapshot["nested"]) != set(self.nested or {}):
            raise ValueError("connected graph rollback lost a region")
        for region_id, pointer in snapshot["regions"].items():
            self.memories[region_id].persistence_manifest = copy.deepcopy(pointer)
        for region_id, nested in (self.nested or {}).items():
            nested.restore_pointers(snapshot["nested"][region_id])

    def stage_graphs(self, instance_root: Path, *, reserve=None, reuse=False) -> dict[str, Any]:
        records = {}
        for region_id, memory in self.memories.items():
            relative = "engine/connected-state/" + region_id
            engine = _inside(Path(instance_root), relative)
            if not (engine / "substrate" / "manifest.json").is_file() and memory.persistence_manifest:
                source_store = self.graph_sources[region_id] / "substrate"
                pointer = dict(memory.persistence_manifest)
                NeuralSubstrate._generation_references(source_store, pointer)
                # Forward-index blobs are derived but referenced by the live
                # lazy view. Carry the complete private store, including those
                # index shards, without materializing its path list in RAM.
                paths = (path.relative_to(source_store).as_posix()
                         for path in source_store.rglob("*") if path.is_file())
                for logical in paths:
                    if logical == "manifest.json":
                        continue
                    source = _inside(source_store, logical)
                    destination = _inside(engine / "substrate", logical)
                    if reserve is not None:
                        reserve(source.stat().st_size, "connected immutable graph materialization")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        try:
                            # These are content-addressed immutable generation
                            # blobs, never mutable/live tensor destinations.
                            os.link(source, destination)
                        except OSError:
                            with source.open("rb") as incoming, destination.open("xb") as output:
                                shutil.copyfileobj(incoming, output, 1024 * 1024)
                                output.flush()
                                os.fsync(output.fileno())
                    if _file_hash(source) != _file_hash(destination):
                        raise ValueError("connected graph materialization lost source bytes")
                atomic_write_json(engine / "substrate" / "manifest.json", pointer)
                # The validated byte-identical blobs now back the *same*
                # lazy overlay in its new private store. Preserve dirty rows,
                # revisions and locators; only the immutable source path moves.
                synapses = memory.synapses
                if hasattr(synapses, "root"):
                    if Path(synapses.root).resolve() != source_store.resolve():
                        raise ValueError("regional synapse source changed during materialization")
                    synapses.root = (engine / "substrate").resolve()
            if not reuse or not (engine / "substrate" / "manifest.json").is_file():
                memory.save_sharded(engine / "substrate", disk_reserve=reserve)
            self.graph_sources[region_id] = engine
            records[region_id] = {"enginePath": relative,
                "substrate": memory.metadata(include_records=False),
                "attentionOverlay": memory.attention_overlay_metadata()}
            nested = (self.nested or {}).get(region_id)
            if nested is not None:
                records[region_id]["nestedStateRoot"] = relative + "/nested"
                records[region_id]["nestedGraphStates"] = nested.stage_graphs(
                    engine / "nested", reserve=reserve, reuse=reuse)
        body = {"format": "omni-connected-graph-states", "formatVersion": 1,
                "descriptorSha256": self.descriptor["sha256"],
                "attentionEpoch": self.attention_epoch, "regions": records}
        return {**body, "sha256": hashlib.sha256(_canonical(body)).hexdigest()}

    def clear_attention(self, epoch: int) -> dict[str, Any]:
        result = {}
        for region in self.decoder.regions:
            region.liquid_state.zero_()
            region.router.reset_activity()
            result[region.identity.region_id] = self.memories[region.identity.region_id].clear_attention_activity(epoch)
            nested = (self.nested or {}).get(region.identity.region_id)
            if nested is not None:
                result[region.identity.region_id]["nestedRegions"] = nested.clear_attention(epoch)
        self.attention_epoch = epoch
        return result

    def close(self) -> None:
        for nested in (self.nested or {}).values():
            nested.close()
        seen = set()
        for owner in reversed(self.index_owners):
            if id(owner) in seen:
                continue
            seen.add(id(owner))
            close = getattr(owner, "close", None)
            if callable(close):
                close()


def load_connected_regions(instance_root, descriptor, config, **options):
    owned = []
    try:
        return _load_connected_regions_impl(instance_root, descriptor, config, _cleanup=owned, **options)
    except BaseException:
        seen = set()
        for value in reversed(owned):
            if id(value) in seen:
                continue
            seen.add(id(value))
            close = getattr(value, "close", None)
            if callable(close):
                close()
        raise


def _load_connected_regions_impl(
    instance_root: Path, descriptor: Mapping[str, Any], config: OmniConfig, *,
    core_pager=None, router_pager=None, resource_policy=None, growth_guard=None,
    cancelled: Optional[Callable[[], bool]] = None,
    graph_states: Optional[Mapping[str, Any]] = None,
    runtime_root: Optional[Path] = None, graph_root: Optional[Path] = None,
    repair_graph_mirrors: bool = True,
    restoring_checkpoint: bool = False,
    _cleanup=None,
) -> LoadedConnectedRegions:
    """Load every parent directly into registered final packed destinations.

    This is the merge-install path. Subsequent loads restore these same
    registered owners from the root's connected checkpoint, using
    ``load_connected_decoder_bounded`` below. Snapshot metadata remains
    available for the host's exact history/pending-work merge.
    """
    descriptor = validate_connected_descriptor(descriptor)
    active_graphs = {}
    epoch = 0
    if graph_states is not None:
        body = dict(graph_states)
        claimed = body.pop("sha256", None)
        if (claimed != hashlib.sha256(_canonical(body)).hexdigest()
                or body.get("format") != "omni-connected-graph-states" or body.get("formatVersion") != 1
                or body.get("descriptorSha256") != descriptor["sha256"]
                or not isinstance(body.get("regions"), Mapping)
                or set(body["regions"]) != {r["regionId"] for r in descriptor["regions"]}
                or type(body.get("attentionEpoch")) is not int or body["attentionEpoch"] < 0):
            raise ValueError("connected graph state is not bound to the full root checkpoint")
        active_graphs = body["regions"]
        epoch = body["attentionEpoch"]
    regions, memories, metadata, owners, graph_sources, nested_regions = [], {}, {}, [], {}, {}
    root = Path(instance_root).resolve()
    live_root = Path(runtime_root).resolve() if runtime_root is not None else root
    graph_base = Path(graph_root).resolve() if graph_root is not None else root
    for declaration in descriptor["regions"]:
        if cancelled is not None and cancelled():
            raise InterruptedError("full merge stopped before regional allocation")
        path = resolve_connected_input(root / "engine", declaration, verify=False)
        for filename, field in (("brain.json", "metadataSha256"), ("core.safetensors", "coreSha256"), ("plasticity.safetensors", "plasticitySha256")):
            if _file_hash(path / filename, cancelled=cancelled) != declaration[field]:
                raise ValueError("parent snapshot changed before connected assembly")
        saved = read_json(path / "brain.json")
        parent_config = OmniConfig.from_dict(dict(declaration["config"]))
        # Context capacity is a runtime setting; preserve learned geometry
        # while applying the newly reviewed common attention window.
        parent_config.max_seq_len = config.max_seq_len
        nested = None
        active = active_graphs.get(declaration["regionId"])
        if saved.get("connected_regions") is not None:
            nested_runtime = live_root / "engine" / "connected-state" / declaration["regionId"] / "nested"
            nested_graphs = saved.get("connected_graph_states")
            nested_graph_root = path.parent
            nested_repair = False
            if active is not None:
                expected_nested = "engine/connected-state/" + declaration["regionId"] + "/nested"
                if (not isinstance(active, Mapping) or active.get("nestedStateRoot") != expected_nested
                        or not isinstance(active.get("nestedGraphStates"), Mapping)):
                    raise ValueError("nested merged graph lost its mutable-state binding")
                nested_graph_root = _inside(graph_base, expected_nested)
                nested_graphs = active["nestedGraphStates"]
                nested_repair = repair_graph_mirrors
            nested = load_connected_regions(path.parent, saved["connected_regions"], parent_config,
                core_pager=core_pager, router_pager=router_pager, resource_policy=resource_policy,
                growth_guard=growth_guard, cancelled=cancelled, graph_states=nested_graphs,
                runtime_root=nested_runtime, graph_root=nested_graph_root,
                repair_graph_mirrors=nested_repair,
                # This parent's committed composite core, loaded below (or
                # by the final root loader), owns every child byte already.
                # Loading the child originals too would populate sparse
                # owners twice and needlessly read an entire model again.
                restoring_checkpoint=True)
            nested.decoder.restore_layout(saved["connected_layout"])
            nested_regions[declaration["regionId"]] = nested
            _cleanup.append(nested)
        with core_pager.construction(from_checkpoint=True) if core_pager is not None else nullcontext():
            with router_pager.construction(loading=True) if router_pager is not None else nullcontext():
                region = NativeRegion(RegionIdentity(declaration["regionId"], declaration["parentId"], declaration["generationSha256"]), parent_config, connected=nested)
                if nested is None:
                    for _ in range(int(saved.get("expert_count", 0))):
                        region.decoder.grow_expert()
        if not restoring_checkpoint:
            core = BoundedTensorFile(path / "core.safetensors")
            plastic = BoundedTensorFile(path / "plasticity.safetensors")
            for module, prefix in ((region.decoder, "decoder."), (region.memory_bridge, "memory_bridge."),
                    (region.idea_adapter, "idea_adapter."), (region.liquid, "liquid."), (region.modalities, "modalities.")):
                if module is region.decoder and nested is not None:
                    load_connected_decoder_bounded(module, core, prefix, cancelled=cancelled)
                else:
                    load_module_bounded(module, core, prefix, cancelled=cancelled)
            region.router.synapses.load_recurrent_state_bounded(plastic, "router.synapses.")
            load_module_bounded(region.router, plastic, "router.", excluded_prefixes=("synapses.",), cancelled=cancelled)
            if "state.liquid" in plastic.specs:
                plastic.copy_into("state.liquid", region.liquid_state, cancelled=cancelled)
        # recall_projection is new, separately admitted/initialized in the
        # regional constructor; parent memory encoders remain unchanged.
        memory_args = dict(growth_guard=growth_guard, lazy_synapses=True)
        graph_engine, graph_metadata = path, saved["substrate"]
        if active is not None:
            expected_path = "engine/connected-state/" + declaration["regionId"]
            if not isinstance(active, Mapping) or active.get("enginePath") != expected_path or not isinstance(active.get("substrate"), Mapping):
                raise ValueError("connected graph active path is invalid")
            graph_engine = _inside(graph_base, expected_path)
            graph_metadata = dict(active["substrate"])
            # These files are recovery mirrors, never independent authority.
            # The already committed root descriptor chooses the old/new
            # generation after a crash between graph staging and root save.
            if repair_graph_mirrors:
                atomic_write_json(graph_engine / "brain.json", {"substrate": graph_metadata})
                atomic_write_json(graph_engine / "substrate" / "manifest.json", dict(graph_metadata["persistence"]))
        if resource_policy is not None:
            with prepare_committed_paged_cache(graph_engine, live_root / "engine" / "state" / "connected-cache" / declaration["regionId"], resource_policy=resource_policy) as prepared:
                memory = NeuralSubstrate.load_sharded(graph_engine / "substrate", graph_metadata,
                    paged_vectors=prepared.vectors, paged_neurons=prepared.neurons,
                    defer_paged_assemblies=True, **memory_args)
                rebuilt = finish_verified_index_from_loaded_vectors(prepared, memory)
                owners.extend([prepared.vectors, prepared.neurons, rebuilt])
                _cleanup.extend([prepared.vectors, prepared.neurons, rebuilt])
        else:
            memory = NeuralSubstrate.load_sharded(graph_engine / "substrate", graph_metadata, **memory_args)
        overlay = (active.get("attentionOverlay") if active is not None else saved.get("attention_overlay"))
        regional_epoch = (int(overlay.get("epoch", 0)) if isinstance(overlay, Mapping)
                          else int((saved.get("fresh_attention_boundary") or {}).get("epoch", 0)))
        memory.configure_attention_overlay(regional_epoch, overlay)
        from .connected_recall_binding import ConnectedRecallBinding
        region.attach_recall(ConnectedRecallBinding(memory, config.working_memory_slots))
        region.attach_memory_learning(lambda cue, event_key, importance, source=memory:
            source.learn_vector(cue, fingerprint=event_key, kind="connected-experience",
                source="shared-neural-workspace", importance=importance))
        regions.append(region)
        memories[declaration["regionId"]] = memory
        metadata[declaration["regionId"]] = saved
        graph_sources[declaration["regionId"]] = graph_engine
    with core_pager.construction(from_checkpoint=False) if core_pager is not None else nullcontext():
        decoder = ConnectedOmniDecoder(config, regions)
        modalities = ConnectedModalityHub(decoder)
    if core_pager is not None:
        core_pager.bind_names((("decoder.", decoder), ("modalities.", modalities)))
    return LoadedConnectedRegions(descriptor, decoder, modalities, memories, metadata, owners, graph_sources, epoch, nested_regions)


def load_connected_decoder_bounded(decoder: ConnectedOmniDecoder, reader: BoundedTensorFile,
                                   prefix: str = "decoder.", *, cancelled=None, overrides=None) -> None:
    """Restore nested sparse router pages before generic packed core owners."""
    exclusions = []
    for name, module in decoder.named_modules():
        if isinstance(module, STDPSynapses):
            relative = name + "."
            module.load_recurrent_state_bounded(reader, prefix + relative)
            exclusions.append(relative)
    load_module_bounded(decoder, reader, prefix, excluded_prefixes=tuple(exclusions),
                        cancelled=cancelled, overrides=overrides)


def connected_state_inventory(decoder: ConnectedOmniDecoder, modalities: ConnectedModalityHub) -> dict[str, Any]:
    """Actual registered owners, including every parent's auxiliary state."""
    tensors, seen = {}, set()
    for prefix, root in (("decoder.", decoder), ("modalities.", modalities)):
        for name, value in root.state_dict(keep_vars=True).items():
            identity = (value.device, value.data_ptr(), value.numel(), value.element_size())
            if value.numel() and identity in seen:
                raise ValueError("connected state registers duplicate storage owners")
            seen.add(identity)
            tensors[prefix + name] = value
    return {"regions": len(decoder.regions), "registeredTensors": len(tensors),
            "tensorBytes": sum(v.numel() * v.element_size() for v in tensors.values()),
            "floatingLearnedParameters": sum(p.numel() for p in decoder.parameters()) + sum(p.numel() for p in modalities.parameters()),
            "sha256": tensor_checksum(tensors.values()),
            "regionIds": [r.identity.region_id for r in decoder.regions]}


def _parents_from_request(root: Path, declarations: Sequence[Mapping[str, Any]], *, cancelled=None):
    if (not isinstance(declarations, list) or len(declarations) < 2
            or any(not isinstance(row, Mapping) for row in declarations)):
        raise ValueError("full merge needs at least two declared private snapshots")
    result = []
    for row in declarations:
        result.append(inspect_parent_snapshot(root, parent_id=str(row.get("brainId", "")),
            engine_relative_path=str(row.get("enginePath", "")),
            generation_sha256=str(row.get("generationSha256", "")), cancelled=cancelled))
    if len({parent.parent_id for parent in result}) != len(result):
        raise ValueError("full merge parent identity is repeated")
    return result


def _merge_origin(brain, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot the actual merged birth state, never a fictional curriculum."""
    origin = brain.engine_path / "origin"
    if origin.exists():
        return verify_connected_origin(origin, brain_id=brain.brain_id,
                                       descriptor_sha256=receipt["descriptorSha256"])
    temporary = brain.engine_path / ".connected-origin.next"
    if temporary.exists():
        # Only this recognizable, incomplete merge-owned staging directory is
        # discarded. The host already holds the destination owner lease.
        if temporary.is_symlink() or not temporary.is_dir():
            raise ValueError("connected origin staging path is unsafe")
        shutil.rmtree(temporary)
    try:
        snapshot_files(brain.engine_path, temporary, resource_policy=brain.resource_policy)
        # snapshot_files includes graph closure and a WAL-consistent runtime
        # backup; never overwrite it with a raw copy of the live SQLite main.
        packed = brain.engine_path / "packed-ternary"
        if not packed.is_dir():
            raise ValueError("merged origin is missing native packed inference state")
        shutil.copytree(packed, temporary / "packed-ternary")
        marker = {"format": "omni-connected-native-origin", "formatVersion": 1,
            "brainId": brain.brain_id, "descriptorSha256": receipt["descriptorSha256"],
            "runtimeAdoptionSha256": receipt["runtimeAdoption"]["sha256"],
            "parameterChecksum": brain.parameter_checksum(),
            "coreSha256": _file_hash(temporary / "core.safetensors"),
            "plasticitySha256": _file_hash(temporary / "plasticity.safetensors"),
            "metadataSha256": _file_hash(temporary / "brain.json"),
            "packedManifestSha256": _file_hash(temporary / "packed-ternary" / "manifest.json"),
            "initialCurriculumRerun": False,
            "source": "complete-native-parent-state-and-new-connection-owners"}
        marker["sha256"] = hashlib.sha256(_canonical(marker)).hexdigest()
        atomic_write_json(temporary / "connected-origin.json", marker)
        os.replace(temporary, origin)
        return marker
    except BaseException:
        # Leave an interrupted staging directory recognizable and restartable;
        # never replace a previously published immutable origin.
        raise


def verify_connected_origin(origin: Path, *, brain_id: str, descriptor_sha256: str,
                            allow_ancestor_owner: bool = False) -> dict[str, Any]:
    """Authenticate a merged origin without inventing initial-training proof."""
    from .connected_runtime import validate_runtime_adoption
    from .ternary_packing import verify_ternary_shards
    origin = Path(origin)
    marker = read_json(origin / "connected-origin.json")
    body = dict(marker)
    claimed = body.pop("sha256", None)
    if (body.get("format") != "omni-connected-native-origin" or body.get("formatVersion") != 1
            or (not allow_ancestor_owner and body.get("brainId") != brain_id)
            or body.get("descriptorSha256") != descriptor_sha256
            or body.get("initialCurriculumRerun") is not False
            or hashlib.sha256(_canonical(body)).hexdigest() != claimed):
        raise ValueError("connected origin declaration is invalid")
    for relative, field in (("brain.json", "metadataSha256"), ("core.safetensors", "coreSha256"),
                            ("plasticity.safetensors", "plasticitySha256"), ("packed-ternary/manifest.json", "packedManifestSha256")):
        if _file_hash(_inside(origin, relative)) != body.get(field):
            raise ValueError("connected origin bytes changed: " + relative)
    metadata = read_json(origin / "brain.json")
    descriptor = validate_connected_descriptor(metadata.get("connected_regions"))
    adoption = validate_runtime_adoption(metadata.get("connected_runtime_adoption"))
    if (metadata.get("brain_id") != body.get("brainId")
            or not _ID.fullmatch(str(metadata.get("brain_id", "")))
            or descriptor["sha256"] != descriptor_sha256
            or not isinstance(adoption, Mapping) or adoption["sha256"] != body["runtimeAdoptionSha256"]):
        raise ValueError("connected origin lost its neural/runtime provenance")
    packed = verify_ternary_shards(origin / "packed-ternary", retain_names=()).manifest
    if (packed.get("metadata", {}).get("parameterChecksum") != body["parameterChecksum"]
            or packed.get("metadata", {}).get("brainId") != body["brainId"]):
        raise ValueError("connected origin packed state is not bound to its neural state")
    return marker


def connected_parent_action_curriculum(instance_root: Path, descriptor, receipt_validator):
    """Resolve the actual inherited curriculum, never a new birth receipt.

    Only exact supported, authenticated parent origins contribute. A fork may
    retain its original owner's immutable origin; its current parent identity
    is still bound separately by the connected descriptor.
    """
    from .ground_up import (GROUND_UP_ACTION_EXAMPLES, resolve_ground_up_curriculum_manifest,
                            validate_ground_up_v3_training_manifest)
    from .ternary_packing import verify_ternary_shards
    try:
        inherited = None
        origins = []
        for region in validate_connected_descriptor(descriptor)["regions"]:
            engine = resolve_connected_input(Path(instance_root) / "engine", region)
            metadata = read_json(engine / "brain.json")
            nested = metadata.get("connected_regions")
            if nested is not None:
                verify_connected_origin(engine / "origin", brain_id=region["parentId"], descriptor_sha256=nested["sha256"],
                                        allow_ancestor_owner=True)
                child = connected_parent_action_curriculum(engine.parent, nested, receipt_validator)
                if child is None:
                    return None
                resolved = child["curriculum"]
                if inherited is not None and inherited != resolved:
                    return None
                inherited = resolved
                origins.extend({**entry, "regionPath": region["regionId"] + "/" + entry["regionPath"]}
                               for entry in child["origins"])
                continue
            origin = engine / "origin"
            saved = read_json(origin / "brain.json")
            manifest = saved.get("ground_up_training_manifest")
            resolved = resolve_ground_up_curriculum_manifest(manifest)
            if (not _ID.fullmatch(str(saved.get("brain_id", ""))) or not isinstance(manifest, Mapping)
                    or resolved is None or (saved.get("config") or {}).get("origin_kind") != "ground-up"
                    or manifest.get("baseFrozen") is not False
                    or not all(manifest.get(key) == value for key, value in resolved.items())
                    or (manifest.get("toolCurriculum") or {}).get("perActionCoverage") is not True
                    or (manifest.get("actionTraining") or {}).get("examples") != len(GROUND_UP_ACTION_EXAMPLES)
                    or not receipt_validator(manifest.get("publicCapabilityReadiness"))):
                return None
            if any(not (origin / path).is_file() or (origin / path).is_symlink() for path in (
                    "core.safetensors", "plasticity.safetensors", "substrate/manifest.json", "state/manifest.json")):
                return None
            packed = verify_ternary_shards(origin / "packed-ternary", retain_names=()).manifest
            if int(resolved.get("formatVersion", 0)) >= 3:
                validate_ground_up_v3_training_manifest(manifest)
                meta = packed.get("metadata", {})
                receipt = manifest.get("trainingReceipt", {})
                if (meta.get("groundUpCurriculumSha256") != resolved["sha256"]
                        or meta.get("groundUpTrainingManifestSha256") != manifest.get("contentSha256")
                        or meta.get("groundUpTrainingReceiptSha256") != receipt.get("contentSha256")
                        or meta.get("parameterChecksum") != receipt.get("parameterChecksumAfter")
                        or meta.get("originKind") != "ground-up" or meta.get("baseFrozen") is not False
                        or meta.get("brainId") != saved["brain_id"]):
                    return None
            if inherited is not None and inherited != resolved:
                return None
            inherited = resolved
            origins.append({"regionPath": region["regionId"], "parentId": region["parentId"],
                "originOwnerId": saved["brain_id"], "trainingManifestSha256": manifest["contentSha256"],
                "trainingReceiptSha256": manifest["trainingReceipt"]["contentSha256"]})
        return {"curriculum": inherited, "origins": origins} if inherited and origins else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def connected_parent_action_origins_verified(instance_root: Path, descriptor, receipt_validator) -> bool:
    return connected_parent_action_curriculum(instance_root, descriptor, receipt_validator) is not None


def connected_graph_snapshot_entries(source_engine: Path, graph_states, *, prefix=""):
    """Enumerate the exact committed graph closure relative to any engine dir.

    Inline authority mirrors are synthesized from the root commit, not copied
    from potentially newer orphaned recovery mirrors. No live parent is edited.
    """
    if not isinstance(graph_states, Mapping):
        raise ValueError("connected snapshot graph declaration is invalid")
    body = dict(graph_states)
    claimed = body.pop("sha256", None)
    if (body.get("format") != "omni-connected-graph-states" or body.get("formatVersion") != 1
            or claimed != hashlib.sha256(_canonical(body)).hexdigest()
            or not isinstance(body.get("regions"), Mapping)):
        raise ValueError("connected snapshot graph checksum is invalid")
    for region_id, record in graph_states["regions"].items():
        if (not isinstance(region_id, str) or not _ID.fullmatch(region_id)
                or not isinstance(record, Mapping)
                or record.get("enginePath") != "engine/connected-state/" + region_id
                or ("nestedGraphStates" in record and record.get("nestedStateRoot") != record["enginePath"] + "/nested")):
            raise ValueError("connected snapshot graph path is invalid")
        source = _inside(Path(source_engine), "connected-state/" + region_id)
        relative = prefix + "connected-state/" + region_id + "/"
        authority = _canonical({"substrate": record["substrate"]})
        yield relative + "brain.json", authority, hashlib.sha256(authority).hexdigest()
        pointer = record["substrate"]["persistence"]
        _generation, refs = NeuralSubstrate._generation_references(source / "substrate", pointer)
        for logical in (str(pointer["generationManifest"]), *sorted(refs)):
            incoming = _inside(source / "substrate", logical)
            expected = pointer["generationManifestSha256"] if logical == pointer["generationManifest"] else Path(logical).stem
            yield relative + "substrate/" + logical, incoming, expected
        authority = _canonical(pointer)
        yield relative + "substrate/manifest.json", authority, hashlib.sha256(authority).hexdigest()
        if "nestedGraphStates" in record:
            yield from connected_graph_snapshot_entries(source / "nested" / "engine", record["nestedGraphStates"],
                prefix=relative + "nested/engine/")


def copy_connected_graph_snapshot(source_root: Path, graph_states, destination_engine: Path, *, reserve=None, source_engine=None):
    """Preserve regional graph closure; immutable parent trees remain separate."""
    from .persistence import atomic_write_bytes
    base = Path(source_root) / "engine" if source_engine is None else Path(source_engine)
    for relative, incoming, expected in connected_graph_snapshot_entries(base, graph_states):
        outgoing = _inside(Path(destination_engine), relative)
        size = len(incoming) if isinstance(incoming, bytes) else incoming.stat().st_size
        if reserve is not None:
            reserve(size, "connected snapshot graph")
        outgoing.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(incoming, bytes):
            atomic_write_bytes(outgoing, incoming)
        elif not outgoing.exists():
            shutil.copy2(incoming, outgoing)
        if _file_hash(outgoing) != expected:
            raise ValueError("connected graph snapshot lost a committed blob")


def assemble_full_merge(root: Path, brain_id: str, parent_declarations,
                        settings: OmniConfig, *, cancelled=None, progress=None) -> dict[str, Any]:
    """Production assembly called only inside the host's unpublished target."""
    from .brain import AdaptiveBrain
    from .connected_runtime import adopt_connected_runtime
    root = Path(root).resolve()
    if not _ID.fullmatch(brain_id):
        raise ValueError("new merged identity is invalid")
    parents = _parents_from_request(root, parent_declarations, cancelled=cancelled)
    if brain_id in {parent.parent_id for parent in parents}:
        raise ValueError("a full merge cannot overwrite a parent identity")
    descriptor = connected_descriptor(parents)
    config = merged_config(descriptor, settings=settings)
    metadata_path = root / "engine" / "brain.json"
    if metadata_path.exists():
        saved = read_json(metadata_path)
        if (saved.get("brain_id") != brain_id or saved.get("connected_regions") != descriptor
                or saved.get("config") != config.to_dict()):
            raise ValueError("existing merge staging checkpoint has different parents or settings")
        brain = AdaptiveBrain.load(root, expected_brain_id=brain_id)
    else:
        # Track the object before construction so cancellation/validation in
        # a later region closes already-open pages and SQLite owners.
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        try:
            brain.__init__(brain_id, root, config, _connected_descriptor=descriptor,
                           _connected_cancelled=cancelled)
        except BaseException:
            brain.close()
            raise
    try:
        if progress is not None:
            progress("adopting", {"regions": len(descriptor["regions"]), "parents": len(parents)})
        if cancelled is not None and cancelled():
            raise InterruptedError("full merge stopped before runtime adoption")
        adoption = adopt_connected_runtime(brain, parents, cancelled=cancelled)
        brain.save()
        if cancelled is not None and cancelled():
            raise InterruptedError("full merge stopped after its committed assembly")
        packed = brain.export_packed_ternary()
        receipt = {"format": "omni-full-native-assembly", "formatVersion": 1,
            "brainId": brain_id, "descriptorSha256": descriptor["sha256"],
            "runtimeAdoption": adoption, "regionalState": connected_state_inventory(brain.decoder, brain.modalities),
            "parameterAccounting": brain.parameter_accounting(), "packedTernary": packed["summary"]}
        receipt["origin"] = _merge_origin(brain, receipt)
        atomic_write_json(root / "engine" / "connected-assembly.json", receipt)
        if progress is not None:
            progress("assembled", {"regions": len(descriptor["regions"]), "parents": len(parents), "counts": adoption["counts"]})
        return receipt
    finally:
        brain.close()


def verify_full_merge(root: Path, brain_id: str, *, cancelled=None) -> dict[str, Any]:
    """Reload actual owners and authenticated activity before host publication."""
    from .brain import AdaptiveBrain
    from .connected_runtime import validate_runtime_adoption, validate_pending_ingestions
    root = Path(root).resolve()
    receipt = read_json(root / "engine" / "connected-assembly.json")
    if receipt.get("format") != "omni-full-native-assembly" or receipt.get("brainId") != brain_id:
        raise ValueError("merged assembly receipt is missing or belongs to another identity")
    if cancelled is not None and cancelled():
        raise InterruptedError("full merge stopped before verification load")
    brain = AdaptiveBrain.load(root, expected_brain_id=brain_id)
    try:
        if brain._connected_regions is None or brain._connected_regions.descriptor["sha256"] != receipt["descriptorSha256"]:
            raise ValueError("merged engine is not its declared connected native graph")
        adoption = validate_runtime_adoption(brain.connected_runtime_adoption)
        if adoption != receipt["runtimeAdoption"]:
            raise ValueError("merged runtime adoption differs from its committed receipt")
        validate_pending_ingestions(brain.connected_pending_ingestions)
        inventory = connected_state_inventory(brain.decoder, brain.modalities)
        if inventory != receipt["regionalState"]:
            raise ValueError("merged connected region owners changed on restart")
        conversation = brain.conversation.integrity()
        replay = brain.replay.checkpoint()
        if conversation["totalEntries"] != adoption["counts"]["historyEntries"]:
            raise ValueError("merged conversation is incomplete")
        if cancelled is not None and cancelled():
            raise InterruptedError("full merge stopped after bounded state verification")
        origin = verify_connected_origin(root / "engine" / "origin", brain_id=brain_id,
                                         descriptor_sha256=receipt["descriptorSha256"])
        if origin != receipt["origin"] or origin["parameterChecksum"] != brain.parameter_checksum():
            raise ValueError("merged origin is not the actual assembled birth state")
        audit = brain.require_complete_packed_runtime()
        return {"brainId": brain_id, "verified": True, "descriptorSha256": receipt["descriptorSha256"],
            "regionalState": inventory, "conversation": conversation, "replay": replay,
            "runtimeAdoption": adoption, "origin": origin, "packedRuntime": audit,
            "pendingIngestions": copy_pending(brain.connected_pending_ingestions)}
    finally:
        brain.close()


def copy_pending(values):
    # JSON-compatible by its closed validator; no tensor or source payload.
    return json.loads(_canonical(values))
