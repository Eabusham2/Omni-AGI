"""Isolated native geometry preparation; never operate on the live parent.

Default factories construct native modules only when an explicitly requested
isolated candidate runs. Tests inject descriptor/module factories. Existing
history, substrate, settings and original durable activity payloads are not
rewritten. Geometry is not a function-preserving or quality claim.
"""

from __future__ import annotations

import copy
import hashlib
import math
import uuid
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import torch

from .architecture_migration import (
    assert_architecture_quiescent, copy_control_geometry, copy_packed_geometry,
    geometry_candidate_config, geometry_owner_plan, normalize_architecture_change,
    preserve_runtime_rng, reseal_native_descriptor,
)
from .bounded_tensor_io import BoundedTensorFile, TRANSFER_BYTES
from .native_architecture import validate_native_architecture


ROOTS = ("decoder", "memory_bridge", "idea_adapter", "liquid", "modalities", "router")
_GEOMETRY_ATTRIBUTES = {"in_features", "out_features", "num_embeddings", "embedding_dim", "dimensions",
    "hidden_dimensions", "hidden_width", "head_dim", "heads", "input_size", "hidden_size", "idea_dim",
    "d_model", "d_ff", "n_heads", "max_cached_tokens"}


def _admit(policy, byte_count, purpose):
    if policy is not None and policy.status(estimated_ram_bytes=max(0, int(byte_count))).get("memoryPressure"):
        raise RuntimeError(purpose + " paused at the live shared memory reserve")
    reserve = getattr(policy, "reserve_ram", None)
    return reserve(max(0, int(byte_count)), purpose) if callable(reserve) else nullcontext()


def _cancel(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("isolated geometry preparation cancelled; original candidate retained")


def _map_activity(value, width, policy):
    if value.ndim not in (1, 2) or not value.shape[-1]:
        raise ValueError("historical geometry activity must be a declared vector/matrix")
    if value.shape[-1] == width:
        return value
    shape = (*value.shape[:-1], width)
    size = math.prod(shape) * value.element_size()
    with _admit(policy, 2 * size, "historical geometry activity") as held:
        target = torch.empty(shape, dtype=value.dtype, device=value.device)
        copy_control_geometry(value, target)
        if held is not None: held.mark_allocated(size)
    return target


class GeometryReplayView(Sequence):
    """Adapt one activity row on demand, preserving its original durable bytes."""

    def __init__(self, source, width):
        self._geometry_source = getattr(source, "_geometry_source", source)
        self.width = int(width)

    @property
    def policy(self): return getattr(self._geometry_source, "policy", None)
    @policy.setter
    def policy(self, value): self._geometry_source.policy = value
    def __len__(self): return len(self._geometry_source)
    def __getattr__(self, name): return getattr(self._geometry_source, name)

    def _preflight(self, index):
        # The ordinary replay reader materializes one blob. Admit its physical
        # lifetime BEFORE that fetch, never turn resource exhaustion into a
        # shortened record or skipped replay row.
        source = self._geometry_source
        if hasattr(source, "_connect"):
            with source._connect() as connection:
                row = connection.execute("SELECT length(payload) FROM replay ORDER BY sequence LIMIT 1 OFFSET ?", (index,)).fetchone()
            if row is not None:
                with _admit(self.policy, int(row[0]) * 3 + self.width * 4 * 2, "historical geometry replay decode"):
                    pass

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        normalized = index if index >= 0 else len(self) + index
        self._preflight(normalized)
        return _map_activity(self._geometry_source[index], self.width, self.policy)

    def __iter__(self):
        source = self._geometry_source
        if hasattr(source, "_connect") and hasattr(source, "_decode"):
            with source._connect() as connection:
                addresses = connection.execute("SELECT sequence,length(payload) FROM replay ORDER BY sequence")
                for sequence, byte_count in addresses:
                    with _admit(self.policy, int(byte_count) * 3 + self.width * 4 * 2, "historical geometry replay decode"):
                        row = connection.execute("SELECT sequence,sha256,dtype,shape_json,payload FROM replay WHERE sequence=?", (sequence,)).fetchone()
                        if row is None: raise ValueError("historical replay changed during geometry traversal")
                        value = _map_activity(source._decode(row), self.width, self.policy)
                    yield value
        else:
            for value in source:
                yield _map_activity(value, self.width, self.policy)


class GeometryWorkingMemoryView:
    """Keep cold pages intact; adapt before a normal transactional page-in."""

    def __init__(self, source, width):
        self._geometry_source = getattr(source, "_geometry_source", source)
        self.width = int(width)

    @property
    def policy(self): return self._geometry_source.policy
    @policy.setter
    def policy(self, value): self._geometry_source.policy = value
    def __getattr__(self, name): return getattr(self._geometry_source, name)

    def read(self, page_id, *, touch=True):
        source = self._geometry_source
        if hasattr(source, "_connect"):
            with source._connect() as connection:
                row = connection.execute("SELECT length(payload) FROM working_pages WHERE page_id=?", (str(page_id),)).fetchone()
            if row is not None:
                with _admit(self.policy, int(row[0]) * 3 + self.width * 4 * 2, "historical geometry working-page decode"):
                    pass
        value, metadata = source.read(page_id, touch=touch)
        return _map_activity(value, self.width, self.policy), metadata

    def page_in(self, page_id):
        # Map/admit before deletion. A physical failure must not consume the
        # original page. The store verifies the row again in its transaction.
        value, metadata = self.read(page_id, touch=False)
        self._geometry_source.page_in(page_id)
        return value, metadata

    def peek_hot(self, **kwargs):
        return [self.read(key, touch=False) for key in self._geometry_source._rank_hot_page_ids(**kwargs)]

    def page_in_hot(self, **kwargs):
        return [self.page_in(key) for key in self._geometry_source._rank_hot_page_ids(**kwargs)]


def install_geometry_runtime_views(brain):
    descriptor = getattr(brain.config, "native_architecture", None) or {}
    history = descriptor.get("evolutionLineage", {}).get("mutations", ())
    if not any(item.get("mutation") in {"resize-width", "repartition-heads"} for item in history):
        return False
    width = int(brain.config.idea_dim)
    if hasattr(brain, "replay"):
        brain.replay = GeometryReplayView(brain.replay, width)
    if hasattr(brain, "paged_working_memory"):
        brain.paged_working_memory = GeometryWorkingMemoryView(brain.paged_working_memory, width)
    return True


def _default_modules(config, brain):
    # Called only at the future authorized runtime boundary, not on import.
    from torch import nn
    from .liquid import LiquidController
    from .modalities import ModalityHub
    from .model import BitLinear, OmniDecoder
    decoder = OmniDecoder(config).to(brain.device)
    for _ in range(brain.decoder.expert_count):
        decoder.grow_expert(torch.zeros(config.d_model, device=brain.device))
    router = copy.copy(brain.router)
    router._modules = dict(brain.router._modules)
    router._buffers = dict(brain.router._buffers)
    router._parameters = dict(brain.router._parameters)
    router.idea_dim = config.idea_dim
    router.input_projection = BitLinear(config.idea_dim, config.router_neurons, bias=True).to(brain.device)
    router.output_projection = BitLinear(config.router_neurons, config.idea_dim, bias=True).to(brain.device)
    return {"decoder": decoder,
        "memory_bridge": BitLinear(config.vsa_dim, config.idea_dim, bias=True).to(brain.device),
        "idea_adapter": nn.Sequential(BitLinear(config.idea_dim, config.idea_dim * 2, bias=True), nn.SiLU(),
            BitLinear(config.idea_dim * 2, config.idea_dim, bias=True)).to(brain.device),
        "liquid": LiquidController(config.idea_dim, mode=config.liquid_mode, solver_steps=config.liquid_steps).to(brain.device),
        "modalities": ModalityHub(config).to(brain.device), "router": router}


def _default_pager(brain, directory):
    from .native_core_paging import NativeCorePager
    def reserve(count, device):
        status = brain.resource_policy.status(estimated_ram_bytes=int(count) if device.type != "cuda" else 0)
        free = status.get("acceleratorFreeMemoryBytes")
        if status.get("memoryPressure") or device.type != "cpu" and isinstance(free, int) and count > free // 2:
            raise RuntimeError("isolated geometry packed admission paused at the shared resource reserve")
    return NativeCorePager(directory / "native-core", cpu_hot_bytes=0, accelerator_hot_bytes=0,
        reserve_disk=brain.resource_policy.require_disk, reserve_admission=reserve,
        resource_policy=brain.resource_policy, budget_provider=brain._native_core_budget_for_device)


def _owner_inventory(proof, owner, role, dtype):
    packed = "copiedLogicalTrits" in proof
    return {"kind": "packed" if packed else "control", "owner": owner,
        "role": role if packed else "control", "dtype": str(dtype),
        "oldShape": [proof["oldLogicalShape"][0], (proof["oldLogicalShape"][1] + 3) // 4] if packed else proof["oldShape"],
        "newShape": [proof["newLogicalShape"][0], (proof["newLogicalShape"][1] + 3) // 4] if packed else proof["newShape"]}


def _copy_file_control(reader, key, target, axes=None, *, cancelled=None):
    """Bounded source reads into one already admitted control destination."""
    spec = reader.specs[key]
    if spec.dtype != target.dtype or len(spec.shape) != target.ndim or target.ndim > 2:
        raise ValueError("unavailable isolated control geometry: " + key)
    mappings = axes or [[[0, 0, min(old, new)]] for old, new in zip(spec.shape, target.shape)]
    from .architecture_migration import _validate_axis_segments
    mappings = [_validate_axis_segments(axis, old, new) for axis, old, new in zip(mappings, spec.shape, target.shape)]
    item, copied, digest = target.element_size(), 0, hashlib.sha256()
    flat = target.reshape(-1)
    step = max(1, TRANSFER_BYTES // item)
    for start in range(0, flat.numel(), step):
        _cancel(cancelled); flat[start:start + step].zero_()
    rows = [[0, 0, 1]] if target.ndim < 2 else mappings[0]
    columns = [[0, 0, 1]] if target.ndim == 0 else mappings[-1]
    with reader.path.open("rb") as handle:
        for old_row, new_row, row_count in rows:
            for row in range(row_count):
                for old_column, new_column, column_count in columns:
                    for start in range(0, column_count, step):
                        _cancel(cancelled)
                        count = min(step, column_count - start)
                        old_start = old_column + start if target.ndim < 2 else (old_row + row) * spec.shape[1] + old_column + start
                        new_start = new_column + start if target.ndim < 2 else (new_row + row) * target.shape[1] + new_column + start
                        handle.seek(spec.offset + old_start * item)
                        payload = handle.read(count * item)
                        if len(payload) != count * item: raise ValueError("geometry control source ended early")
                        block = torch.frombuffer(bytearray(payload), dtype=target.dtype).to(target.device)
                        flat[new_start:new_start + count].copy_(block)
                        if not torch.equal(flat[new_start:new_start + count].reshape(-1).view(torch.uint8).cpu(), block.reshape(-1).view(torch.uint8).cpu()):
                            raise ValueError("geometry control copy changed source bytes")
                        digest.update(payload); copied += count
    return {"copiedElements": copied, "initializedElements": target.numel() - copied,
        "droppedElements": spec.numel - copied, "oldShape": list(spec.shape), "newShape": list(target.shape),
        "axisSegments": mappings, "copiedCoordinatesSha256": digest.hexdigest(), "peakTransferBytes": TRANSFER_BYTES}


def apply_isolated_geometry_candidate(brain, change, candidate_id, *, module_factory=None, pager_factory=None, cancelled=None):
    """Prepare/validate replacements before committing an isolated object swap."""
    engine = Path(brain.engine_path).resolve()
    if (engine.name != "engine" or engine.parent.name != "model" or engine.parent.parent.name != candidate_id
        or engine.parent.parent.parent.name != "candidates" or not candidate_id
        or any(character not in "0123456789abcdef" for character in candidate_id)):
        raise PermissionError("geometry application is restricted to an isolated candidate model directory")
    assert_architecture_quiescent(brain)
    from .packed_collective_hooks import packed_derivative_collective_active
    if packed_derivative_collective_active():
        raise RuntimeError("geometry preparation cannot cross an active packed derivative collective")
    old_roots = {name: getattr(brain, name) for name in ROOTS}
    if any(getattr(child, "_online_transaction", None) is not None for root in old_roots.values() for child in root.modules()):
        raise RuntimeError("geometry preparation cannot cross an active native online update")
    if brain.core_pager.status().get("pinnedOwners", 0) or getattr(brain, "router_state_pager", None) is not None and brain.router_state_pager.status().get("activeOperations", 0):
        raise RuntimeError("geometry preparation requires quiescent projection/router owners")
    canonical = normalize_architecture_change(change)
    if canonical is None:
        raise ValueError("geometry application requires an explicit typed mutation")
    canonical = {key: value for key, value in canonical.items() if key != "compatibilityBoundary"}
    old_geometry = {key: getattr(brain.config, key) for key in ("d_model", "d_ff", "n_heads", "idea_dim")}
    proposed = geometry_candidate_config(old_geometry, canonical)
    validate_native_architecture(brain.config.native_architecture)
    config = copy.deepcopy(brain.config)
    for key, value in proposed.items(): setattr(config, key, value)
    config.native_architecture = reseal_native_descriptor(config, canonical)
    config.validate()
    source = {"core": BoundedTensorFile(engine / "core.safetensors"), "plasticity": BoundedTensorFile(engine / "plasticity.safetensors")}
    directory = Path(brain._live_paging_cache_directory) / ("geometry-" + candidate_id + "-" + uuid.uuid4().hex)
    pager, replacements = None, None
    proofs, inventory = {}, {}
    changed_attributes = {*ROOTS, "config", "core_pager", "_live_paging_cache_directory", "working_attention_pager",
        "_working_attention_envelope", "_optimizer", "_optimizer_offloaded", "_optimizer_scratch_pointer",
        "liquid_state", "working_memory", "replay", "paged_working_memory", "_paged_vector_cache",
        "_native_residency_baseline_bytes"}
    original = {name: getattr(brain, name) for name in changed_attributes if hasattr(brain, name)}
    committed = False
    try:
        brain.core_pager.flush()
        brain.core_pager.cool_to_budget(0)
        _cancel(cancelled)
        # Ordinary controls, registries and at least one idea row are admitted
        # separately from cold packed weights; no FP projection master exists.
        ordinary = sum(spec.byte_count for label, reader in source.items() for key, spec in reader.specs.items()
            if not (label == "plasticity" and key.startswith("router."))
            and spec.dtype != torch.uint8)
        reserve_bytes = ordinary * max(2, math.ceil(config.idea_dim / old_geometry["idea_dim"]) * 2) + config.idea_dim * 128 + 2 * TRANSFER_BYTES
        with _admit(brain.resource_policy, reserve_bytes, "isolated geometry nonweight preparation"):
            pager = (pager_factory or _default_pager)(brain, directory)
            with preserve_runtime_rng(brain.device), pager.construction(from_checkpoint=True):
                replacements = (module_factory or _default_modules)(config, brain)
            if set(replacements) != set(ROOTS): raise ValueError("geometry factory omitted a native cortical root")
            for root_name, target_root in replacements.items():
                if target_root is old_roots[root_name]: raise ValueError("geometry factory reused an original cortical root")
                if list(target_root.parameters()) or list(old_roots[root_name].parameters()):
                    raise ValueError("geometry migration does not create or migrate floating projection parameters")
                old_modules, new_modules = dict(old_roots[root_name].named_modules()), dict(target_root.named_modules())
                if set(old_modules) != set(new_modules): raise ValueError("geometry factory changed undeclared module topology")
                reader = source["plasticity" if root_name == "router" else "core"]
                label = "plasticity" if root_name == "router" else "core"
                for relative, child in new_modules.items():
                    before = old_modules[relative]
                    for key, value in before.__dict__.items():
                        if key not in _GEOMETRY_ATTRIBUTES and key in child.__dict__ and type(value) in (bool, int, float, str):
                            setattr(child, key, value)
                old_state = old_roots[root_name].state_dict()
                if set(old_state) != set(target_root.state_dict()):
                    raise ValueError("geometry factory removed or added undeclared tensor state")
                for relative_key, target in target_root.state_dict().items():
                    _cancel(cancelled)
                    key = root_name + "." + relative_key
                    qualified = label + ":" + key
                    if key not in reader.specs or relative_key not in old_state:
                        raise ValueError("geometry factory introduced an undeclared tensor: " + key)
                    spec, old_tensor = reader.specs[key], old_state[relative_key]
                    parent_name, _, field = relative_key.rpartition(".")
                    owner, previous = new_modules[parent_name], old_modules[parent_name]
                    alias = (old_tensor.numel() and target.numel() and old_tensor.device == target.device
                        and old_tensor.untyped_storage().data_ptr() == target.untyped_storage().data_ptr())
                    retained_router = root_name == "router" and (relative_key.startswith(("population.", "synapses.")) or relative_key in {"active_prefix_neurons", "region_ends"})
                    if alias and retained_router:
                        continue
                    if alias: raise ValueError("geometry target aliases an original cortical tensor")
                    if field in {"_packed_forward_weight", "_packed_forward_bias"}:
                        role = "bias" if field.endswith("bias") else "weight"
                        logical_name = "ternary_bias_shape" if role == "bias" else "ternary_weight_shape"
                        old_shape, new_shape = getattr(previous, logical_name), getattr(owner, logical_name)
                        old_shape = old_shape() if callable(old_shape) else old_shape
                        new_shape = new_shape() if callable(new_shape) else new_shape
                        old_shape = (1, old_shape[0]) if role == "bias" and len(old_shape) == 1 else tuple(old_shape)
                        new_shape = (1, new_shape[0]) if role == "bias" and len(new_shape) == 1 else tuple(new_shape)
                        plan = geometry_owner_plan(root_name + "." + parent_name, old_shape, new_shape, old_geometry, proposed, role=role)
                        proof = copy_packed_geometry(reader, target, old_shape[1], new_shape[1], source_name=key,
                            seed=candidate_id, tensor_name=qualified, row_segments=plan["rowSegments"],
                            column_segments=plan["columnSegments"], cancelled=cancelled)
                    else:
                        if tuple(target.shape) != spec.shape and any(size == 0 for size in target.shape):
                            # Dynamic route/schema registries are not new empty
                            # registries: retain every existing admitted row.
                            with _admit(brain.resource_policy, spec.byte_count, "geometry dynamic control registry"):
                                target = torch.empty(spec.shape, dtype=spec.dtype, device=target.device)
                            owner._buffers[field] = target
                        if not target.numel() and spec.numel == 0: continue
                        axes = None
                        if field == "_row_stability":
                            shape = getattr(previous, "ternary_weight_shape")
                            shape = shape() if callable(shape) else shape
                            target_shape = getattr(owner, "ternary_weight_shape")
                            target_shape = target_shape() if callable(target_shape) else target_shape
                            plan = geometry_owner_plan(root_name + "." + parent_name, tuple(shape), tuple(target_shape), old_geometry, proposed)
                            axes = [plan["rowSegments"]]
                        if target.ndim > 2 and tuple(target.shape) == spec.shape:
                            reader.copy_into(key, target, cancelled=cancelled); continue
                        proof = _copy_file_control(reader, key, target, axes, cancelled=cancelled)
                        role = "control"
                    proofs[qualified] = proof
                    inventory[qualified] = _owner_inventory(proof, root_name + "." + parent_name, role, target.dtype)
                for child in target_root.modules():
                    if getattr(child, "_native_core_pager", None) is pager:
                        pager.attach(child, loaded=True)
                        child._native_loading_checkpoint = False
                        for field in ("_packed_validated_version", "_bias_validated_version", "_scale_validated_version"):
                            if hasattr(child, field): setattr(child, field, -1)
                target_root.train(old_roots[root_name].training)
            # Hot activity is transformed without stacking the source corpus.
            liquid = torch.empty((1, config.idea_dim), dtype=brain.liquid_state.dtype, device=brain.device)
            proof = _copy_file_control(source["plasticity"], "state.liquid", liquid, cancelled=cancelled)
            proofs["plasticity:state.liquid"] = proof
            inventory["plasticity:state.liquid"] = _owner_inventory(proof, "state.liquid", "control", liquid.dtype)
            working = []
            if "state.working_memory" in source["plasticity"].specs:
                spec = source["plasticity"].specs["state.working_memory"]
                with _admit(brain.resource_policy, spec.shape[0] * config.idea_dim * 4, "geometry hot working activity"):
                    values = torch.empty((spec.shape[0], config.idea_dim), dtype=spec.dtype, device=brain.device)
                proof = _copy_file_control(source["plasticity"], "state.working_memory", values, cancelled=cancelled)
                working = [row for row in values]
                proofs["plasticity:state.working_memory"] = proof
                inventory["plasticity:state.working_memory"] = _owner_inventory(proof, "state.working_memory", "control", values.dtype)
            pager.bind_names(tuple((name + ".", root) for name, root in replacements.items()))
            _cancel(cancelled)
            for name, root in replacements.items(): setattr(brain, name, root)
            brain.config, brain.core_pager = config, pager
            brain._live_paging_cache_directory = directory
            pager.refresh_budget(force=True)
            if hasattr(brain, "working_attention_pager"): del brain.working_attention_pager
            brain.liquid_state, brain.working_memory = liquid, working
            brain._paged_vector_cache = None
            install_geometry_runtime_views(brain)
            brain._configure_working_attention_resources()
            brain._replace_optimizer()
            _cancel(cancelled)
            pager.flush()
            committed = True
    except BaseException:
        new_attention = getattr(brain, "working_attention_pager", None)
        if new_attention is not None and new_attention is not original.get("working_attention_pager"):
            try: new_attention.close()
            except Exception: pass
        for name in changed_attributes:
            if name in original: setattr(brain, name, original[name])
            elif hasattr(brain, name): delattr(brain, name)
        if pager is not None:
            try: pager.close()
            except Exception: pass
        raise
    if committed:
        # Only retired scratch is cleaned; immutable checkpoint/history files
        # are never deleted. Cleanup failure cannot undo a completed swap.
        cleanup = {}
        for name in ("core_pager", "working_attention_pager"):
            old = original.get(name)
            if old is not None:
                try: cleanup[name] = old.close()
                except Exception as error: cleanup[name] = {"deferred": True, "error": str(error)}
        return {"mutation": canonical, "functionPreserved": False, "oldGeometry": old_geometry,
            "newGeometry": proposed, "tensorProofs": proofs, "ownerInventory": inventory,
            "nonpersistentCachesRebuilt": True, "originalDurableActivityBytesPreserved": True, "retiredScratchCleanup": cleanup}
