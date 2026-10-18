"""Complete cached-step callbacks for unified Auto, never a second model.

Definitions/import perform no neural work. The same loaded packed decoder uses
private COW activity and original-home sampling. Only exact declared owners,
typed read-only connected recall and current admitted layouts are eligible.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import ExitStack, contextmanager

import torch

from .cached_decoder_native import NativeCachedDecoderOps
from .cached_operation_layout import cached_operation_range, maximum_safe_key_tokens, sampling_inference_class
from .cached_operation_snapshot import fork_operation_cache
from .connected_recall_binding import ConnectedRecallBinding, PrivateConnectedRecallBinding, recall_source_snapshot
from .decoder_execution_native import _tensor_guard
from .operation_auto_registry import CACHED_KINDS, CACHED_STAGES, PRIVATE_OPERATION
from .operation_auto_policy import OperationAutoRestoreRequired
from .operation_transfer_observation import recurrent_transfer_observation
from .parameter_diagnostics import native_parameter_read_only
from .regional_execution import _ACTIVE_REGIONAL, canonical_device, transient_regional_scope
from .whole_path_execution import digest


class CompleteCachedTrial:
    def __init__(self, fork, sample, rng, activity, contract, observations=None):
        self.fork, self.sample, self.rng_state, self.activity, self.contract = fork, sample, rng, activity, contract
        self.observations = observations
    @property
    def cache(self): return self.fork.cache
    @property
    def neural_state(self): return self.activity[0][2]
    def take_cache(self): return self.fork.take()
    def close(self): self.fork.close(); self.sample = self.rng_state = self.activity = self.observations = None


class _Target:
    def __init__(self, ops, candidate, stopped, operation=None):
        from .operation_auto_native import _PerceptionTarget
        self._inner = _PerceptionTarget(ops, candidate, stopped)
        self._mixed = None
        if candidate.get("backendMode") == "native-mixed-sequential":
            from .mixed_region_native import MixedDecoderBlockConsumer
            self._mixed = MixedDecoderBlockConsumer(ops, candidate, operation, stopped)
    @property
    def active(self): return self._inner.active
    @active.setter
    def active(self, value): self._inner.active = value
    @property
    def _lease(self): return self._inner._lease
    def policy_identity(self): return self._inner.policy_identity()
    def packed_limits(self, policy): return self._inner.packed_limits(policy)
    def execute(self, module, method, hidden, args, kwargs):
        # All selected owners already share this one operation target. The
        # existing block wrapper must not create another block-only schedule.
        if self._inner.stopped.is_set(): raise InterruptedError("complete cached operation stopped")
        if self._mixed is not None: return self._mixed.execute(self, module, method, hidden, args, kwargs)
        return method(module, hidden, *args, **kwargs)


class NativeCachedWholeOperationOps(NativeCachedDecoderOps):
    def __init__(self, brain, host):
        super().__init__(brain)
        self.host, self._template = host, None

    def mixed_candidates(self, operation, review, *, current):
        from .mixed_region_native import prepare_mixed_operation_candidates
        return prepare_mixed_operation_candidates(self, operation, review, current=current)

    def _hash_tensor(self, value):
        # The complete-operation proof may visit a large real cue/cache. Use
        # the current proof scope's cancellation at every bounded checksum
        # chunk, including initial capture; restoration deliberately enters a
        # non-cancellable proof scope so Stop cannot strand original owners.
        if value is None: return None
        from contextlib import nullcontext
        pager = self.brain.working_attention_pager
        hasher = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        block = max(1, 262144 // value.element_size())
        layout = (nullcontext() if value.is_contiguous() else self.brain.resource_policy.reserve_ram(
            value.numel() * value.element_size(), "complete cached checksum layout"))
        with layout, self.brain.resource_policy.reserve_ram(min(value.numel(), block) * value.element_size() * 3,
                "bounded complete cached checksum"):
            flat, version = value.detach().reshape(-1), int(value._version)
            for start in range(0, value.numel(), block):
                pager.check_cancelled()
                payload = flat[start:start + block].to(device="cpu", non_blocking=False).contiguous().view(torch.uint8).numpy()
                hasher.update(memoryview(payload))
            if int(value._version) != version: raise RuntimeError("complete cached tensor changed during its exact proof")
        return hasher.hexdigest()

    @staticmethod
    def _kind(cache):
        from .model import _DecoderInferenceCache
        from .connected_regions import _ConnectedInferenceCache
        return "ordinary" if type(cache) is _DecoderInferenceCache else "connected" if type(cache) is _ConnectedInferenceCache else None

    def _closure(self):
        from . import model as m, connected_regions as c, liquid as l, spiking as s, sparse_router_state as r, packed_connections as p
        brain = self.brain
        roots, regions, decoders, fixed, bindings, visiting = [], [], [], {}, [], set()
        def walk(decoder, path):
            if type(decoder) not in {m.OmniDecoder, c.ConnectedOmniDecoder}: raise ValueError("opaque decoder implementation cannot enter complete Auto")
            if id(decoder) in visiting: raise ValueError("cyclic connected decoder ownership cannot enter complete Auto")
            if any(name in decoder.__dict__ for name in ("forward", "_generation_step", "_generation_step_bounded", "_generation_signature", "_apply_experts")):
                raise ValueError("instance decoder computation is not a declared complete operation")
            visiting.add(id(decoder))
            decoders.append((path, decoder))
            if type(decoder) is c.ConnectedOmniDecoder:
                for name in ("workspace_exchange", "cortical_exchange"): roots.append((path + "." + name, getattr(decoder, name)))
                for index, region in enumerate(decoder.regions):
                    if type(region) is not c.NativeRegion or any(name in region.__dict__ for name in ("condition", "observe", "attach_recall")):
                        raise ValueError("unknown connected region callback is not a closed pure operation")
                    key = path + ".region" + str(index)
                    regions.append((key, region)); walk(region.decoder, key + ".decoder")
                    for name in ("memory_bridge", "idea_adapter", "recall_projection", "liquid", "router"):
                        roots.append((key + "." + name, getattr(region, name)))
                    for module in region.router.synapses.modules(): fixed[id(module)] = module
                    fixed[id(region.router.population)] = region.router.population
                    callback = region._recall
                    if callback is not None:
                        if (type(callback) not in {ConnectedRecallBinding, PrivateConnectedRecallBinding}
                                or "private" in vars(callback)):
                            raise ValueError("opaque connected recall callback lacks a current read-only owner fence")
                        bindings.append((key, region, callback))
            else:
                for name in ("embedding", "global_workspace", "workspace_strength", "memory_projection", "memory_strength",
                             "blocks", "experts", "expert_prototypes", "final_norm", "language_head"):
                    roots.append((path + "." + name, getattr(decoder, name)))
            visiting.remove(id(decoder))
        walk(brain.decoder, "decoder")
        allowed = {m.PackedAdaptiveBitLinear, m.PackedAdaptiveTernaryEmbedding, m.PackedAdaptiveTernaryScalar, m.RMSNorm,
            m.RotaryEmbedding, m.GlobalWorkspace, m.DecoderBlock, m.CausalSelfAttention, m.SwiGLU, m.TernaryExpert,
            c.RegionExchange, l.LiquidController, l.CfCCell, l.LTCCell, s.AssociativeSpikingRouter, s.LIFPopulation,
            s.STDPSynapses, r.SparseRouterBlock, p.PooledParallelConnections, p.PackedContactPage,
            torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.SiLU}
        modules, names = {}, {}
        for path, root in roots:
            for suffix, module in root.named_modules():
                if type(module) not in allowed or any(name in module.__dict__ for name in ("forward", "forward_cached", "causal_step", "causal_summaries", "recurrent")) or any(getattr(module, name, {})
                        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
                    raise ValueError("unknown decoder module/hook prevents a complete read-only comparison")
                if type(module) is torch.nn.SiLU and module.inplace: raise ValueError("in-place decoder activity cannot enter private Auto")
                if getattr(module, "_online_transaction", None) is not None: raise ValueError("packed owner is in a live learned transaction")
                modules[id(module)] = module; names.setdefault(id(module), path + ("." + suffix if suffix else ""))
        if any(getattr(torch.nn.modules.module, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks")):
            raise ValueError("global module callbacks prevent isolated decoder comparison")
        packed = [module for module in modules.values() if module in brain.core_pager._owners]
        if not packed or any(not brain.core_pager._owners[module].loaded for module in packed): raise ValueError("complete loaded packed text owners are required")
        # The scalar is a container around its registered ``levels`` embedding,
        # not another packed owner. Requiring the container itself in the pager
        # would incorrectly make every real ordinary decoder ineligible.
        packed_types = {m.PackedAdaptiveBitLinear, m.PackedAdaptiveTernaryEmbedding, p.PackedContactPage}
        if any(type(module) in packed_types and key not in fixed and (module not in brain.core_pager._owners
                    or getattr(module, "_native_core_pager", None) is not brain.core_pager)
                for key, module in modules.items()):
            raise ValueError("complete cached operation has an unregistered packed or foreign learned owner")
        for _key, region in regions:
            synapses = region.router.synapses
            sparse = getattr(synapses, "_sparse_state", None)
            if sparse is None or not sparse.ready or sparse._transaction_active or getattr(synapses, "_router_failed_journal", None) is not None:
                raise ValueError("unfenced/failed recurrent learned owner cannot enter connected Auto")
            if getattr(synapses, "_router_state_pager", None) is None: raise ValueError("connected recurrent storage has no owned pager")
        return {"roots": roots, "modules": modules, "names": names, "packed": packed, "fixed": fixed,
                "regions": regions, "decoders": decoders, "bindings": bindings,
                "kind": CACHED_KINDS[1] if type(brain.decoder) is c.ConnectedOmniDecoder else CACHED_KINDS[0]}

    def _registry(self):
        closure = self._closure()
        rows = []
        for key, module in closure["modules"].items():
            fields = set(getattr(module, "_packed_structural_fields", ())) | {"dimensions", "in_features", "out_features", "embedding_dim",
                "num_embeddings", "n_heads", "head_dim", "slots", "iterations", "eps", "base", "leak", "threshold", "neurons",
                "query_chunk_tokens", "key_chunk_tokens", "query_chunk_slots", "max_seq_len", "max_cached_tokens", "solver_steps",
                "widths", "input_size", "hidden_size", "mode", "dropout", "pre_neurons", "post_neurons", "cached_seq_len"}
            structural = {name: getattr(module, name) for name in sorted(fields) if type(getattr(module, name, None)) in {int, float, str, bool, tuple}}
            rows.append([closure["names"][key], str(key), type(module).__module__ + "." + type(module).__name__, structural,
                [[name, list(value.shape), str(value.dtype)] for name, value in module._buffers.items()
                 if isinstance(value, torch.Tensor) and name not in module._non_persistent_buffers_set]])
        identities = [[name, region.identity.region_id, region.identity.parent_id, region.identity.generation_sha256] for name, region in closure["regions"]]
        recalls = [[key, callback.authority_id, str(id(callback.authoritative_source)), callback.workspace_slots] for key, _region, callback in closure["bindings"]]
        decoder_configs = [[path, decoder.config.to_dict()] for path, decoder in closure["decoders"]]
        return digest([rows, identities, recalls, decoder_configs]), self.brain.core_pager

    def supported(self):
        try:
            if getattr(self.brain, "_native_execution_restore_required", False): return False, "exact-owner-restore-required"
            self._closure()
            return True, "complete-current-owned-cached-step"
        except (ValueError, RuntimeError) as error: return False, str(error)

    def work_kinds(self):
        supported, _reason = self.supported()
        return [self._closure()["kind"]] if supported else []

    def retire_observation(self):
        if self._template is not None:
            self._template["scope"].close(); self._template["lease"].release(); self._template = None

    def _scope(self, closure):
        registry, _pager = self._registry()
        if self._template is not None and self._template["registry"] != registry: self.retire_observation()
        if self._template is None:
            lease = self.brain.resource_policy.reserve_ram(16384 + len(closure["modules"]) * 1024, "complete decoder owner epoch/registry metadata")
            try:
                scope = self.brain.core_pager.observe_learning_owners(closure["packed"])
                self._template = {"registry": registry, "scope": scope, "lease": lease, "routerTokens": None, "routerEpoch": 0}
                lease.mark_allocated(16384 + len(closure["modules"]) * 1024)
            except BaseException: lease.release(); raise
        return self._template

    def _state(self, closure, scope, session):
        from .model import RotaryEmbedding
        versions, controls, router = scope["scope"].snapshot(), hashlib.sha256(), []
        for key, module in closure["modules"].items():
            if module in self.brain.core_pager._owners: continue
            for name, value in module._buffers.items():
                if not isinstance(value, torch.Tensor) or (name in module._non_persistent_buffers_set
                        and not (type(module) is RotaryEmbedding and name == "inverse_frequency")): continue
                if key in closure["fixed"]:
                    # Actual registered CPU-backed sparse state is guarded by
                    # object/version/shape/device tokens, not a whole N² hash.
                    router.append([str(key), name, _tensor_guard(value)])
                else: controls.update(json.dumps([closure["names"][key], name, self._hash_tensor(value)], separators=(",", ":")).encode())
        token = digest(router)
        if scope["routerTokens"] is not None and scope["routerTokens"] != token: scope["routerEpoch"] += 1
        scope["routerTokens"] = token
        if closure["regions"]: versions["connected-sparse-router-binding"] = scope["routerEpoch"]
        graph = []
        for key, _region, binding in closure["bindings"]:
            source = recall_source_snapshot(binding, reserve=lambda size: self.brain.resource_policy.reserve_ram(size, "exact connected graph revision metadata"),
                check_cancelled=self.brain.working_attention_pager.check_cancelled)
            graph.append([key, source]); versions["graph:" + key] = source["sourceStateRevision"]
        if not versions: raise RuntimeError("complete decoder state lacks a proven native owner")
        return {"kind": "live-owner-epochs", "bindingSha256": digest([scope["registry"], versions, controls.hexdigest(), token, graph]),
            "ownerSessionId": session, "complete": True, "ownerVersions": versions}

    def _cache_nodes(self, decoder, cache, prefix="decoder"):
        from .connected_regions import ConnectedOmniDecoder
        mode = self._kind(cache)
        if mode is None or mode != ("connected" if type(decoder) is ConnectedOmniDecoder else "ordinary"):
            raise ValueError("cache shape does not match its exact decoder owner")
        yield prefix, decoder, cache
        if mode == "connected":
            if not len(decoder.regions) == len(cache.workspace) == len(cache.attention) == len(cache.temporal) == len(cache.hidden_sum) == len(cache.nested):
                raise ValueError("connected cache has incomplete parent state")
            for index, region in enumerate(decoder.regions):
                state = cache.temporal[index]
                if not isinstance(state, tuple) or len(state) != 3 or any(not isinstance(value, torch.Tensor) for value in state):
                    raise ValueError("connected retained cache lacks its full temporal state")
                if type(region.decoder) is ConnectedOmniDecoder:
                    if cache.nested[index] is None: raise ValueError("nested connected retained cache was not initialized")
                    yield from self._cache_nodes(region.decoder, cache.nested[index], prefix + ".r" + str(index))
                elif cache.nested[index] is not None: raise ValueError("ordinary parent acquired an undeclared nested cache")

    def _layout(self, window, cache):
        from .working_attention_paging import PagedTensorSequence, attention_tile_bytes
        rows, length, capacity = [], int(window.shape[1]), int(self.brain.config.max_seq_len)
        pager = self.brain.working_attention_pager
        for path, decoder, node in self._cache_nodes(self.brain.decoder, cache):
            capacity = min(capacity, int(decoder.config.max_seq_len))
            if node.length + 1 != length: raise ValueError("nested retained cache frontier changed")
            groups = [(path, decoder, node.attention)] if self._kind(node) == "ordinary" else [
                (path + ".r" + str(index), region.decoder, node.attention[index]) for index, region in enumerate(decoder.regions)
                if self._kind(node.nested[index]) != "connected"]
            for owner, ordinary, layers in groups:
                capacity = min(capacity, int(ordinary.config.max_seq_len))
                if len(layers) != len(ordinary.blocks): raise ValueError("retained cache lost an actual attention block")
                for index, (block, layer) in enumerate(zip(ordinary.blocks, layers)):
                    if layer is None: raise ValueError("retained attention cache is missing")
                    for role, sequence in (("key", layer.key), ("value", layer.value)):
                        if not isinstance(sequence, PagedTensorSequence) or sequence.pager is not pager: raise ValueError("opaque/foreign cached attention pages")
                        shape, axis = list(sequence.shape), sequence.axis
                        element = torch.empty((), dtype=sequence.dtype).element_size()
                        token_bytes = math.prod(value for j, value in enumerate(shape) if j != axis) * element
                        configured = int(block.attention.key_chunk_tokens)
                        safe = maximum_safe_key_tokens(configured, budget=pager.device_tile_budget_bytes,
                            cost=lambda k, b=shape[0], h=block.attention.n_heads, d=block.attention.head_dim, e=element:
                                attention_tile_bytes(b, h, d, 1, k, e, training=False))
                        rows.append({"owner": owner + ".block" + str(index) + "." + role, "axis": axis, "shape": shape, "dtype": str(sequence.dtype),
                            "tokenBytes": token_bytes, "pageTokens": pager.page_tokens, "tileBudgetBytes": pager.device_tile_budget_bytes,
                            "retainedPoolPolicySha256": digest([{key: pool.get(key) for key in
                                ("deviceId", "executionDevice", "memoryGroupId", "budgetBytes", "pageBytes", "partitionMode",
                                 "minimumPackedOwnerBytes", "controlReserveBytes", "transferReserveBytes")}
                                for pool in getattr(pager, "_retained_context_pools", ())]),
                            "previousPageTokens": sequence._page_tokens, "ends": list(sequence.ends),
                            "configuredKeyTokens": configured, "maximumSafeKeyTokens": safe})
        return cached_operation_range(length, capacity, rows)

    def valid_frontier(self, window, cue, cache):
        try:
            if (torch.is_grad_enabled() or self.brain.decoder.training or not isinstance(window, torch.Tensor)
                    or window.dtype != torch.long or window.ndim != 2 or window.shape[0] < 1 or window.shape[1] < 2
                    or canonical_device(window.device) != canonical_device(self.brain.device)):
                return False
            if self.brain.core_pager.status()["pinnedOwners"] or getattr(getattr(self.brain, "_regional_execution", None), "active", False): return False
            if cache.signature != self.brain.decoder._generation_signature(cue): return False
            closure = self._closure()
            # Existing transmission capture is a live independent observer,
            # not a clonable private recurrent state. Preserve that actual
            # ordinary operation rather than replaying or omitting its events.
            if any(getattr(region.router.synapses, "_native_transmission_observer", None) is not None
                    for _key, region in closure["regions"]): return False
            self._layout(window, cache)
            return True
        except (ValueError, RuntimeError, AttributeError): return False

    def cache_sha(self, cache):
        from .working_attention_paging import PagedTensorSequence, SavedActivityTensor
        hasher = hashlib.sha256()
        def emit(value): hasher.update(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        def activity(value):
            if value is None: emit(None)
            elif isinstance(value, (PagedTensorSequence, SavedActivityTensor)):
                emit([type(value).__name__, list(value.shape), str(value.dtype), getattr(value, "axis", None), getattr(value, "ends", None)])
                for identifier in value.pages:
                    page = value.pager._pages[identifier]
                    emit([list(page.shape), str(page.dtype), self._page_sha(value.pager, identifier)])
            else: emit(self._hash_tensor(value))
        stack = [cache]
        while stack:
            node = stack.pop(); mode = self._kind(node)
            if mode is None: raise ValueError("unknown retained cache cannot be authenticated")
            emit([mode, node.length])
            groups = [node.workspace] if mode == "ordinary" else node.workspace
            for group in groups:
                emit(len(group))
                for chunk in group:
                    for value in (chunk.queries, chunk.mass, chunk.weighted_values): activity(value)
            layers = [node.attention] if mode == "ordinary" else node.attention
            for group in layers:
                emit(len(group))
                for layer in group:
                    if layer is None: emit(None)
                    else: activity(layer.key); activity(layer.value)
            if mode == "ordinary":
                for value in (node.hidden_sum, node.memory_hidden, node.neural_state): activity(value)
            else:
                for state in node.temporal:
                    for value in state: activity(value)
                for value in node.hidden_sum: activity(value)
                emit([child is not None for child in node.nested])
                stack.extend(child for child in reversed(node.nested) if child is not None)
        return hasher.hexdigest()

    @contextmanager
    def proof_scope(self, stopped=None):
        cancelled = ((lambda: False) if stopped is None else stopped.is_set)
        with self.brain.working_attention_pager.activate(cancelled=cancelled): yield

    def _activities(self, closure):
        from .connected_regions import ConnectedOmniDecoder
        rows = [(decoder, "last_generation_neural_state", getattr(decoder, "last_generation_neural_state", None))
                for path, decoder in closure["decoders"] if path == "decoder" or type(decoder) is ConnectedOmniDecoder]
        rows.extend((decoder, "last_region_activity", decoder.last_region_activity) for _path, decoder in closure["decoders"] if hasattr(decoder, "last_region_activity"))
        return rows

    def capture_frontier(self, window, cue, cache, generator, sampling, session, review, *, stopped=None):
        closure = self._closure(); stack = ExitStack()
        try:
            # Keep CPU router representation stable through the private pool.
            # Existing reentrant read operations do not bypass learned writes.
            pagers = {id(region.router.synapses._router_state_pager): region.router.synapses._router_state_pager for _key, region in closure["regions"]}
            for pager in pagers.values(): stack.enter_context(pager.operation())
            scope = self._scope(closure)
            lease = self.brain.resource_policy.reserve_ram(131072 + len(closure["modules"]) * 512, "complete cached frontier metadata and home RNG")
            stack.callback(lease.release)
            with self.proof_scope(stopped if stopped is not None else getattr(self.host, "_stop_event", None)):
                guard, layout = (_tensor_guard(window), _tensor_guard(cue)), self._layout(window, cache)
                rng = generator.get_state().clone()
                source_sha = self.cache_sha(cache)
                state = self._state(closure, scope, session)
                values = {key: value for key, value in sampling.items() if key != "text_boundaries"}
                values["textBoundaries"] = None if sampling["text_boundaries"] is None else [dict(vars(value)) for value in sampling["text_boundaries"]]
                sampling_sha = digest(values)
                geometry = digest([scope["registry"], layout, list(window.shape), None if cue is None else list(cue.shape)])
                binding = review["binding"]
                identity = {"ownerId": self.brain.brain_id, "ownerSessionId": session, "workKind": closure["kind"], "stageCoverage": list(CACHED_STAGES),
                    "workloadSha256": digest([self._hash_tensor(window), self._hash_tensor(cue), source_sha, self._hash_tensor(rng)]),
                    "geometrySha256": geometry, "samplingContractSha256": sampling_sha,
                    **{key: binding[key] for key in ("policySha256", "topologySha256", "budgetSha256")},
                    "cache": {"heat": "warm", "logicalStateSha256": source_sha, "backendInitialization": "warm"}, "nativeState": state}
                lease.mark_allocated(131072 + len(closure["modules"]) * 512)
            return {"window": window, "cue": cue, "cache": cache, "generator": generator, "sampling": sampling, "rng": rng,
                "identity": identity, "guards": guard, "originalCacheSha256": source_sha, "closure": closure, "scope": scope, "stack": stack,
                "contextRange": layout, "originalActivity": self._activities(closure),
                "originalCorticalObservations": getattr(self.brain.core_pager, "_observations", None),
                "residency": [(module, self.brain.core_pager._owners[module].device) for module in closure["packed"]],
                "ownerObjects": {module: self.brain.core_pager._owners[module] for module in closure["packed"]},
                "stratumKey": digest([closure["kind"], scope["registry"], layout, sampling_inference_class(sampling), binding]),
                "planContract": {"geometrySha256": digest([scope["registry"], layout]), "contextRange": layout,
                    "numericGate": "exact-cache-connected-activity-neural-sampling-equality",
                    "reuseBasis": "same-validated-operator-page-layout-range-current-owner-policy-inference"}}
        except BaseException: stack.close(); raise

    def operation_state(self, operation, session): return self._state(operation["closure"], operation["scope"], session)

    def frontier_unchanged(self, operation, session):
        with self.proof_scope():
            return (operation["guards"] == (_tensor_guard(operation["window"]), _tensor_guard(operation["cue"]))
                and torch.equal(operation["generator"].get_state(), operation["rng"])
                and self.operation_state(operation, session) == operation["identity"]["nativeState"]
                and self.cache_sha(operation["cache"]) == operation["originalCacheSha256"])

    def release_frontier(self, operation): operation["stack"].close()

    def rebind_original(self, operation, session):
        if getattr(self.brain, "_native_execution_restore_required", False): return
        with self.proof_scope():
            if (operation["guards"] == (_tensor_guard(operation["window"]), _tensor_guard(operation["cue"]))
                    and self.operation_state(operation, session) == operation["identity"]["nativeState"]
                    and not getattr(operation["cache"], "_closed", False)):
                try: same = self.cache_sha(operation["cache"]) == operation["originalCacheSha256"]
                except (ValueError, RuntimeError, KeyError): return
                if same: self._signatures(self.brain.decoder, operation["cache"], operation["cue"])

    def _signatures(self, decoder, cache, cue):
        from .connected_regions import ConnectedOmniDecoder
        cache.signature = decoder._generation_signature(cue)
        if type(decoder) is ConnectedOmniDecoder:
            cues = [None] * len(decoder.regions) if cue is None else list(cue.split(decoder.widths, -1))
            for index, region in enumerate(decoder.regions):
                if cache.nested[index] is not None: self._signatures(region.decoder, cache.nested[index], cues[index])

    @contextmanager
    def pure_scope(self, stopped):
        token = PRIVATE_OPERATION.set(True)
        try:
            with torch.no_grad(), native_parameter_read_only(), self.proof_scope(stopped), recurrent_transfer_observation(
                    lambda count: setattr(self, "_activation_transfer_bytes", self._activation_transfer_bytes + count)):
                yield
        finally: PRIVATE_OPERATION.reset(token)

    @contextmanager
    def _placement(self, candidate, operation, stopped, observations=None):
        brain, pager, closure = self.brain, self.brain.core_pager, operation["closure"]
        self.admit_candidate(candidate)
        controls = [(module, dict(module._buffers), dict(module._parameters), getattr(module, "_native_compute_device", None))
                    for key, module in closure["modules"].items() if module not in pager._owners and key not in closure["fixed"]]
        packed = [(module, hasattr(module, "_native_compute_device"), getattr(module, "_native_compute_device", None))
                  for module in closure["packed"]]
        activity = self._activities(closure)
        callbacks, views, graph_observers, transmissions = [], [], [], []
        original_observations = getattr(pager, "_observations", None)
        observation_swapped = False
        if observations is not None and isinstance(original_observations, dict):
            observations.update({"requests": dict(pager._observation_requests), "scopes": dict(pager._observation_scopes)})
        home = canonical_device(candidate["executionDevice"]) == canonical_device(brain.device)
        consumer = _Target(self, candidate, stopped, operation)
        active = _ACTIVE_REGIONAL.set((consumer, consumer._lease["epoch"], candidate["executionDevice"]))
        try:
            if observations is not None and isinstance(original_observations, dict):
                pager._observations = {}; observation_swapped = True
            with transient_regional_scope(consumer):
                for module, _present, _device in packed:
                    if not home: module._native_compute_device = torch.device(candidate["executionDevice"])
                for module, buffers, parameters, _device in controls:
                    if any(isinstance(value, torch.Tensor) for value in parameters.values()):
                        raise ValueError("floating learned decoder parameter is not eligible for temporary placement")
                    for name, value in buffers.items():
                        if not home and isinstance(value, torch.Tensor) and name != "_next_page":
                            module._buffers[name] = self.move_activity(value, candidate["executionDevice"])
                for _key, region, binding in closure["bindings"]:
                    view, private, lease = binding.private(cancelled=stopped.is_set, transfer=self.move_activity,
                        reserve_metadata=lambda size: brain.resource_policy.reserve_ram(size, "private connected attention membership"))
                    views.append((view, lease)); callbacks.append((region, region._recall))
                    for store in (binding.authoritative_source.neurons, binding.authoritative_source.synapses):
                        graph_observers.append((store, getattr(store, "_recall_graph_observer", None)))
                    region.attach_recall(private)
                for _key, region in closure["regions"]:
                    owner = region.router.synapses
                    transmissions.append((owner, getattr(owner, "_native_transmission_observer", None)))
                    # Private trials cannot publish repeated diagnostic contact
                    # events as if every candidate were an actual user step.
                    owner._native_transmission_observer = None
                yield
        finally:
            consumer.active = False; _ACTIVE_REGIONAL.reset(active)
            if observation_swapped:
                observations["rows"] = pager._observations
                pager._observations = original_observations
            try:
                scratch_error = None
                for view, lease in views:
                    try: view.release_paged_recall_scratch()
                    except BaseException as error:
                        if scratch_error is None: scratch_error = error
                    finally: lease.release()
                if scratch_error is not None: raise scratch_error
            finally:
                try:
                    for region, callback in callbacks: region.attach_recall(callback)
                    for store, observer in graph_observers:
                        if observer is None:
                            if hasattr(store, "_recall_graph_observer"): del store._recall_graph_observer
                        else: store._recall_graph_observer = observer
                    for owner, observer in transmissions: owner._native_transmission_observer = observer
                    for owner, name, value in activity: setattr(owner, name, value)
                    for module, buffers, parameters, compute in controls:
                        module._buffers.clear(); module._buffers.update(buffers)
                        module._parameters.clear(); module._parameters.update(parameters)
                        if compute is not None: module._native_compute_device = compute
                    for module, present, compute in packed:
                        if present: module._native_compute_device = compute
                        elif hasattr(module, "_native_compute_device"): del module._native_compute_device
                    for module, original in operation["ownerObjects"].items():
                        with pager._lock:
                            if pager._owners.get(module) is not original or not original.loaded:
                                raise RuntimeError("complete cached original packed owner was replaced")
                            pager._observe_mutations(module, original)
                            old = next(device for value, device in operation["residency"] if value is module)
                            current = None if original.device is None else canonical_device(original.device)
                            expected = None if old is None else canonical_device(old)
                        if current != expected:
                            pager.evict(module)
                            if old is not None:
                                with pager.use(module, old): pass
                    pager._regional_limits()
                except BaseException as error:
                    brain._native_execution_restore_required = True
                    raise OperationAutoRestoreRequired("complete cached reference/packed restoration requires explicit Unload") from error

    def _fork(self, cache, target):
        from .model import _DecoderInferenceCache, _AttentionInferenceCache, _WorkspaceInferenceChunk
        from .connected_regions import _ConnectedInferenceCache
        from .working_attention_paging import PagedTensorSequence, SavedActivityTensor
        def tensor(value):
            # Small running/temporal state is an independent admitted copy on
            # the candidate. Paged K/V never materialize in full on a GPU.
            size = value.numel() * value.element_size()
            self.brain.resource_policy.require_device(target, size, "private complete cached running state")
            lease = self.brain.resource_policy.reserve_ram(size, "private complete cached running-state copy")
            try:
                copied = self.move_activity(value, target).detach().clone(); lease.mark_allocated(size)
                return copied, lease
            except BaseException: lease.release(); raise
        return fork_operation_cache(cache, kind=self._kind,
            activity=lambda value: value.fork() if isinstance(value, (PagedTensorSequence, SavedActivityTensor)) else None,
            tensor=tensor, workspace=_WorkspaceInferenceChunk, attention=_AttentionInferenceCache,
            ordinary=_DecoderInferenceCache, connected=_ConnectedInferenceCache,
            reserve_metadata=lambda size: self.brain.resource_policy.reserve_ram(size, "complete recursive cached COW containers"))

    def _cache_home(self, cache, target):
        stack = [cache]
        while stack:
            node = stack.pop()
            if self._kind(node) == "ordinary":
                for name in ("hidden_sum", "memory_hidden", "neural_state"):
                    value = getattr(node, name)
                    if value is not None: setattr(node, name, self.move_activity(value, target))
            else:
                node.hidden_sum = [self.move_activity(value, target) for value in node.hidden_sum]
                node.temporal = [tuple(self.move_activity(value, target) for value in state) for state in node.temporal]
                stack.extend(value for value in node.nested if value is not None)

    def _home_activity(self, closure, fork, target):
        rows = []
        for owner, name, value in self._activities(closure):
            def copy_value(item):
                if not isinstance(item, torch.Tensor): raise ValueError("complete decoder activity is not a native tensor")
                size = item.numel() * item.element_size()
                self.brain.resource_policy.require_device(target, size, "complete returned native activity")
                lease = self.brain.resource_policy.reserve_ram(size, "complete returned native activity copy")
                try:
                    copied = self.move_activity(item, target).detach().clone()
                    if not bool(torch.isfinite(copied).all()): raise ValueError("complete decoder activity is nonfinite")
                    lease.mark_allocated(size); fork.leases.append(lease)
                    return copied
                except BaseException: lease.release(); raise
            copied = tuple(copy_value(item) for item in value) if isinstance(value, tuple) else copy_value(value)
            rows.append((owner, name, copied))
        return rows

    def _activity_sha(self, activity):
        return digest([[str(id(owner)), name, [self._hash_tensor(item) for item in value] if isinstance(value, tuple)
                        else self._hash_tensor(value)] for owner, name, value in activity])

    def execute(self, candidate, operation, observer, stopped):
        from .native_token_sampling import sample_native_token
        fork = transient = None
        def stage(name):
            if observer is not None: observer.stage_completed(name)
            if stopped.is_set(): raise InterruptedError("complete cached trial stopped at a real stage boundary")
        try:
            self.admit_candidate(candidate)
            # Existing per-kernel/tile/paged-activity guards remain in force.
            # Add the complete one-frontier known working/output envelope.
            size = int(self.brain.decoder._forward_window_bytes(operation["window"].shape[0], 1))
            size += int(operation["window"].shape[0] * self.brain.config.vocab_size * 16)
            self.brain.resource_policy.require_device(candidate["executionDevice"], size, "complete cached frontier working/readout")
            if not candidate["executionDevice"].startswith("cuda:"):
                transient = self.brain.resource_policy.reserve_ram(size, "complete cached frontier transient working/readout")
            fork = self._fork(operation["cache"], candidate["executionDevice"])
            observations = {}
            requests = getattr(self.brain.core_pager, "_observation_requests", {})
            if requests:
                observed_size = 4096 + sum(int(count) * 64 + 1024 for _start, count in requests.values())
                lease = self.brain.resource_policy.reserve_ram(observed_size, "private current cortical viewport rows")
                fork.leases.append(lease); lease.mark_allocated(observed_size)
            stage(CACHED_STAGES[0])
            with self._placement(candidate, operation, stopped, observations):
                window = self.move_activity(operation["window"], candidate["executionDevice"])
                cue = None if operation["cue"] is None else self.move_activity(operation["cue"], candidate["executionDevice"])
                self._signatures(self.brain.decoder, fork.cache, cue)
                stage(CACHED_STAGES[1])
                before = self.brain.working_attention_pager._metrics["processedPrefillTokens"]
                logits, returned = self.brain.decoder._generation_step(window, cue, fork.cache)
                fork.cache = returned
                # A connected root notes each recursively executed decoder
                # frontier; every one must advance exactly one, never a prefix.
                expected = sum(1 for _path, _decoder, _cache in self._cache_nodes(self.brain.decoder, operation["cache"]))
                if returned is None or returned.length != operation["cache"].length + 1 or self.brain.working_attention_pager._metrics["processedPrefillTokens"] - before != expected:
                    raise RuntimeError("complete cached candidate replayed/refilled or lost an actual retained frontier")
                for name in CACHED_STAGES[2:5]: stage(name)
                home = operation["generator"].device
                logits = self.move_activity(logits, home)
                self._cache_home(returned, home)
                activity = self._home_activity(operation["closure"], fork, home)
                stage(CACHED_STAGES[5])
            stage(CACHED_STAGES[6])
            self._signatures(self.brain.decoder, fork.cache, operation["cue"])
            generator = torch.Generator(device=operation["generator"].device); generator.set_state(operation["rng"])
            sample = sample_native_token(logits, generator, **operation["sampling"])
            rng = generator.get_state().clone()
            stage(CACHED_STAGES[7])
            self.synchronize(candidate["asyncDevices"])
            contract = digest({"cache": self.cache_sha(fork.cache), "activity": self._activity_sha(activity), "sampling": self._sample_sha(sample, rng)})
            return CompleteCachedTrial(fork, sample, rng, activity, contract, observations)
        except BaseException:
            if fork is not None: fork.close()
            raise
        finally:
            if transient is not None: transient.release()

    def output_descriptor(self, trial):
        return {"cacheKind": self._kind(trial.cache), "activity": [[name,
            [[list(item.shape), str(item.dtype)] for item in value] if isinstance(value, tuple) else [list(value.shape), str(value.dtype)]]
            for _owner, name, value in trial.activity], "token": [list(trial.sample[0].shape), str(trial.sample[0].dtype)]}

    def check_output(self, trial, descriptor):
        if self.output_descriptor(trial) != descriptor: raise ValueError("fresh cached inference output geometry changed")
        token, log_probability, entropy = trial.sample
        if not math.isfinite(log_probability) or not math.isfinite(entropy) or not bool(torch.isfinite(trial.neural_state).all()):
            raise ValueError("fresh cached output is nonfinite")

    def publish_activity(self, trial):
        for owner, name, value in trial.activity: setattr(owner, name, value)
        pager, rows = self.brain.core_pager, trial.observations
        if (isinstance(rows, dict) and "rows" in rows and rows["requests"] == pager._observation_requests
                and rows["scopes"] == pager._observation_scopes):
            # Candidate/warmup rows never touch the live viewer. Only the
            # accepted real computation's observed viewport becomes visible.
            pager._observations = {**pager._observations, **rows["rows"]}

    def restore_published_activity(self, operation):
        for owner, name, value in operation["originalActivity"]: setattr(owner, name, value)
        original = operation.get("originalCorticalObservations")
        if original is not None: self.brain.core_pager._observations = original

    def commit_sampling(self, operation, trial):
        if not torch.equal(operation["generator"].get_state(), operation["rng"]): return None
        operation["generator"].set_state(trial.rng_state)
        # Signature was bound after private restoration BEFORE the real world
        # boundary. Never relabel old KV current if that boundary learned.
        return trial.sample
