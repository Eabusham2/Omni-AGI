"""Actual eval/no-grad decoder full-forward callbacks on existing native owners.

The complete output window retains its ordinary admission. Trials own only
activity and read-only placement; all learned signs/strengths remain shared.
"""
from __future__ import annotations

from contextlib import ExitStack

import torch

from .cached_activity_snapshot import CacheFork
from .cached_whole_operation_native import NativeCachedWholeOperationOps
from .decoder_execution_native import _tensor_guard
from .forward_whole_operation_auto import FORWARD_NUMERIC_GATE, FORWARD_REUSE_BASIS
from .operation_auto_registry import FORWARD_KINDS, FORWARD_STAGES
from .operation_auto_policy import OperationAutoRestoreRequired
from .regional_execution import canonical_device
from .whole_path_execution import digest


class CompleteForwardTrial:
    def __init__(self, output, activity, contract, lifetime, observations):
        self.output, self.activity, self.contract = output, activity, contract
        self.lifetime, self.observations = lifetime, observations
    def take(self):
        output, self.output = self.output, None
        return output
    def close(self):
        self.output = self.activity = self.observations = None
        self.lifetime.close()


class NativeFullForwardOperationOps(NativeCachedWholeOperationOps):
    def _closure(self):
        from . import model as m, connected_regions as c
        closure = super()._closure()
        # The cached closure deliberately excludes action readout and borrowed
        # full-forward wrappers. This entry actually executes those owners.
        modules, names = closure["modules"], closure["names"]
        extra, borrowed = [], []
        for path, decoder in closure["decoders"]:
            if (any(name in decoder.__dict__ for name in ("_forward_bounded", "_working_pager",
                    "maximum_forward_tokens", "_forward_window_bytes"))
                    or self._has_hooks(decoder)):
                raise ValueError("unreviewed decoder full-forward override or callback")
            # Include decoder-owned controls, without walking unexecuted tool,
            # argument, imagination or internal-action subtrees.
            extra.append((path, decoder))
            if type(decoder) is m.OmniDecoder:
                if type(decoder.action_policy) is not m.ActionPolicyHead:
                    raise ValueError("full forward requires the actual native action policy")
                for suffix, module in decoder.action_policy.named_modules():
                    extra.append((path + ".action_policy" + ("." + suffix if suffix else ""), module))
                continue
            expected = {"embedding": c._JoinedEmbedding, "final_norm": c._JoinedNorm,
                "language_head": c._JoinedHead, "action_policy": c._JoinedHead,
                "global_workspace": c._JoinedWorkspace, "memory_projection": c._JoinedMemoryProjection,
                "workspace_strength": c._BorrowedGains, "memory_strength": c._BorrowedGains,
                "blocks": c._BorrowedSequence, "experts": c._BorrowedSequence,
                "expert_prototypes": c._BorrowedSequence}
            regions, widths = tuple(decoder.regions), tuple(decoder.widths)
            if widths != tuple(int(region.config.d_model) for region in regions):
                raise ValueError("connected full-forward widths no longer match actual owners")
            for field, kind in expected.items():
                wrapper = getattr(decoder, field)
                if type(wrapper) is not kind or self._has_hooks(wrapper) or "forward" in vars(wrapper):
                    raise ValueError("unknown or overridden borrowed full-forward wrapper")
                if kind is c._BorrowedSequence:
                    actual = tuple(value for region in regions for value in getattr(region.decoder, field))
                    if tuple(wrapper.values) != actual:
                        raise ValueError("borrowed sequence differs from its connected learned owners")
                    refs = [str(id(value)) for value in actual]
                else:
                    if tuple(wrapper.regions) != regions or tuple(wrapper.widths) != widths:
                        raise ValueError("borrowed wrapper differs from its connected owners")
                    if kind in {c._JoinedHead, c._BorrowedGains} and wrapper.field != field:
                        raise ValueError("borrowed full-forward field points at another operation")
                    if kind is c._JoinedHead and tuple(wrapper._borrowed_learning_heads) != tuple(
                            getattr(region.decoder, field) for region in regions):
                        raise ValueError("borrowed action/language readout owners changed")
                    if kind is c._JoinedWorkspace and wrapper.exchange is not decoder.workspace_exchange:
                        raise ValueError("borrowed workspace exchange changed")
                    refs = [str(id(region)) for region in regions]
                borrowed.append([path, field, str(id(wrapper)), widths, refs,
                    getattr(wrapper, "field", None), str(id(getattr(wrapper, "exchange", None)))])
                extra.append((path + "." + field, wrapper))
        for key, region in closure["regions"]:
            # condition() reads this real initial recurrent buffer. It is not
            # a registered child of the liquid controller and must not vanish
            # from state hashing or selected control placement.
            if (set(region._buffers) != {"liquid_state"} or region._parameters
                    or not isinstance(region.liquid_state, torch.Tensor)
                    or region.liquid_state.dtype != torch.float32
                    or tuple(region.liquid_state.shape) != (1, region.config.idea_dim)
                    or self._has_hooks(region)):
                raise ValueError("connected regional liquid initial state is not declared")
            extra.append((key, region))
        allowed = {m.OmniDecoder, c.ConnectedOmniDecoder, c.NativeRegion, m.ActionPolicyHead,
            m.RMSNorm, m.PackedAdaptiveBitLinear, m.PackedAdaptiveTernaryEmbedding,
            c._JoinedEmbedding, c._JoinedNorm, c._JoinedHead, c._JoinedWorkspace,
            c._JoinedMemoryProjection, c._BorrowedGains, c._BorrowedSequence}
        packed_types = {m.PackedAdaptiveBitLinear, m.PackedAdaptiveTernaryEmbedding}
        from .packed_connections import PackedContactPage, PooledParallelConnections
        allowed.update({PackedContactPage, PooledParallelConnections, torch.nn.ModuleDict})
        packed_types.add(PackedContactPage)
        for name, module in extra:
            if (type(module) not in allowed or "forward" in vars(module) or self._has_hooks(module)
                    or getattr(module, "_online_transaction", None) is not None):
                raise ValueError("unreviewed full-forward action/control implementation")
            if type(module) in packed_types:
                if (getattr(module, "_native_core_pager", None) is not self.brain.core_pager
                        or module not in self.brain.core_pager._owners
                        or not self.brain.core_pager._owners[module].loaded):
                    raise ValueError("full-forward action readout has an unowned packed connection")
                if module not in closure["packed"]: closure["packed"].append(module)
            modules[id(module)] = module
            names.setdefault(id(module), name)
        closure["borrowedForwardBindings"] = borrowed
        closure["kind"] = FORWARD_KINDS[int(type(self.brain.decoder) is c.ConnectedOmniDecoder)]
        for module in modules.values():
            if any(name in vars(module) for name in ("_validated_attention_mask", "_latent_rows", "_attend",
                    "summarize", "_packed_residency_scope", "packed_forward_weight", "effective_weighted_bias")):
                raise ValueError("unreviewed full-forward helper override cannot be privately replayed")
        return closure

    @staticmethod
    def _has_hooks(module):
        return any(getattr(module, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))

    def _registry(self):
        from .model import ActionPolicyHead
        base, pager = super()._registry()
        closure = self._closure()
        heads = [[closure["names"][key], module.hidden_width, module.logit_gain]
            for key, module in closure["modules"].items() if type(module) is ActionPolicyHead]
        return digest([base, closure["borrowedForwardBindings"], heads]), pager

    def _activities(self, closure):
        # Full forward does not set last_generation_neural_state, create a KV
        # cursor or sample tokens. Preserve exactly its real connected effect.
        from .connected_regions import ConnectedOmniDecoder
        return [(decoder, "last_region_activity", decoder.last_region_activity)
            for _path, decoder in closure["decoders"] if type(decoder) is ConnectedOmniDecoder]

    def _geometry(self, window, cue, use_global_workspace, attention_mask, closure):
        pager = self.brain.working_attention_pager
        rows = []
        from .connected_regions import ConnectedOmniDecoder
        for path, decoder in closure["decoders"]:
            rows.append([path, type(decoder).__name__, int(decoder.config.max_seq_len),
                int(decoder._forward_window_bytes(*window.shape)),
                int(decoder.maximum_forward_tokens(window.shape[0])),
                int(decoder.global_workspace.slots), int(decoder.global_workspace.query_chunk_slots),
                [] if type(decoder) is ConnectedOmniDecoder else [[int(block.attention.n_heads),
                    int(block.attention.head_dim), int(block.attention.query_chunk_tokens),
                    int(block.attention.key_chunk_tokens)] for block in decoder.blocks]])
        def tensor_layout(value):
            return None if value is None else [list(value.shape), str(value.dtype), list(value.stride())]
        return {"input": tensor_layout(window), "cue": tensor_layout(cue), "mask": tensor_layout(attention_mask),
            "wholeWorkspace": use_global_workspace, "owners": rows,
            "pageTokens": int(pager.page_tokens), "tileBudgetBytes": int(pager.device_tile_budget_bytes),
            "residentBudgetBytes": int(pager.resident_budget_bytes), "scratchBudgetBytes": int(pager.scratch_budget_bytes)}

    def valid_forward(self, window, cue, use_global_workspace, attention_mask):
        try:
            decoder = self.brain.decoder
            if (torch.is_grad_enabled() or decoder.training or not isinstance(window, torch.Tensor)
                    or window.dtype != torch.long or window.ndim != 2 or window.shape[0] < 1
                    or not 1 <= window.shape[1] <= decoder.config.max_seq_len
                    or canonical_device(window.device) != canonical_device(self.brain.device)
                    or use_global_workspace is not None and type(use_global_workspace) is not bool):
                return False
            if cue is not None and (not isinstance(cue, torch.Tensor) or cue.dtype != torch.float32
                    or cue.ndim not in {1, 2} or cue.shape[-1] != decoder.config.idea_dim
                    or cue.ndim == 2 and cue.shape[0] not in {1, window.shape[0]}
                    or canonical_device(cue.device) != canonical_device(window.device)):
                return False
            if attention_mask is not None and (not isinstance(attention_mask, torch.Tensor)
                    or attention_mask.shape != window.shape
                    or canonical_device(attention_mask.device) != canonical_device(window.device)):
                return False
            if (self.brain.core_pager.status()["pinnedOwners"]
                    or getattr(getattr(self.brain, "_regional_execution", None), "active", False)):
                return False
            for name in ("_decoder_execution", "_cached_decoder_execution"):
                controller = getattr(self.brain, name, None)
                if controller is not None and controller.status()["phase"] not in {"idle", "cancelled", "failed", "unsupported", "completed"}:
                    return False
            closure = self._closure()
            if any(module.training for module in closure["modules"].values()): return False
            if any(getattr(region.router.synapses, "_native_transmission_observer", None) is not None
                    for _key, region in closure["regions"]): return False
            self._geometry(window, cue, use_global_workspace, attention_mask, closure)
            return True
        except (ValueError, RuntimeError, AttributeError, IndexError): return False

    @staticmethod
    def _rng_frontiers(devices):
        # Read actual process frontiers. There is no seed translation, reset,
        # draw or set_rng_state in this declared deterministic eval operation.
        rows = {"cpu": torch.get_rng_state()}
        for name in sorted(set(devices) - {"cpu"}):
            device = torch.device(name)
            if device.type == "cuda": rows[name] = torch.cuda.get_rng_state(device)
            elif device.type == "mps": rows[name] = torch.mps.get_rng_state()
            else: raise ValueError("full-forward RNG frontier is not supported for this backend")
        return rows

    def _restore_required(self, message):
        self.brain._native_execution_restore_required = True
        raise OperationAutoRestoreRequired(message + "; explicit Unload required")

    def require_original_frontiers(self, operation, session):
        try:
            with self.proof_scope():
                actual = self._rng_frontiers(operation["rng"])
                if (actual.keys() != operation["rng"].keys()
                        or any(not torch.equal(actual[key], value) for key, value in operation["rng"].items())):
                    self._restore_required("deterministic full forward changed an original RNG frontier")
                if self.operation_state(operation, session) != operation["identity"]["nativeState"]:
                    self._restore_required("private full forward changed an authoritative native control or learned owner")
        except OperationAutoRestoreRequired: raise
        except Exception as error:
            self.brain._native_execution_restore_required = True
            raise OperationAutoRestoreRequired("original full-forward frontier cannot be proven; explicit Unload required") from error

    def capture_forward(self, window, cue, use_global_workspace, attention_mask, session, review, *, stopped):
        closure, stack = self._closure(), ExitStack()
        try:
            pagers = {id(region.router.synapses._router_state_pager): region.router.synapses._router_state_pager
                for _key, region in closure["regions"]}
            for pager in pagers.values(): stack.enter_context(pager.operation())
            scope = self._scope(closure)
            devices = {canonical_device(self.brain.device), *(canonical_device(row["executionDevice"]) for row in review["candidates"])}
            size = 131072 + len(closure["modules"]) * 1024 + len(devices) * 32768
            # Holding the source also holds any larger backing storage view.
            storages = {(_tensor_guard(value)[2], _tensor_guard(value)[-1]): int(value.untyped_storage().nbytes())
                for value in (window, cue, attention_mask) if value is not None}
            size += sum(storages.values())
            lease = self.brain.resource_policy.reserve_ram(size, "uncached full-forward input, owner and RNG frontier custody")
            stack.callback(lease.release)
            with self.proof_scope(stopped):
                guards = tuple(_tensor_guard(value) for value in (window, cue, attention_mask))
                geometry = self._geometry(window, cue, use_global_workspace, attention_mask, closure)
                geometry_sha = digest([scope["registry"], geometry])
                rng, state, binding = self._rng_frontiers(devices), self._state(closure, scope, session), review["binding"]
                identity = {"ownerId": self.brain.brain_id, "ownerSessionId": session, "workKind": closure["kind"],
                    "stageCoverage": list(FORWARD_STAGES), "geometrySha256": geometry_sha,
                    "workloadSha256": digest([*[self._hash_tensor(value) for value in (window, cue, attention_mask)],
                        [[key, self._hash_tensor(value)] for key, value in rng.items()]]),
                    "samplingContractSha256": digest("deterministic-eval-full-forward-no-sampling-or-RNG-draw"),
                    **{key: binding[key] for key in ("policySha256", "topologySha256", "budgetSha256")},
                    "cache": {"heat": "cold", "logicalStateSha256": digest({"retainedKV": None,
                        "operation": "uncached-full-forward"}), "backendInitialization": "warm"}, "nativeState": state}
                lease.mark_allocated(size)
            return {"window": window, "cue": cue, "attention_mask": attention_mask, "use_global_workspace": use_global_workspace,
                "rng": rng, "guards": guards, "closure": closure, "scope": scope, "stack": stack,
                "identity": identity, "forwardGeometry": geometry, "session": session,
                "originalActivity": self._activities(closure),
                "originalCorticalObservations": getattr(self.brain.core_pager, "_observations", None),
                "residency": [(module, self.brain.core_pager._owners[module].device) for module in closure["packed"]],
                "ownerObjects": {module: self.brain.core_pager._owners[module] for module in closure["packed"]},
                "stratumKey": digest([closure["kind"], identity["workloadSha256"], geometry_sha]),
                "planContract": {"forwardLayout": {"shapeSha256": geometry_sha},
                    "numericGate": FORWARD_NUMERIC_GATE, "reuseBasis": FORWARD_REUSE_BASIS}}
        except BaseException: stack.close(); raise

    def forward_unchanged(self, operation, session):
        with self.proof_scope():
            return (operation["guards"] == tuple(_tensor_guard(operation[name]) for name in ("window", "cue", "attention_mask"))
                and self.operation_state(operation, session) == operation["identity"]["nativeState"]
                and self._geometry(operation["window"], operation["cue"], operation["use_global_workspace"],
                    operation["attention_mask"], operation["closure"]) == operation["forwardGeometry"])

    def release_forward(self, operation): operation["stack"].close()

    def _returned_home(self, output, activity, home, lifetime):
        # Preserve actual output/activity view aliases and strides. Returning
        # connected activity must not turn it into a separate generation state.
        storage_copies = {}
        def move(value):
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError("full-forward output/activity must be finite native float32")
            storage = value.untyped_storage()
            key = (str(value.device), int(storage.data_ptr()), int(storage.nbytes()), str(value.dtype))
            if key not in storage_copies:
                size = int(storage.nbytes())
                self.brain.resource_policy.require_device(home, size, "complete full-forward returned storage")
                lease = self.brain.resource_policy.reserve_ram(size, "retained complete full-forward output/activity storage")
                lifetime.leases.append(lease)
                flat = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                    storage, 0, (size // value.element_size(),), (1,))
                copied = self.move_activity(flat, home)
                lease.mark_allocated(size)
                storage_copies[key] = copied
            return storage_copies[key].as_strided(value.shape, value.stride(), value.storage_offset())
        returned = {key: move(value) for key, value in output.items()}
        rows = [(owner, name, tuple(move(value) for value in values)) for owner, name, values in activity]
        return returned, rows

    def execute(self, candidate, operation, observer, stopped):
        lifetime, transient = CacheFork(), None
        def stage(name):
            if observer is not None: observer.stage_completed(name)
            if stopped.is_set(): raise InterruptedError("private full forward stopped at a real boundary")
        try:
            self.require_original_frontiers(operation, operation["session"])
            self.admit_candidate(candidate)
            size = int(self.brain.decoder._forward_window_bytes(*operation["window"].shape))
            self.brain.resource_policy.require_device(candidate["executionDevice"], size, "complete full-output forward")
            if not candidate["executionDevice"].startswith("cuda:"):
                transient = self.brain.resource_policy.reserve_ram(size, "complete full-output transient window")
            observations = {}
            requests = getattr(self.brain.core_pager, "_observation_requests", {})
            if requests:
                observed_size = 4096 + sum(int(count) * 64 + 1024 for _start, count in requests.values())
                lease = self.brain.resource_policy.reserve_ram(observed_size, "private full-forward viewport rows")
                lifetime.leases.append(lease); lease.mark_allocated(observed_size)
            stage(FORWARD_STAGES[0])
            try:
                with self._placement(candidate, operation, stopped, observations):
                    window = self.move_activity(operation["window"], candidate["executionDevice"])
                    cue = None if operation["cue"] is None else self.move_activity(operation["cue"], candidate["executionDevice"])
                    mask = None if operation["attention_mask"] is None else self.move_activity(operation["attention_mask"], candidate["executionDevice"])
                    stage(FORWARD_STAGES[1])
                    # Actual public entry, including unchanged full-window
                    # pager admission. PRIVATE_OPERATION prevents recursion.
                    output = self.brain.decoder.forward(window, memory_bias=cue, labels=None,
                        use_global_workspace=operation["use_global_workspace"], attention_mask=mask)
                    self._validate_output(output, operation)
                    for name in FORWARD_STAGES[2:5]: stage(name)
                    returned, activity = self._returned_home(output, self._activities(operation["closure"]),
                        self.brain.device, lifetime)
                    stage(FORWARD_STAGES[5])
            finally:
                # A declared no-draw eval body that changed RNG/native state
                # cannot safely be repeated as an ordinary fallback. Do not
                # reset/reseed frontiers or relabel a changed epoch as current.
                self.require_original_frontiers(operation, operation["session"])
            stage(FORWARD_STAGES[6])
            self.synchronize(candidate["asyncDevices"])
            contract = digest({"outputs": [[key, self._hash_tensor(value), list(value.stride())]
                for key, value in sorted(returned.items())], "activity": self._activity_sha(activity),
                "rng": [[key, self._hash_tensor(value)] for key, value in operation["rng"].items()]})
            return CompleteForwardTrial(returned, activity, contract, lifetime, observations)
        except BaseException: lifetime.close(); raise
        finally:
            if transient is not None: transient.release()

    def _validate_output(self, output, operation):
        from .model import ACTION_KINDS
        decoder, (batch, length) = self.brain.decoder, operation["window"].shape
        shapes = {"logits": (batch, length, decoder.config.vocab_size),
            "hidden": (batch, length, decoder.config.d_model), "action_logits": (batch, len(ACTION_KINDS))}
        if operation["use_global_workspace"] is True:
            shapes["workspace"] = (batch, 1 if operation["closure"]["regions"] else decoder.global_workspace.slots, decoder.config.d_model)
        if not operation["closure"]["regions"] and decoder.expert_count:
            shapes["expert_routing"] = (batch, decoder.expert_count)
        if not isinstance(output, dict) or output.keys() != shapes.keys():
            raise ValueError("actual full-forward output keys are incomplete or undeclared")
        for key, shape in shapes.items():
            value = output[key]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError("actual full-forward tensor output has a different shape, dtype or finite contract")

    def output_descriptor(self, trial):
        return {"outputs": [[key, list(value.shape), str(value.dtype), list(value.stride())]
            for key, value in sorted(trial.output.items())], "activity": [[name,
                [[list(value.shape), str(value.dtype), list(value.stride())] for value in values]]
                for _owner, name, values in trial.activity]}
