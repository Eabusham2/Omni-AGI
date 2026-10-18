"""Scoped, same-brain perception placement; import starts no neural operation.

Only the declared pure encoder/connected workspace closure is eligible. Its
source preprocessing, lasting learning, recurrent state and callbacks are not
repeated. Exact qualification of one input supports a current operator/geometry
route inference for later inputs, not a promise of equal future measured speed.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from contextlib import contextmanager, nullcontext

import torch

from .decoder_execution_native import NativeDecoderExecutionOps, _tensor_guard
from .native_compute_profile import native_projection_tensor_budget
from .operation_auto_policy import OperationAutoPolicy, OperationAutoRestoreRequired, STAGES, _BYPASS, bind_operation_auto
from .operation_auto_registry import PERCEPTION_KINDS
from .parameter_diagnostics import native_parameter_read_only
from .regional_execution import _ACTIVE_REGIONAL, canonical_device, topology_identity, transient_regional_scope
from .whole_path_execution import digest


class PerceptionOutput:
    def __init__(self, value, lease, contract):
        self.value, self.lease, self.contract = value, lease, contract
    def take(self):
        value, self.value = self.value, None
        return value
    def close(self):
        self.value = None
        if self.lease is not None: self.lease.release(); self.lease = None


class _PerceptionTarget:
    def __init__(self, ops, candidate, stopped):
        self.ops, self.candidate, self.stopped = ops, candidate, stopped
        self._policy, self._lease, self.active = ops.brain.resource_policy, {"epoch": uuid.uuid4().hex}, True
    def policy_identity(self): return self._policy
    def packed_limits(self, policy):
        if not self.active or policy is not self._policy or self.stopped.is_set():
            raise InterruptedError("perception packed placement authority was retired")
        result = {}
        for name in {self.candidate["executionDevice"], canonical_device(self.ops.brain.device), *self.candidate.get("asyncDevices", ())} - {"cpu"}:
            actual = next((row for row in self.ops.brain._regional_topology()["devices"]
                           if canonical_device(row.get("executionDevice", "")) == name), None)
            if actual is None: raise RuntimeError("perception compute interface disappeared")
            self.ops.admit_candidate({"executionDevice": name})
            hot = self.ops.brain._native_core_budget_for_device(self.ops.brain.core_pager.status(), torch.device(name))
            result[name] = {"memoryGroupId": actual["memoryGroupId"], "hotBytes": int(hot["acceleratorCoreHotBytes"])}
        return result
    def execute(self, *_args, **_kwargs):
        raise RuntimeError("perception Auto cannot route or repeat a decoder operation")


class NativeOperationAutoOps(NativeDecoderExecutionOps):
    def __init__(self, brain):
        super().__init__(brain)
        self._scopes, self._stop_event = {}, None
        from .cached_whole_operation_native import NativeCachedWholeOperationOps
        self.cached_operations = NativeCachedWholeOperationOps(brain, self)
        from .prefill_whole_operation_native import NativeFullPrefixOperationOps
        self.prefill_operations = NativeFullPrefixOperationOps(brain, self)
        from .forward_whole_operation_native import NativeFullForwardOperationOps
        self.forward_operations = NativeFullForwardOperationOps(brain, self)
        from .media_whole_operation_native import NativeWholeMediaOperationOps
        self.media_operations = NativeWholeMediaOperationOps(brain, self)
        from .training_whole_operation import NativeTrainingWaveOps
        self.training_operations = NativeTrainingWaveOps(brain, self)
        from .whole_path_native import NativeFixedMemoryOps
        from .whole_path_scheduler import WholePathScheduler
        self.whole_path_scheduler = WholePathScheduler(NativeFixedMemoryOps(brain, self))

    def _metadata_reservation(self):
        return self.brain.resource_policy.reserve_ram(max(65536, len(self.brain.core_pager._owners) * 8192),
                                                     "declared perception registry traversal metadata")

    def _hash_tensor(self, value):
        if value is None: return None
        hasher = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        block = max(1, 262144 // value.element_size())
        layout = (nullcontext() if value.is_contiguous() else self.brain.resource_policy.reserve_ram(
            value.numel() * value.element_size(), "exact perception checksum layout"))
        with layout, self.brain.resource_policy.reserve_ram(min(value.numel(), block) * value.element_size() * 3,
                                                         "bounded actual perception checksum"):
            flat, version = value.detach().reshape(-1), int(value._version)
            for start in range(0, value.numel(), block):
                if self._stop_event is not None and self._stop_event.is_set() or self.parent_cancelled():
                    raise InterruptedError("private perception checksum stopped")
                payload = flat[start:start + block].to(device="cpu", non_blocking=False).contiguous().view(torch.uint8).numpy()
                hasher.update(memoryview(payload))
            if int(value._version) != version: raise RuntimeError("actual perception tensor changed while hashing")
        return hasher.hexdigest()

    def _closure(self, hub, modality):
        from .modalities import ModalityHub, TinyImageImagination, TinyAudioCodec, TinyVideoImagination
        from .connected_regions import ConnectedModalityHub
        if modality not in {"image", "audio", "video"}: raise ValueError("unknown pure perception operation")
        roots, identities, seen = [], [], set()
        def visit(value, path):
            if id(value) in seen: raise RuntimeError("cyclic connected perception hub")
            seen.add(id(value))
            if type(value) is ConnectedModalityHub:
                if any(name in value.__dict__ for name in ("perception_embedding", "_resize")):
                    raise RuntimeError("connected perception override is not reviewed")
                decoder = value.connected_decoder
                children = []
                for index, region in enumerate(decoder.regions):
                    identity = region.identity
                    identities.append([path, index, identity.region_id, identity.parent_id, identity.generation_sha256,
                                       region.config.to_dict()])
                    children.append(visit(region.modalities, path + ".r" + str(index)))
                roots.append((path + ".workspace_exchange", decoder.workspace_exchange))
                seen.remove(id(value))
                return {"kind": "connected", "hub": value, "children": children, "exchange": decoder.workspace_exchange}
            if type(value) is not ModalityHub: raise RuntimeError("unknown perception implementation is not a pure admitted closure")
            if "perception_embedding" in value.__dict__: raise RuntimeError("instance perception override is not reviewed")
            if modality == "image" and value.config.vision_enabled:
                selected = [(path + ".vision", value.vision)]
                encoder, projection = value.vision.features, value.vision.projection
            else:
                if not getattr(value.config, modality + "_enabled", False): raise ValueError("perception pack is disabled")
                pack = getattr(value, modality)
                if type(pack) is not {"image": TinyImageImagination, "audio": TinyAudioCodec, "video": TinyVideoImagination}[modality] or "encode" in pack.__dict__:
                    raise RuntimeError("unknown or overridden sensory encoder is not a pure declared closure")
                selected = [(path + "." + modality + ".encoder", getattr(value, modality).encoder),
                            (path + "." + modality + ".encoder_projection", getattr(value, modality).encoder_projection)]
                encoder, projection = getattr(value, modality).encoder, getattr(value, modality).encoder_projection
            roots.extend(selected)
            identities.append([path, modality, value.config.to_dict()])
            seen.remove(id(value))
            return {"kind": "leaf", "hub": value, "encoder": encoder, "projection": projection}
        tree = visit(hub, "modalities")
        modules, names = {}, {}
        for path, root in roots:
            for suffix, module in root.named_modules():
                modules[id(module)] = module
                names.setdefault(id(module), path + ("." + suffix if suffix else ""))
        return {"tree": tree, "roots": roots, "modules": modules, "names": names, "identities": identities}

    def _validate_closure(self, closure):
        from .modalities import TinyVisionEncoder
        from .connected_regions import RegionExchange
        from .model import PackedAdaptiveBitConv1d, PackedAdaptiveBitConv2d, PackedAdaptiveBitConv3d, PackedAdaptiveBitLinear
        from .packed_connections import PackedContactPage, PooledParallelConnections
        allowed = {TinyVisionEncoder, RegionExchange, PackedAdaptiveBitConv1d, PackedAdaptiveBitConv2d,
            PackedAdaptiveBitConv3d, PackedAdaptiveBitLinear, PackedContactPage, PooledParallelConnections,
            torch.nn.Sequential, torch.nn.ModuleDict, torch.nn.SiLU, torch.nn.AdaptiveAvgPool2d}
        globals_ = torch.nn.modules.module
        if any(getattr(globals_, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks")):
            raise RuntimeError("global module hooks prevent a pure perception comparison")
        for module in closure["modules"].values():
            if (type(module) not in allowed or "forward" in module.__dict__ or any(getattr(module, name, {})
                    for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))):
                raise RuntimeError("perception closure has an unreviewed class, override or callback")
            if type(module) is torch.nn.SiLU and module.inplace:
                raise RuntimeError("in-place perception activity cannot be privately compared")
            if getattr(module, "_online_transaction", None) is not None: raise RuntimeError("perception learned owner is in a transaction")
        pager = self.brain.core_pager
        projected = {PackedAdaptiveBitConv1d, PackedAdaptiveBitConv2d, PackedAdaptiveBitConv3d, PackedAdaptiveBitLinear, PackedContactPage}
        packed = [value for value in closure["modules"].values() if type(value) in projected]
        if not packed or any(getattr(value, "_native_core_pager", None) is not pager or value not in pager._owners or not pager._owners[value].loaded for value in packed):
            raise RuntimeError("complete loaded packed perception owners are required")
        return packed

    def _closure_sha(self, closure):
        hasher = hashlib.sha256()
        hasher.update(json.dumps(closure["identities"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        for key, module in closure["modules"].items():
            fields = set(getattr(module, "_packed_structural_fields", ())) | {"in_features", "out_features", "dimensions", "in_channels", "out_channels",
                "kernel_size", "stride", "padding", "dilation", "groups", "padding_mode", "widths", "output_size", "inplace"}
            structural = {name: getattr(module, name) for name in sorted(fields)
                if type(getattr(module, name, None)) in {int, float, str, bool, tuple, list}}
            row = [closure["names"][key], str(key), type(module).__module__ + "." + type(module).__name__, structural,
                [[name, list(value.shape), str(value.dtype)] for name, value in module._buffers.items()
                 if isinstance(value, torch.Tensor) and name not in module._non_persistent_buffers_set]]
            encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            hasher.update(len(encoded).to_bytes(8, "little")); hasher.update(encoded)
        return hasher.hexdigest()

    def _registry(self):
        # Actual selected owners, including borrowed connected parent hubs and
        # learned workspace bridges absent from hub.named_modules().
        with self._metadata_reservation():
            rows = []
            for modality in ("image", "audio", "video"):
                try:
                    closure = self._closure(self.brain.modalities, modality)
                    self._validate_closure(closure); rows.append([modality, self._closure_sha(closure)])
                except ValueError: pass  # An explicitly disabled pack is not an available work kind.
            if self.cached_operations.work_kinds():
                rows.append(["complete-cached-decoder", self.cached_operations._registry()[0]])
            from .operation_auto_registry import STAGE_COVERAGE
            from .operation_auto_registry import PREFILL_KINDS
            if any(kind in STAGE_COVERAGE for kind in PREFILL_KINDS) and self.prefill_operations.work_kinds():
                rows.append(["complete-native-full-prefix", self.prefill_operations._registry()[0]])
            from .operation_auto_registry import FORWARD_KINDS
            if any(kind in STAGE_COVERAGE for kind in FORWARD_KINDS) and self.forward_operations.work_kinds():
                rows.append(["complete-native-uncached-full-forward", self.forward_operations._registry()[0]])
            from .media_whole_operation_native import MEDIA_KINDS
            if any(kind in STAGE_COVERAGE for kind in MEDIA_KINDS):
                media_registry, media_rows = self.media_operations.registry()
                if media_rows: rows.append(["complete-native-media", media_registry])
            from .training_whole_operation import TRAINING_KIND
            if TRAINING_KIND in STAGE_COVERAGE and self.training_operations.work_kinds():
                rows.append(["complete-prepared-native-learner-wave", self.training_operations.registry()])
            if not rows: raise RuntimeError("no completely declared native Auto operation is available")
            return digest(rows), self.brain.core_pager

    def supported(self):
        if getattr(self.brain, "_native_execution_restore_required", False): return False, "native-owner-restore-requires-explicit-unload", False
        if not getattr(self.brain, "_ground_up_action_origin_verified", False): return False, "verified-current-native-origin-required", False
        if self.brain.resource_policy.device_resources is None: return False, "current-per-device-resource-settings-required", False
        if getattr(getattr(self.brain, "_regional_execution", None), "active", False): return False, "manual-regional-policy-is-separate", False
        for name in ("_decoder_execution", "_cached_decoder_execution"):
            value = getattr(self.brain, name, None)
            if value is not None and value.status()["phase"] not in {"idle", "cancelled", "failed", "unsupported", "completed"}:
                return False, "stop-the-separate-decoder-review-before-perception-auto", False
        from .connected_regions import ConnectedModalityHub
        self._registry()
        return True, "current-native-operation-closures-supported-not-whole-brain", type(self.brain.modalities) is ConnectedModalityHub

    def work_kinds(self):
        result = []
        with self._metadata_reservation():
            for modality, kind in zip(("image", "audio", "video"), PERCEPTION_KINDS):
                try: self._validate_closure(self._closure(self.brain.modalities, modality)); result.append(kind)
                except ValueError: pass
            result.extend(self.cached_operations.work_kinds())
            # The main/shared typed contract is explicitly composed before
            # these work kinds are allowed onto the reviewed public surface.
            from .operation_auto_registry import STAGE_COVERAGE
            from .operation_auto_registry import PREFILL_KINDS
            if any(kind in STAGE_COVERAGE for kind in PREFILL_KINDS):
                result.extend(kind for kind in self.prefill_operations.work_kinds() if kind in STAGE_COVERAGE)
            from .operation_auto_registry import FORWARD_KINDS
            if any(kind in STAGE_COVERAGE for kind in FORWARD_KINDS):
                result.extend(kind for kind in self.forward_operations.work_kinds() if kind in STAGE_COVERAGE)
            from .media_whole_operation_native import MEDIA_KINDS
            if any(kind in STAGE_COVERAGE for kind in MEDIA_KINDS):
                result.extend(kind for kind in self.media_operations.work_kinds() if kind in STAGE_COVERAGE)
            from .training_whole_operation import TRAINING_KIND
            if TRAINING_KIND in STAGE_COVERAGE:
                result.extend(self.training_operations.work_kinds())
        return result

    def quiescent(self):
        self.brain._regional_quiescent()
        supported, reason, _connected = self.supported()
        if not supported: raise RuntimeError(reason)

    def hardware_binding(self, session, *, topology=None):
        actual = digest(topology_identity(self.brain._regional_topology()))
        if topology is not None and digest(topology_identity(topology)) != actual: raise RuntimeError("perception main topology is stale")
        registry, _pager = self._registry()
        return {"ownerSessionId": session, "policySha256": digest({"resource": self.brain._regional_policy_sha(),
                    "configuration": self.brain.config.to_dict()}), "topologySha256": actual, "registrySha256": registry,
                "budgetSha256": digest(self._budgets()), "homeExecutionDevice": canonical_device(self.brain.device)}

    def eligible(self, hub, modality, tensor):
        try:
            if hub is not self.brain.modalities or torch.is_grad_enabled() or not isinstance(tensor, torch.Tensor): return False
            if canonical_device(tensor.device) != canonical_device(self.brain.device) or tensor.dtype != torch.float32: return False
            if any(int(size) < 1 for size in tensor.shape): return False
            if (modality == "image" and (tensor.ndim != 4 or tensor.shape[1] != 3) or
                    modality == "video" and (tensor.ndim != 5 or tensor.shape[1] != 3) or
                    modality == "audio" and (tensor.ndim not in {2, 3} or tensor.ndim == 3 and tensor.shape[1] != 1)):
                return False
            self.quiescent()
            with self._metadata_reservation(): self._validate_closure(self._closure(hub, modality))
            return True
        except (RuntimeError, ValueError): return False

    def _state(self, closure, scope, session):
        if self._stop_event is not None and self._stop_event.is_set() or self.parent_cancelled():
            return {"kind": "unproven", "bindingSha256": None, "ownerSessionId": session, "complete": False, "ownerVersions": {}}
        versions, controls = scope.snapshot(), hashlib.sha256()
        for key, module in closure["modules"].items():
            if module in self.brain.core_pager._owners: continue
            for kind, values in (("buffer", module._buffers), ("parameter", module._parameters)):
                for name, value in values.items():
                    if (not isinstance(value, torch.Tensor) or name == "_autograd_trigger"
                            or kind == "buffer" and name in module._non_persistent_buffers_set): continue
                    controls.update(json.dumps([closure["names"][key], kind, name, self._hash_tensor(value)], separators=(",", ":")).encode())
        return {"kind": "live-owner-epochs", "bindingSha256": digest([self._closure_sha(closure), versions, controls.hexdigest()]),
            "ownerSessionId": session, "complete": True, "ownerVersions": versions}

    def capture_operation(self, hub, modality, tensor, session, review):
        with self._metadata_reservation(): return self._capture_operation(hub, modality, tensor, session, review)

    def _capture_operation(self, hub, modality, tensor, session, review):
        closure = self._closure(hub, modality); packed = self._validate_closure(closure)
        closure_sha, scope_key = self._closure_sha(closure), (id(hub), modality)
        cached = self._scopes.get(scope_key)
        if cached is not None and cached[0] != closure_sha: cached[1].close(); del self._scopes[scope_key]; cached = None
        if cached is None:
            scope = self.brain.core_pager.observe_learning_owners(packed)
            self._scopes[scope_key] = (closure_sha, scope)
        else: scope = cached[1]
        size = tensor.untyped_storage().nbytes() + len(closure["modules"]) * 1024
        lease = self.brain.resource_policy.reserve_ram(size, "actual perception input and scoped runtime metadata")
        try:
            guard = _tensor_guard(tensor)
            workload = self._hash_tensor(tensor)
            if _tensor_guard(tensor) != guard: raise RuntimeError("actual sensory input changed during its exact binding")
            state = self._state(closure, scope, session)
            geometry = digest({"closure": closure_sha, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "stride": list(tensor.stride()), "home": canonical_device(tensor.device), "operation": modality})
            binding = review["binding"]
            residency = [[str(id(module)), None if self.brain.core_pager._owners[module].device is None else
                          canonical_device(self.brain.core_pager._owners[module].device)] for module in packed]
            identity = {"ownerId": self.brain.brain_id, "ownerSessionId": session, "workKind": "native.perception." + modality + "-embedding",
                "stageCoverage": list(STAGES), "workloadSha256": workload, "geometrySha256": geometry,
                "samplingContractSha256": digest({"draws": 0, "returns": "complete-perception-embedding", "exactEquality": True}),
                **{key: binding[key] for key in ("policySha256", "topologySha256", "budgetSha256")},
                "cache": {"heat": "warm", "logicalStateSha256": digest(residency), "backendInitialization": "warm"}, "nativeState": state}
            lease.mark_allocated(len(closure["modules"]) * 1024)
            return {"hub": hub, "modality": modality, "tensor": tensor, "guard": guard, "lease": lease, "scope": scope,
                "closure": closure, "closureSha": closure_sha, "identity": identity, "residency": [(module, self.brain.core_pager._owners[module].device) for module in packed],
                "ownerObjects": {module: self.brain.core_pager._owners[module] for module in packed},
                "stratumKey": digest({"kind": identity["workKind"], "geometry": geometry, "binding": binding})}
        except BaseException: lease.release(); raise

    def operation_state(self, operation, session): return self._state(operation["closure"], operation["scope"], session)
    def unchanged(self, operation, session):
        with self._metadata_reservation():
            return (_tensor_guard(operation["tensor"]) == operation["guard"] and
                self._closure_sha(self._closure(operation["hub"], operation["modality"])) == operation["closureSha"] and
                self.operation_state(operation, session) == operation["identity"]["nativeState"])
    def release_operation(self, operation): operation["lease"].release()
    def retire_observations(self):
        for _sha, scope in self._scopes.values(): scope.close()
        self._scopes.clear()
        self.cached_operations.retire_observation()
        self.prefill_operations.retire_observation()
        self.forward_operations.retire_observation()
        if getattr(self, "media_operations", None) is not None: self.media_operations.retire_observation()
    def reserve_plan(self, size): return self.brain.resource_policy.reserve_ram(size, "derived exact perception Auto route metadata")
    def parent_cancelled(self):
        # Never call a chat's world-ticking/receipt-generating cancel callback.
        flags = getattr(self.brain, "_native_operation_auto_cancel_flags", None)
        return bool(flags is not None and flags.is_set())

    @contextmanager
    def pure_scope(self, stopped):
        token = _BYPASS.set(True)
        try:
            # Selected encoders have no dropout, batch norm or recurrent writes;
            # normal live perception can be no-grad while Module.training=True.
            # Preserve flags exactly instead of silently changing their mode.
            with torch.no_grad(), native_parameter_read_only(), self.brain.working_attention_pager.activate(cancelled=stopped.is_set): yield
        finally: _BYPASS.reset(token)

    def _activity_bytes(self, operation):
        """Conservative actual-geometry transient bound, not an allocator peak.

        Sum declared encoder intermediates (including int64 weighted kernels)
        rather than sizing them as int32 or crediting unconsumed GPU capacity.
        Packed/repeated-contact owner pages retain their existing admission.
        """
        from .model import _standard_padding_pairs, _standard_output_shape, PACKED_CONV_POSITION_BLOCK
        batch, element = int(operation["tensor"].shape[0]), operation["tensor"].element_size()
        def leaf(node, shape):
            if operation["modality"] == "audio" and len(shape) == 2: shape = (shape[0], 1, shape[1])
            total = math.prod(shape) * element * 3
            for layer in node["encoder"]:
                if isinstance(layer, torch.nn.SiLU): total += math.prod(shape) * element * 2; continue
                if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
                    shape = (shape[0], shape[1], 1, 1); total += math.prod(shape) * element * 2; continue
                kernel, spatial = tuple(layer.kernel_size), shape[2:]
                pairs = _standard_padding_pairs(layer.padding, spatial, kernel, layer.stride, layer.dilation)
                padded = tuple(size + a + b for size, (a, b) in zip(spatial, pairs))
                output = _standard_output_shape(padded, kernel, layer.stride, layer.dilation)
                width, outputs = int(layer._matrix_width), int(layer.out_channels // layer.groups)
                samples = batch * min(math.prod(output), PACKED_CONV_POSITION_BLOCK)
                cost = native_projection_tensor_budget(samples, width, outputs)["forwardTensorReserveBytes"]
                total += math.prod((shape[0], shape[1], *padded)) * element + samples * width * element * 2 + cost
                shape = (batch, layer.out_channels, *output)
                total += math.prod(shape) * element * 3
            projection = node["projection"]
            total += native_projection_tensor_budget(batch, projection.in_features, projection.out_features)["forwardTensorReserveBytes"]
            total += batch * (projection.in_features + projection.out_features * 4) * element
            return total, projection.out_features
        def walk(node, shape):
            if node["kind"] == "leaf": return leaf(node, shape)
            total, width = 0, 0
            for child in node["children"]:
                config = child["hub"].config
                target = ((config.audio_samples,) if operation["modality"] == "audio" else
                    (config.video_frames, config.image_size, config.image_size) if operation["modality"] == "video" else (config.image_size, config.image_size))
                child_shape = (shape[0], *shape[1:-len(target)], *target)
                cost, child_width = walk(child, child_shape); total += cost + math.prod(child_shape) * element * 2; width += child_width
            total += batch * width * element * 6  # concatenation/normalization/bridge sums
            for bridge in node["exchange"].bridges.values():
                total += native_projection_tensor_budget(batch, bridge.in_features, bridge.out_features)["forwardTensorReserveBytes"]
            return total, width
        needed, _width = walk(operation["closure"]["tree"], tuple(operation["tensor"].shape))
        return needed + 65536

    @contextmanager
    def _placement(self, candidate, operation, stopped):
        brain, pager = self.brain, self.brain.core_pager
        self.quiescent(); self.admit_candidate(candidate)
        modules = operation["closure"]["modules"].values()
        controls = [(module, dict(module._buffers), dict(module._parameters), getattr(module, "_native_compute_device", None))
                    for module in modules if module not in pager._owners]
        packed = [(module, getattr(module, "_native_compute_device", None)) for module, _device in operation["residency"]]
        consumer = _PerceptionTarget(self, candidate, stopped)
        token = _ACTIVE_REGIONAL.set((consumer, consumer._lease["epoch"], candidate["executionDevice"]))
        try:
            with transient_regional_scope(consumer):
                # No brain/device/policy/context/global .to() mutation.
                for module, device in packed: module._native_compute_device = torch.device(candidate["executionDevice"])
                for module, buffers, parameters, _device in controls:
                    for name, value in buffers.items():
                        # The only non-paged persistent leaf in this reviewed
                        # closure is the pool's tiny CPU ordinal counter. It
                        # must not be moved or learned during inference.
                        if isinstance(value, torch.Tensor) and name != "_next_page":
                            module._buffers[name] = self._move(value, candidate["executionDevice"])
                    for name, value in parameters.items():
                        if isinstance(value, torch.Tensor):
                            raise RuntimeError("unknown floating learned perception parameter cannot be privately remapped")
                yield
        finally:
            consumer.active = False; _ACTIVE_REGIONAL.reset(token)
            try:
                for module, old_device in operation["residency"]:
                    with pager._lock:
                        original_owner = operation["ownerObjects"][module]
                        if pager._owners.get(module) is not original_owner or not original_owner.loaded:
                            raise RuntimeError("cannot restore a replaced or retired packed perception owner")
                        # Verify actual writes even for the unchanged-device
                        # fast path. Avoid needless evict/readmit/validation of
                        # an already exact original residency, especially the
                        # current candidate's controlled baseline.
                        pager._observe_mutations(module, original_owner)
                        actual = None if original_owner.device is None else canonical_device(original_owner.device)
                        expected = None if old_device is None else canonical_device(old_device)
                    if actual != expected:
                        pager.evict(module)
                        if old_device is not None:
                            with pager.use(module, old_device): pass
            except BaseException as error:
                brain._native_execution_restore_required = True
                raise OperationAutoRestoreRequired("exact packed perception residency restore failed; explicit Unload required") from error
            finally:
                for module, buffers, parameters, compute_device in controls:
                    module._buffers.clear(); module._buffers.update(buffers)
                    module._parameters.clear(); module._parameters.update(parameters)
                    if compute_device is not None: module._native_compute_device = compute_device
                for module, device in packed:
                    if device is not None: module._native_compute_device = device
                pager._regional_limits()

    def _move(self, value, target):
        if canonical_device(value.device) == canonical_device(target): return value
        size = value.numel() * value.element_size()
        with self.brain._regional_transfer_reservation(target, size, "pure perception actual activity transfer") as held:
            result = value.to(device=target, non_blocking=False)
            if held is not None: held.mark_allocated(size)
        self._activation_transfer_bytes += size
        return result

    def execute(self, candidate, operation, observer, stopped):
        result = None
        def stage(name):
            if observer is not None: observer.stage_completed(name)
            if stopped.is_set(): raise InterruptedError("private perception stopped at its actual stage boundary")
        try:
            needed = self._activity_bytes(operation)
            self.brain.resource_policy.require_device(candidate["executionDevice"], needed, "complete declared perception transient tensors")
            reservation = (self.brain.resource_policy.reserve_ram(needed, "declared same-brain perception intermediates")
                if not candidate["executionDevice"].startswith("cuda:") else nullcontext())
            with reservation, self._placement(candidate, operation, stopped):
                stage(STAGES[0])
                tensor = self._move(operation["tensor"], candidate["executionDevice"]); stage(STAGES[1])
                output = operation["hub"].perception_embedding(operation["modality"], tensor); stage(STAGES[2])
                size = output.numel() * output.element_size()
                lease = self.brain.resource_policy.reserve_ram(size * 2, "complete checked perception home output")
                try:
                    output = self._move(output, operation["tensor"].device).detach().clone()
                    if not bool(torch.isfinite(output).all()): raise RuntimeError("perception embedding is nonfinite")
                    lease.mark_allocated(size)
                    result = PerceptionOutput(output, lease, self._hash_tensor(output))
                except BaseException: lease.release(); raise
                stage(STAGES[3])
            stage(STAGES[4])
            self.synchronize(candidate["asyncDevices"])
            return result
        except BaseException:
            if result is not None: result.close()
            raise

    def exact_output(self, reference, trial):
        return (reference.value.shape == trial.value.shape and reference.value.dtype == trial.value.dtype
                and reference.contract == trial.contract and bool(torch.equal(reference.value, trial.value)))
    def output_descriptor(self, output): return {"shape": list(output.value.shape), "dtype": str(output.value.dtype)}
    def check_output(self, output, descriptor):
        if self.output_descriptor(output) != descriptor or not bool(torch.isfinite(output.value).all()):
            raise RuntimeError("reused perception route returned a different output geometry or nonfinite values")

    def append_receipt(self, job_id, payload):
        events = self.brain.events
        if self._event_range is None or self._event_range[0] != job_id:
            first = int(events.connection.execute("SELECT COALESCE(MAX(sequence),0) FROM events").fetchone()[0])
            self._event_range = [job_id, first, first]
        size = len(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        self.brain.resource_policy.require_disk(size * 2 + 65536, "native perception Auto evidence journal")
        with self.brain.resource_policy.reserve_ram(size * 4, "perception evidence serialization"):
            event_id = events.append("native-operation-auto", payload, job_id=job_id)
        self._event_range[2] = int(events.connection.execute("SELECT MAX(sequence) FROM events").fetchone()[0])
        return event_id
    def receipts(self, job_id):
        if self._event_range is None or self._event_range[0] != job_id: raise RuntimeError("perception evidence belongs to another live operation")
        events = self.brain.events
        first, last = self._event_range[1:]
        rows = events.connection.execute("SELECT sequence,length(CAST(payload_json AS BLOB)),payload_sha256 FROM events "
            "WHERE sequence>? AND sequence<=? AND job_id=? AND brain_id=? AND kind='native-operation-auto' ORDER BY sequence",
            (first, last, job_id, self.brain.brain_id))
        for sequence, size, sha in rows:
            with self.brain.resource_policy.reserve_ram(int(size) * 6, "one exact perception Auto evidence row"):
                raw = events.connection.execute("SELECT payload_json FROM events WHERE sequence=?", (sequence,)).fetchone()[0]
                if hashlib.sha256(raw.encode()).hexdigest() != sha: raise RuntimeError("perception evidence row was modified")
                row = json.loads(raw)
                if row["phase"] == "measured": yield row["observation"]


def create_native_operation_auto(brain):
    controller = OperationAutoPolicy(brain.brain_id, NativeOperationAutoOps(brain), owner_session=brain._native_operation_auto_session)
    controller.ops._stop_event = controller._cancel
    # Bind only the public complete entry. Private nested connected encoders run
    # under _BYPASS and cannot independently benchmark/repeat parent effects.
    bind_operation_auto(brain.modalities, controller)
    from .cached_whole_operation_auto import bind_cached_whole_operation
    bind_cached_whole_operation(brain.decoder, controller)
    from .prefill_whole_operation_auto import bind_prefill_whole_operation
    bind_prefill_whole_operation(brain.decoder, controller)
    from .forward_whole_operation_auto import bind_full_forward_operation
    bind_full_forward_operation(brain.decoder, controller)
    return controller
