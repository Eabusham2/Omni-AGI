"""Actual loaded ordinary-decoder callbacks for measured isolated prefill.

Import defines adapters only. No model construction, Load or calibration is
performed. Controls use the separate metadata-only controller; these callbacks
run on the already-owned serialized neural lane.
"""
from __future__ import annotations

import copy
import hashlib
import json
import threading
import weakref
from contextlib import contextmanager, nullcontext

import torch

from .decoder_path_execution import DecoderPathExecution, bind_decoder_execution
from .parameter_diagnostics import native_parameter_read_only
from .regional_execution import canonical_device, topology_identity
from .whole_path_execution import digest


def _tensor_guard(value):
    if value is None:
        return None
    return (id(value), int(value._version), str(value.device), str(value.dtype), tuple(value.shape),
            tuple(value.stride()), int(value.storage_offset()), int(value.untyped_storage().data_ptr()))


class NativeDecoderExecutionOps:
    relative_tolerance = 2e-5
    absolute_tolerance = 1e-4

    def __init__(self, brain):
        self._owner = weakref.ref(brain)
        self._activation_transfer_bytes = 0
        self._event_range = None
        self._home = None

    @property
    def brain(self):
        value = self._owner()
        if value is None:
            raise RuntimeError("loaded decoder execution owner was released")
        return value

    def supported(self):
        from .model import OmniDecoder
        brain = self.brain
        if getattr(brain, "_native_execution_restore_required", False):
            return False, "failed-native-placement-restore-requires-explicit-unload"
        if type(brain.decoder) is not OmniDecoder:
            return False, "connected-decoder-awaits-complete-fast-recurrent-owner-fence"
        if brain.resource_policy.device_resources is None:
            return False, "measured-decoder-placement-requires-current-per-device-budget-settings"
        if getattr(getattr(brain, "_regional_execution", None), "active", False):
            return False, "manual-regional-lease-remains-separate-from-measured-decoder-auto"
        return True, "ordinary-native-decoder-supported"

    def quiescent(self):
        self.brain._regional_quiescent()
        supported, reason = self.supported()
        if not supported:
            raise RuntimeError(reason)

    def retain_input(self, window, cue):
        # Borrow only the actual input; retained storage (including a view's
        # backing) is admitted, not a secretly synthesized/retrieved prompt.
        if window.ndim != 2 or window.shape[0] < 1 or window.shape[1] < 1 or window.dtype != torch.long:
            raise ValueError("actual decoder input must be a non-empty int64 batch")
        tensors = [value for value in (window, cue) if value is not None]
        unique = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes() for value in tensors}
        size = sum(unique.values())
        lease = self.brain.resource_policy.reserve_ram(size, "retained actual decoder input storage")
        try:
            lease.mark_allocated(size)
            return {"window": window.detach(), "cue": None if cue is None else cue.detach(),
                "originals": (window, cue), "guards": (_tensor_guard(window), _tensor_guard(cue)),
                "lease": lease, "materialized": False, "homeDevice": str(window.device)}
        except BaseException:
            lease.release()
            raise

    def release_input(self, capture):
        capture["lease"].release()
        capture.clear()

    def retire_observation(self):
        self.brain.core_pager.retire_learning_observation()

    def _hash_tensor(self, value):
        if value is None:
            return None
        hasher = hashlib.sha256()
        hasher.update(json.dumps([str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        block = max(1, 256 * 1024 // value.element_size())
        layout_lease = (nullcontext() if value.is_contiguous() else self.brain.resource_policy.reserve_ram(
            value.numel() * value.element_size(), "noncontiguous exact decoder checksum layout"))
        with layout_lease, self.brain.resource_policy.reserve_ram(min(value.numel(), block) * value.element_size() * 3,
                "bounded decoder input/control checksum"):
            flat = value.detach().reshape(-1)
            version = int(value._version)
            for start in range(0, value.numel(), block):
                payload = flat[start:start + block].to(device="cpu", non_blocking=False).contiguous().view(torch.uint8).numpy()
                hasher.update(memoryview(payload))
            if int(value._version) != version:
                raise RuntimeError("decoder tensor changed while binding its exact bytes")
        return hasher.hexdigest()

    def materialize_input(self, capture):
        if capture["materialized"]:
            if (_tensor_guard(capture["window"]), _tensor_guard(capture["cue"])) != capture["guards"]:
                raise RuntimeError("private measured input was mutated")
            return capture
        if tuple(_tensor_guard(value) for value in capture["originals"]) != capture["guards"]:
            raise RuntimeError("normal input/cue changed before quiescent capture materialization")
        size = sum(value.numel() * value.element_size() for value in (capture["window"], capture["cue"]) if value is not None)
        lease = self.brain.resource_policy.reserve_ram(size, "exact isolated-prefill CPU input copy")
        try:
            window = capture["window"].to(device="cpu", non_blocking=False).clone()
            cue = None if capture["cue"] is None else capture["cue"].to(device="cpu", non_blocking=False).clone()
            if tuple(_tensor_guard(value) for value in capture["originals"]) != capture["guards"]:
                raise RuntimeError("normal input/cue changed during capture copy")
            lease.mark_allocated(size)
            result = {"window": window, "cue": cue, "lease": lease, "materialized": True,
                "homeDevice": capture["homeDevice"], "guards": (_tensor_guard(window), _tensor_guard(cue))}
            result["workloadSha256"] = digest({"input": self._hash_tensor(window), "cue": self._hash_tensor(cue)})
            return result
        except BaseException:
            lease.release()
            raise

    def describe_input(self, capture):
        return {"inputShape": list(capture["window"].shape), "cueShape": None if capture["cue"] is None else list(capture["cue"].shape)}

    def _registry(self):
        decoder, pager = self.brain.decoder, self.brain.core_pager
        hasher = hashlib.sha256()
        # Metadata is streamed per real module, not one growing registry JSON.
        # Torch's named_modules memo is admitted separately from learned bytes.
        with self.brain.resource_policy.reserve_ram(max(64 * 1024, len(pager._owners) * 4096),
                                                  "decoder registry traversal metadata"):
            for name, module in decoder.named_modules():
                keys = set(getattr(module, "_packed_structural_fields", ())) | {
                    "eps", "n_heads", "head_dim", "dimensions", "in_features", "out_features", "num_embeddings",
                    "slots", "iterations", "query_chunk_slots", "query_chunk_tokens", "key_chunk_tokens", "max_seq_len"}
                structural = {key: getattr(module, key) for key in sorted(keys)
                              if type(getattr(module, key, None)) in {int, float, str, bool}}
                row = [name, str(id(module)), type(module).__module__ + "." + type(module).__name__, structural,
                    [(key, list(value.shape), str(value.dtype)) for key, value in module._buffers.items()
                     if isinstance(value, torch.Tensor) and key not in module._non_persistent_buffers_set]]
                encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                hasher.update(len(encoded).to_bytes(8, "little")); hasher.update(encoded)
        return hasher.hexdigest(), pager

    def native_state(self, owner_session):
        registry, pager = self._registry()
        # Hash ONLY the small non-pager controls. Actual packed weights use the
        # complete persistent-owner mutation epoch, not a per-token model hash.
        controls = hashlib.sha256()
        for name, module in self.brain.decoder.named_modules():
            if getattr(module, "_native_core_pager", None) is pager:
                continue
            for kind, values in (("buffer", module._buffers), ("parameter", module._parameters)):
                for key, value in values.items():
                    if (not isinstance(value, torch.Tensor) or key == "_autograd_trigger"
                            or kind == "buffer" and key in module._non_persistent_buffers_set): continue
                    controls.update(json.dumps([name, kind, key, self._hash_tensor(value)], separators=(",", ":")).encode())
        version = int(pager.learning_revision)
        return {"kind": "live-owner-epochs", "bindingSha256": digest({"registry": registry, "epoch": version, "controls": controls.hexdigest()}),
                "ownerSessionId": owner_session, "complete": True, "ownerVersions": {"ordinary-decoder-packed": version}}

    def _geometry(self, capture):
        decoder = self.brain.decoder
        return digest({"input": self.describe_input(capture), "vocabulary": decoder.config.vocab_size,
            "configSha256": digest(decoder.config.to_dict()),
            "dimensions": decoder.config.d_model, "feedForward": decoder.config.d_ff,
            "layers": len(decoder.blocks), "heads": decoder.config.n_heads,
            "maxSequence": decoder.config.max_seq_len, "workspaceSlots": decoder.global_workspace.slots,
            "experts": len(decoder.experts) if hasattr(decoder, "experts") else None,
            "home": str(self.brain.device), "operation": "isolated-prefix-causal-prefill-no-rng-no-retained-cache"})

    def _budgets(self):
        policy = self.brain.resource_policy
        reading = policy.status()
        rows = policy.device_resources.budgets(reading)
        # Selected capacities/policy only, NOT mutable currently-free readings.
        return [dict((key, row[key]) for key in ("deviceId", "memoryGroupId", "kind", "enabled", "ceilingBytes", "selection", "autoIncrease"))
                for row in rows if row["kind"] in {"ram", "gpu"}]

    def binding(self, capture, owner_session, capture_id, *, topology=None):
        self.materialize_input(capture)
        if canonical_device(self.brain.device) != canonical_device(capture["homeDevice"]):
            raise RuntimeError("captured prefill home backend changed; capture a new normal input")
        actual_topology = self.brain._regional_topology()
        topo_sha = digest(topology_identity(actual_topology))
        if topology is not None and digest(topology_identity(topology)) != topo_sha:
            raise RuntimeError("main-observed compute topology changed before native review")
        registry, _pager = self._registry()
        state = self.native_state(owner_session)
        return {"captureId": capture_id, "ownerSessionId": owner_session,
            "policySha256": self.brain._regional_policy_sha(), "topologySha256": topo_sha,
            "registrySha256": registry, "nativeStateSha256": state["bindingSha256"],
            "workloadSha256": capture["workloadSha256"], "geometrySha256": self._geometry(capture),
            "budgetSha256": digest(self._budgets())}

    def candidates(self, acknowledge_cuda):
        brain = self.brain
        result = []
        for device in brain._regional_topology()["devices"]:
            name = canonical_device(device.get("executionDevice", ""))
            if device["kind"] not in {"ram", "gpu"} or name != "cpu" and not (name == "mps" or name.startswith("cuda:")):
                continue
            cuda = name.startswith("cuda:")
            if cuda and not acknowledge_cuda or not cuda and not device["nativeComputeSupported"]:
                continue
            budget = brain.resource_policy.device_budget(torch.device(name), fresh=True)
            if not budget or not budget["enabled"] or not budget["capacityMeasured"]:
                continue
            home = canonical_device(brain.device)
            asynchronous = sorted({item for item in (name, home) if item != "cpu"})
            descriptor = {"executionDevice": name, "deviceId": device["id"], "memoryGroupId": device["memoryGroupId"],
                "backendMode": "explicit-cuda-cpu-tile" if cuda else "native-metal" if name == "mps" else "native-cpu",
                "asyncDevices": asynchronous, "supported": True, "fallbackAcknowledged": cuda}
            descriptor["planSha256"] = digest(descriptor)
            result.append(descriptor)
        current = next((item for item in result if item["executionDevice"] == canonical_device(brain.device)), None)
        if current is None:
            raise RuntimeError("current placement lacks measured admission or explicit CUDA fallback acknowledgment")
        return current, sorted(result, key=lambda item: (item != current, item["executionDevice"]))

    def identity(self, binding, owner_session, work_kind, stages):
        return {"ownerId": self.brain.brain_id, "ownerSessionId": owner_session, "workKind": work_kind,
            "stageCoverage": list(stages), **{key: binding[key] for key in ("workloadSha256", "geometrySha256", "policySha256", "topologySha256", "budgetSha256")},
            "samplingContractSha256": digest({"draws": 0, "relativeTolerance": self.relative_tolerance,
                "absoluteTolerance": self.absolute_tolerance, "returns": ["next-logits", "normalized-last-neural-state"]}),
            "cache": {"heat": "warm", "logicalStateSha256": digest({"privateActivity": "empty-each-operation", "returnsToHome": True}),
                "backendInitialization": "warm"}, "nativeState": self.native_state(owner_session)}

    @contextmanager
    def pure_scope(self, cancelled):
        brain, decoder = self.brain, self.brain.decoder
        training = [(module, module.training) for module in decoder.modules()]
        previous_state = getattr(decoder, "last_generation_neural_state", None)
        try:
            decoder.eval()
            with torch.no_grad(), native_parameter_read_only(), brain.working_attention_pager.activate(cancelled=cancelled.is_set):
                yield
        finally:
            decoder.last_generation_neural_state = previous_state
            for module, was_training in training:
                module.training = was_training

    def _copy_output(self, logits, state, *, device="cpu"):
        size = sum(value.numel() * value.element_size() for value in (logits, state))
        self.brain.resource_policy.require_device(torch.device(device), size, "isolated-prefill exact home output")
        lease = self.brain.resource_policy.reserve_ram(size * 2, "isolated-prefill checked output transfer")
        try:
            values = tuple(value.detach().to(device=device, non_blocking=False).clone() for value in (logits, state))
            lease.mark_allocated(size)
            for original, copied in zip((logits, state), values):
                if original.device != copied.device:
                    self._activation_transfer_bytes += original.numel() * original.element_size()
            return {"logits": values[0], "state": values[1], "lease": lease}
        except BaseException:
            lease.release()
            raise

    def release_output(self, output):
        output["lease"].release()
        output.clear()

    def consumer_result(self, output):
        result = (output["logits"], output["state"])
        self.release_output(output)
        return result

    def calibration_reservation(self, samples, candidates):
        # Physical admission, not an arbitrary lifetime/sample count cap.
        return self.brain.resource_policy.reserve_ram((samples + 1) * candidates * 256,
                                                     "decoder calibration receipt identifiers")

    def reference_output(self, capture, cancelled):
        if cancelled.is_set():
            raise InterruptedError("decoder reference cancelled before compute")
        decoder, device = self.brain.decoder, self.brain.device
        self.admit_candidate({"executionDevice": canonical_device(device)})
        size = sum(value.numel() * value.element_size() for value in (capture["window"], capture["cue"]) if value is not None)
        with self.brain.resource_policy.reserve_ram(size * 2, "exact reference input/cue transfer"):
            window, cue = self._inputs(capture, device)
            # Existing full-forward admission is retained, never expanded to force
            # a reference through an insufficient RAM/device envelope.
            output = decoder.forward(window, memory_bias=cue)
            result = self._copy_output(output["logits"][:, -1], output["hidden"][:, -1], device=device)
        if cancelled.is_set():
            self.release_output(result)
            raise InterruptedError("decoder reference cancelled after compute")
        return result

    def output_contract(self, reference):
        return digest({"referenceLogitsSha256": self._hash_tensor(reference["logits"]),
            "referenceNeuralStateSha256": self._hash_tensor(reference["state"]),
            "relativeTolerance": self.relative_tolerance, "absoluteTolerance": self.absolute_tolerance,
            "finiteRequired": True, "exactShapeAndDtype": True, "samplingDraws": 0})

    def compare(self, reference, output):
        differences = {}
        equivalent = True
        for name in ("logits", "state"):
            a, b = reference[name], output[name]
            shapes = a.shape == b.shape and a.dtype == b.dtype
            finite = shapes and bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            equal = finite and bool(torch.allclose(a, b, rtol=self.relative_tolerance, atol=self.absolute_tolerance))
            equivalent = equivalent and equal
            differences[name] = float((a - b).abs().max().item()) if finite else None
        return {"equivalent": equivalent, "maximumAbsoluteLogitError": differences["logits"],
            "maximumAbsoluteStateError": differences["state"],
            "actualLogitsSha256": self._hash_tensor(output["logits"]), "actualNeuralStateSha256": self._hash_tensor(output["state"]),
            "argmaxAgreement": bool(torch.equal(reference["logits"].argmax(-1), output["logits"].argmax(-1))),
            "maximumAbsoluteProbabilityError": float((reference["logits"].float().softmax(-1) - output["logits"].float().softmax(-1)).abs().max().item()) if equivalent else None}

    def _inputs(self, capture, device):
        size = sum(value.numel() * value.element_size() for value in (capture["window"], capture["cue"]) if value is not None)
        self.brain.resource_policy.require_device(device, size, "exact measured-prefill input/cue transfer")
        window = capture["window"].to(device=device, non_blocking=False)
        cue = None if capture["cue"] is None else capture["cue"].to(device=device, non_blocking=False)
        for source, target in ((capture["window"], window), (capture["cue"], cue)):
            if source is not None and source.device != target.device:
                self._activation_transfer_bytes += source.numel() * source.element_size()
        return window, cue

    def admit_candidate(self, candidate):
        if candidate.get("backendMode") == "native-mixed-sequential":
            from .mixed_region_native import admit_mixed_candidate
            return admit_mixed_candidate(self, candidate)
        device = torch.device(candidate["executionDevice"])
        policy = self.brain.resource_policy
        budget = policy.device_budget(device, fresh=True)
        if not budget or not budget["enabled"] or not budget["capacityMeasured"]:
            raise RuntimeError("candidate lacks current measured per-device admission")
        if device.type not in {"cpu", "mps", "cuda"}:
            raise RuntimeError("unsupported native candidate backend")
        return budget

    @contextmanager
    def _placement(self, candidate):
        brain, decoder = self.brain, self.brain.decoder
        self.quiescent()
        # Hold the SAME original controls for exact rollback, not a learned
        # master/model copy. Packed owners remain under their existing pager.
        roots = tuple(brain._trainable_modules())
        modules = {id(module): module for root in roots for module in root.modules()}
        registered = [(module, dict(module._buffers), dict(module._parameters), getattr(module, "_native_compute_device", None))
                      for module in modules.values() if getattr(module, "_native_core_pager", None) is not brain.core_pager]
        attrs = {name: value for name, value in brain.__dict__.items()
            if not isinstance(value, torch.nn.Module) and next(brain._native_state_tensors(value), None) is not None}
        old = {"device": brain.device, "backend": brain.device_backend,
            "residency": copy.deepcopy(brain._native_backend_residency), "lastChanged": brain._native_backend_last_changed,
            "policyDevice": brain.resource_policy.execution_device,
            "gpuAttributes": copy.copy(getattr(brain, "_native_gpu_state_attributes", set())),
            "working": (brain.working_attention_pager.compute_device, brain.working_attention_pager.resident_budget_bytes,
                brain.working_attention_pager.scratch_budget_bytes, brain.working_attention_pager.device_tile_budget_bytes,
                tuple(getattr(brain.working_attention_pager, "_retained_context_pools", ()))),
            "chunks": [(block.attention, block.attention.query_chunk_tokens, block.attention.key_chunk_tokens) for block in decoder.blocks],
            "workspaceChunk": decoder.global_workspace.query_chunk_slots}
        try:
            brain.prepare_native_execution(operation="measured-isolated-decoder-prefill", quiescent=True,
                reviewed_target=candidate["executionDevice"], acknowledge_packed_cpu_fallback=candidate.get("fallbackAcknowledged", False))
            if canonical_device(brain.device) != candidate["executionDevice"]:
                raise RuntimeError("admitted native backend differs from the measured candidate")
            yield
        finally:
            # Cache owners must be closed by execute's inner finally first.
            # Exact packed copies are evicted before old controls are restored.
            try:
                brain.core_pager.flush()
            except BaseException:
                brain._native_execution_restore_required = True
                raise
            finally:
                # Even an unexpected packed flush error cannot skip restoring
                # the original controls. A failed packed restore stays visibly
                # paused until explicit Unload; it never silently resumes chat.
                for module, buffers, parameters, compute_device in registered:
                    module._buffers.clear(); module._buffers.update(buffers)
                    module._parameters.clear(); module._parameters.update(parameters)
                    if compute_device is not None: module._native_compute_device = compute_device
                for root in roots:
                    for module in root.modules():
                        if getattr(module, "_native_core_pager", None) is brain.core_pager:
                            module._native_compute_device = old["device"]
                for name, value in attrs.items(): setattr(brain, name, value)
                brain.device, brain.device_backend = old["device"], old["backend"]
                brain._native_backend_residency, brain._native_backend_last_changed = old["residency"], old["lastChanged"]
                brain._native_gpu_state_attributes = old["gpuAttributes"]
                brain.resource_policy.execution_device = old["policyDevice"]
                pager = brain.working_attention_pager
                pager.compute_device, pager.resident_budget_bytes, pager.scratch_budget_bytes, pager.device_tile_budget_bytes = old["working"][:4]
                pager.configure_retained_context_pools(old["working"][4])
                for attention, query, key in old["chunks"]:
                    attention.query_chunk_tokens, attention.key_chunk_tokens = query, key
                decoder.global_workspace.query_chunk_slots = old["workspaceChunk"]
                brain.core_pager.refresh_budget(force=True)

    def execute(self, candidate, capture, observation, cancelled):
        if cancelled.is_set():
            raise InterruptedError("decoder operation stopped before placement")
        def stage(name):
            if observation is not None: observation.stage_completed(name)
        cache, result = None, None
        previous_state = getattr(self.brain.decoder, "last_generation_neural_state", None)
        try:
            with self._placement(candidate):
                stage("exact-placement-admission")
                if cancelled.is_set(): raise InterruptedError("decoder operation stopped before input transfer")
                size = sum(value.numel() * value.element_size() for value in (capture["window"], capture["cue"]) if value is not None)
                with self.brain.resource_policy.reserve_ram(size * 2, "measured-prefill input/cue activity"):
                    window, cue = self._inputs(capture, self.brain.device)
                    stage("input-and-cue-transfer")
                    try:
                        logits, cache = self.brain.decoder._generation_step(window, cue, None)
                        stage("native-prefill")
                        if cancelled.is_set(): raise InterruptedError("decoder operation stopped after native prefill")
                        result = self._copy_output(logits, self.brain.decoder.last_generation_neural_state, device=capture["homeDevice"])
                        stage("output-home-transfer")
                    finally:
                        if cache is not None: cache.close()
                        stage("private-cache-release")
            stage("home-placement-restore")
            return result
        except BaseException:
            if result is not None: self.release_output(result)
            raise
        finally:
            self.brain.decoder.last_generation_neural_state = previous_state

    def synchronize(self, devices):
        completed = []
        for name in devices:
            if name == "mps": torch.mps.synchronize()
            elif name.startswith("cuda:"): torch.cuda.synchronize(torch.device(name))
            else: raise RuntimeError("unsupported async completion device")
            completed.append(name)
        return completed

    def counters(self):
        return {**self.brain.core_pager.execution_counters(),
            "attentionReadBytes": self.brain.working_attention_pager._metrics["bytesRead"],
            "attentionWriteBytes": self.brain.working_attention_pager._metrics["bytesWritten"],
            "activationTransferBytes": self._activation_transfer_bytes}

    @contextmanager
    def observation_resources(self, candidate, budget_sha):
        lock, stop = threading.RLock(), threading.Event()
        peaks, ceilings = {}, {}
        state = {"verified": True, "pressure": False, "sampled": False}
        participants = {canonical_device(self.brain.device), canonical_device(candidate["executionDevice"]),
            *(canonical_device(value) for value in candidate.get("asyncDevices", ()))}
        def sample():
            try:
                reading = self.brain.resource_policy.status(fresh_memory_sample=True)
                budgets = self._budgets()
                participating = {}
                for name in participants:
                    observed = self.brain.resource_policy.device_budget(name, fresh=True)
                    if not observed: raise RuntimeError("whole-path participant budget is unavailable")
                    participating[observed["deviceId"]] = observed
                with lock:
                    state["verified"] &= reading.get("ramAdmissionVerified") is True and digest(budgets) == budget_sha
                    state["verified"] &= all(value["enabled"] and value["capacityMeasured"] and any(
                        row["deviceId"] == identifier and row["enabled"] for row in budgets)
                        for identifier, value in participating.items())
                    state["pressure"] |= bool(reading.get("memoryPressure"))
                    resident = reading.get("admissionResidentMemoryBytes")
                    group_interfaces = {}
                    for row in budgets:
                        group = row["memoryGroupId"]
                        if not row["enabled"]: continue
                        observed = participating.get(row["deviceId"])
                        if observed is not None:
                            allocated = observed.get("allocatedBytes")
                            state["verified"] &= bool(observed["enabled"] and observed["capacityMeasured"])
                            state["pressure"] |= type(allocated) is int and allocated > row["ceilingBytes"]
                        # Shared physical groups get ONE ceiling/peak, not RAM
                        # plus Metal added as separate physical capacity.
                        if group == "ram:system":
                            ceilings[group] = int(reading["systemRamBudgetBytes"])
                            used = resident
                        elif observed is not None:
                            ceilings[group] = row["ceilingBytes"]
                            used = observed.get("allocatedBytes")
                            previous = group_interfaces.setdefault(group, row["deviceId"])
                            if previous != row["deviceId"]:
                                # Without a physical-group usage counter,
                                # separate non-RAM allocator readings cannot
                                # prove whether their bytes overlap. Refuse a
                                # route instead of inventing capacity or a sum.
                                state["verified"] = False
                        else: continue  # Unused distinct accelerator not credited.
                        if type(used) is not int: state["verified"] = False
                        else: peaks[group] = max(peaks.get(group, 0), used)
            except Exception:
                with lock: state["verified"] = False
        sample()
        def watcher():
            while not stop.wait(0.05): sample()
        thread = threading.Thread(target=watcher, name="omni-prefill-observation", daemon=True)
        thread.start()
        def reading():
            sample()
            with lock:
                result = {"budgetSha256": budget_sha, "admissionComplete": state["verified"],
                    "pressureObserved": state["pressure"], "physicalGroups": {group: {"ceilingBytes": ceilings[group],
                        "observedPeakBytes": peak, "peakObservation": "sampled-full-operation" if state["sampled"] else "operation-start"}
                        for group, peak in peaks.items()}}
                state["sampled"] = True
                return result
        try: yield reading
        finally:
            stop.set(); thread.join()

    def append_receipt(self, job_id, payload):
        events = self.brain.events
        if self._event_range is None or self._event_range[0] != job_id:
            first = events.connection.execute("SELECT COALESCE(MAX(sequence),0) FROM events").fetchone()[0]
            self._event_range = [job_id, int(first), int(first)]
        size = len(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        self.brain.resource_policy.require_disk(size * 2 + 64 * 1024, "decoder operation evidence journal")
        with self.brain.resource_policy.reserve_ram(size * 4, "decoder operation evidence serialization"):
            event_id = events.append("decoder-execution", payload, job_id=job_id)
        self._event_range[2] = int(events.connection.execute("SELECT MAX(sequence) FROM events").fetchone()[0])
        return event_id

    def receipts(self, job_id):
        if self._event_range is None or self._event_range[0] != job_id:
            raise RuntimeError("decoder receipt source does not belong to this live operation")
        events = self.brain.events
        first, last = self._event_range[1:]
        cursor = events.connection.execute("SELECT sequence,length(CAST(payload_json AS BLOB)),payload_sha256 FROM events "
            "WHERE sequence>? AND sequence<=? AND job_id=? AND brain_id=? AND kind='decoder-execution' ORDER BY sequence",
            (first, last, job_id, self.brain.brain_id))
        for sequence, size, sha in cursor:
            with self.brain.resource_policy.reserve_ram(int(size) * 6, "bounded exact decoder evidence row"):
                raw = events.connection.execute("SELECT payload_json FROM events WHERE sequence=?", (sequence,)).fetchone()[0]
                if hashlib.sha256(raw.encode()).hexdigest() != sha:
                    raise RuntimeError("decoder evidence row checksum mismatch")
                payload = json.loads(raw)
                if payload["phase"] == "measured": yield payload["observation"]

    def receipt_head(self, job_id):
        head, count = "0" * 64, 0
        for receipt in self.receipts(job_id):
            head, count = digest([head, receipt["receiptSha256"]]), count + 1
        return head, count

    def auto_allowed(self, candidate):
        return self.admit_candidate(candidate)["selection"] == "auto"

    def input_may_match(self, capture, window, cue):
        return (capture is not None and capture["materialized"] and tuple(window.shape) == tuple(capture["window"].shape)
            and window.dtype == capture["window"].dtype and str(window.device) == capture["homeDevice"]
            and (cue is None) == (capture["cue"] is None)
            and (cue is None or cue.shape == capture["cue"].shape and cue.dtype == capture["cue"].dtype))

    def matches_input(self, capture, window, cue):
        return digest({"input": self._hash_tensor(window), "cue": self._hash_tensor(cue)}) == capture["workloadSha256"]


def create_decoder_execution(brain):
    controller = DecoderPathExecution(brain.brain_id, NativeDecoderExecutionOps(brain),
        owner_session=brain._decoder_execution_session)
    bind_decoder_execution(brain.decoder, controller)
    return controller
