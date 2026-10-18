"""Read-only same-owner complete media consumers and private effect sinks.

No model is constructed/copied and no codec is run on import. Borrowed connected
regions, codec tables, exchanges and native generators form one closed route;
opaque callbacks or an unpinned external codec simply retain ordinary output.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
import weakref
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from .cached_decoder_native import NativeCachedDecoderOps
from .decoder_execution_native import _tensor_guard
from .media_artifact_operation import media_artifact_scope
from .media_canvas import media_storage_lifetime
from .media_randomness import MediaRandomScope
from .media_streaming_codec import media_view, payload_sha256, existing_media_sha256, write_media_payload
from .operation_auto_registry import PRIVATE_OPERATION, MEDIA_KINDS, MEDIA_STAGES
from .operation_auto_policy import OperationAutoRestoreRequired
from .operation_auto_native import _PerceptionTarget
from .parameter_diagnostics import native_parameter_read_only
from .regional_execution import _ACTIVE_REGIONAL, canonical_device, transient_regional_scope
from .whole_path_execution import digest


def _plan_layout(plan):
    if plan is None: return None
    return {name: getattr(plan, name) for name in ("modality", "width", "height", "sample_rate", "total_samples", "fps",
        "total_frames", "spatial_overlap", "temporal_overlap", "work_units", "preview_every_units", "canvas_storage")}


class _MediaEffects:
    def __init__(self, brain, modality, *, private=False, directory=None, cancelled=lambda: False):
        self.brain, self.modality, self.private, self.directory = weakref.ref(brain), modality, private, directory
        self.cancelled, self.resource_paused = cancelled, False
        self._sha, self.previews, self.outputs = hashlib.sha256(), 0, 0
        self._files, self.failure = [], None
        self._directory_identity = None
        if private:
            directory = Path(directory); value = directory.lstat()
            if directory.is_symlink() or not directory.is_dir(): raise RuntimeError("private media output directory is not owned")
            self.directory, self._directory_identity = directory, (value.st_dev, value.st_ino)
    def _observe(self, callback):
        if self.failure is not None: return
        try: callback()
        except BaseException as error:
            if self.private: raise
            # Auto's evidence is optional; lack of its private proof budget
            # never interrupts the original public generator or callback.
            self.failure = error
    @contextmanager
    def payload_write(self, path, size):
        path = Path(path)
        directory = self.artifact_directory()
        if not path.is_relative_to(directory) or not path.parent.resolve().is_relative_to(directory.resolve()) or path.is_symlink():
            raise RuntimeError("private media write escaped its admitted effect sink")
        lease = self.brain().resource_policy.reserve_spill(size + 65536, "private complete-media artifact bytes")
        row = [lease, path, None]; self._files.append(row)
        try:
            yield
            current = path.lstat()
            if path.is_symlink() or current.st_size != size: raise RuntimeError("private complete-media artifact write changed shape")
            row[2] = (current.st_dev, current.st_ino)
            lease.commit(path=path, promised_bytes=size, retain_open_backing=True)
        except BaseException:
            # Keep the escrow while its actual file can still exist. The
            # caller removes only its own TemporaryDirectory before close.
            raise
    def close_files(self):
        first = None
        for lease, _path, identity in self._files:
            try:
                if identity is None: lease.release()
                else: lease.backing_closed()
            except BaseException as error:
                if first is None: first = error
        self._files.clear()
        if first is not None: raise first
    def _record(self, row):
        if self.cancelled(): raise InterruptedError("complete media semantic proof retired")
        raw = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self._sha.update(len(raw).to_bytes(8, "little")); self._sha.update(raw)
    def artifact_directory(self):
        if not self.private or self.directory is None: raise RuntimeError("private media effect sink is not owned")
        value = self.directory.lstat()
        if self.directory.is_symlink() or (value.st_dev, value.st_ino) != self._directory_identity:
            raise RuntimeError("private media effect directory was replaced")
        return self.directory
    def output(self, role, neural):
        modality = "audio" if role == "companion-audio" else self.modality
        view = media_view(neural, modality)
        def observe():
            with self.brain().resource_policy.reserve_ram(2*1024*1024, "bounded exact media output proof"):
                self._record(["output", role, list(view.shape), str(view.dtype), view.neural_sha256()]); self.outputs += 1
        self._observe(observe)
    def preview(self, progress, neural, mime_type, payload, details):
        view = media_view(neural, self.modality)
        def observe():
            with self.brain().resource_policy.reserve_ram(2*1024*1024, "bounded exact media preview proof"):
                self._record(["preview", float(progress), list(view.shape), str(view.dtype), view.neural_sha256(),
                    mime_type, payload_sha256(payload), details]); self.previews += 1
        self._observe(observe)
    def sink(self, _progress, mime_type, payload, _details):
        # Match the real synchronous content-addressed host-preview durability,
        # but publish no notification, URL, journal or learning event.
        suffix = {"image/png": ".png", "image/apng": ".png", "audio/wav": ".wav", "video/mp4": ".mp4"}.get(mime_type)
        if suffix is None: raise ValueError("private media preview has an unknown container")
        directory = self.artifact_directory() / "previews"; directory.mkdir(exist_ok=True)
        target = directory / (payload_sha256(payload) + suffix)
        write_media_payload(target, payload)
        with target.open("rb") as handle: os.fsync(handle.fileno())
        if existing_media_sha256(target) != payload_sha256(payload): raise RuntimeError("private preview lost its exact encoded bytes")
    def finish(self, produced, random):
        if self.failure is not None: raise RuntimeError("original media evidence declined without cancelling its result") from self.failure
        result = produced["result"]
        if self.outputs < 1 or self.previews != produced["previewCount"]:
            raise RuntimeError("complete media did not visit its actual neural/preview boundaries")
        if self.resource_paused: raise RuntimeError("private complete media reached a resource watermark")
        # Paths, UUIDs, elapsed time and transport embedding are not neural
        # content. All semantic output/timing/codec evidence remains exact.
        result_contract = {key: value for key, value in result.items() if key not in {"path", "dataUrl"}}
        if existing_media_sha256(Path(result["path"])) != result["artifactSha256"]:
            raise RuntimeError("complete media artifact checksum changed before its receipt")
        if self.private:
            final = Path(result["path"]); value = final.lstat(); identity = (value.st_dev, value.st_ino)
            for row in self._files:
                if row[2] == identity:
                    row[0].commit(path=final, promised_bytes=value.st_size, retain_open_backing=True); row[1] = final
        self._record(["final", result_contract, random.summary()])
        descriptor = {"shape": result["shape"], "mimeType": result["mimeType"], "mediaTimeline": result.get("mediaTimeline"),
            "companionAudio": bool(result.get("synchronizedAudio", {}).get("generated")), "previewCount": produced["previewCount"]}
        return SimpleNamespace(contract=self._sha.hexdigest(), descriptor=descriptor)


class NativeWholeMediaOperationOps(NativeCachedDecoderOps):
    def __init__(self, brain, host):
        super().__init__(brain); self.host, self._scopes = host, {}
        self.serial_thread = threading.get_ident()

    def _codec(self):
        gateway = getattr(self.brain, "_native_operation_auto_codec_gateway", None)
        if gateway is None: return None
        with gateway.condition:
            configuration = gateway.selected_configuration
            return None if configuration is None else dict(configuration)

    def _closure(self, modalities):
        from . import modalities as m, connected_regions as c, model as b, packed_connections as p
        roots, hubs, identities, seen = [], [], [], set()
        def visit(hub, path):
            if id(hub) in seen: raise ValueError("cyclic native media region ownership")
            seen.add(id(hub)); hubs.append(hub)
            if any(name in hub.__dict__ for name in ("generate", "_generate", "generate_scaled", "_resize", "eval", "train",
                    "_generate_scaled_image", "_generate_scaled_audio", "_generate_scaled_video")):
                raise ValueError("overridden media producer is not a closed native route")
            if type(hub) is c.ConnectedModalityHub:
                decoder = hub.connected_decoder
                roots.extend([(path + ".exchange", decoder.workspace_exchange), (path + ".selector", hub.selector)])
                identities.append([path, "connected", hub.config.to_dict(), list(hub.widths)])
                for index, region in enumerate(decoder.regions):
                    identity = region.identity
                    identities.append([path, index, identity.region_id, identity.parent_id, identity.generation_sha256])
                    visit(region.modalities, path + ".r" + str(index))
            elif type(hub) is m.ModalityHub:
                identities.append([path, "leaf", hub.config.to_dict()])
                for modality in modalities:
                    if not getattr(hub.config, modality + "_enabled", False): raise ValueError("native media pack is disabled")
                    pack = getattr(hub, modality)
                    if type(pack) is not {"image": m.TinyImageImagination, "audio": m.TinyAudioCodec, "video": m.TinyVideoImagination}[modality]:
                        raise ValueError("unknown native media generator")
                    roots.append((path + "." + modality, pack))
            else: raise ValueError("unknown same-brain media owner")
            seen.remove(id(hub))
        visit(self.brain.modalities, "modalities")
        allowed = {m.TinyImageImagination, m.TinyAudioCodec, m.TinyVideoImagination, m.VectorQuantizer,
            m.PackedTernaryTable, m.TernaryTransformerBlock, m.TernaryLatentTransformer, m.LiquidTemporalGate,
            c.RegionExchange, b.PackedAdaptiveBitLinear, b.PackedAdaptiveBitConv1d, b.PackedAdaptiveBitConv2d,
            b.PackedAdaptiveBitConv3d, b.PackedAdaptiveBitConvTranspose1d, b.PackedAdaptiveBitConvTranspose2d,
            b.PackedAdaptiveBitConvTranspose3d, p.PackedContactPage, p.PooledParallelConnections,
            torch.nn.Sequential, torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.SiLU, torch.nn.Tanh, torch.nn.LayerNorm}
        globals_ = torch.nn.modules.module
        if any(getattr(globals_, name, {}) for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks")):
            raise ValueError("global callbacks prevent private media effects isolation")
        modules, names, pager = {}, {}, self.brain.core_pager
        packed_types = {b.PackedAdaptiveBitLinear, b.PackedAdaptiveBitConv1d, b.PackedAdaptiveBitConv2d,
            b.PackedAdaptiveBitConv3d, b.PackedAdaptiveBitConvTranspose1d, b.PackedAdaptiveBitConvTranspose2d,
            b.PackedAdaptiveBitConvTranspose3d, p.PackedContactPage}
        for path, root in roots:
            for suffix, module in root.named_modules():
                if (type(module) not in allowed or any(name in module.__dict__ for name in ("forward", "generate", "encode", "_condition", "_evolve"))
                        or any(getattr(module, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks"))
                        or getattr(module, "_online_transaction", None) is not None):
                    raise ValueError("unreviewed/transactional native media callback")
                if type(module) is torch.nn.SiLU and module.inplace: raise ValueError("in-place media module is not private")
                if type(module) is torch.nn.LayerNorm and module.elementwise_affine: raise ValueError("floating learned media normalization is unsupported")
                if type(module) in packed_types and (module not in pager._owners or getattr(module, "_native_core_pager", None) is not pager
                        or not pager._owners[module].loaded): raise ValueError("media has an unloaded or foreign packed learned owner")
                modules[id(module)] = module; names.setdefault(id(module), path + ("."+suffix if suffix else ""))
        packed = [module for module in modules.values() if module in pager._owners]
        if not packed: raise ValueError("media has no proven loaded packed owners")
        mode_types = allowed | {m.ModalityHub, c.ConnectedModalityHub, c._ConnectedMediaModule, m.TinyVisionEncoder, torch.nn.AdaptiveAvgPool2d}
        for module in self.brain.modalities.modules():
            if type(module) not in mode_types or any(name in module.__dict__ for name in ("eval", "train")):
                raise ValueError("media eval traversal has an unreviewed mode callback")
        return {"modules": modules, "names": names, "identities": identities, "hubs": hubs, "packed": packed}

    def _closure_sha(self, closure):
        rows = []
        fields = {"in_features", "out_features", "in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation",
            "groups", "output_padding", "image_size", "samples", "frames", "channels", "latent_size", "latent_samples",
            "rows", "dimensions", "max_tokens", "heads", "head_dim", "eps", "normalized_shape", "elementwise_affine"}
        for key, module in closure["modules"].items():
            structural = {name: getattr(module, name) for name in sorted(fields | set(getattr(module, "_packed_structural_fields", ())))
                if type(getattr(module, name, None)) in {int, float, bool, str, tuple, list}}
            rows.append([closure["names"][key], str(key), type(module).__module__ + "." + type(module).__name__, structural])
        return digest([closure["identities"], rows])

    def registry(self):
        rows = []
        for modality in ("image", "audio", "video"):
            try:
                closure = self._closure((modality,)); rows.append([modality, self._closure_sha(closure)])
            except ValueError: continue
        return digest([rows, self._codec()]), rows

    def work_kinds(self):
        _sha, rows = self.registry()
        return ["native.media." + modality + "-artifact" for modality, _sha in rows if modality != "video" or self._codec() is not None]

    def eligible(self, brain, prepared):
        try:
            if brain is not self.brain or threading.get_ident() != self.serial_thread: return False
            if getattr(brain, "_native_execution_restore_required", False): return False
            value = prepared["idea"]
            if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.dtype != torch.float32 or value.shape[0] != 1: return False
            if canonical_device(value.device) != canonical_device(brain.device): return False
            if prepared["modality"] == "video" and self._codec() is None: return False
            closure = self._closure(tuple(dict.fromkeys((prepared["modality"], *(["audio"] if prepared["companionAudio"] else [])))))
            return all(not brain.core_pager._owners[value].pins for value in closure["packed"])
        except (KeyError, ValueError, RuntimeError): return False

    def _scope(self, closure):
        key = self._closure_sha(closure)
        if key not in self._scopes:
            self._scopes[key] = self.brain.core_pager.observe_learning_owners(closure["packed"])
        return self._scopes[key]
    def retire_observation(self):
        for scope in self._scopes.values(): scope.close()
        self._scopes.clear()
    def operation_state(self, operation, session): return self.host._state(operation["closure"], operation["scope"], session)

    def capture_operation(self, prepared, producer, previews, session, review):
        modalities = tuple(dict.fromkeys((prepared["modality"], *(["audio"] if prepared["companionAudio"] else []))))
        closure, idea = self._closure(modalities), prepared["idea"]
        previews_bound = int(prepared["primaryPlan"].estimated_previews) if prepared["primaryPlan"] is not None else 4
        size = idea.numel()*idea.element_size() + 16384 + len(closure["modules"])*1024 + (previews_bound+2)*1024
        self.brain.resource_policy.require_device(idea.device, idea.numel()*idea.element_size(), "retained actual media cue")
        lease = self.brain.resource_policy.reserve_ram(size, "actual prepared media cue and owner metadata")
        try:
            guard = _tensor_guard(idea); copied = idea.detach().clone(); lease.mark_allocated(size)
            if guard != _tensor_guard(idea): raise RuntimeError("real media cue changed during its private retention")
            scope, codec = self._scope(closure), self._codec() if prepared["modality"] == "video" else None
            shape = {"modality": prepared["modality"], "primary": _plan_layout(prepared["primaryPlan"]),
                "companion": _plan_layout(prepared["audioPlan"]), "previews": bool(previews), "codec": codec,
                "settings": prepared["settings"],
                "nativeWindows": [self.brain.config.image_size, self.brain.config.audio_samples, self.brain.config.video_frames]}
            layout_sha, closure_sha = digest(shape), self._closure_sha(closure)
            operation = {"closure": closure, "closureSha": closure_sha, "scope": scope, "lease": lease, "producer": producer,
                "prepared": {**prepared, "idea": copied}, "cueGuard": _tensor_guard(copied), "session": session,
                "codec": codec, "layout": shape, "previews": bool(previews), "released": False}
            binding, state = review["binding"], self.operation_state(operation, session)
            identity = {"ownerId": self.brain.brain_id, "ownerSessionId": session, "workKind": "native.media."+prepared["modality"]+"-artifact",
                "stageCoverage": list(MEDIA_STAGES), "workloadSha256": digest([self.host._hash_tensor(copied), prepared["seed"], shape]),
                "geometrySha256": digest([closure_sha, shape]), "samplingContractSha256": digest(["same-home-media-RNG", canonical_device(idea.device), "fresh-seeded-generator-per-native-unit"]),
                **{key: binding[key] for key in ("policySha256", "topologySha256", "budgetSha256")},
                "cache": {"heat": "warm", "logicalStateSha256": closure_sha, "backendInitialization": "warm"}, "nativeState": state}
            operation.update(identity=identity, stratumKey=digest([identity["geometrySha256"], identity["samplingContractSha256"],
                binding["policySha256"], binding["topologySha256"], binding["budgetSha256"]]),
                planContract={"numericGate": "exact-media-neural-timeline-rng-and-encoded-artifact-equality",
                    "reuseBasis": "same-validated-media-operator-output-layout-current-owner-policy-inference",
                    "mediaLayout": {"modality": prepared["modality"], "shapeSha256": layout_sha}})
            self._residency(operation)
            return operation
        except BaseException: lease.release(); raise

    def _residency(self, operation):
        pager = self.brain.core_pager
        operation["ownerObjects"] = {module: pager._owners[module] for module in operation["closure"]["packed"]}
        operation["residency"] = [(module, pager._owners[module].device) for module in operation["closure"]["packed"]]
    def after_genuine(self, operation, baseline, session):
        if baseline.contract is None: raise RuntimeError("real media could not retain an exact optional qualification proof")
        if not self.unchanged(operation, session): raise RuntimeError("real media mutated a learned owner; private pool declined")
        self._residency(operation); operation["baseline"] = baseline
    def unchanged(self, operation, session):
        modalities = tuple(dict.fromkeys((operation["prepared"]["modality"], *(["audio"] if operation["prepared"]["companionAudio"] else []))))
        return (not operation["released"] and _tensor_guard(operation["prepared"]["idea"]) == operation["cueGuard"]
            and self._closure_sha(self._closure(modalities)) == operation["closureSha"]
            and self.operation_state(operation, session) == operation["identity"]["nativeState"]
            and (operation["codec"] is None or operation["codec"] == self._codec()))
    def release_operation(self, operation):
        if operation["released"]: return
        operation["released"] = True; operation["lease"].release(); operation["producer"] = None; operation["prepared"]["idea"] = None

    @contextmanager
    def _placement(self, candidate, operation, stopped, *, private):
        brain, pager, closure = self.brain, self.brain.core_pager, operation["closure"]
        self.admit_candidate(candidate)
        controls = [(module, dict(module._buffers), dict(module._parameters), module.training,
            hasattr(module, "_native_compute_device"), getattr(module, "_native_compute_device", None))
            for module in closure["modules"].values() if module not in pager._owners]
        packed = [(module, hasattr(module, "_native_compute_device"), getattr(module, "_native_compute_device", None), module.training)
            for module in closure["packed"]]
        # generate_scaled calls eval() on the public hub. Include even its
        # unselected attached packs/wrappers in this tiny mode-only snapshot;
        # no unrelated learned tensors are copied, moved or read.
        modes = [(module, module.training) for module in brain.modalities.modules()]
        original_observations = getattr(pager, "_observations", None)
        consumer = _PerceptionTarget(self, candidate, stopped)
        active = _ACTIVE_REGIONAL.set((consumer, consumer._lease["epoch"], candidate["executionDevice"]))
        try:
            if private and isinstance(original_observations, dict): pager._observations = {}
            with transient_regional_scope(consumer):
                for module, _present, _compute, _training in packed:
                    module._native_compute_device = torch.device(candidate["executionDevice"])
                for module, buffers, parameters, _training, _present, _compute in controls:
                    if any(isinstance(value, torch.Tensor) for value in parameters.values()): raise ValueError("floating learned media control cannot be remapped")
                    for name, value in buffers.items():
                        if isinstance(value, torch.Tensor) and name != "_next_page": module._buffers[name] = self.move_activity(value, candidate["executionDevice"])
                yield
        finally:
            consumer.active = False; _ACTIVE_REGIONAL.reset(active)
            if private and isinstance(original_observations, dict): pager._observations = original_observations
            try:
                for module, old_device in operation["residency"]:
                    with pager._lock:
                        original = operation["ownerObjects"][module]
                        if pager._owners.get(module) is not original or not original.loaded: raise RuntimeError("native media original packed owner was replaced")
                        pager._observe_mutations(module, original)
                        actual = None if original.device is None else canonical_device(original.device)
                        expected = None if old_device is None else canonical_device(old_device)
                    if actual != expected:
                        pager.evict(module)
                        if old_device is not None:
                            with pager.use(module, old_device): pass
            except BaseException as error:
                brain._native_execution_restore_required = True
                raise OperationAutoRestoreRequired("exact media packed residency restore failed; explicit Unload required") from error
            finally:
                for module, buffers, parameters, training, present, compute in controls:
                    module._buffers.clear(); module._buffers.update(buffers); module._parameters.clear(); module._parameters.update(parameters); module.training = training
                    if present: module._native_compute_device = compute
                    elif hasattr(module, "_native_compute_device"): del module._native_compute_device
                for module, present, compute, training in packed:
                    module.training = training
                    if present: module._native_compute_device = compute
                    elif hasattr(module, "_native_compute_device"): del module._native_compute_device
                for module, training in modes: module.training = training
                pager._regional_limits()

    def _run(self, operation, preview, directory, cancelled, started_at, *, private=False, effects_out=None):
        effects = _MediaEffects(self.brain, operation["prepared"]["modality"], private=private, directory=directory,
            cancelled=(cancelled or (lambda: False)) if private else self.host.parent_cancelled)
        if effects_out is not None: effects_out.append(effects)
        random = MediaRandomScope(self.brain.device, transfer=self.move_activity, cancelled=effects.cancelled)
        with random.activate(), media_artifact_scope(effects):
            produced = operation["producer"](operation["prepared"]["idea"], effects.sink if private and operation["previews"] else preview,
                directory, cancelled, started_at)
            try: actual = effects.finish(produced, random)
            except BaseException:
                if private: raise
                actual = SimpleNamespace(contract=None, descriptor=None)
        return produced, actual

    def execute_genuine(self, operation, preview, directory, cancelled, started_at):
        return self._run(operation, preview, directory, cancelled, started_at)

    def _codec_scope(self, operation, stopped):
        if operation["codec"] is None: return nullcontext()
        from codec_gateway import CodecOwner
        gateway = self.brain._native_operation_auto_codec_gateway
        return gateway.pinned_scope(CodecOwner("media-auto-"+operation["identity"]["workloadSha256"][:32], self.brain.brain_id),
            stopped.is_set, operation["codec"])

    @media_storage_lifetime
    def execute_private(self, candidate, operation, observer, stopped):
        if stopped.is_set(): raise InterruptedError("private media retired before its actual owner boundary")
        def stage(name):
            if observer is not None: observer.stage_completed(name)
            if stopped.is_set(): raise InterruptedError("private media retired at its actual stage boundary")
        root = self.brain._live_paging_cache_directory / "media-auto-private"
        self.brain.resource_policy.require_disk(65536, "private complete media directory")
        root.mkdir(parents=True, exist_ok=True)
        prepared = operation["prepared"]; original = prepared["idea"]; effects = []
        token = PRIVATE_OPERATION.set(True)
        try:
            with tempfile.TemporaryDirectory(prefix="operation-", dir=root) as directory, torch.no_grad(), native_parameter_read_only(), \
                    self.brain.working_attention_pager.activate(cancelled=stopped.is_set), self._codec_scope(operation, stopped):
                with self._placement(candidate, operation, stopped, private=True):
                    stage(MEDIA_STAGES[0]); prepared["idea"] = self.move_activity(original, candidate["executionDevice"])
                    stage(MEDIA_STAGES[1])
                    _produced, actual = self._run(operation, None, Path(directory), stopped.is_set, time.perf_counter(), private=True, effects_out=effects)
                    for name in MEDIA_STAGES[2:5]: stage(name)
                stage(MEDIA_STAGES[5])
                return actual
        finally:
            prepared["idea"] = original; PRIVATE_OPERATION.reset(token)
            for sink in effects: sink.close_files()

    def execute_live(self, candidate, operation, preview, directory, cancelled, started_at, begin):
        class Flags:
            def is_set(_self): return bool(cancelled is not None and cancelled())
        prepared, original = operation["prepared"], operation["prepared"]["idea"]
        try:
            with torch.no_grad(), native_parameter_read_only(), self._placement(candidate, operation, Flags(), private=False):
                prepared["idea"] = self.move_activity(original, candidate["executionDevice"])
                begin(); return self._run(operation, preview, directory, cancelled, started_at)
        finally: prepared["idea"] = original

    def check_output(self, trial, descriptor):
        if trial.descriptor != descriptor: raise RuntimeError("fresh media route changed its complete output/timeline descriptor")
