"""Actual ingestion learner waves: private qualification, one genuine update.

This consumes the same prepared token windows and the existing local learner;
it does not build a model, teach a fixture answer, time a primitive or replay a
checkpoint commit. Input traversal, STDP/graph admission, source receipts and
checkpoint/cursor publication remain on their original once-only caller path.
Cross-rank jobs have no reviewed all-rank protocol and stay ordinary.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
import weakref
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

import torch

from .decoder_execution_native import _tensor_guard
from .operation_auto_policy import OperationAutoRestoreRequired
from .operation_auto_registry import PRIVATE_OPERATION, TRAINING_KINDS, TRAINING_STAGES
from .operation_transfer_observation import recurrent_transfer_observation
from .training_cow import TrainingNamespace, PrivateTrainingSQLite, private_training_diagnostics, training_owner_read_only
from .training_randomness import TrainingRandomScope, current_rng_state, training_execution
from .whole_path_execution import COUNTERS, digest, select_whole_path_candidate
from .whole_path_scheduler import select_complete_operation, qualification_allowed, owned_operation_choice
from .outer_learning_pipeline import genuine_learning_operation
from .window_wave_buffer import PreparedWindowWave

TRAINING_KIND = TRAINING_KINDS[0]
_JOB = ContextVar("native_ingestion_operation_job_intent", default=None)


def _local_world_size():
    distributed = torch.distributed
    if distributed.is_available() and distributed.is_initialized(): return int(distributed.get_world_size())
    try: return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError: return 0


def capture_training_intent(brain, job_id, source=None):
    """A current live review is intent, never restart-persistent authority."""
    controller = getattr(brain, "_native_operation_auto", None)
    value = {"format": "omni-native-training-job-intent", "version": 1,
        "jobId": str(job_id), "ownerId": brain.brain_id, "worldSize": _local_world_size(),
        "enabled": False, "reason": "no-explicit-current-operation-Auto-review"}
    if not job_id: return {**value, "reason": "actual-app-job-identity-required"}, None
    if value["worldSize"] != 1: return {**value, "reason": "no-reviewed-cross-rank-whole-wave-protocol"}, None
    if controller is None: return value, None
    with controller._lock:
        active = (not controller._closed and not controller._cancel.is_set() and not controller._active
            and controller._phase in {"awaiting-operation", "active", "failed-fallback"} and controller._review is not None)
        if not active: return value, None
        review = copy.deepcopy(controller._review)
        operation, revision = controller._operation, controller._authority_revision
    if TRAINING_KIND not in review["supportedWorkKinds"]:
        return {**value, "reason": "actual-prepared-learner-wave-kind-not-reviewed"}, None
    try:
        if controller.ops.hardware_binding(controller.owner_session) != review["binding"]:
            return {**value, "reason": "current-native-owner-device-policy-review-is-stale"}, None
    except Exception as error:
        if getattr(error, "native_owner_restore_required", False): raise
        return {**value, "reason": "current-native-review-unavailable:"+type(error).__name__}, None
    selected = {key: source[key] for key in ("manifestId", "manifestHash", "transactionKey", "epoch", "contentHash")
        if source is not None and key in source and isinstance(source[key], (str, int)) and not isinstance(source[key], bool)}
    value.update({"enabled": True, "reason": "explicit-current-loaded-owner-operation-Auto",
        "ownerSessionId": controller.owner_session, "operationId": operation,
        "authorityRevision": revision, "binding": review["binding"],
        "reviewSha256": digest(review), "currentCandidate": review["currentCandidate"],
        "source": selected, "savedIntentRestoresAuthority": False})
    return value, {"controller": weakref.ref(controller), "review": review, "revision": revision, "operation": operation}


@contextmanager
def training_job(brain, job_id, source=None):
    intent, authority = capture_training_intent(brain, job_id, source)
    # Source-free per-attempt provenance, not another neural generation or a
    # resumable capability token. Read-back never arms/restores this policy.
    if job_id:
        from .persistence import atomic_write_json
        path = Path(brain.storage_path)/"engine"/"native-operation-jobs"/(digest([job_id, uuid.uuid4().hex])+".json")
        brain.resource_policy.require_disk(len(json.dumps(intent))*3+65536, "native learner job provenance")
        atomic_write_json(path, intent)
        brain.events.append("native-training-job-intent", intent, job_id=job_id)
    token = _JOB.set({"brain": brain, "intent": intent, "authority": authority})
    try: yield intent
    finally: _JOB.reset(token)


class _Flags:
    def __init__(self, controller): self.controller = controller
    def is_set(self): return self.controller._cancel.is_set() or self.controller.ops.parent_cancelled()


class PrivateWaveEffects:
    """Discardable effect/cursor namespace, never the live source journal."""
    def __init__(self, directory, policy, cancelled):
        self.path, self.policy, self.cancelled = Path(directory)/"effects.sqlite3", policy, cancelled
        self.quota = policy.reserve_spill(65536, "private learner effects namespace")
        self.quota.bind_path(self.path)
        try:
            self.connection = sqlite3.connect(self.path)
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("CREATE TABLE effects (ordinal INTEGER PRIMARY KEY, kind TEXT, payload TEXT)")
            info = self.path.stat(); self.identity = info.st_dev, info.st_ino
            self.quota.commit(path=self.path, promised_bytes=info.st_size, retain_open_backing=True)
        except BaseException:
            if hasattr(self, "connection"): self.connection.close()
            self.path.unlink(missing_ok=True); self.quota.release(); raise
    def append(self, kind, payload=None, **kwargs):
        if self.cancelled(): raise InterruptedError("private learner effects stopped")
        encoded = json.dumps([payload, kwargs], sort_keys=True, separators=(",", ":"), allow_nan=False)
        quota = self.policy.reserve_spill(len(encoded)*2+8192, "private learner effect")
        quota.bind_path(self.path)
        try:
            self.connection.execute("INSERT INTO effects(kind,payload) VALUES (?,?)", (str(kind), encoded))
            pages = int(self.connection.execute("PRAGMA page_count").fetchone()[0])
            size = pages*int(self.connection.execute("PRAGMA page_size").fetchone()[0])
            quota.commit(path=self.path, promised_bytes=size, retain_open_backing=True)
            self.quota.backing_closed(); self.quota = quota
        except BaseException: quota.release(); raise
        return "private-effect"
    def summary(self):
        result = hashlib.sha256()
        for row in self.connection.execute("SELECT kind,payload FROM effects ORDER BY ordinal"):
            raw = json.dumps(row, separators=(",", ":")).encode(); result.update(len(raw).to_bytes(8, "little")); result.update(raw)
        return result.hexdigest()
    def close(self):
        self.connection.close()
        try:
            info = self.path.stat()
            if not self.path.is_symlink() and (info.st_dev, info.st_ino) == self.identity: self.path.unlink()
        except FileNotFoundError: pass
        self.quota.backing_closed()


@contextmanager
def genuine_wave_effects(brain, directory, cancelled):
    """Observe normal genuine effects while forwarding each to its real owner."""
    original = brain.events
    sink = PrivateWaveEffects(directory, brain.resource_policy, cancelled)
    class Forward:
        def append(self, kind, payload=None, **kwargs):
            sink.append(kind, payload, **kwargs)
            return original.append(kind, payload, **kwargs)
        def __getattr__(self, name): return getattr(original, name)
    try:
        brain.events = Forward(); yield sink
    finally: brain.events = original; sink.close()


class NativeTrainingWaveOps:
    def __init__(self, brain, host):
        self._owner, self.host = weakref.ref(brain), host
        self.total = {name: 0 for name in COUNTERS}
    @property
    def brain(self):
        value = self._owner()
        if value is None: raise RuntimeError("native learner Auto owner was released")
        return value
    def _closure(self):
        from .optimizers import PackedOnlyOptimizer
        from .connected_transaction import _ACTIVE
        if _local_world_size() != 1:
            raise ValueError("whole-wave Auto has no reviewed cross-rank ownership protocol")
        if _ACTIVE.get() is not None:
            raise ValueError("whole-wave Auto awaits the original graph transaction boundary")
        from .packed_collective_hooks import packed_derivative_collective_active
        if packed_derivative_collective_active():
            raise ValueError("whole-wave Auto has no reviewed cross-rank derivative protocol")
        if type(self.brain._optimizer) is not PackedOnlyOptimizer or self.brain._optimizer_offloaded:
            raise ValueError("whole-wave Auto requires its resident stateless native optimizer")
        # Reuse the reviewed complete ordinary/connected region inventory;
        # extend it with the real root memory/temporal objective and every
        # owner whose stability event is drained by the existing learner.
        closure = self.host.cached_operations._closure()
        modules = dict(closure["modules"])
        roots = list(self.brain._slow_transaction_modules().items())
        for root in self.brain._trainable_modules():
            for module in root.modules(): modules[id(module)] = module
        for _name, root in roots:
            for module in root.modules(): modules[id(module)] = module
        mutable = {id(module) for _name, root in roots for module in root.modules()}
        # Text corpus forward does not execute any attached modality pack.
        # Their modes/event counters still receive private dictionaries, but
        # do not copy unrelated media activity or move it onto the candidate.
        for hub in [self.brain.modalities, *(region.modalities for _name, region in closure["regions"])]:
            mutable.difference_update(id(module) for module in hub.modules())
        mutable.difference_update(closure["fixed"])
        for module in modules.values():
            if any(isinstance(value, torch.Tensor) for value in module._parameters.values()):
                raise ValueError("whole-wave Auto cannot substitute dense learned tensors")
            if any(getattr(module, name, {}) for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks")):
                raise ValueError("whole-wave Auto cannot privately repeat opaque neural callbacks")
        if any(self.brain.core_pager._owners[module].pins for module in modules.values() if module in self.brain.core_pager._owners):
            raise ValueError("whole-wave Auto requires the actual unpinned serial learner boundary")
        return {**closure, "modules": modules, "learningRoots": roots, "mutableControls": mutable}
    def registry(self):
        closure = self._closure()
        cached = self.host.cached_operations._registry()[0]
        fields = {"in_features", "out_features", "dimensions", "input_size", "hidden_size", "mode", "solver_steps", "eps",
            "in_channels", "out_channels", "kernel_size", "stride", "padding", "dilation", "groups", "num_embeddings", "embedding_dim"}
        additional = [[name, type(root).__module__+"."+type(root).__name__,
            [[path, str(id(module)), type(module).__module__+"."+type(module).__name__,
              {key: getattr(module, key) for key in sorted(fields | set(getattr(module, "_packed_structural_fields", ())))
               if type(getattr(module, key, None)) in {str, int, float, bool, tuple}},
              [[key, list(value.shape), str(value.dtype)] for key, value in module._buffers.items()
               if isinstance(value, torch.Tensor) and key not in module._non_persistent_buffers_set]]
             for path, module in root.named_modules()]] for name, root in closure["learningRoots"]]
        return digest([cached, additional, "local-prepared-wave-native-learner-v1"])
    def work_kinds(self):
        try: self.registry(); return [TRAINING_KIND]
        except (ValueError, RuntimeError): return []
    def counters(self): return dict(self.total)
    def recurrent_transfer(self, size): self.total["activationTransferBytes"] += int(size)
    def move(self, value, target):
        target = torch.device(target)
        if value.device == target: return value
        size = value.numel()*value.element_size()
        with self.brain._regional_transfer_reservation(target, size, "whole learner actual activity transfer") as held:
            result = value.to(device=target, non_blocking=False)
            if held is not None: held.mark_allocated(size)
        self.total["activationTransferBytes"] += size
        return result
    def _tensor_hash(self, value, flags):
        if not value.is_contiguous(): raise ValueError("whole-wave proof requires contiguous native/control fields")
        output = hashlib.sha256()
        block = max(1, self.brain.core_pager.chunk_bytes//value.element_size())
        flat, guard = value.detach().reshape(-1), _tensor_guard(value)
        with self.brain.resource_policy.reserve_ram(min(value.numel(), block)*value.element_size()*3+4096, "bounded exact whole learner state proof"):
            for offset in range(0, value.numel(), block):
                if flags.is_set(): raise InterruptedError("whole learner state proof stopped")
                payload = flat[offset:offset+block].to("cpu").contiguous().view(torch.uint8).numpy()
                output.update(memoryview(payload))
        if _tensor_guard(value) != guard: raise RuntimeError("learner state changed during its exact proof")
        return output.hexdigest()
    def state(self, roots, flags):
        values, seen = [], set()
        for name, root in roots:
            for path, module in root.named_modules():
                if id(module) in seen: continue
                seen.add(id(module))
                pager = getattr(module, "_native_core_pager", None)
                context = pager.use(module, "cpu") if pager is not None and module in pager._owners else nullcontext()
                with training_owner_read_only(), context:
                    values.append([name+"."+path,
                        [[key, list(value.shape), str(value.dtype), self._tensor_hash(value, flags)]
                         for key, value in sorted(module._buffers.items()) if isinstance(value, torch.Tensor)
                         and key not in module._non_persistent_buffers_set],
                        int(getattr(module, "_pending_stability_events", 0)), int(getattr(module, "_revision", 0))])
        return digest([values, self.brain.counters])
    def graph_guard(self, closure, flags):
        from .connected_recall_binding import recall_source_snapshot
        result = []
        for name, _region, binding in closure["bindings"]:
            snapshot = recall_source_snapshot(binding,
                reserve=lambda size: self.brain.resource_policy.reserve_ram(size, "whole learner graph revision fence"),
                check_cancelled=lambda: (_ for _ in ()).throw(InterruptedError("learner graph fence stopped")) if flags.is_set() else None)
            dirty = getattr(binding.authoritative_source.synapses, "_dirty", None)
            if not isinstance(dirty, dict): raise ValueError("whole learner graph has an opaque mutable synapse overlay")
            hasher = hashlib.sha256()
            with self.brain.resource_policy.reserve_ram(4096+len(dirty)*128, "exact learner synapse-overlay index"):
                for identifier in sorted(dirty):
                    if flags.is_set(): raise InterruptedError("learner synapse-overlay proof stopped")
                    raw = json.dumps([identifier, dirty[identifier]], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                    with self.brain.resource_policy.reserve_ram(len(raw)*4+4096, "one exact learner synapse-overlay record"):
                        hasher.update(len(raw).to_bytes(8, "little")); hasher.update(raw)
            result.append([name, snapshot, hasher.hexdigest()])
        return result
    def _graph_paths(self, closure):
        paths, stores = set(), {}
        for _name, _region, binding in closure["bindings"]:
            memory = binding.authoritative_source
            for item in (memory.neurons, memory.neuron_vectors, memory.synapses, memory.assemblies, getattr(memory.assemblies, "index", None)):
                if item is not None and hasattr(item, "__dict__"):
                    stores[id(item)] = item
                    if getattr(item, "path", None) is not None: paths.add(Path(item.path))
        return paths, list(stores.values())
    def genuine_graph_summary(self, closure, flags):
        paths, _stores = self._graph_paths(closure)
        # No connection is requested, so this reads originals in mode=ro and
        # creates no copied database. The actual declared graph rows are
        # compared with the private COW result, not merely a revision number.
        proof = PrivateTrainingSQLite(paths, self.brain._live_paging_cache_directory,
            self.brain.resource_policy, flags.is_set)
        try: return proof.summary()
        finally: proof.close()
    @contextmanager
    def _private_graph(self, closure, namespace, directory, flags):
        paths, stores = self._graph_paths(closure)
        database = PrivateTrainingSQLite(paths, Path(directory)/"graph", self.brain.resource_policy, flags.is_set, read_only=True)
        original, leases, views = [], [], []
        try:
            for store in stores:
                before = store.__dict__; private = {}
                for name, value in before.items():
                    private[name] = None if name in {"_recall_graph_observer", "_native_transmission_observer"} else namespace.controls.value(value)
                original.append((store, before)); store.__dict__ = private
            for _key, region, binding in closure["bindings"]:
                view, private, lease = binding.private(cancelled=flags.is_set, transfer=self.move,
                    reserve_metadata=lambda size: self.brain.resource_policy.reserve_ram(size, "private learner graph activity metadata"))
                views.append(view); leases.append(lease); region._recall = private
            with database.activate(): yield database
        finally:
            with ExitStack() as cleanup:
                for lease in leases: cleanup.callback(lease.release)
                for store, before in original: cleanup.callback(setattr, store, "__dict__", before)
                cleanup.callback(database.close)
                for view in views: cleanup.callback(view.release_paged_recall_scratch)
    def execute_private(self, candidate, wave, closure, baseline_rng, invoke, observer, flags):
        from .cached_whole_operation_native import _Target
        from .regional_execution import _ACTIVE_REGIONAL, transient_regional_scope
        from .working_attention_paging import WorkingAttentionPager
        self.host.admit_candidate(candidate)
        def stage(value):
            if observer is not None: observer.stage_completed(value)
            if flags.is_set(): raise InterruptedError("whole learner private operation stopped")
        root = Path(self.brain._live_paging_cache_directory)/"training-auto-private"
        self.brain.resource_policy.require_disk(131072, "private whole learner namespace")
        root.mkdir(parents=True, exist_ok=True)
        home, attention = self.brain.device, self.brain.working_attention_pager
        random = TrainingRandomScope(home, baseline_rng, transfer=self.move, cancelled=flags.is_set,
            reserve_conversion=lambda size: self.brain.resource_policy.reserve_ram(size, "whole learner input dtype activity"))
        token = PRIVATE_OPERATION.set(True)
        try:
            with tempfile.TemporaryDirectory(prefix="wave-", dir=root) as directory, ExitStack() as stack:
                namespace = stack.enter_context(TrainingNamespace(self.brain, closure["modules"].values(), directory,
                    self.brain.resource_policy, flags.is_set, mutable_controls=closure.get("mutableControls")))
                stage(TRAINING_STAGES[0])
                private_attention = WorkingAttentionPager(Path(directory)/"activity",
                    resident_budget_bytes=attention.resident_budget_bytes, scratch_budget_bytes=attention.scratch_budget_bytes,
                    device_tile_budget_bytes=attention.device_tile_budget_bytes, resource_policy=self.brain.resource_policy,
                    page_tokens=attention.page_tokens, compute_device=torch.device(candidate["executionDevice"]))
                stack.callback(private_attention.close); self.brain.working_attention_pager = private_attention
                for module in closure["modules"].values():
                    if getattr(module, "working_attention_pager", None) is attention:
                        module.working_attention_pager = private_attention
                effects = PrivateWaveEffects(directory, self.brain.resource_policy, flags.is_set)
                stack.callback(effects.close); self.brain.events = effects
                graph = stack.enter_context(self._private_graph(closure, namespace, directory, flags))
                stack.enter_context(private_training_diagnostics(namespace.pager, namespace.controls))
                private_epochs = namespace.pager.observe_learning_owners(list(namespace.pager.sources))
                stack.callback(private_epochs.close)
                stage(TRAINING_STAGES[1])
                for module in closure["modules"].values():
                    if module in namespace.pager._owners: module._native_compute_device = torch.device(candidate["executionDevice"])
                    elif id(module) in closure.get("mutableControls", closure["modules"]):
                        for name, value in tuple(module._buffers.items()):
                            if isinstance(value, torch.Tensor) and name != "_next_page": module._buffers[name] = self.move(value, candidate["executionDevice"])
                target = _Target(self.host, candidate, flags)
                active = _ACTIVE_REGIONAL.set((target, target._lease["epoch"], candidate["executionDevice"]))
                try:
                    with transient_regional_scope(target), private_attention.activate(cancelled=flags.is_set), random.activate(), \
                            recurrent_transfer_observation(self.recurrent_transfer), \
                            training_execution(candidate["executionDevice"], private=True, batches=wave.batches):
                        stage(TRAINING_STAGES[2]); report = invoke()
                    stage(TRAINING_STAGES[3])
                    state = self.state(closure["learningRoots"], flags)
                    graph_state = [self.graph_guard(closure, flags), graph.summary()]
                    effect_state = effects.summary()
                    contract = digest([state, report, random.summary(), effect_state, graph_state, wave.identity])
                    stage(TRAINING_STAGES[4])
                finally:
                    target.active = False; _ACTIVE_REGIONAL.reset(active)
            counters = namespace.pager.execution_counters()
            for key in counters: self.total[key] += counters[key]
            self.total["attentionReadBytes"] += private_attention._metrics["bytesRead"]
            self.total["attentionWriteBytes"] += private_attention._metrics["bytesWritten"]
            stage(TRAINING_STAGES[5])
            if not torch.equal(current_rng_state(home), baseline_rng):
                raise RuntimeError("live learner RNG changed outside its private generator scope")
            return {"contract": contract, "state": state, "report": report, "rng": random.summary(), "graph": graph_state, "effects": effect_state}
        finally: PRIVATE_OPERATION.reset(token)

    @contextmanager
    def genuine_placement(self, candidate, closure, flags):
        """One live update, without changing the brain's home/device policy.

        Packed learned changes stay with their original owners. Only target
        hints and unmodified transient controls are restored; sparse births,
        strength/resistance changes, counters and normal observations survive.
        """
        from .cached_whole_operation_native import _Target
        from .regional_execution import _ACTIVE_REGIONAL, transient_regional_scope, canonical_device
        brain, pager = self.brain, self.brain.core_pager
        self.host.admit_candidate(candidate)
        packed = [(module, pager._owners[module], pager._owners[module].device)
            for module in closure["modules"].values() if module in pager._owners]
        hints, controls = [], []
        target = _Target(self.host, candidate, flags)
        token = _ACTIVE_REGIONAL.set((target, target._lease["epoch"], candidate["executionDevice"]))
        try:
            with transient_regional_scope(target):
                for module in closure["modules"].values():
                    if module in pager._owners:
                        hints.append((module, "_native_compute_device" in vars(module), getattr(module, "_native_compute_device", None)))
                        module._native_compute_device = torch.device(candidate["executionDevice"])
                    elif id(module) in closure.get("mutableControls", closure["modules"]):
                        for name, value in tuple(module._buffers.items()):
                            # _next_page is the mutable persistent contact ID
                            # counter. Keep its actual CPU owner, never undo it.
                            if not isinstance(value, torch.Tensor) or name == "_next_page": continue
                            moved = self.move(value, candidate["executionDevice"])
                            if moved is not value:
                                controls.append((module, name, value, _tensor_guard(moved)))
                                module._buffers[name] = moved
                yield
        finally:
            target.active = False; _ACTIVE_REGIONAL.reset(token)
            restore_error = None
            try:
                for module, name, original, guard in controls:
                    actual = module._buffers[name]
                    if _tensor_guard(actual) != guard:
                        raise RuntimeError("genuine learner changed a remapped non-packed control")
                    module._buffers[name] = original
                for module, original_owner, old_device in packed:
                    if pager._owners.get(module) is not original_owner:
                        raise RuntimeError("genuine learner replaced an original packed owner")
                    actual = None if original_owner.device is None else canonical_device(original_owner.device)
                    expected = None if old_device is None else canonical_device(old_device)
                    if actual != expected:
                        pager.evict(module)
                        if old_device is not None:
                            with pager.use(module, old_device): pass
                for module in closure["modules"].values():
                    for name in ("last_generation_neural_state", "last_region_activity"):
                        value = getattr(module, name, None)
                        if isinstance(value, torch.Tensor): setattr(module, name, self.move(value, brain.device))
                        elif isinstance(value, tuple) and all(isinstance(item, torch.Tensor) for item in value):
                            setattr(module, name, tuple(self.move(item, brain.device) for item in value))
            except BaseException as error: restore_error = error
            finally:
                for module, present, value in hints:
                    if present: module._native_compute_device = value
                    else: module.__dict__.pop("_native_compute_device", None)
                pager._regional_limits()
            if restore_error is not None:
                brain._native_execution_restore_required = True
                raise OperationAutoRestoreRequired("genuine learner target could not return exact original residency") from restore_error


class _PreparedWave:
    def __init__(self, brain, experiences, schedule, directory, flags):
        policy = brain.resource_policy
        self.wave, self.pending_disk, self.current_disk = None, None, None
        self.ram_leases, self.preparing = [], True
        def settle_disk():
            if self.pending_disk is None or self.wave is None or self.wave.path is None: return
            size = (int(self.wave.database.execute("PRAGMA page_count").fetchone()[0])
                * int(self.wave.database.execute("PRAGMA page_size").fetchone()[0]))
            self.pending_disk.bind_path(self.wave.path)
            self.pending_disk.commit(path=self.wave.path, promised_bytes=size, retain_open_backing=True)
            if self.current_disk is not None: self.current_disk.backing_closed()
            self.current_disk, self.pending_disk = self.pending_disk, None
        def reserve(ram_bytes=0, disk_bytes=0):
            if flags.is_set(): raise InterruptedError("whole learner preparation stopped")
            if ram_bytes:
                policy.require_device("cpu", ram_bytes, "prepared whole learner input")
                if self.preparing:
                    lease = policy.reserve_ram(ram_bytes, "retained prepared learner input")
                    lease.mark_allocated(ram_bytes); self.ram_leases.append(lease)
            if disk_bytes:
                settle_disk()
                self.pending_disk = policy.reserve_spill(disk_bytes, "prepared whole learner input spill")
        hasher, geometries = hashlib.sha256(), hashlib.sha256()
        try:
            # Escrow the SQLite schema before the common spill writer creates
            # its first file. A RAM-only wave releases this unused promise.
            self.pending_disk = policy.reserve_spill(65536, "prepared learner optional SQLite schema")
            self.wave = PreparedWindowWave(physical_batch=int(schedule["physicalBatchRecords"]),
                ram_budget=max(0, brain.working_attention_pager.resident_budget_bytes//4), directory=Path(directory)/"input",
                reserve=reserve, allow_spill=True)
            for encoded, vectors in brain._streaming_experience_window_batches(experiences,
                    physical_batch=int(schedule["physicalBatchRecords"]), sequence_tokens=int(schedule["trainingSequenceTokens"])):
                for ids, cue in zip(encoded, vectors):
                    for value in (ids, cue):
                        size = value.numel()*value.element_size()
                        hasher.update(size.to_bytes(8, "little"))
                        with policy.reserve_ram(size+4096, "bounded actual learner input row layout"):
                            flat = value.detach().reshape(-1)
                            block = max(1, min(size, 262144)//value.element_size())
                            for start in range(0, value.numel(), block):
                                if flags.is_set(): raise InterruptedError("whole learner input hash stopped")
                                raw = flat[start:start+block].to("cpu").contiguous().numpy().tobytes(); hasher.update(raw)
                    shape = json.dumps([list(ids.shape), list(cue.shape)], separators=(",", ":")).encode()
                    geometries.update(len(shape).to_bytes(8, "little")); geometries.update(shape)
                    # The common spill container has a typed noise column.
                    # Local-wave noise is drawn by its private home-RNG scope,
                    # so retain only a fixed framing scalar, never stale noise.
                    self.wave.append(ids, cue.reshape(-1), torch.zeros(1, dtype=torch.float32))
                    settle_disk()
                    if self.wave.database is not None:
                        for lease in self.ram_leases: lease.release()
                        self.ram_leases.clear()
            self.identity = {"inputSha256": hasher.hexdigest(), "geometrySha256": geometries.hexdigest(),
                "windows": self.wave.window_count, "labels": self.wave.label_count, "schedule": dict(schedule)}
            if not self.wave: raise ValueError("prepared whole learner wave contains no windows")
            self.preparing = False
        except BaseException: self.close(); raise
    def batches(self):
        for batch in self.wave.batches():
            if self.current_disk is not None: self.current_disk.refresh_physical()
            yield [row[0] for row in batch], [row[1] for row in batch]
    def close(self):
        if self.wave is not None: self.wave.close()
        if self.pending_disk is not None: self.pending_disk.release(); self.pending_disk = None
        if self.current_disk is not None: self.current_disk.backing_closed(); self.current_disk = None
        for lease in self.ram_leases: lease.release()
        self.ram_leases.clear()


def consume_streaming_training_wave(brain, experiences, schedule, locked):
    if not qualification_allowed(): return None
    job = _JOB.get()
    if job is None or job.get("retired") or job["brain"] is not brain or not job["intent"]["enabled"]: return None
    authority = job["authority"]; controller = authority["controller"]()
    if controller is None or not locked or schedule is None: return None
    # Do not advertise an operation until its main/shared contract is composed.
    if TRAINING_KIND not in authority["review"]["supportedWorkKinds"]: return None
    return _consume_prepared(brain, experiences, schedule, job, controller)


def _consume_prepared(brain, experiences, schedule, job, controller):
    """A comparison refusal returns None: the original learner then runs once."""
    authority, review, flags = job["authority"], job["authority"]["review"], _Flags(controller)
    ops = getattr(controller.ops, "training_operations", None)
    if ops is None: return None
    def current():
        with controller._lock:
            if (controller._closed or controller._cancel.is_set() or controller._operation != authority["operation"]
                    or controller._authority_revision != authority["revision"]): raise InterruptedError("learner job Auto authority retired")
        if controller.owner_session != job["intent"]["ownerSessionId"] or controller.ops.hardware_binding(controller.owner_session) != review["binding"]:
            raise RuntimeError("learner job owner, topology or selected policy changed")
    def invoke(): return brain._optimize_streaming_experience_batch(experiences, learning_schedule=schedule, schedule_locked=True)
    root, wave, acquired = Path(brain._live_paging_cache_directory)/"training-auto-input", None, False
    genuine_started = False
    metadata_lease = None
    router_stack = None
    receipt_id = "native-training-wave-"+uuid.uuid4().hex
    try:
        current()
        metadata_size = max(65536, len(brain.core_pager._owners)*8192)
        metadata_lease = brain.resource_policy.reserve_ram(metadata_size, "complete prepared learner registry and namespace metadata")
        closure = ops._closure(); metadata_lease.mark_allocated(metadata_size)
        router_stack = ExitStack()
        router_pagers = {id(region.router.synapses._router_state_pager): region.router.synapses._router_state_pager
            for _name, region in closure.get("regions", [])}
        for pager in router_pagers.values(): router_stack.enter_context(pager.operation())
        with controller._lock:
            if controller._active: return None
            controller._active, controller._kind, acquired = True, TRAINING_KIND, True
        brain.resource_policy.require_disk(65536, "actual prepared whole learner input namespace")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="wave-", dir=root) as directory:
            wave = _PreparedWave(brain, experiences, schedule, directory, flags)
            rng = current_rng_state(brain.device).clone()
            registry = ops.registry()
            # Native owner epochs are a live serial fence, not a saved model.
            scope = brain.core_pager.observe_learning_owners([value for value in closure["modules"].values() if value in brain.core_pager._owners])
            try:
                graph_before = ops.graph_guard(closure, flags)
                state = {"kind": "live-owner-epochs", "bindingSha256": digest([registry, scope.snapshot(), graph_before]),
                    "ownerSessionId": controller.owner_session, "complete": True, "ownerVersions": scope.snapshot()}
                identity = {"ownerId": brain.brain_id, "ownerSessionId": controller.owner_session,
                    "workKind": TRAINING_KIND, "stageCoverage": list(TRAINING_STAGES),
                    "workloadSha256": digest(wave.identity), "geometrySha256": digest([registry, wave.identity["geometrySha256"], schedule]),
                    "samplingContractSha256": digest(["same-home-learning-RNG", hashlib.sha256(rng.numpy().tobytes()).hexdigest()]),
                    **{key: review["binding"][key] for key in ("policySha256", "topologySha256", "budgetSha256")},
                    "cache": {"heat": "cold", "logicalStateSha256": digest([registry, "private-first-use-native-copies"]), "backendInitialization": "warm"},
                    "nativeState": state}
                baseline, receipts, issued, issued_count = None, [], "0"*64, 0
                candidates = [review["currentCandidate"], *[item for item in review["candidates"] if item != review["currentCandidate"]]]
                for sample in range(-1, controller._samples):
                    for index, candidate in enumerate(candidates):
                        current()
                        if scope.snapshot() != state["ownerVersions"] or ops.graph_guard(closure, flags) != graph_before:
                            raise OperationAutoRestoreRequired("original learner owner/RNG changed before canonical publication")
                        if not torch.equal(current_rng_state(brain.device), rng):
                            raise RuntimeError("original learner RNG advanced outside the private operation")
                        with controller._lock:
                            controller._phase = "qualifying"
                            controller._progress = {"candidate": index+1, "candidates": len(candidates), "sample": max(0, sample+1),
                                "samplesPerCandidate": controller._samples, "stage": "warming" if sample < 0 else "measuring"}
                        observed = identity if sample >= 0 else {**identity, "cache": {**identity["cache"], "backendInitialization": "unknown"}}
                        failure, actual = None, None
                        with controller.ops.observation_resources(candidate, identity["budgetSha256"]) as resources:
                            observer = controller.recorder.begin(observed, controller.primitive(candidate), synchronize=controller.ops.synchronize,
                                counters=ops.counters, resources=resources)
                            try:
                                actual = ops.execute_private(candidate, wave, closure, rng, invoke, observer, flags)
                                if baseline is None: baseline = actual
                                if actual["contract"] != baseline["contract"]: raise RuntimeError("complete learner mutation, activity, RNG or effects differs")
                                current()
                            except BaseException as error: failure = error
                            after_versions = scope.snapshot()
                            after_graph = ops.graph_guard(closure, flags) if not flags.is_set() else graph_before
                            after_state = {**state, "ownerVersions": after_versions,
                                "bindingSha256": digest([registry, after_versions, after_graph])}
                            receipt = observer.finish(native_state_after=after_state,
                                output_contract_sha256=actual["contract"] if failure is None else None,
                                status="cancelled" if failure and flags.is_set() else "failed" if failure else "completed",
                                error=type(failure).__name__ if failure else None)
                        receipt["exactCompleteLearnerStateAndRngEquality"] = failure is None
                        receipt["receiptSha256"] = digest({key: value for key, value in receipt.items() if key != "receiptSha256"})
                        controller.ops.append_receipt(receipt_id, {"phase": "warmup" if sample < 0 else "measured", "observation": receipt,
                            "jobProvenanceSha256": digest(job["intent"]), "originalLearningWasCommitted": False})
                        if sample >= 0:
                            receipts.append(receipt)
                            issued, issued_count = digest([issued, receipt["receiptSha256"]]), issued_count+1
                        if failure is not None: raise failure
                decision = select_complete_operation(controller, receipts, selector=select_whole_path_candidate, identity=identity, current_candidate=controller.primitive(review["currentCandidate"]),
                    minimum_samples=controller._samples, verify_receipt=lambda row: row["observerId"] == controller.recorder.observer_id)
                if controller.ops.receipt_head(receipt_id) != (issued, issued_count):
                    raise RuntimeError("whole learner issued receipt head changed")
                current()
                selected = review["currentCandidate"]
                if decision["decision"] == "propose-at-quiescence":
                    selected = next(value for value in candidates if value["planSha256"] == decision["proposedCandidate"]["planSha256"])
                    if not controller.ops.auto_allowed(selected): selected = review["currentCandidate"]
                shape = digest([registry, wave.identity["geometrySha256"], schedule])
                operation = {"identity": identity, "planContract": {
                    "numericGate": "exact-prepared-wave-native-state-rng-and-effects-equality",
                    "reuseBasis": "same-prepared-wave-current-owner-policy-no-cross-wave-inference",
                    "trainingLayout": {"shapeSha256": shape}}}
                controller._store_plan(digest([TRAINING_KIND, identity["geometrySha256"], review["binding"]]), operation,
                    selected, None, (issued, issued_count), descriptor={"shapeSha256": shape, "workloadSha256": identity["workloadSha256"]})
                controller.ops.append_receipt(receipt_id, {"phase": "private-pool-qualified", "comparison": decision,
                    "selectedGenuineRoute": selected, "candidateAdoptionSupported": True,
                    "outerCheckpointAndCursorNotReplayed": True})
                class GenuineFlags:
                    # Stop Auto retires qualification, not a canonical update
                    # already in progress. The job's own Stop still applies.
                    def is_set(self): return controller.ops.parent_cancelled()
                current()
                if scope.snapshot() != state["ownerVersions"] or ops.graph_guard(closure, flags) != graph_before:
                    raise OperationAutoRestoreRequired("learner source changed before once-only genuine update")
                genuine_started = True
                with controller._lock: controller._committing = True
                with owned_operation_choice(controller, identity, selected), \
                        genuine_learning_operation(brain, controller, identity, selected), \
                        TrainingRandomScope(brain.device, rng, transfer=ops.move, cancelled=controller.ops.parent_cancelled,
                        reserve_conversion=lambda size: brain.resource_policy.reserve_ram(size, "whole learner input dtype activity")).activate() as random, \
                        training_execution(selected["executionDevice"], private=False, batches=wave.batches), \
                        recurrent_transfer_observation(ops.recurrent_transfer), \
                        genuine_wave_effects(brain, directory, controller.ops.parent_cancelled) as real_effects, \
                        ops.genuine_placement(selected, closure, GenuineFlags()):
                    result = invoke()
                    if (ops.state(closure["learningRoots"], GenuineFlags()) != baseline["state"]
                            or result != baseline["report"] or random.summary() != baseline["rng"]
                            or real_effects.summary() != baseline["effects"]
                            or [ops.graph_guard(closure, GenuineFlags()), ops.genuine_graph_summary(closure, GenuineFlags())] != baseline["graph"]):
                        raise RuntimeError("genuine learner state differs from its complete private proof")
                    random.commit()
                with controller._lock:
                    controller._phase, controller._reason = (("cancelled", "Auto-stopped-current-canonical-learner-update-finished-once")
                        if controller._cancel.is_set() else ("active", "prepared-learner-qualified-current-target-update-once-outer-cursor-unchanged"))
                return result
            finally: scope.close()
    except BaseException as error:
        if getattr(error, "native_owner_restore_required", False) or genuine_started: raise
        job["retired"] = True
        with controller._lock:
            controller._phase = "cancelled" if controller._cancel.is_set() else "failed-fallback"
            controller._reason = "prepared-learner-comparison-kept-ordinary:"+type(error).__name__
        return None
    finally:
        cleanup_errors = []
        for cleanup in ([wave.close] if wave is not None else []) + ([router_stack.close] if router_stack is not None else []) \
                + ([metadata_lease.release] if metadata_lease is not None else []):
            try: cleanup()
            except Exception as error: cleanup_errors.append(type(error).__name__)
        if acquired:
            with controller._lock:
                controller._active = controller._committing = False; controller._progress = None
                if cleanup_errors:
                    # A private backing can remain charged for later reclaim;
                    # it cannot undo a verified canonical update or leave the
                    # user's composer trapped in a running Auto state.
                    controller._reason = "private-learner-resource-retirement-pending:"+",".join(cleanup_errors)
