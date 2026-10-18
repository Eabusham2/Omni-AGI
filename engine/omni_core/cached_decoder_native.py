"""Actual same-owner retained-cache callbacks; no model construction on import."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import struct
import uuid
from contextlib import contextmanager

import torch

from .cached_activity_snapshot import fork_decoder_cache
from .cached_decoder_execution import CachedDecoderExecution, bind_cached_decoder, WORK_KIND, STAGES
from .decoder_execution_native import NativeDecoderExecutionOps, _tensor_guard
from .native_token_sampling import sample_native_token
from .paged_substrate_writer import _identity_proof_supported
from .regional_execution import canonical_device
from .regional_execution import _ACTIVE_REGIONAL, transient_regional_scope, topology_identity
from .whole_path_execution import digest


class CachedTrial:
    def __init__(self, fork, sample, rng_state, contract):
        self.fork, self.sample, self.rng_state, self.contract = fork, sample, rng_state, contract
    @property
    def cache(self): return self.fork.cache
    @property
    def neural_state(self): return self.cache.neural_state
    def take_cache(self): return self.fork.take()
    def close(self):
        self.fork.close()
        self.sample = self.rng_state = None


class _TransientBlocks:
    """Exact registered block dispatch, never a saved/manual/global lease."""
    def __init__(self, ops, candidate, stopped):
        self.ops, self.candidate, self.stopped = ops, candidate, stopped
        self._policy = ops.brain.resource_policy
        self._lease = {"epoch": uuid.uuid4().hex}
        self.active = True
        self.blocks = {id(block): block for block in ops.brain.decoder.blocks}
    def policy_identity(self): return self._policy
    def packed_limits(self, policy):
        if policy is not self._policy or not self.active:
            raise RuntimeError("cached transient packed budget belongs to a different owner")
        brain = self.ops.brain
        result = {}
        for name in {self.candidate["executionDevice"], canonical_device(brain.device)} - {"cpu"}:
            device = next((row for row in brain._regional_topology()["devices"] if canonical_device(row.get("executionDevice", "")) == name), None)
            if device is None: raise RuntimeError("cached physical compute interface disappeared")
            selected = brain.resource_policy.device_budget(name, fresh=True)
            if not selected or not selected["enabled"] or not selected["capacityMeasured"]:
                raise RuntimeError("cached target lost its current measured allocation")
            paging = brain.core_pager.status()
            hot = brain._native_core_budget_for_device(paging, torch.device(name))["acceleratorCoreHotBytes"]
            result[name] = {"memoryGroupId": device["memoryGroupId"], "hotBytes": int(hot)}
        return result
    def execute(self, module, method, hidden, args, kwargs):
        if self.blocks.get(id(module)) is not module:
            raise RuntimeError("cached routing encountered an unreviewed decoder block")
        if self.stopped.is_set(): raise InterruptedError("cached operation stopped before region transfer")
        if module.training or torch.is_grad_enabled(): raise RuntimeError("cached routing is eval/read-only only")
        brain = self.ops.brain
        target, home = self.candidate["executionDevice"], str(hidden.device)
        token = _ACTIVE_REGIONAL.set((self, self._lease["epoch"], target))
        try:
            descriptor = brain._regional_describe(module)
            from .native_compute_profile import native_projection_tensor_budget
            width, ff = descriptor["dimensions"], descriptor["feedForward"]
            count = max(1, hidden.numel() // width)
            needed = max(native_projection_tensor_budget(count, first, second)["forwardTensorReserveBytes"]
                for first, second in ((width, width * 3), (width, ff * 2), (ff, width)))
            brain.working_attention_pager.admit_compute(needed + hidden.numel() * hidden.element_size() * 4 + 16,
                                                      "cached decoder block transient compute", device=torch.device(target))
            moved = self.ops.move_activity(hidden, target)
            output = method(module, moved, *args, **kwargs)
            if self.stopped.is_set(): raise InterruptedError("cached operation stopped after region computation")
            value = output[0] if isinstance(output, tuple) else output
            value = self.ops.move_activity(value, home)
            return (value, *output[1:]) if isinstance(output, tuple) else value
        finally:
            _ACTIVE_REGIONAL.reset(token)


class NativeCachedDecoderOps(NativeDecoderExecutionOps):
    def supported(self):
        supported, reason = super().supported()
        if not supported: return supported, reason
        old = getattr(self.brain, "_decoder_execution", None)
        if old is not None and old.status()["phase"] not in {"idle", "cancelled", "failed", "unsupported"}:
            return False, "finish-or-stop-the-separate-isolated-prefill-review-first"
        return True, "ordinary-paged-retained-cache-supported"

    def hardware_binding(self, session, *, topology=None):
        actual = self.brain._regional_topology()
        topology_sha = digest(topology_identity(actual))
        if topology is not None and digest(topology_identity(topology)) != topology_sha:
            raise RuntimeError("cached main-observed compute topology changed")
        registry, _pager = self._registry()
        return {"ownerSessionId": session, "policySha256": digest({"resource": self.brain._regional_policy_sha(),
                    "configuration": self.brain.config.to_dict()}), "topologySha256": topology_sha,
            "registrySha256": registry, "budgetSha256": digest(self._budgets()), "homeExecutionDevice": canonical_device(self.brain.device)}

    def valid_frontier(self, window, cue, cache):
        from .model import _DecoderInferenceCache
        from .working_attention_paging import PagedTensorSequence
        if (type(cache) is not _DecoderInferenceCache or cache.length + 1 != window.shape[1]
                or cache.signature != self.brain.decoder._generation_signature(cue)
                or not cache.attention or any(not isinstance(layer.key, PagedTensorSequence)
                    or not isinstance(layer.value, PagedTensorSequence) for layer in cache.attention)):
            return False
        supported, _reason = self.supported()
        if not supported: return False
        if self.brain.core_pager.status()["pinnedOwners"] or any(getattr(module, "_online_transaction", None) is not None
                for root in self.brain._trainable_modules() for module in root.modules()):
            return False
        return True

    def _page_sha(self, pager, identifier):
        pager.check_cancelled()
        page = pager._pages[identifier]
        if page.tensor is not None:
            value = page.tensor
            signature = ("hot", id(value), int(value._version), value.untyped_storage().data_ptr(), tuple(value.shape), str(value.dtype))
            old = getattr(page, "_cached_operation_proof", None)
            if old is not None and old[0] == signature: return old[1]
            hasher = hashlib.sha256()
            with self.brain.resource_policy.reserve_ram(min(page.size, 256 * 1024) * 2, "cached activity page checksum"):
                raw = value.detach().view(torch.uint8).reshape(-1)
                for start in range(0, raw.numel(), 256 * 1024):
                    pager.check_cancelled()
                    # Retained immutable pages may now own admitted VRAM.
                    # Hash one bounded host fragment, never a whole GPU cache.
                    hasher.update(memoryview(raw[start:start + 256 * 1024].to(device="cpu", non_blocking=False).contiguous().numpy()))
                if signature[2] != int(value._version): raise RuntimeError("hot cache activity changed while hashing")
            result = hasher.hexdigest()
        else:
            path = page.path
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode): raise RuntimeError("cached activity requires an owned regular file, not a symlink")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                key = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                if not stat.S_ISREG(opened.st_mode) or key(before) != key(opened) or opened.st_size != page.size:
                    raise RuntimeError("cached activity file was replaced")
                signature = ("file", *key(opened), str(page.dtype), page.shape, page.digest)
                old = getattr(page, "_cached_operation_proof", None)
                # Unknown/Windows filesystem paths still hash the exact opened
                # bytes each time; POSIX no-follow proof is metadata-bounded.
                cached = (os.name == "posix" and hasattr(os, "O_NOFOLLOW") and old is not None
                    and old[0] == signature and _identity_proof_supported(path, opened))
                if cached: result = old[1]
                else:
                    hasher = hashlib.sha256()
                    with self.brain.resource_policy.reserve_ram(512 * 1024, "exact opened cached activity hash"):
                        while True:
                            pager.check_cancelled()
                            block = os.read(descriptor, 256 * 1024)
                            if not block: break
                            hasher.update(block)
                            pager._metrics["bytesRead"] += len(block)
                    result = hasher.hexdigest()
                    if result != page.digest: raise RuntimeError("cached activity byte integrity failed")
                if key(os.fstat(descriptor)) != key(opened) or key(path.lstat()) != key(opened) or path.is_symlink():
                    raise RuntimeError("cached activity changed during its exact FD proof")
            finally:
                os.close(descriptor)
        pager.check_cancelled()
        page._cached_operation_proof = (signature, result)
        return result

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
        emit(cache.length)
        for chunk in cache.workspace:
            for value in (chunk.queries, chunk.mass, chunk.weighted_values): activity(value)
        for layer in cache.attention:
            activity(layer.key); activity(layer.value)
        for value in (cache.hidden_sum, cache.memory_hidden, cache.neural_state): activity(value)
        return hasher.hexdigest()

    def capture_frontier(self, window, cue, cache, generator, sampling, session, review):
        lease = self.brain.resource_policy.reserve_ram(128 * 1024 + len(self.brain.core_pager._owners) * 128,
                                                      "actual cached home RNG/residency capture")
        try:
            rng = generator.get_state().clone()
            if rng.numel() * rng.element_size() > 128 * 1024:
                raise RuntimeError("home RNG state exceeds the pre-admitted runtime state observation")
            lease.mark_allocated(rng.numel() * rng.element_size())
            original_sha = self.cache_sha(cache)
            sampling_descriptor = {key: value for key, value in sampling.items() if key != "text_boundaries"}
            sampling_descriptor["textBoundaries"] = None if sampling["text_boundaries"] is None else [dict(vars(value)) for value in sampling["text_boundaries"]]
            native = self.native_state(session)
            core_plan = [(module, owner.device) for module, owner in self.brain.core_pager._owners.items() if owner.device is not None]
            core_plan_sha = self._core_plan_sha(core_plan)
            binding = review["binding"]
            identity = {"ownerId": self.brain.brain_id, "ownerSessionId": session, "workKind": WORK_KIND,
                "stageCoverage": list(STAGES), "workloadSha256": digest({"window": self._hash_tensor(window),
                    "cue": self._hash_tensor(cue), "cache": original_sha, "rng": self._hash_tensor(rng)}),
                "geometrySha256": digest({"registry": binding["registrySha256"], "window": list(window.shape), "cacheLength": cache.length}),
                "samplingContractSha256": digest(sampling_descriptor), "policySha256": binding["policySha256"],
                "topologySha256": binding["topologySha256"], "budgetSha256": binding["budgetSha256"],
                "cache": {"heat": "warm", "logicalStateSha256": digest([original_sha, core_plan_sha]), "backendInitialization": "warm"}, "nativeState": native}
            return {"window": window, "cue": cue, "cache": cache, "generator": generator, "sampling": sampling,
                "rng": rng, "lease": lease, "originalCacheSha256": original_sha, "identity": identity,
                "guards": (_tensor_guard(window), _tensor_guard(cue)), "corePlan": core_plan, "corePlanSha256": core_plan_sha}
        except BaseException:
            lease.release(); raise

    def frontier_unchanged(self, frontier, session):
        return (frontier["guards"] == (_tensor_guard(frontier["window"]), _tensor_guard(frontier["cue"]))
            and torch.equal(frontier["generator"].get_state(), frontier["rng"])
            and self.native_state(session) == frontier["identity"]["nativeState"]
            and self.cache_sha(frontier["cache"]) == frontier["originalCacheSha256"]
            and self._core_plan_sha([(module, owner.device) for module, owner in self.brain.core_pager._owners.items() if owner.device is not None]) == frontier["corePlanSha256"])

    @staticmethod
    def _core_plan_sha(plan):
        return digest(sorted((str(id(module)), canonical_device(device)) for module, device in plan))

    def _restore_core_plan(self, frontier):
        # Only disposable PACKED residency moves. Model controls, working
        # pager/home, retained KV and learned epochs are never migrated/reset.
        pager = self.brain.core_pager
        pager.flush()
        pager._regional_limits()  # Retire the transient budget view, not prefs.
        for module, device in frontier["corePlan"]:
            with pager.use(module, device): pass

    def release_frontier(self, frontier): frontier["lease"].release()

    def rebind_original(self, frontier, session):
        # Only the known unchanged original can be rebound after disposable
        # sort/device-cache churn. Never reset a learned epoch or replay input.
        if self.frontier_unchanged(frontier, session):
            frontier["cache"].signature = self.brain.decoder._generation_signature(frontier["cue"])

    def move_activity(self, value, device):
        target = torch.device(device)
        if value.device == target: return value
        size = value.numel() * value.element_size()
        with self.brain._regional_transfer_reservation(target, size, "cached block actual activation transfer") as held:
            result = value.to(device=target, non_blocking=False)
            if held is not None: held.mark_allocated(size)
        self._activation_transfer_bytes += size
        return result

    def _copy_small_tensor(self, value):
        size = value.numel() * value.element_size()
        self.brain.resource_policy.require_device(value.device, size, "private cached sufficient-statistic copy")
        lease = self.brain.resource_policy.reserve_ram(size, "private cached sufficient-statistic copy")
        try:
            result = value.detach().clone()
            lease.mark_allocated(size)
            return result, lease
        except BaseException: lease.release(); raise

    def execute_cached(self, candidate, frontier, observation, stopped):
        from .model import _DecoderInferenceCache, _AttentionInferenceCache, _WorkspaceInferenceChunk
        from .working_attention_paging import PagedTensorSequence, SavedActivityTensor
        if stopped.is_set(): raise InterruptedError("cached comparison stopped before COW")
        fork = fork_decoder_cache(frontier["cache"],
            activity=lambda value: value.fork() if isinstance(value, (PagedTensorSequence, SavedActivityTensor)) else None,
            tensor=self._copy_small_tensor, workspace=_WorkspaceInferenceChunk, attention=_AttentionInferenceCache,
            decoder=_DecoderInferenceCache, reserve_metadata=lambda size: self.brain.resource_policy.reserve_ram(size, "private cached cursor containers"))
        returned = restoration_started = False
        try:
            observation.stage_completed("cache-cow-snapshot")
            fork.cache.signature = self.brain.decoder._generation_signature(frontier["cue"])
            consumer = _TransientBlocks(self, candidate, stopped)
            before = self.brain.working_attention_pager._metrics["processedPrefillTokens"]
            try:
                with transient_regional_scope(consumer):
                    logits, new_cache = self.brain.decoder._generation_step(frontier["window"], frontier["cue"], fork.cache)
                fork.cache = new_cache
            finally: consumer.active = False
            if new_cache is None or new_cache.length != frontier["cache"].length + 1 or self.brain.working_attention_pager._metrics["processedPrefillTokens"] - before != 1:
                raise RuntimeError("private cached operation unexpectedly replayed/refilled the prefix")
            for stage in ("cached-workspace", "routed-decoder-blocks", "home-readout"): observation.stage_completed(stage)
            generator = torch.Generator(device=frontier["generator"].device)
            generator.set_state(frontier["rng"])
            sample = sample_native_token(logits, generator, **frontier["sampling"])
            observation.stage_completed("same-home-sampling")
            rng = generator.get_state().clone()
            contract = digest({"cache": self.cache_sha(new_cache), "sample": self._sample_sha(sample, rng)})
            restoration_started = True
            self._restore_core_plan(frontier)
            observation.stage_completed("retained-cache-return")
            returned = True
            return CachedTrial(fork, sample, rng, contract)
        except BaseException:
            fork.close(); raise
        finally:
            if not returned and not restoration_started: self._restore_core_plan(frontier)

    def _sample_sha(self, sample, rng):
        token, log_probability, entropy = sample
        if not math.isfinite(log_probability) or not math.isfinite(entropy): raise RuntimeError("nonfinite cached sampling output")
        return digest({"token": self._hash_tensor(token), "postRng": self._hash_tensor(rng),
            "logProbabilityFloat64": struct.pack("<d", log_probability).hex(), "entropyFloat64": struct.pack("<d", entropy).hex()})

    def exact_output(self, reference, trial): return reference.contract == trial.contract

    def commit_sampling(self, frontier, trial):
        if not torch.equal(frontier["generator"].get_state(), frontier["rng"]): return None
        frontier["generator"].set_state(trial.rng_state)
        # Same home law, exact token/entropy/log-probability and exact post-RNG
        # state were checked. No second world callback or extra actual draw.
        trial.cache.signature = self.brain.decoder._generation_signature(frontier["cue"])
        return trial.sample

    def counters(self):
        result = super().counters()
        result["activationTransferBytes"] += self.brain.working_attention_pager._metrics["bytesTransferred"]
        return result


def create_cached_decoder_execution(brain):
    controller = CachedDecoderExecution(brain.brain_id, NativeCachedDecoderOps(brain),
        owner_session=brain._cached_decoder_execution_session)
    bind_cached_decoder(brain.decoder, controller)
    return controller
