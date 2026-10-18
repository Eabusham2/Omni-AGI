"""Actual streamed whole-prefix callbacks on the existing learned owners.

Each private trial begins with cache=None and owns only its newly created
activity/cache pages. The caller's old cache, context and real effects remain
untouched until the existing once-only adoption boundary.
"""
from __future__ import annotations

from contextlib import ExitStack

import torch

from .cached_activity_snapshot import CacheFork
from .cached_whole_operation_native import CompleteCachedTrial, NativeCachedWholeOperationOps
from .decoder_execution_native import _tensor_guard
from .prefill_whole_operation_auto import (PREFILL_KINDS, PREFILL_STAGES,
    PREFILL_NUMERIC_GATE, PREFILL_REUSE_BASIS)
from .regional_execution import canonical_device
from .whole_path_execution import digest


class NativeFullPrefixOperationOps(NativeCachedWholeOperationOps):
    def _closure(self):
        closure = super()._closure()
        closure["kind"] = PREFILL_KINDS[1 if closure["regions"] else 0]
        return closure

    def _prefix_geometry(self, window, closure):
        """Hash the actual complete prefix/query/page geometry, not a token cap."""
        from .connected_regions import ConnectedOmniDecoder
        pager, rows = self.brain.working_attention_pager, []
        batch, length = map(int, window.shape)
        for path, decoder in closure["decoders"]:
            connected = type(decoder) is ConnectedOmniDecoder
            # Connected generation visits one causal token at a time, including
            # every ordinary leaf borrowed by its parent. An ordinary root uses
            # exactly its admitted streamed prefill block calculation.
            query = 1 if connected or path != "decoder" else int(decoder._prefill_block_tokens(batch))
            blocks = [] if connected else [[index, int(block.attention.n_heads), int(block.attention.head_dim),
                int(block.attention.query_chunk_tokens), int(block.attention.key_chunk_tokens)]
                for index, block in enumerate(decoder.blocks)]
            rows.append({"owner": path, "kind": "connected" if connected else "ordinary",
                "configuredContext": int(decoder.config.max_seq_len), "queryBlockTokens": query,
                "actualQueryTokens": min(length, query), "prefixBlocks": (length + query - 1) // query,
                "lastQueryTokens": (length - 1) % query + 1, "blocks": blocks})
        return {"batch": batch, "prefixTokens": length, "inputDtype": str(window.dtype),
            "pageTokens": int(pager.page_tokens), "tileBudgetBytes": int(pager.device_tile_budget_bytes),
            "residentBudgetBytes": int(pager.resident_budget_bytes), "scratchBudgetBytes": int(pager.scratch_budget_bytes),
            "retainedContextPools": [{key: pool.get(key) for key in ("deviceId", "executionDevice", "memoryGroupId", "budgetBytes", "pageBytes",
                                         "partitionMode", "minimumPackedOwnerBytes", "controlReserveBytes", "transferReserveBytes")}
                                     for pool in getattr(pager, "_retained_context_pools", ())],
            "owners": rows}

    def valid_prefix(self, window, cue, cache):
        try:
            decoder = self.brain.decoder
            if (torch.is_grad_enabled() or decoder.training or not isinstance(window, torch.Tensor)
                    or window.dtype != torch.long or window.ndim != 2 or window.shape[0] < 1
                    or not 1 <= window.shape[1] <= decoder.config.max_seq_len
                    or canonical_device(window.device) != canonical_device(self.brain.device)):
                return False
            if self.brain.core_pager.status()["pinnedOwners"] or getattr(getattr(self.brain, "_regional_execution", None), "active", False):
                return False
            for name in ("_decoder_execution", "_cached_decoder_execution"):
                old = getattr(self.brain, name, None)
                if old is not None and old.status()["phase"] not in {"idle", "cancelled", "failed", "unsupported", "completed"}:
                    return False
            signature = decoder._generation_signature(cue)
            if signature is None: return False
            if cache is not None:
                if self._kind(cache) is None or getattr(cache, "_closed", False): return False
                # A valid retained one-token step belongs to the already frozen
                # consumer. Refills happen only when normal generation itself
                # would reject its old signature, length, or batch shape.
                if (cache.signature == signature and cache.length + 1 == window.shape[1]
                        and (cache.hidden_sum[0] if self._kind(cache) == "connected" else cache.hidden_sum).shape[0] == window.shape[0]):
                    return False
            closure = self._closure()
            if any(getattr(region.router.synapses, "_native_transmission_observer", None) is not None
                   for _key, region in closure["regions"]): return False
            self._prefix_geometry(window, closure)
            return True
        except (ValueError, RuntimeError, AttributeError, IndexError): return False

    def capture_prefix(self, window, cue, cache, generator, sampling, session, review, *, stopped):
        closure, stack = self._closure(), ExitStack()
        try:
            pagers = {id(region.router.synapses._router_state_pager): region.router.synapses._router_state_pager
                for _key, region in closure["regions"]}
            for pager in pagers.values(): stack.enter_context(pager.operation())
            scope = self._scope(closure)
            size = 131072 + len(closure["modules"]) * 1024
            lease = self.brain.resource_policy.reserve_ram(size, "full-prefix source/empty-cache/RNG metadata")
            stack.callback(lease.release)
            with self.proof_scope(stopped):
                guards, layout = (_tensor_guard(window), _tensor_guard(cue)), self._prefix_geometry(window, closure)
                rng = generator.get_state().clone()
                old_sha = None if cache is None else self.cache_sha(cache)
                state, binding = self._state(closure, scope, session), review["binding"]
                sampling_contract = {key: value for key, value in sampling.items() if key != "text_boundaries"}
                sampling_contract["textBoundaries"] = (None if sampling["text_boundaries"] is None
                    else [dict(vars(value)) for value in sampling["text_boundaries"]])
                geometry_sha = digest([scope["registry"], layout, None if cue is None else list(cue.shape)])
                identity = {"ownerId": self.brain.brain_id, "ownerSessionId": session, "workKind": closure["kind"],
                    "stageCoverage": list(PREFILL_STAGES), "geometrySha256": geometry_sha,
                    "workloadSha256": digest([self._hash_tensor(window), self._hash_tensor(cue), old_sha, self._hash_tensor(rng)]),
                    "samplingContractSha256": digest(sampling_contract),
                    **{key: binding[key] for key in ("policySha256", "topologySha256", "budgetSha256")},
                    "cache": {"heat": "cold", "logicalStateSha256": digest({"newCache": None, "retainedOriginal": old_sha}),
                        "backendInitialization": "warm"}, "nativeState": state}
                lease.mark_allocated(size)
            return {"window": window, "cue": cue, "cache": cache, "generator": generator, "sampling": sampling,
                "rng": rng, "guards": guards, "originalCacheSha256": old_sha, "closure": closure, "scope": scope,
                "stack": stack, "identity": identity, "prefixGeometry": layout,
                "originalActivity": self._activities(closure),
                "originalCorticalObservations": getattr(self.brain.core_pager, "_observations", None),
                "residency": [(module, self.brain.core_pager._owners[module].device) for module in closure["packed"]],
                "ownerObjects": {module: self.brain.core_pager._owners[module] for module in closure["packed"]},
                "stratumKey": digest([closure["kind"], geometry_sha]),
                "planContract": {"prefillLayout": {"shapeSha256": geometry_sha},
                    "numericGate": PREFILL_NUMERIC_GATE, "reuseBasis": PREFILL_REUSE_BASIS}}
        except BaseException: stack.close(); raise

    def prefix_unchanged(self, operation, session):
        with self.proof_scope():
            cache = operation["cache"]
            return (operation["guards"] == (_tensor_guard(operation["window"]), _tensor_guard(operation["cue"]))
                and torch.equal(operation["generator"].get_state(), operation["rng"])
                and self.operation_state(operation, session) == operation["identity"]["nativeState"]
                and self._prefix_geometry(operation["window"], operation["closure"]) == operation["prefixGeometry"]
                and (cache is None or not getattr(cache, "_closed", False)
                    and self.cache_sha(cache) == operation["originalCacheSha256"]))

    def release_prefix(self, operation): operation["stack"].close()

    def rebind_original(self, operation, session):
        # A rejected old cache is never relabelled current: the ordinary refill
        # must still see its original signature/length and close it normally.
        # An initial operation has no old cache to restore or manufacture.
        return None

    def execute(self, candidate, operation, observer, stopped):
        from .native_token_sampling import sample_native_token
        fork, transient = CacheFork(), None
        def stage(name):
            if observer is not None: observer.stage_completed(name)
            if stopped.is_set(): raise InterruptedError("complete full-prefix trial stopped at a real stage boundary")
        try:
            self.admit_candidate(candidate)
            # Streamed blocks keep their existing per-tile guards. Reserve only
            # the actual maximal one-block/readout envelope, not a full-prefix
            # expanded hidden/logit tensor or a synthetic reduced workload.
            rows = operation["prefixGeometry"]["owners"]
            query = max(row["actualQueryTokens"] for row in rows)
            estimate = (self.brain.decoder._forward_window_bytes if operation["closure"]["regions"]
                else self.brain.decoder._prefill_block_bytes)
            size = int(estimate(operation["window"].shape[0], query))
            size += int(operation["window"].shape[0] * self.brain.config.vocab_size * 16)
            self.brain.resource_policy.require_device(candidate["executionDevice"], size, "full-prefix streamed working/readout")
            if not candidate["executionDevice"].startswith("cuda:"):
                transient = self.brain.resource_policy.reserve_ram(size, "full-prefix streamed working/readout")
            observations = {}
            requests = getattr(self.brain.core_pager, "_observation_requests", {})
            if requests:
                observed_size = 4096 + sum(int(count) * 64 + 1024 for _start, count in requests.values())
                lease = self.brain.resource_policy.reserve_ram(observed_size, "private full-prefix viewport rows")
                fork.leases.append(lease); lease.mark_allocated(observed_size)
            stage(PREFILL_STAGES[0])
            with self._placement(candidate, operation, stopped, observations):
                window = self.move_activity(operation["window"], candidate["executionDevice"])
                cue = None if operation["cue"] is None else self.move_activity(operation["cue"], candidate["executionDevice"])
                stage(PREFILL_STAGES[1])
                before = self.brain.working_attention_pager._metrics["processedPrefillTokens"]
                # Bypass only the separate manual capture decorator, never the
                # actual streamed ordinary/connected prefix implementation.
                logits, returned = self.brain.decoder._generation_step_bounded(window, cue, None)
                fork.cache = returned
                if returned is None or returned.length != window.shape[1]:
                    raise RuntimeError("full-prefix candidate did not return the entire new cache")
                nodes = list(self._cache_nodes(self.brain.decoder, returned))
                if (any(node.length != window.shape[1] for _path, _decoder, node in nodes)
                        or self.brain.working_attention_pager._metrics["processedPrefillTokens"] - before != window.shape[1] * len(nodes)):
                    raise RuntimeError("full-prefix candidate omitted or replayed a causal regional frontier")
                for name in PREFILL_STAGES[2:5]: stage(name)
                home = operation["generator"].device
                logits = self.move_activity(logits, home)
                self._cache_home(returned, home)
                activity = self._home_activity(operation["closure"], fork, home)
                stage(PREFILL_STAGES[5])
            stage(PREFILL_STAGES[6])
            self._signatures(self.brain.decoder, fork.cache, operation["cue"])
            generator = torch.Generator(device=operation["generator"].device)
            generator.set_state(operation["rng"])
            sample = sample_native_token(logits, generator, **operation["sampling"])
            rng = generator.get_state().clone()
            stage(PREFILL_STAGES[7])
            self.synchronize(candidate["asyncDevices"])
            contract = digest({"prefixCache": self.cache_sha(fork.cache), "activity": self._activity_sha(activity),
                "sampling": self._sample_sha(sample, rng)})
            return CompleteCachedTrial(fork, sample, rng, activity, contract, observations)
        except BaseException: fork.close(); raise
        finally:
            if transient is not None: transient.release()
