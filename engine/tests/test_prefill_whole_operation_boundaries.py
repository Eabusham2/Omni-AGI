"""NEW raw/inert native prefix cases, no model/forward/Load/train/hardware.

Methods inspect lightweight owners and raw CPU tensors; the one execution
callback is explicitly inert, not an Omni decoder or learned neural kernel.
"""
from __future__ import annotations

import importlib
import threading
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext

import torch

from test_operation_auto_boundaries import PACKAGE, MODEL, ROOT, Owner, Policy

NATIVE = importlib.import_module(PACKAGE + ".prefill_whole_operation_native")
PREFIX = importlib.import_module(PACKAGE + ".prefill_whole_operation_auto")
WHOLE = importlib.import_module(PACKAGE + ".whole_path_execution")
GUARD = importlib.import_module(PACKAGE + ".decoder_execution_native")._tensor_guard


def owner_fixture():
    decoder = Owner(training=False, config=Owner(max_seq_len=100000, vocab_size=6),
        _generation_signature=lambda _cue: "source-signature", _prefill_block_tokens=lambda _batch: 7,
        _prefill_block_bytes=lambda batch, tokens: batch * tokens * 48, blocks=[])
    pager = Owner(page_tokens=11, device_tile_budget_bytes=65536, resident_budget_bytes=131072,
        scratch_budget_bytes=1048576, _metrics={"processedPrefillTokens": 0})
    brain = Owner(brain_id="raw-prefix-owner", decoder=decoder, config=decoder.config,
        resource_policy=Policy(), working_attention_pager=pager, core_pager=Owner(status=lambda: {"pinnedOwners": 0}),
        device=torch.device("cpu"))
    ops = object.__new__(NATIVE.NativeFullPrefixOperationOps)
    ops._owner, ops._template = weakref.ref(brain), None
    ops.host, ops._activation_transfer_bytes = Owner(_stop_event=threading.Event()), 0
    closure = {"regions": [], "decoders": [("decoder", decoder)], "modules": {}, "packed": [], "bindings": []}
    ops._closure = lambda: closure
    return brain, ops, closure


class RawPrefixBoundaryCases(unittest.TestCase):
    def test_actual_full_prefix_geometry_is_complete_and_scales_without_fixed_limit(self):
        brain, ops, closure = owner_fixture()
        block = Owner(attention=Owner(n_heads=2, head_dim=4, query_chunk_tokens=5, key_chunk_tokens=17))
        brain.decoder.blocks = [block]
        geometry = ops._prefix_geometry(torch.zeros((2, 71), dtype=torch.long), closure)
        self.assertEqual((geometry["batch"], geometry["prefixTokens"], geometry["pageTokens"]), (2, 71, 11))
        self.assertEqual(geometry["owners"][0]["queryBlockTokens"], 7)
        self.assertEqual(geometry["owners"][0]["prefixBlocks"], 11)
        self.assertEqual(geometry["owners"][0]["lastQueryTokens"], 1)
        self.assertEqual(geometry["owners"][0]["blocks"], [[0, 2, 4, 5, 17]])
        extended = ops._prefix_geometry(torch.zeros((1, 70001), dtype=torch.long), closure)
        self.assertEqual(extended["prefixTokens"], 70001)
        self.assertEqual(extended["owners"][0]["prefixBlocks"], 10001)
        self.assertNotEqual(WHOLE.digest(extended), WHOLE.digest(geometry))

    def test_only_genuine_initial_or_invalid_cache_can_enter_not_retained_step_or_manual_lease(self):
        brain, ops, _closure = owner_fixture()
        window = torch.tensor([[1, 2, 3]], dtype=torch.long)
        cache = MODEL._DecoderInferenceCache(2, "source-signature", [], [], torch.zeros((1, 2)), None)
        with torch.no_grad():
            self.assertTrue(ops.valid_prefix(window, None, None))
            self.assertFalse(ops.valid_prefix(window, None, cache))
            cache.length = 3; self.assertTrue(ops.valid_prefix(window, None, cache))
            cache.length = 2; cache.signature = "old-cue"; self.assertTrue(ops.valid_prefix(window, None, cache))
            brain._decoder_execution = Owner(status=lambda: {"phase": "active"})
            self.assertFalse(ops.valid_prefix(window, None, None))
            brain._decoder_execution = None
            brain.core_pager.status = lambda: {"pinnedOwners": 1}
            self.assertFalse(ops.valid_prefix(window, None, None))
        self.assertFalse(ops.valid_prefix(window, None, None))

    def test_source_window_cue_rng_and_prefix_geometry_have_independent_current_guards(self):
        brain, ops, closure = owner_fixture()
        window, cue = torch.tensor([[1, 2, 3]], dtype=torch.long), torch.tensor([[0.25, 0.75]])
        rng = torch.Generator().manual_seed(8)
        epoch = [1]
        ops.operation_state = lambda _operation, _session: {"epoch": epoch[0]}
        ops.proof_scope = lambda *args: nullcontext()
        operation = {"window": window, "cue": cue, "guards": (GUARD(window), GUARD(cue)), "generator": rng,
            "rng": rng.get_state().clone(), "cache": None, "closure": closure,
            "identity": {"nativeState": {"epoch": 1}}, "prefixGeometry": ops._prefix_geometry(window, closure)}
        self.assertTrue(ops.prefix_unchanged(operation, "owner-session"))
        brain.working_attention_pager.page_tokens += 1
        self.assertFalse(ops.prefix_unchanged(operation, "owner-session"))
        brain.working_attention_pager.page_tokens -= 1
        epoch[0] += 1; self.assertFalse(ops.prefix_unchanged(operation, "owner-session")); epoch[0] -= 1
        torch.rand((1,), generator=rng)
        self.assertFalse(ops.prefix_unchanged(operation, "owner-session")); rng.set_state(operation["rng"])
        cue.add_(1); self.assertFalse(ops.prefix_unchanged(operation, "owner-session"))

    def test_inert_actual_boundary_gets_new_empty_cache_each_trial_and_home_rng_is_not_reset(self):
        brain, ops, closure = owner_fixture()
        decoder, calls, stages, leases = brain.decoder, [], [], brain.resource_policy.leases
        original = torch.tensor([[9.0, 8.0]])
        decoder.last_generation_neural_state = original
        real_rng = torch.Generator().manual_seed(17)
        state = real_rng.get_state().clone()
        returned = []
        def inert_prefix(window, cue, cache):
            self.assertIsNone(cache)
            calls.append(window.clone())
            fresh = Owner(length=int(window.shape[1]), closed=False)
            fresh.close = lambda: setattr(fresh, "closed", True)
            returned.append(fresh)
            brain.working_attention_pager._metrics["processedPrefillTokens"] += int(window.shape[1])
            decoder.last_generation_neural_state = torch.tensor([[1.0, 2.0]])
            return torch.tensor([[0.1, 0.4, -0.1, 0.2, 0.5, 0.3]]), fresh
        decoder._generation_step_bounded = inert_prefix
        @contextmanager
        def placement(*_args):
            before = decoder.last_generation_neural_state
            try: yield
            finally: decoder.last_generation_neural_state = before
        ops._placement = placement
        ops.admit_candidate = lambda _candidate: None
        ops.move_activity = lambda value, target: value
        ops._cache_home = lambda cache, home: None
        ops._signatures = lambda decoder, cache, cue: None
        ops._cache_nodes = lambda decoder, cache: [("decoder", decoder, cache)]
        ops._home_activity = lambda closure, fork, home: [(decoder, "last_generation_neural_state", decoder.last_generation_neural_state.clone())]
        ops.cache_sha = lambda cache: WHOLE.digest(cache.length)
        ops._activity_sha = lambda activity: WHOLE.digest(activity[0][2].tolist())
        ops._sample_sha = lambda sample, rng: WHOLE.digest([sample[0].tolist(), sample[1], sample[2], rng.tolist()])
        ops.synchronize = lambda devices: None
        window = torch.tensor([[1, 2, 3]], dtype=torch.long)
        operation = {"window": window, "cue": None, "generator": real_rng, "rng": state,
            "closure": closure, "prefixGeometry": ops._prefix_geometry(window, closure),
            "sampling": {"temperature": 0.65, "top_k": 4, "noise": 0.0, "text_boundaries": None, "remaining_tokens": 6}}
        candidate, flags = {"executionDevice": "cpu", "asyncDevices": []}, threading.Event()
        first = ops.execute(candidate, operation, Owner(stage_completed=stages.append), flags)
        second = ops.execute(candidate, operation, Owner(stage_completed=lambda _stage: None), flags)
        self.assertEqual(stages, PREFIX.PREFILL_STAGES)
        self.assertEqual(first.contract, second.contract); self.assertIsNot(first.cache, second.cache)
        self.assertTrue(all(torch.equal(value, window) for value in calls)); self.assertEqual(len(calls), 2)
        self.assertIs(decoder.last_generation_neural_state, original)
        self.assertTrue(torch.equal(real_rng.get_state(), state)); self.assertFalse(torch.equal(first.rng_state, state))
        first.close(); second.close()
        self.assertTrue(all(cache.closed for cache in returned)); self.assertTrue(all(lease.closed for lease in leases))

    def test_inert_incomplete_prefix_and_stop_release_only_new_cache_not_source(self):
        brain, ops, closure = owner_fixture()
        ops.admit_candidate = lambda candidate: None
        ops.move_activity = lambda value, target: value
        ops._placement = lambda *args: nullcontext()
        window, caches = torch.tensor([[1, 2, 3]], dtype=torch.long), []
        def incomplete(window, cue, cache):
            self.assertIsNone(cache)
            fresh = Owner(length=2, closed=False); fresh.close = lambda: setattr(fresh, "closed", True)
            caches.append(fresh); return torch.zeros((1, 6)), fresh
        brain.decoder._generation_step_bounded = incomplete
        operation = {"window": window, "cue": None, "closure": closure,
            "prefixGeometry": ops._prefix_geometry(window, closure)}
        with self.assertRaisesRegex(RuntimeError, "entire new cache"):
            ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation, None, threading.Event())
        self.assertTrue(caches[0].closed)
        stop = threading.Event(); stop.set()
        with self.assertRaises(InterruptedError):
            ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation, None, stop)
        self.assertEqual(len(caches), 1)
        self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_rebind_does_not_relabel_rejected_old_cache_or_manufacture_a_new_one(self):
        brain, ops, _closure = owner_fixture()
        old = Owner(signature="rejected-cue", length=7)
        ops.rebind_original({"cache": old}, "session")
        ops.rebind_original({"cache": None}, "session")
        self.assertEqual((old.signature, old.length), ("rejected-cue", 7))


if __name__ == "__main__": unittest.main()
