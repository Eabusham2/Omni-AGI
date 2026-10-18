"""NEW complete-step source seams with constructor-free owners/raw CPU tensors.

No learned module/brain constructor, native packed kernel, Load, training,
hardware comparison or real generation runs. Inert callbacks are not neural
performance or capability evidence.
"""
from __future__ import annotations

import importlib
import ast
import copy
import threading
import types
import unittest
import weakref
from contextlib import contextmanager

import torch

from test_operation_auto_boundaries import PACKAGE, MODEL, CONNECTED, ROOT, Owner, Policy, empty_module

NATIVE = importlib.import_module(PACKAGE + ".cached_whole_operation_native")
REGISTRY = importlib.import_module(PACKAGE + ".operation_auto_registry")
RECALL = importlib.import_module(PACKAGE + ".connected_recall_binding")
SPIKING = importlib.import_module(PACKAGE + ".spiking")
LIQUID = importlib.import_module(PACKAGE + ".liquid")
SAMPLING = importlib.import_module(PACKAGE + ".native_token_sampling")
PAGING = importlib.import_module(PACKAGE + ".working_attention_paging")
SPARSE = importlib.import_module(PACKAGE + ".sparse_router_state")
ROUTER_PAGING = importlib.import_module(PACKAGE + ".router_state_paging")


class ActivityPager:
    def __init__(self): self._metrics, self.cancelled = {"processedPrefillTokens": 0}, None
    def check_cancelled(self):
        if self.cancelled and self.cancelled(): raise InterruptedError("tiny proof interrupted")
    @contextmanager
    def activate(self, cancelled=None):
        previous, self.cancelled = self.cancelled, cancelled
        try: self.check_cancelled(); yield
        finally: self.cancelled = previous


def config(width=2):
    return types.SimpleNamespace(max_seq_len=64, vocab_size=4, working_memory_slots=32,
                                 to_dict=lambda: {"width": width, "maxSeqLen": 64})


def leaf():
    def projection(kind=MODEL.PackedAdaptiveBitLinear):
        return empty_module(kind, buffers={"_packed_forward_weight": torch.ones((1, 1), dtype=torch.uint8)},
                            attrs={"in_features": 2, "out_features": 2})
    def scalar():
        value = empty_module(MODEL.PackedAdaptiveTernaryScalar, children={"levels": projection(MODEL.PackedAdaptiveTernaryEmbedding)},
                             buffers={"_index": torch.zeros(1, dtype=torch.long)}, attrs={"base": 0.125})
        value._non_persistent_buffers_set.add("_index")
        return value
    roots = {"embedding": projection(MODEL.PackedAdaptiveTernaryEmbedding),
        "global_workspace": empty_module(MODEL.GlobalWorkspace, children={"query": projection()}, attrs={"slots": 4}),
        "workspace_strength": scalar(), "memory_projection": projection(),
        "memory_strength": scalar(), "blocks": empty_module(torch.nn.ModuleList),
        "experts": empty_module(torch.nn.ModuleList), "expert_prototypes": empty_module(torch.nn.ModuleList),
        "final_norm": empty_module(MODEL.RMSNorm, buffers={"control": torch.ones(1)}), "language_head": projection()}
    return empty_module(MODEL.OmniDecoder, children=roots, attrs={"config": config(), "last_generation_neural_state": torch.ones((1, 2)), "training": False})


def fixture(*, connected=False):
    decoder = leaf()
    if connected:
        synapses = empty_module(SPIKING.STDPSynapses, buffers={"pre_trace": torch.zeros(2)}, attrs={
            "_sparse_state": types.SimpleNamespace(ready=True, _transaction_active=False), "_router_state_pager": object()})
        router = empty_module(SPIKING.AssociativeSpikingRouter, children={"synapses": synapses,
            "population": empty_module(SPIKING.LIFPopulation, buffers={"membrane": torch.zeros(2)})})
        region = empty_module(CONNECTED.NativeRegion, children={"decoder": decoder, "router": router,
            "memory_bridge": decoder.memory_projection, "idea_adapter": empty_module(torch.nn.Sequential),
            "recall_projection": decoder.language_head, "liquid": empty_module(LIQUID.LiquidController)}, attrs={
            "identity": CONNECTED.RegionIdentity("real-region", "parent", "a" * 64), "config": config(), "_recall": None})
        decoder = empty_module(CONNECTED.ConnectedOmniDecoder, children={
            "workspace_exchange": empty_module(CONNECTED.RegionExchange, attrs={"widths": (2,)}),
            "cortical_exchange": empty_module(CONNECTED.RegionExchange, attrs={"widths": (2,)})}, attrs={
            "regions": [region], "config": config(), "last_generation_neural_state": torch.ones((1, 2)),
            "last_region_activity": (torch.ones((1, 2)),), "training": False})
    pager = types.SimpleNamespace(_owners={}, _lock=threading.RLock(), _regional_limits=lambda: None)
    brain = Owner(decoder=decoder, config=decoder.config, core_pager=pager, device=torch.device("cpu"),
                  resource_policy=Policy(), working_attention_pager=ActivityPager())
    for root in (decoder.regions[0].decoder,) if connected else (decoder,):
        for module in root.modules():
            if type(module) in {MODEL.PackedAdaptiveBitLinear, MODEL.PackedAdaptiveTernaryEmbedding}:
                module._native_core_pager = pager
                pager._owners[module] = types.SimpleNamespace(loaded=True, device=torch.device("cpu"))
    ops = object.__new__(NATIVE.NativeCachedWholeOperationOps)
    ops._owner, ops.host, ops._template, ops._activation_transfer_bytes = weakref.ref(brain), types.SimpleNamespace(_stop_event=threading.Event()), None, 0
    return brain, ops


class CompleteNativeBoundaries(unittest.TestCase):
    def test_exact_closure_traverses_connected_owners_and_rejects_opaque_callbacks_overrides_and_unregistered_weights(self):
        brain, ops = fixture(connected=True)
        closed = ops._closure()
        self.assertEqual(closed["kind"], REGISTRY.CACHED_KINDS[1]); self.assertEqual(len(closed["regions"]), 1)
        self.assertTrue(all(module in brain.core_pager._owners for module in closed["packed"]))
        region = brain.decoder.regions[0]
        region._recall = lambda cue: cue
        with self.assertRaisesRegex(ValueError, "opaque connected recall"): ops._closure()
        region._recall = None
        region.decoder.__dict__["_generation_step_bounded"] = lambda *_args: None
        with self.assertRaisesRegex(ValueError, "instance decoder"): ops._closure()
        del region.decoder.__dict__["_generation_step_bounded"]
        missing = region.decoder.memory_projection; brain.core_pager._owners.pop(missing)
        with self.assertRaisesRegex(ValueError, "unregistered packed"): ops._closure()

    def test_connected_cycle_and_changed_exchange_or_parent_configuration_do_not_reuse_the_same_registry(self):
        brain, ops = fixture(connected=True)
        before = ops._registry()[0]
        brain.decoder.workspace_exchange.widths = (3,)
        self.assertNotEqual(before, ops._registry()[0])
        brain.decoder.workspace_exchange.widths = (2,)
        brain.decoder.regions[0].decoder.config = config(3)
        self.assertNotEqual(before, ops._registry()[0])
        brain.decoder.regions[0].decoder = brain.decoder
        with self.assertRaisesRegex(ValueError, "cyclic connected"): ops._closure()

    def test_native_recursive_fork_and_exact_hash_include_nested_temporal_running_and_neural_state(self):
        brain, ops = fixture()
        child = MODEL._DecoderInferenceCache(3, "child", [], [], torch.ones((1, 2)), None, torch.ones((1, 2)))
        source = CONNECTED._ConnectedInferenceCache(3, "root", [[]], [[]],
            [(torch.ones((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)))], [torch.ones((1, 2))], [child])
        with ops.proof_scope():
            before = ops.cache_sha(source)
            fork = ops._fork(source, "cpu")
            self.assertEqual(before, ops.cache_sha(fork.cache))
            fork.cache.temporal[0][1].add_(1)
            self.assertNotEqual(before, ops.cache_sha(fork.cache)); self.assertEqual(before, ops.cache_sha(source))
            fork.cache.nested[0].hidden_sum.add_(2)
            self.assertTrue(torch.equal(child.hidden_sum, torch.ones((1, 2))))
            fork.close()
        self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_bounded_proof_observes_stop_but_original_restore_scope_can_still_verify(self):
        brain, ops = fixture(); stopped = threading.Event(); stopped.set()
        value = torch.arange(4, dtype=torch.float32)
        with self.assertRaisesRegex(InterruptedError, "proof interrupted"), ops.proof_scope(stopped): ops._hash_tensor(value)
        with ops.proof_scope(): self.assertEqual(len(ops._hash_tensor(value)), 64)

    def test_placement_restores_control_activity_compute_presence_and_refuses_replaced_owner(self):
        brain, ops = fixture(); closed = ops._closure(); packed = closed["packed"][0]
        # This is only a home-device metadata label. No MPS tensor, kernel,
        # admission probe or real accelerator operation is performed.
        brain.device = torch.device("mps")
        packed._native_compute_device = None
        before = brain.decoder.final_norm._buffers["control"]
        old_activity = brain.decoder.last_generation_neural_state
        calls = []; brain.core_pager._observe_mutations = lambda *_args: calls.append("observe")
        ops.admit_candidate = lambda candidate: None
        operation = {"closure": closed, "ownerObjects": dict(brain.core_pager._owners),
                     "residency": [(module, owner.device) for module, owner in brain.core_pager._owners.items()]}
        with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()):
            self.assertEqual(packed._native_compute_device, torch.device("cpu"))
            brain.decoder.final_norm._buffers["control"] = torch.zeros(1)
            brain.decoder.last_generation_neural_state = torch.zeros((1, 2))
        self.assertIsNone(packed._native_compute_device)
        self.assertIs(brain.decoder.final_norm._buffers["control"], before)
        self.assertIs(brain.decoder.last_generation_neural_state, old_activity)
        self.assertEqual(len(calls), len(closed["packed"]))
        with self.assertRaisesRegex(NATIVE.OperationAutoRestoreRequired, "explicit Unload"):
            with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()):
                brain.core_pager._owners[packed] = types.SimpleNamespace(loaded=True, device=torch.device("cpu"))
        self.assertTrue(brain._native_execution_restore_required)

    def test_inert_complete_step_returns_same_home_sample_and_retained_state_without_advancing_real_rng(self):
        brain, ops = fixture(); decoder = brain.decoder
        cache = MODEL._DecoderInferenceCache(2, "same", [], [], torch.ones((1, 2)), None, torch.ones((1, 2)))
        window, rng = torch.tensor([[1, 2, 3]]), torch.Generator().manual_seed(7)
        initial = rng.get_state().clone(); logits = torch.tensor([[0.2, 0.5, 0.7, -0.3]])
        sampling = {"temperature": 0.65, "top_k": 4, "noise": 0.0, "text_boundaries": None, "remaining_tokens": 8}
        decoder._forward_window_bytes = lambda *_args: 64
        decoder._generation_signature = lambda cue: "same"
        calls = []
        def inert_step(_window, _cue, private):
            calls.append(private); private.length += 1
            private.hidden_sum = private.hidden_sum + 1
            private.neural_state = torch.tensor([[3.0, 4.0]])
            decoder.last_generation_neural_state = private.neural_state
            brain.working_attention_pager._metrics["processedPrefillTokens"] += 1
            return logits, private
        decoder._generation_step = inert_step
        @contextmanager
        def inert_placement(*_args):
            previous = decoder.last_generation_neural_state
            try: yield
            finally: decoder.last_generation_neural_state = previous
        ops._placement = inert_placement
        ops.admit_candidate, ops.synchronize = lambda *_args: None, lambda *_args: None
        operation = {"cache": cache, "window": window, "cue": None, "rng": initial, "generator": rng,
                     "sampling": sampling, "closure": {"decoders": [("decoder", decoder)]}}
        stages = []; observer = types.SimpleNamespace(stage_completed=stages.append)
        with ops.pure_scope(threading.Event()):
            trial = ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation, observer, threading.Event())
        expected_rng = torch.Generator(); expected_rng.set_state(initial)
        expected = SAMPLING.sample_native_token(logits, expected_rng, **sampling)
        self.assertTrue(torch.equal(rng.get_state(), initial)); self.assertEqual(cache.length, 2)
        self.assertIsNot(calls[0], cache); self.assertEqual(trial.cache.length, 3)
        self.assertTrue(torch.equal(trial.sample[0], expected[0])); self.assertEqual(trial.sample[1:], expected[1:])
        self.assertTrue(torch.equal(trial.rng_state, expected_rng.get_state())); self.assertEqual(stages, REGISTRY.CACHED_STAGES)
        ops.check_output(trial, ops.output_descriptor(trial)); trial.close()
        self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_declared_recall_binding_uses_exact_configured_slots_and_never_marks_read_as_learning(self):
        calls = []
        source = Owner(recall_vector=lambda cue, **kwargs: calls.append((cue, kwargs)) or ("neural-vector", []))
        binding = RECALL.ConnectedRecallBinding(source, 37)
        self.assertEqual(binding("live-cue"), "neural-vector")
        self.assertEqual(calls, [("live-cue", {"workspace_slots": 37, "record_activity": False})])
        with self.assertRaisesRegex(ValueError, "declared native source"): RECALL.recall_source_snapshot(binding, reserve=lambda *_args: None)

    def test_native_layout_uses_real_pager_and_attention_geometry_and_the_actual_parent_capacity(self):
        brain, ops = fixture(connected=True)
        root, child = brain.decoder, brain.decoder.regions[0].decoder
        child.config.max_seq_len = 7
        child.blocks = empty_module(torch.nn.ModuleList, children={"0": empty_module(MODEL.DecoderBlock, children={
            "attention": empty_module(MODEL.CausalSelfAttention, attrs={"key_chunk_tokens": 64, "n_heads": 2, "head_dim": 4})})})
        pager = brain.working_attention_pager
        pager.page_tokens, pager.device_tile_budget_bytes = 16, 65536
        def sequence():
            value = object.__new__(PAGING.PagedTensorSequence)
            value.pager, value.axis, value.dtype = pager, 2, torch.float32
            value._shape, value.ends, value._page_tokens = [1, 2, 4, 4], [4], 16
            return value
        layer = MODEL._AttentionInferenceCache(sequence(), sequence())
        cache = CONNECTED._ConnectedInferenceCache(4, "current", [[]], [[layer]],
            [(torch.ones((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)))], [torch.ones((1, 2))], [None])
        result = ops._layout(torch.tensor([[1, 2, 3, 4, 5]]), cache)
        self.assertEqual((result["minimumTokens"], result["maximumTokens"]), (2, 7))
        layer.key._page_tokens = 8
        with self.assertRaisesRegex(ValueError, "allocation policy changed"): ops._layout(torch.tensor([[1, 2, 3, 4, 5]]), cache)

    def test_private_recurrent_guards_allow_existing_lookup_but_refuse_birth_transaction_and_journal_before_any_write(self):
        state = object.__new__(SPARSE.SparseRouterState)
        existing = object(); state.owner = Owner(blocks={SPARSE.block_key(0, 0, 0): existing})
        token = REGISTRY.PRIVATE_OPERATION.set(True)
        try:
            self.assertIs(state.ensure(0, 0), existing)
            with self.assertRaisesRegex(RuntimeError, "cannot grow"): state.ensure(0, 1)
            with self.assertRaisesRegex(RuntimeError, "cannot begin"), state.transaction(): self.fail("private transaction entered")
            with self.assertRaisesRegex(RuntimeError, "cannot journal"):
                ROUTER_PAGING.RouterMutationJournal(object(), None, ram_bytes=0)
        finally: REGISTRY.PRIVATE_OPERATION.reset(token)
        self.assertEqual(len(state.owner.blocks), 1)

    def test_inert_connected_step_returns_complete_temporal_activity_without_modifying_original_connected_state(self):
        brain, ops = fixture(connected=True); decoder = brain.decoder
        state = (torch.ones((1, 2)), torch.zeros((1, 2)), torch.ones((1, 2)))
        cache = CONNECTED._ConnectedInferenceCache(2, "same", [[]], [[]], [state], [torch.ones((1, 2))], [None])
        window, rng = torch.tensor([[1, 2, 3]]), torch.Generator().manual_seed(9)
        initial = rng.get_state().clone(); logits = torch.tensor([[0.2, 0.5, 0.7, -0.3]])
        sampling = {"temperature": 0.65, "top_k": 4, "noise": 0.0, "text_boundaries": None, "remaining_tokens": 8}
        decoder._forward_window_bytes, decoder._generation_signature = lambda *_args: 64, lambda cue: "same"
        def inert_step(_window, _cue, private):
            private.length += 1
            private.temporal = [tuple(item + 1 for item in value) for value in private.temporal]
            private.hidden_sum = [value + 1 for value in private.hidden_sum]
            decoder.last_generation_neural_state = torch.tensor([[3.0, 4.0]])
            decoder.last_region_activity = (torch.tensor([[5.0, 6.0]]),)
            brain.working_attention_pager._metrics["processedPrefillTokens"] += 1
            return logits, private
        decoder._generation_step = inert_step
        @contextmanager
        def inert_placement(*_args):
            before = decoder.last_generation_neural_state, decoder.last_region_activity
            try: yield
            finally: decoder.last_generation_neural_state, decoder.last_region_activity = before
        ops._placement, ops.admit_candidate, ops.synchronize = inert_placement, lambda *_args: None, lambda *_args: None
        operation = {"cache": cache, "window": window, "cue": None, "rng": initial, "generator": rng,
                     "sampling": sampling, "closure": {"decoders": [("decoder", decoder)]}}
        with ops.proof_scope(): before = ops.cache_sha(cache)
        with ops.pure_scope(threading.Event()):
            trial = ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation, None, threading.Event())
            self.assertEqual(before, ops.cache_sha(cache)); self.assertNotEqual(before, ops.cache_sha(trial.cache))
        self.assertTrue(torch.equal(rng.get_state(), initial)); self.assertEqual(cache.length, 2)
        self.assertEqual(trial.cache.length, 3); self.assertEqual(len(trial.activity), 2)
        self.assertTrue(torch.equal(trial.neural_state, torch.tensor([[3.0, 4.0]])))
        self.assertTrue(torch.equal(trial.activity[1][2][0], torch.tensor([[5.0, 6.0]])))
        ops.check_output(trial, ops.output_descriptor(trial)); trial.close()
        self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_private_recall_forwards_exact_cancellation_and_releases_only_its_own_scratch_on_stop(self):
        stopped, calls, moved = threading.Event(), [], []
        state = Owner(closed=False)
        state.close = lambda: setattr(state, "closed", True)
        original_source, view = Owner(), Owner(_last_recall_paged_state=None)
        def recall(cue, **kwargs):
            calls.append(kwargs); view._last_recall_paged_state = state
            stopped.set()
            return cue, []
        view.recall_vector = recall
        original = RECALL.ConnectedRecallBinding(original_source, 37)
        binding = RECALL.PrivateConnectedRecallBinding(view, original,
            lambda value, target: moved.append(str(target)) or value, stopped.is_set)
        with self.assertRaisesRegex(InterruptedError, "before home delivery"): binding(torch.ones(2))
        self.assertEqual(moved, ["cpu"]); self.assertTrue(state.closed)
        self.assertIsNone(view._last_recall_paged_state)
        self.assertEqual(calls[0]["workspace_slots"], 37); self.assertFalse(calls[0]["record_activity"])
        self.assertTrue(calls[0]["cancelled"]())
        with self.assertRaisesRegex(InterruptedError, "before its source read"): binding(torch.ones(2))
        self.assertEqual(len(calls), 1)

    def test_actual_paged_recall_reserve_checks_cancellation_before_any_disk_or_memory_side_effect(self):
        tree = ast.parse((ROOT / "omni_core/vsa.py").read_text())
        substrate = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NeuralSubstrate")
        method = next(node for node in substrate.body if isinstance(node, ast.FunctionDef) and node.name == "_recall_vector_paged")
        reserve = next(copy.deepcopy(node) for node in method.body if isinstance(node, ast.FunctionDef) and node.name == "reserve")
        stopped, calls = threading.Event(), []
        index = Owner(_reserve_disk=lambda *args: calls.append(("disk", args)), _reserve_memory=lambda *args: calls.append(("ram", args)), _disk_reserve=object())
        source = Owner(assemblies=Owner(index=index), _check_growth=lambda *_args: self.fail("existing reserve policy was changed"))
        namespace = {"cancelled": stopped.is_set, "source_reference": lambda: source}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[reserve], type_ignores=[])), "actual-private-recall-reserve", "exec"), namespace)
        stopped.set()
        with self.assertRaisesRegex(InterruptedError, "resource boundary"): namespace["reserve"](32, "tiny-frontier")
        self.assertEqual(calls, [])
        stopped.clear(); self.assertTrue(namespace["reserve"](32, "tiny-frontier"))
        self.assertEqual(calls, [("disk", (32, "tiny-frontier")), ("ram", (320 * 1024, "tiny-frontier"))])

    def test_home_baseline_keeps_original_control_devices_and_compute_references_instead_of_rewriting_the_reference(self):
        brain, ops = fixture(); closed = ops._closure()
        ops.admit_candidate = lambda *_args: None
        ops.move_activity = lambda *_args: self.fail("current-home baseline remapped its original controls")
        brain.core_pager._observe_mutations = lambda *_args: None
        operation = {"closure": closed, "ownerObjects": dict(brain.core_pager._owners),
            "residency": [(module, owner.device) for module, owner in brain.core_pager._owners.items()]}
        with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()):
            self.assertTrue(all(not hasattr(module, "_native_compute_device") for module in closed["packed"]))
        self.assertTrue(all(not hasattr(module, "_native_compute_device") for module in closed["packed"]))

    def test_private_cortical_viewport_rows_restore_after_trials_and_only_the_exact_accepted_scope_is_published(self):
        brain, ops = fixture(); closed = ops._closure(); pager = brain.core_pager
        old = {"selected": {"values": [1.0]}}
        pager._observations, pager._observation_requests, pager._observation_scopes = old, {"selected": (0, 1)}, {"selected": "capture"}
        pager._observe_mutations = lambda *_args: None
        ops.admit_candidate = lambda *_args: None
        operation = {"closure": closed, "ownerObjects": dict(pager._owners), "originalActivity": ops._activities(closed),
            "originalCorticalObservations": old, "residency": [(module, owner.device) for module, owner in pager._owners.items()]}
        observations = {}
        with ops._placement({"executionDevice": "cpu"}, operation, threading.Event(), observations):
            pager._observations["selected"] = {"values": [2.0]}
            self.assertEqual(old, {"selected": {"values": [1.0]}})
        self.assertIs(pager._observations, old)
        trial = Owner(activity=operation["originalActivity"], observations=observations)
        ops.publish_activity(trial)
        self.assertEqual(pager._observations, {"selected": {"values": [2.0]}})
        self.assertEqual(old, {"selected": {"values": [1.0]}})
        ops.restore_published_activity(operation); self.assertIs(pager._observations, old)
        pager._observation_scopes["selected"] = "new-capture"
        ops.publish_activity(trial); self.assertIs(pager._observations, old)

    def test_private_attention_membership_copy_is_admitted_and_immutable_without_changing_source_sets(self):
        policy = Policy()
        source = Owner(growth_guard=None, attention_active_neuron_ids={"neuron"},
            attention_recalled_assembly_ids={"assembly"}, attention_eligible_synapse_ids={"edge"})
        binding = RECALL.ConnectedRecallBinding(source, 37)
        view, private, lease = binding.private(cancelled=lambda: False, transfer=lambda value, target: value,
            reserve_metadata=lambda size: policy.reserve_ram(size, "private memberships"))
        source.attention_active_neuron_ids.add("later")
        self.assertEqual(view.attention_active_neuron_ids, frozenset({"neuron"}))
        self.assertIs(private.authoritative_source, source); self.assertEqual(private.workspace_slots, 37)
        self.assertEqual(lease.size, 4096 + 3 * 64); self.assertFalse(lease.closed)
        lease.release(); self.assertTrue(lease.closed)
        with self.assertRaisesRegex(RuntimeError, "metadata refused"):
            binding.private(cancelled=lambda: False, transfer=lambda value, target: value,
                reserve_metadata=lambda size: (_ for _ in ()).throw(RuntimeError("metadata refused")))

    def test_foreign_packed_projection_pager_is_not_mistaken_for_the_complete_current_owner(self):
        brain, ops = fixture()
        self.assertTrue(ops._closure()["packed"])
        brain.decoder.memory_projection._native_core_pager = object()
        with self.assertRaisesRegex(ValueError, "foreign learned owner"): ops._closure()

    def test_connected_private_callback_and_graph_observer_references_restore_and_release_admitted_memberships(self):
        brain, ops = fixture(connected=True); closed = ops._closure(); region = brain.decoder.regions[0]
        class Memory(Owner):
            def release_paged_recall_scratch(self): self.release_count += 1
        before = object()
        source = Memory(growth_guard=None, release_count=0, attention_active_neuron_ids={"n"},
            attention_recalled_assembly_ids={"a"}, attention_eligible_synapse_ids={"s"},
            neurons=Owner(_recall_graph_observer=before), synapses=Owner())
        original = RECALL.ConnectedRecallBinding(source, 37); region.attach_recall(original)
        closed["bindings"] = [("decoder.region0", region, original)]
        brain.core_pager._observe_mutations = lambda *_args: None
        ops.admit_candidate = lambda *_args: None
        operation = {"closure": closed, "ownerObjects": dict(brain.core_pager._owners),
            "residency": [(module, owner.device) for module, owner in brain.core_pager._owners.items()]}
        with ops._placement({"executionDevice": "cpu"}, operation, threading.Event()):
            private = region._recall; view = private.source
            self.assertIsNot(view, source); self.assertEqual(private.workspace_slots, 37)
            self.assertIs(private.authoritative_source, source)
            source.neurons._recall_graph_observer = object()
            source.synapses._recall_graph_observer = object()
        self.assertIs(region._recall, original); self.assertEqual(view.release_count, 1)
        self.assertEqual(source.release_count, 0); self.assertIs(source.neurons._recall_graph_observer, before)
        self.assertFalse(hasattr(source.synapses, "_recall_graph_observer"))
        self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))


if __name__ == "__main__": unittest.main()
