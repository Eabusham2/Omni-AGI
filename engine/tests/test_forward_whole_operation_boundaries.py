"""New raw CPU, metadata and inert-callback full-forward ownership cases.

No decoder/model/brain constructor, neural forward/backward, Load, training,
codec, hardware route or speed measurement is executed.
"""
from __future__ import annotations

import importlib
import threading
import types
import unittest
import weakref
from contextlib import contextmanager, nullcontext

import torch

from test_operation_auto_boundaries import PACKAGE, MODEL, CONNECTED, Owner, empty_module
from test_cached_whole_operation_boundaries import fixture

NATIVE = importlib.import_module(PACKAGE + ".forward_whole_operation_native")
AUTO = importlib.import_module(PACKAGE + ".forward_whole_operation_auto")
REGISTRY = importlib.import_module(PACKAGE + ".operation_auto_registry")
WHOLE = importlib.import_module(PACKAGE + ".whole_path_execution")
GUARD = importlib.import_module(PACKAGE + ".decoder_execution_native")._tensor_guard


def full_fixture(*, connected=False):
    brain, _cached = fixture(connected=connected)
    root = brain.decoder
    leaves = [region.decoder for region in root.regions] if connected else [root]
    for leaf in leaves:
        leaf.config.d_model, leaf.config.d_ff, leaf.config.idea_dim = 2, 4, 2
        leaf.global_workspace.query_chunk_slots = 4
        def projection(width):
            value = empty_module(MODEL.PackedAdaptiveBitLinear,
                buffers={"_packed_forward_weight": torch.ones((1, 1), dtype=torch.uint8)},
                attrs={"in_features": 2, "out_features": width, "_native_core_pager": brain.core_pager})
            brain.core_pager._owners[value] = Owner(loaded=True, device=torch.device("cpu"))
            return value
        leaf.action_policy = empty_module(MODEL.ActionPolicyHead, attrs={"hidden_width": 8}, children={
            "norm": empty_module(MODEL.RMSNorm, buffers={"control": torch.ones(1)}),
            "hidden": projection(8), "projection": projection(len(MODEL.ACTION_KINDS))})
    if connected:
        root.config.d_model, root.config.d_ff, root.config.idea_dim = 2, 4, 2
        root.widths = (2,)
        region = root.regions[0]
        region.config.d_model, region.config.d_ff, region.config.idea_dim = 2, 4, 2
        region._buffers["liquid_state"] = torch.zeros((1, 2))
        fields = {"embedding": CONNECTED._JoinedEmbedding, "final_norm": CONNECTED._JoinedNorm,
            "language_head": CONNECTED._JoinedHead, "action_policy": CONNECTED._JoinedHead,
            "global_workspace": CONNECTED._JoinedWorkspace, "memory_projection": CONNECTED._JoinedMemoryProjection,
            "workspace_strength": CONNECTED._BorrowedGains, "memory_strength": CONNECTED._BorrowedGains,
            "blocks": CONNECTED._BorrowedSequence, "experts": CONNECTED._BorrowedSequence,
            "expert_prototypes": CONNECTED._BorrowedSequence}
        for field, kind in fields.items():
            if kind is CONNECTED._BorrowedSequence:
                attrs = {"values": tuple(value for reg in root.regions for value in getattr(reg.decoder, field))}
            else:
                attrs = {"regions": tuple(root.regions), "widths": root.widths}
                if kind in {CONNECTED._JoinedHead, CONNECTED._BorrowedGains}: attrs["field"] = field
                if kind is CONNECTED._JoinedHead:
                    attrs["_borrowed_learning_heads"] = tuple(getattr(reg.decoder, field) for reg in root.regions)
                if kind is CONNECTED._JoinedWorkspace:
                    attrs.update(exchange=root.workspace_exchange, slots=4, query_chunk_slots=4)
            setattr(root, field, empty_module(kind, attrs=attrs))
        # fixture regions are borrowed Python objects rather than registered
        # constructors; set their exact metadata flags without calling train().
        for module in region.modules(): module.training = False
    for leaf in leaves:
        for module in leaf.modules(): module.training = False
    for module in root.modules(): module.training = False
    brain.brain_id = "raw-full-forward-owner"
    brain.core_pager.status = lambda: {"pinnedOwners": 0}
    pager = brain.working_attention_pager
    pager.page_tokens, pager.device_tile_budget_bytes = 11, 1048576
    pager.resident_budget_bytes, pager.scratch_budget_bytes = 1048576, 2097152
    ops = object.__new__(NATIVE.NativeFullForwardOperationOps)
    ops._owner, ops._template = weakref.ref(brain), None
    ops.host, ops._activation_transfer_bytes = Owner(_stop_event=threading.Event()), 0
    return brain, ops


class FullForwardNativeBoundaries(unittest.TestCase):
    def test_declared_action_owners_wrappers_and_region_liquid_state_are_in_exact_closure(self):
        for connected in (False, True):
            with self.subTest(connected=connected):
                brain, ops = full_fixture(connected=connected)
                closure = ops._closure()
                leaf = brain.decoder.regions[0].decoder if connected else brain.decoder
                self.assertIn(leaf.action_policy.hidden, closure["packed"])
                self.assertIn(leaf.action_policy.projection, closure["packed"])
                self.assertIn(id(leaf.action_policy.norm), closure["modules"])
                self.assertEqual(closure["kind"], REGISTRY.FORWARD_KINDS[int(connected)])
                if connected:
                    region = brain.decoder.regions[0]
                    self.assertIn(id(region), closure["modules"])
                    self.assertIn(id(brain.decoder.action_policy), closure["modules"])
                    old = ops._registry()[0]
                    brain.decoder.action_policy._borrowed_learning_heads = (leaf.language_head,)
                    with self.assertRaisesRegex(ValueError, "readout owners"): ops._registry()
                    brain.decoder.action_policy._borrowed_learning_heads = (leaf.action_policy,)
                    self.assertEqual(ops._registry()[0], old)

    def test_missing_action_owners_hooks_and_foreign_regional_initial_state_remain_ordinary(self):
        brain, ops = full_fixture()
        brain.core_pager._owners.pop(brain.decoder.action_policy.hidden)
        with self.assertRaisesRegex(ValueError, "unowned packed"): ops._closure()
        brain, ops = full_fixture()
        brain.decoder.action_policy._forward_hooks[1] = object()
        with self.assertRaisesRegex(ValueError, "action/control"): ops._closure()
        brain, ops = full_fixture(connected=True)
        brain.decoder.regions[0]._buffers["liquid_state"] = torch.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "liquid initial"): ops._closure()
        brain, ops = full_fixture()
        brain.decoder._forward_bounded = lambda *_args: None
        with self.assertRaisesRegex(ValueError, "override"): ops._closure()
        brain, ops = full_fixture()
        brain.decoder.global_workspace._attend = lambda *_args: None
        with self.assertRaisesRegex(ValueError, "helper override"): ops._closure()

    def test_eval_no_grad_mask_cue_and_manual_owner_admission_preserve_ordinary_path(self):
        brain, ops = full_fixture()
        window, cue, mask = torch.tensor([[1, 2, 3]], dtype=torch.long), torch.ones((1, 2)), torch.tensor([[True, True, False]])
        with torch.no_grad():
            self.assertTrue(ops.valid_forward(window, cue, True, mask))
            self.assertFalse(ops.valid_forward(window, cue[:, :1], True, mask))
            self.assertFalse(ops.valid_forward(window, cue, True, mask[:, :1]))
            self.assertFalse(ops.valid_forward(window, cue, 1, mask))
            brain._cached_decoder_execution = Owner(status=lambda: {"phase": "active"})
            self.assertFalse(ops.valid_forward(window, cue, None, mask))
            brain._cached_decoder_execution = None
            brain.core_pager.status = lambda: {"pinnedOwners": 1}
            self.assertFalse(ops.valid_forward(window, cue, None, mask))
        self.assertFalse(ops.valid_forward(window, cue, None, mask))
        brain.core_pager.status = lambda: {"pinnedOwners": 0}
        brain.decoder.training = True
        with torch.no_grad(): self.assertFalse(ops.valid_forward(window, cue, None, mask))

    def test_actual_geometry_and_state_include_mask_whole_workspace_and_liquid_initial_value(self):
        brain, ops = full_fixture(connected=True)
        window, mask = torch.tensor([[1, 2, 3]]), torch.tensor([[True, True, False]])
        closure = ops._closure()
        plain = ops._geometry(window, None, None, None, closure)
        whole = ops._geometry(window, None, True, mask, closure)
        self.assertNotEqual(WHOLE.digest(plain), WHOLE.digest(whole))
        self.assertEqual(whole["input"][0], [1, 3]); self.assertEqual(whole["mask"][1], "torch.bool")
        scope = {"scope": Owner(snapshot=lambda: {"packed": 1}), "registry": "r", "routerTokens": None, "routerEpoch": 0}
        before = ops._state(closure, scope, "same-owner")
        brain.decoder.regions[0].liquid_state.add_(1)
        after = ops._state(closure, scope, "same-owner")
        self.assertNotEqual(before["bindingSha256"], after["bindingSha256"])
        self.assertNotIn("last_generation_neural_state", [name for _owner, name, _value in ops._activities(closure)])

    def test_returned_storage_keeps_real_output_and_connected_activity_aliases_without_generation_state(self):
        brain, ops = full_fixture(connected=True)
        storage = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        output = {"hidden": storage.view(1, 3, 2), "workspace": storage[-1:].view(1, 1, 2)}
        rows = [(brain.decoder, "last_region_activity", (output["hidden"][:, -1],))]
        lifetime = NATIVE.CacheFork()
        result, activity = ops._returned_home(output, rows, torch.device("cpu"), lifetime)
        self.assertEqual(len(lifetime.leases), 1)
        self.assertEqual(result["hidden"].untyped_storage().data_ptr(), activity[0][2][0].untyped_storage().data_ptr())
        self.assertEqual(activity[0][2][0].storage_offset(), output["hidden"][:, -1].storage_offset())
        self.assertTrue(torch.equal(result["workspace"], output["workspace"]))
        lifetime.close(); self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_actual_capture_binds_complete_input_cue_mask_geometry_and_releases_borrowed_storage(self):
        brain, ops = full_fixture()
        state = {"kind": "live-owner-epochs", "bindingSha256": "1" * 64,
            "ownerSessionId": "s", "complete": True, "ownerVersions": {"native-owner": 1}}
        ops._scope = lambda _closure: {"registry": "registry"}
        ops._state = lambda *_args: state
        backing = torch.tensor([[9, 1, 2, 3, 8]], dtype=torch.long)
        window, cue, mask = backing[:, 1:4], torch.tensor([[0.25, 0.5]]), torch.tensor([[True, True, False]])
        binding = {key: "2" * 64 for key in ("policySha256", "topologySha256", "budgetSha256")}
        operation = ops.capture_forward(window, cue, True, mask, "s",
            {"binding": binding, "candidates": [{"executionDevice": "cpu"}]}, stopped=threading.Event())
        self.assertIs(operation["window"], window); self.assertIs(operation["attention_mask"], mask)
        self.assertEqual(operation["identity"]["stageCoverage"], REGISTRY.FORWARD_STAGES)
        self.assertEqual(operation["identity"]["cache"]["heat"], "cold")
        self.assertEqual(len(operation["identity"]["cache"]["logicalStateSha256"]), 64)
        self.assertEqual(operation["planContract"]["forwardLayout"],
            {"shapeSha256": operation["identity"]["geometrySha256"]})
        self.assertTrue(ops.forward_unchanged(operation, "s"))
        mask.logical_not_(); self.assertFalse(ops.forward_unchanged(operation, "s"))
        self.assertTrue(any(size >= backing.untyped_storage().nbytes() for size, _label in brain.resource_policy.admissions))
        ops.release_forward(operation)
        self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))

    def test_original_rng_frontier_change_requires_unload_without_reset_or_hidden_retry(self):
        brain, ops = full_fixture()
        rng = ops._rng_frontiers(["cpu"])
        ops.operation_state = lambda _operation, _session: {"same": True}
        operation = {"rng": rng, "identity": {"nativeState": {"same": True}}}
        ops.require_original_frontiers(operation, "session")
        # Alter the expected snapshot, not the real process generator. The
        # test leaves the process RNG untouched even on its failure path.
        rng["cpu"] = rng["cpu"].clone(); rng["cpu"][0] ^= 1
        actual = torch.get_rng_state().clone()
        with self.assertRaises(NATIVE.OperationAutoRestoreRequired): ops.require_original_frontiers(operation, "session")
        self.assertTrue(brain._native_execution_restore_required)
        self.assertTrue(torch.equal(torch.get_rng_state(), actual))

    def test_unprovable_original_owner_frontier_also_requires_unload_without_normal_retry(self):
        brain, ops = full_fixture()
        def replaced(*_args): raise RuntimeError("scoped learned owner was replaced")
        ops.operation_state = replaced
        original = torch.get_rng_state().clone()
        with self.assertRaisesRegex(NATIVE.OperationAutoRestoreRequired, "cannot be proven"):
            ops.require_original_frontiers({"rng": {"cpu": original}, "identity": {"nativeState": {}}}, "s")
        self.assertTrue(brain._native_execution_restore_required)
        self.assertTrue(torch.equal(torch.get_rng_state(), original))

    def test_inert_actual_entry_returns_all_outputs_and_restores_only_real_connected_activity(self):
        for connected in (False, True):
            with self.subTest(connected=connected):
                brain, ops = full_fixture(connected=connected)
                closure, decoder = ops._closure(), brain.decoder
                old_generation = decoder.last_generation_neural_state
                old_activity = getattr(decoder, "last_region_activity", None)
                calls, stages = [], []
                def inert_forward(window, **kwargs):
                    self.assertTrue(REGISTRY.PRIVATE_OPERATION.get())
                    calls.append((window.clone(), kwargs))
                    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
                    if connected: decoder.last_region_activity = (hidden[:, -1],)
                    return {"hidden": hidden, "logits": torch.ones((1, 3, decoder.config.vocab_size)),
                        "action_logits": torch.ones((1, len(MODEL.ACTION_KINDS))),
                        "workspace": hidden[:, -1:] if connected else torch.ones((1, 4, 2))}
                decoder.forward = inert_forward
                @contextmanager
                def placement(*_args):
                    try: yield
                    finally:
                        if connected: decoder.last_region_activity = old_activity
                ops._placement, ops.admit_candidate, ops.synchronize = placement, lambda *_a: None, lambda *_a: None
                ops.operation_state = lambda *_a: {"same": True}
                window, cue = torch.tensor([[1, 2, 3]], dtype=torch.long), torch.tensor([[0.25, 0.5]])
                mask = torch.tensor([[True, True, False]])
                operation = {"window": window, "cue": cue, "attention_mask": mask, "use_global_workspace": True,
                    "session": "session", "rng": ops._rng_frontiers(["cpu"]), "identity": {"nativeState": {"same": True}},
                    "closure": closure}
                with ops.pure_scope(threading.Event()):
                    trial = ops.execute({"executionDevice": "cpu", "asyncDevices": []}, operation,
                        Owner(stage_completed=stages.append), threading.Event())
                self.assertEqual(stages, REGISTRY.FORWARD_STAGES)
                self.assertEqual(set(trial.output), {"hidden", "logits", "action_logits", "workspace"})
                self.assertIs(calls[0][1]["attention_mask"], mask); self.assertIs(calls[0][1]["memory_bias"], cue)
                self.assertIsNone(calls[0][1]["labels"])
                self.assertIs(decoder.last_generation_neural_state, old_generation)
                if connected:
                    self.assertIs(decoder.last_region_activity, old_activity)
                    self.assertEqual(trial.activity[0][1], "last_region_activity")
                    self.assertEqual(trial.output["hidden"].untyped_storage().data_ptr(), trial.activity[0][2][0].untyped_storage().data_ptr())
                else: self.assertEqual(trial.activity, [])
                trial.close(); self.assertTrue(all(lease.closed for lease in brain.resource_policy.leases))


if __name__ == "__main__": unittest.main()
