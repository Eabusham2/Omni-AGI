"""Constructor-free tiny router state/math/storage fixtures, no neural run."""
import ast
import gc
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from omni_core.offload import NeuralStateResourcePause
from omni_core.parameter_diagnostics import ParameterDeltaJournal
from omni_core.persistence import tensor_checksum
from omni_core.router_state_paging import MATRIX_FIELDS, RouterStatePager
from omni_core.sparse_router_state import SparseRouterState, block_key, parse_block_key
from omni_core.spiking import STDPSynapses
from omni_core.optimizers import PackedMutationSnapshot
from omni_core.ternary_packing import (collect_module_ternary_tensors, inspect_module_ternary_layout,
    export_module_ternary_shards, verify_ternary_shards)
if __package__:
    from .test_router_state_paging import owner as dense_owner, dense_step
    from . import test_router_state_paging as dense_fixture
else:
    from test_router_state_paging import owner as dense_owner, dense_step
    import test_router_state_paging as dense_fixture


def sparse_owner(pager=None, *, rows=3, columns=5, loading=False):
    # Deliberately never call STDPSynapses/Router/Brain constructors.
    value = STDPSynapses.__new__(STDPSynapses)
    torch.nn.Module.__init__(value)
    value.pre_neurons, value.post_neurons = columns, rows
    value.learning_rate, value.pre_decay, value.post_decay = .035, .8, .7
    value.a_plus, value.a_minus, value.metaplasticity_rate = 1., 1.05, .025
    value.weight_limit, value.ternary = 1., True
    value._router_state_pager, value._router_loading_checkpoint = pager, loading
    value._sparse_state = SparseRouterState(value, loading=loading)
    value.register_buffer("pre_trace", torch.arange(columns).float() / 9)
    value.register_buffer("post_trace", torch.arange(rows).float() / 7)
    value.register_buffer("plasticity_events", torch.tensor(11))
    value.register_buffer("decay_cycles", torch.tensor(2))
    return value


def virtual_field(value, name):
    return torch.cat(tuple(value._sparse_state.dense_field_chunks(name))).reshape(value.post_neurons, value.pre_neurons)


class SparseRouterStateFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sparse-router-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def pager(self, *, hot=0, policy=None):
        return RouterStatePager(self.root, hot_bytes=hot, tile_bytes=2048, journal_ram_bytes=0,
                                resource_policy=policy)

    def native_file(self, value, *, path="router.safetensors", prefix="router.synapses."):
        target = self.root / path
        atomic_save_tensors_bounded(target, {prefix + name: tensor for name, tensor in value.state_dict().items()},
                                   chunk_bytes=32)
        return BoundedTensorFile(target, chunk_bytes=32)

    def test_fresh_constructor_source_has_no_full_recurrent_fields(self):
        source = Path(__file__).parents[1] / "omni_core" / "spiking.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "STDPSynapses")
        constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        strings = {node.value for node in ast.walk(constructor) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
        self.assertNotIn("_packed_weights", strings)
        self.assertNotIn("stability", strings)
        self.assertNotIn("uses", strings)
        self.assertNotIn("eligibility_accumulator", strings)
        value = sparse_owner()
        self.assertEqual(list(value.blocks), [])
        self.assertEqual(value.logical_ternary_parameter_count, 0)
        self.assertEqual(value.logical_connectivity_count, 15)
        self.assertEqual(value._sparse_state.status()["allocatedStateBytes"], 0)

    def test_no_population_or_block_count_product_cap(self):
        value = STDPSynapses.__new__(STDPSynapses)
        torch.nn.Module.__init__(value)
        value.pre_neurons, value.post_neurons = 1_000_001, 2_000_001
        value._router_state_pager = None
        value._sparse_state = SparseRouterState(value)
        self.assertGreater(value.logical_connectivity_count, 1_000_000_000_000)
        self.assertEqual(value.logical_ternary_parameter_count, 0)
        with value._sparse_state.transaction():
            block = value._sparse_state.ensure(31_250, 15_625)
            self.assertEqual((block.rows, block.columns), (1, 1))
        self.assertEqual(value.logical_ternary_parameter_count, 1)

    def test_exact_dense_native_migration_retains_nonzero_controls_on_zero_edges(self):
        old = dense_owner(rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8)
        levels[0, 0], levels[66, 68] = 1, -1
        old.set_effective_weights(levels)
        old.stability[65, 0] = 1.25
        old.uses[3, 67] = 8.5
        old.eligibility_accumulator[66, 1] = -255
        reader = self.native_file(old)
        pager = self.pager()
        value = sparse_owner(pager, rows=67, columns=69, loading=True)
        with patch.object(reader, "tensor", side_effect=AssertionError("full-tensor read is forbidden")):
            value.load_recurrent_state_bounded(reader, "router.synapses.")
        self.assertEqual(set(value.blocks), {"r0_c0", "r0_c1", "r1_c0", "r1_c1"})
        for field in ("weights", "eligibility_accumulator", "stability", "uses"):
            self.assertTrue(torch.equal(virtual_field(value, field), levels if field == "weights" else getattr(old, field)), field)
        self.assertEqual(value.historical_checksum_with_controls(), old.checksum_with_controls())
        self.assertTrue(value._sparse_state.migration_proof["exactNativeFieldsPreserved"])
        self.assertEqual(value._sparse_state.migration_proof["sourceHistoricalChecksum"], old.checksum_with_controls())
        self.assertLessEqual(reader.peak_transfer_bytes, 32)
        self.assertGreater(pager.status()["mappedLogicalBytes"], 0)
        pager.close()

    def test_all_zero_dense_migrates_without_allocating_any_block(self):
        old = dense_owner(rows=67, columns=69)
        pager = self.pager()
        value = sparse_owner(pager, rows=67, columns=69, loading=True)
        value.load_recurrent_state_bounded(self.native_file(old), "router.synapses.")
        self.assertEqual(len(value.blocks), 0)
        self.assertEqual(pager.status()["registeredStateBytes"], 0)
        self.assertEqual(value.historical_decoded_weight_checksum(), tensor_checksum((torch.zeros(67, 69, dtype=torch.int8),)))
        pager.close()

    def test_sparse_checkpoint_roundtrip_restores_declared_blocks_and_exact_controls(self):
        value = sparse_owner(rows=5, columns=7)
        levels = torch.zeros(5, 7, dtype=torch.int8)
        levels[4, 6] = -1
        value.set_effective_weights(levels)
        value.blocks["r0_c0"].stability[3, 2] = 4.25
        value.blocks["r0_c0"].uses[2, 1] = 9.5
        value.blocks["r0_c0"].eligibility_accumulator[1, 4] = 243
        reader = self.native_file(value)
        pager = self.pager()
        loaded = sparse_owner(pager, rows=5, columns=7, loading=True)
        loaded.load_recurrent_state_bounded(reader, "router.synapses.")
        for name, tensor in value.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
        self.assertEqual(loaded._sparse_state.status()["checkpointReady"], True)
        self.assertEqual(loaded.checksum_with_controls(), value.checksum_with_controls())
        pager.close()

    def test_sparse_recurrence_count_mean_and_historical_checksums(self):
        value = sparse_owner(rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8)
        levels[0, 0], levels[66, 68] = 1, -1
        value.set_effective_weights(levels)
        value.blocks["r1_c1"].stability[2, 4] = 4.
        inputs = torch.arange(69).float() / 5.
        torch.testing.assert_close(value.recurrent(inputs), torch.mv(levels.float(), inputs), rtol=0, atol=0)
        self.assertEqual(value.active_synapse_count(), 2)
        self.assertEqual(value.stability_mean(), 4. / (67*69))
        self.assertEqual(value.historical_decoded_weight_checksum(), tensor_checksum((levels,)))
        self.assertEqual(value.historical_checksum_with_controls(), tensor_checksum((levels, virtual_field(value, "stability"),
            virtual_field(value, "uses"), value.plasticity_events)))

    def test_exact_causal_anti_causal_and_quiet_pair_controls(self):
        old = dense_owner(rows=3, columns=5)
        levels = torch.tensor([[1, 0, -1, 0, 1], [0, 1, 0, -1, 0], [-1, 0, 1, 0, -1]], dtype=torch.int8)
        old.set_effective_weights(levels)
        old.stability.copy_(torch.arange(15).reshape(3, 5).float() / 13.)
        old.uses.copy_(torch.arange(15).reshape(3, 5).float() / 11.)
        old.eligibility_accumulator.copy_(torch.arange(-7, 8).reshape(3, 5).short() * 13)
        value = sparse_owner(loading=True)
        value.load_recurrent_state_bounded(self.native_file(old), "router.synapses.")
        pre, post = torch.tensor([1., 0., .25, 0., .5]), torch.tensor([.5, 0., 1.])
        expected, delta = dense_step(old, pre, post)
        summary = value.step(pre, post)
        self.assertEqual(summary.active_pairs, int((delta != 0).sum()))
        for field in MATRIX_FIELDS:
            actual = value.blocks["r0_c0"]._buffers[field]
            self.assertTrue(torch.equal(actual, expected[field]), field)
        for field in ("pre_trace", "post_trace", "plasticity_events", "decay_cycles"):
            self.assertTrue(torch.equal(getattr(value, field), expected[field]), field)

    def test_silent_activity_retains_controls_without_allocating_cross_products(self):
        value = sparse_owner(rows=67, columns=69)
        value.reset_activity()
        self.assertEqual(value.step(torch.zeros(69), torch.zeros(67)).active_pairs, 0)
        self.assertEqual(len(value.blocks), 0)
        with value._sparse_state.transaction():
            block = value._sparse_state.ensure(1, 1)
            block.eligibility_accumulator[2, 3] = 193
        value.step(torch.zeros(69), torch.zeros(67))
        self.assertEqual(int(block.eligibility_accumulator[2, 3]), 193)

    def test_activity_allocates_only_required_blocks_including_zero_rate_controls(self):
        value = sparse_owner(rows=67, columns=69)
        value.reset_activity()
        value.learning_rate = 0.
        value.pre_trace[68] = 1.
        post = torch.zeros(67); post[66] = 1.
        summary = value.step(torch.zeros(69), post)
        self.assertEqual(summary.active_pairs, 1)
        self.assertEqual(list(value.blocks), ["r1_c1"])
        block = value.blocks["r1_c1"]
        self.assertEqual(float(block.uses[2, 4]), 1.)
        self.assertEqual(float(block.stability[2, 4]), float(torch.tensor(.025)))
        self.assertEqual(value.active_synapse_count(), 0)

    def test_exact_decay_uses_original_absolute_positions_not_block_local_hashes(self):
        old = dense_owner(rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8)
        levels[65:, 65:] = 1
        old.set_effective_weights(levels)
        old.stability[66, 68], old.uses[66, 68] = 1., 4.
        value = sparse_owner(rows=67, columns=69, loading=True)
        value.load_recurrent_state_bounded(self.native_file(old), "router.synapses.")
        old.decay_unused(.8)
        value.decay_unused(.8)
        self.assertTrue(torch.equal(virtual_field(value, "weights"), old.weights))
        self.assertTrue(torch.equal(virtual_field(value, "stability"), old.stability))
        self.assertEqual(int(value.decay_cycles), int(old.decay_cycles))

    def test_existing_and_new_blocks_both_roll_back_on_late_admission_failure(self):
        pager = self.pager()
        value = sparse_owner(pager, rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8); levels[0, 0] = 1
        value.set_effective_weights(levels)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        ensure = value._sparse_state.ensure
        def fail_late(row, column, **kwargs):
            if row == 1 and column == 0:
                raise NeuralStateResourcePause("fixture new block exceeded shared pool", {})
            return ensure(row, column, **kwargs)
        with patch.object(value._sparse_state, "ensure", side_effect=fail_late):
            with self.assertRaises(NeuralStateResourcePause):
                value.step(torch.ones(69), torch.ones(67))
        self.assertEqual(set(value.state_dict()), set(before))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(value.state_dict()[name], tensor), name)
        self.assertEqual(len(value.blocks), 1)
        self.assertEqual(list(pager.directory.glob("*.rollback")), [])
        pager.close()

    def test_cancel_at_exact_tile_boundary_rolls_back_control_and_topology(self):
        pager = self.pager()
        value = sparse_owner(pager, rows=5, columns=7)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        calls = 0
        def cancelled():
            nonlocal calls
            calls += 1
            return calls == 12
        pager.cancelled = cancelled
        with self.assertRaises(InterruptedError):
            value.step(torch.ones(7), torch.ones(5))
        pager.cancelled = None
        self.assertEqual(set(value.state_dict()), set(before))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(value.state_dict()[name], tensor), name)
        self.assertEqual(pager.status()["registeredStateBytes"], 0)
        pager.close()

    def test_late_invalid_dense_control_does_not_drop_earlier_valid_edges(self):
        old = dense_owner(rows=67, columns=69)
        old._packed_weights[0, 0] = 0x56
        old.uses[-1, -1] = float("nan")
        pager = self.pager()
        value = sparse_owner(pager, rows=67, columns=69, loading=True)
        with self.assertRaises(ValueError):
            value.load_recurrent_state_bounded(self.native_file(old), "router.synapses.")
        self.assertFalse(value._sparse_state.ready)
        self.assertEqual(len(value.blocks), 0)
        self.assertEqual(pager.status()["registeredStateBytes"], 0)
        with self.assertRaises(RuntimeError):
            value.active_synapse_count()
        pager.close()

    def test_sparse_schema_rejects_missing_field_reserved_code_and_floating_archive(self):
        value = sparse_owner(rows=3, columns=5)
        levels = torch.zeros(3, 5, dtype=torch.int8); levels[0, 0] = 1
        value.set_effective_weights(levels)
        for corruption in ("missing", "code", "floating"):
            state = {"router.synapses." + name: tensor.clone() for name, tensor in value.state_dict().items()}
            if corruption == "missing":
                state.pop("router.synapses.blocks.r0_c0.uses")
            elif corruption == "code":
                state["router.synapses.blocks.r0_c0._packed_weights"][0, 0] = 0x57
            else:
                state["router.synapses.weights"] = torch.zeros(3, 5)
            path = self.root / (corruption + ".safetensors")
            atomic_save_tensors_bounded(path, state)
            destination = sparse_owner(rows=3, columns=5, loading=True)
            with self.assertRaises(ValueError):
                destination.load_recurrent_state_bounded(BoundedTensorFile(path), "router.synapses.")
            self.assertEqual(len(destination.blocks), 0)
            self.assertFalse(destination._sparse_state.ready)

    def test_growth_prefix_keeps_exact_border_trits_controls_and_zero_new_edges(self):
        old = sparse_owner(rows=5, columns=7)
        levels = (torch.arange(35).reshape(5, 7) % 3 - 1).to(torch.int8)
        old.set_effective_weights(levels)
        block = old.blocks["r0_c0"]
        block.stability.copy_(torch.arange(35).reshape(5, 7).float()/3)
        block.uses.copy_(torch.arange(35).reshape(5, 7).float()/2)
        block.eligibility_accumulator.copy_(torch.arange(-17, 18).reshape(5, 7).short())
        pager = self.pager()
        new = sparse_owner(pager, rows=67, columns=69)
        new._sparse_state.copy_from(old._sparse_state)
        self.assertEqual(list(new.blocks), ["r0_c0"])
        for field in ("weights", "stability", "uses", "eligibility_accumulator"):
            expected = torch.zeros((67, 69), dtype=torch.int8 if field == "weights" else getattr(block, field).dtype)
            expected[:5, :7] = levels if field == "weights" else getattr(block, field)
            self.assertTrue(torch.equal(virtual_field(new, field), expected), field)
        new._sparse_state.validate()
        pager.close()

    def test_ternary_inventory_and_layout_are_allocated_only_not_virtual_dense_export(self):
        value = sparse_owner(rows=67, columns=69)
        self.assertEqual(collect_module_ternary_tensors({"router": value}), ())
        levels = torch.zeros(67, 69, dtype=torch.int8); levels[66, 68] = -1
        value.set_effective_weights(levels)
        specs = collect_module_ternary_tensors({"router": value})
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].name, "router.blocks.r1_c1.weights")
        self.assertEqual(specs[0].values.shape, (3, 5))
        self.assertEqual(inspect_module_ternary_layout({"router": value}, dynamic_synapses={}),
                         {"router.blocks.r1_c1.weights": ((3, 5), "dynamic-synapse")})
        self.assertEqual(value.logical_ternary_parameter_count, 15)
        self.assertEqual(value.authoritative_packed_tensors(), ())
        self.assertEqual(sum(tensor.numel() for tensor in value.iter_authoritative_packed_tensors()), 6)
        self.assertEqual(sum(tensor.numel() for child in value.modules()
                             if callable(getattr(child, "authoritative_packed_tensors", None))
                             for tensor in child.authoritative_packed_tensors()), 6)

    def test_block_keys_are_canonical_and_out_of_range_blocks_are_rejected(self):
        self.assertEqual(parse_block_key(block_key(10, 1000)), (10, 1000))
        for bad in ("r01_c0", "r0_c-1", "r0_c0.x", "r1_c00"):
            with self.assertRaises(ValueError):
                parse_block_key(bad)
        value = sparse_owner()
        with self.assertRaises(ValueError):
            value._sparse_state.ensure(1, 0)

    def test_normal_checksum_never_virtualizes_potential_and_is_canonical(self):
        value = sparse_owner(rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8)
        levels[0, 0], levels[66, 68] = 1, -1
        value.set_effective_weights(levels)
        with patch.object(value._sparse_state, "dense_field_chunks", side_effect=AssertionError("implicit NxN scan")):
            before = value.checksum_with_controls()
            weights_before = value.decoded_weight_checksum()
            value.blocks = torch.nn.ModuleDict(reversed(tuple(value.blocks.items())))
            self.assertEqual(value.checksum_with_controls(), before)
            self.assertEqual(value.decoded_weight_checksum(), weights_before)
            value.pre_trace.add_(1.)
            self.assertNotEqual(value.checksum_with_controls(), before)
            self.assertEqual(value.decoded_weight_checksum(), weights_before)

    def test_shared_pool_quota_and_surviving_tensor_views_stay_charged(self):
        policy, ledger = dense_fixture.RouterStatePagingFixtures.policy(self)
        pager = self.pager(policy=policy)
        value = sparse_owner(pager, rows=3, columns=5)
        levels = torch.zeros(3, 5, dtype=torch.int8); levels[0, 0] = 1
        value.set_effective_weights(levels)
        held = value.blocks["r0_c0"]._packed_weights
        used = ledger.status()["physicalSpillUsageBytes"]
        self.assertGreater(used, 0)
        pager.close()
        self.assertGreater(ledger.status()["physicalSpillUsageBytes"], 0)
        self.assertLessEqual(ledger.status()["physicalSpillUsageBytes"], used)
        del held; gc.collect()
        self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_shared_pool_refusal_precedes_topology_and_timing_mutation(self):
        policy, ledger = dense_fixture.RouterStatePagingFixtures.policy(self, pool=1)
        pager = self.pager(policy=policy)
        value = sparse_owner(pager)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        with self.assertRaises(NeuralStateResourcePause):
            value.step(torch.ones(5), torch.ones(3))
        self.assertEqual(len(value.blocks), 0)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(value.state_dict()[name], tensor), name)
        self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)
        pager.close()

    def journal(self, value, *, rollback=False):
        journal = ParameterDeltaJournal((("router", value),), directory=self.root / "diagnostic",
            reserve_ram=lambda _size: None, reserve_disk=lambda _size, _operation: None, include_state=rollback)
        self.addCleanup(journal.close)
        return journal

    def test_dynamic_zero_birth_tracks_actual_net_trits_and_not_padding_or_flips(self):
        value = sparse_owner()
        journal = self.journal(value)
        levels = torch.zeros(3, 5, dtype=torch.int8); levels[2, 4] = 1
        value.set_effective_weights(levels)
        self.assertEqual(journal.delta_norm(), 1.)
        self.assertIsNone(journal.database)  # original is known implicit 0x55
        levels[2, 4] = -1
        value.set_effective_weights(levels)
        self.assertEqual(journal.delta_norm(), 1.)
        levels.zero_(); value.set_effective_weights(levels)
        self.assertEqual(journal.delta_norm(), 0.)
        item = next(item for item in journal.original.values() if item.get("implicit_zero_birth"))
        self.assertEqual(item["logical_columns"], 5)

    def test_existing_sparse_owner_journal_hooks_and_failure_restore_are_exact(self):
        pager = self.pager()
        value = sparse_owner(pager, rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8); levels[0, 0] = 1
        value.set_effective_weights(levels)
        journal = self.journal(value)
        ensure = value._sparse_state.ensure
        def fail_late(row, column, **kwargs):
            if row == 1 and column == 0:
                raise NeuralStateResourcePause("fixture later new block", {})
            return ensure(row, column, **kwargs)
        with patch.object(value._sparse_state, "ensure", side_effect=fail_late):
            with self.assertRaises(NeuralStateResourcePause):
                value.step(torch.ones(69), torch.ones(67))
        self.assertEqual(journal.delta_norm(), 0.)
        levels[0, 0] = -1
        value.set_effective_weights(levels)
        self.assertEqual(journal.delta_norm(), 2.)
        journal.close(); pager.close()

    def test_snapshot_journal_retirement_restores_absent_block_topology(self):
        pager = self.pager()
        value = sparse_owner(pager)
        journal = self.journal(value, rollback=True)
        value.step(torch.ones(5), torch.ones(3))
        self.assertEqual(len(value.blocks), 1)
        journal.restore()
        self.assertEqual(len(value.blocks), 0)
        self.assertEqual(pager.status()["registeredStateBytes"], 0)
        self.assertEqual(journal.delta_norm(), 0.)
        pager.close()

    def test_birth_observer_refusal_does_not_leave_module_or_page_metadata(self):
        pager = self.pager()
        value = sparse_owner(pager)
        journal = self.journal(value)
        def refuse_birth(size):
            if size == 2048:
                raise NeuralStateResourcePause("fixture birth metadata ceiling", {})
        journal.reserve_ram = refuse_birth
        levels = torch.zeros(3, 5, dtype=torch.int8); levels[0, 0] = 1
        with self.assertRaises(NeuralStateResourcePause):
            value.set_effective_weights(levels)
        self.assertEqual(len(value.blocks), 0)
        self.assertEqual(pager.status()["registeredStateBytes"], 0)
        journal.reserve_ram = lambda _size: None
        self.assertEqual(journal.delta_norm(), 0.)
        pager.close()

    def test_packed_snapshot_walks_actual_children_and_skips_only_declared_container(self):
        value = sparse_owner()
        empty = PackedMutationSnapshot.capture((value,), reserve=lambda _size: None)
        empty.close()
        levels = torch.zeros(3, 5, dtype=torch.int8); levels[0, 0] = 1
        value.set_effective_weights(levels)
        snapshot = PackedMutationSnapshot.capture((value,), reserve=lambda _size: None)
        value.set_effective_weights(-levels)
        snapshot.restore()
        self.assertTrue(torch.equal(virtual_field(value, "weights"), levels))
        snapshot.close()
        unknown = torch.nn.Module()
        unknown.authoritative_packed_tensors = lambda: ()
        with self.assertRaises(ValueError):
            PackedMutationSnapshot.capture((unknown,), reserve=lambda _size: None)

    def test_packed_batch_retry_removes_only_post_boundary_sparse_births(self):
        pager = self.pager()
        value = sparse_owner(pager, rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8); levels[0, 0] = 1
        value.set_effective_weights(levels)
        journal = self.journal(value)
        snapshot = PackedMutationSnapshot.capture((value,), reserve=lambda _size: None)
        levels[66, 68] = -1; levels[0, 0] = -1
        value.set_effective_weights(levels)
        self.assertEqual(len(value.blocks), 2)
        snapshot.restore()
        self.assertEqual(list(value.blocks), ["r0_c0"])
        self.assertEqual(journal.delta_norm(), 0.)
        snapshot.restore()
        self.assertEqual(list(value.blocks), ["r0_c0"])
        snapshot.close(); journal.close(); pager.close()

    def test_settings_policy_alias_rebinds_actual_router_admission(self):
        calls = []
        class Policy:
            def __init__(self, name): self.name = name
            def reserve_ram(self, amount, operation):
                calls.append((self.name, amount, operation))
                return nullcontext()
        old, new = Policy("old"), Policy("new")
        pager = self.pager(policy=old)
        pager.resource_policy = new
        self.assertIs(pager.policy, new)
        with pager.ram(123, "fixture selected-budget admission"):
            pass
        self.assertEqual(calls, [("new", 123, "fixture selected-budget admission")])
        pager.close()

    def test_real_ternary_export_roundtrip_uses_only_allocated_border_shape(self):
        value = sparse_owner(rows=67, columns=69)
        levels = torch.zeros(67, 69, dtype=torch.int8); levels[66, 68] = -1
        value.set_effective_weights(levels)
        destination = self.root / "packed"
        manifest = export_module_ternary_shards(destination, {"router": value})
        verified = verify_ternary_shards(destination, expected_names=("router.blocks.r1_c1.weights",))
        self.assertEqual(len(manifest["tensors"]), 1)
        self.assertEqual(manifest["tensors"][0]["shape"], [3, 5])
        self.assertEqual(verified.tensors["router.blocks.r1_c1.weights"].dtype, torch.int8)
        self.assertTrue(torch.equal(verified.tensors["router.blocks.r1_c1.weights"], levels[64:, 64:]))


if __name__ == "__main__":
    unittest.main()
