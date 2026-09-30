"""Tiny constructor-free tensor/control/storage fixtures, no neural run."""
import ast
import gc
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded, load_module_bounded
from omni_core.model import pack_ternary_weight, unpack_ternary_weight_rows
from omni_core.router_state_paging import MATRIX_FIELDS, RouterStatePager, RouterMutationJournal
from omni_core.spiking import STDPSynapses
from omni_core.offload import NeuralStateResourcePause, ResourcePolicy, ResourceReading, GIB, MIB
from omni_core.shared_resource_ledger import SharedResourceLedger
from omni_core.persistence import tensor_checksum


def owner(pager=None, *, loading=False, rows=3, columns=5):
    value = STDPSynapses.__new__(STDPSynapses)
    torch.nn.Module.__init__(value)
    value.pre_neurons, value.post_neurons = columns, rows
    value.learning_rate, value.pre_decay, value.post_decay = .035, .8, .7
    value.a_plus, value.a_minus, value.metaplasticity_rate = 1., 1.05, .025
    value.weight_limit, value.ternary = 1., True
    value._router_state_pager, value._router_loading_checkpoint = pager, loading
    for name, shape, dtype, fill in (
        ("_packed_weights", (rows, (columns+3)//4), torch.uint8, 0x55),
        ("eligibility_accumulator", (rows, columns), torch.int16, 0),
        ("stability", (rows, columns), torch.float32, 0),
        ("uses", (rows, columns), torch.float32, 0),
    ):
        value.register_buffer(name, value._router_buffer(name, shape, dtype, fill))
    value.register_buffer("pre_trace", torch.arange(columns).float() / 9)
    value.register_buffer("post_trace", torch.arange(rows).float() / 7)
    value.register_buffer("plasticity_events", torch.tensor(11))
    value.register_buffer("decay_cycles", torch.tensor(2))
    return value


def dense_step(value, pre, post):
    state = {name: tensor.clone() for name, tensor in value.state_dict().items()}
    levels = unpack_ternary_weight_rows(state["_packed_weights"], value.pre_neurons)
    signal = value.a_plus * torch.outer(post, state["pre_trace"]) - value.a_minus * torch.outer(state["post_trace"], pre)
    delta = value.learning_rate / (1 + state["stability"]) * signal
    active = signal.ne(0)
    if bool(active.any()):
        agreement = levels.sign().eq(delta.sign()) | levels.eq(0)
        change = torch.where(agreement, torch.full_like(state["stability"], value.metaplasticity_rate),
                             torch.full_like(state["stability"], -value.metaplasticity_rate * .25))
        state["stability"].add_(change * active).clamp_(0, 20)
        state["uses"].add_(active.to(state["uses"]))
        increments = (delta / max(value.learning_rate, 1e-6) * 256).round().clamp(-256, 256).int()
        pressure = (state["eligibility_accumulator"].int() + increments).clamp(-256, 256)
        transition = torch.where(pressure >= 256, torch.ones_like(pressure),
            torch.where(pressure <= -256, -torch.ones_like(pressure), torch.zeros_like(pressure)))
        updated = (levels.int() + transition).clamp(-1, 1)
        pressure -= transition * 256
        state["eligibility_accumulator"] = torch.where(updated.eq(levels), 0, pressure).short()
        state["_packed_weights"] = pack_ternary_weight(updated.to(torch.int8))
        state["plasticity_events"].add_(int(active.sum()))
    state["pre_trace"].mul_(value.pre_decay).add_(pre)
    state["post_trace"].mul_(value.post_decay).add_(post)
    return state, delta


class RouterStatePagingFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="router-storage-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def pager(self, *, hot=0, journal=0, policy=None, provider=None):
        return RouterStatePager(self.root, hot_bytes=hot, tile_bytes=2048, journal_ram_bytes=journal,
                                resource_policy=policy, budget_provider=provider)

    def policy(self, pool=MIB):
        ledger = SharedResourceLedger(self.root / "quota.sqlite3", process_alive=lambda _: True)
        reading = ResourceReading(16 * GIB, 14 * GIB, 128 * MIB, 100 * GIB, 90 * GIB)
        policy = ResourcePolicy(self.root, reading_provider=lambda: reading,
                                include_accelerator_memory=False,
                                shared_resource_owner_id="router-fixture", shared_storage_pool_bytes=pool,
                                shared_ledger=ledger)
        return policy, ledger

    def test_exact_fitting_ram_budget_is_not_double_counted_or_eagerly_written(self):
        required = 3 * 2 + 3 * 5 * 10
        pager = self.pager(hot=required)
        owner(pager)
        self.assertEqual(pager.status()["cpuHeapBytes"], required)
        self.assertEqual(pager.status()["mappedLogicalBytes"], 0)
        self.assertEqual(list(pager.directory.iterdir()), [])
        pager.close()

    def test_hot_residency_recovers_and_pressure_spill_does_not_change_learned_shape(self):
        budget = {"hotBytes": 0, "tileBytes": 2048, "journalRamBytes": 0}
        pager = self.pager(provider=lambda _: budget)
        value = owner(pager)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        budget["hotBytes"] = 4096
        with pager.operation(value):
            pass
        self.assertEqual(pager.status()["mappedLogicalBytes"], 0)
        revision = pager.mutation_revision
        budget["hotBytes"] = 0
        with pager.operation(value):
            pass
        self.assertGreater(pager.status()["mappedLogicalBytes"], 0)
        self.assertEqual(pager.mutation_revision, revision)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(getattr(value, name), tensor), name)
        pager.close()

    def test_router_pages_and_dirty_journal_share_real_inode_quota_until_views_die(self):
        policy, ledger = self.policy()
        pager = self.pager(policy=policy)
        value = owner(pager)
        value.step(torch.ones(5), torch.ones(3))
        self.assertEqual(pager.status()["rollbackJournalBytes"], 0)
        used = ledger.status()["physicalSpillUsageBytes"]
        self.assertGreater(used, 0)
        held = value._packed_weights
        pager.close()
        self.assertGreater(ledger.status()["physicalSpillUsageBytes"], 0)
        self.assertLessEqual(ledger.status()["physicalSpillUsageBytes"], used)
        del held
        gc.collect()
        self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_designated_pool_exhaustion_before_write_preserves_every_state_value(self):
        policy, ledger = self.policy(pool=1)
        pager = self.pager(hot=4096, policy=policy)
        value = owner(pager)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        with self.assertRaises(NeuralStateResourcePause):
            value.step(torch.ones(5), torch.ones(3))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(getattr(value, name), tensor), name)
        self.assertEqual(list(pager.directory.iterdir()), [])
        self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)
        pager.close()

    def test_failed_later_dirty_tile_reservation_rolls_back_prior_committed_tiles(self):
        policy, ledger = self.policy()
        pager = self.pager(policy=policy)
        value = owner(pager)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        reserve = policy.reserve_spill
        calls = 0
        def limited(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 7:
                raise NeuralStateResourcePause("fixture later tile pool exhausted", {})
            return reserve(*args, **kwargs)
        policy.reserve_spill = limited
        with self.assertRaises(NeuralStateResourcePause):
            value.step(torch.ones(5), torch.ones(3))
        self.assertEqual(calls, 7)
        for name, tensor in before.items():
            self.assertTrue(torch.equal(getattr(value, name), tensor), name)
        self.assertEqual(list(pager.directory.glob("*.rollback")), [])
        pager.close(); gc.collect()
        self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_explicit_late_invalid_trit_replacement_rolls_back_previous_tiles(self):
        pager = self.pager()
        value = owner(pager)
        before = value._packed_weights.clone()
        levels = torch.ones(3, 5, dtype=torch.int8)
        levels[-1, -1] = 2
        with self.assertRaises(ValueError):
            value.set_effective_weights(levels)
        self.assertTrue(torch.equal(value._packed_weights, before))
        pager.close()

    def test_exact_short_write_loop_and_ram_to_disk_before_image_replay(self):
        pager = self.pager(hot=4096, journal=280)
        value = owner(pager)
        journal = RouterMutationJournal(value, pager, ram_bytes=280)
        journal.capture("pre_trace", 0, 2)
        value.pre_trace[:2].fill_(7.)
        journal.capture("post_trace", 0, 2)
        value.post_trace[:2].fill_(9.)
        original = journal.file
        class ShortWriter:
            def __getattr__(self, name):
                return getattr(original, name)
            def write(self, block):
                return original.write(block[:max(1, len(block) // 2)])
        journal.file = ShortWriter()
        journal.capture("plasticity_events")
        value.plasticity_events.fill_(27)
        journal.rollback()
        self.assertTrue(torch.equal(value.pre_trace, torch.arange(5).float() / 9))
        self.assertTrue(torch.equal(value.post_trace, torch.arange(3).float() / 7))
        self.assertEqual(int(value.plasticity_events), 11)
        journal.close(); pager.close()

    def test_ram_first_and_exact_mapped_control_ownership(self):
        hot, cold = self.pager(hot=4096, journal=4096), self.pager()
        first, second = owner(hot), owner(cold)
        self.assertEqual(hot.status()["mappedLogicalBytes"], 0)
        self.assertGreater(cold.status()["mappedLogicalBytes"], 0)
        for name in MATRIX_FIELDS:
            self.assertTrue(torch.equal(getattr(first, name), getattr(second, name)))
            self.assertEqual(getattr(second, name).device.type, "cpu")
        held = second._packed_weights
        status = cold.close()
        self.assertGreater(status["mappedBytesStillReferencedAfterClose"], 0)
        self.assertEqual(int(held[0, 0]), 0x55)
        hot.close()

    def test_tiled_recurrence_includes_all_existing_inhibitory_and_positive_edges(self):
        pager = self.pager()
        value = owner(pager)
        levels = torch.tensor([[1,-1,0,1,-1], [0,1,-1,0,1], [-1,-1,1,1,0]], dtype=torch.int8)
        value.set_effective_weights(levels)
        vector = torch.tensor([1., 0., 1., 1., 0.])
        with patch.object(value, "effective_weight", side_effect=AssertionError("whole recurrent decode")):
            self.assertTrue(torch.equal(value.recurrent(vector), torch.mv(levels.float(), vector)))
            self.assertEqual(value.active_synapse_count(), int(levels.ne(0).sum()))
        pager.close()

    def test_exact_causal_and_anti_causal_control_values_match_old_dense_equations(self):
        for pre, post in ((torch.tensor([1.,0.,1.,.5,1.]), torch.tensor([0.,1.,.5])),
                          (torch.zeros(5), torch.ones(3)), (torch.ones(5), torch.zeros(3)),
                          (torch.zeros(5), torch.tensor([0.,0.,1.]))):
            pager = self.pager()
            value = owner(pager)
            value.eligibility_accumulator.copy_(torch.arange(15).reshape(3,5).short() - 7)
            value.stability.copy_(torch.arange(15).reshape(3,5).float() / 4)
            expected, delta = dense_step(value, pre, post)
            with patch.object(value, "effective_weight", side_effect=AssertionError("whole STDP decode")), patch("torch.outer", side_effect=AssertionError("whole outer product")):
                summary = value.step(pre, post)
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(getattr(value, name), tensor), name)
            self.assertAlmostEqual(summary.delta_absolute_sum, float(delta.abs().double().sum()), places=10)
            self.assertEqual(list(pager.directory.glob("*.rollback")), [])
            pager.close()

    def test_crossing_saturation_and_inactive_eligibility_follow_identical_dense_equations(self):
        for pre, post in ((torch.zeros(5), torch.ones(3)), (torch.ones(5), torch.zeros(3))):
            pager = self.pager()
            value = owner(pager)
            value.set_effective_weights(torch.tensor([[1,-1,0,1,-1],[0,1,-1,0,1],[-1,-1,1,1,0]], dtype=torch.int8))
            value.pre_trace.fill_(2.)
            value.post_trace.fill_(2.)
            value.eligibility_accumulator.copy_(torch.linspace(-255,255,15).round().short().reshape(3,5))
            expected, _ = dense_step(value, pre, post)
            before = unpack_ternary_weight_rows(value._packed_weights, 5).clone()
            summary = value.step(pre, post)
            for name, tensor in expected.items():
                self.assertTrue(torch.equal(getattr(value, name), tensor), name)
            difference = unpack_ternary_weight_rows(value._packed_weights, 5).int() - before.int()
            self.assertEqual(summary.changed_absolute_sum, int(difference.abs().sum()))
            self.assertEqual(summary.changed_signed_sum, int(difference.sum()))
            self.assertGreater(summary.changed_pairs, 0)
            pager.close()

    def test_zero_timing_keeps_controls_and_advances_exact_trace_timing(self):
        value = owner(self.pager())
        value.reset_activity()
        value.eligibility_accumulator.fill_(37)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        summary = value.step(torch.ones(5), torch.zeros(3))
        self.assertEqual(summary.active_pairs, 0)
        for name in MATRIX_FIELDS:
            self.assertTrue(torch.equal(getattr(value, name), before[name]))
        self.assertTrue(torch.equal(value.pre_trace, torch.ones(5)))
        value._router_state_pager.close()

    def test_cancel_during_global_quiet_pair_reset_replays_original_eligibility(self):
        pager = self.pager()
        value = owner(pager)
        value.eligibility_accumulator.fill_(17)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks == 7
        pager.cancelled = cancel
        with self.assertRaises(InterruptedError):
            value.step(torch.zeros(5), torch.tensor([0.,0.,1.]))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(getattr(value, name), tensor), name)
        pager.cancelled = None
        pager.close()

    def test_managed_controls_stay_cpu_under_module_apply_without_dtype_shadow(self):
        pager = self.pager()
        value = owner(pager)
        controls = {name: getattr(value, name) for name in MATRIX_FIELDS}
        value._apply(lambda tensor: tensor.double() if tensor.is_floating_point() else tensor)
        self.assertEqual(value.pre_trace.dtype, torch.float64)
        for name, tensor in controls.items():
            self.assertIs(getattr(value, name), tensor)
            self.assertEqual(getattr(value, name).device.type, "cpu")
        pager.close()

    def test_replaced_owner_retirement_and_residue_are_not_an_ever_growing_live_cache(self):
        pager = self.pager()
        old = owner(pager)
        replacement = owner(pager)
        logical = sum(getattr(replacement, name).numel() * getattr(replacement, name).element_size() for name in MATRIX_FIELDS)
        held = old._packed_weights
        with pager.operation(replacement):
            with self.assertRaises(RuntimeError):
                pager.release_owner(old)
        pager.release_owner(old)
        self.assertEqual(pager.status()["mappedLogicalBytes"], logical)
        self.assertTrue(all(getattr(old, name) is None for name in MATRIX_FIELDS))
        self.assertEqual(int(held[0,0]), 0x55)
        pager.close()

    def test_historical_decoded_weight_and_control_checksums_remain_byte_identical(self):
        for rows, columns in ((3,5), (2,131), (1,4), (5,1)):
            pager = self.pager()
            value = owner(pager, rows=rows, columns=columns)
            levels = ((torch.arange(rows*columns).reshape(rows,columns) % 3) - 1).to(torch.int8)
            value.set_effective_weights(levels)
            value.stability.copy_(torch.arange(rows*columns).reshape(rows,columns).float() % 20)
            value.uses.copy_(torch.arange(rows*columns).reshape(rows,columns).float() / 3)
            historical_decoded = value.effective_weight()
            self.assertEqual(historical_decoded.dtype, torch.int8)
            old_weight = tensor_checksum([historical_decoded], chunk_bytes=7)
            old_fresh = tensor_checksum([historical_decoded, value.stability, value.uses, value.plasticity_events], chunk_bytes=7)
            with patch.object(value, "effective_weight", side_effect=AssertionError("checksum whole decode")):
                self.assertEqual(value.decoded_weight_checksum(), old_weight)
                self.assertEqual(value.checksum_with_controls(), old_fresh)
                blocks = list(value.iter_decoded_weight_chunks())
            self.assertTrue(torch.equal(torch.cat(blocks), levels.reshape(-1)))
            self.assertTrue(all(block.numel()*16+1024 <= pager.tile_bytes for block in blocks))
            self.assertEqual(pager.status()["mappedHotRetainedBytesEstimate"], 0)
            pager.close()

    def test_historical_control_checksum_also_preserves_strided_c_order_bytes(self):
        value = owner()
        value.stability = torch.arange(15).reshape(5,3).float().t()
        value.uses = torch.arange(15).reshape(3,5).float().flip(1)
        expected = tensor_checksum([value.effective_weight(), value.stability, value.uses, value.plasticity_events])
        self.assertEqual(value.checksum_with_controls(), expected)

    def test_mid_tile_cancel_rolls_back_exact_codes_controls_traces_and_counters(self):
        pager = self.pager()
        value = owner(pager)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            return checks >= 4
        pager.cancelled = cancel
        with self.assertRaises(InterruptedError):
            value.step(torch.ones(5), torch.ones(3))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(getattr(value, name), tensor), name)
        pager.cancelled = None
        pager.close()

    def test_decay_uses_original_global_edge_positions_not_tile_local_randomness(self):
        pager = self.pager()
        value = owner(pager)
        levels = torch.tensor([[1,-1,0,1,-1], [0,1,-1,0,1], [-1,-1,1,1,0]], dtype=torch.int8)
        value.set_effective_weights(levels)
        value.stability.fill_(2.)
        positions = torch.arange(15).reshape(3,5)
        draw = ((positions * 1664525 + 3 * 1013904223) & 0xffffffff).float() / 4294967296.
        expected = torch.where(levels.ne(0) & (draw < .5 / (1+value.uses)), 0, levels).to(torch.int8)
        value.decay_unused(.5)
        self.assertTrue(torch.equal(unpack_ternary_weight_rows(value._packed_weights, 5), expected))
        self.assertTrue(torch.equal(value.stability, torch.full((3,5), 1.9)))
        self.assertEqual(int(value.decay_cycles), 3)
        pager.close()

    def test_bounded_loader_writer_preserve_immutable_file_and_all_original_tensor_keys(self):
        original = owner()
        path = self.root / "plasticity.safetensors"
        values = {"router.synapses."+name: tensor for name, tensor in original.state_dict().items()}
        atomic_save_tensors_bounded(path, values)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        pager = self.pager()
        value = owner(pager, loading=True)
        with self.assertRaisesRegex(RuntimeError, "before.*load"):
            value.active_synapse_count()
        load_module_bounded(value, BoundedTensorFile(path, chunk_bytes=24), "router.synapses.")
        self.assertEqual(set(value.state_dict()), set(original.state_dict()))
        for name, tensor in original.state_dict().items():
            self.assertTrue(torch.equal(getattr(value, name), tensor))
        value.step(torch.ones(5), torch.ones(3))
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
        pager.close()

    def test_failed_construction_unlinks_only_session_owned_backing(self):
        pager = self.pager()
        with self.assertRaisesRegex(RuntimeError, "fixture abort"):
            with pager.construction():
                owner(pager)
                raise RuntimeError("fixture abort")
        self.assertEqual(pager.status()["mappedLogicalBytes"], 0)
        self.assertEqual(list(pager.directory.glob("*.router-page")), [])
        pager.close()

    def test_production_route_feedback_and_step_source_have_no_whole_decode_or_outer(self):
        tree = ast.parse((Path(__file__).parents[1]/"omni_core"/"spiking.py").read_text())
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                for method in cls.body:
                    if isinstance(method, ast.FunctionDef) and method.name in {"route", "apply_feedback", "step", "decay_unused"}:
                        attrs = [call.func.attr for call in ast.walk(method) if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)]
                        self.assertNotIn("outer", attrs)
                        self.assertNotIn("effective_weight", attrs)


if __name__ == "__main__":
    unittest.main()
