"""Tiny tensor/storage tests for weighted native STDP; no brain training."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from omni_core.offload import NeuralStateResourcePause
from omni_core.packed_connections import connection_strengths, connection_resistance
from omni_core.router_state_paging import RouterStatePager
from omni_core.sparse_router_state import block_key
from omni_core.spiking import AssociativeSpikingRouter
from omni_core.parameter_diagnostics import ParameterDeltaJournal
from omni_core.optimizers import PackedMutationSnapshot
from .test_sparse_router_state import sparse_owner


class WeightedSparseRouterFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="weighted-router-fixture-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def one(self, pager=None, *, sign=1, strength=1, resistance=0):
        value = sparse_owner(pager, rows=1, columns=1)
        value.reset_activity()
        value.a_plus = value.a_minus = 1.0
        value.metaplasticity_rate = 0.0
        value.set_effective_weights(torch.tensor([[sign]], dtype=torch.int8))
        if not value.blocks:
            with value._sparse_state.transaction():
                value._sparse_state.ensure(0, 0)
        value.blocks["r0_c0"]._connection_state.fill_((strength - 1) | (resistance << 4))
        return value

    def causal(self, value, amount=1.0):
        value.reset_activity()
        value.pre_trace.fill_(amount)
        return value.step(torch.zeros(value.pre_neurons), torch.ones(value.post_neurons))

    def anti(self, value, amount=1.0):
        value.reset_activity()
        value.post_trace.fill_(amount)
        return value.step(torch.ones(value.pre_neurons), torch.zeros(value.post_neurons))

    def save(self, value, name="router.safetensors"):
        path = self.root / name
        atomic_save_tensors_bounded(path, {"router."+key: tensor for key, tensor in value.state_dict().items()}, chunk_bytes=32)
        return BoundedTensorFile(path, chunk_bytes=32)

    def test_signed_strength_is_used_by_recurrence_and_changes_its_checksum(self):
        value = self.one(sign=-1, strength=6)
        self.assertEqual(float(value.recurrent(torch.tensor([2.]))[0]), -12.)
        self.assertEqual(int(value.effective_weight()[0, 0]), -6)
        before = value.decoded_weight_checksum()
        value.blocks["r0_c0"]._connection_state.fill_(6)
        self.assertNotEqual(value.decoded_weight_checksum(), before)
        self.assertEqual(int(value.effective_weight()[0, 0]), -7)
        self.assertTrue(all(tensor.dtype == torch.uint8 for tensor in value.iter_authoritative_packed_tensors()))

    def test_fractional_causal_events_accumulate_instead_of_being_erased(self):
        value = self.one(sign=0)
        block = value.blocks["r0_c0"]
        for index in range(3):
            summary = self.causal(value, .25)
            self.assertEqual(summary.changed_pairs, 0)
            self.assertEqual(int(block.eligibility_accumulator), (index+1)*64)
        summary = self.causal(value, .25)
        self.assertEqual(summary.changed_pairs, 1)
        self.assertEqual(int(block.eligibility_accumulator), 0)
        self.assertEqual(int(value.effective_weight()), 1)

    def test_opposite_timing_weakens_strength_before_changing_polarity(self):
        value = self.one(sign=1, strength=3)
        self.anti(value)
        self.assertEqual(int(value.effective_weight()), 2)
        self.anti(value)
        self.assertEqual(int(value.effective_weight()), 1)
        self.anti(value)
        self.assertEqual(int(value.effective_weight()), 0)
        self.anti(value)
        self.assertEqual(int(value.effective_weight()), -1)

    def test_each_contact_resistance_changes_its_required_timing_evidence(self):
        value = self.one(strength=1, resistance=3)
        block = value.blocks["r0_c0"]
        for index in range(3):
            self.causal(value)
            self.assertEqual(int(value.effective_weight()), 1)
            self.assertEqual(int(block.eligibility_accumulator), (index+1)*256)
        self.causal(value)
        self.assertEqual(int(value.effective_weight()), 2)
        self.assertEqual(int(connection_resistance(block._connection_state)), 4)

    def test_unrelated_active_pairs_do_not_erase_waiting_eligibility(self):
        value = sparse_owner(rows=2, columns=2)
        value.metaplasticity_rate = 0.0
        value.reset_activity()
        value.pre_trace[0] = .25
        value.step(torch.zeros(2), torch.tensor([1., 0.]))
        block = value.blocks["r0_c0"]
        self.assertEqual(int(block.eligibility_accumulator[0, 0]), 64)
        value.reset_activity()
        value.pre_trace[1] = 1.
        value.step(torch.zeros(2), torch.tensor([0., 1.]))
        self.assertEqual(int(block.eligibility_accumulator[0, 0]), 64)

    def test_saturated_positive_and_negative_contacts_grow_independent_parallel_paths(self):
        for sign in (-1, 1):
            value = self.one(sign=sign, strength=16)
            summary = self.causal(value) if sign > 0 else self.anti(value)
            self.assertEqual(summary.new_contacts, 1)
            self.assertEqual(summary.active_pairs, 1)
            self.assertEqual(int(value.effective_weight()), sign*17)
            self.assertEqual(value.active_synapse_count(), 2)
            self.assertEqual(list(value.blocks), ["r0_c0", "r0_c0_p1"])
            extra = value.blocks["r0_c0_p1"]
            self.assertEqual(int(connection_strengths(extra._connection_state)), 1)
            self.assertEqual(int(connection_resistance(extra._connection_state)), 0)
            summary = self.causal(value) if sign > 0 else self.anti(value)
            self.assertEqual(summary.new_contacts, 0)
            self.assertEqual(int(value.effective_weight()), sign*18)
            self.assertEqual(len(value.blocks), 2)
            self.assertEqual(summary.contact_updates, 2)

    def test_parallel_layers_have_no_fixed_index_ceiling_and_share_a_block_for_many_pairs(self):
        value = sparse_owner(rows=2, columns=2)
        value.reset_activity(); value.metaplasticity_rate = 0.0
        value.set_effective_weights(torch.ones(2, 2, dtype=torch.int8))
        value.blocks["r0_c0"]._connection_state.fill_(15)
        value.pre_trace.fill_(1.)
        summary = value.step(torch.zeros(2), torch.ones(2))
        self.assertEqual(summary.new_contacts, 4)
        self.assertEqual(len(value.blocks), 2)
        with value._sparse_state.transaction():
            value._sparse_state.ensure(0, 0, layer=1_000_001)
        self.assertIn(block_key(0, 0, 1_000_001), value.blocks)
        self.assertEqual(value.sparse_state_status()["connectionStateBytes"], 12)

    def test_parallel_state_roundtrip_and_prefix_growth_preserve_all_learned_bytes(self):
        value = self.one(strength=16)
        self.causal(value)
        self.causal(value)
        pager = RouterStatePager(self.root, hot_bytes=0, tile_bytes=2048, journal_ram_bytes=0)
        loaded = sparse_owner(pager, rows=1, columns=1, loading=True)
        loaded.load_recurrent_state_bounded(self.save(value), "router.")
        for name, tensor in value.state_dict().items():
            self.assertTrue(torch.equal(tensor, loaded.state_dict()[name]), name)
        self.assertEqual(value.checksum_with_controls(), loaded.checksum_with_controls())
        grown = sparse_owner(rows=3, columns=5)
        grown._sparse_state.copy_from(loaded._sparse_state)
        self.assertEqual(list(grown.blocks), ["r0_c0", "r0_c0_p1"])
        self.assertEqual(int(grown.effective_weight()[0, 0]), 18)
        self.assertEqual(int(grown.effective_weight().sum()), 18)
        pager.close()

    def test_pre_strength_sparse_checkpoint_migrates_to_unit_connections_exactly(self):
        original = self.one(sign=-1)
        old_state = {"router."+key: tensor.clone() for key, tensor in original.state_dict().items() if not key.endswith("_connection_state")}
        old_state["router._sparse_layout"][0] = 1
        path = self.root / "old-native.safetensors"
        atomic_save_tensors_bounded(path, old_state, chunk_bytes=16)
        loaded = sparse_owner(rows=1, columns=1, loading=True)
        proof = loaded.load_recurrent_state_bounded(BoundedTensorFile(path, chunk_bytes=16), "router.")
        self.assertTrue(proof["unitStrengthMigration"])
        self.assertEqual(proof["sourceConnectionStateVersion"], 1)
        self.assertEqual(int(loaded.blocks["r0_c0"]._connection_state), 0)
        self.assertEqual(float(loaded.recurrent(torch.tensor([3.]))), -3.)
        for name, tensor in old_state.items():
            if name != "router._sparse_layout":
                self.assertTrue(torch.equal(tensor, loaded.state_dict()[name[len("router."):]]), name)

    def test_v2_missing_strength_is_corruption_not_an_implicit_migration(self):
        original = self.one(strength=3)
        state = {"router."+key: tensor.clone() for key, tensor in original.state_dict().items() if not key.endswith("_connection_state")}
        path = self.root / "incomplete.safetensors"
        atomic_save_tensors_bounded(path, state)
        loaded = sparse_owner(rows=1, columns=1, loading=True)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            loaded.load_recurrent_state_bounded(BoundedTensorFile(path), "router.")
        self.assertEqual(len(loaded.blocks), 0)

    def test_contact_growth_failure_restores_weights_controls_and_topology(self):
        pager = RouterStatePager(self.root, hot_bytes=0, tile_bytes=2048, journal_ram_bytes=0)
        value = self.one(pager, strength=16)
        value.pre_trace.fill_(1.)
        before = {name: tensor.clone() for name, tensor in value.state_dict().items()}
        ensure = value._sparse_state.ensure
        def fail_growth(row, column, **kwargs):
            if kwargs.get("layer", 0):
                raise NeuralStateResourcePause("fixture parallel layer admission failure", {})
            return ensure(row, column, **kwargs)
        with patch.object(value._sparse_state, "ensure", side_effect=fail_growth):
            with self.assertRaises(NeuralStateResourcePause):
                value.step(torch.zeros(1), torch.ones(1))
        self.assertEqual(set(before), set(value.state_dict()))
        for name, tensor in before.items():
            self.assertTrue(torch.equal(tensor, value.state_dict()[name]), name)
        self.assertEqual(list(value.blocks), ["r0_c0"])
        self.assertEqual(value._sparse_state._layers, {(0, 0): {0: "r0_c0"}})
        self.assertEqual(list(pager.directory.glob("*.rollback")), [])
        pager.close()

    def test_decay_weakens_one_strength_unit_before_erasing_a_connection(self):
        value = self.one(strength=4)
        value.blocks["r0_c0"].uses.zero_()
        value.decay_unused(1.)
        self.assertEqual(int(value.effective_weight()), 3)
        self.assertEqual(value.active_synapse_count(), 1)

    def test_strength_only_updates_are_traced_and_packed_rollback_restores_them(self):
        value = self.one()
        before = value.decoded_weight_checksum()
        snapshot = PackedMutationSnapshot.capture((value,), reserve=lambda _size: None)
        journal = ParameterDeltaJournal((("router", value),), directory=self.root / "diagnostics",
            reserve_ram=lambda _size: None, reserve_disk=lambda _size, _operation: None)
        signs = value.blocks["r0_c0"]._packed_weights.clone()
        self.causal(value)
        self.assertTrue(torch.equal(signs, value.blocks["r0_c0"]._packed_weights))
        self.assertGreater(journal.delta_norm(), 0.)
        snapshot.restore()
        self.assertEqual(value.decoded_weight_checksum(), before)
        self.assertEqual(journal.delta_norm(), 0.)
        journal.close(); snapshot.close()

    def test_tiny_projection_growth_preserves_strengths_and_repeated_contact_identity(self):
        router = AssociativeSpikingRouter(idea_dim=2, neurons=2)
        prior = []
        for projection in (router.input_projection, router.output_projection):
            projection._connection_state.fill_(0x23)
            projection._bias_connection_state.fill_(0x12)
            projection.add_parallel_connection_(0, 0, sign=-1, strength=4)
            projection.add_parallel_connection_(0, 1, sign=1, strength=3, bias=True)
            prior.append((projection._connection_state.clone(), projection._bias_connection_state.clone(),
                          tuple(projection.iter_parallel_connection_records()), tuple(projection.iter_parallel_connection_records(bias=True))))
        router.grow_neurons(1)
        for projection, (state, bias, weights, biases) in zip((router.input_projection, router.output_projection), prior):
            self.assertTrue(torch.equal(projection._connection_state[:state.shape[0], :state.shape[1]], state))
            self.assertTrue(torch.equal(projection._bias_connection_state[:bias.shape[0], :bias.shape[1]], bias))
            self.assertEqual(tuple(projection.iter_parallel_connection_records()), weights)
            self.assertEqual(tuple(projection.iter_parallel_connection_records(bias=True)), biases)


if __name__ == "__main__":
    unittest.main()
