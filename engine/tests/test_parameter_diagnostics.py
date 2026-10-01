"""Constructor-free primitive/storage diagnostic and batch-lifetime fixtures.

No brain/model construction, backward, optimizer step, training, app or CI run.
"""
import hashlib
import inspect
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.parameter_diagnostics import ParameterDeltaJournal, packed_diagnostic_write, diagnostic_call
from omni_core.optimizers import PackedMutationSnapshot, packed_snapshot_lifetime
from omni_core.persistence import tensor_checksum
from omni_core.brain import AdaptiveBrain


class Owner:
    """Only registered primitive tensors and names; deliberately no nn.Model."""
    def __init__(self, rows=2, columns=1, control=False):
        self._buffers = {"packed": torch.full((rows, columns), 0x55, dtype=torch.uint8),
            "_row_stability": torch.zeros(rows, dtype=torch.uint8)}
        self._pending_stability_events = 0
        self.controls = {"gain": torch.tensor([1.0])} if control else {}
        self.children = {}
        self._native_core_pager = None
    def __getattr__(self, name):
        if name in self._buffers:
            return self._buffers[name]
        raise AttributeError(name)
    def named_buffers(self, recurse=False):
        return iter(self._buffers.items())
    def authoritative_packed_tensors(self):
        return (self._buffers["packed"],)
    def named_parameters(self):
        return iter(self.controls.items())
    def parameters(self):
        return iter(self.controls.values())
    def named_modules(self):
        yield "", self
        yield from self.children.items()
    def modules(self):
        yield self
        yield from self.children.values()
    def _validate_packed(self):
        pass


class ParameterDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="core-diagnostic-fixture-")
        self.root = Path(self.temporary.name)
        self.owner = Owner(control=True)
        self.journals = []
    def tearDown(self):
        for journal in self.journals:
            journal.close()
        self.temporary.cleanup()
    def journal(self, ram=None, disk=None):
        journal = ParameterDeltaJournal((("decoder", self.owner),), directory=self.root,
            reserve_ram=ram or (lambda _bytes: None), reserve_disk=disk or (lambda _bytes, _stage: None))
        self.journals.append(journal)
        return journal
    def install(self, values):
        target = self.owner._buffers["packed"]
        replacement = torch.tensor(values, dtype=torch.uint8).reshape(target.shape)
        with packed_diagnostic_write(self.owner, target, replacement):
            target.copy_(replacement)

    def test_baseline_never_clones_packed_core_or_creates_a_full_disk_snapshot(self):
        original = torch.Tensor.clone
        def no_packed_clone(value, *args, **kwargs):
            if value.dtype == torch.uint8:
                raise AssertionError("packed diagnostic clone")
            return original(value, *args, **kwargs)
        with patch.object(torch.Tensor, "clone", no_packed_clone):
            journal = self.journal()
        self.assertIsNone(journal.database)
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(journal.delta_norm(), 0.0)

    def test_exact_net_delta_for_repeated_changes_reversal_and_real_final_state(self):
        journal = self.journal()
        self.install([[0x56], [0x55]]) # zero -> +1 for one trit
        self.assertEqual(journal.delta_norm(), 1.0)
        self.install([[0x54], [0x55]]) # +1 -> -1; baseline remains zero
        self.assertEqual(journal.delta_norm(), 1.0) # not cumulative three flips
        self.install([[0x55], [0x55]])
        self.assertEqual(journal.delta_norm(), 0.0)
        self.owner.controls["gain"].fill_(3.0)
        self.assertEqual(journal.delta_norm(), 2.0)
        self.assertFalse(journal.scope["substrateDeltaMeasured"])

    def test_first_original_bytes_are_only_changed_positions(self):
        self.owner = Owner(rows=4, columns=32)
        journal = self.journal()
        replacement = self.owner.packed.clone()
        replacement[1, 7] = 0x56
        with packed_diagnostic_write(self.owner, self.owner.packed, replacement):
            self.owner.packed.copy_(replacement)
        record = journal.database.execute("SELECT positions,codes FROM original").fetchone()
        self.assertEqual(len(record[0]), 2)
        self.assertEqual(len(record[1]), 1)
        self.assertEqual(journal.delta_norm(), 1.0)

    def test_disk_and_scratch_refusals_precede_real_weight_mutation(self):
        journal = self.journal(disk=lambda _bytes, _stage: (_ for _ in ()).throw(RuntimeError("disk reserve")))
        before = self.owner.packed.clone()
        with self.assertRaisesRegex(RuntimeError, "disk reserve"):
            self.install([[0xAA], [0x55]])
        self.assertTrue(torch.equal(before, self.owner.packed))
        journal.close()
        def refuse_scratch(size):
            if size > 100000: raise RuntimeError("scratch reserve")
        journal = self.journal(ram=refuse_scratch)
        with self.assertRaisesRegex(RuntimeError, "scratch reserve"):
            self.install([[0xAA], [0x55]])
        self.assertTrue(torch.equal(before, self.owner.packed))

    def test_unknown_same_tensor_write_is_not_silently_reported_as_zero(self):
        journal = self.journal()
        self.owner.packed.fill_(0xAA)
        with self.assertRaisesRegex(RuntimeError, "outside its before-write journal"):
            journal.delta_norm()

    def test_failed_write_before_copy_does_not_count_a_proposed_mutation(self):
        journal = self.journal()
        with self.assertRaisesRegex(RuntimeError, "before actual copy"):
            with packed_diagnostic_write(self.owner, self.owner.packed, torch.full_like(self.owner.packed, 0xAA)):
                raise RuntimeError("before actual copy")
        self.assertEqual(journal.delta_norm(), 0.0)

    def test_stable_name_after_paging_replacement_uses_actual_final_buffer(self):
        journal = self.journal()
        self.install([[0x56], [0x55]])
        self.owner._buffers["packed"] = self.owner.packed.clone()
        self.install([[0x54], [0x55]])
        self.assertEqual(journal.delta_norm(), 1.0)

    def test_new_grown_packed_owner_is_not_lost_from_core_delta(self):
        journal = self.journal()
        grown = Owner(rows=1)
        grown.packed.fill_(0xAA)
        self.owner.children["grown"] = grown
        self.assertEqual(journal.delta_norm(), 2.0)

    def test_scope_finally_cleans_early_return_and_exception_journals(self):
        created = []
        @diagnostic_call
        def request(fail):
            journal = self.journal()
            self.install([[0xAA], [0x55]])
            created.append(journal)
            if fail: raise RuntimeError("early request failure")
            return "typed early yield"
        self.assertEqual(request(False), "typed early yield")
        with self.assertRaises(RuntimeError): request(True)
        self.assertTrue(all(journal.closed for journal in created))
        self.assertEqual(list(self.root.iterdir()), [])

    def test_real_ram_rollback_keeps_net_delta_exact_after_retry(self):
        journal = self.journal()
        snapshot = PackedMutationSnapshot.capture((self.owner,), reserve=lambda _bytes: None)
        self.install([[0xAA], [0x55]])
        self.owner._row_stability.fill_(7)
        snapshot.restore()
        self.assertEqual(journal.delta_norm(), 0.0)
        self.assertTrue(bool(self.owner._row_stability.eq(0).all()))
        self.install([[0x56], [0x55]])
        self.assertEqual(journal.delta_norm(), 1.0)
        snapshot.close()

    def test_same_operation_rollback_is_not_obstructed_by_diagnostic_ram_pressure(self):
        pressure = [False]
        def reserve(_size):
            if pressure[0]: raise RuntimeError("no new diagnostic scratch under pressure")
        journal = self.journal(ram=reserve)
        snapshot = PackedMutationSnapshot.capture((self.owner,), reserve=lambda _bytes: None)
        self.install([[0xAA], [0x55]])
        pressure[0] = True
        snapshot.restore()
        self.assertTrue(bool(self.owner.packed.eq(0x55).all()))
        pressure[0] = False
        self.assertEqual(journal.delta_norm(), 0.0)
        snapshot.close()

    def test_disk_fallback_is_admitted_stable_named_and_repeatable(self):
        reservations, disk = [], []
        def reserve(size):
            reservations.append(size)
            if len(reservations) == 1: raise RuntimeError("refuse full CPU copy")
        snapshot = PackedMutationSnapshot.capture((self.owner,), reserve=reserve, disk_directory=self.root / "rollback",
            disk_reserve=lambda size, stage: disk.append((size, stage)))
        self.assertEqual(snapshot.storage_mode, "private-disk-packed-snapshot")
        self.assertEqual(len(reservations), 2)
        self.assertTrue(disk)
        self.owner._buffers["packed"] = torch.full_like(self.owner.packed, 0xAA)
        snapshot.restore()
        self.assertTrue(bool(self.owner.packed.eq(0x55).all()))
        self.owner.packed.fill_(0)
        snapshot.restore()
        self.assertTrue(bool(self.owner.packed.eq(0x55).all()))
        private = snapshot._disk_snapshot.directory
        snapshot.close()
        self.assertFalse(private.exists())

    def test_batch_snapshot_closed_at_final_exit_not_between_retry_attempts(self):
        seen = []
        @packed_snapshot_lifetime
        def logical_batch(fail):
            snapshot = PackedMutationSnapshot.capture((self.owner,), reserve=lambda _bytes: None)
            seen.append(snapshot)
            for attempt in range(2):
                self.owner.packed.fill_(0xAA)
                if attempt == 0:
                    snapshot.restore()
                    self.assertTrue(snapshot._entries)
                    continue
                if fail: raise RuntimeError("final logical batch failure")
            self.assertTrue(snapshot._entries)
        logical_batch(False)
        with self.assertRaises(RuntimeError): logical_batch(True)
        self.assertTrue(all(not snapshot._entries for snapshot in seen))

    def test_ordinary_pre_neural_steer_still_uses_no_baseline_or_model(self):
        from types import SimpleNamespace
        owner = SimpleNamespace(brain_id="brain", completed_chat_turns=[], config=SimpleNamespace(max_seq_len=64),
            resource_policy=SimpleNamespace(),
            _validated_chat_turn_id=AdaptiveBrain._validated_chat_turn_id)
        result = AdaptiveBrain.chat(owner, "actual input", turn_id="old", steer_check=lambda: True)
        self.assertTrue(result["zeroTokenYield"])

    def test_actual_brain_diagnostic_methods_route_to_journal_with_physical_admission(self):
        from types import SimpleNamespace
        readings, disk = [], []
        def status(**kwargs):
            readings.append(kwargs)
            return {"memoryPressure": False}
        fake = SimpleNamespace(decoder=self.owner, memory_bridge=Owner(), idea_adapter=Owner(), liquid=Owner(),
            _live_paging_cache_directory=self.root,
            resource_policy=SimpleNamespace(status=status, require_disk=lambda size, stage: disk.append((size, stage))))
        fake._reserve_core_diagnostic_state = lambda size: AdaptiveBrain._reserve_core_diagnostic_state(fake, size)
        journal = AdaptiveBrain._parameter_copy(fake)
        self.journals.append(journal)
        self.install([[0x56], [0x55]])
        self.assertEqual(AdaptiveBrain._parameter_delta_norm(fake, journal), 1.0)
        self.assertTrue(journal.closed)
        self.assertTrue(readings)
        self.assertTrue(disk)
        source = inspect.getsource(AdaptiveBrain._optimize_streaming_experience_batch)
        self.assertIn("@packed_snapshot_lifetime", source)
        self.assertIn("disk_directory=self._live_paging_cache_directory", source)
        self.assertIn("disk_reserve=self.resource_policy.require_disk", source)
        self.assertLess(source.index("packed_snapshot.close()"), source.index("self._maintain_neural_state_resources()"))

    def test_checksum_preserves_old_wire_bytes_with_bounded_contiguous_and_strided_chunks(self):
        values = [torch.arange(35, dtype=torch.float32).reshape(5, 7).t(), torch.tensor([0x55, 0xAA], dtype=torch.uint8)]
        reference = hashlib.sha256()
        for value in values:
            reference.update(str(tuple(value.shape)).encode("ascii"))
            reference.update(str(value.dtype).encode("ascii"))
            reference.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        chunks, reservations = [], []
        self.assertEqual(tensor_checksum(values, chunk_bytes=16, reserve=reservations.append, on_chunk=chunks.append), reference.hexdigest())
        self.assertLessEqual(max(chunks), 16)
        self.assertTrue(reservations)


if __name__ == "__main__": unittest.main()
