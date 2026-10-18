"""Durable diagnostic baseline fixtures using primitive/native state only."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from omni_core.parameter_diagnostics import ParameterDeltaJournal, active_diagnostic_scopes, packed_diagnostic_write
from tests.test_parameter_diagnostics import Owner
from tests.test_capability_rollback import Owner as PackedOwner
from tests.test_sparse_router_state import sparse_owner


BINDING = hashlib.sha256(b'verified paused neural checkpoint and turn').hexdigest()


def recreate(owner):
    if not isinstance(owner, Owner):
        return copy.deepcopy(owner)
    result = Owner(rows=owner.packed.shape[0], columns=owner.packed.shape[1], control=bool(owner.controls))
    result._buffers = {key: value.clone() for key, value in owner._buffers.items()}
    result.controls = {key: value.clone() for key, value in owner.controls.items()}
    result.children = {key: recreate(value) for key, value in owner.children.items()}
    return result


class ParameterJournalResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='omni-diagnostic-resume-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def journal(self, owner, *, prefix='decoder', rollback=False):
        result = ParameterDeltaJournal(((prefix, owner),), directory=self.root / 'live',
            reserve_ram=lambda _size: None, reserve_disk=lambda _size, _label: None,
            include_state=rollback)
        self.addCleanup(result.close)
        return result

    def resume(self, owner, receipt, *, prefix='decoder', binding=BINDING, ram=None, disk=None, policy=None):
        result = ParameterDeltaJournal.from_resume_state(((prefix, owner),), self.root / 'saved',
            binding_id=binding, expected_manifest_sha256=receipt['manifestSha256'], directory=self.root / 'rebound',
            reserve_ram=ram or (lambda _size: None), reserve_disk=disk or (lambda _size, _label: None), resource_policy=policy)
        self.addCleanup(result.close)
        return result

    @staticmethod
    def install(owner, code):
        replacement = torch.full_like(owner.packed, code)
        with packed_diagnostic_write(owner, owner.packed, replacement):
            owner.packed.copy_(replacement)

    def test_rebind_preserves_original_control_and_net_changed_bytes_not_current_baseline(self):
        owner = Owner(control=True)
        journal = self.journal(owner)
        self.install(owner, 0x56)
        owner.controls['gain'].fill_(3.)
        expected = journal.delta_norm()
        recreated = recreate(owner)
        original_clone = torch.Tensor.clone
        def no_packed_clone(value, *args, **kwargs):
            if value.dtype == torch.uint8: raise AssertionError('full packed snapshot is forbidden')
            return original_clone(value, *args, **kwargs)
        with patch.object(torch.Tensor, 'clone', no_packed_clone):
            receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
            self.assertFalse(journal.closed)
            journal.close()
            resumed = self.resume(recreated, receipt)
        self.assertEqual(resumed.delta_norm(), expected)
        self.install(recreated, 0x54)
        self.assertEqual(resumed.delta_norm(), expected)
        self.install(recreated, 0x55)
        recreated.controls['gain'].fill_(1.)
        self.assertEqual(resumed.delta_norm(), 0.)
        resumed.close()
        self.assertEqual(list((self.root / 'rebound').iterdir()), [])
        self.assertTrue((self.root / 'saved/manifest.json').is_file())

    def test_unchanged_empty_journal_roundtrip_does_not_create_mutable_byte_database(self):
        owner = Owner()
        journal = self.journal(owner)
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        journal.close()
        resumed = self.resume(Owner(), receipt)
        self.assertIsNone(resumed.database)
        self.assertEqual(resumed.delta_norm(), 0.)

    def test_wrong_checkpoint_or_corrupt_file_is_rejected_before_observer_registration(self):
        owner = Owner()
        journal = self.journal(owner)
        self.install(owner, 0x56)
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        active = active_diagnostic_scopes()
        with self.assertRaisesRegex(ValueError, 'binding'):
            self.resume(recreate(owner), receipt, binding='1' * 64)
        self.assertEqual(active_diagnostic_scopes(), active)
        with (self.root / 'saved/changed.sqlite').open('ab') as handle: handle.write(b'tamper')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.resume(recreate(owner), receipt)
        self.assertEqual(active_diagnostic_scopes(), active)
        self.assertFalse((self.root / 'rebound').exists())

    def test_wrong_named_shape_rejected_without_touching_loaded_state(self):
        journal = self.journal(Owner())
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        recreated = Owner(rows=3)
        before = recreated.packed.clone()
        active = active_diagnostic_scopes()
        with self.assertRaisesRegex(ValueError, 'shape'):
            self.resume(recreated, receipt)
        self.assertTrue(torch.equal(before, recreated.packed))
        self.assertEqual(active_diagnostic_scopes(), active)

    def test_export_and_import_refusal_preserve_source_and_cleanup_only_private_attempt(self):
        owner = Owner()
        journal = self.journal(owner)
        self.install(owner, 0x56)
        previous = journal.delta_norm()
        def denied(*_args): raise RuntimeError('refused admission')
        journal.reserve_disk = denied
        with self.assertRaisesRegex(RuntimeError, 'refused admission'):
            journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        self.assertFalse((self.root / 'saved').exists())
        self.assertEqual(journal.delta_norm(), previous)
        journal.reserve_disk = lambda _size, _label: None
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        active = active_diagnostic_scopes()
        with self.assertRaisesRegex(RuntimeError, 'refused admission'):
            self.resume(recreate(owner), receipt, disk=denied)
        self.assertEqual(active_diagnostic_scopes(), active)
        self.assertEqual(journal.delta_norm(), previous)
        self.assertFalse((self.root / 'rebound').exists())

    def test_parallel_growth_baseline_rebinds_and_exact_rollback_removes_new_contacts(self):
        owner = PackedOwner()
        journal = self.journal(owner, rollback=True)
        owner.parallel_connections.add(1, 2, sign=1, strength=5)
        delta = journal.delta_norm()
        recreated = recreate(owner)
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        journal.close()
        resumed = self.resume(recreated, receipt)
        self.assertEqual(resumed.delta_norm(), delta)
        resumed.restore()
        self.assertEqual(recreated.parallel_connections.storage_status()['parallelConnectionCount'], 0)
        self.assertEqual(resumed.delta_norm(), 0.)

    def test_sparse_implicit_birth_survives_real_bounded_reload_and_can_be_retired(self):
        owner = sparse_owner()
        journal = self.journal(owner, prefix='router', rollback=True)
        levels = torch.zeros((3, 5), dtype=torch.int8)
        levels[2, 4] = -1
        owner.set_effective_weights(levels)
        delta = journal.delta_norm()
        path = self.root / 'router.safetensors'
        atomic_save_tensors_bounded(path, owner.state_dict(), chunk_bytes=32)
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        journal.close()
        recreated = sparse_owner(loading=True)
        recreated.load_recurrent_state_bounded(BoundedTensorFile(path, chunk_bytes=32), '')
        resumed = self.resume(recreated, receipt, prefix='router')
        self.assertEqual(resumed.delta_norm(), delta)
        self.assertTrue(any(item.get('implicit_zero_birth') for item in resumed.original.values()))
        resumed.restore()
        self.assertEqual(len(recreated.blocks), 0)
        self.assertEqual(resumed.delta_norm(), 0.)

    def test_before_image_offsets_are_validated_even_with_a_fresh_file_seal(self):
        import sqlite3
        owner = Owner()
        journal = self.journal(owner)
        self.install(owner, 0x56)
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        file = self.root / 'saved/changed.sqlite'
        database = sqlite3.connect(str(file))
        database.execute('UPDATE original SET block=99999')
        database.commit(); database.close()
        manifest_path = self.root / 'saved/manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files']['changed.sqlite'] = {'sha256': hashlib.sha256(file.read_bytes()).hexdigest(), 'bytes': file.stat().st_size}
        encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
        manifest_path.write_bytes(encoded)
        receipt['manifestSha256'] = hashlib.sha256(encoded).hexdigest()
        active = active_diagnostic_scopes()
        with self.assertRaisesRegex(ValueError, 'offsets'):
            self.resume(recreate(owner), receipt)
        self.assertEqual(active_diagnostic_scopes(), active)
        self.assertFalse((self.root / 'rebound').exists())

    def test_closed_journal_does_not_keep_an_unloaded_owner_alive(self):
        import gc
        import weakref
        owner = Owner()
        reference = weakref.ref(owner)
        journal = self.journal(owner)
        journal.close()
        del owner
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(journal.roots, ())

    def test_exported_artifacts_keep_real_shared_pool_charges_after_journals_close(self):
        from types import SimpleNamespace
        from omni_core.shared_resource_ledger import SharedResourceLedger
        ledger = SharedResourceLedger(self.root / 'quota.sqlite')
        ledger.register_owner('diagnostic', 4 * 1024 * 1024)
        policy = SimpleNamespace(reserve_spill=lambda size, _label: ledger.reserve('diagnostic', 'spill', size))
        owner = Owner()
        journal = self.journal(owner)
        self.install(owner, 0x56)
        journal.resource_policy = policy
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        charged = ledger.status()['physicalSpillUsageBytes']
        self.assertGreater(charged, 0)
        self.assertEqual(ledger.status()['pendingSpillBytes'], 0)
        journal.close()
        resumed = self.resume(recreate(owner), receipt, policy=policy)
        self.assertEqual(resumed.delta_norm(), 2**.5)
        self.assertGreater(ledger.status()['physicalSpillUsageBytes'], charged)
        resumed.close()
        self.assertEqual(ledger.status()['physicalSpillUsageBytes'], charged)
        self.assertEqual(ledger.status()['pendingSpillBytes'], 0)

    def test_real_native_control_bytes_and_ephemeral_sort_cache_are_not_ternary_signs(self):
        from omni_core.modalities import PackedTernaryTable

        owner = PackedTernaryTable(69, 2, scale=1.)
        projection = owner.projection
        projection.fill_ternary_(1)
        projection._packed_stability_strength = .5
        projection._row_stability.fill_(3)  # Valid resistance, reserved as a sign lane.
        projection.online_learning_rate = 1000.
        projection.add_parallel_connection_(0, 64, sign=1, strength=2)
        before = {name: value.clone() for name, value in owner.state_dict().items()}
        journal = self.journal(owner, rollback=True)
        owner.learn_selected(torch.tensor([0, 64, 68]), -torch.ones(3, 2))
        delta = journal.delta_norm()
        self.assertGreater(delta, 0.)
        receipt = journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        checkpoint = self.root / 'native.safetensors'
        atomic_save_tensors_bounded(checkpoint, owner.state_dict(), chunk_bytes=32)
        journal.close()
        recreated = PackedTernaryTable(69, 2, scale=1.)
        reader = BoundedTensorFile(checkpoint, chunk_bytes=32)
        recreated.load_state_dict({key: reader.tensor(key) for key in reader.specs})
        resumed = self.resume(recreated, receipt)
        self.assertEqual(resumed.delta_norm(), delta)
        self.assertFalse(any(item['name'] == '_sort_order' for item in resumed.original.values()))
        row = next(item for item in resumed.original.values() if item['name'] == '_row_stability')
        self.assertTrue(row['structural_state'])
        resumed.restore()
        for name, value in before.items():
            self.assertTrue(torch.equal(recreated.state_dict()[name], value), name)
        self.assertEqual(resumed.delta_norm(), 0.)

    def test_failed_final_quota_commit_does_not_leave_a_published_diagnostic_copy(self):
        from types import SimpleNamespace

        owner = Owner()
        journal = self.journal(owner)
        self.install(owner, 0x56)
        before = journal.delta_norm()
        leases = []

        class Lease:
            def __init__(self): self.path, self.released = None, False
            def bind_path(self, path): self.path = path
            def commit(self, **_):
                if self.path.parent.name == 'saved':
                    raise RuntimeError('injected final quota failure')
            def release(self): self.released = True

        def reserve(*_):
            lease = Lease()
            leases.append(lease)
            return lease

        journal.resource_policy = SimpleNamespace(reserve_spill=reserve)
        with self.assertRaisesRegex(RuntimeError, 'final quota failure'):
            journal.export_resume_state(self.root / 'saved', binding_id=BINDING)
        self.assertFalse((self.root / 'saved').exists())
        self.assertFalse(any(path.name.startswith('.diagnostic-') for path in self.root.iterdir()))
        self.assertTrue(all(lease.released for lease in leases))
        self.assertEqual(journal.delta_norm(), before)
        self.assertFalse(journal.closed)


if __name__ == '__main__': unittest.main()
