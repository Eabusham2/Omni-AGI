"""Primitive packed owner rollback; no brain initialization or corpus work."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omni_core.capability_rollback import CapabilityRollbackSnapshot
from omni_core.packed_connections import PooledParallelConnections
from omni_core.persistence import tensor_checksum


class Lease:
    def __init__(self, policy, size): self.policy, self.size, self.released = policy, size, False
    def release(self):
        if not self.released: self.policy.used -= self.size; self.released = True


class Owner(torch.nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.register_buffer('packed', torch.full((64, width), 0x55, dtype=torch.uint8))
        self.register_buffer('_connection_state', torch.zeros((64, 4 * width), dtype=torch.uint8))
        self.register_buffer('_row_stability', torch.zeros(64, dtype=torch.uint8))
        self.register_buffer('updates', torch.tensor(3, dtype=torch.int64))
        self.register_buffer('control', torch.tensor([.25, .5]))
        self.register_buffer('structural_counter_bytes', torch.tensor([1, 2, 3], dtype=torch.uint8))
        self.parallel_connections = PooledParallelConnections((64, 4 * width))
        self._pending_stability_events = 5
    def authoritative_packed_tensors(self): return self.packed, self._connection_state
    def capture_packed_topology_boundary(self): return self.parallel_connections.capture_packed_topology_boundary()


class CapabilityRollbackTests(unittest.TestCase):
    def fixture(self, *, limit=10**9, spill=True, width=16, refuse_disk=False):
        temporary = tempfile.TemporaryDirectory(prefix='omni-capability-rollback-')
        self.addCleanup(temporary.cleanup)
        decoder = torch.nn.Module()
        decoder.head = Owner(width)
        decoder.head.parallel_connections.eval()
        policy = SimpleNamespace(used=0, disk_calls=[])
        def reserve(size, _label):
            if policy.used + size > limit: raise RuntimeError('RAM reserve')
            policy.used += size
            return Lease(policy, size)
        def disk(size, label):
            if refuse_disk: raise RuntimeError('disk reserve')
            policy.disk_calls.append((size, label))
        policy.reserve_ram, policy.require_disk = reserve, disk
        brain = SimpleNamespace(decoder=decoder, resource_policy=policy,
            config=SimpleNamespace(disk_state_offload=spill), counters={'steps': 1},
            _live_paging_cache_directory=Path(temporary.name))
        return brain, {'head': decoder.head}, Path(temporary.name)

    def exercise(self, brain, heads):
        head = heads['head']
        head.parallel_connections.add(0, 2, sign=-1, strength=4)
        before = tensor_checksum(head.state_dict().values())
        modes = [module.training for module in brain.decoder.modules()]
        snapshot = CapabilityRollbackSnapshot(brain, heads)
        mode = snapshot.storage_mode
        try:
            self.assertEqual(before, snapshot.before_checksum)
            head.packed.fill_(0xAA)
            head._connection_state.fill_(0xF7)
            head._row_stability.fill_(7)
            head.updates.add_(4)
            head.control.add_(2)
            head.structural_counter_bytes.fill_(8)
            head._pending_stability_events = 91
            head.parallel_connections.add(1, 3, sign=1, strength=3)
            head.eval()
            brain.counters.update(steps=99, uncommitted=2)
            snapshot.restore()
            snapshot.restore_modes()
            self.assertEqual(before, tensor_checksum(head.state_dict().values()))
            self.assertEqual(brain.counters, {'steps': 1})
            self.assertEqual(head._pending_stability_events, 5)
            self.assertEqual([module.training for module in brain.decoder.modules()], modes)
            snapshot.restore()
        finally:
            snapshot.close()
        self.assertEqual(brain.resource_policy.used, 0)
        return mode

    def test_ram_first_exact_controls_topology_counter_keys_and_mixed_modes(self):
        brain, heads, directory = self.fixture()
        self.assertEqual(self.exercise(brain, heads), 'reserved-cpu-packed-bytes')
        self.assertEqual(list(directory.iterdir()), [])

    def test_low_ram_uses_authorized_bounded_disk_snapshot_then_cleans(self):
        brain, heads, directory = self.fixture(limit=360000, width=1024)
        self.assertEqual(self.exercise(brain, heads), 'private-disk-packed-snapshot')
        self.assertTrue(brain.resource_policy.disk_calls)
        self.assertEqual(list((directory / 'capability-rollback').iterdir()), [])

    def test_no_spill_or_disk_refusal_leaves_source_and_modes_untouched(self):
        for enabled, denied in ((False, False), (True, True)):
            with self.subTest(spill=enabled, disk_refused=denied):
                brain, heads, directory = self.fixture(limit=360000, width=1024, spill=enabled, refuse_disk=denied)
                before = tensor_checksum(brain.decoder.state_dict().values())
                modes = [module.training for module in brain.decoder.modules()]
                with self.assertRaisesRegex(RuntimeError, 'reserve'):
                    CapabilityRollbackSnapshot(brain, heads)
                self.assertEqual(before, tensor_checksum(brain.decoder.state_dict().values()))
                self.assertEqual([module.training for module in brain.decoder.modules()], modes)
                self.assertEqual(brain.resource_policy.used, 0)
                self.assertEqual(list(directory.iterdir()), [])

    def test_authorized_snapshot_is_charged_to_real_shared_pool_until_its_owned_file_is_removed(self):
        from omni_core.shared_resource_ledger import SharedResourceLedger

        brain, heads, directory = self.fixture(limit=360000, width=1024)
        ledger = SharedResourceLedger(directory / 'quota.sqlite')
        ledger.register_owner('capability', 4 * 1024 * 1024)
        brain.resource_policy.reserve_spill = lambda size, _label: ledger.reserve('capability', 'spill', size)
        before = tensor_checksum(heads['head'].state_dict().values())
        snapshot = CapabilityRollbackSnapshot(brain, heads)
        try:
            self.assertEqual(snapshot.storage_mode, 'private-disk-packed-snapshot')
            self.assertGreater(ledger.status()['physicalSpillUsageBytes'], 0)
            self.assertEqual(ledger.status()['pendingSpillBytes'], 0)
            heads['head'].packed.fill_(0xaa)
            heads['head'].parallel_connections.add(40, 7, sign=1, strength=5)
            snapshot.restore()
            self.assertEqual(tensor_checksum(heads['head'].state_dict().values()), before)
        finally:
            snapshot.close()
        self.assertEqual(ledger.status()['physicalSpillUsageBytes'], 0)
        self.assertEqual(ledger.status()['pendingSpillBytes'], 0)
        self.assertEqual(brain.resource_policy.used, 0)

    def test_native_rehearsal_uses_shared_disk_fallback_and_cleans_after_measured_candidate(self):
        from tests.test_capability_rehearsal import CapabilityRehearsalTests
        from omni_core.capability_rehearsal import rehearse_capabilities
        from omni_core.optimizers import PackedMutationSnapshot
        from unittest.mock import patch
        case = CapabilityRehearsalTests()
        case.setUp()
        brain, policy = case.brain, case.policy
        with tempfile.TemporaryDirectory(prefix='omni-rehearsal-spill-fixture-') as temporary:
            brain.config.disk_state_offload = True
            brain._live_paging_cache_directory = Path(temporary)
            brain.resource_policy.require_disk = lambda _size, _label: None
            reserve = brain.resource_policy.reserve_ram
            refused = False
            def limited(size, label):
                nonlocal refused
                if label.endswith('packed rollback') and not refused:
                    refused = True
                    raise RuntimeError('simulated full-copy RAM pressure')
                return reserve(size, label)
            brain.resource_policy.reserve_ram = limited
            capture, modes = PackedMutationSnapshot.capture, []
            def observe(*args, **kwargs):
                result = capture(*args, **kwargs)
                modes.append(result.storage_mode)
                return result
            with patch.object(PackedMutationSnapshot, 'capture', side_effect=observe):
                receipt = rehearse_capabilities(brain, phase='middle', committed_global_waves=3, policy=policy)
            self.assertEqual(modes, ['private-disk-packed-snapshot'])
            self.assertEqual(receipt['action']['attemptedSteps'], 2)
            self.assertEqual(case.released, len(case.reserved))
            self.assertEqual(list((Path(temporary) / 'capability-rollback').iterdir()), [])

    def test_selected_drive_fallback_uses_its_local_grant_not_the_primary_cache(self):
        from tests.test_multi_drive_spill import MultiDriveSpillTests
        from omni_core.offload import GIB
        from unittest.mock import Mock

        fixture = MultiDriveSpillTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        fixture.free[str(fixture.roots[0])] = 20 * GIB  # Fastest volume is full at reserve.
        fixture.free[str(fixture.roots[1])] = 20 * GIB + 4 * 1024 * 1024
        policy = fixture.policy(pool=4 * 1024 * 1024)
        policy.reserve_spill = Mock(side_effect=AssertionError('primary-drive/global spill fallback'))
        decoder = torch.nn.Module()
        decoder.head = Owner(width=1024)
        rejected = False

        def ram(_size, label):
            nonlocal rejected
            if label.endswith('packed rollback') and not rejected:
                rejected = True
                raise RuntimeError('whole-copy RAM pressure')
            return None

        policy.reserve_ram = ram
        local = fixture.root / 'primary-cache'
        brain = SimpleNamespace(decoder=decoder, resource_policy=policy,
            config=SimpleNamespace(disk_state_offload=True), counters={'steps': 1},
            _live_paging_cache_directory=local)
        initial = tensor_checksum(decoder.state_dict().values())
        snapshot = CapabilityRollbackSnapshot(brain, {'head': decoder.head})
        path = snapshot.packed._disk_snapshot.path
        try:
            self.assertEqual(path.parent.parent, fixture.backings[1])
            self.assertEqual(path.suffix, '.packed-rollback')
            self.assertFalse(local.exists())
            self.assertGreater(fixture.ledger.status()['physicalSpillUsageBytes'], 0)
            self.assertEqual(fixture.ledger.status()['pendingSpillBytes'], 0)
            decoder.head.packed.fill_(0xaa)
            decoder.head.parallel_connections.add(40, 1, sign=-1, strength=5)
            snapshot.restore()
            self.assertEqual(tensor_checksum(decoder.state_dict().values()), initial)
        finally:
            snapshot.close()
        self.assertFalse(path.exists())
        self.assertEqual(fixture.ledger.status()['physicalSpillUsageBytes'], 0)
        self.assertEqual(fixture.ledger.status()['pendingSpillBytes'], 0)

        # A write can fail after its private bytes reached the selected drive.
        # Only that owned attempt is removed; no primary-cache fallback appears.
        from omni_core.bounded_tensor_io import atomic_save_tensors_bounded
        from unittest.mock import patch
        rejected = False

        def fail_after_write(*args, **kwargs):
            atomic_save_tensors_bounded(*args, **kwargs)
            raise OSError('injected selected rollback publication failure')

        with patch('omni_core.native_core_paging.atomic_save_tensors_bounded', side_effect=fail_after_write), \
                self.assertRaisesRegex(OSError, 'selected rollback publication failure'):
            CapabilityRollbackSnapshot(brain, {'head': decoder.head})
        self.assertEqual(tensor_checksum(decoder.state_dict().values()), initial)
        self.assertFalse(local.exists())
        self.assertFalse(any(list(root.iterdir()) for root in fixture.backings))
        self.assertEqual(fixture.ledger.status()['physicalSpillUsageBytes'], 0)
        self.assertEqual(fixture.ledger.status()['pendingSpillBytes'], 0)

    def test_selected_but_unauthorized_drive_does_not_fall_back_to_an_unselected_path(self):
        from tests.test_multi_drive_spill import MultiDriveSpillTests
        from omni_core.offload import NeuralStateResourcePause

        fixture = MultiDriveSpillTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        policy = fixture.policy(authorize=False, pool=4 * 1024 * 1024)
        decoder = torch.nn.Module()
        decoder.head = Owner()
        rejected = False

        def ram(_size, label):
            nonlocal rejected
            if label.endswith('packed rollback') and not rejected:
                rejected = True
                raise RuntimeError('whole-copy RAM pressure')

        policy.reserve_ram = ram
        local = fixture.root / 'primary-cache'
        brain = SimpleNamespace(decoder=decoder, resource_policy=policy,
            config=SimpleNamespace(disk_state_offload=True), counters={},
            _live_paging_cache_directory=local)
        initial = tensor_checksum(decoder.state_dict().values())
        with self.assertRaises(NeuralStateResourcePause) as failure:
            CapabilityRollbackSnapshot(brain, {'head': decoder.head})
        self.assertTrue(failure.exception.status['storageAuthorizationRequired'])
        self.assertEqual(tensor_checksum(decoder.state_dict().values()), initial)
        self.assertFalse(local.exists())
        self.assertEqual(fixture.ledger.status()['pendingSpillBytes'], 0)


if __name__ == '__main__': unittest.main()
