"""Actual tiny page files; injected two-volume capacity/identity fixtures."""
import gc
import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from omni_core.device_resources import DeviceResourceManager
from omni_core.bounded_tensor_io import BoundedTensorFile, atomic_save_tensors_bounded
from omni_core.drive_pool import DrivePoolManager
from omni_core.native_core_paging import NativeCorePager
from omni_core.router_state_paging import RouterStatePager
from omni_core.offload import GIB, NeuralStateResourcePause, ResourcePolicy, ResourceReading
from omni_core.shared_resource_ledger import SharedResourceLedger
from omni_core.working_attention_paging import WorkingAttentionPager, WorkingAttentionCancelled, SavedActivityTensor


class MultiDriveSpillTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-two-drive-")
        self.root = Path(self.temporary.name)
        self.roots = [self.root / "drive-a", self.root / "drive-b"]
        self.backings = []
        self.ledger = SharedResourceLedger(self.root / "quota.sqlite3")
        self.free = {str(root): 20 * GIB + 16 * 1024 + 1 for root in self.roots}
        with self.ledger._transaction() as db:
            for index, root in enumerate(self.roots):
                root.mkdir()
                backing = root / (".omni-spill-fixture-%d" % index); backing.mkdir()
                self.backings.append(backing)
                original, owned = root.stat(), backing.stat()
                db.execute("INSERT INTO drive_pools VALUES(?,?,?,?,?,?,?,?,?)", (str(index), str(root), str(backing),
                    str(original.st_dev), str(original.st_ino), str(owned.st_dev), str(owned.st_ino), "fixture-volume-%d" % index, 100-index))
        self.settings = {"version": 1, "allocations": [{"deviceId": "storage:" + hashlib.sha256(str(root).encode()).hexdigest()[:24],
            "enabled": True, "mode": "manual", "percent": 100, "autoIncrease": False} for root in self.roots],
            "allowStorageSpill": True, "autoIncreaseStorage": False, "storageDirectories": [str(root) for root in self.roots]}
        self.pagers = []

    def tearDown(self):
        for pager in self.pagers:
            pager.close()
        gc.collect()
        self.temporary.cleanup()

    def policy(self, owner="fixture-a", *, authorize=True, pool=32 * 1024):
        if authorize:
            with self.ledger._transaction() as db:
                for index in range(2): db.execute("INSERT OR IGNORE INTO drive_access VALUES(?,?)", (owner, str(index)))
        value = ResourcePolicy(self.root / "instance" / owner, reading_provider=lambda: ResourceReading(
            total_memory_bytes=16*GIB, available_memory_bytes=12*GIB, process_memory_bytes=1024,
            disk_total_bytes=500*GIB, disk_free_bytes=300*GIB), shared_resource_owner_id=owner,
            shared_storage_pool_bytes=pool, shared_ledger=self.ledger)
        value.device_resources = DeviceResourceManager(self.settings, policy=value, torch_module=SimpleNamespace())
        value.device_resources._drive_pool = DrivePoolManager(value, self.settings,
            disk_usage=lambda path: SimpleNamespace(total=500*GIB, free=self.free[str(path)]))
        return value

    def native(self, policy):
        pager = NativeCorePager(self.root / "local-pages", cpu_hot_bytes=0, accelerator_hot_bytes=0,
            resource_policy=policy, reserve_disk=policy.require_disk)
        self.pagers.append(pager)
        return pager

    def test_native_pages_really_span_two_drives_and_live_maps_remain_charged(self):
        policy = self.policy()
        pager = self.native(policy)
        first, second = nn.Module(), nn.Module()
        a = pager.allocate(first, "value", (12 * 1024,), fill=7); first.register_buffer("value", a); pager.attach(first, loaded=True)
        b = pager.allocate(second, "value", (12 * 1024,), fill=9); second.register_buffer("value", b); pager.attach(second, loaded=True)
        pager.flush()
        paths = list(pager._owned_files)
        self.assertEqual({path.parent.parent for path in paths}, set(self.backings))
        self.assertFalse(pager.directory.exists())
        self.assertEqual(int(a.sum()), 12 * 1024 * 7)
        self.assertEqual(int(b.sum()), 12 * 1024 * 9)
        exported = self.root / "portable-state.safetensors"
        atomic_save_tensors_bounded(exported, {"first": a, "second": b})
        portable = BoundedTensorFile(exported)
        self.assertTrue(torch.equal(portable.tensor("first"), a))
        self.assertTrue(torch.equal(portable.tensor("second"), b))
        self.assertNotIn(b".omni-spill-", exported.read_bytes())
        charged = self.ledger.status()["physicalSpillUsageBytes"]
        self.assertGreaterEqual(charged, 24 * 1024)
        pager.close()
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], charged)
        self.assertFalse(any(path.exists() for path in paths))
        del a, b
        gc.collect()
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)

    def test_activity_pages_roundtrip_on_both_selected_volumes(self):
        policy = self.policy()
        pager = WorkingAttentionPager(self.root / "local-activity", resident_budget_bytes=0,
            scratch_budget_bytes=32*1024, device_tile_budget_bytes=128*1024, resource_policy=policy)
        self.pagers.append(pager)
        values = [torch.arange(3072, dtype=torch.float32), torch.arange(3072, dtype=torch.float32) * 2]
        identifiers = [pager.save_page(value) for value in values]
        self.assertEqual({page.path.parent.parent for page in pager._pages.values()}, set(self.backings))
        for identifier, expected in zip(identifiers, values):
            self.assertTrue(torch.equal(pager.read_page(identifier), expected))
        self.assertEqual(pager.status()["logicalSpillBytes"], 24 * 1024)
        pager.close()
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)

    def test_global_pool_is_largest_instance_not_sum_and_shared_volume_is_not_doubled(self):
        one, two = self.policy("one"), self.policy("two")
        a = one.allocate_spill_file(16*1024, "fixture", ".activity")
        b = two.allocate_spill_file(16*1024, "fixture", ".activity")
        try:
            self.assertNotEqual(a.grant["pool_id"], b.grant["pool_id"])
            self.assertEqual(self.ledger.status()["largestConfiguredPoolBytes"], 32*1024)
            with self.assertRaises(NeuralStateResourcePause):
                two.allocate_spill_file(1, "fixture", ".activity")
        finally:
            a.release(); b.release()
        with self.ledger._transaction() as db:
            db.execute("UPDATE drive_pools SET physical_group='shared-fixture-volume'")
        a = one.allocate_spill_file(12*1024, "fixture", ".activity")
        try:
            with self.assertRaises(NeuralStateResourcePause):
                two.allocate_spill_file(12*1024, "fixture", ".activity")
        finally: a.release()

    def test_imported_paths_without_owner_grants_cannot_write(self):
        policy = self.policy(authorize=False)
        with self.assertRaises(NeuralStateResourcePause) as error:
            policy.allocate_spill_file(4096, "fixture", ".activity")
        self.assertTrue(error.exception.status["storageAuthorizationRequired"])
        self.assertFalse(any(list(root.iterdir()) for root in self.backings))
        self.assertEqual(self.ledger.status()["pendingSpillBytes"], 0)

    def test_allocation_failure_and_cancellation_release_only_owned_pending_files(self):
        policy = self.policy()
        pager = self.native(policy)
        sentinel = self.roots[0] / "user-file.txt"; sentinel.write_text("untouched")
        with patch("omni_core.native_core_paging.mmap.mmap", side_effect=OSError("fixture map failed")), self.assertRaises(OSError):
            pager.allocate(nn.Module(), "value", (4096,), fill=1)
        self.assertEqual(self.ledger.status()["pendingSpillBytes"], 0)
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 2: raise WorkingAttentionCancelled("fixture stop")
        with self.assertRaises(WorkingAttentionCancelled):
            policy.allocate_spill_file(4096, "fixture", ".activity", cancelled=cancel)
        self.assertEqual(self.ledger.status()["pendingSpillBytes"], 0)
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertFalse(any(list(root.iterdir()) for root in self.backings))

    def test_rate_order_actual_free_space_and_explicit_auto_increase(self):
        with self.ledger._transaction() as db: db.execute("UPDATE drive_pools SET rate=1000 WHERE pool_id='1'")
        policy = self.policy(pool=64*1024)
        selected = policy.allocate_spill_file(4096, "fixture", ".activity")
        self.assertEqual(selected.grant["pool_id"], "1")
        selected.release()
        self.free[str(self.roots[1])] = 20 * GIB
        fallback = policy.allocate_spill_file(4096, "fixture", ".activity")
        self.assertEqual(fallback.grant["pool_id"], "0"); fallback.release()
        self.free[str(self.roots[0])] = 20 * GIB + 32 * 1024
        with self.assertRaises(NeuralStateResourcePause): policy.allocate_spill_file(24*1024, "fixture", ".activity")
        self.settings["allocations"][0]["autoIncrease"] = True
        with self.assertRaises(NeuralStateResourcePause): policy.allocate_spill_file(24*1024, "fixture", ".activity")
        self.settings["autoIncreaseStorage"] = True
        grown = policy.allocate_spill_file(24*1024, "fixture", ".activity")
        self.assertEqual(grown.grant["pool_id"], "0"); grown.release()

    def test_replaced_root_and_caller_selected_file_path_fail_closed(self):
        policy = self.policy()
        lease = policy.allocate_spill_file(4096, "fixture", ".activity")
        try:
            with self.assertRaises(ValueError): lease.bind_path(self.root / "user-data")
        finally: lease.release()
        old = self.backings[0].with_name("old-backing")
        self.backings[0].rename(old)
        self.backings[0].mkdir()
        with self.assertRaises(ValueError): policy.allocate_spill_file(4096, "fixture", ".activity")

    def test_activity_staging_failure_does_not_leak_pending_quota(self):
        policy = self.policy()
        pager = WorkingAttentionPager(self.root / "local-activity", resident_budget_bytes=0,
            scratch_budget_bytes=32*1024, device_tile_budget_bytes=128*1024, resource_policy=policy)
        self.pagers.append(pager)
        with patch.object(torch.Tensor, "numpy", side_effect=MemoryError("fixture staging failure")), self.assertRaises(MemoryError):
            pager.save_page(torch.ones(1024))
        self.assertEqual(self.ledger.status()["pendingSpillBytes"], 0)
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)
        self.assertEqual(pager.status()["livePages"], 0)
        self.assertFalse(any(list(root.iterdir()) for root in self.backings))

    def test_saved_activation_restores_from_another_drive_and_outlives_pager(self):
        policy = self.policy()
        pager = WorkingAttentionPager(self.root / "local-activity", resident_budget_bytes=0,
            scratch_budget_bytes=32*1024, device_tile_budget_bytes=128*1024, resource_policy=policy)
        self.pagers.append(pager)
        expected = torch.arange(3072, dtype=torch.float32)
        saved = SavedActivityTensor(pager, expected)
        restored = saved.restore(torch.device("cpu"))
        self.assertTrue(torch.equal(restored, expected))
        self.assertEqual(saved._restore_image.path.parent.parent, self.backings[1])
        saved.close(); pager.close()
        self.assertTrue(torch.equal(restored, expected))
        self.assertGreaterEqual(self.ledger.status()["physicalSpillUsageBytes"], 12*1024)
        del restored, saved
        gc.collect()
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)

    def test_new_manager_reclaims_only_registered_dead_job_backings(self):
        policy = self.policy()
        previous = policy.allocate_spill_file(4096, "fixture", ".activity")
        descriptor = previous.open(os.O_WRONLY)
        os.write(descriptor, b"x" * 4096); os.close(descriptor)
        previous.commit(path=previous.path)
        old_path = previous.path
        sentinel = self.backings[0] / "user-file.txt"; sentinel.write_text("untouched")
        with self.ledger._transaction() as db:
            db.execute("UPDATE leases SET pid=2000000000 WHERE token=?", (previous.quota.token,))
        self.ledger.process_alive = lambda pid: pid == os.getpid()
        recovered = self.policy()
        current = recovered.allocate_spill_file(4096, "fixture", ".activity")
        try:
            self.assertFalse(old_path.exists())
            self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)
            self.assertEqual(sentinel.read_text(), "untouched")
        finally: current.release()

    def test_private_mcp_media_uses_authorized_drive_and_registered_dead_source_cleanup_only(self):
        policy = self.policy()
        previous = policy.allocate_spill_file(4096, "private MCP media", ".mcp-media")
        descriptor = previous.open(os.O_WRONLY)
        os.write(descriptor, b"private reviewed source" * 32); os.close(descriptor)
        previous.commit(path=previous.path)
        old_path = previous.path
        self.assertIn(old_path.parent.parent, self.backings)
        sentinel = self.backings[0] / "user-photo.png"; sentinel.write_bytes(b"untouched")
        with self.ledger._transaction() as db:
            db.execute("UPDATE leases SET pid=2000000000 WHERE token=?", (previous.quota.token,))
        self.ledger.process_alive = lambda pid: pid == os.getpid()
        recovered = self.policy()
        current = recovered.allocate_spill_file(4096, "private MCP media", ".mcp-media")
        try:
            self.assertFalse(old_path.exists())
            self.assertEqual(sentinel.read_bytes(), b"untouched")
            self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)
        finally:
            current.release()
        denied = self.policy("no-media-authority", authorize=False)
        with self.assertRaises(NeuralStateResourcePause):
            denied.allocate_spill_file(4096, "private MCP media", ".mcp-media")

    def test_live_backing_alias_prevents_dead_job_cleanup(self):
        policy = self.policy()
        previous = policy.allocate_spill_file(4096, "fixture", ".activity")
        descriptor = previous.open(os.O_WRONLY)
        os.write(descriptor, b"x" * 4096); os.close(descriptor)
        previous.commit(path=previous.path)
        alias = previous.path.with_name("f" * 32 + ".activity")
        os.link(previous.path, alias)
        shared = self.ledger.reserve("fixture-a", "spill", 4096)
        shared.bind_path(alias); shared.commit(path=alias)
        with self.ledger._transaction() as db:
            self.assertEqual(self.ledger._drive_status(db, "fixture-volume-0")["chargedBytes"], 4096)
        with self.ledger._transaction() as db:
            db.execute("UPDATE leases SET pid=2000000000 WHERE token=?", (previous.quota.token,))
        self.ledger.process_alive = lambda pid: pid == os.getpid()
        recovered = self.policy()
        current = recovered.allocate_spill_file(4096, "fixture", ".activity")
        try:
            self.assertTrue(previous.path.exists())
            self.assertTrue(alias.exists())
        finally:
            current.release(); previous.path.unlink(); previous.release(); alias.unlink(); shared.release()

    def test_detached_drive_is_not_mistaken_for_a_deleted_backing(self):
        policy = self.policy()
        lease = policy.allocate_spill_file(4096, "fixture", ".activity")
        descriptor = lease.open(os.O_WRONLY)
        os.write(descriptor, b"x" * 4096); os.close(descriptor)
        lease.commit(path=lease.path)
        detached = self.root / "temporarily-detached-drive"
        self.roots[0].rename(detached)
        try:
            lease.release()
            self.ledger.reconcile_owner("fixture-a")
            self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 4096)
        finally:
            detached.rename(self.roots[0])
        lease.path.unlink(); lease.release()
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)

    def test_disabled_spill_does_not_fall_back_to_an_unselected_local_drive(self):
        for roots in ([], self.settings["storageDirectories"]):
            self.settings["storageDirectories"] = roots
            self.settings["allowStorageSpill"] = False
            policy = self.policy()
            with self.assertRaisesRegex(NeuralStateResourcePause, "spill is disabled"):
                policy.allocate_spill_file(4096, "fixture", ".activity")
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)

    def test_ram_only_saved_activation_round_trip_needs_no_disk_image(self):
        self.settings["storageDirectories"] = []
        self.settings["allowStorageSpill"] = False
        policy = self.policy()
        pager = WorkingAttentionPager(self.root / "attention-ram-only", resident_budget_bytes=65536,
            scratch_budget_bytes=65536, device_tile_budget_bytes=65536, resource_policy=policy)
        self.pagers.append(pager)
        expected = torch.arange(128, dtype=torch.float32).reshape(8, 16)
        saved = SavedActivityTensor(pager, expected)
        restored = saved.restore()
        self.assertTrue(torch.equal(restored, expected))
        self.assertIsNone(saved._restore_image)
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)
        saved.close()

    def test_router_controls_use_both_selected_drives_and_retain_live_export_charge(self):
        policy = self.policy()
        pager = RouterStatePager(self.root / "local-router", hot_bytes=0, tile_bytes=65536, journal_ram_bytes=65536, resource_policy=policy)
        self.pagers.append(pager)
        owner = nn.Module()
        first = pager.allocate(owner, "first", (16384,), torch.uint8, fill=0, loading=True)
        owner.register_buffer("first", first)
        second = pager.allocate(owner, "second", (16384,), torch.uint8, fill=0, loading=True)
        owner.register_buffer("second", second)
        entries = list(pager._entries.values())
        self.assertEqual({entry.path.parent.parent for entry in entries}, set(self.backings))
        first.fill_(37); second.fill_(84)
        pager.release_chunk(first, 0, first.numel()); pager.release_chunk(second, 0, second.numel())
        pager.close()
        self.assertEqual(int(first[0]), 37); self.assertEqual(int(second[-1]), 84)
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 32768)
        del first, second, entries
        gc.collect()
        self.assertEqual(self.ledger.status()["physicalSpillUsageBytes"], 0)


if __name__ == "__main__":
    unittest.main()
