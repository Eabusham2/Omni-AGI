"""Constructor-free quota/storage fixtures, including real SQLite IPC only."""

import gc
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import sqlite3
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import torch
from omni_core.shared_resource_ledger import SharedResourceLedger, SharedQuotaPause
from omni_core.offload import ResourcePolicy, ResourceReading, GIB, MIB
from omni_core.working_attention_paging import WorkingAttentionPager
from omni_core.native_core_paging import NativeCorePager


class SharedResourceLedgerFixtures(unittest.TestCase):
    def test_standalone_auto_pool_does_not_erase_a_configured_app_pool(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ledger = SharedResourceLedger(root / "quota.sqlite3", process_alive=lambda _: True)
            data = ResourceReading(16 * GIB, 14 * GIB, 128 * MIB, 100 * GIB, 90 * GIB)
            policy = ResourcePolicy(root, reading_provider=lambda: data, shared_resource_owner_id="legacy", shared_ledger=ledger)
            with patch.dict("os.environ", {}, clear=True):
                lease = policy.reserve_spill(4096, "standalone legacy Auto")
            lease.release()
            self.assertEqual(ledger.status()["largestConfiguredPoolBytes"], 70 * GIB)
            ledger.register_owner("legacy", 65536)
            other = ResourcePolicy(root, reading_provider=lambda: data, shared_resource_owner_id="legacy", shared_ledger=ledger)
            with patch.dict("os.environ", {"OMNI_SHARED_RESOURCE_LEDGER": str(ledger.path)}, clear=True):
                self.assertEqual(other.shared_ledger.status()["largestConfiguredPoolBytes"], 65536)

    def test_main_global_ceiling_and_removed_owners_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = SharedResourceLedger(Path(folder) / "quota.sqlite3", process_alive=lambda _: True)
            ledger.register_owner("a", 100); ledger.register_owner("b", 200); ledger.set_ram_ceiling(100)
            lease = ledger.reserve("a", "ram", 40, observed_ram_bytes=50, ram_budget_bytes=1000, verified=True)
            with self.assertRaises(SharedQuotaPause):
                ledger.reserve("b", "ram", 30, observed_ram_bytes=50, ram_budget_bytes=1000, verified=True)
            ledger.remove_owner("a")
            with self.assertRaises(SharedQuotaPause):
                ledger.reserve("a", "spill", 1)
            lease.release()
            self.assertEqual(ledger.status()["largestConfiguredPoolBytes"], 200)

    def test_largest_pool_not_sum_and_live_ram_pending_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = SharedResourceLedger(Path(folder) / "quota.sqlite3", process_alive=lambda _: True)
            ledger.register_owner("a", 100); ledger.register_owner("b", 200)
            self.assertEqual(ledger.status()["largestConfiguredPoolBytes"], 200)
            one = ledger.reserve("a", "ram", 40, observed_ram_bytes=40, ram_budget_bytes=100, verified=True)
            with self.assertRaises(SharedQuotaPause):
                ledger.reserve("b", "ram", 40, observed_ram_bytes=40, ram_budget_bytes=100, verified=True)
            ledger.remove_owner("a")
            self.assertEqual(ledger.status()["ramEscrowBytes"], 40)
            one.release()
            self.assertEqual(ledger.status()["ramEscrowBytes"], 0)

    def test_unsampled_allocations_remain_until_a_new_verified_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            now = [20]
            ledger = SharedResourceLedger(Path(folder) / "quota.sqlite3", now_ns=lambda: now[0], process_alive=lambda _: True)
            ledger.register_owner("a", 100)
            lease = ledger.reserve("a", "ram", 40, observed_ram_bytes=40, ram_budget_bytes=100, verified=True)
            lease.mark_allocated(); lease.release()
            self.assertEqual(ledger.status(observed_ns=10, verified=True)["ramEscrowBytes"], 40)
            self.assertEqual(ledger.status(observed_ns=30, verified=False)["ramEscrowBytes"], 40)
            self.assertEqual(ledger.status(observed_ns=30, verified=True)["ramEscrowBytes"], 0)

    def test_actual_file_identity_deduplicates_hardlinks_and_keeps_orphans_charged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); alive = {11, 22}
            ledger = SharedResourceLedger(root / "quota.sqlite3", pid=11, process_alive=lambda pid: pid in alive)
            ledger.register_owner("a", 65536)
            a, b = root / "a.raw", root / "b.raw"
            lease = ledger.reserve("a", "spill", 8192); lease.bind_path(a); a.write_bytes(b"fixture")
            lease.commit(path=a)
            used = ledger.status()["physicalSpillUsageBytes"]
            os.link(a, b)
            alias = ledger.reserve("a", "spill", 0); alias.bind_path(b); alias.commit(path=b)
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], used)
            a.unlink(); lease.release()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], used)
            b.unlink(); alias.release()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)
            orphan = root / "orphan.raw"
            pending = ledger.reserve("a", "spill", 8192); pending.bind_path(orphan); orphan.write_bytes(b"partial")
            alive.remove(11)
            self.assertGreater(ledger.status()["physicalSpillUsageBytes"], 0)
            orphan.unlink(); ledger.reconcile_owner("a")
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_unlinked_open_mapping_remains_charged_until_backing_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ledger = SharedResourceLedger(root / "quota.sqlite3", process_alive=lambda _: True)
            ledger.register_owner("a", 65536)
            path = root / "mapped.raw"
            lease = ledger.reserve("a", "spill", 8192); lease.bind_path(path); path.write_bytes(b"fixture")
            lease.commit(path=path, retain_open_backing=True)
            used = ledger.status()["physicalSpillUsageBytes"]
            path.unlink(); lease.release(); ledger.reconcile_owner("a")
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], used)
            lease.backing_closed()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_append_delta_reserves_only_growth_and_updates_one_inode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ledger = SharedResourceLedger(root / "quota.sqlite3", process_alive=lambda _: True)
            ledger.register_owner("a", 65536); path = root / "journal.raw"
            initial = ledger.reserve("a", "spill", 8192); initial.bind_path(path); path.write_bytes(bytes(4096)); initial.commit(path=path)
            delta = ledger.reserve("a", "spill", 8192); delta.bind_path(path)
            with path.open("ab") as handle:
                handle.write(bytes(4096))
            delta.commit(path=path)
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], path.stat().st_blocks * 512 if hasattr(path.stat(), "st_blocks") else path.stat().st_size)
            path.unlink(); delta.release(); initial.release()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_journal_same_path_alias_metadata_is_bounded_without_dropping_hardlinks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ledger = SharedResourceLedger(root / "quota.sqlite3", process_alive=lambda _: True)
            ledger.register_owner("a", 65536); path = root / "journal.raw"; alias = root / "alias.raw"
            first = ledger.reserve("a", "spill", 8192); first.bind_path(path); path.write_bytes(b"fixture"); first.commit(path=path)
            os.link(path, alias)
            linked = ledger.reserve("a", "spill", 0); linked.bind_path(alias); linked.commit(path=alias)
            for _ in range(100):
                final = ledger.reserve("a", "spill", 0); final.bind_path(path); final.commit(path=path)
            with sqlite3.connect(ledger.path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM leases WHERE kind='spill'").fetchone()[0], 2)
            path.unlink(); final.release()
            self.assertGreater(ledger.status()["physicalSpillUsageBytes"], 0)
            alias.unlink(); linked.release()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)

    def test_two_actual_processes_cannot_both_spend_the_same_ram_headroom(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quota.sqlite3"
            ledger = SharedResourceLedger(path, process_alive=lambda _: True)
            ledger.register_owner("a", 65536); ledger.register_owner("b", 65536)
            source = str(ENGINE / "omni_core" / "shared_resource_ledger.py")
            code = '''import importlib.util,sys
spec=importlib.util.spec_from_file_location("quota_only",sys.argv[1]); module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
ledger=module.SharedResourceLedger(sys.argv[2],process_alive=lambda _:True)
input()
try:
 lease=ledger.reserve(sys.argv[3],"ram",40,observed_ram_bytes=40,ram_budget_bytes=100,verified=True);print("accepted",flush=True)
except module.SharedQuotaPause:
 lease=None;print("paused",flush=True)
input()
if lease:lease.release()
'''
            children = [subprocess.Popen([sys.executable, "-c", code, source, str(path), owner], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for owner in ("a", "b")]
            try:
                for child in children:
                    child.stdin.write("start\n"); child.stdin.flush()
                outcomes = [child.stdout.readline().strip() for child in children]
                self.assertEqual(sorted(outcomes), ["accepted", "paused"])
                for child in children:
                    child.stdin.write("done\n"); child.stdin.flush()
                for child in children:
                    _output, error = child.communicate(timeout=10)
                    self.assertEqual(child.returncode, 0, error)
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill(); child.wait(timeout=10)
            self.assertEqual(ledger.status()["ramEscrowBytes"], 0)

    def test_real_pagers_share_quota_and_views_survive_close_without_false_credit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); ledger = SharedResourceLedger(root / "quota.sqlite3", process_alive=lambda _: True)
            data = ResourceReading(16 * GIB, 14 * GIB, 128 * MIB, 100 * GIB, 90 * GIB)
            policy = ResourcePolicy(root, reading_provider=lambda: data, shared_resource_owner_id="a",
                shared_storage_pool_bytes=128 * 1024, shared_ledger=ledger)
            activity = WorkingAttentionPager(root / "activity", resident_budget_bytes=0,
                scratch_budget_bytes=128 * 1024, device_tile_budget_bytes=MIB, resource_policy=policy)
            identifier = activity.save_page(torch.ones(8))
            page_used = ledger.status()["physicalSpillUsageBytes"]
            self.assertGreater(page_used, 0)
            core = NativeCorePager(root / "core", cpu_hot_bytes=0, resource_policy=policy)
            module = torch.nn.Module()  # Empty ownership stub, no native model.
            value = core.allocate(module, "weight", (4, 4), fill=85); module.register_buffer("weight", value); core.attach(module, loaded=True)
            total = ledger.status()["physicalSpillUsageBytes"]
            core.close()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], total)
            self.assertEqual(int(value[0, 0]), 85)
            del value; gc.collect()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], page_used)
            activity.release_page(identifier); activity.close()
            self.assertEqual(ledger.status()["physicalSpillUsageBytes"], 0)


if __name__ == "__main__":
    unittest.main()
