"""Injected process tables/resource/control fixtures; never an OS/device probe."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.managed_process_memory import (
    ManagedMemorySample, ManagedProcessMemorySampler, ProcessRow, ProcessTable,
    managed_family_sample, parse_posix_process_table,
)
from omni_core.offload import GIB, MIB, ResourcePolicy, ResourceReading
from omni_core.native_core_paging import shared_native_residency_budget
from omni_core.working_attention_paging import WorkingAttentionPager


def reading(**overrides):
    fields = dict(total_memory_bytes=16 * GIB, available_memory_bytes=13 * GIB,
        process_memory_bytes=256 * MIB, disk_total_bytes=500 * GIB, disk_free_bytes=300 * GIB)
    fields.update(overrides)
    return ResourceReading(**fields)


class ManagedRamCeilingFixtures(unittest.TestCase):
    def test_trusted_app_root_sums_only_main_renderer_gpu_and_owned_children(self):
        table = parse_posix_process_table(b"1 0 100\n10 1 1000\n11 10 2000\n12 10 3000\n13 11 4000\n20 1 99999\n21 20 99999\n")
        result = managed_family_sample(table, 10, 11, "managed-app-process-family")
        self.assertEqual(result.rss_bytes, 10_000 * 1024)
        self.assertEqual(result.worker_rss_bytes, 2000 * 1024)
        self.assertEqual(result.process_count, 4)
        with self.assertRaisesRegex(ValueError, "ancestor"):
            managed_family_sample(table, 20, 11, "managed-app-process-family")

    def test_standalone_default_never_uses_the_terminal_shell_as_owner(self):
        table = parse_posix_process_table(b"1 0 1\n5 1 99999\n10 5 1000\n11 10 2000\n12 5 99999\n")
        with patch.dict("os.environ", {}, clear=True):
            sampler = ManagedProcessMemorySampler(worker_pid=10, provider=lambda: table)
            self.assertEqual(sampler.root_pid, 10)
            self.assertEqual(sampler.sample().rss_bytes, 3000 * 1024)

    def test_windows_style_native_reader_queries_only_owned_ids(self):
        rss = Mock(side_effect=lambda pid: {10: 100, 11: 200}[pid])
        table = ProcessTable({pid: ProcessRow(pid, parent, None) for pid, parent in ((1, 0), (10, 1), (11, 10), (20, 1))}, rss)
        result = managed_family_sample(table, 10, 11, "managed-app-process-family")
        self.assertEqual(result.rss_bytes, 300)
        self.assertEqual({call.args[0] for call in rss.call_args_list}, {10, 11})

    def test_cache_and_measurement_failures_are_explicit_not_zero_residency(self):
        clock = [0.0]
        provider = Mock(return_value=parse_posix_process_table(b"10 1 100\n11 10 200\n"))
        sampler = ManagedProcessMemorySampler(worker_pid=11, owner_pid="10", provider=provider, now=lambda: clock[0])
        self.assertFalse(sampler.sample().cached)
        clock[0] = 0.5
        cached = sampler.sample()
        self.assertTrue(cached.cached)
        self.assertEqual(cached.sample_age_seconds, 0.5)
        provider.assert_called_once()
        clock[0] = 1.1; provider.side_effect = OSError("fixture inaccessible")
        failure = sampler.sample()
        self.assertFalse(failure.verified)
        self.assertIsNone(failure.rss_bytes)
        for invalid in ("", "0", "-1", "x", "9" * 5000):
            self.assertFalse(ManagedProcessMemorySampler(worker_pid=11, owner_pid=invalid, provider=provider).sample().verified)

    def test_manual_and_auto_caps_do_not_raise_for_a_larger_allocation(self):
        for share in (0, 50):
            policy = ResourcePolicy(Path("/fixture"), system_ram_share_percent=share, reading_provider=lambda: reading())
            before = policy.status()
            projected = policy.status(estimated_ram_bytes=10 * GIB)
            self.assertEqual(projected["systemRamBudgetBytes"], before["systemRamBudgetBytes"])
            self.assertEqual(projected["systemRamSharePercent"], before["systemRamSharePercent"])
            self.assertTrue(projected["memoryPressure"])
            self.assertFalse(projected["hardRssIsolation"])
            self.assertFalse(projected["osPhysicalPagePinning"])

    def test_family_not_worker_alone_governs_training_and_admission(self):
        data = reading(managed_process_memory_bytes=7 * GIB, managed_worker_rss_bytes=256 * MIB,
            managed_memory_verified=True, managed_memory_scope="managed-app-process-family", managed_memory_process_count=4)
        policy = ResourcePolicy(Path("/fixture"), system_ram_share_percent=50, reading_provider=lambda: data)
        projected = policy.status(estimated_ram_bytes=GIB)
        self.assertTrue(projected["memoryPressure"])
        self.assertEqual(projected["admissionResidentMemoryBytes"], 7 * GIB)
        self.assertEqual(projected["processMemoryBytes"], 256 * MIB)
        self.assertEqual(projected["projectedProcessMemoryBytes"], 8 * GIB)
        self.assertIn("double-counted", projected["memoryAccountingBasis"])
        plan = policy.training_plan(max_window_tokens=4096, requested_batch_size=1,
            requested_gradient_accumulation=1, trainable_parameter_bytes=0, activation_bytes_per_token=MIB)
        self.assertEqual(plan["memory"]["ramHeadroomBytes"], 512 * MIB)

    def test_worker_footprint_floor_and_unknown_residency_are_conservative(self):
        data = reading(process_memory_bytes=2 * GIB, managed_process_memory_bytes=3 * GIB,
            managed_worker_rss_bytes=GIB, managed_memory_verified=True)
        policy = ResourcePolicy(Path("/fixture"), reading_provider=lambda: data)
        self.assertEqual(policy.status()["admissionResidentMemoryBytes"], 4 * GIB)
        for data in (reading(process_memory_bytes=None, process_peak_memory_bytes=2 * GIB),
                     reading(process_memory_bytes=None), reading(managed_memory_verified=False)):
            policy = ResourcePolicy(Path("/fixture"), reading_provider=lambda: data)
            status = policy.status(estimated_ram_bytes=MIB)
            self.assertTrue(status["memoryPressure"])
            self.assertFalse(status["ramAdmissionVerified"])
            self.assertNotEqual(status["projectedProcessMemoryBytes"], MIB)
            plan = policy.training_plan(max_window_tokens=4096, requested_batch_size=1,
                requested_gradient_accumulation=1, trainable_parameter_bytes=0, activation_bytes_per_token=MIB)
            self.assertTrue(plan["pauseBeforeStep"])
            self.assertEqual(plan["admittedWindowTokens"], 0)

    def test_policy_production_readings_consumes_real_sampler_result_without_any_probe(self):
        policy = ResourcePolicy(Path("/fixture"), include_accelerator_memory=False)
        sample = ManagedMemorySample(900 * MIB, 100 * MIB, 10, 4, True, "managed-app-process-family", sample_duration_ms=2.5)
        fake_sampler = Mock(); fake_sampler.sample.return_value = sample
        with patch("omni_core.offload.shutil.disk_usage", return_value=type("Disk", (), {"total": 500 * GIB, "free": 300 * GIB})()), \
             patch("omni_core.offload._mac_memory", return_value=(16 * GIB, 13 * GIB)), \
             patch("omni_core.offload._linux_memory", return_value=(16 * GIB, 13 * GIB)), \
             patch("omni_core.offload._windows_memory", return_value=(16 * GIB, 13 * GIB)), \
             patch("omni_core.offload._fallback_memory", return_value=(16 * GIB, 13 * GIB)), \
             patch("omni_core.offload._process_memory", return_value=(100 * MIB, 200 * MIB)), \
             patch("omni_core.offload.default_managed_process_sampler", return_value=fake_sampler):
            status = policy.status()
        self.assertEqual(status["admissionResidentMemoryBytes"], 900 * MIB)
        self.assertEqual(status["managedMemorySampleDurationMs"], 2.5)

    def test_existing_cached_activity_and_core_headroom_do_not_erase_family_pressure(self):
        status = {"availableMemoryBytes": 12 * GIB, "ramReserveBytes": GIB,
            "systemRamBudgetBytes": 4 * GIB, "processMemoryBytes": 128 * MIB,
            "admissionResidentMemoryBytes": 5 * GIB, "ramAdmissionVerified": True}
        pager = WorkingAttentionPager.__new__(WorkingAttentionPager)
        pager.resource_policy = Mock(); pager._policy_snapshot = status
        pager._policy_at = 0.0
        with patch("omni_core.working_attention_paging.time.monotonic", return_value=0.1):
            self.assertTrue(pager._resource_status(MIB)["memoryPressure"])
        budget = shared_native_residency_budget(status, baseline_bytes=128 * MIB)
        self.assertEqual(budget["liveCoreHotBytes"], 0)
        self.assertEqual(budget["measuredBaselineBytes"], 4 * GIB)
        status.update(admissionResidentMemoryBytes=128 * MIB, ramAdmissionVerified=False)
        with patch("omni_core.working_attention_paging.time.monotonic", return_value=0.1):
            self.assertTrue(pager._resource_status(0)["memoryPressure"])

    def test_family_baseline_is_not_repromised_to_each_ram_subpool(self):
        snapshot = {"systemRamBudgetBytes": 8 * GIB, "availableSafeRamBytes": 12 * GIB,
            "processMemoryBytes": GIB, "admissionResidentMemoryBytes": 3 * GIB}
        budget = shared_native_residency_budget(snapshot, baseline_bytes=GIB)
        self.assertEqual(budget["measuredBaselineBytes"], 3 * GIB)
        self.assertEqual(budget["otherManagedProcessResidentBytes"], 2 * GIB)
        self.assertEqual(sum(budget[key] for key in ("corePartitionBytes", "workingActivityPartitionBytes", "trainingTransferPartitionBytes")), 5 * GIB)


if __name__ == "__main__":
    unittest.main()
