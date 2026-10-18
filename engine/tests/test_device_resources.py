"""Device budget and allocator fixtures; no model, GPU allocation or training."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from omni_core.device_resources import (
    DeviceResourceAdmissionError, DeviceResourceManager,
    device_resource_budgets, validate_device_resource_settings,
)
from omni_core.offload import GIB, MIB, ResourcePolicy, ResourceReading


def settings(*allocations, **changes):
    return {"version": 1, "allocations": list(allocations), "allowStorageSpill": True,
            "autoIncreaseStorage": False, "storageDirectories": [], **changes}


def allocation(identifier, *, percent=50, enabled=True, auto=False):
    return {"deviceId": identifier, "enabled": enabled, "mode": "manual",
            "percent": percent, "autoIncrease": auto}


def topology():
    common = {"availableBytes": 7 * GIB, "allocatedBytes": GIB, "reservedBytes": GIB,
              "totalBytes": 8 * GIB, "nativeComputeSupported": True, "unifiedMemory": False}
    return {"measuredAt": "2026-10-06T00:00:00Z", "devices": [
        {**common, "id": "ram:system", "memoryGroupId": "ram:system", "kind": "ram", "backend": "cpu", "executionDevice": "cpu"},
        {**common, "id": "cuda:fixture", "memoryGroupId": "cuda:fixture", "kind": "gpu", "backend": "cuda", "executionDevice": "cuda:0"},
        {**common, "id": "mps:0", "memoryGroupId": "ram:system", "kind": "gpu", "backend": "mps", "executionDevice": "mps",
         "unifiedMemory": True, "recommendedLimitBytes": 6 * GIB},
        {**common, "id": "storage:fixture", "memoryGroupId": "volume:fixture", "kind": "storage", "backend": "filesystem",
         "totalBytes": 500 * GIB, "availableBytes": 100 * GIB, "readOnly": False},
    ]}


def host(**changes):
    return {"systemRamBudgetBytes": 4 * GIB, "diskReserveBytes": 20 * GIB,
            "admissionResidentMemoryBytes": GIB, "availableSafeRamBytes": 6 * GIB,
            "ramAdmissionVerified": True, **changes}


def fake_torch():
    return SimpleNamespace(
        cuda=SimpleNamespace(current_device=Mock(return_value=0), mem_get_info=Mock(return_value=(7 * GIB, 8 * GIB)),
            memory_allocated=Mock(return_value=GIB), memory_reserved=Mock(return_value=GIB), set_per_process_memory_fraction=Mock()),
        mps=SimpleNamespace(recommended_max_memory=Mock(return_value=6 * GIB), driver_allocated_memory=Mock(return_value=GIB),
            current_allocated_memory=Mock(return_value=GIB), set_per_process_memory_fraction=Mock()),
    )


class DeviceResourceTests(unittest.TestCase):
    def manager(self, selected=None, observed=None):
        device_torch = fake_torch()
        manager = DeviceResourceManager(selected or settings(), policy=None, torch_module=device_torch,
            topology_provider=lambda: copy.deepcopy(observed or topology()))
        return manager, device_torch

    def test_settings_are_exact_and_reject_unsafe_values(self):
        valid = settings(allocation("ram:system", percent=30))
        result = validate_device_resource_settings(valid)
        result["allocations"][0]["percent"] = 45
        self.assertEqual(valid["allocations"][0]["percent"], 30)
        invalid = [settings(allocation("ram:system", percent=29)), settings(allocation("cuda:fixture", percent=True)),
                   settings(allocation("cuda:fixture", percent=float("nan"))), settings(version=True),
                   settings(allocation("cuda:fixture"), allocation("cuda:fixture")), settings(unknown=True),
                   settings(storageDirectories=["/one", "/one"])]
        wrong_mode = settings(allocation("cuda:fixture")); wrong_mode["allocations"][0]["mode"] = []
        invalid.append(wrong_mode)
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_device_resource_settings(value)

    def test_worker_config_roundtrip_preserves_device_choices_without_changing_geometry(self):
        from omni_core.config import OmniConfig
        raw = settings(allocation("ram:system", percent=47), allocation("mps:0", percent=60))
        config = OmniConfig.from_external({"deviceResources": raw, "hardwareTier": "micro"})
        self.assertEqual(config.device_resource_settings, raw)
        self.assertEqual(config.system_ram_share_percent, 47)
        saved = config.to_dict()
        loaded = OmniConfig(**saved)
        loaded.validate()
        self.assertEqual(loaded.device_resource_settings, raw)
        self.assertEqual(loaded.d_model, config.d_model)
        automatic = copy.deepcopy(raw); automatic["allocations"][0]["mode"] = "auto"
        self.assertEqual(OmniConfig.from_external({"deviceResources": automatic}).system_ram_share_percent, 0)

    def test_safe_host_ceiling_and_shared_apple_memory_are_not_added_twice(self):
        selected = settings(allocation("ram:system", percent=100), allocation("mps:0", percent=100),
                            allocation("cuda:fixture", enabled=False), allocation("storage:fixture", enabled=False))
        manager, _ = self.manager(selected)
        status = manager.status(host())
        self.assertEqual(status["physicalGroupCeilings"], {"ram:system": 4 * GIB})
        self.assertFalse(status["multiGpuComputeScheduling"])
        oversized = device_resource_budgets(topology(), selected, system_ram_budget_bytes=100 * GIB, disk_reserve_bytes=20 * GIB)
        self.assertEqual(oversized[0]["ceilingBytes"], 8 * GIB)

    def test_fresh_cuda_free_memory_limits_allocation_and_failure_does_not_reuse_cache(self):
        manager, device_torch = self.manager(settings(allocation("cuda:fixture", percent=50)))
        baseline = manager.topology()
        device_torch.cuda.mem_get_info.return_value = (GIB, 8 * GIB)
        status = manager.device_budget("cuda:0", host(), fresh=True)
        self.assertEqual(status["remainingBytes"], GIB - (8 * GIB // 10))
        with self.assertRaises(DeviceResourceAdmissionError):
            manager.require_device("cuda:0", GIB, "fixture", host())
        self.assertEqual(baseline["devices"][1]["availableBytes"], 7 * GIB)
        device_torch.cuda.mem_get_info.side_effect = RuntimeError("unavailable")
        with self.assertRaises(DeviceResourceAdmissionError) as error:
            manager.require_device("cuda:0", 1, "fixture", host())
        self.assertFalse(error.exception.status["capacityMeasured"])
        device_torch.cuda.set_per_process_memory_fraction.assert_not_called()

    def test_unified_device_respects_other_managed_family_residency(self):
        manager, _ = self.manager(settings(allocation("mps:0", percent=100)))
        status = manager.device_budget("mps", host(admissionResidentMemoryBytes=7 * GIB // 2), fresh=True)
        self.assertEqual(status["remainingBytes"], GIB // 2)
        with self.assertRaises(DeviceResourceAdmissionError):
            manager.require_device("mps", GIB, "fixture", host(admissionResidentMemoryBytes=7 * GIB // 2))
        with self.assertRaises(DeviceResourceAdmissionError) as error:
            manager.require_device("mps", 1, "fixture", host(ramAdmissionVerified=False))
        self.assertFalse(error.exception.status["capacityMeasured"])
        self.assertEqual(manager.device_budget("cpu", host(admissionResidentMemoryBytes=3 * GIB))["allocatedBytes"], 3 * GIB)

    def test_fresh_device_capacity_change_recalculates_percentage_not_just_free_bytes(self):
        manager, device_torch = self.manager(settings(allocation("cuda:fixture", percent=50)))
        manager.topology()
        device_torch.cuda.mem_get_info.return_value = (3 * GIB, 4 * GIB)
        result = manager.device_budget("cuda:0", host(), fresh=True)
        self.assertEqual(result["capacityBytes"], 4 * GIB)
        self.assertEqual(result["ceilingBytes"], (4 * GIB - 4 * GIB // 10) // 2)

    def test_disabled_device_never_installs_zero_fraction_and_process_limit_is_reasserted(self):
        manager, device_torch = self.manager(settings(allocation("mps:0", enabled=False)))
        with self.assertRaises(DeviceResourceAdmissionError):
            manager.require_device("mps", 1, "fixture", host())
        device_torch.mps.set_per_process_memory_fraction.assert_not_called()
        manager, device_torch = self.manager(settings(allocation("cuda:fixture", percent=50)))
        for _ in range(2):
            manager.require_device("cuda:0", MIB, "fixture", host())
        self.assertEqual(device_torch.cuda.set_per_process_memory_fraction.call_count, 2)
        self.assertAlmostEqual(device_torch.cuda.set_per_process_memory_fraction.call_args.args[0], 0.45, places=8)
        for bad in (-1, True, 1.25):
            with self.assertRaises(ValueError):
                manager.require_device("cuda:0", bad, "fixture", host())
        device_torch.cuda.set_per_process_memory_fraction.side_effect = RuntimeError("fixture allocator unavailable")
        with self.assertRaises(DeviceResourceAdmissionError) as error:
            manager.require_device("cuda:0", MIB, "fixture", host())
        self.assertFalse(error.exception.status["allocatorLimitApplied"])

    def test_storage_choice_does_not_silently_expand_or_shrink_its_quota(self):
        current = topology()
        manager = DeviceResourceManager(settings(allocation("storage:fixture", auto=True)), policy=None,
            torch_module=fake_torch(), topology_provider=lambda: copy.deepcopy(current))
        original = manager.budgets(host())[-1]["ceilingBytes"]
        current["devices"][-1]["availableBytes"] = 200 * GIB
        self.assertEqual(manager.budgets(host(), fresh=True)[-1]["ceilingBytes"], original)
        manager.settings["autoIncreaseStorage"] = True
        increased = manager.budgets(host(), fresh=True)[-1]["ceilingBytes"]
        self.assertGreater(increased, original)
        current["devices"][-1]["availableBytes"] = 30 * GIB
        self.assertEqual(manager.budgets(host(), fresh=True)[-1]["ceilingBytes"], increased)
        self.assertEqual(manager.budgets(host())[-1]["currentlyAvailableBytes"], 30 * GIB)

    def test_training_planner_consumes_selected_live_gpu_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            reading = ResourceReading(total_memory_bytes=16 * GIB, available_memory_bytes=12 * GIB,
                process_memory_bytes=GIB, disk_total_bytes=500 * GIB, disk_free_bytes=100 * GIB,
                accelerator_total_memory_bytes=8 * GIB, accelerator_free_memory_bytes=7 * GIB,
                accelerator_allocated_memory_bytes=GIB)
            policy = ResourcePolicy(Path(directory), reading_provider=lambda: reading, execution_device="cuda:0")
            manager, _ = self.manager(settings(allocation("cuda:fixture", percent=20)))
            policy.device_resources = manager
            plan = policy.training_plan(max_window_tokens=4096, requested_batch_size=4,
                requested_gradient_accumulation=1, trainable_parameter_bytes=0, activation_bytes_per_token=MIB)
            chosen = plan["memory"]["selectedDeviceBudget"]
            self.assertLess(plan["memory"]["acceleratorBudgetBytes"], GIB)
            self.assertEqual(plan["memory"]["acceleratorBudgetBytes"], chosen["remainingBytes"])
            self.assertTrue(plan["allSourceBytesVisited"])
            self.assertLess(plan["physicalBatchRecords"], 4)
            manager.settings["allocations"][0]["enabled"] = False
            blocked = policy.training_plan(max_window_tokens=4096, requested_batch_size=4,
                requested_gradient_accumulation=1, trainable_parameter_bytes=0, activation_bytes_per_token=MIB)
            self.assertTrue(blocked["pauseBeforeStep"])
            self.assertEqual(blocked["admittedWindowTokens"], 0)


if __name__ == "__main__":
    unittest.main()
