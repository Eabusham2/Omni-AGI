"""Hardware inventory and bounded-probe controls with tiny numeric fixtures."""

import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from omni_core.device_topology import collect_device_topology
from omni_core.offload import GIB, MIB


def fake_torch(*, mps=True, cuda=True):
    tensor = Mock()
    tensor.copy_.return_value = tensor
    tensor.fill_.return_value = tensor
    return SimpleNamespace(uint8="uint8", empty=Mock(return_value=tensor),
        cuda=SimpleNamespace(is_available=Mock(return_value=cuda), device_count=Mock(return_value=2),
            get_device_properties=Mock(side_effect=lambda index: SimpleNamespace(uuid=f"fixture-{index}", name=f"GPU {index}", total_memory=8 * GIB)),
            mem_get_info=Mock(return_value=(7 * GIB, 8 * GIB)), memory_allocated=Mock(return_value=0), memory_reserved=Mock(return_value=0),
            can_device_access_peer=Mock(return_value=True), synchronize=Mock()),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=Mock(return_value=mps))),
        mps=SimpleNamespace(recommended_max_memory=Mock(return_value=12 * GIB), driver_allocated_memory=Mock(return_value=GIB),
            current_allocated_memory=Mock(return_value=GIB), synchronize=Mock()))


def reading():
    return SimpleNamespace(total_memory_bytes=16 * GIB, available_memory_bytes=10 * GIB, process_memory_bytes=MIB)


class DeviceTopologyTests(unittest.TestCase):
    def collect(self, module=None, **kwargs):
        return collect_device_topology(torch_module=module or fake_torch(), host_reading=reading(),
            native_support={"cpu": True, "mps": True, "cuda": False}, **kwargs)

    def test_inventory_never_constructs_model_or_allocates_probe_without_request(self):
        module = fake_torch()
        with patch("omni_core.device_topology.platform.system", return_value="Darwin"), \
                patch("omni_core.device_topology.platform.machine", return_value="arm64"):
            value = self.collect(module)
        module.empty.assert_not_called()
        self.assertFalse(value["neuralModelConstructed"])
        self.assertFalse(value["measurementRequested"])
        devices = {entry["id"]: entry for entry in value["devices"]}
        self.assertEqual(devices["ram:system"]["memoryGroupId"], devices["mps:0"]["memoryGroupId"])
        self.assertEqual(devices["mps:0"]["availableBytes"], 10 * GIB)
        self.assertEqual(devices["cuda:fixture-0"]["executionDevice"], "cuda:0")
        self.assertFalse(devices["cuda:fixture-0"]["nativeComputeSupported"])
        self.assertEqual(len(value["links"]), 12)
        peers = [link for link in value["links"] if link["from"].startswith("cuda:") and link["to"].startswith("cuda:")]
        self.assertTrue(all(link["peerAccess"] for link in peers))

    def test_unknown_accelerators_are_reported_without_claiming_native_support(self):
        module = fake_torch(mps=False, cuda=False)
        module.npu = SimpleNamespace(is_available=Mock(return_value=True), device_count=Mock(return_value=1),
            get_device_name=Mock(return_value="Fixture NPU"), memory_allocated=Mock(return_value=0), memory_reserved=Mock(return_value=0))
        value = self.collect(module)
        npu = value["devices"][1]
        self.assertEqual(npu["kind"], "npu")
        self.assertIsNone(npu["totalBytes"])
        self.assertFalse(npu["nativeComputeSupported"])
        self.assertIn("unverified", npu["unavailableReason"])

    def test_probe_payload_follows_budget_and_every_gpu_copy_rechecks_free_memory(self):
        module = fake_torch(mps=False)
        ticks = itertools.count(step=1000)
        value = self.collect(module, measure=True, probe_budget_bytes=32 * 1024, clock_ns=lambda: next(ticks))
        self.assertTrue(value["measurementRequested"])
        self.assertTrue(module.empty.called)
        self.assertTrue(all(call.args == (4096,) for call in module.empty.call_args_list))
        self.assertGreater(module.cuda.mem_get_info.call_count, 2)
        self.assertEqual(value["devices"][0]["measurement"]["sampleBytes"], 4096)
        self.assertEqual(value["devices"][0]["measurement"]["runs"], 3)
        # Two enumeration calls succeed; subsequent fresh allocation readings
        # report no free GPU memory. There must be no GPU tensor allocation.
        module = fake_torch(mps=False)
        module.cuda.mem_get_info.side_effect = [(7 * GIB, 8 * GIB), (7 * GIB, 8 * GIB)] + [(0, 8 * GIB)] * 20
        value = self.collect(module, measure=True, probe_budget_bytes=32 * 1024, clock_ns=lambda: next(ticks))
        self.assertTrue(all(call.kwargs["device"] == "cpu" for call in module.empty.call_args_list))
        self.assertGreater(len(value["warnings"]), 0)

    def test_too_small_probe_budget_and_cancellation_do_not_allocate(self):
        module = fake_torch()
        value = self.collect(module, measure=True, probe_budget_bytes=1)
        module.empty.assert_not_called()
        self.assertIn("32 KiB", value["warnings"][0])
        with self.assertRaises(InterruptedError):
            self.collect(module, measure=True, cancelled=lambda: True)
        module.empty.assert_not_called()

    def test_selected_directory_is_deduplicated_and_probe_file_is_removed(self):
        module = fake_torch(mps=False, cuda=False)
        ticks = itertools.count(step=1000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("omni_core.device_topology.shutil.disk_usage", return_value=SimpleNamespace(total=500 * GIB, free=100 * GIB)):
                value = self.collect(module, measure=True, probe_budget_bytes=32 * 1024,
                    storage_paths=[directory, directory], clock_ns=lambda: next(ticks))
            storage = [entry for entry in value["devices"] if entry["kind"] == "storage"]
            self.assertEqual(len(storage), 1)
            self.assertEqual(storage[0]["measurement"]["scope"], "durable-file-write")
            self.assertEqual(list(root.iterdir()), [])
            # Real cancellation at the first write boundary leaves no probe.
            def cancel_after_open():
                return bool(list(root.iterdir()))
            with patch("omni_core.device_topology.shutil.disk_usage", return_value=SimpleNamespace(total=500 * GIB, free=100 * GIB)), \
                    self.assertRaises(InterruptedError):
                self.collect(module, measure=True, probe_budget_bytes=32 * 1024, storage_paths=[directory],
                    cancelled=cancel_after_open, clock_ns=lambda: next(ticks))
            self.assertEqual(list(root.iterdir()), [])

    def test_drive_probe_preserves_disk_reserve(self):
        module = fake_torch(mps=False, cuda=False)
        with tempfile.TemporaryDirectory() as directory:
            with patch("omni_core.device_topology.shutil.disk_usage", return_value=SimpleNamespace(total=500 * GIB, free=20 * GIB)):
                value = self.collect(module, measure=True, probe_budget_bytes=32 * 1024, storage_paths=[directory])
            storage = next(entry for entry in value["devices"] if entry["kind"] == "storage")
            self.assertNotIn("measurement", storage)
            self.assertIn("disk reserve", " ".join(value["warnings"]))
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
