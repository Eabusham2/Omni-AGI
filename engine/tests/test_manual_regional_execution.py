"""New inert regional routing/pager cases. No Torch/module/brain construction."""
import ast
import copy
import importlib.util
import threading
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("owned_manual_region_fixture", ROOT / "omni_core/regional_execution.py")
REGIONAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REGIONAL)


def topology():
    devices = [{"id": "ram:system", "kind": "ram", "backend": "cpu", "executionDevice": "cpu", "memoryGroupId": "ram:system",
                "totalBytes": 1000, "availableBytes": 800, "nativeComputeSupported": True, "unifiedMemory": False},
               {"id": "mps:0", "kind": "gpu", "backend": "mps", "executionDevice": "mps", "memoryGroupId": "ram:system",
                "totalBytes": 1000, "availableBytes": 700, "nativeComputeSupported": True, "unifiedMemory": True}]
    devices += [{"id": "cuda:" + str(index), "kind": "gpu", "backend": "cuda", "executionDevice": "cuda:" + str(index),
                 "memoryGroupId": "physical:" + str(index), "totalBytes": 1000, "availableBytes": 800,
                 "nativeComputeSupported": False, "unifiedMemory": False, "unavailableReason": "no-native-packed-CUDA"} for index in range(2)]
    devices += [{"id": "npu:0", "kind": "npu", "backend": "npu", "executionDevice": "npu:0", "memoryGroupId": "npu:0",
                 "totalBytes": None, "availableBytes": None, "nativeComputeSupported": False, "unifiedMemory": None}]
    links = [{"from": first["id"], "to": second["id"], "peerAccess": None,
              "measurement": {"bytesPerSecond": 1000}} for first in devices for second in devices if first is not second]
    return {"format": "omni-device-resource-topology", "version": 1, "devices": devices, "links": links,
            "measuredAt": "2026-10-15T00:00:00Z", "measurementRequested": False, "neuralModelConstructed": False,
            "platform": "fixture", "architecture": "fixture", "warnings": []}


class Activity:
    def __init__(self, values=(1, 2), device="cpu"):
        self.values, self.device = tuple(values), device
    @property
    def bytes(self):
        return len(self.values) * 4


class InertBlock:
    def __init__(self):
        self.calls = []
        self.training = False
        self.width = 2
        self.learned = b"exact-packed-signs-state-and-repeat-provenance"
    @REGIONAL.regional_block_call
    def forward(self, hidden):
        self.calls.append(hidden.device)
        return Activity((value + 1 for value in hidden.values), hidden.device)
    @REGIONAL.regional_block_call
    def forward_cached(self, hidden, cache):
        self.calls.append(hidden.device)
        return Activity(hidden.values, hidden.device), cache


class RoutingFixture:
    def __init__(self, routes=None):
        self.first, self.second = InertBlock(), InertBlock()
        self.topology = topology()
        self.policy, self.policy_sha = object(), "a" * 64
        self.busy = self.pressure = self.cancelled = False
        self.admissions, self.moves, self.reservations = [], [], []
        self.preferences = {"version": 1, "enabled": True, "mode": "manual", "routes": routes or [
            {"regionId": "decoder.blocks.0", "deviceId": "cuda:0", "executionMode": "packed-cpu-tile-fallback", "acknowledgePackedCpuFallback": True},
            {"regionId": "decoder.regions.0.decoder.blocks.1", "deviceId": "cuda:1", "executionMode": "packed-cpu-tile-fallback", "acknowledgePackedCpuFallback": True}]}
        self.consumer = REGIONAL.RegionalExecutionConsumer(owner_id="owned-fixture", settings=lambda: self.preferences,
            policy_identity=lambda: self.policy, policy_sha=lambda: self.policy_sha, topology=lambda: self.topology,
            quiescent=self.quiescent, describe=lambda module: {"dimensions": module.width, "packedOwnerBytes": len(module.learned)},
            tensor_bytes=lambda value: value.bytes, tensor_device=lambda value: value.device, transfer=self.transfer,
            reserve_transfer=self.reserve, admit_compute=self.admit, training=lambda module: module.training,
            cancelled=lambda: self.cancelled)
        REGIONAL.bind_regional_blocks(self.consumer, [("decoder.blocks.0", self.first), ("decoder.regions.0.decoder.blocks.1", self.second)])
    def quiescent(self):
        if self.busy:
            raise RuntimeError("live cache/pin/transaction held")
    def transfer(self, value, destination):
        self.moves.append((value.device, destination, value.bytes))
        return Activity(value.values, destination)
    @contextmanager
    def reserve(self, device, size, label):
        item = {"device": device, "size": size, "closed": False, "allocated": 0}
        self.reservations.append(item)
        try:
            yield SimpleNamespace(mark_allocated=lambda count: item.update(allocated=count))
        finally:
            item["closed"] = True
    def admit(self, module, hidden, target, route):
        self.admissions.append((target, hidden.bytes, route["executionMode"]))
        if self.pressure:
            raise RuntimeError("fake device pressure")
    def activate(self):
        view = self.consumer.prepare()
        return self.consumer.activate({key: view[key] for key in ("policySha256", "topologyIdentitySha256", "registrySha256", "topology")})


class ManualRegionalRoutingFixtures(unittest.TestCase):
    def test_real_decorator_dispatches_two_registered_regions_with_exact_admitted_roundtrips(self):
        fixture = RoutingFixture()
        self.addCleanup(fixture.consumer.close)
        original = (fixture.first.learned, fixture.second.learned)
        self.assertFalse(fixture.consumer.active)  # Saved enabled preferences are not activation.
        state = fixture.activate()
        self.assertTrue(state["leaseValid"])
        self.assertFalse(state["contextGpuCapacityCredited"])
        result = fixture.second.forward(fixture.first.forward(Activity()))
        self.assertEqual((result.values, result.device), ((3, 4), "cpu"))
        self.assertEqual(fixture.first.calls, ["cuda:0"])
        self.assertEqual(fixture.second.calls, ["cuda:1"])
        self.assertEqual(fixture.moves, [("cpu", "cuda:0", 8), ("cuda:0", "cpu", 8), ("cpu", "cuda:1", 8), ("cuda:1", "cpu", 8)])
        self.assertTrue(all(item["closed"] and item["allocated"] == 8 for item in fixture.reservations))
        status = fixture.consumer.status()
        self.assertEqual(status["completedRegionDispatches"], 2)
        self.assertEqual(status["actualActivationTransferBytes"], 32)
        self.assertAlmostEqual(status["estimatedKnownTransferSeconds"], .032)
        self.assertEqual(status["unknownTransferCosts"], 0)
        self.assertEqual(original, (fixture.first.learned, fixture.second.learned))

    def test_cache_is_preserved_and_changes_wait_instead_of_clearing_context(self):
        fixture = RoutingFixture()
        self.addCleanup(fixture.consumer.close)
        fixture.activate()
        cache = object()
        _output, returned = fixture.first.forward_cached(Activity(), cache)
        self.assertIs(returned, cache)
        epoch = fixture.consumer.status()["epoch"]
        fixture.busy = True
        with self.assertRaisesRegex(RuntimeError, "live cache"):
            fixture.consumer.retire(expected_epoch=epoch)
        self.assertEqual(fixture.consumer.status()["epoch"], epoch)
        fixture.busy = False
        fixture.consumer.retire(expected_epoch=epoch)
        self.assertFalse(fixture.consumer.active)

    def test_stale_policy_topology_owner_and_geometry_refuse_without_silent_fallback(self):
        for change in ("policy", "topology", "geometry"):
            with self.subTest(change=change):
                fixture = RoutingFixture()
                self.addCleanup(fixture.consumer.close)
                fixture.activate()
                if change == "policy":
                    fixture.policy = object()
                elif change == "topology":
                    fixture.topology["devices"][2]["memoryGroupId"] = "replacement-physical-owner"
                else:
                    fixture.first.width = 3
                with self.assertRaises(RuntimeError):
                    fixture.first.forward(Activity())
                self.assertFalse(fixture.consumer.status()["leaseValid"])
                self.assertEqual(fixture.first.calls, [])
                with self.assertRaisesRegex(RuntimeError, "invalidated"):
                    fixture.first.forward(Activity())

    def test_pressure_cancel_and_failure_never_retry_an_effect_or_leak_reservations(self):
        for failure in ("pressure", "cancel", "transfer"):
            with self.subTest(failure=failure):
                fixture = RoutingFixture()
                self.addCleanup(fixture.consumer.close)
                fixture.activate()
                if failure == "pressure":
                    fixture.pressure = True
                elif failure == "cancel":
                    fixture.cancelled = True
                else:
                    fixture.consumer.transfer = lambda *_args: (_ for _ in ()).throw(RuntimeError("fake transfer failed"))
                with self.assertRaises((RuntimeError, InterruptedError)):
                    fixture.first.forward(Activity())
                self.assertEqual(fixture.first.calls, [])
                self.assertTrue(all(item["closed"] for item in fixture.reservations))
                self.assertEqual(fixture.consumer.status()["activeCalls"], 0)

    def test_training_is_explicitly_inapplicable_and_copies_never_inherit_armed_leases(self):
        fixture = RoutingFixture()
        self.addCleanup(fixture.consumer.close)
        fixture.activate()
        fixture.first.training = True
        fixture.first.forward(Activity())
        self.assertEqual(fixture.first.calls, ["cpu"])
        self.assertEqual(fixture.consumer.status()["trainingInapplicableCalls"], 1)
        self.assertFalse(fixture.consumer.status()["trainingSupported"])
        duplicate = copy.deepcopy(fixture.second)
        duplicate.forward(Activity())
        self.assertEqual(duplicate.calls, ["cpu"])
        self.assertEqual(fixture.consumer.status()["completedRegionDispatches"], 0)

    def test_direct_metal_and_explicit_fallback_modes_remain_distinct_and_npu_unavailable(self):
        fixture = RoutingFixture(routes=[{"regionId": "decoder.blocks.0", "deviceId": "mps:0", "executionMode": "native"}])
        self.addCleanup(fixture.consumer.close)
        fixture.activate()
        fixture.first.forward(Activity())
        self.assertEqual(fixture.first.calls, ["mps"])
        self.assertEqual(REGIONAL.canonical_device("mps:0"), "mps")
        for invalid in ({"deviceId": "cuda:0", "executionMode": "native"}, {"deviceId": "npu:0", "executionMode": "native"}):
            fixture.consumer.retire()
            fixture.preferences["routes"] = [{"regionId": "decoder.blocks.0", **invalid}]
            with self.assertRaises(ValueError):
                fixture.activate()
        with self.assertRaisesRegex(ValueError, "acknowledgment"):
            REGIONAL.validate_regional_execution_settings({"version": 1, "enabled": True, "mode": "manual", "routes": [
                {"regionId": "decoder.blocks.0", "deviceId": "cuda:0", "executionMode": "packed-cpu-tile-fallback"}]})

    def test_compute_topology_ignores_mutable_capacity_and_apfs_storage_but_not_peer_changes(self):
        fixture = RoutingFixture()
        self.addCleanup(fixture.consumer.close)
        reviewed = copy.deepcopy(fixture.topology)
        reviewed["devices"][0]["availableBytes"] = 1
        reviewed["devices"].append({"id": "storage:pool", "kind": "storage", "memoryGroupId": "apfs:rewritten"})
        self.assertEqual(REGIONAL.topology_identity(reviewed), REGIONAL.topology_identity(fixture.topology))
        fixture.consumer.prepare(topology=reviewed)
        reviewed["links"][0]["peerAccess"] = True
        with self.assertRaisesRegex(ValueError, "stale"):
            fixture.consumer.prepare(topology=reviewed)

    def test_per_device_and_shared_physical_hot_budgets_are_both_consumed(self):
        limits = {"cuda:0": {"memoryGroupId": "a", "hotBytes": 18}, "cuda:1": {"memoryGroupId": "b", "hotBytes": 18}}
        self.assertTrue(REGIONAL.regional_group_admission({"cuda:0": 12}, limits, "cuda:1", 12))
        self.assertFalse(REGIONAL.regional_group_admission({"cuda:0": 12}, limits, "cuda:0", 12))
        shared = {"mps": {"memoryGroupId": "ram", "hotBytes": 18}, "alias": {"memoryGroupId": "ram", "hotBytes": 18}}
        self.assertFalse(REGIONAL.regional_group_admission({"mps": 12}, shared, "alias", 12))


class Device:
    def __init__(self, name):
        self.name = str(name)
        self.type = self.name.split(":")[0]
    def __str__(self):
        return self.name
    def __eq__(self, other):
        return isinstance(other, Device) and self.name == other.name


class Buffer:
    def __init__(self, data):
        self.data = bytes(data)
        self._version = 0


class RealPagerTransitionFixture(unittest.TestCase):
    def test_actual_use_method_keeps_distinct_device_owners_and_rolls_back_failed_admission(self):
        source = ast.parse((ROOT / "omni_core/native_core_paging.py").read_text())
        owner = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "NativeCorePager")
        method = copy.deepcopy(next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "use"))
        method.returns = None
        for argument in method.args.args:
            argument.annotation = None
        failing = {"value": False}
        def empty_like(value, device):
            if failing["value"] and len(value.data) == 8:
                raise RuntimeError("fake allocation refused")
            return Buffer(bytes(len(value.data)))
        namespace = {"contextmanager": contextmanager, "nullcontext": nullcontext,
            "torch": SimpleNamespace(device=Device, empty_like=empty_like), "canonical_device": REGIONAL.canonical_device,
            "regional_group_admission": REGIONAL.regional_group_admission}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "actual-owned-pager-use", "exec"), namespace)
        class Pager:
            use = namespace["use"]
            def __init__(self):
                self._owners, self._lock, self._closed, self._clock = {}, threading.RLock(), False, 0
                self._learning_revision_active = False
                self.accelerator_hot_bytes = 18  # Old global budget could not retain both 12-byte owners.
                self.reserve_admission, self.shared_resource_policy, self._page_ins = lambda *_args: None, None, 0
            def refresh_budget(self): pass
            def _observe_mutations(self, *_args): pass
            def _invalidate_validation(self, *_args): pass
            def _event(self, *_args): pass
            def cool_to_budget(self): pass
            def _owner_bytes(self, owner): return sum(len(value.data) for value in owner.cold.values())
            def _regional_limits(self): return {"cuda:0": {"memoryGroupId": "a", "hotBytes": 18}, "cuda:1": {"memoryGroupId": "b", "hotBytes": 18}}
            def _accelerator_usage(self):
                return {name: sum(self._owner_bytes(owner) for owner in self._owners.values() if owner.device == Device(name)) for name in ("cuda:0", "cuda:1")}
            def _copy_bounded(self, source, target): target.data = source.data
            def _pause(self, message): raise RuntimeError(message)
            def evict(self, module):
                owner = self._owners[module]
                module._buffers.update(owner.cold)
                owner.device = None
        class Module:
            def __init__(self): self._buffers = {}
        pager = Pager()
        first, second, third = Module(), Module(), Module()
        for module in (first, second, third):
            cold = {"signs": Buffer(b"1234"), "state": Buffer(b"strength")}
            module._buffers.update(cold)
            pager._owners[module] = SimpleNamespace(cold=cold, loaded=True, pins=0, device=None, heat=0,
                versions={key: 0 for key in cold}, dirty_names=set())
        with pager.use(first, Device("cuda:0")):
            self.assertEqual(pager._owners[first].pins, 1)
        with pager.use(second, Device("cuda:1")):
            pass
        self.assertEqual(pager._accelerator_usage(), {"cuda:0": 12, "cuda:1": 12})
        failing["value"] = True
        with self.assertRaisesRegex(RuntimeError, "allocation refused"):
            with pager.use(third, Device("cuda:1")):
                self.fail("failed partial allocation entered the body")
        self.assertIsNone(pager._owners[third].device)
        self.assertEqual(third._buffers["state"].data, b"strength")
        self.assertEqual(pager._owners[third].pins, 0)
        self.assertEqual(pager._owners[first].device, Device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
