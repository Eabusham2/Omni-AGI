"""New actual-source admission/worker boundaries with inert injected owners."""
import ast
import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def methods(path, class_name, names, namespace):
    tree = ast.parse(path.read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = []
    for node in owner.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node = copy.deepcopy(node)
            node.returns = None
            for argument in (*node.args.args, *node.args.kwonlyargs):
                argument.annotation = None
            selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), "inert-actual-regional-boundary", "exec"), namespace)
    return namespace


class RegionalBoundaryFixtures(unittest.TestCase):
    def test_inapplicable_training_does_not_become_blocked_by_stale_manual_inference_policy(self):
        spec = importlib.util.spec_from_file_location("owned_region_fixture_reuse", Path(__file__).with_name("test_manual_regional_execution.py"))
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        fixture = fixtures.RoutingFixture()
        self.addCleanup(fixture.consumer.close)
        fixture.activate()
        fixture.first.training = True
        fixture.policy_sha = "b" * 64
        fixture.first.forward(fixtures.Activity())
        self.assertEqual(fixture.first.calls, ["cpu"])
        self.assertEqual(fixture.moves, [])
        self.assertEqual(fixture.consumer.status()["trainingInapplicableCalls"], 1)

    def test_actual_attention_admission_uses_target_gpu_not_roots_unrelated_free_reading(self):
        regional = {"target": "cuda:1"}
        namespace = methods(ROOT / "omni_core/working_attention_paging.py", "WorkingAttentionPager", {"admit_compute"},
            {"current_regional_target": lambda _policy: regional["target"], "canonical_device": str})
        calls, host_estimates = [], []
        class Device:
            type = "cuda"
            def __str__(self): return "cuda:1"
        def pause(message, **_extra): raise RuntimeError(message)
        def status(size):
            host_estimates.append(size)
            return {"memoryPressure": False, "acceleratorFreeMemoryBytes": 1}  # Root cuda:0, not target cuda:1.
        pager = SimpleNamespace(device_tile_budget_bytes=1024, compute_device=Device(), check_cancelled=lambda: None,
            _resource_status=status, _pause=pause, _metrics={"peakComputeTileBytes": 0},
            resource_policy=SimpleNamespace(require_device=lambda target, size, label: calls.append((str(target), size, label))))
        namespace["admit_compute"](pager, 32, "new regional tile", device=Device())
        self.assertEqual(calls[0][:2], ("cuda:1", 32))
        self.assertEqual(host_estimates, [0])
        regional["target"] = None
        with self.assertRaisesRegex(RuntimeError, "accelerator headroom"):
            namespace["admit_compute"](pager, 32, "original path", device=Device())
        self.assertEqual(host_estimates, [0, 32])

    def test_actual_device_entry_normalizes_metal_tensor_index_without_aliasing_physical_groups(self):
        namespace = methods(ROOT / "omni_core/device_resources.py", "DeviceResourceManager", {"_entry"},
            {"canonical_device": lambda value: "mps" if str(value) == "mps:0" else str(value), "DeviceResourceAdmissionError": RuntimeError})
        observed = {"devices": [{"id": "mps:0", "executionDevice": "mps", "backend": "mps", "memoryGroupId": "ram:system", "availableBytes": 64}]}
        manager = SimpleNamespace(topology=lambda: observed, budgets=lambda *_args, **_kwargs: [{"deviceId": "mps:0", "ceilingBytes": 64}])
        device, budget = namespace["_entry"](manager, "mps:0", {"systemRamBudgetBytes": 64})
        self.assertEqual(device["memoryGroupId"], "ram:system")
        self.assertEqual(budget["deviceId"], "mps:0")

    def test_actual_worker_delegates_only_to_loaded_owner_and_refuses_foreign_settings(self):
        class Fault(RuntimeError): pass
        names = {"_regional_loaded_owner", "regional_execution_prepare", "regional_execution_status", "regional_execution_activate", "regional_execution_deactivate"}
        namespace = methods(ROOT / "worker.py", "Worker", names, {"RpcFault": Fault})
        calls = []
        brain = SimpleNamespace(regional_execution_prepare=lambda **kwargs: kwargs,
            regional_execution_status=lambda: {"loaded": True}, regional_execution_activate=lambda reviewed: reviewed,
            regional_execution_deactivate=lambda **kwargs: kwargs)
        class Worker:
            worker_role = "neural"
            def _get(self, params, **kwargs):
                if kwargs.get("allow_load", False): raise AssertionError("regional control tried to load a brain")
                calls.append(dict(params))
                return brain
        for name in names:
            setattr(Worker, name, namespace[name])
        worker = Worker()
        self.assertEqual(worker.regional_execution_status({"brainId": "owned"}, None), {"loaded": True})
        self.assertEqual(worker.regional_execution_prepare({"brainId": "owned", "start": 3, "count": 2, "topology": {"fixture": True}}, None),
                         {"start": 3, "count": 2, "topology": {"fixture": True}})
        self.assertEqual(worker.regional_execution_activate({"brainId": "owned", "reviewed": {"exact": "review"}}, None), {"exact": "review"})
        self.assertEqual(worker.regional_execution_deactivate({"brainId": "owned", "epoch": "current"}, None), {"epoch": "current"})
        with self.assertRaises(Fault):
            worker.regional_execution_activate({"brainId": "owned", "reviewed": {}, "settings": {}}, None)
        worker.worker_role = "inspection"
        with self.assertRaises(Fault):
            worker.regional_execution_status({"brainId": "owned"}, None)
        self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
