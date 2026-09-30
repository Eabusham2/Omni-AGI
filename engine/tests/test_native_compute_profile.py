"""Injected timers/RPC source checks only: never run a hardware primitive."""
import ast
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.native_compute_profile import profile_native_projection_compute
from omni_core.offload import NeuralStateResourcePause
from worker import RpcFault, Worker


class NativeComputeProfileFixtures(unittest.TestCase):
    def worker(self):
        value = Worker.__new__(Worker)
        value.worker_role = "neural"
        value._inline_generations = {}
        value._inline_lock = threading.RLock()
        value._cooperative_cancel = threading.Event()
        value._get = lambda _params: self.fail("hardware profiling tried to load a brain")
        value._default_root = lambda: Path("/fixture-not-probed")
        return value

    def test_injected_clock_profiles_whole_declared_primitive_not_bare_integer_mm(self):
        ticks = iter([0, .001, .002, .003, .004, .005, .006, .007])
        calls, reserves = [], []
        value = profile_native_projection_compute(primitive=lambda: calls.append(True), clock=lambda: next(ticks),
                                                   reserve=lambda count, device: reserves.append((count, device)))
        self.assertEqual(len(calls), 4)
        self.assertEqual(reserves, [(1053700, "cpu")])
        self.assertEqual(value["projectionMacsPerRun"], 32768)
        self.assertEqual(value["sampleTensorBytes"], 5124)
        self.assertEqual(value["projectionMacsPerSecond"], 32768 * 1_000_000 // value["medianRunMicroseconds"])
        self.assertEqual(value["activityDtype"], "float32")
        self.assertEqual(value["implementation"], "injected-fixture")
        self.assertFalse(value["neuralModelConstructed"])
        self.assertFalse(value["fullNeuralThroughputMeasured"])
        self.assertFalse(value["neuralQualityMeasured"])

    def test_admission_rejection_cancel_and_invalid_clock_never_invent_a_speed(self):
        def reject(*_args):
            raise NeuralStateResourcePause("fixture admission", {})
        calls = []
        with self.assertRaises(NeuralStateResourcePause):
            profile_native_projection_compute(primitive=lambda: calls.append(True), reserve=reject)
        with self.assertRaises(InterruptedError):
            profile_native_projection_compute(primitive=lambda: calls.append(True), cancelled=lambda: True)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(ValueError, "timer"):
            profile_native_projection_compute(primitive=lambda: None, clock=lambda: 1)

    def test_sampling_stops_after_bounded_slow_primitive_window(self):
        ticks = iter([0, .25])
        value = profile_native_projection_compute(primitive=lambda: None, clock=lambda: next(ticks))
        self.assertEqual(value["runs"], 1)
        self.assertEqual(value["elapsedMicroseconds"], 250000)

    def test_worker_route_only_forwards_hardware_and_admits_before_injected_profiler(self):
        value = self.worker()
        captured = []
        policy = SimpleNamespace(status=lambda **kwargs: {"memoryPressure": False,
            "projectedProcessMemoryBytes": kwargs["estimated_ram_bytes"] + 1000})
        def profiler(**kwargs):
            captured.append(kwargs["device"])
            kwargs["reserve"](1053700, "cpu")
            self.assertFalse(kwargs["cancelled"]())
            return {"fixture": "no probe performed"}
        with patch("omni_core.offload.ResourcePolicy", return_value=policy), patch(
            "omni_core.native_compute_profile.profile_native_projection_compute", side_effect=profiler
        ):
            result = value.hardware_projection_profile({"device": "cpu", "ramBudgetBytes": 2 * 1024 ** 2}, "profile")
        self.assertEqual(result, {"fixture": "no probe performed"})
        self.assertEqual(captured, ["cpu"])

    def test_worker_rejects_model_input_untrusted_device_and_budget_before_profiling(self):
        value = self.worker()
        for params in ({"brainId": "brain"}, {"device": "remote-model"}, {"ramBudgetBytes": True}, {"ramBudgetBytes": 0},
                       {"ramBudgetBytes": 2 * 1024 ** 2, "hardwareTier": []}):
            with self.assertRaises(RpcFault):
                value.hardware_projection_profile(params, "profile")

    def test_inline_work_defers_measurement_without_interfering_with_it(self):
        value = self.worker()
        value._inline_generations = {("brain", "action"): SimpleNamespace(future=SimpleNamespace(done=lambda: False))}
        result = value.hardware_projection_profile({"device": "cpu", "ramBudgetBytes": 2 * 1024 ** 2}, "profile")
        self.assertFalse(result["available"])
        self.assertIn("active-inline-job", result["reason"])

    def test_production_profile_source_calls_actual_packed_primitive_without_module_construction(self):
        source = Path(__file__).parents[1] / "omni_core" / "native_compute_profile.py"
        tree = ast.parse(source.read_text())
        calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertIn("_packed_ternary_forward", calls)
        self.assertIn("pack_ternary_weight", calls)
        self.assertNotIn("OmniDecoder", calls)
        self.assertNotIn("AdaptiveBrain", calls)
        self.assertNotIn("PackedAdaptiveBitLinear", calls)
        self.assertTrue(all(not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call) for node in tree.body))


if __name__ == "__main__":
    unittest.main()
