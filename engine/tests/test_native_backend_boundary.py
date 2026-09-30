"""Metadata/stub device migrations: no real accelerator, brain or model init."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.offload import NeuralStateResourcePause


class MovableStateStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.full((4, 8), 85, dtype=torch.uint8))
        self.register_buffer("control", torch.tensor([0.25, 0.5]))
        self.moves = []
        self.fail_move = False

    def to(self, device):
        self.moves.append(str(device))
        if self.fail_move:
            self._buffers["weight"] = torch.zeros_like(self.weight)
            raise RuntimeError("stub transfer failed")
        return self


class BackendBoundaryTests(unittest.TestCase):
    def fixture(self, *, pressure=False):
        brain = object.__new__(AdaptiveBrain)
        module = MovableStateStub()
        brain.device = torch.device("cuda:0")  # Metadata only; tensors stay CPU.
        brain.device_backend = "cuda"
        brain._native_preferred_device = torch.device("cuda:0")
        brain._native_preferred_backend = "cuda"
        brain._native_backend_last_changed = 0.0
        brain._native_backend_residency = {"requestedDevice": "cuda:0", "actualBackend": "cuda"}
        state = {"pinnedOwners": 0, "maximumPackedOwnerBytes": 1024, "cpuHeapBytes": 2048}
        calls = []
        brain.core_pager = SimpleNamespace(
            status=lambda: dict(state), flush=lambda: calls.append("flush"),
            refresh_budget=lambda **_: calls.append("refresh"),
            cool_to_budget=lambda size: calls.append(("cool", size)),
        )
        brain.resource_policy = SimpleNamespace(status=lambda **_: {
            "memoryPressure": pressure, "acceleratorFreeMemoryBytes": 16 * 1024 ** 2,
        })
        brain._trainable_modules = lambda: (module,)
        limit = [128]
        brain._native_core_budget_for_device = lambda _status, _device=None: {"acceleratorCoreHotBytes": limit[0]}
        brain._configure_working_attention_resources = lambda: calls.append("activity-configure")
        brain._maintain_neural_state_resources = lambda: calls.append("reclaim")
        brain.config = SimpleNamespace(device="cuda:0", d_model=8, d_ff=16, vsa_dim=32,
                                       working_memory_slots=128, max_seq_len=1024)
        brain.liquid_state = torch.tensor([[0.5, -0.5]])
        brain.working_memory = [torch.tensor([1., 2.])]
        brain._native_move_tensor = lambda value, _device: value.detach().clone()  # CPU-only simulation.
        return brain, module, state, limit, calls

    def test_fallback_and_safe_restore_preserve_exact_values_and_declared_shape(self):
        brain, module, _, limit, calls = self.fixture()
        expected = module.weight.clone()
        recurrent = brain.liquid_state.clone()
        old_config = dict(vars(brain.config))
        report = brain.prepare_native_execution(operation="chat", quiescent=True)
        self.assertEqual(brain.device.type, "cpu")
        self.assertEqual(report["actualBackend"], "cpu")
        self.assertFalse(report["alternateModelLoaded"])
        self.assertFalse(report["weightOrContextShapeChanged"])
        self.assertIsNone(report["slowdown"]["percent"])
        self.assertTrue(torch.equal(module.weight, expected))
        self.assertTrue(torch.equal(brain.liquid_state, recurrent))
        self.assertEqual(vars(brain.config), old_config)
        self.assertIn("activity-configure", calls)
        limit[0] = 16384
        brain._native_backend_last_changed = 0.0
        report = brain.prepare_native_execution(operation="ingest", quiescent=True)
        self.assertEqual(report["actualBackend"], "cuda")
        self.assertEqual(module.moves, ["cpu", "cuda:0"])
        self.assertTrue(torch.equal(module.weight, expected))

    def test_pinned_nonboundary_and_cpu_insufficiency_pause_without_moving(self):
        brain, module, state, _, _ = self.fixture()
        with self.assertRaises(NeuralStateResourcePause):
            brain.prepare_native_execution(operation="chat")
        state["pinnedOwners"] = 1
        with self.assertRaises(NeuralStateResourcePause):
            brain.prepare_native_execution(operation="chat", quiescent=True)
        self.assertEqual(module.moves, [])
        brain, module, _, _, _ = self.fixture(pressure=True)
        with self.assertRaisesRegex(NeuralStateResourcePause, "RAM ceiling"):
            brain.prepare_native_execution(operation="chat", quiescent=True)
        self.assertEqual(module.moves, [])
        self.assertEqual(brain.device.type, "cuda")

    def test_already_hot_unchanged_backend_still_reclaims_and_pauses_at_ram_ceiling(self):
        brain, module, _, limit, calls = self.fixture(pressure=True)
        limit[0] = 16384
        with self.assertRaisesRegex(NeuralStateResourcePause, "RAM ceiling"):
            brain.prepare_native_execution(operation="chat", quiescent=True)
        self.assertIn(("cool", 0), calls)
        self.assertIn("reclaim", calls)
        self.assertEqual(module.moves, [])
        brain, module, _, limit, calls = self.fixture()
        limit[0] = 16384
        samples = iter(({"memoryPressure": True}, {"memoryPressure": False}))
        brain.resource_policy.status = lambda **_: next(samples)
        brain.prepare_native_execution(operation="chat", quiescent=True)
        self.assertIn("reclaim", calls)
        self.assertEqual(module.moves, [])

    def test_failed_transfer_restores_registered_values_and_recurrent_refs(self):
        brain, module, _, _, _ = self.fixture()
        expected = module.weight
        recurrent = brain.liquid_state
        module.fail_move = True
        with self.assertRaisesRegex(RuntimeError, "stub transfer failed"):
            brain.prepare_native_execution(operation="chat", quiescent=True)
        self.assertIs(module.weight, expected)
        self.assertIs(brain.liquid_state, recurrent)
        self.assertEqual(brain.device.type, "cuda")

    def test_fresh_create_initializes_but_load_only_defers_constructor(self):
        source = Path(__file__).parents[1] / "omni_core" / "brain.py"
        tree = ast.parse(source.read_text())
        brain = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        methods = {node.name: node for node in brain.body if isinstance(node, ast.FunctionDef)}
        for name, expected in (("create", False), ("_load_impl", True)):
            calls = [node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "cls"]
            self.assertEqual(len(calls), 1)
            flags = {keyword.arg: keyword.value for keyword in calls[0].keywords}
            self.assertEqual("_loading_checkpoint" in flags, expected)
            if expected:
                self.assertIs(flags["_loading_checkpoint"].value, True)


if __name__ == "__main__":
    unittest.main()
