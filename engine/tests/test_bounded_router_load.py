"""Pure router-state/file/admission fixtures; no brain/model constructor/run."""
import ast
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from omni_core.bounded_tensor_io import (
    AdmittedTensorMapping, BoundedTensorFile, atomic_save_tensors_bounded, load_module_bounded,
)
from omni_core.brain import AdaptiveBrain
from omni_core.offload import NeuralStateResourcePause
from omni_core.spiking import AssociativeSpikingRouter, STDPSynapses


class RouterStateFixture(torch.nn.Module):
    """Storage only, borrowing actual control-header/default/validation methods."""
    prepare_bounded_state_load = AssociativeSpikingRouter.prepare_bounded_state_load
    bounded_optional_state_defaults = AssociativeSpikingRouter.bounded_optional_state_defaults
    validate_bounded_state_load = AssociativeSpikingRouter.validate_bounded_state_load

    def __init__(self, neurons=17):
        super().__init__()
        self.neurons = neurons
        self.register_buffer("active_prefix_neurons", torch.tensor(neurons))
        self.register_buffer("region_ends", torch.tensor([neurons]))
        self.population = torch.nn.Module()
        self.population.register_buffer("membrane", torch.zeros(neurons))
        self.population.register_buffer("spike_count", torch.zeros(neurons))
        self.synapses = torch.nn.Module()
        self.synapses.pre_neurons = self.synapses.post_neurons = neurons
        self.synapses._validate_packed = STDPSynapses._validate_packed.__get__(self.synapses)
        self.synapses._tile_budget = STDPSynapses._tile_budget.__get__(self.synapses)
        self.synapses._ram = STDPSynapses._ram.__get__(self.synapses)
        self.synapses.register_buffer("_packed_weights", torch.full((neurons, (neurons + 3) // 4), 0x55, dtype=torch.uint8))
        self.synapses.register_buffer("eligibility_accumulator", torch.zeros(neurons, neurons, dtype=torch.int16))
        self.synapses.register_buffer("stability", torch.zeros(neurons, neurons))
        self.synapses.register_buffer("uses", torch.zeros(neurons, neurons))
        self.synapses.register_buffer("pre_trace", torch.zeros(neurons))
        self.synapses.register_buffer("post_trace", torch.zeros(neurons))
        self.synapses.register_buffer("plasticity_events", torch.zeros((), dtype=torch.long))
        self.synapses.register_buffer("decay_cycles", torch.zeros((), dtype=torch.long))


class BoundedRouterLoadFixtures(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="omni-router-load-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def state(self, owner):
        values = {"router." + name: value.clone() for name, value in owner.state_dict().items()}
        for name, value in values.items():
            if name.endswith(("membrane", "spike_count", "stability", "uses", "pre_trace", "post_trace", "eligibility_accumulator")):
                value.copy_(torch.arange(value.numel()).reshape_as(value).to(value.dtype))
        values["router.active_prefix_neurons"] = torch.tensor(5)
        values["router.region_ends"] = torch.tensor([5, owner.neurons])
        return values

    def test_actual_matrix_prefixes_and_controls_load_into_existing_destinations_in_chunks(self):
        owner = RouterStateFixture()
        values = self.state(owner)
        path = self.root / "plasticity.safetensors"
        atomic_save_tensors_bounded(path, values, chunk_bytes=24)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        pointers = {name: value.data_ptr() for name, value in owner.state_dict().items() if name != "region_ends"}
        reader = BoundedTensorFile(path, chunk_bytes=24)
        chunks = []
        with patch.object(reader, "tensor", side_effect=AssertionError("full source tensor materialized")):
            load_module_bounded(owner, reader, "router.", on_chunk=lambda *entry: chunks.append(entry))
        for name, value in owner.state_dict().items():
            self.assertTrue(torch.equal(value, values["router." + name]), name)
            if name in pointers:
                self.assertEqual(value.data_ptr(), pointers[name], name)
        self.assertGreater(len(chunks), 100)
        self.assertLessEqual(reader.peak_transfer_bytes, 24)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_only_two_missing_additive_nonweight_controls_receive_native_defaults(self):
        owner = RouterStateFixture()
        values = self.state(owner)
        for name in ("router.active_prefix_neurons", "router.region_ends"):
            values.pop(name)
        path = self.root / "legacy-controls.safetensors"
        atomic_save_tensors_bounded(path, values)
        load_module_bounded(owner, BoundedTensorFile(path), "router.")
        self.assertEqual(int(owner.active_prefix_neurons), owner.neurons)
        self.assertEqual(owner.region_ends.tolist(), [owner.neurons])
        for name, value in owner.state_dict().items():
            if "router." + name in values:
                self.assertTrue(torch.equal(value, values["router." + name]))

    def test_missing_actual_nonweight_matrix_or_floating_learned_weight_is_rejected(self):
        for kind in ("missing-eligibility", "float-learned-weight"):
            owner = RouterStateFixture()
            values = self.state(owner)
            if kind == "missing-eligibility":
                values.pop("router.synapses.eligibility_accumulator")
            else:
                values.pop("router.synapses._packed_weights")
                values["router.synapses.weights"] = torch.zeros(owner.neurons, owner.neurons)
            path = self.root / (kind + ".safetensors")
            atomic_save_tensors_bounded(path, values)
            old = owner.synapses._packed_weights.clone()
            with self.assertRaisesRegex(ValueError, "checkpoint mismatch"):
                load_module_bounded(owner, BoundedTensorFile(path), "router.")
            self.assertTrue(torch.equal(old, owner.synapses._packed_weights))

    def test_invalid_control_headers_and_loaded_region_values_fail_closed(self):
        for invalid in (torch.tensor([5, 3, 17]), torch.tensor([5, 16]), torch.tensor([5.0, 17.0]), torch.tensor([0, 17])):
            owner = RouterStateFixture()
            values = self.state(owner)
            values["router.region_ends"] = invalid
            path = self.root / "invalid-regions.safetensors"
            atomic_save_tensors_bounded(path, values)
            with self.assertRaisesRegex(ValueError, "region geometry"):
                load_module_bounded(owner, BoundedTensorFile(path), "router.")

    def test_dtype_preflight_rejects_before_any_existing_matrix_changes(self):
        owner = RouterStateFixture()
        values = self.state(owner)
        values["router.synapses.uses"] = values["router.synapses.uses"].double()
        path = self.root / "wrong-control-dtype.safetensors"
        atomic_save_tensors_bounded(path, values)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            load_module_bounded(owner, BoundedTensorFile(path), "router.")
        self.assertEqual(float(owner.population.membrane.sum()), 0.0)

    def test_actual_packed_validation_rejects_reserved_and_padding_without_dense_decode(self):
        for invalid in ("reserved", "padding"):
            owner = RouterStateFixture()
            values = self.state(owner)
            packed = values["router.synapses._packed_weights"]
            if invalid == "reserved":
                packed[0, 0] = 0xFF
            else:
                packed[0, -1] = 0
            path = self.root / "invalid-packed.safetensors"
            atomic_save_tensors_bounded(path, values)
            with patch("omni_core.spiking.unpack_ternary_weight_rows", side_effect=AssertionError("dense validation shadow")):
                with self.assertRaisesRegex(ValueError, "reserved|padding"):
                    load_module_bounded(owner, BoundedTensorFile(path, chunk_bytes=24), "router.")

    def test_auxiliary_resident_view_excludes_router_and_admits_only_requested_final_state(self):
        path = self.root / "state.safetensors"
        values = {"router.synapses.stability": torch.zeros(100, 100),
                  "state.liquid": torch.arange(6).reshape(1, 6).float(),
                  "state.working_memory": torch.arange(90).reshape(15, 6).float()}
        atomic_save_tensors_bounded(path, values)
        reader = BoundedTensorFile(path, chunk_bytes=24)
        reserves = []
        view = AdmittedTensorMapping(reader, names=["state.liquid", "state.working_memory"],
                                     reserve=lambda size, name: reserves.append((size, name)))
        self.assertEqual(list(view), ["state.liquid", "state.working_memory"])
        self.assertNotIn("router.synapses.stability", view)
        self.assertEqual(reserves, [])
        self.assertTrue(torch.equal(view["state.working_memory"], values["state.working_memory"]))
        self.assertEqual(reserves, [(360 + 24, "state.working_memory")])
        self.assertEqual(view.destination_bytes_read, 360)
        self.assertLessEqual(reader.peak_transfer_bytes, 24)
        with self.assertRaises(KeyError):
            view["router.synapses.stability"]

    def test_auxiliary_admission_rejection_and_cancel_precede_allocation(self):
        path = self.root / "state.safetensors"
        atomic_save_tensors_bounded(path, {"state.liquid": torch.zeros(1, 6)})
        reader = BoundedTensorFile(path)
        def reject(_size, _name):
            raise NeuralStateResourcePause("fixture physical pause", {})
        with patch("omni_core.bounded_tensor_io.torch.empty", side_effect=AssertionError("allocated before admission")):
            with self.assertRaises(NeuralStateResourcePause):
                AdmittedTensorMapping(reader, reserve=reject)["state.liquid"]
            with self.assertRaises(InterruptedError):
                AdmittedTensorMapping(reader, cancelled=lambda: True)["state.liquid"]

    def test_actual_cpu_admission_ignores_unrelated_accelerator_free_memory(self):
        tree = ast.parse((Path(__file__).parents[1] / "omni_core" / "brain.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        initializer = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        reserve = next(node for node in ast.walk(initializer) if isinstance(node, ast.FunctionDef) and node.name == "reserve_core_admission")
        calls = []
        owner = SimpleNamespace(resource_policy=SimpleNamespace(status=lambda **kwargs: calls.append(kwargs) or {
            "memoryPressure": False, "acceleratorFreeMemoryBytes": 0}))
        namespace = {"self": owner, "torch": torch, "NeuralStateResourcePause": NeuralStateResourcePause}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[reserve], type_ignores=[])), "<actual-admission-method>", "exec"), namespace)
        namespace["reserve_core_admission"](100, torch.device("cpu"))
        self.assertEqual(calls[-1], {"estimated_ram_bytes": 100})
        with self.assertRaises(NeuralStateResourcePause):
            namespace["reserve_core_admission"](100, SimpleNamespace(type="cuda"))

    def test_production_loader_has_no_full_plasticity_map_or_origin_router_read(self):
        tree = ast.parse((Path(__file__).parents[1] / "omni_core" / "brain.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AdaptiveBrain")
        loader = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_load_impl")
        calls = [node.func.id for node in ast.walk(loader) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("load_tensors", calls)
        self.assertIn("load_module_bounded", calls)
        self.assertIn("AdmittedTensorMapping", calls)
        self.assertIn("validate_compatible_architecture_lineage", calls)


if __name__ == "__main__":
    unittest.main()
