"""Only fake factories, real module imports, primitive tensors and storage."""

import copy
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.architecture_migration import geometry_candidate_manifest, verify_geometry_checkpoint_migration
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded
from omni_core.evolution import NeuralEvolutionManager
from omni_core.evolution_anchors import save_retention_anchors
from omni_core.geometry_candidate_application import (
    ROOTS, GeometryReplayView, GeometryWorkingMemoryView, apply_isolated_geometry_candidate,
    install_geometry_runtime_views,
)
from omni_core.native_architecture import native_architecture_sha256, native_core_inventory
from omni_core.offload import DurableReplayBuffer, PagedWorkingMemory


class Policy:
    def status(self, **kwargs): return {"memoryPressure": False, "diskPressure": False}
    def require_disk(self, *args): pass


class Pager:
    def __init__(self): self.closed = False; self.refreshed = False
    def status(self): return {"pinnedOwners": 0}
    def flush(self): pass
    def cool_to_budget(self, *args): pass
    def construction(self, **kwargs): return nullcontext()
    def bind_names(self, roots): pass
    def refresh_budget(self, **kwargs): self.refreshed = kwargs.get("force") is True
    def close(self): self.closed = True; return {"checkpointDeleted": False}


class Module:
    def __init__(self, children=None, buffers=None):
        self.children = children or {}; self._buffers = buffers or {}; self.training = False
    def named_modules(self):
        yield "", self
        for name, child in self.children.items():
            for relative, module in child.named_modules(): yield name + ("." + relative if relative else ""), module
    def modules(self): return (module for _, module in self.named_modules())
    def parameters(self): return iter(())
    def state_dict(self):
        return {name + ("." if name else "") + field: value for name, module in self.named_modules()
            for field, value in module._buffers.items()}
    def train(self, value): self.training = value


class Projection(Module):
    def __init__(self, width):
        super().__init__(buffers={"_packed_forward_weight": torch.full((2, (width + 3) // 4), 0x55, dtype=torch.uint8),
            "_row_stability": torch.tensor([4, 5], dtype=torch.uint8), "_packed_forward_scale": torch.tensor(.25)})
        self.in_features, self.out_features = width, 2
    @property
    def ternary_weight_shape(self): return (2, self.in_features)


def descriptor():
    shape = {"dModel": 48, "layers": 1, "feedForward": 96, "nHeads": 4, "vsaDimensions": 64,
        "routerNeurons": 5, "modalityChannels": 4, "imageSize": 16, "audioSamples": 256, "videoFrames": 4,
        "workingMemoryItems": 32, "workspaceLatents": 8, "vocabSize": 261, "liquidMode": "cfc"}
    value = {"format": "omni-main-selected-native-architecture", "formatVersion": 1, "architecture": "OmniCortex",
        "externalPretrainedWeights": False, "qualityEvidence": "unmeasured-native-quality-deferred", "hardwareTier": "micro",
        "shape": shape, "inventory": native_core_inventory(shape), "sizing": {"selected": 1}}
    value["sha256"] = native_architecture_sha256(value)
    return value


def roots(width, router=None):
    values = {name: Module() for name in ROOTS}
    values["decoder"] = Module(children={"proj": Projection(width)})
    values["router"] = Module(children={"population": router or Module(buffers={"membrane": torch.arange(5).float()})})
    return values


class ApplicationFixtures(unittest.TestCase):
    def fixture(self, directory):
        candidate_id = "a" * 32
        engine = Path(directory) / "candidates" / candidate_id / "model" / "engine"
        engine.mkdir(parents=True)
        config = SimpleNamespace(d_model=48, d_ff=96, n_heads=4, idea_dim=48, n_layers=1, router_neurons=5,
            native_architecture=descriptor(), validate=lambda: None)
        brain = SimpleNamespace(engine_path=engine, config=config, device=torch.device("cpu"), resource_policy=Policy(),
            core_pager=Pager(), working_attention_pager=Pager(), _live_paging_cache_directory=Path(directory) / "cache",
            liquid_state=torch.arange(48).float().reshape(1, -1), working_memory=[], replay=[],
            _optimizer=object(), messages=[{"content": "unchanged"}], history={"unchanged": True})
        for name, module in roots(48).items(): setattr(brain, name, module)
        brain._configure_working_attention_resources = lambda: setattr(brain, "working_attention_pager", Pager())
        brain._replace_optimizer = lambda: setattr(brain, "_optimizer", object())
        core = {name + "." + key: value for name in ROOTS if name != "router" for key, value in getattr(brain, name).state_dict().items()}
        plastic = {"router." + key: value for key, value in brain.router.state_dict().items()}
        plastic["state.liquid"] = brain.liquid_state
        atomic_save_tensors_bounded(engine / "core.safetensors", core)
        atomic_save_tensors_bounded(engine / "plasticity.safetensors", plastic)
        return brain, candidate_id

    def test_prepared_swap_preserves_controls_and_independent_pager_and_proves_actual_files(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, candidate_id = self.fixture(directory)
            original_history, old_population, old_pager = brain.history, brain.router.children["population"], brain.core_pager
            before = brain.engine_path
            pager = Pager()
            result = apply_isolated_geometry_candidate(brain, {"mutation": "resize-width", "dModel": 64}, candidate_id,
                module_factory=lambda config, source: roots(config.d_model, source.router.children["population"]),
                pager_factory=lambda *_: pager)
            self.assertEqual(brain.config.d_model, 64)
            self.assertTrue(pager.refreshed); self.assertTrue(old_pager.closed)
            self.assertIs(brain.history, original_history)
            self.assertIs(brain.router.children["population"], old_population)
            self.assertEqual(brain.messages, [{"content": "unchanged"}])
            self.assertTrue(torch.equal(brain.liquid_state[:, :48], torch.arange(48).reshape(1, -1).float()))
            after = Path(directory) / "insertion"; after.mkdir()
            atomic_save_tensors_bounded(after / "core.safetensors", {"decoder." + key: value for key, value in brain.decoder.state_dict().items()})
            atomic_save_tensors_bounded(after / "plasticity.safetensors", {**{"router." + key: value for key, value in brain.router.state_dict().items()}, "state.liquid": brain.liquid_state})
            manifest = geometry_candidate_manifest(candidate_id=candidate_id, parent_metadata_sha256="a" * 64,
                parent_architecture_sha256=descriptor()["sha256"], root_architecture_sha256=descriptor()["sha256"],
                candidate_architecture_sha256=brain.config.native_architecture["sha256"], mutation=result["mutation"],
                old_geometry=result["oldGeometry"], new_geometry=result["newGeometry"],
                tensor_proofs=result["tensorProofs"], owner_inventory=result["ownerInventory"])
            self.assertTrue(verify_geometry_checkpoint_migration(before, after, manifest, chunk_bytes=32)["verified"])

    def test_late_preparation_failure_restores_original_refs_bytes_and_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, candidate_id = self.fixture(directory)
            originals = {key: getattr(brain, key) for key in (*ROOTS, "config", "core_pager", "liquid_state", "working_attention_pager", "_optimizer")}
            original_bytes = (brain.engine_path / "core.safetensors").read_bytes()
            pager = Pager()
            def fail(): raise RuntimeError("fixture late optimizer failure")
            brain._replace_optimizer = fail
            with self.assertRaisesRegex(RuntimeError, "late optimizer"):
                apply_isolated_geometry_candidate(brain, {"mutation": "resize-width", "dModel": 64}, candidate_id,
                    module_factory=lambda config, source: roots(config.d_model, source.router.children["population"]),
                    pager_factory=lambda *_: pager)
            for key, value in originals.items(): self.assertIs(getattr(brain, key), value)
            self.assertTrue(pager.closed); self.assertFalse(brain.core_pager.closed)
            self.assertEqual((brain.engine_path / "core.safetensors").read_bytes(), original_bytes)

    def test_cancellation_and_live_parent_paths_are_rejected_before_swap(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, candidate_id = self.fixture(directory)
            original = brain.config
            with self.assertRaises(InterruptedError):
                apply_isolated_geometry_candidate(brain, {"mutation": "resize-width", "dModel": 64}, candidate_id,
                    cancelled=lambda: True)
            self.assertIs(brain.config, original)
            brain.engine_path = Path(directory) / "live" / "engine"
            with self.assertRaises(PermissionError):
                apply_isolated_geometry_candidate(brain, {"mutation": "resize-width", "dModel": 64}, candidate_id)

    def test_restart_views_keep_original_durable_hashes_and_all_records(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy()
            replay = DurableReplayBuffer(Path(directory) / "replay.sqlite3", policy)
            replay.extend((torch.arange(4).float() + index for index in range(7)))
            checkpoint = replay.checkpoint()
            cold = PagedWorkingMemory(Path(directory) / "working.sqlite3", policy)
            page = cold.append(torch.arange(4).float(), {"assemblyId": "unchanged"})
            brain = SimpleNamespace(config=SimpleNamespace(idea_dim=6, native_architecture={"evolutionLineage": {
                "mutations": [{"mutation": "resize-width", "dModel": 6}]}}), replay=replay, paged_working_memory=cold)
            self.assertTrue(install_geometry_runtime_views(brain))
            values = list(brain.replay)
            self.assertEqual(len(values), 7)
            self.assertEqual(replay.checkpoint(), checkpoint)
            for index, value in enumerate(values):
                self.assertTrue(torch.equal(value[:4], torch.arange(4).float() + index))
                self.assertTrue(bool(value[4:].eq(0).all()))
            value, metadata = brain.paged_working_memory.read(page, touch=False)
            self.assertEqual(cold.count(), 1); self.assertEqual(metadata["assemblyId"], "unchanged")
            self.assertEqual(value.shape, (6,))

    def test_narrow_retention_penalizes_every_missing_parent_axis_without_mean_dilution(self):
        class Adapter:
            def eval(self): pass
            def __call__(self, values): return values
        with tempfile.TemporaryDirectory() as directory:
            anchors = save_retention_anchors(Path(directory) / "anchors.safetensors", [torch.tensor([1., 2., 3., 4.])],
                count=1, width=4, candidate_id="fixture")
            def brain(width): return SimpleNamespace(config=SimpleNamespace(idea_dim=width), device=torch.device("cpu"),
                decoder=SimpleNamespace(working_attention_pager=SimpleNamespace(device_tile_budget_bytes=65536)), idea_adapter=Adapter())
            self.assertEqual(NeuralEvolutionManager._latent_loss(brain(4), anchors, 2), 0)
            self.assertEqual(NeuralEvolutionManager._latent_loss(brain(2), anchors, 2), 25 / 4)
            self.assertEqual(NeuralEvolutionManager._latent_loss(brain(8), anchors, 4), 0)


if __name__ == "__main__": unittest.main()
