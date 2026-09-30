"""Constructor-free packed migration/math/file proofs, never model training."""

import copy
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import torch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.architecture_migration import (
    assert_architecture_quiescent, copy_packed_prefix, copy_tensor_prefix,
    normalize_architecture_change, preserve_runtime_rng,
    reseal_native_descriptor, validate_compatible_architecture_lineage, verify_preserved_tensor_prefixes,
)
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded
from omni_core.evolution import NeuralEvolutionManager, _architecture_compatible, _architecture_signature, _bundle_checksum, _diff_checksum, _file_sha256
from omni_core.evolution_anchors import RetentionAnchorFile, save_retention_anchors
from omni_core.config import OmniConfig
from omni_core.distributed_seal import make_distributed_training_seal
from omni_core.model import OmniDecoder, pack_ternary_weight, unpack_ternary_weight_rows
from omni_core.native_architecture import native_architecture_sha256, native_core_inventory
from omni_core.spiking import AssociativeSpikingRouter, LIFPopulation, STDPSynapses


class CompatibleArchitectureMigrationFixtures(unittest.TestCase):
    def test_packed_prefix_keeps_exact_old_trits_with_nonbyte_aligned_dimensions(self):
        old = torch.tensor([[1, -1, 0, 1, -1], [0, 1, 1, -1, 0], [-1, 0, 1, 0, 1]], dtype=torch.int8)
        source = pack_ternary_weight(old)
        target = torch.empty((5, 3), dtype=torch.uint8)
        peak = copy_packed_prefix(source, target, 5, 9, chunk_bytes=8)
        levels = unpack_ternary_weight_rows(target, 9)
        self.assertTrue(torch.equal(levels[:3, :5], old))
        self.assertEqual(int(levels[:, 5:].abs().sum()), 0)
        self.assertEqual(int(levels[3:].abs().sum()), 0)
        self.assertLessEqual(peak, 8)

    def test_nonweight_timing_and_activity_prefixes_are_exact_not_averaged(self):
        for dtype in (torch.int16, torch.float32, torch.int64):
            source = torch.arange(15).reshape(3, 5).to(dtype)
            target = torch.zeros((7, 9), dtype=dtype)
            peak = copy_tensor_prefix(source, target, chunk_bytes=24)
            self.assertTrue(torch.equal(target[:3, :5], source))
            self.assertEqual(int(target[3:].abs().sum()), 0)
            self.assertLessEqual(peak, 24)

    def test_zero_residual_depth_and_dormant_router_region_preserve_math(self):
        torch.manual_seed(71)
        hidden = torch.randn(2, 3, 4)
        attention_values = torch.randn_like(hidden)
        feed_forward_values = torch.randn(2, 3, 7)
        appended = hidden + attention_values @ torch.zeros(4, 4)
        appended = appended + feed_forward_values @ torch.zeros(7, 4)
        self.assertTrue(torch.equal(hidden, appended))
        old_activity = torch.tensor([0.2, 0.7, 0.0])
        old_output = torch.randn(4, 3)
        expanded_output = torch.cat((old_output, torch.zeros(4, 2)), dim=1)
        expanded_activity = torch.cat((old_activity, torch.zeros(2)))
        self.assertTrue(torch.equal(old_output @ old_activity, expanded_output @ expanded_activity))
        self.assertEqual(float(old_activity.mean()), float(expanded_activity[:3].mean()))
        old_logits = torch.tensor([[0.2, -0.7]])
        old_residuals = torch.randn(1, 2, 3, 4)
        old_mix = (old_residuals * old_logits.softmax(-1)[:, :, None, None]).sum(1)
        expanded_logits = torch.cat((old_logits, torch.tensor([[0.9]])), dim=-1)
        routing = torch.cat((expanded_logits[:, :2].softmax(-1), expanded_logits[:, 2:].sigmoid()), dim=-1)
        expanded_residuals = torch.cat((old_residuals, torch.zeros(1, 1, 3, 4)), dim=1)
        self.assertTrue(torch.equal(old_mix, (expanded_residuals * routing[:, :, None, None]).sum(1)))

    def test_current_descriptor_reseals_shape_without_rewriting_origin_descriptor(self):
        shape = {"dModel": 48, "layers": 1, "feedForward": 96, "nHeads": 4,
                 "vsaDimensions": 64, "routerNeurons": 5, "modalityChannels": 4,
                 "imageSize": 16, "audioSamples": 256, "videoFrames": 4,
                 "workingMemoryItems": 32, "workspaceLatents": 8, "vocabSize": 261, "liquidMode": "cfc"}
        descriptor = {"format": "omni-main-selected-native-architecture", "formatVersion": 1,
                      "architecture": "OmniCortex", "externalPretrainedWeights": False,
                      "qualityEvidence": "unmeasured-native-quality-deferred", "hardwareTier": "micro",
                      "shape": shape, "inventory": native_core_inventory(shape), "sizing": {"selected": 1}}
        descriptor["sha256"] = native_architecture_sha256(descriptor)
        original = copy.deepcopy(descriptor)
        config = SimpleNamespace(native_architecture=descriptor, n_layers=3, router_neurons=5)
        evolved = reseal_native_descriptor(config, {"mutation": "grow-depth", "addLayers": 2})
        self.assertEqual(descriptor, original)
        self.assertEqual(evolved["shape"]["layers"], 3)
        self.assertEqual(evolved["shape"]["dModel"], 48)
        self.assertEqual(evolved["shape"]["nHeads"], 4)
        self.assertEqual(evolved["evolutionLineage"]["rootArchitectureSha256"], original["sha256"])
        self.assertFalse(evolved["evolutionLineage"]["qualityVerified"])
        validate_compatible_architecture_lineage(evolved, original)
        config.native_architecture = evolved
        config.router_neurons = 7
        expanded = reseal_native_descriptor(config, {"mutation": "grow-regions", "addRegions": 2, "neuronsPerRegion": 1})
        validate_compatible_architecture_lineage(expanded, original)
        self.assertEqual(expanded["evolutionLineage"]["parentArchitectureSha256"], evolved["sha256"])
        corrupted = copy.deepcopy(expanded)
        corrupted["evolutionLineage"]["parentArchitectureSha256"] = "0" * 64
        corrupted["sha256"] = native_architecture_sha256(corrupted)
        with self.assertRaisesRegex(ValueError, "parent chain"):
            validate_compatible_architecture_lineage(corrupted, original)

    def test_rng_quiescence_and_invalid_geometry_are_explicit(self):
        before = torch.get_rng_state().clone()
        with preserve_runtime_rng(torch.device("cpu")):
            torch.rand(50)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        with self.assertRaisesRegex(RuntimeError, "collective"):
            assert_architecture_quiescent(SimpleNamespace(_packed_collective_controller=SimpleNamespace(step_id=0)))
        with self.assertRaisesRegex(RuntimeError, "window"):
            assert_architecture_quiescent(SimpleNamespace(ingestion_checkpoints={"source": {"activeRecordWindowSha256": "a"}}))
        with self.assertRaisesRegex(RuntimeError, "outstanding"):
            assert_architecture_quiescent(SimpleNamespace(ingestion_checkpoints={"source": {"status": "active"}}))
        assert_architecture_quiescent(SimpleNamespace(ingestion_checkpoints={}, completed_ingestions=[{"status": "completed", "parameterChecksumAfter": "old"}]))
        for epoch in (0, 1):
            seal = make_distributed_training_seal(manifest_sha256="a" * 64, topology_sha256="b" * 64,
                training_policy_sha256="c" * 64, record_count=2, epochs=1,
                cursors=[{"rank": 0, "worldSize": 1, "epoch": epoch, "nextGlobalOrdinal": 0,
                    "ownedRecordsCompleted": epoch * 2, "optimizerStepsCompleted": 0, "manifestSha256": "a" * 64}],
                committed_record_stop=epoch * 2, global_steps=0)
            if epoch == 0:
                with self.assertRaisesRegex(RuntimeError, "unfinished distributed"):
                    assert_architecture_quiescent(SimpleNamespace(distributed_training_seal=seal))
            else:
                assert_architecture_quiescent(SimpleNamespace(distributed_training_seal=seal))
        for payload in ({"mutation": "grow-width", "dimensions": 512}, {"mutation": "grow-depth", "addLayers": True}, {"mutation": "grow-regions", "addRegions": 2}):
            with self.assertRaises(ValueError):
                normalize_architecture_change(payload)

    def test_retention_anchor_snapshot_and_evaluation_are_bounded_without_corpus_truncation(self):
        class IdentityAdapter:
            sizes = []
            def eval(self):
                pass
            def __call__(self, values):
                self.sizes.append(values.shape[0])
                return values
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.safetensors"
            anchors = save_retention_anchors(path, (torch.arange(4).float() + index for index in range(203)), count=203, width=4, candidate_id="fixture")
            self.assertEqual(anchors.shape, (203, 4))
            seen = 0
            for batch in anchors.batches(max_rows=7, byte_budget=80):
                self.assertLessEqual(batch.numel() * 4, 80)
                for row in batch:
                    self.assertTrue(torch.equal(row, torch.arange(4).float() + seen))
                    seen += 1
            self.assertEqual(seen, 203)
            adapter = IdentityAdapter()
            brain = SimpleNamespace(device=torch.device("cpu"), decoder=SimpleNamespace(working_attention_pager=SimpleNamespace(device_tile_budget_bytes=1024)), idea_adapter=adapter)
            self.assertEqual(NeuralEvolutionManager._latent_loss(brain, anchors), 0.0)
            self.assertEqual(sum(adapter.sizes), 203)
            self.assertLessEqual(max(adapter.sizes), 2)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "count"):
                save_retention_anchors(path, [torch.ones(4)], count=2, width=4, candidate_id="bad")
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, "geometry"):
                RetentionAnchorFile(path, expected_width=5)

    def test_invalid_expert_normalization_control_cannot_be_saved_config(self):
        for value in (-2, True, 1.5):
            config = OmniConfig.micro()
            config.expert_routing_baseline_count = value
            with self.assertRaisesRegex(ValueError, "expert routing"):
                config.validate()

    def test_float_synapse_archive_rejects_before_mutating_existing_packed_bytes(self):
        synapses = STDPSynapses.__new__(STDPSynapses)
        torch.nn.Module.__init__(synapses)
        synapses.register_buffer("_packed_weights", torch.tensor([[0x55]], dtype=torch.uint8))
        before = synapses._packed_weights.clone()
        payload = {"weights": torch.tensor([[0.6]])}
        with self.assertRaisesRegex(RuntimeError, "floating learned synapse weights"):
            synapses.load_state_dict(payload)
        self.assertTrue(torch.equal(before, synapses._packed_weights))
        self.assertEqual(set(payload), {"weights"})

    def test_synapse_authority_validates_only_bounded_packed_bytes_not_a_dense_shadow(self):
        synapses = STDPSynapses.__new__(STDPSynapses)
        torch.nn.Module.__init__(synapses)
        synapses.pre_neurons, synapses.post_neurons = 5, 3
        synapses.register_buffer("_packed_weights", torch.full((3, 2), 0x55, dtype=torch.uint8))
        with patch("omni_core.spiking.unpack_ternary_weight_rows", side_effect=AssertionError("no dense validation shadow")):
            with patch("omni_core.spiking.TRANSFER_BYTES", 16):
                self.assertIs(synapses.authoritative_packed_tensors()[0], synapses._packed_weights)
                synapses._packed_weights[0, 0] = 0xFF
                with self.assertRaisesRegex(ValueError, "reserved"):
                    synapses._validate_packed()
                synapses._packed_weights[0, 0] = 0x55
                synapses._packed_weights[1, 1] = 0x45
                with self.assertRaisesRegex(ValueError, "padding"):
                    synapses._validate_packed()

    def test_promotion_binds_parent_and_candidate_metadata_before_any_overlay_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            live, candidate_dir = Path(directory) / "live", Path(directory) / "candidate"
            model = candidate_dir / "model"
            candidate = model / "engine"
            live.mkdir(); candidate.mkdir(parents=True)
            for path in (live, candidate):
                atomic_save_tensors_bounded(path / "core.safetensors", {"decoder.test": torch.tensor([0x55], dtype=torch.uint8)})
                atomic_save_tensors_bounded(path / "plasticity.safetensors", {"router.test": torch.tensor([0x55], dtype=torch.uint8)})
                (path / "brain.json").write_text(json.dumps({"config": {"n_layers": 1, "router_neurons": 3}, "expert_count": 0}))
            original = (live / "brain.json").read_bytes()
            record = {"status": "evaluated", "parentParameterChecksum": "fixed",
                "parentStateChecksum": _bundle_checksum(live), "parentMetadataSha256": _file_sha256(live / "brain.json"),
                "candidateStateChecksum": _bundle_checksum(candidate), "candidateMetadataSha256": _file_sha256(candidate / "brain.json"),
                "evaluation": {"passed": True, "architecture": _architecture_signature(candidate)}}
            manager = NeuralEvolutionManager.__new__(NeuralEvolutionManager)
            manager.engine_path = live
            manager.brain = SimpleNamespace(parameter_checksum=lambda: "fixed", _record_candidate=lambda *a, **kw: None)
            manager._record = lambda _: (candidate_dir, record)
            manager._model_path = lambda _: model
            (live / "brain.json").write_text(json.dumps({"config": {"n_layers": 1, "router_neurons": 3}, "freshContext": "changed"}))
            with self.assertRaisesRegex(ValueError, "baseline is stale"):
                manager.promote("fixture")
            (live / "brain.json").write_bytes(original)
            (candidate / "brain.json").write_text(json.dumps({"config": {"n_layers": 1, "router_neurons": 3}, "lineage": "altered"}))
            with self.assertRaisesRegex(ValueError, "metadata/context/lineage"):
                manager.promote("fixture")
            self.assertEqual((live / "brain.json").read_bytes(), original)

    def test_actual_expert_application_preserves_old_pool_and_gradients_with_zero_added_map(self):
        class FixedRoute(torch.nn.Module):
            def __init__(self, logit):
                super().__init__(); self.logit = logit
            def forward(self, values):
                return values.new_full((values.shape[0], 1), self.logit)
        class FixedResidual(torch.nn.Module):
            def __init__(self, scale):
                super().__init__(); self.scale = scale
            def forward(self, values):
                return values * self.scale
        decoder = OmniDecoder.__new__(OmniDecoder)
        torch.nn.Module.__init__(decoder)
        decoder.config = SimpleNamespace(expert_routing_baseline_count=-1)
        decoder.experts = torch.nn.ModuleList([FixedResidual(0.25), FixedResidual(-0.5)])
        decoder.expert_prototypes = torch.nn.ModuleList([FixedRoute(0.2), FixedRoute(-0.7)])
        hidden = torch.randn(1, 3, 4, requires_grad=True)
        before, _ = decoder._apply_experts(hidden)
        old_gradient = torch.autograd.grad(before.sum(), hidden)[0]
        decoder.config.expert_routing_baseline_count = 2
        decoder.experts.append(FixedResidual(0.0))
        decoder.expert_prototypes.append(FixedRoute(0.9))
        after, _ = decoder._apply_experts(hidden)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(old_gradient, torch.autograd.grad(after.sum(), hidden)[0]))

    def test_actual_router_route_preserves_old_activity_and_metrics_in_dormant_region(self):
        def router(size):
            value = AssociativeSpikingRouter.__new__(AssociativeSpikingRouter)
            torch.nn.Module.__init__(value)
            value.idea_dim, value.neurons = 2, size
            value.register_buffer("active_prefix_neurons", torch.tensor(3))
            value.register_buffer("region_ends", torch.tensor([3] if size == 3 else [3, size]))
            population = SimpleNamespace(neurons=size, leak=0.88, threshold=0.55, membrane=torch.zeros(size), spike_count=torch.zeros(size))
            population.membrane[:3] = torch.tensor([0.1, 0.2, 0.3])
            population.step = lambda current, threshold_offset=0.0: LIFPopulation.step(population, current, threshold_offset)
            value.population = population
            recurrent = torch.zeros(size, size); recurrent[:3, :3] = torch.tensor([[0., 1., -1.], [1., 0., 0.], [-1., 1., 0.]])
            stability = torch.zeros(size, size); stability[:3, :3] = torch.arange(9).reshape(3, 3).float()
            value.synapses = SimpleNamespace(stability=stability, effective_weight=lambda: recurrent)
            value.input_projection = lambda idea: torch.cat((idea.new_tensor([[0.1, -0.2, 0.3]]), idea.new_zeros(1, size - 3)), dim=-1)
            value.output_projection = lambda activity: torch.stack((activity[:, :3].sum(-1), activity[:, 1:3].sum(-1)), dim=-1)
            return value
        original, expanded = router(3), router(5)
        original.validate_bounded_state_load(); expanded.validate_bounded_state_load()
        idea = torch.tensor([0.1, -0.2])
        before, before_metrics = original.route(idea, steps=4, learn=False)
        after, after_metrics = expanded.route(idea, steps=4, learn=False)
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(before_metrics, after_metrics)
        self.assertTrue(torch.equal(original.population.membrane, expanded.population.membrane[:3]))
        self.assertEqual(float(expanded.population.membrane[3:].abs().sum()), 0.0)
        self.assertEqual(set(expanded.bounded_optional_state_defaults({}, "router.")), {"active_prefix_neurons", "region_ends"})
        expanded.region_ends = torch.tensor([-1, 5])
        with self.assertRaisesRegex(ValueError, "region"):
            expanded.validate_bounded_state_load()

    def test_bounded_file_prefix_proof_and_evolution_signatures(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / "base", Path(directory) / "new"
            baseline.mkdir(); candidate.mkdir()
            core = {"decoder.blocks.0.attention.output._packed_forward_weight": torch.tensor([[0x55, 0x55]], dtype=torch.uint8)}
            expanded = {**core, "decoder.blocks.1.attention.output._packed_forward_weight": torch.tensor([[0x55, 0x55]], dtype=torch.uint8)}
            plastic = {"router.active_prefix_neurons": torch.tensor(3), "router.region_ends": torch.tensor([3]), "state.working_memory": torch.arange(8).reshape(2, 4).float()}
            for path, tensors, layers in ((baseline, core, 1), (candidate, expanded, 2)):
                atomic_save_tensors_bounded(path / "core.safetensors", tensors, chunk_bytes=8)
                atomic_save_tensors_bounded(path / "plasticity.safetensors", plastic, chunk_bytes=8)
                (path / "brain.json").write_text(json.dumps({"config": {"n_layers": layers, "router_neurons": 3, "d_model": 4, "d_ff": 7, "n_heads": 1, "idea_dim": 4, "vsa_dim": 16}, "expert_count": 0}))
            proof = verify_preserved_tensor_prefixes(baseline / "core.safetensors", candidate / "core.safetensors", chunk_bytes=1)
            self.assertTrue(proof["verified"])
            self.assertLessEqual(proof["peakTransferBytes"], 1)
            self.assertTrue(_architecture_compatible(_architecture_signature(baseline), _architecture_signature(candidate), {"mutation": "grow-depth", "addLayers": 1}))
            changed_geometry = copy.deepcopy(_architecture_signature(candidate))
            changed_geometry["protectedGeometry"]["dropout"] = 0.5
            self.assertFalse(_architecture_compatible(_architecture_signature(baseline), changed_geometry, {"mutation": "grow-depth", "addLayers": 1}))
            self.assertNotEqual(_bundle_checksum(baseline), _bundle_checksum(candidate))
            digest, norm = _diff_checksum(baseline, candidate)
            self.assertEqual(len(digest), 64)
            self.assertGreater(norm, 0)
            expanded["decoder.blocks.0.attention.output._packed_forward_weight"] = torch.tensor([[0, 0x55]], dtype=torch.uint8)
            atomic_save_tensors_bounded(candidate / "core.safetensors", expanded)
            with self.assertRaisesRegex(ValueError, "changed learned"):
                verify_preserved_tensor_prefixes(baseline / "core.safetensors", candidate / "core.safetensors")


if __name__ == "__main__":
    unittest.main()
