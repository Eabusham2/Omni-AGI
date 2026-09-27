import sys
import tempfile
import unittest
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.config import OmniConfig
from omni_core.modalities import (
    ModalityHub,
    TernaryLatentTransformer,
    TernaryTransformerBlock,
)
from omni_core.model import PACKED_AUTHORITATIVE_PROJECTION_TYPES
from omni_core.optimizers import adamw_for_remaining_parameters
from omni_core.persistence import atomic_write_json, copy_substrate_snapshot
from omni_core.vsa import (
    ConceptMemory,
    HypervectorSpace,
    SubstrateResourcePause,
)


class MemoryAndModalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)

    def test_modality_transformer_norms_have_no_float_affine_state(self):
        block = TernaryTransformerBlock(8)
        latent = TernaryLatentTransformer(8, 8, max_tokens=4, layers=1)
        for norm in (
            block.norm_attention,
            block.norm_feed_forward,
            latent.output_norm,
            latent.blocks[0].norm_attention,
            latent.blocks[0].norm_feed_forward,
        ):
            self.assertFalse(norm.elementwise_affine)
            self.assertEqual(tuple(norm.parameters()), ())
        self.assertFalse(
            any("norm_attention.weight" in name or "output_norm.weight" in name
                for name in latent.state_dict())
        )

    def test_vsa_binding_unbinding_and_similar_recall(self):
        space = HypervectorSpace(128, seed=2)
        left = space.symbol("left")
        right = space.symbol("right")
        bound = space.bind(left, right)
        recovered = space.bind(bound, right)
        self.assertGreater(space.similarity(left, recovered), 0.99)
        sequenced = space.permute(left, steps=7)
        unsequenced = space.inverse_permute(sequenced, steps=7)
        self.assertTrue(torch.equal(left, unsequenced))
        self.assertLess(space.similarity(left, sequenced), 0.5)

        memory = ConceptMemory(128, seed=2)
        learned = memory.learn(
            "liquid neurons adapt through continuous time",
            retain_source_text=False,
        )
        cue = memory.vector_for_text("liquid neurons")
        _, recalled = memory.recall_vector(cue)
        self.assertTrue(recalled)
        self.assertNotIn("source_text", memory.ideas[0])
        self.assertIn(learned["idea_id"], memory.idea_vectors)

    def test_relation_growth_has_no_cardinality_limit(self):
        memory = ConceptMemory(64)
        memory.learn("alpha beta gamma delta epsilon")
        self.assertGreater(len(memory.relations), 2)
        self.assertIsNone(memory.metadata()["cardinality_limit"])

    def test_sensory_vector_becomes_an_authoritative_cross_modal_assembly(self):
        memory = ConceptMemory(64, seed=21)
        sensory = torch.linspace(-1.0, 1.0, 64)
        learned = memory.learn_vector(
            sensory,
            fingerprint="a" * 64 + ":window:1",
            kind="audio",
            source="media",
            source_label="harbor bell recording.wav",
            importance=0.8,
        )

        assembly_id = learned["assembly_id"]
        self.assertIn(assembly_id, memory.assembly_vectors)
        self.assertEqual(memory.neurons[assembly_id]["region"], "assembly")
        self.assertIn("audio-perception", learned["labels"])
        self.assertIn("harbor", learned["labels"])
        self.assertNotIn("source_text", next(
            item for item in memory.assemblies if item["id"] == assembly_id
        ))
        self.assertTrue(
            any(
                edge["source_id"] == assembly_id
                and edge["kind"] == "contains"
                for edge in memory.synapses.values()
            )
        )
        self.assertTrue(
            torch.allclose(
                memory.assembly_vectors[assembly_id].norm(),
                torch.tensor(1.0),
            )
        )

    def test_decay_never_cardinality_evicts_long_term_substrate(self):
        memory = ConceptMemory(64, seed=17)
        memory.learn(
            "persistent assemblies connect repeated experience through stable synapses"
        )
        neuron_ids = set(memory.neurons)
        assembly_ids = {
            str(record["id"]) for record in memory.assemblies
        }
        synapse_ids = set(memory.synapses)
        neuron_vectors = set(memory.neuron_vectors)
        assembly_vectors = set(memory.assembly_vectors)
        before_activation = {
            key: float(value["activation"])
            for key, value in memory.neurons.items()
        }
        before_strength = {
            key: abs(int(value["effective_weight"]))
            for key, value in memory.synapses.items()
        }

        # Even extreme repeated decay only changes activity and plastic
        # strength. Stable-v1 never deletes a learned neuron, distributed
        # assembly, synapse record, or vector because it became quiet.
        for _ in range(8):
            memory.decay(1.0)

        self.assertEqual(set(memory.neurons), neuron_ids)
        self.assertEqual(
            {str(record["id"]) for record in memory.assemblies}, assembly_ids
        )
        self.assertEqual(set(memory.synapses), synapse_ids)
        self.assertEqual(set(memory.neuron_vectors), neuron_vectors)
        self.assertEqual(set(memory.assembly_vectors), assembly_vectors)
        self.assertTrue(
            any(
                float(memory.neurons[key]["activation"])
                < before_activation[key]
                for key in neuron_ids
            )
        )
        self.assertTrue(
            all(
                abs(int(memory.synapses[key]["effective_weight"]))
                <= before_strength[key]
                for key in synapse_ids
            )
        )
        self.assertTrue(
            all(
                int(record["effective_weight"]) in {-1, 0, 1}
                for record in memory.synapses.values()
            )
        )

    def test_spreading_activation_is_not_limited_to_four_hops(self):
        memory = ConceptMemory(16, seed=9)
        for index in range(7):
            assembly_id = "assembly-%d" % index
            vector = torch.zeros(16)
            vector[index] = 1.0
            memory.assemblies.append(
                {
                    "id": assembly_id,
                    "neuron_ids": [],
                    "importance": 0.5,
                    "rehearsals": 1,
                }
            )
            memory.assembly_vectors[assembly_id] = vector
            memory.neurons[assembly_id] = {
                "id": assembly_id,
                "label": assembly_id,
                "region": "assembly",
                "activation": 0.0,
                "importance": 0.5,
                "uncertainty": 0.5,
                "exposures": 1,
                "last_activated_at": 0.0,
            }
            if index:
                source = "assembly-%d" % (index - 1)
                synapse_id = "%s>%s:test" % (source, assembly_id)
                memory.synapses[synapse_id] = {
                    "id": synapse_id,
                    "source_id": source,
                    "target_id": assembly_id,
                    "effective_weight": 1,
                }

        _, recalled = memory.recall_vector(
            memory.assembly_vectors["assembly-0"],
            workspace_slots=16,
        )
        recalled_ids = {item["assembly_id"] for item in recalled}
        self.assertIn("assembly-6", recalled_ids)
        self.assertGreater(memory._last_recall_rounds, 4)

    @staticmethod
    def _add_test_assembly(memory, assembly_id, vector):
        memory.assemblies.append(
            {
                "id": assembly_id,
                "fingerprint": assembly_id,
                "neuron_ids": [],
                "importance": 0.5,
                "rehearsals": 1,
            }
        )
        memory.assembly_vectors[assembly_id] = vector
        memory.neurons[assembly_id] = {
            "id": assembly_id,
            "label": assembly_id,
            "region": "assembly",
            "activation": 0.0,
            "importance": 0.5,
            "uncertainty": 0.5,
            "exposures": 1,
            "last_activated_at": 0.0,
        }
        memory.neuron_vectors[assembly_id] = vector

    def test_live_spreading_uses_exact_ternary_weight_not_timing_trace(self):
        memory = ConceptMemory(16, seed=3)
        source = torch.zeros(16)
        source[0] = 1.0
        target = torch.zeros(16)
        target[1] = 1.0
        self._add_test_assembly(memory, "source", source)
        self._add_test_assembly(memory, "target", target)
        memory.synapses["source>target:test"] = {
            "id": "source>target:test",
            "source_id": "source",
            "target_id": "target",
            "kind": "test",
            "effective_weight": 1,
            "eligibility": 0.0,
            "plasticity": 1.0,
            "uses": 1,
            "stability": 0.0,
            "last_updated_at": 0.0,
        }
        memory.recall_vector(source, workspace_slots=16)
        first = memory._last_recall_audit["activationByAssembly"]["target"]
        memory.synapses["source>target:test"]["eligibility"] = 0.8
        memory.recall_vector(source, workspace_slots=16)
        second = memory._last_recall_audit["activationByAssembly"]["target"]
        self.assertAlmostEqual(first, 0.52, places=7)
        self.assertEqual(first, second)
        self.assertTrue(memory._last_recall_audit["exactTernaryContribution"])
        self.assertFalse(memory._last_recall_audit["latentMagnitudeUsed"])

    def test_fractional_live_synapse_fails_before_recall_or_persistence(self):
        memory = ConceptMemory(16, seed=3)
        source = torch.zeros(16)
        source[0] = 1.0
        target = torch.zeros(16)
        target[1] = 1.0
        self._add_test_assembly(memory, "source", source)
        self._add_test_assembly(memory, "target", target)
        memory.synapses["source>target:test"] = {
            "id": "source>target:test",
            "source_id": "source",
            "target_id": "target",
            "kind": "test",
            "effective_weight": 0.5,
            "eligibility": 0.0,
            "plasticity": 1.0,
            "uses": 1,
            "stability": 0.0,
            "last_updated_at": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "exact ternary"):
            memory.recall_vector(source, workspace_slots=16)
        with tempfile.TemporaryDirectory(prefix="omni-substrate-") as folder:
            store = Path(folder) / "substrate"
            with self.assertRaisesRegex(ValueError, "exact ternary"):
                memory.save_sharded(store)
            self.assertFalse(store.exists())

    def test_inhibitory_synapse_competes_and_suppresses_positive_seed(self):
        memory = ConceptMemory(16, seed=4)
        source = torch.zeros(16)
        source[0] = 1.0
        target = torch.zeros(16)
        target[0] = 0.5
        target[1] = 0.8660254
        self._add_test_assembly(memory, "source", source)
        self._add_test_assembly(memory, "target", target)
        memory.synapses["source>target:inhibits"] = {
            "id": "source>target:inhibits",
            "source_id": "source",
            "target_id": "target",
            "kind": "inhibits",
            "effective_weight": -1,
            "eligibility": 0.0,
            "plasticity": 1.0,
            "uses": 1,
            "stability": 0.0,
            "last_updated_at": 0.0,
        }
        _signal, recalled = memory.recall_vector(source, workspace_slots=16)
        self.assertNotIn(
            "target", {item["assembly_id"] for item in recalled}
        )
        audit = memory._last_recall_audit
        self.assertEqual(audit["inhibitoryEdges"], 1)
        self.assertGreater(audit["inhibitorySignals"], 0)
        self.assertEqual(audit["suppressedAssemblies"], 1)
        self.assertLess(audit["activationByAssembly"]["target"], 0.0)

    def test_online_decay_can_be_scoped_to_causally_related_synapses(self):
        memory = ConceptMemory(16, seed=4)
        for record_id in ("a", "b", "c"):
            self._add_test_assembly(
                memory, record_id, torch.nn.functional.one_hot(
                    torch.tensor(ord(record_id) - ord("a")),
                    num_classes=16,
                ).float()
            )

        def synapse(record_id, source, target):
            return {
                "id": record_id,
                "source_id": source,
                "target_id": target,
                "kind": "test",
                "effective_weight": 1,
                "eligibility": 0.8,
                "plasticity": 1.0,
                "uses": 1,
                "stability": 0.4,
                "last_updated_at": 1.0,
            }

        related = synapse("a>b:test", "a", "b")
        unrelated = synapse("b>c:test", "b", "c")
        memory.synapses = {
            related["id"]: related,
            unrelated["id"]: unrelated,
        }
        unrelated_before = dict(unrelated)

        memory.decay(0.1, synapses=[related])

        self.assertEqual(related["effective_weight"], 1)
        self.assertLess(related["eligibility"], 0.8)
        self.assertEqual(unrelated, unrelated_before)

    def test_sharded_substrate_roundtrip_reuses_unchanged_blobs(self):
        memory = ConceptMemory(32, seed=8)
        memory.learn(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        )
        with tempfile.TemporaryDirectory(prefix="omni-substrate-") as folder:
            store = Path(folder) / "substrate"
            first = memory.save_sharded(store, records_per_shard=2)
            self.assertGreater(first["shardCount"], 3)
            generation = json.loads(
                (store / first["generationManifest"]).read_text("utf-8")
            )
            self.assertEqual(
                {item["kind"] for item in generation["shards"]},
                {"neurons", "assemblies", "synapses"},
            )
            self.assertGreater(
                sum(
                    1
                    for item in generation["shards"]
                    if item.get("tensors")
                ),
                1,
            )
            old_blobs = {
                spec["path"]: (store / spec["path"]).stat().st_mtime_ns
                for item in generation["shards"]
                for spec in (item.get("records"), item.get("tensors"))
                if spec
            }

            compact = memory.metadata(include_records=False)
            restored = ConceptMemory.load_sharded(store, compact)
            self.assertEqual(memory.neurons, restored.neurons)
            self.assertEqual(memory.assemblies, restored.assemblies)
            self.assertEqual(memory.synapses, restored.synapses)
            self.assertEqual(set(memory.neuron_vectors), set(restored.neuron_vectors))
            self.assertEqual(
                set(memory.assembly_vectors), set(restored.assembly_vectors)
            )
            for key, expected in memory.neuron_vectors.items():
                self.assertTrue(torch.equal(expected, restored.neuron_vectors[key]))
            for key, expected in memory.assembly_vectors.items():
                self.assertTrue(
                    torch.equal(expected, restored.assembly_vectors[key])
                )

            second = memory.save_sharded(store, records_per_shard=2)
            self.assertEqual(first["activeGeneration"], second["activeGeneration"])
            for relative, modified_at in old_blobs.items():
                self.assertEqual(
                    (store / relative).stat().st_mtime_ns,
                    modified_at,
                )
            memory._ensure_neuron("new-growth-only", 42.0)
            third = memory.save_sharded(store, records_per_shard=2)
            self.assertNotEqual(
                second["activeGeneration"], third["activeGeneration"]
            )
            third_generation = json.loads(
                (store / third["generationManifest"]).read_text("utf-8")
            )
            old_shards = {
                (item["kind"], item["bucket"], item["part"]): item
                for item in generation["shards"]
            }
            new_shards = {
                (item["kind"], item["bucket"], item["part"]): item
                for item in third_generation["shards"]
            }
            changed_existing = [
                key
                for key in set(old_shards).intersection(new_shards)
                if old_shards[key] != new_shards[key]
            ]
            self.assertLessEqual(len(changed_existing), 1)
            self.assertTrue(
                all(key[0] == "neurons" for key in changed_existing)
            )
            third_blobs = {
                spec["path"]
                for item in third_generation["shards"]
                for spec in (item.get("records"), item.get("tensors"))
                if spec
            }
            reused = set(old_blobs).intersection(third_blobs)
            self.assertGreater(len(reused), len(old_blobs) // 2)
            for relative in reused:
                self.assertEqual(
                    (store / relative).stat().st_mtime_ns,
                    old_blobs[relative],
                )
            plastic_names = set(memory.tensor_state())
            self.assertEqual(
                plastic_names,
                {
                    "substrate.vectors.packed_rows",
                    "substrate.vectors.update_counters_le",
                },
            )

    def test_shard_promotion_pauses_before_resource_reserve_is_crossed(self):
        memory = ConceptMemory(16)
        memory.learn("reserve guarded substrate")
        memory.growth_guard = lambda _estimated: False
        with tempfile.TemporaryDirectory(prefix="omni-substrate-pause-") as folder:
            store = Path(folder) / "substrate"
            with self.assertRaises(SubstrateResourcePause):
                memory.save_sharded(store, records_per_shard=1)
            self.assertFalse((store / "manifest.json").exists())
            self.assertGreater(memory.growth_pauses, 0)

    def test_snapshot_copy_rejects_a_corrupt_referenced_shard(self):
        memory = ConceptMemory(16)
        memory.learn("checksummed snapshot shard")
        with tempfile.TemporaryDirectory(prefix="omni-substrate-copy-") as folder:
            root = Path(folder)
            source = root / "source"
            destination = root / "destination"
            memory.save_sharded(source / "substrate", records_per_shard=1)
            atomic_write_json(
                source / "brain.json",
                {"substrate": memory.metadata(include_records=False)},
            )
            generation = json.loads(
                (
                    source
                    / "substrate"
                    / memory.persistence_manifest["generationManifest"]
                ).read_text("utf-8")
            )
            blob_relative = generation["shards"][0]["records"]["path"]
            blob = source / "substrate" / blob_relative
            blob.write_bytes(blob.read_bytes() + b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                copy_substrate_snapshot(source, destination)
            self.assertFalse(
                (destination / "substrate" / "manifest.json").exists()
            )

    def test_all_modality_baselines_forward_generate_and_backpropagate(self):
        config = OmniConfig.micro()
        hub = ModalityHub(config)
        idea = torch.randn(1, config.idea_dim)

        image = torch.randn(1, 3, config.image_size, config.image_size).clamp(-1, 1)
        image_result = hub.image(image, idea)
        self.assertEqual(tuple(image_result["reconstruction"].shape), tuple(image.shape))
        self.assertEqual(tuple(image_result["embedding"].shape), tuple(idea.shape))
        self.assertGreater(float(image_result["diffusion_loss"].item()), 0.0)

        audio = torch.randn(1, 1, config.audio_samples).clamp(-1, 1)
        audio_result = hub.audio(audio, idea)
        self.assertEqual(tuple(audio_result["reconstruction"].shape), tuple(audio.shape))

        video = torch.randn(
            1,
            3,
            config.video_frames,
            config.image_size,
            config.image_size,
        ).clamp(-1, 1)
        video_result = hub.video(video, idea)
        self.assertEqual(tuple(video_result["reconstruction"].shape), tuple(video.shape))
        self.assertGreater(float(video_result["diffusion_loss"].item()), 0.0)

        loss = (
            image_result["loss"] + audio_result["loss"] + video_result["loss"]
        )
        packed = [
            module
            for module in hub.modules()
            if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES)
        ]
        before = [
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors())
            for module in packed
        ]
        loss.backward()
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for module, snapshot in zip(packed, before)
                for previous, current in zip(
                    snapshot, module.authoritative_packed_tensors()
                )
            )
        )
        for kind, expected in (
            ("image", (1, 3, config.image_size, config.image_size)),
            ("audio", (1, config.audio_samples)),
            (
                "video",
                (
                    1,
                    3,
                    config.video_frames,
                    config.image_size,
                    config.image_size,
                ),
            ),
        ):
            previews = []
            generated = hub.generate(
                kind,
                idea,
                seed=1,
                preview_callback=lambda progress, value: previews.append(
                    (progress, tuple(value.shape))
                ),
            )
            self.assertEqual(tuple(generated.shape), expected)
            self.assertTrue(torch.isfinite(generated).all())
            self.assertGreaterEqual(len(previews), 3)
            self.assertEqual(previews[-1][0], 1.0)
            if kind == "image":
                self.assertTrue(
                    all(shape == expected for _progress, shape in previews)
                )
            elif kind == "audio":
                sample_counts = [shape[-1] for _progress, shape in previews]
                self.assertEqual(previews[-1][1], expected)
                self.assertEqual(sample_counts, sorted(sample_counts))
                self.assertLess(sample_counts[0], sample_counts[-1])
            else:
                frame_counts = [shape[2] for _progress, shape in previews]
                self.assertEqual(previews[-1][1], expected)
                self.assertEqual(frame_counts, sorted(frame_counts))
                self.assertLess(frame_counts[0], frame_counts[-1])
                self.assertTrue(
                    all(
                        shape[-2:] == expected[-2:]
                        for _progress, shape in previews
                    )
                )

    def test_each_modality_overfits_a_fixture_and_safe_reload_is_exact(self):
        config = OmniConfig.micro(learning_rate=0.01)
        hub = ModalityHub(config)
        idea = torch.randn(1, config.idea_dim)
        fixtures = {
            "image": torch.linspace(
                -1, 1, 3 * config.image_size * config.image_size
            ).reshape(1, 3, config.image_size, config.image_size),
            "audio": torch.sin(
                torch.linspace(0, 8, config.audio_samples)
            ).reshape(1, 1, config.audio_samples),
            "video": torch.linspace(
                -1,
                1,
                3
                * config.video_frames
                * config.image_size
                * config.image_size,
            ).reshape(
                1,
                3,
                config.video_frames,
                config.image_size,
                config.image_size,
            ),
        }
        for name, target in fixtures.items():
            module = getattr(hub, name)
            optimizer = adamw_for_remaining_parameters(
                module.parameters(), lr=0.01, weight_decay=0.01
            )
            packed = [
                projection
                for projection in module.modules()
                if isinstance(projection, PACKED_AUTHORITATIVE_PROJECTION_TYPES)
            ]
            before = [
                tuple(tensor.clone() for tensor in projection.authoritative_packed_tensors())
                for projection in packed
            ]
            with torch.no_grad():
                initial = float(module(target, idea)["loss"].item())
            for _ in range(18):
                optimizer.zero_grad(set_to_none=True)
                loss = module(target, idea)["loss"]
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                final = float(module(target, idea)["loss"].item())
            self.assertTrue(
                any(
                    not torch.equal(previous, current)
                    for projection, snapshot in zip(packed, before)
                    for previous, current in zip(
                        snapshot, projection.authoritative_packed_tensors()
                    )
                ),
                msg=name,
            )
            self.assertLess(final, initial, msg=name)

        with tempfile.TemporaryDirectory(prefix="omni-modalities-") as folder:
            path = Path(folder) / "modalities.safetensors"
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in hub.state_dict().items()
                },
                str(path),
            )
            reloaded = ModalityHub(config)
            reloaded.load_state_dict(load_file(str(path)))
            for name in ("image", "audio", "video"):
                expected = hub.generate(name, idea, seed=91)
                actual = reloaded.generate(name, idea, seed=91)
                self.assertTrue(torch.equal(expected, actual), msg=name)


if __name__ == "__main__":
    unittest.main()
