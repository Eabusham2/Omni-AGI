import json
import sqlite3
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

import torch
from safetensors.torch import save_file


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.model import BitLinear


class AdaptiveBrainTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-brain-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, **overrides):
        config = OmniConfig.micro(
            parallel_thoughts=1,
            max_seq_len=40,
            learn_from_own_messages=False,
            **overrides,
        )
        return AdaptiveBrain.create("brain-test", self.root, config)

    def test_parameter_only_ingest_mutates_weights_without_storing_source(self):
        brain = self.make_brain(
            memory_recipe="synapses-only",
            retain_source_text=False,
        )
        before = brain.parameter_checksum()
        source = "OmegaUniqueSyntax teaches adaptive widgets through blue lattices."
        result = brain.ingest(text=source, name="secret.txt", policy="encode")
        self.assertNotEqual(before, result["parameterChecksumAfter"])
        metadata_text = (brain.engine_path / "brain.json").read_text("utf-8")
        self.assertNotIn(source, metadata_text)
        self.assertFalse(result["source"]["raw_text_retained"])
        self.assertTrue((brain.engine_path / "core.safetensors").is_file())
        self.assertTrue((brain.engine_path / "plasticity.safetensors").is_file())
        self.assertTrue((brain.engine_path / "events.sqlite3").is_file())
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(reloaded.parameter_checksum(), result["parameterChecksumAfter"])
        self.assertEqual(len(reloaded.training_sources), 1)
        reloaded.events.close()

    def test_chat_trace_proves_learning_and_no_textual_retrieval(self):
        brain = self.make_brain()
        result = brain.chat("Hello adaptive brain", max_new_tokens=4, seed=4)
        self.assertTrue(result["text"])
        trace = result["trace"]
        for field in (
            "parameter_checksum_before",
            "parameter_checksum_after",
            "parameter_delta_norm",
            "stdp_update",
            "spike_rate",
            "liquid_controls",
            "recalled_idea_ids",
            "expert_route",
            "seed",
            "train_loss",
            "ponder_factors",
            "branches",
        ):
            self.assertIn(field, trace)
        self.assertFalse(trace["textual_memory_injected"])
        self.assertNotEqual(
            trace["parameter_checksum_before"],
            trace["parameter_checksum_after"],
        )
        self.assertEqual(len(trace["branches"]), 1)
        self.assertFalse(result["runtimeCard"]["hidden_behavioral_prompt"])
        brain.events.close()

    def test_candidate_exception_rolls_back_all_core_parameters(self):
        brain = self.make_brain()
        before = brain.parameter_checksum()

        def explode(*args, **kwargs):
            del args, kwargs
            with torch.no_grad():
                next(brain.decoder.parameters()).add_(100.0)
            raise RuntimeError("deliberate candidate failure")

        brain._optimize_experience = explode
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            brain.train(texts=["candidate rollback fixture"], epochs=1)
        self.assertEqual(before, brain.parameter_checksum())
        candidate_records = list(
            (brain.engine_path / "candidates").glob("*/candidate.json")
        )
        self.assertEqual(len(candidate_records), 1)
        record = json.loads(candidate_records[0].read_text("utf-8"))
        self.assertEqual(record["status"], "rejected")
        brain.events.close()

    def test_recipe_booleans_change_runtime_behavior(self):
        brain = self.make_brain(
            ternary_weights=False,
            spiking_dynamics=False,
            stdp_plasticity=False,
            liquid_dynamics=False,
            vector_symbolic_memory=False,
            online_learning=False,
            consolidation_enabled=False,
        )
        self.assertTrue(
            all(
                not module.ternary
                for root in brain._trainable_modules()
                for module in root.modules()
                if isinstance(module, BitLinear)
            )
        )
        learned = brain.learn_experience("transient dense fixture", steps=0)
        self.assertEqual(learned["spiking"]["spikes"], 0.0)
        self.assertEqual(len(brain.memory.ideas), 0)
        self.assertEqual(learned["training"]["loss"], 0.0)
        consolidation = brain.consolidate()
        self.assertTrue(consolidation["disabled"])
        card = brain.runtime_card()
        self.assertTrue(card["active_modules"]["dense"])
        self.assertFalse(card["active_modules"]["spiking"])
        brain.events.close()

    def test_wav_ingestion_trains_audio_parameters(self):
        brain = self.make_brain()
        wav_path = self.root / "tone.wav"
        samples = array("h", [int(12000 * ((index % 8) / 7.0 - 0.5)) for index in range(128)])
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(samples.tobytes())
        before = brain.parameter_checksum()
        result = brain.ingest(
            path=str(wav_path), kind="audio", policy="encode"
        )
        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertNotEqual(before, result["parameterChecksumAfter"])
        brain.events.close()

    def test_generated_artifacts_are_chromium_viewable_and_embedded(self):
        brain = self.make_brain()
        image = brain.generate_modality("image", prompt="blue geometry", seed=1)
        audio = brain.generate_modality("audio", prompt="short tone", seed=2)
        video = brain.generate_modality("video", prompt="moving light", seed=3)
        self.assertEqual(Path(image["path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(Path(audio["path"]).read_bytes()[:4], b"RIFF")
        self.assertEqual(Path(video["path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertTrue(image["dataUrl"].startswith("data:image/png;base64,"))
        self.assertTrue(audio["dataUrl"].startswith("data:audio/wav;base64,"))
        self.assertTrue(video["dataUrl"].startswith("data:image/apng;base64,"))
        brain.events.close()

    def test_snapshot_and_append_only_event_log(self):
        brain = self.make_brain()
        result = brain.snapshot("checkpoint")
        self.assertTrue(Path(result["path"], "core.safetensors").is_file())
        self.assertEqual(brain.events.integrity(), "ok")
        with self.assertRaises(sqlite3.DatabaseError):
            brain.events.connection.execute(
                "UPDATE events SET kind='changed' WHERE sequence=1"
            )
        brain.events.close()

    def test_hardware_profile_and_timescales_are_exposed(self):
        config = OmniConfig.from_external(
            {
                "name": "GPU brain",
                "hardwareTier": "gpu",
                "liquidMode": "ltc",
                "workingMemorySlots": 7,
                "shortTermHalfLifeMinutes": 12,
                "longTermThreshold": 0.75,
                "forgettingRate": 0.01,
                "parallelThoughts": 2,
            }
        )
        self.assertEqual(config.hardware_tier, "gpu")
        self.assertEqual(config.d_model, 96)
        self.assertEqual(config.liquid_mode, "ltc")
        self.assertEqual(config.working_memory_slots, 7)
        self.assertEqual(config.parallel_thoughts, 2)

    def test_working_memory_is_internal_bounded_and_persistent(self):
        parameter_config = OmniConfig.micro(
            name="parameter",
            seed=81,
            parallel_thoughts=1,
            max_seq_len=40,
            online_learning=False,
            learn_from_own_messages=False,
            memory_injection="parameter-only",
            working_memory_slots=2,
        )
        working_config = OmniConfig.micro(
            name="working",
            seed=81,
            parallel_thoughts=1,
            max_seq_len=40,
            online_learning=False,
            learn_from_own_messages=False,
            memory_injection="working-memory",
            working_memory_slots=2,
        )
        parameter = AdaptiveBrain.create(
            "parameter", self.root / "parameter", parameter_config
        )
        working = AdaptiveBrain.create(
            "working", self.root / "working", working_config
        )
        left = parameter.chat("Remember the cobalt route.", seed=5, max_new_tokens=3)
        right = working.chat("Remember the cobalt route.", seed=5, max_new_tokens=3)
        self.assertEqual(
            left["trace"]["prompt_token_ids_sha256"],
            right["trace"]["prompt_token_ids_sha256"],
        )
        self.assertFalse(left["trace"]["prompt_text_expanded"])
        self.assertFalse(right["trace"]["prompt_text_expanded"])
        self.assertEqual(left["trace"]["working_memory_channel"], "disabled")
        self.assertEqual(
            right["trace"]["working_memory_channel"], "recurrent-vector"
        )
        self.assertEqual(right["trace"]["working_memory_vectors"], 1)
        working.chat("Now follow it twice.", seed=6, max_new_tokens=3)
        working.chat("And a third time.", seed=7, max_new_tokens=3)
        self.assertEqual(len(working.working_memory), 2)
        working.save()
        parameter.events.close()
        working.events.close()
        reloaded = AdaptiveBrain.load(self.root / "working", "working")
        self.assertEqual(len(reloaded.working_memory), 2)
        self.assertIsNotNone(reloaded._working_memory_vector())
        reloaded.events.close()

    def test_slow_metaplastic_anchors_persist_and_penalize_drift(self):
        brain = self.make_brain(metaplasticity=True)
        brain.learn_experience("Stable amber knowledge should resist drift.")
        self.assertGreater(brain.counters["metaplastic_updates"], 0)
        nonzero = sum(
            int(value.gt(0).sum().item())
            for value in brain.slow_importance.values()
        )
        self.assertGreater(nonzero, 0)
        with torch.no_grad():
            parameter = next(brain.decoder.parameters())
            parameter.add_(0.25)
        penalty = float(brain._stability_penalty().item())
        self.assertGreater(penalty, 0.0)
        expected_importance = {
            key: value.clone() for key, value in brain.slow_importance.items()
        }
        brain.save()
        brain.events.close()
        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(set(reloaded.slow_importance), set(expected_importance))
        for key, expected in expected_importance.items():
            self.assertTrue(torch.equal(reloaded.slow_importance[key], expected))
        reloaded.events.close()

    def test_unbounded_sparse_memory_expands_until_resource_guard(self):
        brain = self.make_brain(
            growth_policy="unbounded",
            max_concepts=1,
            max_ideas=1,
            max_synapses=1,
            growth_novelty_threshold=2.0,
        )
        brain._resource_readings = lambda: {
            "diskFreeBytes": 8 * 1024**3,
            "availableMemoryBytes": 8 * 1024**3,
        }
        brain.learn_experience("alpha beta gamma", steps=0)
        brain.learn_experience("delta epsilon zeta", steps=0)
        self.assertGreater(brain.memory.max_concepts, 1)
        self.assertGreater(brain.memory.max_ideas, 1)
        self.assertGreater(brain.memory.max_relations, 1)
        self.assertGreater(brain.memory.capacity_expansions, 0)
        card = brain.runtime_card()
        self.assertGreater(card["growth"]["memoryCapacities"]["concepts"], 1)
        brain.events.close()

    def test_modality_pack_install_is_namespace_limited_and_transactional(self):
        brain = self.make_brain()
        pack_path = self.root / "vision.safetensors"
        vision = {
            "modalities."
            + key: (value.detach().cpu().clone() + 0.01).contiguous()
            for key, value in brain.modalities.state_dict().items()
            if key.startswith("vision.")
        }
        save_file(vision, str(pack_path))
        manifest = {
            "format": "omni-modality-pack",
            "formatVersion": 1,
            "architecture": "OmniCortex",
            "architectureSchemaVersion": 1,
            "pack": {
                "id": "vision-fixture",
                "name": "Vision fixture",
                "modalities": ["vision"],
            },
            "compatibility": {
                "dModel": brain.config.d_model,
                "modalityChannels": brain.config.modality_channels,
                "imageSize": brain.config.image_size,
                "audioSamples": brain.config.audio_samples,
                "videoFrames": brain.config.video_frames,
            },
            "licenseLedger": {
                "license": "MIT",
                "provenanceUrl": "https://example.invalid/vision-fixture",
            },
        }
        result = brain.install_modality_pack(pack_path, manifest)
        self.assertEqual(result["pack"]["id"], "vision-fixture")
        self.assertIn("vision", result["pack"]["modalities"])
        self.assertFalse(result["trace"]["code_executed"])
        before = brain.parameter_checksum()
        invalid_path = self.root / "invalid.safetensors"
        save_file(
            {
                "decoder.illegal": torch.ones(1),
            },
            str(invalid_path),
        )
        with self.assertRaisesRegex(ValueError, "outside modalities"):
            brain.install_modality_pack(invalid_path, manifest)
        self.assertEqual(before, brain.parameter_checksum())
        brain.events.close()


if __name__ == "__main__":
    unittest.main()
