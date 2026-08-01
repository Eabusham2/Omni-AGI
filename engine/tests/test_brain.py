import io
import json
import sqlite3
import sys
import tarfile
import tempfile
import types
import unittest
import wave
from array import array
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file, save_file


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.model import BitLinear
from omni_core.starter import STARTER_CORPUS
from omni_core.ternary_packing import verify_ternary_shards
from omni_core.vsa import ConceptMemory, SubstrateResourcePause


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
        self.assertFalse(
            any(
                name.startswith("substrate.")
                for name in load_file(
                    str(brain.engine_path / "plasticity.safetensors")
                )
            )
        )
        engine_metadata = json.loads(
            (brain.engine_path / "brain.json").read_text("utf-8")
        )
        self.assertNotIn("neurons", engine_metadata["substrate"])
        self.assertGreater(
            engine_metadata["substrate"]["persistence"]["shardCount"],
            1,
        )
        self.assertTrue(
            (brain.engine_path / "substrate" / "manifest.json").is_file()
        )
        self.assertTrue((brain.engine_path / "events.sqlite3").is_file())
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(reloaded.parameter_checksum(), result["parameterChecksumAfter"])
        self.assertEqual(len(reloaded.training_sources), 1)
        reloaded.events.close()

    def test_bundled_starter_is_trained_before_immutable_origin_snapshot(self):
        blank_root = self.root / "blank"
        starter_root = self.root / "starter"
        blank = AdaptiveBrain.create(
            "blank-brain",
            blank_root,
            OmniConfig.micro(
                origin_kind="blank",
                max_seq_len=40,
            ),
        )
        blank_checksum = blank.parameter_checksum()
        self.assertEqual(blank.counters["training_steps"], 0)
        self.assertIsNone(blank.starter_training_manifest)
        blank.events.close()

        starter = AdaptiveBrain.create(
            "starter-brain",
            starter_root,
            OmniConfig.micro(
                origin_kind="starter",
                max_seq_len=40,
            ),
        )
        manifest = starter.starter_training_manifest
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest["corpusPassagesVisited"], len(STARTER_CORPUS))
        self.assertEqual(len(manifest["corpusLossCurve"]), len(STARTER_CORPUS))
        self.assertTrue(
            all(loss >= 0.0 for loss in manifest["corpusLossCurve"])
        )
        self.assertEqual(
            sum(entry["records"] for entry in manifest["datasetLedger"]),
            len(STARTER_CORPUS) + manifest["actionTrajectories"],
        )
        self.assertTrue(
            all(
                len(entry["sha256"]) == 64
                and entry["upstreamModel"] is None
                for entry in manifest["datasetLedger"]
            )
        )
        self.assertFalse(manifest["rlhf"])
        self.assertFalse(manifest["dpo"])
        self.assertFalse(manifest["rewardModel"])
        self.assertFalse(manifest["hiddenBehavioralPrompt"])
        self.assertNotEqual(blank_checksum, starter.parameter_checksum())
        self.assertEqual(starter.counters["experiences"], len(STARTER_CORPUS))
        self.assertGreater(starter.counters["training_steps"], len(STARTER_CORPUS))
        self.assertEqual(
            set(manifest["modalityTraining"]["modalities"]),
            {"vision", "image", "audio", "video"},
        )
        self.assertTrue(all(starter.modality_training.values()))
        self.assertEqual(
            (starter.engine_path / "core.safetensors").read_bytes(),
            (starter.engine_path / "origin" / "core.safetensors").read_bytes(),
        )
        self.assertEqual(
            (starter.engine_path / "plasticity.safetensors").read_bytes(),
            (
                starter.engine_path / "origin" / "plasticity.safetensors"
            ).read_bytes(),
        )
        self.assertTrue(
            (
                starter.engine_path
                / "origin"
                / "substrate"
                / "manifest.json"
            ).is_file()
        )
        current_packed = verify_ternary_shards(
            starter.engine_path / "packed-ternary"
        )
        origin_packed = verify_ternary_shards(
            starter.engine_path / "origin" / "packed-ternary"
        )
        self.assertEqual(current_packed.manifest, origin_packed.manifest)
        self.assertEqual(
            starter.packed_ternary_manifest["eligibleTensorCount"],
            len(current_packed.tensors),
        )
        action_text = "make an image from this internal scene"
        action_ids = starter.tokenizer.tensor(
            action_text,
            starter.device,
            starter.config.max_seq_len,
        )
        action_cue = starter._idea_model_vector(
            starter.memory.vector_for_text(action_text)
        )
        with torch.no_grad():
            decoder_logits = starter.decoder(
                action_ids,
                use_global_workspace=True,
            )["action_logits"]
            internal_logits = starter.decoder.internal_action_policy(action_cue)
            _scores, actions = starter._select_structured_actions(
                0.35 * decoder_logits + 0.65 * internal_logits,
                schemas=[
                    {
                        "id": "modality.imagine",
                        "actions": ["generate"],
                        "grant": "ask",
                    }
                ],
                input_text=action_text,
                assembly_ids=[],
                organic_state={"computeDemand": 0.7},
            )
        self.assertEqual(actions[0]["kind"], "imagine")
        self.assertNotIn("prompt", actions[0]["arguments"])
        metadata = (starter.engine_path / "brain.json").read_text("utf-8")
        self.assertNotIn(STARTER_CORPUS[0], metadata)
        starter.events.close()

        reloaded = AdaptiveBrain.load(starter_root, "starter-brain")
        self.assertEqual(
            reloaded.starter_training_manifest["sha256"],
            manifest["sha256"],
        )
        self.assertEqual(reloaded.parameter_checksum(), starter.parameter_checksum())
        reloaded.events.close()

    def test_packed_inference_shards_refresh_after_dynamic_growth(self):
        brain = self.make_brain()
        initial = verify_ternary_shards(brain.engine_path / "packed-ternary")
        initial_hash = initial.manifest["contentSha256"]
        brain.learn_experience(
            "Packed dynamic synapses connect refreshed assemblies.",
            steps=0,
        )
        result = brain.export_packed_ternary()
        refreshed = verify_ternary_shards(brain.engine_path / "packed-ternary")
        self.assertNotEqual(initial_hash, refreshed.manifest["contentSha256"])
        dynamic = refreshed.tensors["substrate.dynamic_synapses.weights"]
        self.assertEqual(dynamic.numel(), len(brain.memory.synapses))
        self.assertTrue(
            set(int(value) for value in torch.unique(dynamic).tolist()).issubset(
                {-1, 0, 1}
            )
        )
        self.assertEqual(
            result["summary"]["dynamicSynapseCount"],
            len(brain.memory.synapses),
        )
        self.assertEqual(
            result["summary"]["parameterChecksum"],
            brain.parameter_checksum(),
        )
        brain.events.close()

    def test_load_verifies_packed_shards_and_refreshes_only_valid_stale_state(self):
        brain = self.make_brain()
        brain.learn_experience(
            "A later checkpoint makes the existing inference pack stale.",
            steps=0,
        )
        brain.save()
        expected_checksum = brain.parameter_checksum()
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        refreshed = verify_ternary_shards(
            reloaded.engine_path / "packed-ternary"
        )
        self.assertEqual(
            refreshed.manifest["metadata"]["parameterChecksum"],
            expected_checksum,
        )
        shard = (
            reloaded.engine_path
            / "packed-ternary"
            / refreshed.manifest["shards"][0]["file"]
        )
        corrupted = bytearray(shard.read_bytes())
        corrupted[0] ^= 0x01
        shard.write_bytes(corrupted)
        reloaded.events.close()
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            AdaptiveBrain.load(self.root, "brain-test")

    def test_load_refreshes_valid_pack_when_dynamic_synapse_order_or_values_are_stale(self):
        brain = self.make_brain()
        brain.learn_experience(
            "Dynamic inference packs must match every persisted ternary synapse.",
            steps=0,
        )
        synapse_ids, expected_before = brain._dynamic_synapse_export()
        self.assertTrue(synapse_ids)
        target_id = synapse_ids[0]
        previous = int(brain.memory.synapses[target_id]["effective_weight"])
        replacement = 1 if previous != 1 else -1
        brain.memory.synapses[target_id]["effective_weight"] = replacement
        brain.save()
        expected_checksum = brain.parameter_checksum()
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        verified = verify_ternary_shards(
            reloaded.engine_path / "packed-ternary"
        )
        reloaded_ids, reloaded_values = reloaded._dynamic_synapse_export()
        packed_values = verified.tensors[
            "substrate.dynamic_synapses.weights"
        ]
        self.assertEqual(
            verified.manifest["metadata"]["parameterChecksum"],
            expected_checksum,
        )
        self.assertEqual(
            verified.manifest["metadata"]["dynamicSynapseCount"],
            len(reloaded_ids),
        )
        self.assertEqual(
            verified.manifest["metadata"]["dynamicSynapseOrderSha256"],
            __import__("hashlib").sha256(
                "\0".join(reloaded_ids).encode("utf-8")
            ).hexdigest(),
        )
        self.assertTrue(torch.equal(packed_values, reloaded_values))
        self.assertNotEqual(
            int(packed_values[0]),
            int(expected_before[0]),
        )
        reloaded.events.close()

    def test_explicit_dataset_epoch_replays_without_duplicate_source_records(self):
        brain = self.make_brain()
        text = "Each requested epoch must revisit this valid record."
        first = brain.ingest(
            text=text,
            name="epochs.txt",
            policy="encode",
            epoch=0,
        )
        first_checksum = first["parameterChecksumAfter"]
        second = brain.ingest(
            text=text,
            name="epochs.txt",
            policy="encode",
            allow_replay=True,
            epoch=1,
        )
        self.assertFalse(second["duplicate"])
        self.assertNotEqual(
            first_checksum, second["parameterChecksumAfter"]
        )
        self.assertEqual(len(brain.training_sources), 1)
        self.assertEqual(brain.training_sources[0]["training_epochs"], 2)
        self.assertEqual(brain.training_sources[0]["last_epoch"], 1)
        brain.events.close()

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
            "spreading_activation",
        ):
            self.assertIn(field, trace)
        self.assertFalse(trace["textual_memory_injected"])
        self.assertNotEqual(
            trace["parameter_checksum_before"],
            trace["parameter_checksum_after"],
        )
        self.assertGreaterEqual(len(trace["branches"]), 1)
        self.assertTrue(
            trace["spreading_activation"]["exactTernaryContribution"]
        )
        self.assertFalse(
            trace["spreading_activation"]["latentMagnitudeUsed"]
        )
        self.assertIn("action_policy_scores", trace)
        self.assertEqual(
            set(trace["action_policy_scores"]),
            {"talk", "tool", "imagine", "agent", "ponder", "learn", "evolve", "stop"},
        )
        self.assertFalse(result["runtimeCard"]["hidden_behavioral_prompt"])
        brain.events.close()

    def test_default_response_budget_is_state_scaled_not_the_context_ceiling(self):
        brain = AdaptiveBrain.create(
            "long-response-brain",
            self.root,
            OmniConfig.micro(
                max_seq_len=520,
                learn_from_own_messages=False,
            ),
        )
        observed: list[int] = []

        def generate(input_ids, **kwargs):
            observed.append(int(kwargs["max_new_tokens"]))
            suffix = torch.tensor(
                [[ord("A") + brain.tokenizer.byte_offset]],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix), dim=1), [0.0]

        with mock.patch.object(brain.decoder, "generate", side_effect=generate):
            result = brain.chat("Use the hardware-sized response workspace.", seed=17)
        self.assertEqual(result["text"], "A")
        self.assertTrue(observed)
        self.assertEqual(len(set(observed)), 1)
        budget = observed[0]
        self.assertGreaterEqual(
            budget, brain.config.generation_token_budget(0.0)
        )
        self.assertLessEqual(
            budget, brain.config.generation_token_budget(1.0)
        )
        self.assertLess(budget, brain.config.max_seq_len)
        self.assertEqual(
            result["trace"]["generation_budget_source"],
            "hardware-and-organic-state",
        )
        self.assertEqual(
            result["trace"]["generation_budget_tokens"], budget
        )
        brain.events.close()

    def test_substrate_inspection_is_paged_read_only_and_invalidates_stale_cursor(self):
        brain = self.make_brain(memory_recipe="total-recall", retain_source_text=True)
        brain.learn_experience(
            "Distributed assemblies connect language memory and action.",
            source="document",
            source_label="private-fixture.txt",
            steps=0,
        )
        brain.learn_experience(
            "A second private assembly proves cursor paging.",
            source="document",
            source_label="private-fixture.txt",
            steps=0,
        )
        before = brain.parameter_checksum()
        first = brain.query_substrate(
            {"entity": "assemblies", "zoom": 1, "pageSize": 1}
        )
        self.assertEqual(len(first["assemblies"]), 1)
        self.assertNotIn("source_text", first["assemblies"][0])
        self.assertTrue(first["assemblies"][0]["retainsSourceText"])
        self.assertEqual(before, brain.parameter_checksum())
        self.assertTrue(first["hasMore"])

        second = brain.query_substrate(
            {
                "entity": "assemblies",
                "zoom": 1,
                "pageSize": 1,
                "cursor": first["nextCursor"],
            }
        )
        self.assertNotEqual(
            first["assemblies"][0]["id"], second["assemblies"][0]["id"]
        )
        brain.learn_experience("Structural growth invalidates the cursor.", steps=0)
        with self.assertRaisesRegex(ValueError, "stale"):
            brain.query_substrate(
                {
                    "entity": "assemblies",
                    "zoom": 1,
                    "pageSize": 1,
                    "cursor": first["nextCursor"],
                }
            )
        brain.events.close()

    def test_feedback_uses_real_stdp_without_a_reward_model(self):
        brain = self.make_brain()
        text = "Causal blue assemblies connect local evidence."
        brain.learn_experience(text, steps=0)
        before = brain.parameter_checksum()
        positive = brain.feedback(
            text,
            "up",
            trace_id="trace-positive",
            message_id="message-positive",
        )
        self.assertFalse(positive["rewardModel"])
        self.assertFalse(positive["rlhf"])
        self.assertGreater(positive["stdp"]["stdp_update"], 0.0)
        self.assertNotEqual(
            positive["synapseChecksumBefore"],
            positive["synapseChecksumAfter"],
        )
        self.assertNotEqual(before, positive["parameterChecksumAfter"])

        negative = brain.feedback(
            text,
            "down",
            trace_id="trace-negative",
            message_id="message-negative",
        )
        self.assertGreater(negative["stdp"]["stdp_update"], 0.0)
        self.assertLess(negative["stdp"]["signed_update"], 0.0)
        self.assertNotEqual(
            negative["synapseChecksumBefore"],
            negative["synapseChecksumAfter"],
        )
        metadata = (brain.engine_path / "brain.json").read_text("utf-8")
        self.assertNotIn(text, metadata)
        brain.events.close()

    def test_idle_cycle_is_prompt_free_organic_and_mutates_neural_state(self):
        brain = self.make_brain()
        brain.learn_experience(
            "Unfinished amber hypotheses activate temporal associations.",
            steps=0,
        )
        messages_before = list(brain.messages)
        result = brain.idle_cycle(
            tool_schemas=[
                {
                    "id": "modality.imagine",
                    "actions": ["generate"],
                    "grant": "ask",
                }
            ],
            minimum_idle_seconds=0,
        )
        self.assertTrue(result["ran"])
        self.assertEqual(result["trace"]["promptTokenCount"], 0)
        self.assertFalse(result["trace"]["hiddenBehavioralPrompt"])
        self.assertGreater(result["trace"]["stdpUpdate"], 0.0)
        self.assertNotEqual(
            result["trace"]["parameterChecksumBefore"],
            result["trace"]["parameterChecksumAfter"],
        )
        self.assertEqual(brain.messages, messages_before)
        self.assertEqual(brain.counters["idle_cognition_cycles"], 1)
        brain.events.close()

    def test_idle_talk_is_generated_from_neural_state_without_a_prompt(self):
        brain = self.make_brain()
        brain.learn_experience(
            "A half-formed question remains active in the workspace.",
            steps=0,
        )
        talk_scores = {
            "talk": 0.92,
            "tool": 0.01,
            "imagine": 0.01,
            "agent": 0.01,
            "ponder": 0.01,
            "learn": 0.01,
            "evolve": 0.01,
            "stop": 0.02,
        }
        message = "Could this unfinished association connect differently?"
        generated = torch.tensor(
            [
                [
                    brain.tokenizer.bos_id,
                    brain.tokenizer.brain_id,
                    *brain.tokenizer.encode(message),
                    brain.tokenizer.eos_id,
                ]
            ],
            dtype=torch.long,
            device=brain.device,
        )
        with mock.patch.object(
            brain,
            "_organic_state",
            return_value={
                "tension": 0.9,
                "curiosity": 0.9,
                "uncertainty": 0.8,
                "novelty": 0.8,
                "learningProgress": 0.6,
            },
        ), mock.patch.object(
            brain,
            "_select_structured_actions",
            return_value=(talk_scores, []),
        ), mock.patch.object(
            brain.decoder,
            "generate",
            return_value=(generated, [0.4]),
        ) as generate:
            result = brain.idle_cycle(minimum_idle_seconds=0)

        self.assertEqual(result["actions"][0]["kind"], "talk")
        self.assertEqual(result["actions"][0]["arguments"]["message"], message)
        self.assertEqual(result["actions"][0]["arguments"]["promptTokenCount"], 0)
        self.assertEqual(brain.messages[-1]["content"], message)
        boundary = generate.call_args.args[0]
        self.assertEqual(
            boundary.detach().cpu().tolist(),
            [[brain.tokenizer.bos_id, brain.tokenizer.brain_id]],
        )
        self.assertEqual(result["trace"]["promptTokenCount"], 0)
        self.assertFalse(result["trace"]["hiddenBehavioralPrompt"])
        brain.events.close()

    def test_candidate_exception_rolls_back_all_core_parameters(self):
        brain = self.make_brain()
        before = brain.parameter_checksum()

        def explode(*args, **kwargs):
            del args, kwargs
            with torch.no_grad():
                next(brain.decoder.parameters()).add_(100.0)
            raise RuntimeError("deliberate candidate failure")

        brain._experience_ids_batch_loss = explode
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

    def test_slow_learning_reaches_a_tail_beyond_one_context_window(self):
        left = self.make_brain(seed=91)
        right = AdaptiveBrain.create(
            "brain-tail-right",
            self.root / "tail-right",
            OmniConfig.micro(
                seed=91,
                max_seq_len=40,
                learn_from_own_messages=False,
            ),
        )
        left_idea = left.memory.space.symbol("fixed-windowed-training-idea")
        right_idea = right.memory.space.symbol("fixed-windowed-training-idea")
        prefix = "shared-prefix-" * 24
        torch.manual_seed(404)
        left._optimize_experience(
            prefix + "TAIL-ALPHA",
            left_idea,
            steps=1,
            commit_stability=False,
        )
        torch.manual_seed(404)
        right._optimize_experience(
            prefix + "TAIL-OMEGA",
            right_idea,
            steps=1,
            commit_stability=False,
        )
        expected_windows = (
            len((prefix + "TAIL-ALPHA").encode("utf-8")) + 37
        ) // 38
        self.assertEqual(left.counters["training_steps"], expected_windows)
        self.assertNotEqual(
            left.parameter_checksum(), right.parameter_checksum()
        )
        left.events.close()
        right.events.close()

    def test_stable_substrate_invariants_cannot_be_disabled(self):
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
                module.ternary
                for root in brain._trainable_modules()
                for module in root.modules()
                if isinstance(module, BitLinear)
            )
        )
        learned = brain.learn_experience("transient dense fixture", steps=0)
        self.assertGreaterEqual(learned["spiking"]["spikes"], 0.0)
        self.assertGreater(len(brain.memory.ideas), 0)
        self.assertEqual(learned["training"]["loss"], 0.0)
        consolidation = brain.consolidate()
        self.assertTrue(consolidation["disabled"])
        card = brain.runtime_card()
        self.assertTrue(card["active_modules"]["ternary"])
        self.assertFalse(card["active_modules"]["dense"])
        self.assertTrue(card["active_modules"]["spiking"])
        self.assertTrue(card["active_modules"]["stdp"])
        self.assertTrue(card["active_modules"]["liquid"])
        self.assertTrue(card["active_modules"]["vsa"])
        brain.events.close()

    def test_wav_ingestion_trains_audio_parameters(self):
        brain = self.make_brain()
        wav_path = self.root / "tone.wav"
        window = brain.config.audio_samples
        tail = 5
        samples = array(
            "h",
            [
                int(12000 * ((index % 8) / 7.0 - 0.5))
                for index in range(window * 2)
            ]
            + [14000] * tail,
        )
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(samples.tobytes())
        observed = []
        hook = brain.modalities.audio.register_forward_pre_hook(
            lambda _module, inputs: observed.append(inputs[0].detach().cpu().clone())
        )
        before = brain.parameter_checksum()
        try:
            result = brain.ingest(
                path=str(wav_path), kind="audio", policy="encode"
            )
        finally:
            hook.remove()
        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertNotEqual(before, result["parameterChecksumAfter"])
        self.assertEqual(result["mediaCoverage"]["windows"], 3)
        self.assertEqual(
            result["mediaCoverage"]["processedSamples"], window * 2 + tail
        )
        self.assertEqual(result["mediaCoverage"]["tailSamples"], tail)
        self.assertTrue(result["mediaCoverage"]["complete"])
        self.assertEqual(result["coverage"]["processedRecords"], 1)
        # Two learning steps see each window. The final two calls therefore
        # prove the non-full tail was admitted rather than silently discarded.
        self.assertEqual(len(observed), 6)
        tail_target = observed[-1].flatten()
        self.assertTrue(torch.all(tail_target[:tail] > 0.4))
        self.assertTrue(torch.all(tail_target[tail:] == 0.0))
        brain.events.close()

    def test_failed_media_decode_is_an_explicit_rejected_record(self):
        brain = self.make_brain(image_enabled=True, vision_enabled=True)
        image_path = self.root / "corrupt.png"
        image_path.write_bytes(b"not-a-decodable-image")

        with mock.patch.object(
            brain,
            "_decode_image",
            side_effect=RuntimeError("corrupt image fixture"),
        ):
            result = brain.ingest(
                path=str(image_path),
                kind="image",
                policy="encode",
            )

        self.assertFalse(result["source"]["modality_trained"])
        self.assertEqual(result["mediaCoverage"]["failedRecords"], 1)
        self.assertFalse(result["mediaCoverage"]["complete"])
        self.assertEqual(
            result["coverage"],
            {
                "discoveredFiles": 1,
                "completedFiles": 1,
                "processedFiles": 1,
                "rejectedFiles": 0,
                "discoveredRecords": 1,
                "processedRecords": 0,
                "rejectedRecords": 1,
                "processedBytes": len(b"not-a-decodable-image"),
                "shards": 0,
                "modalityCounts": {"image": 1},
                "errors": [
                    {
                        "source": "corrupt.png",
                        "message": "corrupt image fixture",
                    }
                ],
                "complete": True,
            },
        )
        self.assertTrue(
            any("corrupt image fixture" in warning for warning in result["warnings"])
        )
        brain.events.close()

    def test_archive_media_is_trained_while_temporary_record_path_is_leased(self):
        brain = self.make_brain()
        wav_path = self.root / "leased.wav"
        samples = array(
            "h",
            [9000 if index % 2 else -9000 for index in range(73)],
        )
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(8000)
            handle.writeframes(samples.tobytes())
        archive_path = self.root / "media.tar"
        with tarfile.open(archive_path, "w") as archive:
            archive.add(wav_path, arcname="nested/tone.wav")

        result = brain.ingest(
            path=str(archive_path),
            kind="archive",
            policy="encode",
        )

        self.assertTrue(result["source"]["modality_trained"])
        self.assertFalse(result["warnings"])
        self.assertEqual(result["coverage"]["modalityCounts"]["audio"], 1)
        self.assertEqual(result["mediaCoverage"]["records"], 1)
        self.assertEqual(result["mediaCoverage"]["trainedRecords"], 1)
        self.assertEqual(result["mediaCoverage"]["windows"], 2)
        media_record = result["source"]["media_records"][0]
        self.assertEqual(media_record["kind"], "audio")
        self.assertEqual(len(media_record["contentSha256"]), 64)
        self.assertNotIn("local_path", json.dumps(media_record))
        brain.events.close()

    def test_ffmpeg_audio_fallback_has_no_duration_cap_and_streams_tail(self):
        brain = self.make_brain()
        total = brain.config.audio_samples * 2 + 7
        samples = array("f", [float(index) / total for index in range(total)])

        class FakeProcess:
            def __init__(self):
                self.stdout = io.BytesIO(samples.tobytes())
                self.returncode = None

            def wait(self, timeout=None):
                self.returncode = 0
                return 0

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def kill(self):
                self.returncode = -9

        process = FakeProcess()
        ffmpeg = types.SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg")
        with mock.patch.dict(
            sys.modules,
            {"soundfile": None, "imageio_ffmpeg": ffmpeg},
        ), mock.patch(
            "omni_core.brain.subprocess.Popen",
            return_value=process,
        ) as launch:
            windows = list(brain._iter_audio_windows("streamed.mp3"))
        command = launch.call_args.args[0]
        self.assertNotIn("-t", command)
        self.assertEqual(len(windows), 3)
        self.assertEqual(sum(actual for _target, actual in windows), total)
        self.assertEqual(windows[-1][1], 7)
        brain.events.close()

    def test_generated_artifacts_are_chromium_viewable_and_embedded(self):
        brain = self.make_brain()
        image = brain.generate_modality("image", prompt="blue geometry", seed=1)
        audio = brain.generate_modality("audio", prompt="short tone", seed=2)
        video = brain.generate_modality("video", prompt="moving light", seed=3)
        self.assertEqual(Path(image["path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(Path(audio["path"]).read_bytes()[:4], b"RIFF")
        self.assertTrue(image["dataUrl"].startswith("data:image/png;base64,"))
        self.assertTrue(audio["dataUrl"].startswith("data:audio/wav;base64,"))
        if video["mimeType"] == "video/mp4":
            self.assertEqual(Path(video["path"]).read_bytes()[4:8], b"ftyp")
            self.assertTrue(video["dataUrl"].startswith("data:video/mp4;base64,"))
            self.assertEqual(video["containerFallback"], "")
        else:
            self.assertEqual(Path(video["path"]).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertTrue(video["dataUrl"].startswith("data:image/apng;base64,"))
            self.assertTrue(video["containerFallback"])
        brain.events.close()

    def test_snapshot_and_append_only_event_log(self):
        brain = self.make_brain()
        brain.learn_experience(
            "Snapshot shards preserve this distributed neural assembly.",
            steps=0,
        )
        result = brain.snapshot("checkpoint")
        self.assertTrue(Path(result["path"], "core.safetensors").is_file())
        snapshot_path = Path(result["path"])
        snapshot_metadata = json.loads(
            (snapshot_path / "brain.json").read_text("utf-8")
        )
        restored_memory = ConceptMemory.load_sharded(
            snapshot_path / "substrate",
            snapshot_metadata["substrate"],
        )
        self.assertEqual(brain.memory.neurons, restored_memory.neurons)
        self.assertEqual(brain.memory.assemblies, restored_memory.assemblies)
        self.assertEqual(brain.memory.synapses, restored_memory.synapses)
        self.assertEqual(brain.events.integrity(), "ok")
        with self.assertRaises(sqlite3.DatabaseError):
            brain.events.connection.execute(
                "UPDATE events SET kind='changed' WHERE sequence=1"
            )
        brain.events.close()

    def test_interrupted_shard_save_keeps_prior_generation_loadable(self):
        brain = self.make_brain()
        prior_metadata_bytes = (
            brain.engine_path / "brain.json"
        ).read_bytes()
        prior_metadata = json.loads(prior_metadata_bytes.decode("utf-8"))
        prior_generation = prior_metadata["substrate"]["persistence"][
            "activeGeneration"
        ]
        prior_assemblies = len(brain.memory.assemblies)

        # Simulate a state mutation followed by the host reserve pausing before
        # any new generation pointer or engine metadata can be promoted.
        saved_guard = brain.memory.growth_guard
        brain.memory.growth_guard = None
        brain.memory.learn(
            "An interrupted save must not replace the committed shard generation."
        )
        brain.memory.growth_guard = lambda _estimated: False
        with self.assertRaises(SubstrateResourcePause):
            brain.save()
        self.assertEqual(
            (brain.engine_path / "brain.json").read_bytes(),
            prior_metadata_bytes,
        )
        self.assertEqual(
            json.loads(
                (
                    brain.engine_path / "substrate" / "manifest.json"
                ).read_text("utf-8")
            )["activeGeneration"],
            prior_generation,
        )
        brain.memory.growth_guard = saved_guard
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(len(reloaded.memory.assemblies), prior_assemblies)
        self.assertEqual(
            reloaded.memory.persistence_manifest["activeGeneration"],
            prior_generation,
        )
        reloaded.events.close()

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
        self.assertEqual(config.working_memory_slots, 512)
        self.assertFalse(hasattr(config, "parallel_thoughts"))
        self.assertEqual(config.train_batch_size, 2)
        self.assertEqual(config.gradient_accumulation, 8)
        self.assertTrue(config.gradient_checkpointing)

    def test_slow_training_uses_profile_batch_and_gradient_accumulation(self):
        brain = self.make_brain(
            train_batch_size=2,
            gradient_accumulation=2,
            gradient_checkpointing=True,
        )
        result = brain.train(
            texts=[
                "alpha learns amber",
                "beta learns blue",
                "gamma learns green",
                "delta learns violet",
            ],
            epochs=1,
        )
        self.assertEqual(result["steps"], 4)
        self.assertEqual(result["optimizerSteps"], 1)
        self.assertEqual(result["physicalBatchSize"], 2)
        self.assertEqual(result["gradientAccumulation"], 2)
        self.assertTrue(
            brain.runtime_card()["scale"]["gradientCheckpointing"]
        )
        brain.events.close()

    def test_working_memory_is_internal_bounded_and_persistent(self):
        parameter_config = OmniConfig.micro(
            name="parameter",
            seed=81,
            max_seq_len=40,
            online_learning=False,
            learn_from_own_messages=False,
            memory_injection="parameter-only",
            working_memory_slots=2,
        )
        working_config = OmniConfig.micro(
            name="working",
            seed=81,
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
        self.assertEqual(
            left["trace"]["working_memory_channel"], "recurrent-vector"
        )
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

    def test_recent_dialogue_tokens_participate_without_hidden_prompt_text(self):
        brain = AdaptiveBrain.create(
            "recent-context",
            self.root / "recent-context",
            OmniConfig.micro(
                max_seq_len=96,
                online_learning=False,
                learn_from_own_messages=False,
            ),
        )
        observed: list[list[int]] = []

        def generate(input_ids, **_kwargs):
            observed.append(input_ids[0].detach().cpu().tolist())
            suffix = torch.tensor(
                [[ord("A") + brain.tokenizer.byte_offset]],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix), dim=1), [0.0]

        with mock.patch.object(brain.decoder, "generate", side_effect=generate):
            first = brain.chat("Remember first-marker.", max_new_tokens=1, seed=3)
            first_call_count = len(observed)
            second = brain.chat(
                "Use it in this turn.",
                max_new_tokens=1,
                seed=4,
                tool_schemas=[
                    {
                        "id": "code.execute",
                        "actions": ["run"],
                        "grant": "ask",
                        "description": "HIDDEN TOOL PROSE MUST NOT ENTER",
                    }
                ],
            )

        completed, _removed = brain._bounded_completed_turn_tokens(
            "Remember first-marker.", first["text"]
        )
        second_prompts = observed[first_call_count:]
        self.assertTrue(second_prompts)
        self.assertTrue(
            all(
                prompt[0] == brain.tokenizer.bos_id
                and completed == prompt[1 : 1 + len(completed)]
                for prompt in second_prompts
            )
        )
        decoded_prompt = brain.tokenizer.decode(second_prompts[0])
        self.assertIn("first-marker", decoded_prompt)
        self.assertNotIn("HIDDEN TOOL PROSE", decoded_prompt)
        self.assertTrue(second["trace"]["recent_dialogue_context_injected"])
        self.assertGreater(second["trace"]["recent_dialogue_token_count"], 0)
        self.assertTrue(second["trace"]["prompt_text_expanded"])
        self.assertFalse(second["trace"]["hidden_prompt_text_expanded"])
        self.assertFalse(second["trace"]["long_term_source_text_injected"])
        self.assertFalse(second["trace"]["textual_memory_injected"])
        self.assertFalse(second["trace"]["tool_schema_text_injected"])
        self.assertFalse(second["runtimeCard"]["hidden_behavioral_prompt"])
        brain.events.close()

    def test_recent_dialogue_ring_evicts_by_capacity_and_survives_reload(self):
        root = self.root / "recent-reload"
        brain = AdaptiveBrain.create(
            "recent-reload",
            root,
            OmniConfig.micro(
                max_seq_len=24,
                online_learning=False,
                learn_from_own_messages=False,
            ),
        )

        def generate(input_ids, **_kwargs):
            suffix = torch.tensor(
                [[ord("R") + brain.tokenizer.byte_offset]],
                dtype=torch.long,
                device=input_ids.device,
            )
            return torch.cat((input_ids, suffix), dim=1), [0.0]

        with mock.patch.object(brain.decoder, "generate", side_effect=generate):
            brain.chat(
                "A deliberately oversized first working-memory turn.",
                max_new_tokens=1,
                seed=8,
            )
            brain.chat(
                "A newer turn replaces the oldest bounded token activity.",
                max_new_tokens=1,
                seed=9,
            )
        snapshot = brain.workspace_snapshot()
        recent_before = list(brain.recent_token_context)
        self.assertLessEqual(len(recent_before), brain.config.max_seq_len)
        self.assertEqual(recent_before[0], brain.tokenizer.human_id)
        self.assertIn(brain.tokenizer.brain_id, recent_before)
        self.assertEqual(recent_before[-1], brain.tokenizer.eos_id)
        self.assertGreater(snapshot["contextWindow"]["evictions"], 0)
        self.assertEqual(
            snapshot["contextWindow"]["recentTokenCount"], len(recent_before)
        )
        self.assertEqual(
            snapshot["contextWindow"]["recentTokenHash"],
            brain._token_sequence_hash(recent_before),
        )
        evictions = brain.counters["context_token_evictions"]
        brain.events.close()

        reloaded = AdaptiveBrain.load(root, "recent-reload")
        self.assertEqual(reloaded.recent_token_context, recent_before)
        self.assertEqual(
            reloaded.counters["context_token_evictions"], evictions
        )
        self.assertEqual(
            reloaded.workspace_snapshot()["contextWindow"]["recentTokenHash"],
            snapshot["contextWindow"]["recentTokenHash"],
        )
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
            growth_novelty_threshold=2.0,
        )
        brain._resource_readings = lambda: {
            "diskFreeBytes": 8 * 1024**3,
            "availableMemoryBytes": 8 * 1024**3,
        }
        brain.learn_experience("alpha beta gamma", steps=0)
        brain.learn_experience("delta epsilon zeta", steps=0)
        self.assertGreater(len(brain.memory.neurons), 1)
        self.assertGreater(len(brain.memory.assemblies), 1)
        self.assertGreater(len(brain.memory.synapses), 1)
        self.assertGreater(brain.memory.growth_events, 0)
        self.assertIsNone(brain.memory.metadata()["cardinality_limit"])
        card = brain.runtime_card()
        self.assertIsNone(card["growth"]["substrate"]["cardinalityLimit"])
        self.assertGreater(card["growth"]["substrate"]["neurons"], 1)
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
