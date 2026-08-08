import copy
import io
import json
import shutil
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
from omni_core.model import ACTION_KINDS, BitLinear
from omni_core.starter import (
    STARTER_ACTION_EXAMPLES,
    STARTER_CORPUS,
    starter_manifest,
)
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
        blank_heads = {
            key: value.detach().clone()
            for key, value in blank.decoder.action_policy.state_dict().items()
        }
        blank_anchors = {
            key: value.clone()
            for key, value in blank.slow_anchors.items()
        }
        blank_counters = dict(blank.counters)
        self.assertFalse(blank._can_retain_bundled_action_policy())
        self.assertEqual(blank.counters, blank_counters)
        self.assertTrue(
            all(
                torch.equal(value, blank_heads[key])
                for key, value in blank.decoder.action_policy.state_dict().items()
            )
        )
        self.assertTrue(
            all(
                torch.equal(value, blank_anchors[key])
                for key, value in blank.slow_anchors.items()
            )
        )
        blank.config.origin_kind = "starter"
        blank.starter_training_manifest = starter_manifest()
        blank.save()
        blank.events.close()
        spoofed = AdaptiveBrain.load(blank_root, "blank-brain")
        self.assertFalse(spoofed._bundled_origin_verified)
        self.assertFalse(spoofed._can_retain_bundled_action_policy())
        spoofed.events.close()

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
        self.assertTrue(starter._bundled_origin_verified)
        self.assertTrue(
            (starter.engine_path / "origin" / "provenance.json").is_file()
        )
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
        action_training = manifest["actionTraining"]
        self.assertEqual(
            action_training["languageChatFraming"],
            ["bos", "human", "text", "brain"],
        )
        self.assertTrue(action_training["calibrated"])
        self.assertGreater(
            action_training["minimumLanguageTargetConfidence"], 0.62
        )
        self.assertGreater(
            action_training["minimumInternalTargetConfidence"], 0.62
        )
        self.assertGreater(action_training["minimumConfidenceMargin"], 0.0)
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
        action_ids = starter._action_chat_tensor(action_text)
        action_cue = starter._idea_model_vector(
            starter.memory.vector_for_text(action_text)
        )
        with torch.no_grad():
            action_hidden = starter.decoder(
                action_ids,
                memory_bias=starter.idea_adapter(action_cue),
                use_global_workspace=True,
            )["hidden"][:, -1]
            decoder_logits = starter.decoder.action_policy(
                action_hidden + 0.5 * action_cue
            )
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
        self.assertTrue(reloaded._bundled_origin_verified)
        reloaded.events.close()

        fork_root = self.root / "starter-fork"
        shutil.copytree(starter_root, fork_root)
        fork_metadata_path = fork_root / "engine" / "brain.json"
        fork_metadata = json.loads(fork_metadata_path.read_text("utf-8"))
        fork_metadata["brain_id"] = "starter-fork"
        fork_metadata_path.write_text(
            json.dumps(fork_metadata),
            encoding="utf-8",
        )
        forked = AdaptiveBrain.load(fork_root, "starter-fork")
        self.assertTrue(forked._bundled_origin_verified)
        self.assertTrue(forked._can_retain_bundled_action_policy())
        forked.events.close()

    def test_starter_action_retention_survives_online_chat_trajectory(self):
        brain = AdaptiveBrain.create(
            "starter-action-retention",
            self.root / "starter-action-retention",
            OmniConfig.micro(
                origin_kind="starter",
                max_seq_len=96,
                learn_from_own_messages=False,
                vision_enabled=False,
                image_enabled=False,
                audio_enabled=False,
                video_enabled=False,
            ),
        )
        schemas = [
            {
                "id": "modality.imagine",
                "actions": ["generate"],
                "grant": "ask",
            },
            {
                "id": "agent.fork",
                "actions": ["start"],
                "grant": "ask",
            },
        ]
        visible_result = "\n".join(
            [
                "[Visible structured action result]",
                "kind: imagine",
                "tool: modality.imagine",
                "action: generate",
                "requested-by: brain",
                "result:",
                '{"status":"complete","artifact":"fixture-image"}',
            ]
        )
        immutable_manifest = json.loads(
            json.dumps(brain.starter_training_manifest)
        )

        # Force two independent representation shifts outside the action
        # heads. Retention must replay all current starter routes through one
        # padded decoder batch, recover every margin, and preserve the exact
        # action that was emitted before the slow mutation.
        agent_index = next(
            index
            for index, (_text, kind) in enumerate(STARTER_ACTION_EXAMPLES)
            if kind == "agent"
        )
        retained_head_parameter = next(
            iter(brain.decoder.action_policy.parameters())
        )
        unrelated_optimizer_parameter = brain.memory_bridge.weight
        for parameter, fill in (
            (retained_head_parameter, 0.75),
            (unrelated_optimizer_parameter, 0.25),
        ):
            brain._optimizer.state[parameter] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.full_like(parameter, fill),
                "exp_avg_sq": torch.full_like(parameter, fill * fill),
            }
        unrelated_optimizer_state = {
            key: value.clone()
            for key, value in brain._optimizer.state[
                unrelated_optimizer_parameter
            ].items()
        }
        retained_head_parameters = (
            *brain.decoder.action_policy.parameters(),
            *brain.decoder.internal_action_policy.parameters(),
        )
        expected_cleared_head_states = sum(
            parameter in brain._optimizer.state
            for parameter in retained_head_parameters
        )
        drift_pattern = torch.linspace(
            0.97,
            1.03,
            brain.config.d_model,
            device=brain.device,
        )
        for drift_cycle in range(2):
            before_language, before_internal, _targets = (
                brain._starter_action_features()
            )
            with torch.no_grad():
                pre_language_logits = brain.decoder.action_policy(
                    before_language[agent_index : agent_index + 1]
                )
                pre_internal_logits = brain.decoder.internal_action_policy(
                    before_internal[agent_index : agent_index + 1]
                )
                brain.decoder.workspace_strength.add_(0.025)
                brain.decoder.final_norm.scale.mul_(drift_pattern)
                brain.memory_bridge.bias.add_(
                    (drift_cycle + 1)
                    * torch.linspace(
                        -0.004,
                        0.004,
                        brain.config.idea_dim,
                        device=brain.device,
                    )
                )
            after_language, after_internal, _targets = (
                brain._starter_action_features()
            )
            self.assertFalse(
                torch.allclose(before_language, after_language, atol=1e-7)
            )
            with mock.patch.object(
                brain.decoder,
                "forward",
                wraps=brain.decoder.forward,
            ) as current_route_forward:
                drift_retention = brain._retain_starter_action_policy(
                    pre_language_logits=pre_language_logits,
                    pre_internal_logits=pre_internal_logits,
                    pre_action_emitted=True,
                    post_language_feature=after_language[
                        agent_index : agent_index + 1
                    ],
                    post_internal_feature=after_internal[
                        agent_index : agent_index + 1
                    ],
                    exact_route_decoder_forwards=1,
                )
            self.assertEqual(current_route_forward.call_count, 1)
            self.assertIsNotNone(drift_retention)
            assert drift_retention is not None
            self.assertTrue(drift_retention["calibrated"])
            self.assertEqual(drift_retention["canonicalDecoderForwards"], 1)
            self.assertTrue(drift_retention["canonicalReplayReady"])
            if drift_cycle == 0:
                self.assertGreater(drift_retention["steps"], 0)
                self.assertTrue(
                    all(
                        parameter not in brain._optimizer.state
                        for parameter in retained_head_parameters
                    )
                )
                self.assertEqual(
                    drift_retention["mainOptimizerHeadStatesCleared"],
                    expected_cleared_head_states,
                )
                self.assertTrue(
                    all(
                        torch.equal(
                            brain._optimizer.state[
                                unrelated_optimizer_parameter
                            ][name],
                            value,
                        )
                        for name, value in unrelated_optimizer_state.items()
                    )
                )
            current_language, current_internal, current_targets = (
                brain._starter_action_features()
            )
            with torch.no_grad():
                current_reading = brain._action_calibration_reading(
                    brain.decoder.action_policy(current_language),
                    brain.decoder.internal_action_policy(current_internal),
                    current_targets,
                )
            self.assertGreater(
                current_reading["minimumDeployedThresholdMargin"],
                0.0,
            )

        for cycle in range(1):
            seed = 1000 + cycle * 10
            brain.chat(
                "hello, tell me what you notice",
                max_new_tokens=1,
                seed=seed,
                tool_schemas=schemas,
            )
            imagined = brain.chat(
                "make an image from this internal scene",
                max_new_tokens=1,
                seed=seed + 1,
                tool_schemas=schemas,
            )
            self.assertTrue(
                any(action["kind"] == "imagine" for action in imagined["actions"])
            )
            brain.chat(
                visible_result,
                max_new_tokens=1,
                seed=seed + 2,
                tool_schemas=schemas,
            )
            delegated = brain.chat(
                "fork agents to investigate these independent parts",
                max_new_tokens=1,
                seed=seed + 3,
                tool_schemas=schemas,
            )
            self.assertTrue(
                any(
                    action["kind"] == "agent"
                    and action["toolId"] == "agent.fork"
                    and action["action"] == "start"
                    for action in delegated["actions"]
                )
            )
            self.assertIn("agent", delegated["trace"]["proposed_action_kinds"])
            self.assertGreaterEqual(
                delegated["trace"]["action_policy_scores"]["agent"], 0.62
            )
            agent_action = next(
                action
                for action in delegated["actions"]
                if action["kind"] == "agent"
            )
            self.assertEqual(
                agent_action["arguments"]["objective"],
                "fork agents to investigate these independent parts",
            )
            self.assertNotIn("prompt", agent_action["arguments"])
            calibration = delegated["trace"]["action_policy_calibration"]
            self.assertTrue(calibration["calibrated"])
            self.assertEqual(
                calibration["mode"],
                "exact-route-self-distillation+current-neural-replay",
            )
            self.assertLessEqual(calibration["steps"], 96)
            self.assertEqual(
                calibration["exactRouteDecoderForwards"], 1
            )
            self.assertEqual(calibration["canonicalDecoderForwards"], 1)
            self.assertEqual(calibration["canonicalFeatureVectors"], 16)
            self.assertEqual(calibration["actualPreKind"], "agent")
            self.assertTrue(calibration["actualPreActionEmitted"])
            self.assertTrue(calibration["actualRoutePreserved"])
            self.assertFalse(calibration["syntheticDeployedGuarantee"])
            self.assertEqual(
                delegated["trace"]["action_policy_channel"],
                "exact-runtime-prompt+internal-memory+idea-fusion",
            )
            self.assertGreater(
                delegated["trace"]["action_policy_recent_dialogue_tokens"],
                0,
            )
            self.assertGreater(
                delegated["trace"]["action_policy_working_memory_vectors"],
                0,
            )
            self.assertTrue(
                delegated["trace"]["action_policy_capability_conditioned"]
            )
            self.assertEqual(
                delegated["trace"]["action_policy_deployed_kind"], "agent"
            )
            self.assertGreaterEqual(
                delegated["trace"]["action_policy_deployed_confidence"],
                0.62,
            )
            self.assertFalse(
                delegated["trace"]["action_policy_synthetic_guarantee"]
            )
            self.assertGreater(
                delegated["trace"]["decision_prediction_loss"], 0.0
            )
            self.assertGreater(
                delegated["trace"]["organic_state"]["predictionError"], 0.0
            )
            self.assertFalse(delegated["trace"]["hidden_prompt_text_expanded"])

        self.assertEqual(brain.counters["action_retention_checks"], 6)
        self.assertGreaterEqual(brain.counters["action_retention_replays"], 0)
        self.assertEqual(brain.counters["action_retention_failures"], 0)
        self.assertEqual(brain.starter_training_manifest, immutable_manifest)
        persisted_checksum = delegated["trace"]["parameter_checksum_after"]
        brain.events.close()

        reloaded = AdaptiveBrain.load(
            self.root / "starter-action-retention",
            "starter-action-retention",
        )
        self.assertEqual(reloaded.parameter_checksum(), persisted_checksum)
        replayed = reloaded.chat(
            "fork agents to investigate these independent parts",
            max_new_tokens=1,
            seed=2024,
            tool_schemas=schemas,
        )
        self.assertTrue(
            any(
                action["kind"] == "agent"
                and action["toolId"] == "agent.fork"
                for action in replayed["actions"]
            )
        )
        self.assertGreaterEqual(
            replayed["trace"]["action_policy_scores"]["agent"], 0.62
        )

        tool_index = ACTION_KINDS.index("tool")
        subthreshold_logits = torch.zeros(
            (1, len(ACTION_KINDS)),
            dtype=torch.float32,
            device=reloaded.device,
        )
        subthreshold_logits[0, tool_index] = 0.35
        generator = torch.Generator(device=reloaded.device).manual_seed(44)
        subthreshold_language_feature = torch.randn(
            (1, reloaded.config.d_model),
            generator=generator,
            device=reloaded.device,
        )
        subthreshold_internal_feature = torch.randn(
            (1, reloaded.config.d_model),
            generator=generator,
            device=reloaded.device,
        )
        with mock.patch.object(
            reloaded.decoder,
            "forward",
            wraps=reloaded.decoder.forward,
        ) as retention_decoder_forward:
            subthreshold = reloaded._retain_starter_action_policy(
                pre_language_logits=subthreshold_logits,
                pre_internal_logits=subthreshold_logits,
                pre_action_emitted=False,
                post_language_feature=subthreshold_language_feature,
                post_internal_feature=subthreshold_internal_feature,
            )
        self.assertEqual(retention_decoder_forward.call_count, 1)
        self.assertIsNotNone(subthreshold)
        assert subthreshold is not None
        self.assertFalse(subthreshold["actualPreActionEmitted"])
        with torch.no_grad():
            retained_subthreshold_logits = (
                0.35
                * reloaded.decoder.action_policy(
                    subthreshold_language_feature
                )
                + 0.65
                * reloaded.decoder.internal_action_policy(
                    subthreshold_internal_feature
                )
            )
            retained_scores, retained_actions = (
                reloaded._select_structured_actions(
                    retained_subthreshold_logits,
                    schemas=[
                        {
                            "id": "web.search",
                            "actions": ["search"],
                            "grant": "ask",
                        }
                    ],
                    input_text="search the web for primary evidence",
                    assembly_ids=[],
                    organic_state={"computeDemand": 0.7},
                )
            )
        self.assertLess(retained_scores["tool"], 0.62)
        self.assertEqual(retained_actions, [])

        reloaded.config.online_steps = 0
        reloaded.config.growth_novelty_threshold = 0.0
        reloaded.config.growth_patience = 1
        reloaded.config.learn_from_own_messages = True
        reloaded.novelty_streak = 0
        slow_before_disabled_turn = reloaded._parameter_copy()
        topology_before_disabled_turn = tuple(
            (name, tuple(parameter.shape))
            for name, parameter in reloaded.decoder.named_parameters()
        )
        experts_before_disabled_turn = reloaded.decoder.expert_count
        training_steps_before_disabled_turn = reloaded.counters[
            "training_steps"
        ]
        retention_before_disabled_turn = reloaded.counters[
            "action_retention_checks"
        ]
        disabled_turn = reloaded.chat(
            "hello, tell me what you notice",
            max_new_tokens=1,
            seed=3030,
            tool_schemas=schemas,
        )
        self.assertFalse(disabled_turn["trace"]["slow_mutation_applied"])
        disabled_slow_step = next(
            step
            for step in disabled_turn["trace"]["steps"]
            if step["stage"] == "slow-learning"
        )
        self.assertIn("disabled", disabled_slow_step["detail"])
        self.assertEqual(
            disabled_slow_step["value"],
            "online_steps=0; no slow parameter update",
        )
        self.assertIsNone(disabled_turn["trace"]["action_policy_calibration"])
        self.assertFalse(disabled_turn["trace"]["expert_grew"])
        self.assertFalse(
            disabled_turn["trace"]["own_response_expert_grew"]
        )
        self.assertGreater(
            disabled_turn["trace"]["decision_prediction_loss"], 0.0
        )
        self.assertEqual(
            reloaded.counters["training_steps"],
            training_steps_before_disabled_turn,
        )
        self.assertEqual(
            reloaded.counters["action_retention_checks"],
            retention_before_disabled_turn,
        )
        self.assertTrue(
            all(
                torch.equal(before, after)
                for before, after in zip(
                    slow_before_disabled_turn,
                    reloaded._parameter_copy(),
                )
            )
        )
        self.assertEqual(
            reloaded.decoder.expert_count,
            experts_before_disabled_turn,
        )
        self.assertEqual(
            tuple(
                (name, tuple(parameter.shape))
                for name, parameter in reloaded.decoder.named_parameters()
            ),
            topology_before_disabled_turn,
        )

        # The same forced novelty may allocate experts only once slow learning
        # is active. Growth occurs after the deployed action decision and the
        # current-route retention pass must cover the expanded topology.
        reloaded.config.online_steps = 1
        reloaded.novelty_streak = 0
        experts_before_growth_turn = reloaded.decoder.expert_count
        topology_before_growth_turn = tuple(
            (name, tuple(parameter.shape))
            for name, parameter in reloaded.decoder.named_parameters()
        )
        growth_turn = reloaded.chat(
            "fork agents to investigate these independent parts",
            max_new_tokens=1,
            seed=3031,
            tool_schemas=schemas,
        )
        self.assertTrue(growth_turn["trace"]["slow_mutation_applied"])
        self.assertTrue(growth_turn["trace"]["expert_grew"])
        self.assertTrue(growth_turn["trace"]["own_response_expert_grew"])
        self.assertGreater(
            reloaded.decoder.expert_count,
            experts_before_growth_turn,
        )
        self.assertGreater(
            len(tuple(reloaded.decoder.named_parameters())),
            len(topology_before_growth_turn),
        )
        growth_calibration = growth_turn["trace"][
            "action_policy_calibration"
        ]
        self.assertIsNotNone(growth_calibration)
        assert growth_calibration is not None
        self.assertTrue(growth_calibration["calibrated"])
        self.assertTrue(growth_calibration["actualRoutePreserved"])
        self.assertEqual(growth_calibration["canonicalDecoderForwards"], 1)
        self.assertTrue(
            any(
                action["kind"] == "agent"
                for action in growth_turn["actions"]
            )
        )

        head_before_failure = {
            "language": {
                key: value.detach().clone()
                for key, value in reloaded.decoder.action_policy.state_dict().items()
            },
            "internal": {
                key: value.detach().clone()
                for key, value in reloaded.decoder.internal_action_policy.state_dict().items()
            },
        }
        action_names = {
            name
            for name, parameter in reloaded._named_slow_parameters().items()
            if id(parameter)
            in {
                id(item)
                for item in (
                    *reloaded.decoder.action_policy.parameters(),
                    *reloaded.decoder.internal_action_policy.parameters(),
                )
            }
        }
        anchors_before_failure = {
            name: (
                reloaded.slow_anchors[name].clone(),
                reloaded.slow_importance[name].clone(),
            )
            for name in action_names
        }
        counters_before_failure = {
            key: reloaded.counters[key]
            for key in ("training_steps", "metaplastic_updates")
        }
        real_reading = reloaded._action_calibration_reading

        def never_calibrated(*args):
            reading = real_reading(*args)
            reading["minimumLanguageThresholdMargin"] = -1.0
            reading["minimumInternalThresholdMargin"] = -1.0
            reading["minimumDeployedThresholdMargin"] = -1.0
            return reading

        with mock.patch.object(
            reloaded,
            "_action_calibration_reading",
            side_effect=never_calibrated,
        ):
            failed = reloaded._calibrate_starter_action_policy(
                max_steps=1,
                strict=False,
            )
        self.assertFalse(failed["calibrated"])
        self.assertTrue(failed["rolledBack"])
        self.assertEqual(
            {
                key: reloaded.counters[key]
                for key in ("training_steps", "metaplastic_updates")
            },
            counters_before_failure,
        )
        self.assertTrue(
            all(
                torch.equal(value, head_before_failure["language"][key])
                for key, value in reloaded.decoder.action_policy.state_dict().items()
            )
        )
        self.assertTrue(
            all(
                torch.equal(value, head_before_failure["internal"][key])
                for key, value in reloaded.decoder.internal_action_policy.state_dict().items()
            )
        )
        self.assertTrue(
            all(
                torch.equal(reloaded.slow_anchors[name], anchor)
                and torch.equal(reloaded.slow_importance[name], importance)
                for name, (anchor, importance) in anchors_before_failure.items()
            )
        )

        # A failed retention gate rejects the entire enclosing slow mutation,
        # not only the action heads. Force both deferred expert growth and a
        # retention failure, then prove the valid fast turn survives while all
        # slow tensors/topology/optimizer/stability/counters return exactly to
        # their pre-transaction state and only that restored state is saved.
        reloaded.config.online_steps = 1
        reloaded.config.growth_novelty_threshold = 0.0
        reloaded.config.growth_patience = 1
        reloaded.config.learn_from_own_messages = True
        reloaded.novelty_streak = 0
        slow_parameters_before_transaction = {
            "%s.%s" % (prefix, name): parameter.detach().cpu().clone()
            for prefix, module in reloaded._slow_transaction_modules().items()
            for name, parameter in module.named_parameters()
        }
        slow_shapes_before_transaction = {
            name: tuple(parameter.shape)
            for name, parameter in slow_parameters_before_transaction.items()
        }
        slow_checksum_before_transaction = (
            reloaded._slow_parameter_checksum()
        )
        experts_before_transaction = reloaded.decoder.expert_count
        optimizer_before_transaction = copy.deepcopy(
            reloaded._optimizer.state_dict()
        )
        anchors_before_transaction = {
            name: value.clone()
            for name, value in reloaded.slow_anchors.items()
        }
        importance_before_transaction = {
            name: value.clone()
            for name, value in reloaded.slow_importance.items()
        }
        slow_counters_before_transaction = {
            name: reloaded.counters[name]
            for name in (
                "training_steps",
                "metaplastic_updates",
                "action_retention_checks",
                "action_retention_replays",
                "action_retention_failures",
            )
        }
        messages_before_transaction = len(reloaded.messages)
        experiences_before_transaction = reloaded.counters["experiences"]
        assemblies_before_transaction = len(reloaded.memory.assemblies)

        def forced_retention_failure(**_kwargs):
            # Ensure the outer transaction restores mutations performed inside
            # retention as well as the preceding shared training and growth.
            with torch.no_grad():
                next(
                    iter(reloaded.decoder.action_policy.parameters())
                ).add_(3.0)
                reloaded.memory_bridge.bias.add_(0.25)
            return {
                "mode": (
                    "exact-route-self-distillation+current-neural-replay"
                ),
                "calibrated": False,
                "rolledBack": True,
                "failureType": "ForcedRetentionFailure",
                "failure": "forced full-transaction regression",
            }

        with mock.patch.object(
            reloaded,
            "_retain_starter_action_policy",
            side_effect=forced_retention_failure,
        ):
            rejected_turn = reloaded.chat(
                "fork agents while retaining this new fast neural experience",
                max_new_tokens=1,
                seed=4040,
                tool_schemas=schemas,
            )

        rejected_trace = rejected_turn["trace"]
        self.assertTrue(rejected_trace["slow_mutation_requested"])
        self.assertFalse(rejected_trace["slow_mutation_applied"])
        self.assertTrue(rejected_trace["slow_mutation_rolled_back"])
        self.assertEqual(rejected_trace["slow_mutation_stage"], "rolled-back")
        self.assertEqual(
            rejected_trace["slow_mutation_failure"]["stage"],
            "action-retention",
        )
        self.assertTrue(
            rejected_trace["action_policy_calibration"][
                "transactionRolledBack"
            ]
        )
        self.assertFalse(rejected_trace["expert_grew"])
        self.assertFalse(rejected_trace["own_response_expert_grew"])
        self.assertEqual(
            rejected_trace["slow_parameter_checksum_before"],
            slow_checksum_before_transaction,
        )
        self.assertEqual(
            rejected_trace["slow_parameter_checksum_after"],
            slow_checksum_before_transaction,
        )
        slow_step = next(
            step
            for step in rejected_trace["steps"]
            if step["stage"] == "slow-learning"
        )
        self.assertIn("restored all slow parameters", slow_step["detail"])
        self.assertIn("rolled back", slow_step["value"])

        slow_parameters_after_transaction = {
            "%s.%s" % (prefix, name): parameter.detach().cpu()
            for prefix, module in reloaded._slow_transaction_modules().items()
            for name, parameter in module.named_parameters()
        }
        self.assertEqual(
            set(slow_parameters_after_transaction),
            set(slow_parameters_before_transaction),
        )
        self.assertEqual(
            {
                name: tuple(parameter.shape)
                for name, parameter in slow_parameters_after_transaction.items()
            },
            slow_shapes_before_transaction,
        )
        self.assertTrue(
            all(
                torch.equal(
                    parameter,
                    slow_parameters_before_transaction[name],
                )
                for name, parameter in slow_parameters_after_transaction.items()
            )
        )
        self.assertEqual(
            reloaded._slow_parameter_checksum(),
            slow_checksum_before_transaction,
        )
        self.assertEqual(
            reloaded.decoder.expert_count,
            experts_before_transaction,
        )
        expected_slow_counters = dict(slow_counters_before_transaction)
        expected_slow_counters["action_retention_checks"] += 1
        expected_slow_counters["action_retention_failures"] += 1
        self.assertEqual(
            {
                name: reloaded.counters[name]
                for name in slow_counters_before_transaction
            },
            expected_slow_counters,
        )
        self.assertEqual(
            set(reloaded.slow_anchors), set(anchors_before_transaction)
        )
        self.assertEqual(
            set(reloaded.slow_importance), set(importance_before_transaction)
        )
        self.assertTrue(
            all(
                torch.equal(reloaded.slow_anchors[name], value)
                for name, value in anchors_before_transaction.items()
            )
        )
        self.assertTrue(
            all(
                torch.equal(reloaded.slow_importance[name], value)
                for name, value in importance_before_transaction.items()
            )
        )

        def assert_optimizer_tree_equal(expected, actual):
            if isinstance(expected, torch.Tensor):
                self.assertIsInstance(actual, torch.Tensor)
                self.assertTrue(torch.equal(expected.cpu(), actual.cpu()))
                return
            if isinstance(expected, dict):
                self.assertEqual(set(expected), set(actual))
                for key in expected:
                    assert_optimizer_tree_equal(expected[key], actual[key])
                return
            if isinstance(expected, (list, tuple)):
                self.assertEqual(len(expected), len(actual))
                for expected_item, actual_item in zip(expected, actual):
                    assert_optimizer_tree_equal(expected_item, actual_item)
                return
            self.assertEqual(expected, actual)

        assert_optimizer_tree_equal(
            optimizer_before_transaction,
            reloaded._optimizer.state_dict(),
        )
        # The rollback boundary begins after fast admission, so this valid turn
        # still advances conversation, substrate assemblies, and experiences.
        self.assertEqual(
            len(reloaded.messages), messages_before_transaction + 2
        )
        self.assertEqual(
            reloaded.counters["experiences"],
            experiences_before_transaction + 2,
        )
        self.assertGreater(
            len(reloaded.memory.assemblies), assemblies_before_transaction
        )
        self.assertTrue(rejected_turn["text"])

        reloaded.events.close()
        persisted_after_rejection = AdaptiveBrain.load(
            self.root / "starter-action-retention",
            "starter-action-retention",
        )
        self.assertEqual(
            persisted_after_rejection._slow_parameter_checksum(),
            slow_checksum_before_transaction,
        )
        self.assertEqual(
            persisted_after_rejection.decoder.expert_count,
            experts_before_transaction,
        )
        self.assertEqual(
            len(persisted_after_rejection.messages),
            messages_before_transaction + 2,
        )
        reloaded = persisted_after_rejection
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
