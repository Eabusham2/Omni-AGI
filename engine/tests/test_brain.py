import copy
import hashlib
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
from omni_core.brain import ChatGenerationCancelled
from omni_core.datasets import sqlite_consistent_snapshot
from omni_core.model import ACTION_KINDS, PackedAdaptiveBitLinear
from omni_core.offload import ResourcePolicy
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
        # A micro fixture is not a product Build: no curriculum training or
        # origin promotion is performed by this constructor.
        return AdaptiveBrain("brain-test", self.root, config)


    def test_parameter_accounting_is_shared_by_runtime_and_metrics_without_state_double_counting(
        self,
    ):
        brain = self.make_brain()
        core_parameters = brain._core_parameter_map()
        self.assertEqual(core_parameters, {})
        trainable_parameters = {
            id(parameter): parameter
            for module in brain._trainable_modules()
            for parameter in module.parameters()
        }
        self.assertEqual(trainable_parameters, {})
        expected_mutable = brain._packed_logical_parameter_count(
            brain._trainable_modules()
        )
        core_packed = brain._packed_logical_parameter_count(
            (
                brain.decoder,
                brain.memory_bridge,
                brain.idea_adapter,
                brain.liquid,
                brain.modalities,
            )
        )
        router_parameters = brain._packed_logical_parameter_count((brain.router,))
        self.assertGreater(router_parameters, 0)
        self.assertEqual(expected_mutable, core_packed + router_parameters)

        accounting = brain.parameter_accounting()
        expected_vector_parameters = (
            len(brain.memory.neuron_vectors) * brain.memory.space.dimensions
        )
        self.assertEqual(accounting["mutableDenseParameters"], expected_mutable)
        self.assertEqual(accounting["substrateVectorParameters"], expected_vector_parameters)
        self.assertEqual(
            accounting["dynamicSparseSynapses"],
            len(brain.memory.synapses),
        )
        self.assertEqual(
            accounting["substrateDynamicSparseSynapses"],
            len(brain.memory.synapses),
        )
        self.assertEqual(
            accounting["totalNeuralParameters"],
            expected_mutable + expected_vector_parameters + len(brain.memory.synapses),
        )
        self.assertEqual(brain.runtime_card()["parameterAccounting"], accounting)
        metrics = brain.metrics()
        self.assertEqual(metrics["parameterAccounting"], accounting)
        self.assertEqual(
            metrics["trainableParameters"],
            expected_mutable,
        )

        # Non-weight buffers are never counted as extra learned parameters.
        brain.decoder.register_buffer(
            "_parameter_accounting_test_buffer",
            torch.zeros(17),
        )
        self.assertEqual(brain.parameter_accounting(), accounting)
        brain.events.close()

    def test_gpu_profile_counts_only_native_trainable_parameters(self):
        config = OmniConfig.micro(
            origin_kind="ground-up",
            hardware_tier="gpu",
            device="cpu",
            d_model=96,
            n_heads=8,
            n_layers=4,
            d_ff=288,
            idea_dim=96,
            vsa_dim=384,
            router_neurons=96,
            working_memory_slots=65_536,
            memory_resident_items=65_536,
            image_size=32,
            audio_samples=512,
            video_frames=6,
            modality_channels=24,
            max_seq_len=40,
        )
        brain = AdaptiveBrain("gpu-parameter-count", self.root, config)

        accounting = brain.parameter_accounting()

        core_parameter_count = brain._packed_logical_parameter_count(
            (
                brain.decoder,
                brain.memory_bridge,
                brain.idea_adapter,
                brain.liquid,
                brain.modalities,
            )
        )
        router_parameter_count = brain._packed_logical_parameter_count((brain.router,))
        self.assertGreater(core_parameter_count, 0)
        self.assertGreater(router_parameter_count, 0)
        self.assertEqual(
            accounting["mutableDenseParameters"],
            core_parameter_count + router_parameter_count,
        )
        self.assertEqual(accounting["floatingTrainableParameters"], 0)
        self.assertGreater(
            sum(tensor.numel() for tensor in brain._core_tensors().values()), 0
        )
        brain.events.close()

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

    def test_committed_desktop_sqlite_snapshot_trains_under_its_exact_hash(self):
        brain = self.make_brain(
            memory_recipe="synapses-only",
            retain_source_text=False,
        )
        source = self.root / "live-source.sqlite3"
        writer = sqlite3.connect(source)
        try:
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("PRAGMA wal_autocheckpoint=0")
            writer.execute("CREATE TABLE facts (id INTEGER PRIMARY KEY, text TEXT)")
            writer.executemany(
                "INSERT INTO facts VALUES (?, ?)",
                [(2, "The second learned fact."), (1, "The first learned fact.")],
            )
            writer.commit()
            with sqlite_consistent_snapshot(source) as snapshot:
                with mock.patch.object(
                    brain,
                    "_can_retain_bundled_action_policy",
                    return_value=False,
                ):
                    result = brain.ingest(
                        path=str(snapshot.path),
                        kind="sqlite",
                        policy="encode",
                        expected_hash=snapshot.sha256,
                        committed_sqlite_snapshot=True,
                    )
                self.assertEqual(result["source"]["content_hash"], snapshot.sha256)
                self.assertEqual(result["coverage"]["processedRecords"], 2)
                self.assertEqual(result["coverage"]["rejectedRecords"], 0)
                self.assertTrue(result["coverage"]["complete"])
                self.assertFalse(result["source"]["raw_text_retained"])
        finally:
            writer.close()
            brain.events.close()




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

    def test_complete_custom_runtime_has_no_dense_linear_or_convolution_blocker(self):
        brain = self.make_brain()
        audit = brain.require_complete_packed_runtime()
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["denseConvolutionBlockers"], [])
        self.assertGreater(audit["packedBitLinearModules"], 0)
        self.assertGreater(audit["packedConvolutionModules"], 0)
        self.assertFalse(audit["denseBf16LinearWeightMaterialized"])
        brain.events.close()

    def test_fractional_dynamic_synapse_fails_audit_and_packed_export(self):
        brain = self.make_brain()
        brain.learn_experience(
            "Exact live synapses connect every distributed assembly.",
            steps=0,
        )
        synapse_id = sorted(brain.memory.synapses)[0]
        brain.memory.synapses[synapse_id]["effective_weight"] = 0.5

        audit = brain._ternary_audit()
        self.assertIn(
            "substrate.dynamic_synapses.weights", audit["violations"]
        )
        self.assertLess(audit["coverage"], 1.0)
        with self.assertRaisesRegex(ValueError, "exact ternary"):
            brain._dynamic_synapse_export()
        destination = self.root / "invalid-packed-ternary"
        with self.assertRaisesRegex(RuntimeError, "coverage audit failed"):
            brain.export_packed_ternary(destination)
        self.assertFalse(destination.exists())
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
        (
            reloaded_values,
            reloaded_count,
            reloaded_order_sha,
            reloaded_order_basis,
        ) = reloaded._dynamic_synapse_pack_state()
        packed_values = verified.tensors[
            "substrate.dynamic_synapses.weights"
        ]
        self.assertEqual(
            verified.manifest["metadata"]["parameterChecksum"],
            expected_checksum,
        )
        self.assertEqual(
            verified.manifest["metadata"]["dynamicSynapseCount"],
            reloaded_count,
        )
        self.assertEqual(
            verified.manifest["metadata"]["dynamicSynapseOrderSha256"],
            reloaded_order_sha,
        )
        self.assertEqual(
            verified.manifest["metadata"]["dynamicSynapseOrderBasis"],
            reloaded_order_basis,
        )
        self.assertTrue(torch.equal(packed_values, reloaded_values))
        self.assertEqual(
            int(reloaded.memory.synapses[target_id]["effective_weight"]),
            replacement,
        )
        reloaded.events.close()

    def test_packed_core_checkpoint_reloads_without_floating_master(self):
        brain = self.make_brain()
        projection = brain.decoder.action_policy.hidden
        levels = projection.effective_weight().clone()
        levels[0, 0] = 1 if int(levels[0, 0]) != 1 else -1
        projection.set_ternary_weight_(levels)
        brain.save()
        checksum = brain.parameter_checksum()
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertTrue(torch.equal(
            reloaded.decoder.action_policy.hidden.effective_weight(), levels
        ))
        self.assertEqual(reloaded.parameter_checksum(), checksum)
        self.assertEqual(
            reloaded.parameter_accounting()["floatingTrainableParameters"], 0
        )
        verified = verify_ternary_shards(
            reloaded.engine_path / "packed-ternary"
        )
        self.assertEqual(
            verified.manifest["metadata"]["parameterChecksum"], checksum
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

    def test_chat_cooperative_cancel_before_generation_preserves_uncommitted_state(self):
        brain = self.make_brain()
        before_checksum = brain.parameter_checksum()
        before_counts = (
            len(brain.memory.neurons),
            len(brain.memory.assemblies),
            len(brain.memory.synapses),
            brain.counters["inference_count"],
        )

        with self.assertRaises(ChatGenerationCancelled):
            brain.chat(
                "cancel this turn before it mutates",
                cancel_check=lambda: True,
            )

        self.assertEqual(brain.parameter_checksum(), before_checksum)
        self.assertEqual(
            (
                len(brain.memory.neurons),
                len(brain.memory.assemblies),
                len(brain.memory.synapses),
                brain.counters["inference_count"],
            ),
            before_counts,
        )
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
            "generation_elapsed_ms",
            "generated_token_count",
            "generation_tokens_per_second",
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
        self.assertGreater(trace["generation_elapsed_ms"], 0)
        self.assertGreater(trace["generated_token_count"], 0)
        self.assertGreater(trace["generation_tokens_per_second"], 0)
        self.assertEqual(
            result["runtimeCard"]["measured_generation"]["generatedTokens"],
            trace["generated_token_count"],
        )
        self.assertEqual(
            trace["mechanism_order"], "event-derived-unordered"
        )
        first_mechanisms = [step["stage"] for step in trace["steps"]]
        self.assertNotIn("capability-conditioning", first_mechanisms)
        conditioned = brain.chat(
            "Search capability is now available",
            max_new_tokens=2,
            seed=5,
            tool_schemas=[
                {
                    "id": "web.search",
                    "actions": ["search"],
                    "grant": "ask",
                }
            ],
        )
        conditioned_mechanisms = [
            step["stage"] for step in conditioned["trace"]["steps"]
        ]
        self.assertIn("capability-conditioning", conditioned_mechanisms)
        self.assertNotEqual(first_mechanisms, conditioned_mechanisms)
        brain.events.close()




    def test_committed_chat_turn_receipt_is_persisted_and_idempotent(self):
        brain = self.make_brain(online_learning=False)
        turn_id = "turn-receipt-fixture"
        input_text = "Persist this exact turn receipt."

        result = brain.chat(
            input_text,
            max_new_tokens=2,
            seed=29,
            turn_id=turn_id,
        )

        receipt = result["turnReceipt"]
        self.assertTrue(result["turnCommitted"])
        self.assertFalse(result["idempotentCompletion"])
        self.assertEqual(receipt["turnId"], turn_id)
        self.assertEqual(result["humanMessage"], brain.messages[-2])
        self.assertEqual(
            receipt["humanMessageId"], brain.messages[-2]["id"]
        )
        self.assertEqual(
            receipt["brainMessageId"], brain.messages[-1]["id"]
        )
        self.assertEqual(receipt["traceId"], result["trace"]["id"])
        self.assertNotIn("content", receipt)
        before_messages = list(brain.messages)
        before_inferences = int(brain.counters["inference_count"])
        brain.events.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        repeated = reloaded.chat(
            input_text,
            max_new_tokens=2,
            seed=999,
            turn_id=turn_id,
        )

        self.assertTrue(repeated["turnCommitted"])
        self.assertTrue(repeated["idempotentCompletion"])
        self.assertEqual(repeated["turnReceipt"], receipt)
        self.assertEqual(repeated["humanMessage"]["id"], receipt["humanMessageId"])
        self.assertEqual(repeated["message"]["id"], receipt["brainMessageId"])
        self.assertEqual(repeated["trace"]["id"], receipt["traceId"])
        self.assertEqual(reloaded.messages, before_messages)
        self.assertEqual(
            reloaded.counters["inference_count"], before_inferences
        )
        follow_up = reloaded.chat(
            "different input",
            max_new_tokens=1,
            seed=31,
            turn_id=turn_id,
        )
        self.assertFalse(follow_up["idempotentCompletion"])
        self.assertEqual(follow_up["turnReceipt"]["turnId"], turn_id)
        self.assertNotEqual(
            follow_up["turnReceipt"]["inputSha256"], receipt["inputSha256"]
        )
        self.assertEqual(
            reloaded.counters["inference_count"], before_inferences + 1
        )
        repeated_follow_up = reloaded.chat(
            "different input",
            max_new_tokens=1,
            seed=999,
            turn_id=turn_id,
        )
        self.assertTrue(repeated_follow_up["idempotentCompletion"])
        self.assertEqual(
            repeated_follow_up["turnReceipt"], follow_up["turnReceipt"]
        )
        self.assertEqual(
            reloaded.counters["inference_count"], before_inferences + 1
        )
        persisted_metadata = json.loads(
            (self.root / "engine" / "brain.json").read_text("utf-8")
        )
        persisted_receipts = AdaptiveBrain._validated_completed_chat_turns(
            persisted_metadata["completed_chat_turns"]
        )
        self.assertEqual(len(persisted_receipts), 2)
        self.assertEqual(
            {
                (value["turnId"], value["inputSha256"])
                for value in persisted_receipts
            },
            {
                (turn_id, receipt["inputSha256"]),
                (turn_id, follow_up["turnReceipt"]["inputSha256"]),
            },
        )
        reloaded.events.close()



    def test_fresh_attention_never_acknowledges_a_precommit_save_failure(self):
        brain = self.make_brain(online_learning=False)
        brain.messages = [
            {
                "id": "visible-history",
                "role": "human",
                "content": "Visible history remains on disk.",
                "created_at": "2026-09-07T07:20:00Z",
            }
        ]
        brain.save()

        with mock.patch.object(
            brain, "save", side_effect=RuntimeError("precommit failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "precommit failure"):
                brain.start_fresh_attention("fresh-precommit-failure")
            with self.assertRaisesRegex(
                RuntimeError, "not atomically committed"
            ):
                brain.start_fresh_attention("fresh-precommit-failure")
        persisted = json.loads(
            (brain.engine_path / "brain.json").read_text("utf-8")
        )
        self.assertIsNone(persisted.get("fresh_attention_boundary"))
        brain.close()

    def test_fresh_attention_reconciles_a_postcommit_publish_failure(self):
        brain = self.make_brain(online_learning=False)
        brain.messages = [
            {
                "id": "visible-history",
                "role": "human",
                "content": "Keep this visible after the boundary.",
                "created_at": "2026-09-07T07:30:00Z",
            }
        ]
        brain._append_recent_dialogue(
            "Keep this visible after the boundary.", "Acknowledged."
        )
        brain.save()

        with mock.patch.object(
            brain.state_store,
            "publish",
            side_effect=RuntimeError("lost postcommit acknowledgement"),
        ):
            result = brain.start_fresh_attention("fresh-postcommit-failure")

        self.assertTrue(result["committed"])
        self.assertEqual(result["boundary"]["epoch"], 1)
        self.assertEqual(brain.recent_token_context, [])
        brain.close()
        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(
            reloaded.fresh_attention_boundary["operationId"],
            "fresh-postcommit-failure",
        )
        self.assertEqual(reloaded.messages, [])
        self.assertEqual(
            reloaded.conversation.payload_by_id("message", "visible-history")[
                "content"
            ],
            "Keep this visible after the boundary.",
        )
        reloaded.close()

    def test_fresh_attention_recovers_deferred_paged_scratch_cleanup_on_reload(self):
        brain = self.make_brain(online_learning=False)
        brain.paged_working_memory.append(
            torch.ones(brain.config.idea_dim),
            {"assemblyId": "temporary-page", "source": "fixture"},
        )
        self.assertEqual(brain.paged_working_memory.count(), 1)
        brain.save()

        with mock.patch.object(
            brain.paged_working_memory,
            "clear",
            side_effect=sqlite3.OperationalError("temporary pager lock"),
        ):
            result = brain.start_fresh_attention(
                "fresh-deferred-page-cleanup"
            )

        self.assertTrue(result["committed"])
        self.assertTrue(result["pagedCleanupPending"])
        persisted = json.loads(
            (brain.engine_path / "brain.json").read_text("utf-8")
        )
        self.assertEqual(persisted["paged_working_memory"]["count"], 0)
        self.assertEqual(brain.paged_working_memory.count(), 1)
        brain.close()

        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(reloaded.paged_working_memory.count(), 0)
        self.assertEqual(
            reloaded.fresh_attention_boundary["operationId"],
            "fresh-deferred-page-cleanup",
        )
        reloaded.close()


    def test_explicit_response_length_parser_is_typed_not_prompt_injection(self):
        literal = AdaptiveBrain._explicit_response_length_constraint(
            "Reply with only remembered."
        )
        self.assertEqual(literal["kind"], "literal")
        self.assertEqual(literal["count"], 1)
        self.assertEqual(literal["literal"], "remembered")
        words = AdaptiveBrain._explicit_response_length_constraint(
            "Answer in exactly two words."
        )
        self.assertEqual(words, {"kind": "count", "unit": "word", "count": 2})
        sentences = AdaptiveBrain._explicit_response_length_constraint(
            "Respond with no more than 3 sentences."
        )
        self.assertEqual(
            sentences,
            {"kind": "count", "unit": "sentence", "count": 3},
        )
        self.assertIsNone(
            AdaptiveBrain._explicit_response_length_constraint(
                "Explain why plants need sunlight."
            )
        )
        for reference in (
            "Reply only with the code.",
            "Answer with the answer.",
            "Respond using the color.",
            "Reply only with it.",
            "Reply only with the call sign.",
            "Reply only with the exact answer.",
        ):
            with self.subTest(reference=reference):
                self.assertIsNone(
                    AdaptiveBrain._explicit_response_length_constraint(reference)
                )
        for inactive in (
            "Do not answer in two words.",
            'The book says "reply in two words".',
            "If you reply in two words, the test is invalid.",
        ):
            with self.subTest(inactive=inactive):
                self.assertIsNone(
                    AdaptiveBrain._explicit_response_length_constraint(inactive)
                )

    def test_ground_up_literal_budget_uses_native_boundary_tokens(self):
        brain = self.make_brain()
        literal = "ORCHID-7421"
        budget, trace = brain._response_generation_budget(
            f"Reply only with {literal}.",
            cognitive_demand=1.0,
            caller_limit=64,
        )
        self.assertGreaterEqual(budget, len(brain.tokenizer.encode(literal)))
        self.assertEqual(trace["explicitConstraint"]["kind"], "literal")
        brain.events.close()




    def test_precommit_phase_cancellation_does_not_admit_fast_experience(self):
        brain = self.make_brain(online_learning=False)
        prompt = "A valid provisional response must remain uncommitted."
        brain.learn_experience(
            prompt,
            kind="experience",
            source="fixture",
            steps=0,
        )
        brain.current_context = {"sentinel": "precommit"}
        brain.decoder.train(True)
        before = {
            "checksum": brain.parameter_checksum(),
            "counters": copy.deepcopy(brain.counters),
            "memoryRevision": brain.memory.state_revision,
            "replay": brain.replay.checkpoint(),
            "pagedWorking": brain.paged_working_memory.checkpoint(),
            "workingMemory": [value.clone() for value in brain.working_memory],
            "workspaceItems": copy.deepcopy(brain.workspace_items),
            "context": copy.deepcopy(brain.current_context),
        }
        streamed = []

        def cancel_at_phase(kind, payload):
            streamed.append((kind, dict(payload)))
            if kind == "phase":
                raise RuntimeError("pre-commit cancellation")

        with self.assertRaisesRegex(RuntimeError, "pre-commit cancellation"):
            brain.chat(
                prompt,
                max_new_tokens=8,
                seed=53,
                stream_callback=cancel_at_phase,
            )

        self.assertEqual([kind for kind, _payload in streamed], ["token", "phase"])
        self.assertEqual(brain.parameter_checksum(), before["checksum"])
        self.assertEqual(brain.counters, before["counters"])
        self.assertEqual(brain.memory.state_revision, before["memoryRevision"])
        self.assertEqual(brain.replay.checkpoint(), before["replay"])
        self.assertEqual(
            brain.paged_working_memory.checkpoint(), before["pagedWorking"]
        )
        self.assertEqual(brain.workspace_items, before["workspaceItems"])
        self.assertEqual(brain.current_context, before["context"])
        self.assertEqual(brain.messages, [])
        self.assertEqual(brain.traces, [])
        self.assertEqual(len(brain.working_memory), len(before["workingMemory"]))
        for current, expected in zip(
            brain.working_memory, before["workingMemory"]
        ):
            self.assertTrue(torch.equal(current, expected))
        self.assertTrue(brain.decoder.training)
        brain.events.close()

    def test_default_response_budget_is_state_scaled_not_the_context_ceiling(self):
        brain = AdaptiveBrain(
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
        self.assertFalse(first["assemblies"][0]["retainsSourceText"])
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

    def test_low_score_idle_state_does_not_force_ponder(self):
        brain = self.make_brain()
        brain.learn_experience("A simple settled association.", steps=0)
        scores = {
            "talk": 0.86,
            "tool": 0.02,
            "imagine": 0.02,
            "agent": 0.02,
            "ponder": 0.02,
            "learn": 0.02,
            "evolve": 0.02,
            "stop": 0.02,
        }
        with mock.patch.object(
            brain,
            "_select_structured_actions",
            return_value=(scores, []),
        ), mock.patch.object(
            brain,
            "_organic_state",
            return_value={
                "tension": 0.0,
                "curiosity": 0.0,
                "uncertainty": 0.0,
                "novelty": 0.0,
                "predictionError": 0.0,
                "learningProgress": 0.0,
                "activeFraction": 0.1,
            },
        ):
            result = brain.idle_cycle(minimum_idle_seconds=0)
        self.assertEqual(result["actions"], [])
        self.assertNotIn("ponder", result["trace"]["proposedActionKinds"])
        brain.events.close()

    def test_idle_ponder_is_completed_in_cycle_without_recursive_action(self):
        brain = self.make_brain()
        brain.learn_experience("An unresolved association remains active.", steps=0)
        captured = {}
        original_idea_model_vector = brain._idea_model_vector

        def capture_idea_model_vector(vector):
            result = original_idea_model_vector(vector)
            captured["modelIdea"] = result.detach().clone()
            return result

        scores = {
            "talk": 0.001,
            "tool": 0.001,
            "imagine": 0.001,
            "agent": 0.001,
            "ponder": 0.993,
            "learn": 0.001,
            "evolve": 0.001,
            "stop": 0.001,
        }
        proposed = [
            {
                "kind": "ponder",
                "arguments": {"organic": True},
                "confidence": scores["ponder"],
            }
        ]
        with mock.patch.object(
            brain,
            "_select_structured_actions",
            return_value=(scores, proposed),
        ), mock.patch.object(
            brain,
            "_idea_model_vector",
            side_effect=capture_idea_model_vector,
        ), mock.patch.object(
            brain.decoder.internal_action_policy,
            "forward",
            wraps=brain.decoder.internal_action_policy.forward,
        ) as action_policy_forward:
            result = brain.idle_cycle(minimum_idle_seconds=0)

        self.assertEqual(result["actions"], [])
        self.assertEqual(result["trace"]["proposedActionKinds"], [])
        self.assertEqual(
            result["trace"]["internallySettledActionKinds"], ["ponder"]
        )
        self.assertEqual(
            result["trace"]["actionPolicyFeatureChannel"],
            "assembly-model",
        )
        self.assertTrue(
            torch.equal(
                action_policy_forward.call_args.args[0],
                captured["modelIdea"],
            )
        )
        brain.events.close()

    def test_idle_visible_actions_have_a_refractory_period(self):
        brain = self.make_brain()
        brain.learn_experience(
            "An unresolved visual association remains active.",
            steps=0,
        )
        scores = {
            "talk": 0.01,
            "tool": 0.92,
            "imagine": 0.01,
            "agent": 0.01,
            "ponder": 0.01,
            "learn": 0.01,
            "evolve": 0.01,
            "stop": 0.02,
        }
        proposed = [
            {
                "kind": "tool",
                "toolId": "studio.ui",
                "action": "open-creativity",
                "arguments": {"organic": True},
                "confidence": 0.92,
            }
        ]
        with mock.patch.object(
            brain,
            "_select_structured_actions",
            return_value=(scores, proposed),
        ):
            first = brain.idle_cycle(minimum_idle_seconds=0)
            second = brain.idle_cycle(minimum_idle_seconds=0)

        self.assertEqual(len(first["actions"]), 1)
        self.assertEqual(first["actions"][0]["action"], "open-creativity")
        self.assertEqual(second["actions"], [])
        self.assertTrue(second["ran"])
        self.assertGreater(second["trace"]["stdpUpdate"], 0.0)
        brain.events.close()

    def test_candidate_exception_rolls_back_all_core_parameters(self):
        brain = self.make_brain()
        before = brain.parameter_checksum()

        def explode(*args, **kwargs):
            del args, kwargs
            projection = brain.decoder.action_policy.hidden
            levels = projection.effective_weight().clone()
            levels[0, 0] = 1 if int(levels[0, 0]) != 1 else -1
            projection.set_ternary_weight_(levels)
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
        right = AdaptiveBrain(
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
                if isinstance(module, PackedAdaptiveBitLinear)
            )
        )
        learned = brain.learn_experience("transient dense fixture", steps=0)
        self.assertGreaterEqual(learned["spiking"]["spikes"], 0.0)
        self.assertGreater(len(brain.memory.ideas), 0)
        self.assertEqual(learned["training"]["loss"], 0.0)
        self.assertFalse(hasattr(brain, "consolidate"))
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
        self.assertEqual(result["mediaCoverage"]["sensoryAssemblies"], 4)
        self.assertTrue(result["source"]["media_records"][0]["sensory"])
        self.assertGreaterEqual(result["source"]["learned_ideas"], 4)
        self.assertGreaterEqual(result["source"]["learned_concepts"], 1)
        self.assertTrue(
            any(
                neuron.get("label") == "audio-perception"
                for neuron in brain.memory.neurons.values()
            )
        )
        self.assertEqual(result["coverage"]["processedRecords"], 1)
        # Two learning steps see each window. The final two calls therefore
        # prove the non-full tail was admitted rather than silently discarded.
        self.assertEqual(len(observed), 6)
        tail_target = observed[-1].flatten()
        self.assertTrue(torch.all(tail_target[:tail] > 0.4))
        self.assertTrue(torch.all(tail_target[tail:] == 0.0))
        whole_assembly_id = result["source"]["media_records"][0]["coverage"][
            "wholeRecordAssemblyId"
        ]
        brain.events.close()
        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertIn(whole_assembly_id, reloaded.memory.assembly_vectors)
        self.assertGreater(reloaded.modality_training["audio"], 0)
        reloaded.events.close()

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
                "errorCount": 1,
                "errorsTruncated": False,
                "complete": True,
            },
        )
        self.assertTrue(
            any("corrupt image fixture" in warning for warning in result["warnings"])
        )
        brain.events.close()

    def test_video_ingestion_trains_visual_and_embedded_audio_tracks(self):
        brain = self.make_brain(video_enabled=True, audio_enabled=True)
        video = torch.linspace(
            -1.0,
            1.0,
            3
            * brain.config.video_frames
            * brain.config.image_size
            * brain.config.image_size,
        ).reshape(
            1,
            3,
            brain.config.video_frames,
            brain.config.image_size,
            brain.config.image_size,
        )
        audio = torch.sin(
            torch.linspace(0.0, 6.0, brain.config.audio_samples)
        ).reshape(1, 1, -1)
        video_before = {
            key: value.detach().clone()
            for key, value in brain.modalities.video.state_dict().items()
        }
        audio_before = {
            key: value.detach().clone()
            for key, value in brain.modalities.audio.state_dict().items()
        }

        with mock.patch.object(
            brain,
            "_iter_video_windows",
            return_value=iter([(video, brain.config.video_frames)]),
        ), mock.patch.object(
            brain,
            "_iter_audio_windows",
            return_value=iter([(audio, brain.config.audio_samples)]),
        ), mock.patch.object(
            brain, "_assert_embedded_audio_track", return_value=None,
        ):
            trained = brain._train_media(
                "movie-with-sound.mp4",
                "video",
                "harbor movie",
                steps=1,
                content_sha256="b" * 64,
            )

        self.assertTrue(trained["trained"])
        self.assertEqual(trained["primarySteps"], 1)
        self.assertEqual(trained["steps"], 2)
        self.assertTrue(trained["embeddedAudio"]["detected"])
        self.assertTrue(trained["embeddedAudio"]["trained"])
        self.assertEqual(
            trained["coverage"]["embeddedAudio"]["coverage"][
                "processedSamples"
            ],
            brain.config.audio_samples,
        )
        self.assertTrue(
            any(
                not torch.equal(value, video_before[key])
                for key, value in brain.modalities.video.state_dict().items()
            )
        )
        self.assertTrue(
            any(
                not torch.equal(value, audio_before[key])
                for key, value in brain.modalities.audio.state_dict().items()
            )
        )
        self.assertGreater(brain.modality_training["video"], 0)
        self.assertGreater(brain.modality_training["audio"], 0)
        labels = {
            str(neuron.get("label", ""))
            for neuron in brain.memory.neurons.values()
        }
        self.assertIn("video-perception", labels)
        self.assertIn("audio-perception", labels)
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
        self.assertFalse(video["synchronizedAudio"]["supported"])
        self.assertFalse(video["synchronizedAudio"]["generated"])
        self.assertFalse(video["synchronizedAudio"]["speechSynthesis"])
        self.assertFalse(video["synchronizedAudio"]["hiddenBehavioralPrompt"])
        brain.events.close()

    def test_video_generation_muxes_length_aligned_same_idea_audio(self):
        brain = self.make_brain()
        brain.modality_training["video"] = 1
        brain.modality_training["audio"] = 1
        generated = brain.generate_modality(
            "video",
            prompt="a moving shape with an abstract sound",
            seed=31,
            settings={"fps": 8, "sampleRate": 16_000},
        )

        synchronized = generated["synchronizedAudio"]
        self.assertTrue(synchronized["requested"])
        self.assertTrue(synchronized["supported"])
        self.assertTrue(synchronized["sameBrainIdea"])
        self.assertTrue(synchronized["lengthAlignedToVideo"])
        self.assertFalse(synchronized["speechSynthesis"])
        self.assertFalse(synchronized["hiddenBehavioralPrompt"])
        if generated["mimeType"] == "video/mp4":
            encoded = Path(generated["path"]).read_bytes()
            self.assertTrue(synchronized["generated"])
            self.assertIn(b"soun", encoded)
            self.assertEqual(generated["containerFallback"], "")
        else:
            self.assertFalse(synchronized["generated"])
            self.assertIn("APNG fallback", synchronized["reason"])
            self.assertTrue(generated["containerFallback"])
        brain.events.close()

    def test_video_audio_can_be_disabled_and_apng_fallback_is_honest(self):
        brain = self.make_brain()
        brain.modality_training["video"] = 1
        brain.modality_training["audio"] = 1
        with mock.patch.object(
            brain.modalities,
            "generate",
            wraps=brain.modalities.generate,
        ) as decode:
            silent = brain.generate_modality(
                "video", seed=32, settings={"includeAudio": False}
            )
        decoded_modalities = [call.args[0] for call in decode.call_args_list]
        self.assertEqual(decoded_modalities, ["video"])
        self.assertFalse(silent["synchronizedAudio"]["requested"])
        self.assertFalse(silent["synchronizedAudio"]["generated"])

        with mock.patch.object(
            brain,
            "_mp4_bytes",
            side_effect=RuntimeError("fixture mux unavailable"),
        ):
            fallback = brain.generate_modality("video", seed=33)
        self.assertEqual(fallback["mimeType"], "image/apng")
        self.assertTrue(fallback["synchronizedAudio"]["supported"])
        self.assertFalse(fallback["synchronizedAudio"]["generated"])
        self.assertIn("fixture mux unavailable", fallback["synchronizedAudio"]["reason"])
        brain.events.close()

    def test_vision_understanding_recalls_persistent_sensory_assemblies(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is required for the vision fixture")
        brain = self.make_brain(image_enabled=True, vision_enabled=True)
        image_path = self.root / "cobalt-harbor.png"
        Image.new("RGB", (19, 13), color=(15, 70, 180)).save(image_path)
        learned = brain.ingest(
            path=str(image_path), kind="image", policy="encode"
        )
        whole_id = learned["source"]["media_records"][0]["coverage"][
            "wholeRecordAssemblyId"
        ]

        understood = brain.generate_modality(
            "vision", input_path=str(image_path)
        )

        self.assertEqual(
            understood["associationMode"],
            "exact-ternary-recurrent-spreading",
        )
        self.assertGreater(understood["associationCount"], 0)
        self.assertIn(
            whole_id,
            {item["assemblyId"] for item in understood["associations"]},
        )
        self.assertTrue(
            any(
                "image-perception" in item["labels"]
                for item in understood["associations"]
            )
        )
        self.assertNotIn("source_text", json.dumps(understood))
        brain.events.close()

    def test_checkpoint_flushes_packed_state_without_creating_a_snapshot(self):
        brain = self.make_brain()
        brain.learn_experience(
            "Checkpoint this exact distributed neural assembly.",
            steps=0,
        )
        snapshot_root = brain.engine_path / "snapshots"
        before = (
            sorted(path.name for path in snapshot_root.iterdir())
            if snapshot_root.exists()
            else []
        )
        snapshot_count = brain.counters["snapshots"]
        result = brain.checkpoint("checkpoint-operation")
        after = (
            sorted(path.name for path in snapshot_root.iterdir())
            if snapshot_root.exists()
            else []
        )
        metadata_bytes = (brain.engine_path / "brain.json").read_bytes()
        packed_bytes = (
            brain.engine_path / "packed-ternary" / "manifest.json"
        ).read_bytes()
        self.assertEqual(result["format"], "omni-neural-checkpoint")
        self.assertEqual(result["operationId"], "checkpoint-operation")
        self.assertTrue(result["committed"])
        self.assertFalse(result["snapshotCreated"])
        self.assertEqual(before, after)
        self.assertEqual(brain.counters["snapshots"], snapshot_count)
        self.assertEqual(
            result["metadataSha256"], hashlib.sha256(metadata_bytes).hexdigest()
        )
        self.assertEqual(
            result["packedManifestSha256"],
            hashlib.sha256(packed_bytes).hexdigest(),
        )
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

    def test_train_rejects_missing_unknown_and_unretained_inputs_before_mutation(self):
        brain = self.make_brain()
        brain.training_sources.append(
            {
                "id": "unretained-source",
                "name": "hash-only fixture",
                "raw_text_retained": False,
            }
        )

        def snapshot():
            return {
                "parameters": brain.parameter_checksum(),
                "counters": copy.deepcopy(brain.counters),
                "replay": copy.deepcopy(brain.replay.status()),
                "training_sources": copy.deepcopy(brain.training_sources),
                "messages": copy.deepcopy(brain.messages),
                "traces": copy.deepcopy(brain.traces),
                "events": brain.events.recent(10_000),
                "files": {
                    str(path.relative_to(self.root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in self.root.rglob("*")
                    if path.is_file()
                },
            }

        cases = (
            ({"texts": []}, "requires non-empty text"),
            ({"texts": [" \x00 "]}, "requires non-empty text"),
            ({"source_ids": ["missing-source"]}, "unknown training source_ids"),
            (
                {"source_ids": ["unretained-source"]},
                "have no retained text",
            ),
        )
        with mock.patch.object(
            brain,
            "_train_evolution_replay_candidate",
            side_effect=AssertionError("training must not fall back to replay"),
        ):
            for arguments, message in cases:
                with self.subTest(arguments=arguments):
                    before = snapshot()
                    with self.assertRaisesRegex(ValueError, message):
                        brain.train(**arguments)
                    self.assertEqual(snapshot(), before)
        brain.events.close()

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
        # Auto preserves the logical target (2 × 2 = 4) while selecting the
        # largest safe physical divisor. On this fixture all four rows fit.
        self.assertEqual(result["physicalBatchSize"], 4)
        self.assertEqual(result["gradientAccumulation"], 1)
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
        parameter = AdaptiveBrain(
            "parameter", self.root / "parameter", parameter_config
        )
        working = AdaptiveBrain(
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
        brain = AdaptiveBrain(
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
        brain = AdaptiveBrain(
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


    def test_packed_metaplastic_resistance_persists_without_float_anchors(self):
        brain = self.make_brain(metaplasticity=True)
        projection = brain.decoder.action_policy.hidden
        self.assertTrue(projection.packed_stability_status()["metaplasticityEnabled"])
        with torch.no_grad():
            projection._row_stability[0] = 3
        self.assertEqual(float(brain._stability_penalty().item()), 0.0)
        self.assertEqual(brain.slow_importance, {})
        brain.save()
        brain.events.close()
        reloaded = AdaptiveBrain.load(self.root, "brain-test")
        self.assertEqual(
            int(reloaded.decoder.action_policy.hidden._row_stability[0]), 3
        )
        self.assertEqual(reloaded.slow_importance, {})
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
