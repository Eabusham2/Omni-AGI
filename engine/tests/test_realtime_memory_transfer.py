"""Fast learned pathways must reach generation without an answer lookup.

These are mechanism tests, not evidence of fluent or correct verbal recall
from an untrained decoder. No optimizer work or manual consolidation is used.
"""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.offload import ResourcePolicy
from omni_core.vsa import NeuralSubstrate


EXPERIENCE = "A silver finch circled the west clock tower at dusk."
CUE = "What circled the west clock tower?"


class RealtimeSubstrateTransferTests(unittest.TestCase):
    def test_first_exposure_changes_live_pathways_and_readout_after_reload(self):
        memory = NeuralSubstrate(64, seed=7)
        self.assertEqual(len(memory.synapses), 0)
        learned = memory.learn(EXPERIENCE, retain_source_text=False)
        learned_id = learned["assembly_id"]
        self.assertGreater(len(memory.synapses), 0)
        self.assertTrue(
            any(record["effective_weight"] != 0 for record in memory.synapses.values())
        )
        self.assertTrue(
            all(record["effective_weight"] in {-1, 0, 1} for record in memory.synapses.values())
        )
        self.assertTrue(all("source_text" not in item for item in memory.assemblies))
        durable_synapses = copy.deepcopy(dict(memory.synapses))
        memory.clear_attention_activity(1)
        self.assertEqual(dict(memory.synapses), durable_synapses)
        self.assertTrue(
            all(memory.effective_activation(item) == 0.0 for item in memory.neurons.values())
        )

        with tempfile.TemporaryDirectory(prefix="omni-realtime-substrate-") as folder:
            store = Path(folder) / "substrate"
            memory.save_sharded(store)
            reloaded = NeuralSubstrate.load_sharded(
                store, memory.metadata(include_records=False), lazy_synapses=False
            )
            # The brain persists this epoch overlay separately from the
            # authoritative learned records. Restore the same fresh boundary.
            reloaded.clear_attention_activity(1)
            self.assertEqual(set(reloaded.synapses), set(durable_synapses))
            for key, synapse in reloaded.synapses.items():
                self.assertEqual(
                    synapse["effective_weight"], durable_synapses[key]["effective_weight"]
                )
                self.assertEqual(synapse["uses"], durable_synapses[key]["uses"])
                self.assertAlmostEqual(
                    synapse["eligibility"], durable_synapses[key]["eligibility"], places=6
                )
                self.assertNotIn("latent_weight", synapse)
            cue = reloaded.vector_for_text(CUE)
            signal, recalled = reloaded.recall_vector(cue)
            score = next(item["score"] for item in recalled if item["assembly_id"] == learned_id)
            self.assertGreater(score, 0.0)
            self.assertLess(score, 1.0)
            self.assertGreater(reloaded._last_recall_audit["settledRounds"], 1)
            self.assertGreater(reloaded._last_recall_audit["eligibleEdges"], 0)
            self.assertIn(learned_id, reloaded.attention_recalled_assembly_ids)
            # A positive sub-unit memory contribution must survive readout.
            # The old sign bundle returned precisely the cue in this case.
            self.assertFalse(torch.equal(signal, cue))

            for synapse in reloaded.synapses.values():
                synapse["effective_weight"] = 0
            ablated, _ = reloaded.recall_vector(cue, record_activity=False)
            self.assertGreater(float((signal - ablated).norm()), 1e-6)


class RealtimeChatTransferTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(711)
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="omni-realtime-chat-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "brain"
        reserve = mock.patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)

    def test_ordinary_chat_survives_fresh_reload_and_conditions_decoder(self):
        brain = AdaptiveBrain.create(
            "realtime-transfer",
            self.path,
            OmniConfig.micro(
                seed=7,
                max_seq_len=64,
                working_memory_slots=3,
                online_learning=False,
                online_steps=0,
                learn_from_own_messages=False,
                growth_novelty_threshold=1.0,
                vision_enabled=False,
                image_enabled=False,
                audio_enabled=False,
                video_enabled=False,
            ),
            initialize_ground_up=True,
        )
        self.addCleanup(brain.close)
        fast_before = brain._fast_synapse_checksum()
        with mock.patch.object(
            brain, "_optimize_experience",
            side_effect=AssertionError("fast memory must not require optimizer work"),
        ):
            result = brain.chat(EXPERIENCE, max_new_tokens=1, seed=713)
        self.assertTrue(result["turnCommitted"])
        self.assertNotEqual(brain._fast_synapse_checksum(), fast_before)
        learned_id = next(
            item["id"] for item in brain.memory.assemblies
            if item["source"] == "conversation"
        )
        self.assertGreater(len(brain.memory.synapses), 0)
        self.assertEqual(brain.pending_chat_slow_learning, [])
        fast_learned = brain._fast_synapse_checksum()
        fresh = brain.start_fresh_attention("realtime-transfer-fresh")
        self.assertTrue(fresh["committed"])
        self.assertEqual(brain._fast_synapse_checksum(), fast_learned)

        reloaded = AdaptiveBrain.load(self.path, "realtime-transfer")
        self.addCleanup(reloaded.close)
        self.assertEqual(reloaded._fast_synapse_checksum(), fast_learned)
        self.assertEqual(reloaded.messages, [])
        self.assertEqual(reloaded.recent_token_context, [])
        self.assertEqual(reloaded.working_memory, [])
        self.assertEqual(reloaded.memory_lifecycle.afterimage_items, [])
        self.assertIn(learned_id, reloaded.memory.assembly_vectors)

        cue = reloaded.memory.vector_for_text(CUE)
        recalled_signal, recalled = reloaded.memory.recall_vector(cue, record_activity=False)
        self.assertIn(learned_id, {item["assembly_id"] for item in recalled})
        live_synapses = reloaded.memory.synapses
        # Isolate the actual forward-edge contribution without changing the
        # learned checkpoint or turning this into an alternate answer path.
        reloaded.memory.synapses = {
            key: {**value, "effective_weight": 0}
            for key, value in live_synapses.items()
        }
        try:
            ablated_signal, _ = reloaded.memory.recall_vector(cue, record_activity=False)
        finally:
            reloaded.memory.synapses = live_synapses
        self.assertGreater(float((recalled_signal - ablated_signal).norm()), 1e-6)
        # Propagate this isolated edge ablation through the same bridge,
        # adapter and decoder used by chat. Merely reporting a recalled id
        # or nonzero memory bias would not establish an effect on prediction.
        reloaded.decoder.eval()
        probe_ids = torch.tensor(
            [reloaded.tokenizer.dialogue(CUE, complete=False)],
            dtype=torch.long,
            device=reloaded.device,
        )
        with torch.no_grad():
            live_bias = reloaded.idea_adapter(reloaded._idea_model_vector(recalled_signal))
            ablated_bias = reloaded.idea_adapter(reloaded._idea_model_vector(ablated_signal))
            live_logits = reloaded.decoder(probe_ids, memory_bias=live_bias)["logits"]
            ablated_logits = reloaded.decoder(probe_ids, memory_bias=ablated_bias)["logits"]
        self.assertGreater(float((live_bias - ablated_bias).norm()), 1e-6)
        self.assertGreater(float((live_logits - ablated_logits).norm()), 1e-6)

        calls = []
        generate = reloaded.decoder.generate

        def capture_generation(ids, *args, **kwargs):
            calls.append((ids.detach().clone(), kwargs["memory_bias"].detach().clone()))
            return generate(ids, *args, **kwargs)

        with mock.patch.object(
            reloaded,
            "_optimize_experience",
            side_effect=AssertionError("queued training is not memory recall"),
        ), mock.patch.object(
            reloaded.decoder, "generate", side_effect=capture_generation
        ):
            response = reloaded.chat(CUE, max_new_tokens=1, seed=719)

        trace = response["trace"]
        self.assertIn(learned_id, trace["recalled_idea_ids"])
        self.assertGreater(trace["spreading_activation"]["settledRounds"], 1)
        self.assertEqual(trace["recent_dialogue_token_count"], 0)
        self.assertFalse(trace["textual_memory_injected"])
        self.assertFalse(trace["long_term_source_text_injected"])
        self.assertTrue(calls)
        ids, memory_bias = calls[0]
        self.assertNotIn("silver finch", reloaded.tokenizer.decode(ids[0].tolist()).lower())
        self.assertTrue(bool(torch.isfinite(memory_bias).all()))
        with torch.no_grad():
            enabled = reloaded.decoder(ids, memory_bias=memory_bias)["logits"]
            disabled = reloaded.decoder(ids, memory_bias=torch.zeros_like(memory_bias))["logits"]
        self.assertGreater(float((enabled - disabled).norm()), 1e-6)
        # Deliberately do not assert a correct answer from random dense cortex.


if __name__ == "__main__":
    unittest.main()
