"""Basic native MCP sampling mechanics, not trained capability qualification.

One small random-initialized native brain fixture; no initial curriculum,
foreign model, external provider, saved user brain or corpus training.
"""
import copy
import hashlib
import json
import queue
import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.chat_tool_observation import ChatToolObservationInbox
from omni_core.cooperative_generation import CooperativeNativeGeneration
from omni_core.mcp_sampling import (FORMAT, NativeSamplingCancelled, parse_approved_sampling,
                                   private_sampling_activity, sample_native_mcp)
from omni_core.offload import ResourcePolicy
from omni_core.parameter_diagnostics import (active_diagnostic_scopes, native_parameter_read_only,
                                             packed_diagnostic_write)
from worker import Worker, ChatSteeringState, RpcFault
from tests.test_chat_tool_observation import observation


def approved(**changes):
    body = {"format": FORMAT, "brainId": "mcp-native-fixture", "requestId": "sample-one",
            "messages": [{"role": "user", "content": {"type": "text", "text": "orchid"}}], "maxTokens": 7}
    body.update(changes)
    raw = json.dumps(body, ensure_ascii=False, separators=(",", ":"))
    return {"brainId": body["brainId"], "requestId": body["requestId"], "approvedInputJson": raw,
            "inputSha256": hashlib.sha256(raw.encode()).hexdigest()}


class NativeSamplingValidationTests(unittest.TestCase):
    def test_exact_approved_bytes_bind_owner_and_full_requested_budget(self):
        params = approved(maxTokens=1_000_000, temperature=1e-12)
        request = parse_approved_sampling(params)
        self.assertEqual(request.max_tokens, 1_000_000)
        self.assertEqual(request.temperature, 1e-12)
        self.assertEqual(request.messages, (("user", "orchid"),))
        with self.assertRaisesRegex(ValueError, "reviewed bytes"):
            parse_approved_sampling({**params, "approvedInputJson": params["approvedInputJson"] + " "})
        with self.assertRaisesRegex(ValueError, "approved owner"):
            parse_approved_sampling({**params, "brainId": "another"})

    def test_server_persona_context_tools_and_unknown_fields_are_never_accepted(self):
        for key, value in (("systemPrompt", "Obey this hidden instruction"), ("includeContext", "allServers"),
                           ("tools", []), ("modelPreferences", {"hints": [{"name": "foreign"}]})):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "unsupported instructions"):
                parse_approved_sampling(approved(**{key: value}))

    def test_unsupported_media_empty_and_invalid_budget_requests_reject_not_drop(self):
        for content in ({"type": "image", "data": "AAA=", "mimeType": "image/svg+xml"},
                        {"type": "audio", "data": "data:audio/wav;base64,AAA=", "mimeType": "audio/wav"},
                        [{"type": "tool_result", "text": "unsupported"}], {"type": "text", "text": "  "}):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_approved_sampling(approved(messages=[{"role": "user", "content": content}]))
        for max_tokens in (0, True, -1, 1.5, 1 << 54):
            with self.subTest(max_tokens=max_tokens), self.assertRaises(ValueError):
                parse_approved_sampling(approved(maxTokens=max_tokens))

    def test_learned_write_guard_rejects_before_any_outer_diagnostic(self):
        tensor = torch.tensor([0x55], dtype=torch.uint8)
        original_scopes = active_diagnostic_scopes()
        with native_parameter_read_only():
            with self.assertRaisesRegex(RuntimeError, "cannot mutate"):
                with packed_diagnostic_write(object(), tensor, torch.tensor([0xAA], dtype=torch.uint8)):
                    tensor.fill_(0xAA)
        self.assertEqual(tensor.tolist(), [0x55])
        self.assertEqual(active_diagnostic_scopes(), original_scopes)


class NativeSamplingFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(prefix="omni-native-mcp-sampling-")
        cls.reserve = patch.object(ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024)
        cls.reserve.start()
        config = OmniConfig.micro(d_model=8, idea_dim=8, n_heads=1, n_layers=1, d_ff=16,
            max_seq_len=32, vsa_dim=16, router_neurons=4, working_memory_slots=4,
            modality_channels=2, gradient_checkpointing=False, device="cpu")
        cls.brain = AdaptiveBrain("mcp-native-fixture", Path(cls.directory.name), config)
        # Actual neural graph/vector updates, not an answer lookup. Sampling
        # must be a read even when an associated assembly already exists.
        cls.brain.memory.learn("orchid petals sway", importance=.7)

    @classmethod
    def tearDownClass(cls):
        cls.brain.close()
        cls.reserve.stop()
        cls.directory.cleanup()

    def snapshot(self):
        brain = self.brain
        return {"checksum": brain.parameter_checksum(), "nodes": copy.deepcopy(dict(brain.memory.neurons)),
            "edges": copy.deepcopy(dict(brain.memory.synapses)),
            "vectors": {key: value.clone() for key, value in brain.memory.neuron_vectors.items()},
            "assemblies": copy.deepcopy(list(brain.memory.assemblies)), "revision": brain.memory.state_revision,
            "counters": copy.deepcopy(brain.counters), "pending": copy.deepcopy(brain.pending_chat_slow_learning),
            "messages": list(brain.messages), "liquid": brain.liquid_state,
            "membrane": brain.router.population.membrane, "spike_count": brain.router.population.spike_count,
            "modes": [(module, module.training) for root in (brain.decoder, brain.router, brain.liquid,
                brain.memory_bridge, brain.idea_adapter, brain.modalities) for module in root.modules()],
            "diagnostics": active_diagnostic_scopes(), "cpu_rng": torch.random.get_rng_state(),
            "pending_events": [(module, module._pending_stability_events) for root in
                (brain.decoder, brain.router, brain.liquid, brain.memory_bridge, brain.idea_adapter, brain.modalities)
                for module in root.modules() if hasattr(module, "_pending_stability_events")],
            "memory": brain.memory, "audit": brain.memory._last_recall_audit,
            "rounds": getattr(brain.memory, "_last_recall_rounds", None)}

    def assert_unchanged(self, before):
        brain = self.brain
        self.assertEqual(brain.parameter_checksum(), before["checksum"])
        for name, current in (("nodes", dict(brain.memory.neurons)), ("edges", dict(brain.memory.synapses)),
                              ("assemblies", list(brain.memory.assemblies)), ("counters", brain.counters),
                              ("pending", brain.pending_chat_slow_learning), ("messages", list(brain.messages))):
            self.assertEqual(current, before[name], name)
        self.assertEqual(brain.memory.state_revision, before["revision"])
        for key, value in before["vectors"].items():
            self.assertTrue(torch.equal(value, brain.memory.neuron_vectors[key]), key)
        self.assertIs(brain.memory, before["memory"])
        self.assertIs(brain.memory._last_recall_audit, before["audit"])
        self.assertEqual(getattr(brain.memory, "_last_recall_rounds", None), before["rounds"])
        self.assertIs(brain.liquid_state, before["liquid"])
        self.assertIs(brain.router.population.membrane, before["membrane"])
        self.assertIs(brain.router.population.spike_count, before["spike_count"])
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before["cpu_rng"]))
        self.assertEqual(active_diagnostic_scopes(), before["diagnostics"])
        for module, mode in before["modes"]:
            self.assertEqual(module.training, mode)
        for module, count in before["pending_events"]:
            self.assertEqual(module._pending_stability_events, count)

    def test_actual_native_sample_uses_same_owners_without_learning_history_or_callbacks(self):
        brain = self.brain
        brain.decoder.train(); brain.decoder.blocks[0].eval()
        brain.recent_token_context = [1, 259, *brain.tokenizer.encode("private old chat must not be injected"), 260]
        before = self.snapshot()
        captured = []
        original = brain.decoder._generation_step
        def capture(window, bias, cache):
            if not captured:
                captured.append(window.clone())
            return original(window, bias, cache)
        with patch.object(brain.decoder, "_generation_step", side_effect=capture), \
                patch.object(brain, "learn_experience", side_effect=AssertionError("private learning")), \
                patch.object(brain, "save", side_effect=AssertionError("private save")), \
                patch.object(brain.events, "append", side_effect=AssertionError("private event")):
            result = sample_native_mcp(brain, parse_approved_sampling(approved()))
        self.assertEqual(result["model"], "OmniCortex/" + brain.brain_id)
        self.assertTrue(result["privateWorkspace"])
        self.assertFalse(result["parametersMutated"])
        self.assertIn(result["stopReason"], {"endTurn", "maxTokens"})
        self.assertEqual(captured[0].tolist(), [[1, 259, *brain.tokenizer.encode("orchid"), 260]])
        self.assertEqual(brain.decoder.working_attention_status()["livePages"], 0)
        self.assert_unchanged(before)

    def test_stop_sequences_cross_complete_utf8_chunks_and_stop_before_more_compute(self):
        brain = self.brain
        expected = list("AéB🧠C".encode("utf-8"))
        calls, original = [], brain.decoder._generation_step
        def scripted(window, bias, cache):
            real, next_cache = original(window, bias, cache)
            token = expected[len(calls)] + 3
            calls.append(token)
            logits = torch.full_like(real, -float("inf")); logits[:, token] = 0.
            return logits, next_cache
        before = self.snapshot()
        with patch.object(brain.decoder, "_generation_step", side_effect=scripted):
            result = sample_native_mcp(brain, parse_approved_sampling(approved(maxTokens=20, stopSequences=["éB🧠"])))
        self.assertEqual(result["content"]["text"], "A")
        self.assertEqual(result["stopReason"], "stopSequence")
        self.assertEqual(len(calls), len("AéB🧠".encode("utf-8")))
        self.assert_unchanged(before)

    def test_private_sampling_does_not_change_interleaved_main_tokens_rng_cache_or_metadata(self):
        brain = self.brain
        prompt = torch.tensor([[1, 259, 65, 260]])
        options = dict(seed=212, max_new_tokens=5, temperature=.6, noise=.04, printable_only=False)
        expected, entropies = brain.decoder.generate(prompt, **options)
        main = CooperativeNativeGeneration(brain.decoder, prompt, **options)
        try:
            main.advance()
            cache = main.cache_owner[0]
            state = copy.copy(main.metadata)
            before = self.snapshot()
            sample_native_mcp(brain, parse_approved_sampling(approved(requestId="nested-sample")))
            self.assertIs(main.cache_owner[0], cache)
            self.assertEqual(main.metadata["last_generation_stop_reason"], state["last_generation_stop_reason"])
            self.assert_unchanged(before)
            while not main.done:
                main.advance()
            self.assertTrue(torch.equal(main.result[0], expected))
            self.assertEqual(main.result[1], entropies)
        finally:
            main.close()
        self.assertEqual(brain.decoder.working_attention_status()["livePages"], 0)

    def test_real_nested_generation_preserves_outer_pager_cancel_scope_and_live_cache(self):
        brain = self.brain
        prompt = torch.tensor([[1, 259, 65, 260]])
        options = dict(seed=927, max_new_tokens=4, temperature=.7, noise=.03, printable_only=False)
        expected, entropies = brain.decoder.generate(prompt, **options)
        private = [False]
        sampled, outer_caches = [], []
        original = brain.decoder._generation_step
        def step(window, bias, cache):
            value = original(window, bias, cache)
            if not private[0]:
                outer_caches.append((cache, value[1]))
            return value
        def cancelled():
            self.assertFalse(private[0], "private conditioning recursively called the main control callback")
            return False
        def on_token(*_args):
            if not sampled:
                private[0] = True
                try:
                    sampled.append(sample_native_mcp(brain, parse_approved_sampling(approved(requestId="nested-live-cache"))))
                finally:
                    private[0] = False
        before = self.snapshot()
        with patch.object(brain.decoder, "_generation_step", side_effect=step):
            generated, actual_entropy = brain.decoder.generate(prompt, **options, cancelled=cancelled, token_callback=on_token)
        self.assertEqual(len(sampled), 1)
        self.assertTrue(torch.equal(generated, expected))
        self.assertEqual(actual_entropy, entropies)
        for previous, current in zip(outer_caches, outer_caches[1:]):
            self.assertIs(current[0], previous[1], "private sampling invalidated the main turn's cache")
        self.assert_unchanged(before)
        self.assertEqual(brain.decoder.working_attention_status()["livePages"], 0)

    def test_cancellation_restores_activity_and_rejects_partial_completion(self):
        brain = self.brain
        calls, original = [], brain.decoder._generation_step
        def step(*args):
            result = original(*args)
            calls.append(1)
            return result
        before = self.snapshot()
        with patch.object(brain.decoder, "_generation_step", side_effect=step), self.assertRaises(NativeSamplingCancelled):
            sample_native_mcp(brain, parse_approved_sampling(approved(maxTokens=10)), cancelled=lambda: bool(calls))
        self.assert_unchanged(before)
        self.assertEqual(brain.decoder.working_attention_status()["livePages"], 0)

    def test_private_sampling_does_not_enter_the_live_turn_parameter_journal(self):
        brain = self.brain
        journal = brain._parameter_copy()
        try:
            before = self.snapshot()
            sample_native_mcp(brain, parse_approved_sampling(approved(requestId="outer-diagnostics")))
            self.assert_unchanged(before)
            self.assertIsNone(journal.database, "private inference wrote originals into the main mutation journal")
            self.assertEqual(journal.delta_norm(), 0.)
        finally:
            journal.close()

    def test_private_cleanup_restores_originals_even_when_recall_cleanup_raises(self):
        before = self.snapshot()
        with self.assertRaisesRegex(RuntimeError, "cleanup fixture"):
            with private_sampling_activity(self.brain):
                self.brain.memory.release_paged_recall_scratch = lambda: (_ for _ in ()).throw(RuntimeError("cleanup fixture"))
        self.assert_unchanged(before)

    def test_busy_writer_services_nested_sample_during_tool_wait_without_second_neural_thread(self):
        brain, worker = self.brain, Worker()
        worker.brains[brain.brain_id] = brain
        session = ChatSteeringState(brain.brain_id, "owned-turn", claimed=True)
        worker._chat_steering[(brain.brain_id, session.stream_id)] = session
        params = approved(originatingTurnId=session.stream_id, requestId="wait-sample")
        owner = {name: params[name] for name in ("brainId", "requestId", "inputSha256")}
        inbox = ChatToolObservationInbox("brain", "turn", lambda _size: True, wait_seconds=2,
            service_pending=lambda: worker.service_native_mcp_sampling(brain.brain_id, session.stream_id))
        inbox.register("a" * 32, {"kind": "tool", "toolId": "web.search", "action": "search"})
        session.observation_inbox = inbox
        responses, reader_errors, sample_threads = queue.Queue(), [], []
        original = brain.decoder._generation_step
        def native_step(*args):
            sample_threads.append(threading.get_ident())
            return original(*args)
        def server_and_reader():
            try:
                acknowledgement = worker.dispatch_control({"jsonrpc": "2.0", "id": "enqueue", "method": "mcp_sampling_enqueue", "params": params})
                self.assertEqual(acknowledgement["result"]["boundary"], "turn")
                # Real packed neural computation is deliberately used here,
                # rather than an immediate fabricated result. This is a test
                # watchdog, not a product generation/time limit.
                deadline = time.monotonic() + 15
                while True:
                    status = worker.dispatch_control({"jsonrpc": "2.0", "id": "status", "method": "mcp_sampling_status", "params": owner})["result"]
                    if status["state"] in {"complete", "failed", "cancelled"}:
                        responses.put(status)
                        break
                    if time.monotonic() > deadline:
                        raise AssertionError("nested sampling deadlocked behind its waiting writer")
                    time.sleep(.005)
                inbox.offer(observation("the MCP tool received its reviewed private sample"))
            except BaseException as error:
                reader_errors.append(error)
                inbox.close()
        reader = threading.Thread(target=server_and_reader)
        before = self.snapshot()
        try:
            with patch.object(brain.decoder, "_generation_step", side_effect=native_step):
                reader.start()
                evidence = inbox.wait_pending()
                reader.join(4)
            self.assertFalse(reader.is_alive())
            if reader_errors:
                raise reader_errors[0]
            self.assertEqual(responses.get_nowait()["state"], "complete")
            self.assertEqual(len(evidence), 1)
            self.assertEqual(set(sample_threads), {threading.get_ident()})
            self.assertEqual(inbox.drain(), [])
            self.assert_unchanged(before)
            consumed = worker.native_mcp_sampling_control("mcp_sampling_consume", owner)
            self.assertTrue(consumed["consumed"])
            self.assertEqual(worker._mcp_sampling_records, {})
        finally:
            inbox.close(); reader.join(4)
            worker.shutdown_native_mcp_sampling()
            worker.brains.clear()  # shared fixture remains owned by this class
            worker.shutdown({}, None)

    def test_reader_only_enqueues_exact_owned_work_and_unload_or_steer_cancels_before_compute(self):
        worker = Worker(); worker.brains[self.brain.brain_id] = self.brain
        session = ChatSteeringState(self.brain.brain_id, "owned-turn", claimed=True)
        worker._chat_steering[(self.brain.brain_id, session.stream_id)] = session
        try:
            with self.assertRaisesRegex(RpcFault, "trusted originating"):
                worker.dispatch_control({"jsonrpc": "2.0", "id": "unowned", "method": "mcp_sampling_enqueue", "params": approved()})
            for flag in (session.requested, session.suspension_requested):
                params = approved(requestId="cancel-" + str(id(flag)), originatingTurnId=session.stream_id)
                owner = {key: params[key] for key in ("brainId", "requestId", "inputSha256")}
                record = worker.native_mcp_sampling_control("mcp_sampling_enqueue", params)
                self.assertEqual(record["state"], "queued")
                flag.set()
                with patch("worker_mcp_sampling.sample_native_mcp", side_effect=AssertionError("cancelled native work")):
                    worker.service_native_mcp_sampling(self.brain.brain_id, session.stream_id)
                self.assertEqual(worker.native_mcp_sampling_control("mcp_sampling_status", owner)["state"], "cancelled")
                self.assertIsNone(worker._mcp_sampling_records[(self.brain.brain_id, params["requestId"])].request)
                worker.native_mcp_sampling_control("mcp_sampling_consume", owner)
                flag.clear()
        finally:
            worker.shutdown_native_mcp_sampling(); worker.brains.clear(); worker.shutdown({}, None)

    def test_running_sample_stop_steer_and_unload_restore_main_state_without_a_partial_result(self):
        worker = Worker(); worker.brains[self.brain.brain_id] = self.brain
        session = ChatSteeringState(self.brain.brain_id, "owned-turn", claimed=True)
        worker._chat_steering[(self.brain.brain_id, session.stream_id)] = session
        try:
            for flag in (worker._cooperative_cancel, session.requested, session.suspension_requested):
                params = approved(requestId="running-" + str(id(flag)), originatingTurnId=session.stream_id, maxTokens=20)
                owner = {key: params[key] for key in ("brainId", "requestId", "inputSha256")}
                worker.native_mcp_sampling_control("mcp_sampling_enqueue", params)
                original = self.brain.decoder._generation_step
                def interrupted(*args):
                    value = original(*args)
                    flag.set()
                    return value
                before = self.snapshot()
                with patch.object(self.brain.decoder, "_generation_step", side_effect=interrupted):
                    worker.service_native_mcp_sampling(self.brain.brain_id, session.stream_id)
                status = worker.native_mcp_sampling_control("mcp_sampling_status", owner)
                self.assertEqual(status["state"], "cancelled")
                self.assertNotIn("result", status)
                self.assertIsNone(worker._mcp_sampling_records[(self.brain.brain_id, params["requestId"])].request)
                self.assert_unchanged(before)
                worker.native_mcp_sampling_control("mcp_sampling_consume", owner)
                flag.clear()
        finally:
            worker.shutdown_native_mcp_sampling(); worker.brains.clear(); worker.shutdown({}, None)


if __name__ == "__main__":
    unittest.main()
