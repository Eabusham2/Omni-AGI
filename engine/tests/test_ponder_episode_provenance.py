"""New bounded operational Ponder contracts: stdlib/AST/inert callbacks only.

No torch import, model/brain construction, neural forward, training, Load,
provider, UI, or learned-capability claim.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import random
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location("inert_" + name, ROOT / "omni_core" / (name + ".py"))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


EPISODE = load("ponder_episode")
TEXT = load("chat_steering")
protocol = ast.parse((ROOT / "omni_core/native_action_protocol.py").read_text())
namespace = dict(hashlib=hashlib, json=json, math=math, Any=Any, Dict=Dict, Mapping=Mapping,
                 Optional=Optional, NativePonderEpisode=EPISODE.NativePonderEpisode)
nodes = [node for node in protocol.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
         and node.name in {"action_fingerprint", "NativeActionEmissionLedger"}]
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / "omni_core/native_action_protocol.py"), "exec"), namespace)
Ledger = namespace["NativeActionEmissionLedger"]
BRAIN = ast.parse((ROOT / "omni_core/brain.py").read_text())
METHODS = {node.name: node for owner in BRAIN.body if isinstance(owner, ast.ClassDef)
           for node in owner.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def trace(reason="learned-action-transition"):
    return {"activated": True, "activated_by": "learned-action-head", "completed": reason in {"converged", "learned-action-transition"},
            "passes": 2, "elapsed_ms": 4., "final_delta": .1, "stop_reason": reason,
            "phase": "mid-generation", "private": True, "visibleMagicTags": False}


class PonderProvenanceCases(unittest.TestCase):
    def parent(self, ledger, *, step=7):
        return ledger.register({"kind": "ponder", "arguments": {"organic": True}}, step=step, phase="mid-generation")

    def child(self):
        return {"kind": "tool", "toolId": "web.search", "action": "search", "arguments": {"query": "inert"}}

    def test_start_child_finish_order_comes_from_one_active_invocation_and_deduplicates_effects(self):
        ledger, frames = Ledger("turn"), []
        parent = self.parent(ledger)
        with ledger.ponder_episode(parent, lambda kind, value: frames.append((kind, copy.deepcopy(value)))) as episode:
            self.assertEqual(frames[0][1]["state"], "started")
            child = episode.select(self.child(), step=7, phase="mid-generation")
            self.assertEqual(child["parentPonderActionId"], parent["actionId"])
            self.assertIsNone(episode.select(self.child(), step=7, phase="mid-generation"))
            with self.assertRaisesRegex(ValueError, "suspension"): ledger.suspension_state()
            episode.finish(trace())
        self.assertEqual([(kind, value.get("state")) for kind, value in frames],
                         [("ponder-episode", "started"), ("action", None), ("ponder-episode", "finished")])
        self.assertIsNone(ledger.active_ponder_action_id)
        for kind, value in frames:
            if kind == "ponder-episode": EPISODE.validate_ponder_episode_frame(value)
        with self.assertRaisesRegex(ValueError, "active Ponder"):
            ledger.register(self.child(), step=8, phase="mid-generation", parent_ponder_action_id=parent["actionId"])

    def test_private_candidate_keeps_its_local_proposal_without_an_executable_emitter(self):
        ledger = Ledger("private"); parent = self.parent(ledger)
        with ledger.ponder_episode(parent) as episode:
            proposal = episode.select(self.child(), step=7, phase="mid-generation")
            self.assertEqual(proposal["parentPonderActionId"], parent["actionId"])
            self.assertIsNone(episode.select(self.child(), step=7, phase="mid-generation"))
            episode.finish(trace())
        self.assertEqual(len(ledger.actions), 2)

    def test_model_arguments_or_metadata_cannot_create_an_ancestry(self):
        ledger = Ledger("turn")
        action = self.child(); action["parentPonderActionId"] = "a" * 32
        action["arguments"]["parentPonderActionId"] = "user payload, not provenance"
        registered = ledger.register(action, step=1, phase="mid-generation")
        self.assertNotIn("parentPonderActionId", registered)
        self.assertEqual(registered["arguments"], action["arguments"])

    def test_new_episodes_have_unique_identity_but_repeated_external_effect_does_not(self):
        ledger, frames = Ledger("turn"), []
        parents = []
        for step in (7, 8):
            parent = self.parent(ledger, step=step); parents.append(parent["actionId"])
            with ledger.ponder_episode(parent, lambda kind, value: frames.append(kind)) as episode:
                child = episode.select(self.child(), step=step, phase="mid-generation")
                self.assertEqual(child is not None, step == 7)
                episode.finish(trace())
        self.assertNotEqual(*parents); self.assertEqual(frames.count("action"), 1)

    def test_failure_finishes_without_success_and_restores_ownership(self):
        ledger, frames = Ledger("turn"), []
        with self.assertRaisesRegex(RuntimeError, "setup"):
            with ledger.ponder_episode(self.parent(ledger), lambda kind, value: frames.append(copy.deepcopy(value))):
                raise RuntimeError("setup")
        self.assertEqual(frames[-1]["state"], "finished")
        self.assertEqual(frames[-1]["action"]["arguments"]["ponderTrace"]["stop_reason"], "failed")
        self.assertFalse(frames[-1]["action"]["arguments"]["completedInTurn"])
        self.assertIsNone(frames[-1]["action"]["arguments"]["ponderTrace"]["passes"])
        EPISODE.validate_ponder_episode_frame(frames[-1])
        self.assertIsNone(ledger.active_ponder_action_id)

    def test_completed_episode_continuation_preserves_linkage_and_rejects_unknown_parent(self):
        ledger = Ledger("turn"); parent = self.parent(ledger)
        with ledger.ponder_episode(parent, lambda *_: None) as episode:
            child = episode.select(self.child(), step=7, phase="mid-generation"); episode.finish(trace("suspended"))
        resumed = Ledger.from_suspension_state(copy.deepcopy(ledger.suspension_state()), "turn")
        self.assertEqual([a for a in resumed.actions.values() if a["kind"] == "tool"][0]["parentPonderActionId"], parent["actionId"])
        self.assertIsNone(resumed.register(self.child(), step=8, phase="mid-generation"))
        broken = copy.deepcopy(ledger.suspension_state())
        next(a for a in broken["actions"].values() if a["kind"] == "tool")["parentPonderActionId"] = "f" * 32
        with self.assertRaisesRegex(ValueError, "Ponder parent"): Ledger.from_suspension_state(broken, "turn")

    def test_actual_selected_wrapper_materializes_existing_selector_inside_active_invocation(self):
        method = copy.deepcopy(METHODS["_selected_native_ponder"]); method.decorator_list = []
        space = {"ChatGenerationCancelled": type("ChatGenerationCancelled", (Exception,), {})}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "selected-wrapper", "exec"), space)
        frames, selections = [], []
        ledger = Ledger("turn"); parent = self.parent(ledger)
        def native(_current, _internal, **options):
            self.assertEqual(ledger.active_ponder_action_id, parent["actionId"])
            options["transition_callback"]("actual-head-logits", "actual-refined-state")
            options["completion_callback"](trace())
            return "actual-refined-state", trace()
        def selector(logits, **options):
            selections.append((logits, options)); self.assertEqual(ledger.active_ponder_action_id, parent["actionId"])
            return {}, [self.child()]
        host = types.SimpleNamespace(resource_policy=types.SimpleNamespace(status=lambda: {}),
            _native_pre_speech_ponder=native, _select_structured_actions=selector)
        result = space["_selected_native_ponder"](host, "current", "internal", registered=parent,
            ledger=ledger, emit=lambda kind, value: frames.append((kind, copy.deepcopy(value))), schemas=["typed schema"],
            input_text="actual user input", assembly_ids=["measured assembly"], organic_state={}, step=7, phase="mid-generation")
        self.assertEqual(result[0], "actual-refined-state"); self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0][0], "actual-head-logits")
        self.assertEqual(selections[0][1]["neural_state"], "actual-refined-state")
        self.assertEqual([kind for kind, _ in frames], ["ponder-episode", "action", "ponder-episode"])

    def test_actual_transition_call_is_in_neural_try_before_finally_completion(self):
        method = METHODS["_native_pre_speech_ponder"]
        block = next(node for node in method.body if isinstance(node, ast.Try))
        calls = lambda nodes, name: [node for item in nodes for node in ast.walk(item)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name]
        self.assertEqual(len(calls(block.body, "transition_callback")), 1)
        self.assertEqual(len(calls(block.finalbody, "transition_callback")), 0)
        self.assertEqual(len(calls(block.finalbody, "completion_callback")), 1)
        self.assertIn("continuation_logits", ast.unparse(calls(block.body, "transition_callback")[0]))
        self.assertIn('return \'suspended\'', ast.unparse(method))
        source = ast.unparse(METHODS["chat"])
        self.assertIn("branch_ledger = canonical_action_ledger", source)
        self.assertIn("copy.deepcopy(canonical_action_ledger.suspension_state())", source)
        self.assertNotIn("emit=stream_callback", ast.unparse(next(node for node in ast.walk(METHODS["chat"])
            if isinstance(node, ast.FunctionDef) and node.name == "start_branch")))

    def test_actual_wrapper_preserves_stop_steer_suspension_and_resource_boundaries(self):
        for blocked in ("cancelled", "steered", "suspended", "resource-pressure"):
            with self.subTest(boundary=blocked):
                method = copy.deepcopy(METHODS["_selected_native_ponder"]); method.decorator_list = []
                cancelled = type("ChatGenerationCancelled", (Exception,), {})
                space = {"ChatGenerationCancelled": cancelled}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "selected-wrapper", "exec"), space)
                ledger, frames = Ledger("turn"), []; parent = self.parent(ledger)
                def native(_current, _internal, **options):
                    try: options["transition_callback"]("real-head-argument", "refined")
                    finally: options["completion_callback"](trace(blocked))
                    return "refined", trace()
                def forbidden_selector(*_args, **_options): self.fail("blocked/private candidate materialized an action")
                host = types.SimpleNamespace(resource_policy=types.SimpleNamespace(status=lambda: {"memoryPressure": blocked == "resource-pressure"}),
                    _native_pre_speech_ponder=native, _select_structured_actions=forbidden_selector)
                options = dict(registered=parent, ledger=ledger, emit=lambda *values: frames.append(values),
                    schemas=[], input_text="unmodified user input", assembly_ids=[], organic_state={}, step=7, phase="mid-generation",
                    cancel_check=lambda: blocked == "cancelled", steer_check=lambda: blocked == "steered", suspension_check=lambda: blocked == "suspended")
                if blocked == "cancelled":
                    with self.assertRaises(cancelled): space["_selected_native_ponder"](host, "current", "memory", **options)
                else: space["_selected_native_ponder"](host, "current", "memory", **options)
                self.assertIsNone(ledger.active_ponder_action_id)
                self.assertEqual(len(ledger.actions), 1)
                self.assertFalse(any(kind == "action" for kind, _ in frames))

    def test_extracted_private_and_visible_transition_compute_identical_stateful_selector_and_local_stop(self):
        outcomes = []
        for visible in (False, True):
            method = copy.deepcopy(METHODS["_selected_native_ponder"]); method.decorator_list = []
            space = {"ChatGenerationCancelled": type("ChatGenerationCancelled", (Exception,), {})}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "actual-ponder-parity", "exec"), space)
            ledger, frames, reads = Ledger("same-turn"), [], []
            parent = self.parent(ledger); generator = random.Random(123)
            diagnostic = {"before": "same"}
            def selector(logits, **options):
                # Inert stand-in for documented diagnostic/cache/recall reads.
                # It intentionally consumes local RNG too, so an emitter-gated
                # skipped read cannot accidentally pass the parity assertion.
                sample = generator.random(); reads.append((logits, options["neural_state"], sample))
                diagnostic.clear(); diagnostic["observed"] = sample
                return {}, [{"kind": "stop", "arguments": {"reason": "inert learned selection"}}]
            def native(_current, _internal, **options):
                options["transition_callback"]("same-head-logits", "same-refined-state")
                options["completion_callback"](trace())
                return "same-refined-state", trace()
            host = types.SimpleNamespace(resource_policy=types.SimpleNamespace(status=lambda: {}),
                _native_pre_speech_ponder=native, _select_structured_actions=selector)
            space["_selected_native_ponder"](host, "current", "memory", registered=parent, ledger=ledger,
                emit=(lambda kind, value: frames.append((kind, copy.deepcopy(value)))) if visible else None,
                schemas=["same typed schema"], input_text="same user input", assembly_ids=["same measured assembly"],
                organic_state={}, step=7, phase="mid-generation")
            outcomes.append((reads, diagnostic, generator.getstate(), ledger.suspension_state(), ledger.stop_selected))
            self.assertEqual(bool(frames), visible)
            self.assertEqual([kind for kind, _ in frames].count("action"), int(visible))
        self.assertEqual(outcomes[0], outcomes[1]); self.assertTrue(outcomes[0][-1])
        # Actual selector source has writes/observation-producing calls; do not
        # claim its materialization is a universally pure read.
        selected_source = ast.unparse(METHODS["_select_structured_actions"])
        self.assertIn("self._last_tool_route_evidence =", selected_source)
        self.assertIn("self._assembly_argument_view(assembly_ids)", selected_source)
        self.assertIn("self._modality_idea('', assembly_ids)", selected_source)

    def test_inherited_mapping_rebinds_explicit_parent_only_outside_tool_arguments(self):
        source = ast.parse((ROOT / "omni_core/connected_execution.py").read_text())
        method = next(copy.deepcopy(node) for owner in source.body if isinstance(owner, ast.ClassDef)
            for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "_outward")
        space = {"copy": copy}; exec(compile(ast.Module(body=[method], type_ignores=[]), "actual-outward", "exec"), space)
        host = types.SimpleNamespace(root=types.SimpleNamespace(brain_id="new-brain"), original_turn_id="old-turn", root_turn_id="new-turn",
            prior_action_ids=set(), action_id=lambda value: hashlib.sha256(("new-brain:" + value).encode()).hexdigest()[:32],
            _outward_arguments=copy.deepcopy)
        host._outward = types.MethodType(space["_outward"], host)
        original = {"actionId": "a" * 32, "parentPonderActionId": "b" * 32,
                    "arguments": {"parentPonderActionId": "payload stays literal", "turnId": "old-turn"}, "turnId": "old-turn"}
        mapped = host._outward(original)
        self.assertEqual(mapped["actionId"], host.action_id(original["actionId"]))
        self.assertEqual(mapped["parentPonderActionId"], host.action_id(original["parentPonderActionId"]))
        self.assertEqual(mapped["turnId"], "new-turn"); self.assertEqual(mapped["arguments"], original["arguments"])

    def test_episode_ownership_cannot_overlap_or_finish_twice(self):
        ledger = Ledger("turn"); parent, other = self.parent(ledger), self.parent(ledger, step=8)
        with ledger.ponder_episode(parent, lambda *_: None) as episode:
            with self.assertRaisesRegex(ValueError, "inactive parent"):
                with ledger.ponder_episode(other): pass
            episode.finish(trace())
            with self.assertRaisesRegex(ValueError, "active invocation"): episode.finish(trace())
        with self.assertRaisesRegex(ValueError, "inactive parent"):
            with ledger.ponder_episode(parent): pass

    def test_a_genuinely_selected_stop_keeps_native_stop_authority_not_a_new_prompt(self):
        ledger = Ledger("turn"); parent = self.parent(ledger)
        self.assertFalse(ledger.stop_selected)
        with ledger.ponder_episode(parent, lambda *_: None) as episode:
            episode.select({"kind": "stop", "arguments": {}}, step=7, phase="mid-generation")
            episode.finish(trace())
        self.assertTrue(ledger.stop_selected)
        resumed = Ledger.from_suspension_state(copy.deepcopy(ledger.suspension_state()), "turn")
        self.assertTrue(resumed.stop_selected)
        callback = ast.unparse(METHODS["_generation_activity_callback"])
        self.assertIn("if ledger.stop_selected:", callback)
        self.assertIn("'stop': stop or ledger.stop_selected", callback)

    def test_whitespace_and_non_bmp_binding_is_measured_at_actual_normalization(self):
        raw = " \t🧠 brain\n "
        completion = TEXT.generation_completion([1], lambda _: raw)
        binding = completion["publicTextBinding"]
        self.assertEqual(completion["text"], "🧠 brain")
        self.assertEqual(binding["removedPrefixUtf16"], 2); self.assertEqual(binding["removedSuffixUtf16"], 2)
        self.assertEqual(binding["rawUtf16Length"], 12); self.assertEqual(binding["canonicalUtf16Length"], 8)
        self.assertEqual(binding["rawSha256"], hashlib.sha256(raw.encode()).hexdigest())
        interrupted = TEXT.generation_completion([1], lambda _: raw, stopped_by_steer=True)
        self.assertEqual(interrupted["text"], raw)
        self.assertEqual(interrupted["publicTextBinding"]["removedPrefixUtf16"], 0)
        self.assertEqual(interrupted["publicTextBinding"]["normalization"], "identity")


if __name__ == "__main__": unittest.main()
