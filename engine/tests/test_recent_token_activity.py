"""Pure activity/source fixtures: no brain construction, training or quiz."""

import copy
import inspect
import json
import unittest
from types import MethodType, SimpleNamespace

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.memory_lifecycle import OrganicMemoryLifecycle
from omni_core.recent_token_activity import RecentTokenActivity, _raw_activity, token_hash
from omni_core.tokenizer import ByteTokenizer


def _episode(identifier, *, cycle=0, strength=0.2, unfinished=0.0, **signals):
    return {
        "id": identifier,
        "assemblyId": "shared-field",
        "lastActiveCycle": cycle,
        "strength": strength,
        "activityScore": strength,
        "salience": strength,
        "unfinishedScore": unfinished,
        "signals": dict(signals),
    }


class RecentTokenActivityTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()

    def turn(self, human, brain):
        return self.tokenizer.dialogue(human, brain)[1:]

    def activity(self, entries):
        state, tokens = RecentTokenActivity(), []
        for identifier, human, brain in entries:
            tokens = state.append(tokens, self.turn(human, brain), identifier)
        return state, tokens

    def fixture(self):
        fixture = SimpleNamespace(
            tokenizer=self.tokenizer,
            config=SimpleNamespace(max_seq_len=4_096, forgetting_rate=0.02,
                                   long_term_threshold=0.5),
            recent_token_context=[],
            recent_token_activity=RecentTokenActivity(),
            memory_lifecycle=SimpleNamespace(cycle=0, afterimage_items=[]),
            current_context={"tokenCount": 3, "tokenHash": "current-prompt"},
            counters={"context_token_evictions": 0,
                      "context_token_cooling_evictions": 0},
            messages=[{"role": "human", "content": "visible ledger stays"}],
            memory=SimpleNamespace(learned_parameters=b"packed-authority"),
            router=SimpleNamespace(),
        )
        fixture._token_sequence_hash = AdaptiveBrain._token_sequence_hash
        for name in (
            "_bounded_completed_turn_tokens", "_append_recent_dialogue",
            "_recent_token_activity", "_recent_dialogue_snapshot",
            "_restore_recent_dialogue_snapshot", "_sync_recent_dialogue_counts",
            "_clear_recent_dialogue", "_cool_recent_dialogue",
            "_prompt_with_recent_context", "_settle_memory_automatically",
        ):
            setattr(fixture, name, MethodType(getattr(AdaptiveBrain, name), fixture))
        return fixture

    def test_weak_older_raw_turn_cools_with_spare_capacity(self):
        state, tokens = self.activity([
            ("weak", "passing fleck", "noticed"),
            ("current", "current task", "working"),
        ])
        current = self.turn("current task", "working")
        self.assertLess(len(tokens), 4_096)
        retained = state.cool(tokens, [_episode("weak", strength=0.01)], cycle=5)
        self.assertEqual(retained, current)
        self.assertEqual(state.summary()["trackedTokens"], len(current))
        self.assertEqual([span["afterimageId"] for span in state.spans], ["current"])

    def test_current_unfinished_and_exact_reused_spans_stay_active(self):
        state, tokens = self.activity([
            ("weak", "old", "done"),
            ("unfinished", "unfinished", "working"),
            ("reused", "reused", "active again"),
            ("current", "now", "current"),
        ])
        items = [
            _episode("weak", strength=0.001),
            _episode("unfinished", strength=0.001, unfinished=0.7),
            _episode("reused", cycle=200, strength=0.001),
        ]
        retained = state.cool(tokens, items, cycle=200)
        self.assertEqual(retained,
                         self.turn("unfinished", "working")
                         + self.turn("reused", "active again")
                         + self.turn("now", "current"))
        self.assertEqual([span["afterimageId"] for span in state.spans],
                         ["unfinished", "reused", "current"])
        # All records share an assembly; its identity must not pin the weak
        # sibling's raw words merely because a different episode was reused.
        self.assertNotIn("weak", [span["afterimageId"] for span in state.spans])

    def test_live_signals_change_cooling_instead_of_a_fixed_turn_count(self):
        base = _episode("old", strength=0.2, salience=0.0)
        # salience lives on the item, not in the measured edge signals.
        base["salience"] = 0.2
        score = _raw_activity(base, 15)
        for key in ("reuse", "recurrence", "stability"):
            changed = copy.deepcopy(base)
            changed["signals"][key] = 1.0
            self.assertGreater(_raw_activity(changed, 15), score, msg=key)
        salient = copy.deepcopy(base)
        salient["salience"] = 1.0
        self.assertGreater(_raw_activity(salient, 15), score)
        interfering = copy.deepcopy(base)
        interfering["signals"]["interference"] = 1.0
        self.assertLess(_raw_activity(interfering, 15), score)
        self.assertLess(_raw_activity(base, 30), score)

        weak, weak_tokens = self.activity([("old", "old", "done"), ("now", "now", "here")])
        stable, stable_tokens = self.activity([("old", "old", "done"), ("now", "now", "here")])
        strong = copy.deepcopy(base)
        strong["signals"] = {"reuse": 1.0, "recurrence": 1.0, "stability": 1.0}
        self.assertLess(len(weak.cool(weak_tokens, [base], 15)), len(weak_tokens))
        self.assertEqual(stable.cool(stable_tokens, [strong], 15), stable_tokens)

    def test_current_turn_has_no_age_or_turn_count_expiry(self):
        state, tokens = self.activity([("current", "active task", "in progress")])
        self.assertIs(state.cool(tokens, [], 10_000_000), tokens)

    def test_retired_raw_words_are_never_restored_by_neural_reactivation(self):
        state, tokens = self.activity([("old", "retire these words", "done"),
                                       ("current", "keep this", "now")])
        retained = state.cool(tokens, [_episode("old", strength=0.001)], 100)
        restored_activity = [_episode("old", cycle=101, strength=1.0,
                                      reuse=1.0, recurrence=1.0, stability=1.0)]
        self.assertEqual(state.cool(retained, restored_activity, 101), retained)
        serialized = json.dumps(state.metadata(retained))
        self.assertNotIn("retire these words", serialized)
        self.assertNotIn('"old"', serialized)
        self.assertFalse(state.metadata(retained)["rawTokenIdsStored"])

    def test_saved_activity_is_hash_bound_and_restart_preserves_exact_spans(self):
        state, tokens = self.activity([("a", "first", "answer"), ("b", "next", "active")])
        metadata = json.loads(json.dumps(state.metadata(tokens)))
        restored = RecentTokenActivity.from_state(metadata, list(tokens))
        self.assertEqual(restored.metadata(tokens), state.metadata(tokens))
        items = [_episode("a", strength=0.001)]
        self.assertEqual(restored.cool(list(tokens), items, 50), state.cool(tokens, items, 50))
        corrupted = list(tokens)
        corrupted[1] += 1
        with self.assertRaisesRegex(ValueError, "token binding changed"):
            RecentTokenActivity.from_state(metadata, corrupted)
        metadata["spans"][0]["tokenCount"] += 1
        with self.assertRaisesRegex(ValueError, "does not align"):
            RecentTokenActivity.from_state(metadata, tokens)

    def test_unlinked_legacy_spans_are_truthfully_reported_not_guessed(self):
        tokens = self.turn("legacy one", "first") + self.turn("legacy two", "second")
        state = RecentTokenActivity.from_state(None, tokens,
                                              human_id=self.tokenizer.human_id)
        self.assertEqual(state.summary()["untrackedTokens"], len(tokens))
        self.assertEqual(state.summary()["trackedTokens"], 0)
        self.assertEqual(state.summary()["spanCount"], 2)
        self.assertEqual(state.cool(tokens, [_episode("wrong", cycle=500)], 500), tokens)
        self.assertFalse(state.summary()["legacyBindingGuessed"])

    def test_capacity_pressure_preserves_complete_current_role_boundaries(self):
        state, tokens = self.activity([("old", "old words", "done"),
                                       ("current", "human payload", "brain payload")])
        retained = state.fit_capacity(tokens, 11, human_id=self.tokenizer.human_id,
                                      brain_id=self.tokenizer.brain_id,
                                      eos_id=self.tokenizer.eos_id)
        self.assertEqual(len(retained), 11)
        self.assertEqual(retained[0], self.tokenizer.human_id)
        self.assertIn(self.tokenizer.brain_id, retained)
        self.assertEqual(retained[-1], self.tokenizer.eos_id)
        self.assertEqual(state.spans, [{"tokenCount": 11, "afterimageId": "current"}])
        self.assertEqual(state.metadata(retained)["tokenHash"], token_hash(retained))

    def test_brain_settle_seam_cools_tokens_but_keeps_ledger_and_packed_authority(self):
        fixture = self.fixture()
        fixture.memory_lifecycle.afterimage_items = [_episode("old", cycle=0, strength=0.01)]
        fixture._append_recent_dialogue("passing", "fleck", afterimage_id="old")
        fixture._append_recent_dialogue("current", "work", afterimage_id="current")
        before = copy.deepcopy(fixture.messages)
        packed_before = fixture.memory.learned_parameters

        def settle(**kwargs):
            fixture.memory_lifecycle.cycle = 50
            fixture.memory_lifecycle.afterimage_items = [_episode("old", strength=0.001)]
            return {"reinforcementDrive": 0.0, "recurringAssemblyId": ""}

        fixture.memory_lifecycle.settle = settle
        report = fixture._settle_memory_automatically(
            None, assembly_id="unrelated", source="rest", salience=0.0,
            novelty=0.0, prediction_error=0.0, importance=0.0, spike_rate=0.0,
            resting=True,
        )
        self.assertEqual(fixture.recent_token_context, self.turn("current", "work"))
        self.assertEqual(fixture.messages, before)
        self.assertEqual(fixture.memory.learned_parameters, packed_before)
        self.assertEqual(fixture.current_context["recentTokenCount"], len(fixture.recent_token_context))
        self.assertEqual(fixture.current_context["recentTokenHash"], token_hash(fixture.recent_token_context))
        self.assertGreater(fixture.counters["context_token_cooling_evictions"], 0)
        self.assertTrue(report["recentTokenActivity"]["dynamicCooling"])
        prompt, history = fixture._prompt_with_recent_context("next")
        self.assertEqual(history, fixture.recent_token_context)
        faded = self.tokenizer.encode("passing")
        self.assertFalse(any(
            prompt[index : index + len(faded)] == faded
            for index in range(len(prompt) - len(faded) + 1)
        ))

    def test_pure_snapshot_rollback_and_fresh_clear_keep_token_metadata_aligned(self):
        fixture = self.fixture()
        fixture._append_recent_dialogue("first", "answer", afterimage_id="a")
        snapshot = fixture._recent_dialogue_snapshot()
        fixture.memory_lifecycle.cycle = 50
        fixture._append_recent_dialogue("second", "answer", afterimage_id="b")
        fixture._restore_recent_dialogue_snapshot(snapshot)
        self.assertEqual(fixture._recent_dialogue_snapshot(), snapshot)
        saved_tokens = list(fixture.recent_token_context)
        fixture._clear_recent_dialogue()
        self.assertEqual(fixture.recent_token_context, [])
        self.assertEqual(fixture.recent_token_activity.spans, [])
        self.assertEqual(fixture.current_context["recentTokenCount"], 0)
        self.assertEqual(fixture.current_context["recentTokenHash"], token_hash([]))
        self.assertEqual(len(fixture.messages), 1)
        fixture.memory_lifecycle.afterimage_items = [_episode("a", cycle=51, strength=1.0)]
        fixture.memory_lifecycle.cycle = 51
        fixture._cool_recent_dialogue()
        self.assertEqual(fixture.recent_token_context, [])
        self.assertNotEqual(fixture.recent_token_context, saved_tokens)

    def test_afterimage_return_ids_separate_same_field_episodes_and_reuse(self):
        lifecycle = OrganicMemoryLifecycle()
        first, other = torch.zeros(8), torch.zeros(8)
        first[0], other[1] = 1.0, 1.0

        def admit(vector):
            return lifecycle._admit_afterimage(
                vector, assembly_id="shared", source="fixture", strength=0.2,
                salience=0.1, novelty=0.0, prediction_error=0.0, rehearsals=1,
                scores={}, signals={},
            )

        first_id, other_id = admit(first), admit(other)
        self.assertNotEqual(first_id, other_id)
        self.assertEqual(admit(first), first_id)
        self.assertEqual([item["id"] for item in lifecycle.afterimage_items],
                         [first_id, other_id])

    def test_source_wiring_covers_atomic_save_load_fresh_and_slow_rollback(self):
        for method in (AdaptiveBrain.start_fresh_attention,
                       AdaptiveBrain._clear_ground_up_transient_state):
            self.assertIn("self._clear_recent_dialogue()", inspect.getsource(method))
        self.assertIn("**self._recent_dialogue_snapshot()",
                      inspect.getsource(AdaptiveBrain._snapshot_slow_transaction_state))
        self.assertIn("self._restore_recent_dialogue_snapshot(snapshot)",
                      inspect.getsource(AdaptiveBrain._restore_slow_transaction_state))
        self.assertIn('"recent_token_activity": self._recent_token_activity().metadata(',
                      inspect.getsource(AdaptiveBrain._metadata))
        self.assertIn('metadata.get("recent_token_activity")',
                      inspect.getsource(AdaptiveBrain._load_impl))


if __name__ == "__main__":
    unittest.main()
