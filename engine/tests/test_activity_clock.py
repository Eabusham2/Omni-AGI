"""Active decay and idle cadence exclude a long unloaded wall interval."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

from omni_core.activity_clock import ActivityClock
from omni_core.brain import AdaptiveBrain


class FakeClock:
    def __init__(self): self.value = 1000.0
    def __call__(self): return self.value
    def advance(self, value): self.value += value


class ActivityClockTests(unittest.TestCase):
    def test_pause_restart_ignores_a_week_but_preserves_active_time_and_cadence(self):
        timer = FakeClock()
        uninterrupted = ActivityClock(clock=timer)
        timer.advance(5)
        state = uninterrupted.pause()
        self.assertEqual(state["elapsed"], 5.0)
        timer.advance(7 * 86400)
        restored = ActivityClock.from_state(copy.deepcopy(state), clock=timer)
        self.assertEqual(restored.now(), 5.0)
        restored.resume(); timer.advance(5)
        self.assertEqual(restored.now(), 10.0)
        self.assertEqual(uninterrupted.now(), 5.0)
        restored.pause(); restored.pause(); timer.advance(400)
        self.assertEqual(restored.snapshot()["elapsed"], 10.0)

    def test_real_experience_decay_and_idle_cooldown_match_equal_active_time(self):
        decays = []
        class EndOfDecay(Exception): pass
        def learned(*_args, **_kwargs): raise EndOfDecay()
        def owner(clock):
            value = SimpleNamespace(activity_clock=clock, last_activity_decay=0.0, last_idle_cycle_at=0.0,
                last_idle_visible_action_at=0.0, brain_id="fixture", config=SimpleNamespace(vector_symbolic_memory=True, short_term_half_life_minutes=1),
                memory=SimpleNamespace(decay=lambda amount, **_kw: decays.append(amount), learn=learned))
            value._owned_activity_clock = MethodType(AdaptiveBrain._owned_activity_clock, value)
            value._activity_now = MethodType(AdaptiveBrain._activity_now, value)
            return value
        timer = FakeClock(); uninterrupted = owner(ActivityClock(clock=timer)); timer.advance(10)
        with self.assertRaises(EndOfDecay):
            AdaptiveBrain._learn_experience_impl(uninterrupted, "new ordinary experience", kind="knowledge", source="chat", source_label="", steps=None, importance=.5, structural_detail=True)
        other_timer = FakeClock(); paused = ActivityClock(clock=other_timer); other_timer.advance(5); state = paused.pause()
        other_timer.advance(7 * 86400); resumed = owner(ActivityClock.from_state(state, clock=other_timer)); resumed.activity_clock.resume(); other_timer.advance(5)
        with self.assertRaises(EndOfDecay):
            AdaptiveBrain._learn_experience_impl(resumed, "new ordinary experience", kind="knowledge", source="chat", source_label="", steps=None, importance=.5, structural_detail=True)
        self.assertEqual(decays[0], decays[1]); self.assertEqual(uninterrupted.last_activity_decay, resumed.last_activity_decay)
        # The real idle implementation returns before allocating any brain state.
        raw_idle = AdaptiveBrain.idle_cycle.__wrapped__
        self.assertEqual(raw_idle(uninterrupted, minimum_idle_seconds=45), raw_idle(resumed, minimum_idle_seconds=45))

    def test_only_known_clock_state_is_accepted_and_legacy_markers_are_not_bulk_shifted(self):
        timer = FakeClock(); old = ActivityClock.from_state(None, legacy_markers=(1700000000.0, 0.0), clock=timer)
        self.assertTrue(old.paused); self.assertEqual(old.now(), 1700000000.0)
        for change in ({"elapsed": float("nan")}, {"paused": 1}, {"elapsed": -1}, {"extra": "untrusted"}, {"version": True}):
            with self.assertRaises(ValueError): ActivityClock.from_state({**old.snapshot(), **change}, clock=timer)

    def test_read_only_supported_frontier_requires_exact_own_clock_before_execution_adoption(self):
        from omni_core.execution_checkpoint import ExecutionCheckpointStore
        from omni_core.execution_state_codec import ExecutionStateWriter
        with TemporaryDirectory(prefix="omni-activity-seal-") as folder:
            engine = Path(folder); state = ActivityClock(12.25, paused=True).snapshot()
            metadata = {"brain_id": "fixture", "activity_clock": state}
            (engine / "brain.json").write_text(json.dumps(metadata))
            payload = ExecutionStateWriter(engine).encode({"activityClock": state})
            value = {"format": "omni-native-execution-checkpoint", "version": 2, "status": "suspended", "brainId": "fixture",
                "backend": "cpu", "runtimeCompatibility": {"verified-by-this-test": True}, "payload": payload}
            store = ExecutionCheckpointStore(engine, "fixture")
            with patch("omni_core.execution_compatibility.require_execution_compatibility") as check:
                self.assertIs(store.require_supported_frontier(value), value)
                check.assert_called_once_with(value["runtimeCompatibility"], "cpu")
                changed = copy.deepcopy(metadata); changed["activity_clock"]["elapsed"] += 1
                (engine / "brain.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "differs"):
                    store.require_supported_frontier(value)
            with self.assertRaisesRegex(ValueError, "predates"):
                store.require_supported_frontier({**value, "runtimeCompatibility": None})


if __name__ == "__main__": unittest.main()
