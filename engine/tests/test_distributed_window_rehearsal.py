"""Committed-window curriculum routing and real snapshot storage, no models."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.capability_rehearsal import CapabilityRehearsalPolicy, CapabilityScheduleState
from omni_core.distributed_runtime import DatasetManifest, DistributedRunStore, RankCursor
from omni_core.distributed_training import (
    DistributedGroundUpTrainer, _canonical_sha256, _due_distributed_rehearsal_phase,
    _rehearse_bound_native_cursors,
)
from omni_core.persistence import atomic_write_json, read_json


class DistributedWindowRehearsal(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        source = root / "source.txt"
        source.write_text("one literal record with many remaining causal windows", encoding="utf-8")
        manifest = DatasetManifest.build(source)
        entry = manifest.entries[0]
        window = {"format": "omni-record-token-windows", "formatVersion": 1,
            "recordId": entry.record_id, "ordinal": 0, "contentSha256": entry.content_sha256,
            "phase": "text", "tokenCursor": 3, "completedWindows": 1, "pairIndex": 0, "sequenceTokens": 4}
        cursors = [RankCursor(0, 2, 0, 0, 0, 1, manifest.content_sha256, window),
            RankCursor(1, 2, 0, 1, 0, 1, manifest.content_sha256)]
        return manifest, cursors

    def due(self, state, *, active=True, waves=1):
        return _due_distributed_rehearsal_phase(state, CapabilityRehearsalPolicy(),
            committed_global_waves=waves, record_count=1, global_batch_records=16,
            epochs=1, completed_epochs=0, next_global_ordinal=0, active_record_window=active)

    def test_real_nonfinal_window_triggers_middle_and_saved_schedule_does_not_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            _manifest, cursors = self.fixture(directory)
            state = CapabilityScheduleState(start_completed=True, event_count=1)
            self.assertIsNone(self.due(state, active=False))
            self.assertIsNone(self.due(state, waves=0))
            self.assertEqual(self.due(state), "middle")
            positions = copy.deepcopy([cursor.to_dict() for cursor in cursors])
            def rehearse(_brain, **kwargs):
                return {"format": "omni-capability-rehearsal", "phase": kwargs["phase"],
                    "committedGlobalWaves": kwargs["committed_global_waves"]}
            with patch("omni_core.distributed_training.rehearse_capabilities", side_effect=rehearse):
                advanced = _rehearse_bound_native_cursors(None, cursors, state,
                    phase="middle", global_steps=1, policy=CapabilityRehearsalPolicy())
            self.assertEqual([cursor.to_dict() for cursor in cursors], positions)
            self.assertEqual(advanced.last_receipt["committedCursorBoundary"]["rankCursorsSha256"], _canonical_sha256(positions))
            resumed = CapabilityScheduleState.from_dict(advanced.to_dict())
            self.assertIsNone(self.due(resumed, waves=2))

    def test_mutated_cursor_or_nonmiddle_partial_record_fails_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            _manifest, cursors = self.fixture(directory)
            state = CapabilityScheduleState(start_completed=True, event_count=1)
            with patch("omni_core.distributed_training.rehearse_capabilities") as run:
                with self.assertRaisesRegex(RuntimeError, "labelled-window"):
                    _rehearse_bound_native_cursors(None, cursors, state,
                        phase="final", global_steps=1, policy=CapabilityRehearsalPolicy())
                run.assert_not_called()
            def corrupt(_brain, **kwargs):
                cursors[0].record_window["tokenCursor"] += 1
                return {"format": "omni-capability-rehearsal", "phase": "middle", "committedGlobalWaves": 1}
            with patch("omni_core.distributed_training.rehearse_capabilities", side_effect=corrupt):
                with self.assertRaisesRegex(RuntimeError, "cursor accounting"):
                    _rehearse_bound_native_cursors(None, cursors, state,
                        phase="middle", global_steps=1, policy=CapabilityRehearsalPolicy())

    def test_full_canonical_storage_binds_rehearsed_state_to_exact_unadvanced_rank_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, cursors = self.fixture(directory)
            trainer = DistributedGroundUpTrainer.__new__(DistributedGroundUpTrainer)
            trainer.store = DistributedRunStore(root / "run")
            trainer.context = SimpleNamespace(rank=0, device=SimpleNamespace(type="cpu"),
                is_rank_zero=True, distributed=False)
            trainer.options = SimpleNamespace(capability_rehearsal_waves=128, keep_checkpoints=2, epochs=1)
            trainer._training_policy_sha256 = lambda: "a" * 64
            storage = trainer.store.path / "checkpoint-brain"
            engine = storage / "engine"
            brain = SimpleNamespace(storage_path=storage, engine_path=engine, learned_trit=0,
                _optimizer=SimpleNamespace(state={}), resource_policy=SimpleNamespace(
                    status=lambda: {}, require_disk=lambda *_: None), close=lambda: None)
            brain.stage_distributed_training_seal = lambda value: setattr(brain, "seal", value)
            def save():
                atomic_write_json(engine / "brain.json", {"distributed_training_seal": brain.seal,
                    "learnedTrit": brain.learned_trit})
                (engine / "packed.fixture").write_bytes(bytes([brain.learned_trit & 0xff]))
            brain.save = save
            def rehearse(target, **kwargs):
                target.learned_trit = 1
                return {"format": "omni-capability-rehearsal", "phase": kwargs["phase"],
                    "committedGlobalWaves": kwargs["committed_global_waves"]}
            replay = SimpleNamespace(position=0, consume_until=lambda stop: iter(()))
            old_positions = copy.deepcopy([cursor.to_dict() for cursor in cursors])
            with patch("omni_core.distributed_training.sample_resources", return_value=None), \
                patch("omni_core.distributed_training.aggregate_resource_readings", return_value={"diskPressure": False, "memoryPressure": False}), \
                patch("omni_core.distributed_training.native_topology_sha256", return_value="b" * 64), \
                patch("omni_core.distributed_training.rehearse_capabilities", side_effect=rehearse):
                stop, schedule, _media = trainer._checkpoint_native_collective(brain=brain, wrapped=None,
                    optimizer=SimpleNamespace(state={}), manifest=manifest, cursors=cursors,
                    dynamic_high_water=0, global_steps=1, strategy="packed", final_pack=False,
                    schedule_state=CapabilityScheduleState(start_completed=True, event_count=1),
                    rehearsal_phase="middle", media_training_state={}, media_replay=replay)
            checkpoint = trainer.store.load_active_checkpoint(manifest_sha256=manifest.content_sha256, world_size=2)
            self.assertEqual(stop, 0)
            self.assertEqual(checkpoint["rankCursors"], old_positions)
            self.assertEqual(checkpoint["capabilityRehearsal"], schedule.to_dict())
            committed_engine = trainer.store.published_native_path(checkpoint) / "engine"
            self.assertEqual(read_json(committed_engine / "brain.json")["learnedTrit"], 1)
            self.assertEqual((committed_engine / "packed.fixture").read_bytes(), b"\x01")
            resumed = CapabilityScheduleState.from_dict(checkpoint["capabilityRehearsal"])
            self.assertIsNone(self.due(resumed, waves=2))


if __name__ == "__main__": unittest.main()
