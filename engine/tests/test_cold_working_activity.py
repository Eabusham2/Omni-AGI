"""Tiny SQLite/numeric pause-clock checks; no AdaptiveBrain construction."""
from __future__ import annotations

import hashlib
import json
import ast
from contextlib import closing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from omni_core.activity_clock import ActivityClock
from omni_core.connected_runtime import _adopt_working_page
from omni_core.offload import PagedWorkingMemory
from omni_core.working_activity import (checkpoint_activity_time, checkpoint_wall_time,
    page_activity_at, read_activity_baseline, rebase_page_activity)


class TinyPolicy:
    def __init__(self): self.writes = []
    def require_disk(self, size, label):
        if size > 1024 * 1024:
            raise AssertionError("numeric fixture requested a large allocation")
        self.writes.append((size, label))


class Tick:
    def __init__(self): self.value = 1000.0
    def __call__(self): return self.value
    def advance(self, seconds): self.value += seconds


def body(pages, identifier):
    with pages._connect() as database:
        row = database.execute("SELECT metadata_json,updated_at,sha256,payload FROM working_pages WHERE page_id=?", (identifier,)).fetchone()
    return row


def priorities(pages):
    with pages._connect() as database:
        return {assembly: pages._cold_priority(assembly_id=assembly, metadata=json.loads(metadata),
            updated_at=pages.activity_timestamp(json.loads(metadata), stamp), active_ids=frozenset(),
            unfinished_ids=frozenset(), now=pages._ranking_now()) for assembly, metadata, stamp
            in database.execute("SELECT assembly_id,metadata_json,updated_at FROM working_pages")}


class ColdWorkingActivityTests(unittest.TestCase):
    def test_week_unloaded_preserves_scores_retrieval_and_trim_with_real_wall_audit(self):
        with TemporaryDirectory(prefix="omni-cold-clock-") as directory:
            tick = Tick(); clock = ActivityClock(clock=tick); wall = 1_700_000_000.0
            pages = PagedWorkingMemory(Path(directory) / "working.sqlite3", TinyPolicy())
            pages.bind_activity_clock(clock.now, checkpoint_wall_at=wall)
            with patch("omni_core.offload.time.time", return_value=wall):
                a = pages.append(torch.tensor([1.0]), {"assemblyId": "a", "salience": .20})
            tick.advance(300)
            with patch("omni_core.offload.time.time", return_value=wall + 300):
                b = pages.append(torch.tensor([2.0]), {"assemblyId": "b", "salience": .19})
            tick.advance(300); before = priorities(pages)
            self.assertGreater(before["b"], before["a"])
            audit = {key: body(pages, key) for key in (a, b)}
            state = json.loads(json.dumps(clock.pause())); tick.advance(7 * 86400)
            restored_clock = ActivityClock.from_state(state, clock=tick)
            restored = PagedWorkingMemory(pages.path, TinyPolicy())
            restored.bind_activity_clock(restored_clock.now, checkpoint_wall_at=wall + 7 * 86400 + 600)
            with patch("omni_core.offload.time.time", return_value=wall + 7 * 86400 + 600):
                self.assertEqual(priorities(restored), before)
                self.assertEqual(restored._rank_hot_page_ids(hot_assembly_ids=("a", "b"), limit=2), [b, a])
                restored.trim_to(1)
            self.assertEqual(restored.page_ids(), (b,))
            self.assertEqual(body(restored, b), audit[b])
            # The former wall-age formula would flip this actual choice.
            old = {name: restored._cold_priority(assembly_id=name, metadata=json.loads(row[0]),
                updated_at=row[1], active_ids=frozenset(), unfinished_ids=frozenset(),
                now=wall + 7 * 86400 + 600) for name, row in (("a", audit[a]), ("b", audit[b]))}
            self.assertGreater(old["a"], old["b"])

    def test_binding_is_read_only_and_legacy_baseline_survives_a_later_checkpoint(self):
        with TemporaryDirectory(prefix="omni-cold-legacy-") as directory:
            policy = TinyPolicy(); pages = PagedWorkingMemory(Path(directory) / "working.sqlite3", policy)
            with patch("omni_core.offload.time.time", return_value=900.0):
                key = pages.append(torch.tensor([3.0]), {"assemblyId": "old", "salience": .5})
            original = body(pages, key); digest = hashlib.sha256(pages.path.read_bytes()).hexdigest()
            before_writes = len(policy.writes); tick = Tick(); clock = ActivityClock(50.0, paused=True, clock=tick)
            pages.bind_activity_clock(clock.now, checkpoint_wall_at=1000.0)
            self.assertEqual(hashlib.sha256(pages.path.read_bytes()).hexdigest(), digest)
            self.assertEqual(len(policy.writes), before_writes)
            with pages._connect() as database:
                self.assertIsNone(read_activity_baseline(database))
            self.assertEqual(pages.activity_timestamp(json.loads(original[0]), original[1]), -50.0)
            score = priorities(pages)
            pages.persist_activity_baseline(); self.assertEqual(body(pages, key), original)
            tick.advance(7 * 86400)
            reopened = PagedWorkingMemory(pages.path, TinyPolicy())
            reopened.bind_activity_clock(ActivityClock.from_state(clock.snapshot(), clock=tick).now,
                checkpoint_wall_at=1000.0 + 7 * 86400)
            self.assertEqual(priorities(reopened), score)
            self.assertEqual(body(reopened, key), original)

    def test_unknown_old_age_is_not_rejuvenated_or_inferred_from_file_time(self):
        with TemporaryDirectory(prefix="omni-cold-unknown-") as directory:
            pages = PagedWorkingMemory(Path(directory) / "working.sqlite3", TinyPolicy())
            with patch("omni_core.offload.time.time", return_value=12345.0):
                key = pages.append(torch.tensor([4.0]), {"assemblyId": "unknown", "salience": 0.0})
            pages.bind_activity_clock(lambda: 10.0, checkpoint_wall_at=None)
            self.assertIsNone(pages.activity_timestamp(json.loads(body(pages, key)[0]), 12345.0))
            self.assertEqual(priorities(pages), {"unknown": 0.0})
            self.assertIsNone(checkpoint_wall_time("no saved timestamp"))
            self.assertIsNone(checkpoint_wall_time("2026-10-13T12:00:00"))

    def test_touch_updates_activity_and_wall_audit_together_without_changing_payload(self):
        with TemporaryDirectory(prefix="omni-cold-touch-") as directory:
            pages = PagedWorkingMemory(Path(directory) / "working.sqlite3", TinyPolicy())
            pages.bind_activity_clock(lambda: 7.0, checkpoint_wall_at=1000.0)
            with patch("omni_core.offload.time.time", return_value=1000.0):
                key = pages.append(torch.tensor([5.0]), {"assemblyId": "touch"})
            original = body(pages, key)
            with pages._connect() as database:
                database.execute("CREATE TRIGGER fail_touch BEFORE UPDATE ON working_pages BEGIN SELECT RAISE(ABORT,'owned touch failure'); END")
            with patch("omni_core.offload.time.time", return_value=999999.0):
                with self.assertRaisesRegex(sqlite3.IntegrityError, "owned touch failure"):
                    pages.read(key)
            self.assertEqual(body(pages, key), original)
            with pages._connect() as database:
                database.execute("DROP TRIGGER fail_touch")
            pages._activity_now = lambda: 9.0
            with patch("omni_core.offload.time.time", return_value=999999.0):
                pages.read(key)
            changed = body(pages, key)
            self.assertEqual(json.loads(changed[0])["activityUpdatedAt"], 9.0)
            self.assertEqual(changed[1], 999999.0)
            self.assertEqual(changed[2:], original[2:])
            self.assertEqual(pages.status()["lastReadAt"], 999999.0)

    def test_actual_adoption_preserves_age_and_audit_including_future_and_unknown_stamps(self):
        with TemporaryDirectory(prefix="omni-cold-adopt-") as directory:
            policy = TinyPolicy(); pages = PagedWorkingMemory(Path(directory) / "working.sqlite3", policy)
            pages.bind_activity_clock(lambda: 20.0, checkpoint_wall_at=1000.0)
            root = SimpleNamespace(paged_working_memory=pages, resource_policy=policy)
            for identifier, source_at in (("older", -50.0), ("future", 120.0), ("unknown", None)):
                metadata = {"assemblyId": identifier, "salience": .5}
                _adopt_working_page(root, identifier, torch.tensor([6.0]), metadata, 8765.0,
                    source_activity_now=100.0, source_activity_at=source_at)
                row = body(pages, identifier); adopted = json.loads(row[0])
                self.assertEqual(row[1], 8765.0)
                actual = pages.activity_timestamp(adopted, row[1])
                if source_at is None:
                    self.assertIsNone(actual)
                else:
                    self.assertEqual(max(0.0, 20.0 - actual), max(0.0, 100.0 - source_at))
            future = rebase_page_activity({}, source_now=10.0, source_at=30.0, target_now=2.0)
            self.assertEqual(future["activityUpdatedAt"], 2.0)

    def test_saved_coordinate_validation_does_not_charge_hydration_or_accept_corruption(self):
        state = ActivityClock(12.25, paused=True).snapshot(); state["paused"] = False
        self.assertEqual(checkpoint_activity_time({"activity_clock": state}), 12.25)
        self.assertEqual(checkpoint_activity_time({"last_activity_decay": 50.0}), 50.0)
        baseline = {"activityAt": 20.0, "wallAt": 1000.0}
        self.assertEqual(page_activity_at({}, 1100.0, baseline), 20.0)
        for value in (float("nan"), float("inf"), True, "5"):
            with self.assertRaises(ValueError):
                page_activity_at({"activityUpdatedAt": value}, 1000.0, baseline)
        with TemporaryDirectory(prefix="omni-cold-corrupt-") as directory:
            pages = PagedWorkingMemory(Path(directory) / "working.sqlite3", TinyPolicy())
            pages.bind_activity_clock(lambda: 20.0, checkpoint_wall_at=1000.0)
            pages.persist_activity_baseline()
            with pages._connect() as database:
                database.execute("DELETE FROM working_activity_clock")
            with self.assertRaisesRegex(ValueError, "missing or duplicated"):
                pages.bind_activity_clock(lambda: 20.0, checkpoint_wall_at=1000.0)

    def test_isolated_sqlite_copy_retains_baseline_but_not_a_live_clock_reference(self):
        with TemporaryDirectory(prefix="omni-cold-copy-") as directory:
            source = PagedWorkingMemory(Path(directory) / "source.sqlite3", TinyPolicy())
            with patch("omni_core.offload.time.time", return_value=800.0):
                key = source.append(torch.tensor([7.0]), {"assemblyId": "copied", "salience": .25})
            source.bind_activity_clock(lambda: 40.0, checkpoint_wall_at=1000.0)
            source.persist_activity_baseline()
            original = body(source, key); expected = priorities(source)
            destination = Path(directory) / "copy.sqlite3"
            with closing(sqlite3.connect(source.path.resolve().as_uri() + "?mode=ro", uri=True)) as incoming, closing(sqlite3.connect(destination)) as outgoing:
                incoming.backup(outgoing, pages=1)
            copied = PagedWorkingMemory(destination, TinyPolicy())
            copied.bind_activity_clock(lambda: 40.0, checkpoint_wall_at=900000.0)
            self.assertEqual(priorities(copied), expected)
            self.assertEqual(body(copied, key), original)
            copied._activity_now = lambda: 50.0
            self.assertNotEqual(priorities(copied), expected)
            self.assertEqual(priorities(source), expected)

    def test_production_binding_stays_read_only_and_new_clock_math_is_runtime_sealed(self):
        directory = Path(__file__).resolve().parents[1] / "omni_core"
        for name in ("brain.py", "connected_execution_owner.py"):
            tree = ast.parse((directory / name).read_text())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "bind_activity_clock"]
            self.assertEqual(len(calls), 2 if name == "brain.py" else 1)
            for call in calls:
                self.assertIsInstance(call.args[0], ast.Lambda)
                attributes = {node.attr for node in ast.walk(call.args[0]) if isinstance(node, ast.Attribute)}
                self.assertIn("_owned_activity_clock", attributes)
                self.assertIn("now", attributes)
                self.assertNotIn("_activity_now", attributes)
        tree = ast.parse((directory / "execution_compatibility.py").read_text())
        names = next(ast.literal_eval(node.value) for node in ast.walk(tree)
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "source_names" for target in node.targets))
        self.assertTrue({"brain.py", "activity_clock.py", "working_activity.py", "offload.py"}.issubset(names))


if __name__ == "__main__": unittest.main()
