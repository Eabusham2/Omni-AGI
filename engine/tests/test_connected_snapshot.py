import hashlib
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omni_core.full_merge import connected_graph_snapshot_entries, copy_connected_graph_snapshot
from omni_core.persistence import atomic_write_json, snapshot_files, snapshot_required_bytes
from omni_core.vsa import NeuralSubstrate


def sealed(regions):
    value = {"format": "omni-connected-graph-states", "formatVersion": 1, "regions": regions}
    value["sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


class ConnectedSnapshotTests(unittest.TestCase):
    def setup_graph(self, engine, region="r1", *, nested=False):
        path = engine / "connected-state" / region
        memory = NeuralSubstrate(16, seed=9)
        memory._ensure_neuron("saved", 1.)
        memory.save_sharded(path / "substrate")
        record = {"enginePath": "engine/connected-state/" + region,
                  "substrate": memory.metadata(include_records=False)}
        if nested:
            record["nestedGraphStates"] = self.setup_graph(path / "nested" / "engine", "r2")
            record["nestedStateRoot"] = record["enginePath"] + "/nested"
        atomic_write_json(path / "brain.json", {"orphanedRecoveryMirror": True})
        return sealed({region: record})

    def test_nested_graph_closure_from_arbitrarily_named_engine_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "birth", root / "isolated"
            state = self.setup_graph(source, nested=True)
            target.mkdir()
            copy_connected_graph_snapshot(root, state, target, source_engine=source)
            for relative, data, expected in connected_graph_snapshot_entries(source, state):
                output = target / relative
                self.assertTrue(output.is_file(), relative)
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), expected)
            self.assertIn("substrate", json.loads((target / "connected-state/r1/brain.json").read_text()))
            self.assertIn("orphanedRecoveryMirror", json.loads((source / "connected-state/r1/brain.json").read_text()))

    def test_snapshot_keeps_committed_wal_rows_and_budget_covers_connected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "birth"
            state = self.setup_graph(source, nested=True)
            atomic_write_json(source / "brain.json", {"connected_regions": {}, "connected_graph_states": state})
            (source / "core.safetensors").write_bytes(b"fixture-core")
            (source / "plasticity.safetensors").write_bytes(b"fixture-plasticity")
            live = sqlite3.connect(source / "connected-runtime.sqlite3")
            try:
                live.execute("PRAGMA journal_mode=WAL")
                live.execute("PRAGMA wal_autocheckpoint=0")
                live.execute("CREATE TABLE history (id INTEGER PRIMARY KEY)")
                live.execute("INSERT INTO history VALUES (7)")
                live.commit()
                # This fixture isolates graph/WAL copying. The complete real
                # parent descriptor/input closure is exercised by the native
                # merge→stable→isolated-engine lifecycle fixture.
                with patch("omni_core.full_merge._connected_input_sources", side_effect=lambda *args, **kwargs: iter(())):
                    expected = snapshot_required_bytes(source)
                    target = root / "copy"
                    snapshot_files(source, target)
                with sqlite3.connect(target / "connected-runtime.sqlite3") as restored:
                    self.assertEqual(restored.execute("SELECT id FROM history").fetchall(), [(7,)])
                self.assertGreaterEqual(expected, sum(path.stat().st_size for path in target.rglob("*") if path.is_file()))
            finally:
                live.close()

    def test_rejects_corrupt_declared_blob_and_redirected_region(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.setup_graph(root / "engine")
            entries = list(connected_graph_snapshot_entries(root / "engine", state))
            source = next(value for relative, value, _ in entries if "/blobs/" in relative)
            source.write_bytes(b"changed")
            target = root / "target"
            target.mkdir()
            with self.assertRaisesRegex(ValueError, "committed blob"):
                copy_connected_graph_snapshot(root, state, target)
            state["regions"]["r1"]["enginePath"] = "../outside"
            state = sealed(state["regions"])
            with self.assertRaisesRegex(ValueError, "path"):
                list(connected_graph_snapshot_entries(root / "engine", state))


if __name__ == "__main__":
    unittest.main()
