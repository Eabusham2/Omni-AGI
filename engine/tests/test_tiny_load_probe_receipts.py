"""New small custody cases only: no Torch/model/Worker import or neural run."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from .tiny_load_probe_receipts import (content_snapshot, prepare_private_copy,
    sqlite_snapshot, strict_private_files, tree_snapshot)


class TinyLoadReceiptTests(unittest.TestCase):
    def make_source(self, root):
        engine = root / "brain" / "engine"
        engine.mkdir(parents=True)
        (engine / "brain.json").write_text(json.dumps({"brain_id": "tiny-load",
            "config": {"origin_kind": "ground-up", "d_model": 8, "n_layers": 1, "router_neurons": 4},
            "counters": {"training_steps": 0, "inference_count": 0, "experiences": 0}}))
        (engine / "core.safetensors").write_bytes(b"small-not-a-model-file")

    def make_sqlite(self, root):
        for relative in ("engine/events.sqlite3", "engine/conversation.sqlite3",
                         "engine/state/replay.sqlite3", "engine/state/working-memory.sqlite3"):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE observations(sequence INTEGER PRIMARY KEY,payload BLOB)")
                connection.execute("INSERT INTO observations VALUES(1,?)", (b"exact tiny bytes",))

    def test_private_copy_is_byte_equal_independent_and_cannot_mutate_source(self):
        with tempfile.TemporaryDirectory(prefix="omni-native-cold-load-source-") as source_name, \
                tempfile.TemporaryDirectory(prefix="omni-native-cold-load-copy-") as copy_name:
            source, root = Path(source_name), Path(copy_name)
            self.make_source(source)
            before = tree_snapshot(source)
            receipt = prepare_private_copy(source, root)
            self.assertTrue(receipt["privateFilesIndependent"])
            self.assertEqual(content_snapshot(tree_snapshot(source / "brain")), content_snapshot(tree_snapshot(root / "brain")))
            (root / "brain" / "engine" / "core.safetensors").write_bytes(b"changed private bytes")
            self.assertEqual(before, tree_snapshot(source))
            with self.assertRaisesRegex(ValueError, "new, empty"):
                prepare_private_copy(source, root)

    def test_symlink_custody_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory(prefix="omni-native-cold-load-source-") as source_name, \
                tempfile.TemporaryDirectory(prefix="omni-native-cold-load-copy-") as copy_name:
            source, root = Path(source_name), Path(copy_name)
            self.make_source(source)
            (source / "brain" / "linked").symlink_to(source / "brain" / "engine")
            with self.assertRaisesRegex(ValueError, "symlink"):
                prepare_private_copy(source, root)
            self.assertEqual(list(root.iterdir()), [])

    def test_sqlite_physical_change_keeps_logical_rows_but_tamper_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_sqlite(root)
            before = sqlite_snapshot(root)
            before_files = tree_snapshot(root)
            with sqlite3.connect(root / "engine/events.sqlite3") as connection:
                connection.execute("VACUUM")
            self.assertEqual(before, sqlite_snapshot(root))
            self.assertEqual(strict_private_files(before_files), strict_private_files(tree_snapshot(root)))
            with sqlite3.connect(root / "engine/events.sqlite3") as connection:
                connection.execute("UPDATE observations SET payload=?", (b"tampered bytes",))
            self.assertNotEqual(before, sqlite_snapshot(root))

    def test_clock_schema_and_authoritative_metadata_are_not_exempted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_sqlite(root)
            metadata = root / "engine/brain.json"
            metadata.write_text('{"activity_clock":{"elapsed":2,"paused":true}}')
            before_sql = sqlite_snapshot(root)
            before_files = tree_snapshot(root)
            with sqlite3.connect(root / "engine/state/working-memory.sqlite3") as connection:
                connection.execute("CREATE TABLE working_activity_clock(singleton INTEGER PRIMARY KEY)")
            self.assertNotEqual(before_sql, sqlite_snapshot(root))
            metadata.write_text('{"activity_clock":{"elapsed":200,"paused":true}}')
            self.assertNotEqual(strict_private_files(before_files), strict_private_files(tree_snapshot(root)))


if __name__ == "__main__":
    unittest.main()
