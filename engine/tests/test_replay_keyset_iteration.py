"""SQLite-only regression coverage for paged durable replay reads."""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.offload import DurableReplayBuffer


class _DiskPolicy:
    def require_disk(self, estimated_write_bytes, operation):
        return {"diskPressure": False}


class _TracedConnection:
    def __init__(self, connection: sqlite3.Connection, statements: list):
        self.connection = connection
        self.statements = statements

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *args):
        return self.connection.__exit__(*args)

    def execute(self, statement, parameters=()):
        self.statements.append((statement, tuple(parameters)))
        return self.connection.execute(statement, parameters)


class ReplayKeysetIterationTests(unittest.TestCase):
    @staticmethod
    def _values(start: int, stop: int):
        return (torch.tensor([float(index)]) for index in range(start, stop))

    @staticmethod
    def _decoded(replay: DurableReplayBuffer):
        return [int(value.item()) for value in replay]

    def test_pages_sparse_rows_after_truncate_without_offset(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-keyset-") as folder:
            replay = DurableReplayBuffer(Path(folder) / "replay.sqlite3", _DiskPolicy())
            replay.append_many(self._values(0, 530))
            self.assertEqual(replay.truncate(400), 130)
            suffix_ids = replay.append_many(self._values(400, 530))
            self.assertEqual(suffix_ids[0], 531)
            with replay._connect() as connection:
                connection.execute("DELETE FROM replay WHERE sequence IN (5, 542)")

            expected = [index for index in range(530) if index not in (4, 411)]
            row_ids = [int(row[0]) for row in replay.verified_rows()]
            statements = []
            connect = replay._connect
            with mock.patch.object(
                replay,
                "_connect",
                side_effect=lambda: _TracedConnection(connect(), statements),
            ):
                self.assertEqual(self._decoded(replay), expected)

            page_queries = [
                (" ".join(sql.upper().split()), parameters)
                for sql, parameters in statements
                if "SELECT sequence, sha256, dtype, shape_json, payload" in sql
            ]
            self.assertEqual(len(page_queries), 4)
            self.assertTrue(
                all("WHERE SEQUENCE > ? ORDER BY SEQUENCE LIMIT ?" in sql
                    for sql, _parameters in page_queries)
            )
            self.assertTrue(all("OFFSET" not in sql for sql, _ in page_queries))
            self.assertEqual(
                [parameters for _sql, parameters in page_queries],
                [(0, 256), (row_ids[255], 256),
                 (row_ids[511], 256), (row_ids[-1], 256)],
            )

    def test_deleting_consumed_rows_does_not_skip_later_rows_or_new_appends(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-append-") as folder:
            replay = DurableReplayBuffer(Path(folder) / "replay.sqlite3", _DiskPolicy())
            replay.append_many(self._values(0, 520))
            iterator = iter(replay)
            first_page = [int(next(iterator).item()) for _ in range(256)]

            with replay._connect() as connection:
                connection.execute("DELETE FROM replay WHERE sequence <= 10")
            replay.append_many(self._values(520, 523))

            self.assertEqual(first_page + [int(value.item()) for value in iterator],
                             list(range(523)))

    def test_corruption_in_later_page_still_raises_checksum_mismatch(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-corrupt-") as folder:
            replay = DurableReplayBuffer(Path(folder) / "replay.sqlite3", _DiskPolicy())
            replay.append_many(self._values(1, 301))
            with replay._connect() as connection:
                connection.execute(
                    "UPDATE replay SET payload = zeroblob(length(payload)) "
                    "WHERE sequence = 280"
                )

            iterator = iter(replay)
            self.assertEqual([int(next(iterator).item()) for _ in range(256)],
                             list(range(1, 257)))
            with self.assertRaisesRegex(ValueError, "replay tensor checksum mismatch"):
                list(iterator)


if __name__ == "__main__":
    unittest.main()
