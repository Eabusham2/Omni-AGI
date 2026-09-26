"""Pure SQLite coverage for durable replay microbatch writes."""

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

from omni_core.offload import DurableReplayBuffer, NeuralStateResourcePause


class _DiskPolicy:
    def __init__(self):
        self.blocked = False
        self.checks = []

    def require_disk(self, estimated_write_bytes, operation):
        self.checks.append((estimated_write_bytes, operation))
        if self.blocked:
            raise NeuralStateResourcePause(
                "disk reserve reached", {"diskPressure": True, "paused": True}
            )
        return {"diskPressure": False}


class _CheckpointFailingConnection:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *args):
        return self.connection.__exit__(*args)

    def execute(self, statement, *args):
        if statement.strip().upper() == "PRAGMA WAL_CHECKPOINT(FULL)":
            raise sqlite3.OperationalError("injected checkpoint failure")
        return self.connection.execute(statement, *args)


class _CheckpointBusyConnection(_CheckpointFailingConnection):
    def execute(self, statement, *args):
        if statement.strip().upper() == "PRAGMA WAL_CHECKPOINT(FULL)":
            return mock.Mock(fetchone=lambda: (1, 2, 0))
        return self.connection.execute(statement, *args)


class _StreamingOnlyCursor:
    def __init__(self, cursor, fetch_sizes, on_first_fetch=None):
        self.cursor = cursor
        self.fetch_sizes = fetch_sizes
        self.on_first_fetch = on_first_fetch

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchmany(self, size):
        if size > 32:
            raise AssertionError("checkpoint scan batch exceeds 32 rows")
        self.fetch_sizes.append(size)
        rows = self.cursor.fetchmany(size)
        if self.on_first_fetch is not None:
            callback = self.on_first_fetch
            self.on_first_fetch = None
            callback()
        return rows

    def fetchall(self):
        raise AssertionError("checkpoint scan must not materialize replay rows")


class _StreamingOnlyConnection:
    def __init__(self, connection, fetch_sizes, on_first_fetch=None):
        self.connection = connection
        self.fetch_sizes = fetch_sizes
        self.on_first_fetch = on_first_fetch

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *args):
        return self.connection.__exit__(*args)

    def execute(self, statement, *args):
        return _StreamingOnlyCursor(
            self.connection.execute(statement, *args),
            self.fetch_sizes,
            self.on_first_fetch,
        )


class ReplayBatchIOTests(unittest.TestCase):
    @staticmethod
    def _content_rows(replay):
        return tuple(
            (row[0], row[2], row[3], row[4], row[5])
            for row in replay.verified_rows()
        )

    def test_batch_matches_sequential_order_checksums_and_high_water(self):
        values = [
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([3, -5, 7], dtype=torch.int16),
            torch.arange(20000, dtype=torch.float32),
        ]
        with tempfile.TemporaryDirectory(prefix="omni-replay-batch-") as folder:
            sequential_policy = _DiskPolicy()
            batch_policy = _DiskPolicy()
            sequential = DurableReplayBuffer(
                Path(folder) / "sequential.sqlite3", sequential_policy
            )
            batch = DurableReplayBuffer(
                Path(folder) / "batch.sqlite3", batch_policy
            )
            sequential_ids = tuple(sequential.append(value) for value in values)
            batch_ids = batch.append_many(iter(values))

            self.assertEqual(batch_ids, sequential_ids)
            self.assertEqual(self._content_rows(batch), self._content_rows(sequential))
            self.assertEqual(batch.checkpoint(), sequential.checkpoint())
            self.assertEqual(batch.checkpoint()["highWaterId"], batch_ids[-1])
            self.assertEqual(len(batch_policy.checks), 1)
            self.assertEqual(
                batch_policy.checks[0],
                (sum(estimate for estimate, _ in sequential_policy.checks),
                 "latent replay spill"),
            )
            self.assertEqual(batch.append_many(()), ())
            self.assertEqual(len(batch_policy.checks), 1)

    def test_reserve_pause_precedes_every_batch_write(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-reserve-") as folder:
            policy = _DiskPolicy()
            replay = DurableReplayBuffer(Path(folder) / "replay.sqlite3", policy)
            baseline = replay.checkpoint()
            policy.blocked = True

            with self.assertRaises(NeuralStateResourcePause):
                replay.append_many(
                    [torch.ones(2), torch.ones(3), torch.ones(4)]
                )

            self.assertEqual(len(policy.checks), 1)
            self.assertEqual(replay.checkpoint(), baseline)

    def test_insert_failure_rolls_back_entire_batch(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-rollback-") as folder:
            replay = DurableReplayBuffer(
                Path(folder) / "replay.sqlite3", _DiskPolicy()
            )
            replay.append(torch.ones(1))
            baseline = replay.checkpoint()
            baseline_rows = self._content_rows(replay)
            with replay._connect() as connection:
                connection.execute(
                    """
                    CREATE TRIGGER reject_second_batch_row
                    BEFORE INSERT ON replay
                    WHEN NEW.shape_json = '[3]'
                    BEGIN
                        SELECT RAISE(ABORT, 'injected insert failure');
                    END
                    """
                )

            with self.assertRaisesRegex(sqlite3.IntegrityError, "injected insert failure"):
                replay.append_many([torch.ones(2), torch.ones(3)])

            self.assertEqual(self._content_rows(replay), baseline_rows)
            self.assertEqual(replay.checkpoint(), baseline)

    def test_checkpoint_failure_does_not_make_committed_batch_retryable(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-checkpoint-") as folder:
            path = Path(folder) / "replay.sqlite3"
            policy = _DiskPolicy()
            replay = DurableReplayBuffer(path, policy)
            connect = replay._connect
            with mock.patch.object(
                replay,
                "_connect",
                side_effect=lambda: _CheckpointFailingConnection(connect()),
            ):
                sequence_ids = replay.append_many([torch.ones(2), torch.ones(3)])

            self.assertEqual(sequence_ids, (1, 2))
            self.assertEqual(replay.last_checkpoint_error, "injected checkpoint failure")
            self.assertEqual(len(policy.checks), 1)
            reopened = DurableReplayBuffer(path, policy, read_only=True)
            self.assertEqual(tuple(row[0] for row in reopened.verified_rows()), sequence_ids)
            self.assertEqual(reopened.checkpoint()["highWaterId"], 2)

    def test_busy_checkpoint_is_observable_after_commit(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-busy-") as folder:
            replay = DurableReplayBuffer(
                Path(folder) / "replay.sqlite3", _DiskPolicy()
            )
            connect = replay._connect
            with mock.patch.object(
                replay,
                "_connect",
                side_effect=lambda: _CheckpointBusyConnection(connect()),
            ):
                sequence_ids = replay.append_many([torch.ones(2)])

            self.assertEqual(sequence_ids, (1,))
            self.assertIn("busy", replay.last_checkpoint_error)
            self.assertEqual(len(replay), 1)

    def test_checkpoint_scan_matches_materialized_digest_and_pending_semantics(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-stream-") as folder:
            replay = DurableReplayBuffer(
                Path(folder) / "replay.sqlite3", _DiskPolicy()
            )
            empty = replay.checkpoint_for_rows(replay.verified_rows())
            self.assertEqual(replay.checkpoint(), empty)
            replay.append_many(
                torch.tensor([float(index + 1), float(index + 2)])
                for index in range(73)
            )
            with replay._connect() as connection:
                connection.execute("DELETE FROM replay WHERE sequence = 5")

            rows = replay.verified_rows()
            committed = replay.checkpoint_for_rows(rows[:40])
            complete = replay.checkpoint_for_rows(rows)
            self.assertEqual(committed["count"], 40)
            self.assertEqual(committed["highWaterId"], 41)
            self.assertEqual(complete["count"], 72)
            self.assertEqual(complete["highWaterId"], 73)

            fetch_sizes = []
            connect = replay._connect
            with mock.patch.object(
                replay, "verified_rows", side_effect=AssertionError("materialized scan")
            ), mock.patch.object(
                replay,
                "_connect",
                side_effect=lambda: _StreamingOnlyConnection(connect(), fetch_sizes),
            ):
                self.assertEqual(replay.checkpoint(), complete)
                self.assertEqual(
                    replay.verify_checkpoint(empty),
                    {
                        "committedExamples": 0,
                        "durableExamples": 72,
                        "pendingExamples": 72,
                        "contentSha256": empty["contentSha256"],
                    },
                )
                self.assertEqual(
                    replay.verify_checkpoint(committed),
                    {
                        "committedExamples": 40,
                        "durableExamples": 72,
                        "pendingExamples": 32,
                        "contentSha256": committed["contentSha256"],
                    },
                )
                self.assertEqual(replay.verify_checkpoint(complete)["pendingExamples"], 0)
            self.assertGreaterEqual(len(fetch_sizes), 9)
            self.assertEqual(set(fetch_sizes), {32})

            read_only = DurableReplayBuffer(replay.path, _DiskPolicy(), read_only=True)
            self.assertEqual(read_only.verify_checkpoint(committed)["pendingExamples"], 32)

            with self.assertRaisesRegex(ValueError, "checkpoint checksum mismatch"):
                replay.verify_checkpoint({**committed, "highWaterId": 42})

    def test_checkpoint_rejects_corrupt_committed_and_pending_payloads(self):
        for corrupt_sequence in (1, 2):
            with self.subTest(corrupt_sequence=corrupt_sequence):
                with tempfile.TemporaryDirectory(prefix="omni-replay-corrupt-") as folder:
                    replay = DurableReplayBuffer(
                        Path(folder) / "replay.sqlite3", _DiskPolicy()
                    )
                    replay.append(torch.tensor([3.0, 4.0]))
                    committed = replay.checkpoint()
                    replay.append(torch.tensor([5.0, 6.0]))
                    with replay._connect() as connection:
                        connection.execute(
                            "UPDATE replay SET payload = zeroblob(length(payload)) "
                            "WHERE sequence = ?",
                            (corrupt_sequence,),
                        )
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        replay.verify_checkpoint(committed)
                    with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                        replay.checkpoint()

    def test_checkpoint_scan_uses_one_snapshot_when_writer_appends(self):
        with tempfile.TemporaryDirectory(prefix="omni-replay-snapshot-") as folder:
            path = Path(folder) / "replay.sqlite3"
            replay = DurableReplayBuffer(path, _DiskPolicy())
            replay.append_many(
                torch.tensor([float(index + 1)]) for index in range(40)
            )
            before = replay.checkpoint()
            writer_ran = []

            def append_during_scan():
                writer = sqlite3.connect(str(path), timeout=5.0)
                try:
                    writer.execute(
                        "INSERT INTO replay "
                        "(created_at, sha256, dtype, shape_json, payload) "
                        "SELECT created_at, sha256, dtype, shape_json, payload "
                        "FROM replay WHERE sequence = 1"
                    )
                    writer.commit()
                    writer_ran.append(True)
                finally:
                    writer.close()

            connect = replay._connect
            with mock.patch.object(
                replay,
                "_connect",
                side_effect=lambda: _StreamingOnlyConnection(
                    connect(), [], append_during_scan
                ),
            ):
                self.assertEqual(replay.checkpoint(), before)

            self.assertEqual(writer_ran, [True])
            self.assertEqual(replay.checkpoint()["count"], before["count"] + 1)


if __name__ == "__main__":
    unittest.main()
