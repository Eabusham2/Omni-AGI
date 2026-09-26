"""Pure indexed cold-page ranking checks; no brain Build or training."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from omni_core.offload import PagedWorkingMemory


class _NoWritesNeeded:
    def require_disk(self, _estimated_bytes, _label):
        raise AssertionError("rank-only tests must not spill new pages")


def _prior_full_scan(pages, hot_ids, unfinished_ids=(), limit=1):
    """Former selection rule, retained in the test as an exact oracle."""

    active = frozenset(str(value) for value in hot_ids if value)
    unfinished = frozenset(str(value) for value in unfinished_ids if value)
    candidates = active.union(unfinished)
    maximum = max(0, int(limit))
    if not candidates or maximum == 0:
        return []
    now = time.time()
    ranked = []
    with pages._connect() as connection:
        rows = connection.execute(
            "SELECT sequence, page_id, assembly_id, metadata_json, "
            "updated_at FROM working_pages"
        ).fetchall()
    for sequence, page_id, assembly_id, metadata_json, updated_at in rows:
        assembly_key = str(assembly_id)
        if assembly_key not in candidates:
            continue
        try:
            metadata_value = json.loads(str(metadata_json))
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata_value = {}
        metadata = metadata_value if isinstance(metadata_value, dict) else {}
        priority = pages._cold_priority(
            assembly_id=assembly_key,
            metadata=metadata,
            updated_at=float(updated_at),
            active_ids=active,
            unfinished_ids=unfinished,
            now=now,
        )
        ranked.append((priority, int(sequence), str(page_id)))
    return [page_id for _priority, _sequence, page_id in sorted(
        ranked, reverse=True
    )[:maximum]]


class PagedWorkingIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-indexed-pages-")
        self.addCleanup(temporary.cleanup)
        self.pages = PagedWorkingMemory(
            Path(temporary.name) / "pages.sqlite3", _NoWritesNeeded()
        )
        rows = []
        for index in range(1_900):
            assembly = (
                f"focus-{index % 1_200:04d}"
                if index < 1_400
                else f"unrelated-{index:04d}"
            )
            metadata = json.dumps({
                "assemblyId": assembly,
                "salience": (index % 10) / 10.0,
                "retentionScore": (index % 7) / 7.0,
                "rehearsals": index % 5,
            }, sort_keys=True)
            rows.append((
                f"page-{index:04d}", assembly, metadata,
                "0" * 64, "torch.float32", "[1]", b"\0\0\0\0",
                1_700_000_000.0 + index,
            ))
        with self.pages._connect() as connection:
            connection.executemany(
                "INSERT INTO working_pages "
                "(page_id, assembly_id, metadata_json, sha256, dtype, "
                "shape_json, payload, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def test_indexed_ranking_matches_full_scan_for_small_and_chunked_candidates(self):
        cases = [
            ({"focus-0001", "focus-0010"}, {"focus-0600"}, 5),
            ({f"focus-{index:04d}" for index in range(1_200)}, set(), 16),
            ({"missing"}, set(), 3),
            (set(), set(), 3),
        ]
        with mock.patch("omni_core.offload.time.time", return_value=1_700_001_000.0):
            for active, unfinished, limit in cases:
                with self.subTest(candidates=len(active) + len(unfinished)):
                    expected = _prior_full_scan(
                        self.pages, active, unfinished, limit
                    )
                    actual = self.pages._rank_hot_page_ids(
                        hot_assembly_ids=active,
                        unfinished_ids=unfinished,
                        limit=limit,
                    )
                    self.assertEqual(actual, expected)

    def test_normal_candidate_query_uses_assembly_index(self):
        with self.pages._connect() as connection:
            plan = connection.execute(
                "EXPLAIN QUERY PLAN SELECT sequence, page_id, assembly_id, "
                "metadata_json, updated_at FROM working_pages "
                "WHERE assembly_id IN (?, ?)",
                ("focus-0001", "focus-0010"),
            ).fetchall()
        self.assertTrue(
            any("USING INDEX working_pages_assembly" in str(row[-1]) for row in plan),
            plan,
        )

    def test_large_candidate_set_respects_legacy_sqlite_bind_limit(self):
        active = {f"focus-{index:04d}" for index in range(1_200)}
        original_connect = self.pages._connect
        observed_bind_counts = []

        class LegacyLimitConnection:
            def __init__(self, connection):
                self.connection = connection

            def __enter__(self):
                self.connection.__enter__()
                return self

            def __exit__(self, *arguments):
                return self.connection.__exit__(*arguments)

            def execute(self, statement, parameters=()):
                if "WHERE assembly_id IN (" in statement:
                    observed_bind_counts.append(len(parameters))
                    if len(parameters) > 999:
                        raise AssertionError("legacy SQLite bind limit exceeded")
                return self.connection.execute(statement, parameters)

        def limited_connect():
            return LegacyLimitConnection(original_connect())

        with mock.patch.object(self.pages, "_connect", side_effect=limited_connect):
            selected = self.pages._rank_hot_page_ids(
                hot_assembly_ids=active, limit=16
            )
        self.assertEqual(len(selected), 16)
        self.assertEqual(sum(observed_bind_counts), 1_200)
        self.assertLessEqual(max(observed_bind_counts), 999)


if __name__ == "__main__":
    unittest.main()
