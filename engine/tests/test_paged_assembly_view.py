"""Storage-only checks for the explicit paged assembly view."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from omni_core.paged_assembly_index import (
    AssemblyIndexResourcePause,
    PagedAssemblyIndex,
)
from omni_core.paged_assembly_view import PagedAssemblyView


def _record(number):
    return {
        "id": "assembly-%04d" % number,
        "fingerprint": "fingerprint-%04d" % number,
        "neuron_ids": ["neuron-%04d" % number],
        "source_provenance": {"sensor": 1},
        "rehearsals": 1,
        "importance": 0.25,
    }


class PagedAssemblyViewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-assembly-view-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "assembly.sqlite3"
        self.index = PagedAssemblyIndex(self.path)
        self.view = PagedAssemblyView(self.index, page_size=3)

    def test_ordered_iteration_position_range_and_exact_lookup_are_read_only(self):
        for number in range(11):
            self.view.append(_record(number))
        self.assertEqual(len(self.view), 11)
        self.assertEqual([item["id"] for item in self.view], [
            "assembly-%04d" % number for number in range(11)
        ])
        self.assertEqual(self.view[0]["id"], "assembly-0000")
        self.assertEqual(self.view[-1]["id"], "assembly-0010")
        self.assertEqual(
            [item["id"] for item in self.view.iter_range(4, 9)],
            ["assembly-%04d" % number for number in range(4, 9)],
        )
        self.assertEqual(
            self.view.get_by_id("assembly-0002")["fingerprint"],
            "fingerprint-0002",
        )
        self.assertEqual(
            self.view.get_by_fingerprint("fingerprint-0002")["id"],
            "assembly-0002",
        )
        self.assertIsNone(self.view.get_by_id("absent"))
        self.assertIsNone(self.view.get_by_fingerprint("absent"))
        with self.assertRaises(TypeError):
            self.view[1]["rehearsals"] = 9
        with self.assertRaises(TypeError):
            self.view[1]["source_provenance"]["sensor"] = 9
        with self.assertRaises(AttributeError):
            self.view[1]["neuron_ids"].append("another")
        with self.assertRaisesRegex(TypeError, "slices"):
            self.view[2:]
        with self.assertRaises(IndexError):
            self.view[11]
        self.assertEqual(self.index.get_by_id("assembly-0001")["rehearsals"], 1)

    def test_append_requires_new_identity_and_preserves_existing_metadata(self):
        original = _record(1)
        self.view.append(original)
        duplicate = {**original, "rehearsals": 99}
        with self.assertRaisesRegex(ValueError, "new id and fingerprint"):
            self.view.append(duplicate)
        with self.assertRaisesRegex(ValueError, "new id and fingerprint"):
            self.view.append({**_record(2), "fingerprint": original["fingerprint"]})
        self.assertEqual(self.index.get_by_id(original["id"]), original)
        self.assertEqual(len(self.view), 1)

    def test_append_can_admit_the_single_shared_packed_vector_row(self):
        packed_index = PagedAssemblyIndex(self.path, dimensions=5)
        packed_view = PagedAssemblyView(packed_index)
        packed_view.append(_record(1), packed_vector=b"\x55\x55")
        self.assertEqual(packed_index.get_packed_vector("assembly-0001"), b"\x55\x55")
        self.assertEqual(packed_index.status()["packedVectorRows"], 1)

    def test_edit_transaction_stages_nested_mutation_and_commits_atomically(self):
        self.view.append(_record(1))
        self.view.append(_record(2))
        escaped = []

        def revise(record):
            escaped.append(record)
            record["rehearsals"] += 1
            record["source_provenance"]["sensor"] += 1

        with self.view.transaction(max_rows=2) as edits:
            first = edits.edit_by_id("assembly-0001", revise)
            self.assertEqual(first["rehearsals"], 2)
            self.assertEqual(edits.pending_count, 1)
            self.assertEqual(edits.get_by_fingerprint("fingerprint-0001")[
                "source_provenance"
            ]["sensor"], 2)
            self.assertEqual(self.view.get_by_id("assembly-0001")["rehearsals"], 1)
            edits.edit_by_id("assembly-0002", lambda record: record.update(
                rehearsals=3
            ))
            escaped[0]["rehearsals"] = 99
        reopened = PagedAssemblyView(PagedAssemblyIndex(self.path))
        self.assertEqual(reopened.get_by_id("assembly-0001")["rehearsals"], 2)
        self.assertEqual(
            reopened.get_by_id("assembly-0001")["source_provenance"]["sensor"], 2
        )
        self.assertEqual(reopened.get_by_id("assembly-0002")["rehearsals"], 3)
        with self.assertRaisesRegex(RuntimeError, "not active"):
            edits.get_by_id("assembly-0001")

    def test_bounded_overlay_never_evicts_or_partially_commits(self):
        for number in range(3):
            self.view.append(_record(number))
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            with self.view.transaction(max_rows=1) as edits:
                edits.edit_by_id("assembly-0000", lambda record: record.update(
                    rehearsals=2
                ))
                with self.assertRaises(AssemblyIndexResourcePause):
                    edits.edit_by_id("assembly-0001", lambda record: record.update(
                        rehearsals=2
                    ))
        for number in range(3):
            self.assertEqual(self.view[number]["rehearsals"], 1)
        with self.assertRaises(AssemblyIndexResourcePause):
            with self.view.transaction(max_rows=1, max_payload_bytes=128) as edits:
                edits.edit_by_id("assembly-0000", lambda record: record.update(
                    rehearsals=2
                ))
        self.assertEqual(self.view[0]["rehearsals"], 1)

    def test_successive_small_transactions_have_no_total_record_cap(self):
        for number in range(30):
            self.view.append(_record(number))
        for number in range(30):
            with self.view.transaction(max_rows=1) as edits:
                edits.edit_by_id(
                    "assembly-%04d" % number,
                    lambda record: record.update(rehearsals=2),
                )
        self.assertEqual(len(self.view), 30)
        self.assertTrue(all(item["rehearsals"] == 2 for item in self.view))

    def test_identity_privacy_and_reserve_failures_roll_back(self):
        self.view.append(_record(1))
        with self.assertRaisesRegex(ValueError, "cannot change id"):
            with self.view.transaction() as edits:
                edits.edit_by_id("assembly-0001", lambda record: record.update(
                    id="other"
                ))
        with self.assertRaisesRegex(ValueError, "non-structural"):
            with self.view.transaction() as edits:
                edits.edit_by_id("assembly-0001", lambda record: record.update(
                    source_text="raw private passage"
                ))
        with self.assertRaisesRegex(ValueError, "unsafe"):
            with self.view.transaction() as edits:
                edits.edit_by_id("assembly-0001", lambda record: record.update(
                    source="token=private"
                ))
        self.assertEqual(self.view[0]["rehearsals"], 1)

        def deny_overlay(_estimated, operation):
            return "dirty overlay" not in operation

        guarded = PagedAssemblyView(PagedAssemblyIndex(
            self.path, memory_reserve=deny_overlay
        ))
        with self.assertRaises(AssemblyIndexResourcePause):
            with guarded.transaction() as edits:
                edits.edit_by_id("assembly-0001", lambda record: record.update(
                    rehearsals=2
                ))
        self.assertEqual(self.view[0]["rehearsals"], 1)

    def test_caught_read_reserve_failure_makes_transaction_abort_only(self):
        self.view.append(_record(1))

        def deny_record_read(_estimated, operation):
            return operation != "assembly edit record read"

        guarded = PagedAssemblyView(PagedAssemblyIndex(
            self.path, memory_reserve=deny_record_read
        ))
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            with guarded.transaction() as edits:
                with self.assertRaises(AssemblyIndexResourcePause):
                    edits.get_by_id("assembly-0001")
        self.assertEqual(self.view[0]["rehearsals"], 1)

    def test_caught_invalid_edit_does_not_commit_earlier_stage(self):
        self.view.append(_record(1))
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            with self.view.transaction() as edits:
                edits.edit_by_id("assembly-0001", lambda record: record.update(
                    rehearsals=2
                ))
                with self.assertRaises(TypeError):
                    edits.edit_by_id("assembly-0001", None)
        self.assertEqual(self.view[0]["rehearsals"], 1)

    def test_commit_reserve_failure_rolls_back_all_staged_rows(self):
        self.view.append(_record(1))
        self.view.append(_record(2))
        upserts = 0

        def deny_second_upsert(_estimated, operation):
            nonlocal upserts
            if operation == "assembly index upsert":
                upserts += 1
                return upserts < 2
            return True

        guarded = PagedAssemblyView(PagedAssemblyIndex(
            self.path, disk_reserve=deny_second_upsert
        ))
        with self.assertRaises(AssemblyIndexResourcePause):
            with guarded.transaction(max_rows=2) as edits:
                edits.edit_by_id("assembly-0001", lambda record: record.update(
                    rehearsals=2
                ))
                edits.edit_by_id("assembly-0002", lambda record: record.update(
                    rehearsals=2
                ))
        self.assertEqual(upserts, 2)
        self.assertEqual(self.view[0]["rehearsals"], 1)
        self.assertEqual(self.view[1]["rehearsals"], 1)

    def test_sequence_gap_fails_closed_instead_of_shifting_positions(self):
        for number in range(3):
            self.view.append(_record(number))
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "DELETE FROM assembly_records WHERE assembly_id=?", ("assembly-0001",)
            )
        with self.assertRaisesRegex(ValueError, "sequence has a gap"):
            len(self.view)
        with self.assertRaisesRegex(ValueError, "sequence has a gap"):
            self.view[1]


if __name__ == "__main__":
    unittest.main()
