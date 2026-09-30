"""Storage/protocol/math stubs only; no brain/model constructors or runs."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from omni_core.concept_id_views import publish_id_view, IdView, validate_descriptor, unique_ids
from omni_core.paged_recurrent_spreading import PagedRecurrentState
from omni_core.recall_views import RecalledIdSequence, id_page


class ConceptIdViewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="omni-id-view-storage-")
        self.addCleanup(directory.cleanup)
        self.engine = Path(directory.name)
        self.state = PagedRecurrentState(self.engine)
        self.addCleanup(self.state.close)
        for number in range(5000):
            self.state.append_recalled({"idea_id": "assembly-%04d" % number, "assembly_id": "assembly-%04d" % number,
                                        "score": 0.3, "neuron_ids": []})
        self.recalled = self.state.recalled()

    def test_lazy_prefix_uniqueness_and_byte_page_never_truncate_neural_ids(self):
        ids = RecalledIdSequence(self.recalled, prefix="assembly-0002")
        self.assertEqual(len(ids), 5000)
        self.assertEqual(ids[:4], ["assembly-0002", "assembly-0000", "assembly-0001", "assembly-0003"])
        page, coverage = id_page(ids, byte_budget=4096)
        self.assertLess(len(page), len(ids))
        self.assertEqual(coverage["totalCount"], 5000)
        self.assertEqual(coverage["nextOffset"], len(page))
        self.assertEqual(coverage["neuralReadoutCoverage"], "all-eligible-assemblies")

    def test_full_argument_view_roundtrips_all_ids_with_stable_content_fingerprint(self):
        ids = RecalledIdSequence(self.recalled, prefix="experience-new")
        first = publish_id_view(self.engine, ids, brain_id="fixture", turn_id="turn-1")
        second = publish_id_view(self.engine, ids, brain_id="fixture", turn_id="turn-1")
        self.assertEqual(first, second)
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("whole ID file read")):
            view = IdView(self.engine, first, brain_id="fixture", turn_id="turn-1")
            actual = sum(1 for _identifier in view)
        self.assertEqual(actual, 5001)
        self.assertEqual(view[0], "experience-new")

    def test_wrong_owner_path_tamper_and_symlink_are_rejected(self):
        descriptor = publish_id_view(self.engine, RecalledIdSequence(self.recalled), brain_id="fixture", turn_id="turn-1")
        for changes in ({"brainId": "another"}, {"turnId": "another"}, {"path": "../secret"}, {"count": True}, {"source_text": "not-structural"}):
            with self.assertRaises(ValueError):
                validate_descriptor({**descriptor, **changes}, brain_id="fixture", turn_id="turn-1")
        path = self.engine / descriptor["path"]
        with path.open("r+b") as handle:
            handle.seek(-3, 2)
            handle.write(b"z")
        with self.assertRaisesRegex(ValueError, "checksum"):
            IdView(self.engine, descriptor, brain_id="fixture", turn_id="turn-1")

    def test_unique_stream_keeps_first_occurrence_and_all_large_members(self):
        values = ("assembly-%04d" % (number % 5000) for number in range(20000))
        unique = unique_ids(values, self.engine / "scratch")
        self.assertEqual(sum(1 for _value in unique), 5000)
        self.assertEqual(list(unique_ids(["b", "a", "b", "c", "a"], self.engine / "scratch")), ["b", "a", "c"])

    def test_reserve_refusal_does_not_publish_partial_argument_view(self):
        with self.assertRaises(RuntimeError):
            publish_id_view(self.engine, RecalledIdSequence(self.recalled), brain_id="fixture", turn_id="turn-1", reserve=lambda *_args: False)
        self.assertFalse((self.engine / "state" / "concept-id-views").exists())

    def test_copied_historical_header_remains_inspectable_but_never_authorizes_new_owner(self):
        import shutil
        descriptor = publish_id_view(self.engine, RecalledIdSequence(self.recalled), brain_id="original", turn_id="original-turn")
        copied = self.engine / "forked-engine"
        target = copied / descriptor["path"]
        target.parent.mkdir(parents=True)
        shutil.copy2(self.engine / descriptor["path"], target)
        history = IdView(copied, descriptor, brain_id="original", turn_id="original-turn")
        page, coverage = id_page(history, byte_budget=4096)
        self.assertTrue(page)
        self.assertEqual(coverage["totalCount"], 5000)
        self.assertEqual(history.descriptor["brainId"], "original")
        with self.assertRaisesRegex(ValueError, "ownership"):
            IdView(copied, descriptor, brain_id="fork", turn_id="original-turn")

    def test_load_free_historical_query_returns_original_provenance_and_never_execution_permission(self):
        import json
        import shutil
        import worker as worker_module
        descriptor = publish_id_view(self.engine, RecalledIdSequence(self.recalled), brain_id="original", turn_id="original-turn")
        storage = self.engine / "copied-brain"
        engine = storage / "engine"
        target = engine / descriptor["path"]
        target.parent.mkdir(parents=True)
        shutil.copy2(self.engine / descriptor["path"], target)
        (engine / "brain.json").write_text(json.dumps({"brain_id": "fork"}), encoding="utf-8")
        holder = object.__new__(worker_module.Worker)
        params = {"brainId": "fork", "storagePath": str(storage), "conceptIdView": descriptor,
                  "sourceTurnId": "original-turn", "historicalInspection": True}
        result = holder.query_concept_id_view(params, None)
        self.assertEqual(result["sourceBrainId"], "original")
        self.assertTrue(result["historicalInspection"])
        self.assertFalse(result["executionAuthorized"])
        self.assertEqual(result["totalCount"], 5000)
        with self.assertRaises(worker_module.RpcFault):
            holder.query_concept_id_view({**params, "historicalInspection": False}, None)


if __name__ == "__main__":
    unittest.main()
