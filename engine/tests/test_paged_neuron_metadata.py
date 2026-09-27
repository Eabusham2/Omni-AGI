"""Pure metadata-storage tests; no neural model is constructed."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from omni_core.paged_neuron_metadata import PagedNeuronMetadata


def _neuron(number):
    identifier = "neuron-%04d" % number
    return {
        "id": identifier,
        "neuron_id": identifier,
        "label": "example-%d" % number,
        "region": "semantic",
        "activation": 0.0,
        "importance": 0.2,
        "uncertainty": 0.5,
        "exposures": 1,
        "aliases": [],
    }


class PagedNeuronMetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-paged-neurons-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "working.sqlite3"
        self.store = PagedNeuronMetadata(self.path)

    def test_import_pages_lookup_iteration_and_reopen(self):
        for start in range(0, 85, 17):
            page = [_neuron(number) for number in range(start, start + 17)]
            self.assertEqual(self.store.import_page(page, max_rows=17), 17)
        self.assertEqual(len(self.store), 85)
        self.assertEqual(
            list(self.store), ["neuron-%04d" % number for number in range(85)]
        )
        self.assertEqual(
            [key for key, _record in self.store.items()], list(self.store)
        )
        self.assertEqual(
            PagedNeuronMetadata(self.path)["neuron-0084"]["label"], "example-84"
        )

    def test_reads_are_recursively_immutable_and_edits_are_atomic(self):
        self.store[_neuron(1)["id"]] = _neuron(1)
        with self.assertRaises(TypeError):
            self.store["neuron-0001"]["activation"] = 1.0
        before = self.store.status()["revision"]
        self.store.edit_by_id(
            "neuron-0001",
            lambda node: node.update({"activation": 0.75, "exposures": 2}),
        )
        self.assertEqual(self.store["neuron-0001"]["activation"], 0.75)
        self.assertEqual(self.store.status()["revision"], before + 1)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.store.edit_by_id(
                "neuron-0001", lambda node: node.update({"id": "other"})
            )
        self.assertEqual(self.store["neuron-0001"]["exposures"], 2)

    def test_checksum_and_generation_drift_fail_closed(self):
        self.store.import_page([_neuron(number) for number in range(8)])
        pages = self.store.iter_pages(page_size=2)
        self.assertEqual(len(next(pages)), 2)
        self.store.edit_by_id("neuron-0004", lambda node: node.update({"exposures": 3}))
        with self.assertRaisesRegex(ValueError, "generation drift"):
            next(pages)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE paged_neuron_records SET record_json=? WHERE neuron_id=?",
                (b'{"id":"neuron-0000"}', "neuron-0000"),
            )
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.store["neuron-0000"]

    def test_batch_collision_rolls_back_and_rejects_unbounded_input(self):
        self.store.import_page([_neuron(1)])
        with self.assertRaisesRegex(ValueError, "collides"):
            self.store.import_page([_neuron(2), _neuron(1)], max_rows=2)
        self.assertNotIn("neuron-0002", self.store)
        with self.assertRaisesRegex(ValueError, "window"):
            self.store.import_page([_neuron(2), _neuron(3)], max_rows=1)
        self.assertEqual(len(self.store), 1)

    def test_lazy_decay_read_edit_exposure_and_zero_reset_survive_reopen(self):
        self.store.import_page([_neuron(1), _neuron(2)])
        original = self.store["neuron-0001"]
        self.store.decay(0.1)
        self.store.decay(0.2)
        first = self.store["neuron-0001"]
        self.assertAlmostEqual(
            first["activation"], original["activation"] * 0.9 * 0.8
        )
        self.assertAlmostEqual(
            first["uncertainty"], original["uncertainty"] + 0.3 / 2.0
        )
        self.store.edit_by_id(
            "neuron-0001",
            lambda node: node.update({"activation": 0.8, "exposures": 3}),
        )
        self.store.decay(0.1)
        touched = self.store["neuron-0001"]
        cold = self.store["neuron-0002"]
        self.assertAlmostEqual(touched["activation"], 0.72)
        self.assertAlmostEqual(touched["uncertainty"], first["uncertainty"] + 0.1 / 4.0)
        self.assertAlmostEqual(cold["uncertainty"], original["uncertainty"] + 0.4 / 2.0)
        self.assertEqual(dict(PagedNeuronMetadata(self.path)["neuron-0001"]), dict(touched))
        self.store.decay(1.0)
        self.assertEqual(self.store["neuron-0001"]["activation"], 0.0)
        self.assertEqual(self.store["neuron-0002"]["activation"], 0.0)
        self.assertEqual(self.store.status()["decayEpoch"], 4)

    def test_decay_epoch_drift_stops_a_page_scan(self):
        self.store.import_page([_neuron(number) for number in range(6)])
        iterator = self.store.iter_pages(page_size=2)
        next(iterator)
        self.store.decay(0.02)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            next(iterator)

    def test_chat_activity_metrics_are_exact_for_small_store_and_lazy_decay(self):
        rows = [_neuron(number) for number in range(3)]
        rows[0]["activation"] = 0.5
        rows[1]["activation"] = 0.2
        rows[2]["activation"] = 0.0
        self.store.import_page(rows)
        self.store.decay(0.5)
        metrics = self.store.activity_metrics(
            active_ids={"neuron-0000", "neuron-0002"},
            legacy_raw_active=False,
            threshold=0.1,
        )
        self.assertFalse(metrics["estimated"])
        self.assertTrue(metrics["activeCountExact"])
        self.assertEqual(metrics["rowCount"], 3)
        self.assertEqual(metrics["activeCount"], 1)
        self.assertAlmostEqual(metrics["activeFraction"], 1 / 3)
        self.assertAlmostEqual(metrics["meanUncertainty"], 0.75)
        # Deletion leaves a sequence gap; exact small-store traversal must
        # still visit only live rows, not scan the entire high-water range.
        del self.store["neuron-0001"]
        after = self.store.activity_metrics(sample_rows=2)
        self.assertFalse(after["estimated"])
        self.assertEqual(after["sampledRows"], 2)

    def test_large_metrics_are_explicitly_sampled_without_full_mapping_walk(self):
        for start in range(0, 160, 32):
            page = [_neuron(number) for number in range(start, start + 32)]
            for record in page:
                record["uncertainty"] = 0.2 if int(record["id"][-1]) % 2 else 0.8
                record["activation"] = 0.4
            self.store.import_page(page, max_rows=32)
        self.store.decay(0.1)
        metrics = self.store.activity_metrics(sample_rows=16)
        self.assertTrue(metrics["estimated"])
        self.assertFalse(metrics["activeCountExact"])
        self.assertEqual(metrics["rowCount"], 160)
        self.assertLessEqual(metrics["sampledRows"], 16)
        self.assertGreater(metrics["meanUncertainty"], 0)
        overlay = self.store.activity_metrics(
            active_ids={"neuron-0001", "neuron-0002"},
            legacy_raw_active=False,
            sample_rows=16,
        )
        self.assertTrue(overlay["estimated"])  # mean remains sampled
        self.assertTrue(overlay["activeCountExact"])
        self.assertEqual(overlay["activeCount"], 2)


if __name__ == "__main__":
    unittest.main()
