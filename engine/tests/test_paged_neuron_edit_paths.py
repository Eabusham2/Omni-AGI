"""Small storage-only checks for explicit neuron edits; no model training."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from omni_core.memory_lifecycle import OrganicMemoryLifecycle
from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.vsa import NeuralSubstrate


class PagedNeuronEditPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-paged-node-edit-")
        self.addCleanup(temporary.cleanup)
        self.memory = NeuralSubstrate(dimensions=16, seed=37)
        self.memory.neurons = PagedNeuronMetadata(
            Path(temporary.name) / "neurons.sqlite3"
        )

    def test_existing_neuron_reexposure_commits_one_transaction(self):
        identifier = self.memory._ensure_neuron("copper", 1.0)
        self.assertEqual(self.memory.neurons[identifier]["exposures"], 1)
        before = self.memory.neurons.status()["revision"]
        same = self.memory._ensure_neuron("copper", 2.0)
        self.assertEqual(same, identifier)
        self.assertEqual(self.memory.neurons[identifier]["exposures"], 2)
        self.assertEqual(self.memory.neurons[identifier]["last_activated_at"], 2.0)
        self.assertEqual(self.memory.neurons.status()["revision"], before + 1)

    def test_lifecycle_scores_use_explicit_paged_edit(self):
        identifier = self.memory._ensure_neuron("harbor", 1.0)
        lifecycle = OrganicMemoryLifecycle()
        lifecycle._write_neuron_scores(
            self.memory,
            identifier,
            scores={
                "retention": 0.7, "activity": 0.6,
                "plasticity": 0.5, "reinforcement": 0.4,
                "unfinished": 0.3,
            },
            signals={"salience": 0.8},
            timestamp="2026-09-26T00:00:00Z",
        )
        record = self.memory.neurons[identifier]
        self.assertEqual(record["retention_score"], 0.7)
        self.assertEqual(record["settling_signals"]["salience"], 0.8)
        self.assertEqual(record["last_settled_at"], "2026-09-26T00:00:00Z")
        with self.assertRaises(TypeError):
            record["settling_signals"]["salience"] = 0.2

    def test_paged_decay_advances_one_epoch_without_rewriting_rows(self):
        identifier = self.memory._ensure_neuron("violet", 1.0)
        before = dict(self.memory.neurons[identifier])
        with sqlite3.connect(self.memory.neurons.path) as connection:
            raw_before = connection.execute(
                "SELECT record_json,record_sha256 FROM paged_neuron_records "
                "WHERE neuron_id=?", (identifier,),
            ).fetchone()
        self.memory.decay(0.01)
        with sqlite3.connect(self.memory.neurons.path) as connection:
            raw_after = connection.execute(
                "SELECT record_json,record_sha256 FROM paged_neuron_records "
                "WHERE neuron_id=?", (identifier,),
            ).fetchone()
        self.assertEqual(raw_after, raw_before)
        after = self.memory.neurons[identifier]
        self.assertAlmostEqual(after["activation"], before["activation"] * 0.99)
        self.assertAlmostEqual(
            after["uncertainty"],
            min(1.0, before["uncertainty"] + 0.01 / (1 + before["exposures"])),
        )
        self.assertEqual(self.memory.neurons.status()["decayEpoch"], 1)
        reopened = PagedNeuronMetadata(self.memory.neurons.path)
        self.assertEqual(dict(reopened[identifier]), dict(after))


if __name__ == "__main__":
    unittest.main()
