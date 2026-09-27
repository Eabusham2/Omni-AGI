"""Resource-pressure and complete-coverage checks for the native substrate.

There is no cue-to-answer sequence table in the production brain. These checks
exercise only shared neural state, transactional growth, and whole-source
traversal; they deliberately do not assert exact answer lookup.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.brain import AdaptiveBrain
from omni_core.config import OmniConfig
from omni_core.offload import GIB, ResourcePolicy, ResourceReading
from omni_core.vsa import NeuralSubstrate, SubstrateResourcePause


class ResourceSafeNeuralCompactionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(211)
        torch.set_num_threads(1)

    def test_every_record_updates_one_shared_substrate_without_raw_passages(self):
        memory = NeuralSubstrate(64, seed=23)
        records = 300
        for index in range(records):
            result = memory.learn_statistical(
                "shared routing atom recordtoken%d" % index,
                source="document",
                kind="knowledge",
                source_label="ignored-row-label-%d" % index,
            )
            self.assertTrue(result["statistical_update"])

        self.assertEqual(len(memory.assemblies), 1)
        field = memory.assemblies[0]
        self.assertTrue(field["compressed_field"])
        self.assertEqual(field["statistical_experiences"], records)
        self.assertNotIn("source_text", field)
        self.assertGreater(len(memory.neurons), records)
        self.assertTrue(
            all(
                int(synapse["effective_weight"]) in {-1, 0, 1}
                for synapse in memory.synapses.values()
            )
        )

    def test_resource_pause_precedes_any_partial_substrate_mutation(self):
        memory = NeuralSubstrate(
            64, seed=29, growth_guard=lambda _estimated: False
        )
        with self.assertRaises(SubstrateResourcePause):
            memory.learn_statistical("new unseen semantic atoms")
        self.assertFalse(memory.neurons)
        self.assertFalse(memory.assemblies)
        self.assertFalse(memory.synapses)

    def test_large_source_keeps_detailed_learning_with_window_admission(self):
        with tempfile.TemporaryDirectory(prefix="omni-compaction-plan-") as folder:
            brain = AdaptiveBrain(
                "resource-plan", Path(folder) / "brain", OmniConfig.micro()
            )
            try:
                reading = ResourceReading(
                    total_memory_bytes=16 * GIB,
                    available_memory_bytes=8 * GIB,
                    process_memory_bytes=1 * GIB,
                    disk_total_bytes=500 * GIB,
                    disk_free_bytes=100 * GIB,
                )
                brain.resource_policy = ResourcePolicy(
                    brain.engine_path, reading_provider=lambda: reading
                )
                small = brain._streaming_neural_storage_plan(1024)
                large = brain._streaming_neural_storage_plan(10 * GIB)
                self.assertTrue(small["detailedRecordAssemblies"])
                self.assertEqual(
                    small["representationDecision"], "detailed-admitted"
                )
                self.assertTrue(large["detailedRecordAssemblies"])
                self.assertEqual(
                    large["representationDecision"], "detailed-admitted"
                )
                self.assertEqual(
                    large["projectedDetailedBytes"], small["projectedDetailedBytes"]
                )
                self.assertEqual(
                    large["projectionScope"], "next-checkpoint-window-not-whole-source"
                )
                self.assertIsNone(large["recordCardinalityLimit"])
                self.assertFalse(large["silentRecordSkipping"])
            finally:
                brain.close()

    def test_transient_pressure_defers_detailed_work_without_downgrade(self):
        with tempfile.TemporaryDirectory(prefix="omni-pressure-wait-") as folder:
            brain = AdaptiveBrain(
                "pressure-wait", Path(folder) / "brain", OmniConfig.micro()
            )
            try:
                pressured = ResourceReading(
                    total_memory_bytes=16 * GIB,
                    available_memory_bytes=64 * 1024 * 1024,
                    process_memory_bytes=1 * GIB,
                    disk_total_bytes=500 * GIB,
                    disk_free_bytes=100 * GIB,
                )
                current_reading = [pressured]
                brain.resource_policy = ResourcePolicy(
                    brain.engine_path,
                    reading_provider=lambda: current_reading[0],
                )
                plan = brain._streaming_neural_storage_plan(128)
                self.assertTrue(plan["detailedRecordAssemblies"])
                self.assertTrue(plan["detailedRepresentationDeferred"])
                self.assertEqual(
                    plan["representationDecision"], "detailed-awaiting-resources"
                )
                self.assertFalse(plan["representationDowngradedForTransientPressure"])
                self.assertFalse(plan["silentRecordSkipping"])

                current_reading[0] = ResourceReading(
                    total_memory_bytes=16 * GIB,
                    available_memory_bytes=8 * GIB,
                    process_memory_bytes=1 * GIB,
                    disk_total_bytes=500 * GIB,
                    disk_free_bytes=100 * GIB,
                )
                admitted = brain._streaming_neural_storage_plan(128)
                self.assertTrue(admitted["detailedRecordAssemblies"])
                self.assertFalse(admitted["detailedRepresentationDeferred"])
                self.assertEqual(admitted["representationDecision"], "detailed-admitted")
            finally:
                brain.close()

    def test_detailed_ingest_visits_every_typed_row_without_answer_table(self):
        with tempfile.TemporaryDirectory(prefix="omni-compact-ingest-") as folder:
            root = Path(folder)
            brain = AdaptiveBrain("compact-ingest", root / "brain", OmniConfig.micro())
            try:
                dataset = root / "dialogues.jsonl"
                rows = 12
                dataset.write_text(
                    "".join(
                        json.dumps(
                            {
                                "messages": [
                                    {"role": "user", "content": "cue number %d" % index},
                                    {
                                        "role": "assistant",
                                        "content": "response symbol %d" % (index + 100),
                                    },
                                ]
                            }
                        )
                        + "\n"
                        for index in range(rows)
                    ),
                    encoding="utf-8",
                )
                result = brain.ingest(path=str(dataset), policy="encode")
                coverage = result["coverage"]
                self.assertEqual(coverage["processedRecords"], rows)
                self.assertEqual(coverage["rejectedRecords"], 0)
                self.assertEqual(coverage["discoveredRecords"], rows)
                stream = result["streamingGradientTraining"]
                self.assertGreaterEqual(stream["records"], rows)
                self.assertTrue(stream["everyQueuedRecordContributed"])
                self.assertTrue(
                    all(
                        int(synapse["effective_weight"]) in {-1, 0, 1}
                        for synapse in brain.memory.synapses.values()
                    )
                )
                self.assertNotIn("neuralSequenceTraining", result)
            finally:
                brain.close()


if __name__ == "__main__":
    unittest.main()
