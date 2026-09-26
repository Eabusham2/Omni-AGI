"""Pure substrate checks for adaptive, non-row-indexed statistical fields."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core.vsa import NeuralSubstrate, SubstrateResourcePause


class StatisticalFieldGrowthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_distinct_same_source_forms_fields_and_cross_source_related_reinforces(self):
        memory = NeuralSubstrate(64, seed=313)
        first = memory.learn_statistical(
            "solar panels harvest sunlight above rooftops",
            source="file-one",
            kind="knowledge",
            source_label="not a saved passage",
        )
        distinct = memory.learn_statistical(
            "ocean currents move cold water below ice",
            source="file-one",
            kind="knowledge",
        )
        self.assertNotEqual(first["assembly_id"], distinct["assembly_id"])
        self.assertEqual(len(memory.assemblies), 2)

        related = memory.learn_statistical(
            "solar panels harvest sunlight over houses",
            source="chat-two",
            kind="conversation",
        )
        self.assertEqual(related["assembly_id"], first["assembly_id"])
        self.assertEqual(related["assemblies_created"], 0)
        self.assertFalse(torch.equal(first["vector"], related["vector"]))
        self.assertEqual(len(memory.assemblies), 2)
        self.assertEqual(
            sum(field["statistical_experiences"] for field in memory.assemblies),
            3,
        )
        field = memory.assembly_by_id[first["assembly_id"]]
        self.assertEqual(field["statistical_experiences"], 2)
        self.assertEqual(field["source_provenance"], {"file-one": 1, "chat-two": 1})
        self.assertEqual(
            field["kind_provenance"], {"knowledge": 1, "conversation": 1}
        )
        self.assertNotIn("source_text", field)
        self.assertNotIn("not a saved passage", str(field))
        self.assertNotIn(
            "solar panels harvest sunlight above rooftops",
            json.dumps(memory.metadata(include_records=True)),
        )
        self.assertTrue(all(
            record["effective_weight"] in {-1, 0, 1}
            for record in memory.synapses.values()
        ))

    def test_repeated_related_rows_do_not_form_one_assembly_per_row(self):
        memory = NeuralSubstrate(32, seed=317)
        identifiers = set()
        for index in range(100):
            learned = memory.learn_statistical(
                f"shared routing atom recordtoken{index}",
                source="dataset",
                source_label=f"row-{index}",
            )
            identifiers.add(learned["assembly_id"])
        self.assertEqual(len(identifiers), 1)
        self.assertEqual(len(memory.assemblies), 1)
        field = memory.assemblies[0]
        self.assertEqual(field["statistical_experiences"], 100)
        self.assertEqual(len(field["statistical_anchor_ids"]), 4)
        self.assertEqual(field["source_provenance"], {"dataset": 100})
        self.assertTrue(all("source_text" not in row for row in memory.assemblies))

    def test_reordered_relations_change_neural_pathways_without_new_raw_store(self):
        memory = NeuralSubstrate(32, seed=331)
        first = memory.learn_statistical("ruby fox follows amber bird")
        before_ids = set(memory.synapses)
        before_vector = memory.assembly_vectors[first["assembly_id"]].clone()
        second = memory.learn_statistical("amber bird follows ruby fox")
        self.assertEqual(second["assembly_id"], first["assembly_id"])
        self.assertGreater(len(set(memory.synapses) - before_ids), 0)
        self.assertFalse(torch.equal(
            before_vector, memory.assembly_vectors[first["assembly_id"]]
        ))
        self.assertEqual(memory.assemblies[0]["statistical_experiences"], 2)
        self.assertNotIn("source_text", memory.assemblies[0])

    def test_sharded_reload_rebuilds_routing_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory(prefix="omni-statistical-fields-") as folder:
            store = Path(folder) / "substrate"
            memory = NeuralSubstrate(32, seed=337)
            solar = memory.learn_statistical(
                "solar panels harvest sunlight above rooftops",
                source="file-a",
            )
            memory.learn_statistical(
                "ocean currents move cold water below ice",
                source="file-b",
            )
            memory.save_sharded(store, records_per_shard=3)
            metadata = memory.metadata(include_records=False)
            durable_solar = memory.assembly_vectors[solar["assembly_id"]].clone()
            for lazy in (False, True):
                with self.subTest(lazy_synapses=lazy):
                    restored = NeuralSubstrate.load_sharded(
                        store, metadata, lazy_synapses=lazy
                    )
                    self.assertEqual(len(restored.assemblies), 2)
                    self.assertTrue(torch.equal(
                        restored.assembly_vectors[solar["assembly_id"]],
                        durable_solar,
                    ))
                    learned = restored.learn_statistical(
                        "solar panels harvest sunlight over houses",
                        source="crawl-c",
                    )
                    self.assertEqual(learned["assembly_id"], solar["assembly_id"])
                    self.assertFalse(torch.equal(learned["vector"], solar["vector"]))
                    self.assertEqual(len(restored.assemblies), 2)
                    field = restored.assembly_by_id[solar["assembly_id"]]
                    self.assertEqual(
                        field["source_provenance"], {"file-a": 1, "crawl-c": 1}
                    )
                    self.assertNotIn("source_text", field)

    def test_growth_reserve_pauses_before_new_cluster_mutation(self):
        allowed = True

        def guard(_estimated: int) -> bool:
            return allowed

        memory = NeuralSubstrate(32, seed=347, growth_guard=guard)
        memory.learn_statistical("solar panels harvest sunlight")
        counts = (len(memory.neurons), len(memory.assemblies), len(memory.synapses))
        revision = memory.state_revision
        allowed = False
        with self.assertRaises(SubstrateResourcePause):
            memory.learn_statistical("ocean currents move cold water")
        self.assertEqual(
            counts,
            (len(memory.neurons), len(memory.assemblies), len(memory.synapses)),
        )
        self.assertEqual(memory.state_revision, revision)

    def test_warm_routing_evaluates_postings_not_every_field(self):
        memory = NeuralSubstrate(16, seed=349)
        for index in range(64):
            memory.learn_statistical(
                f"topic{index}a topic{index}b",
                source="one-source",
            )
        self.assertEqual(len(memory.assemblies), 64)
        with mock.patch.object(
            memory.space, "similarity", wraps=memory.space.similarity
        ) as compared:
            memory.learn_statistical("topic32a topic32b", source="new-source")
        self.assertLess(compared.call_count, 8)
        self.assertEqual(len(memory.assemblies), 64)


if __name__ == "__main__":
    unittest.main()
