import unittest
from types import SimpleNamespace
from unittest import mock

from omni_core.brain import AdaptiveBrain
from omni_core.vsa import NeuralSubstrate


def inspection_brain(*, neurons=None, assemblies=None, synapses=None):
    brain = object.__new__(AdaptiveBrain)
    brain.brain_id = "inspection-fixture"
    brain.memory = SimpleNamespace(
        neurons=neurons or {},
        assemblies=assemblies or [],
        synapses=synapses or {},
        growth_events=0,
        state_revision=0,
    )
    return brain


def neuron(identifier: str, *, region: str = "semantic"):
    return {
        "id": identifier,
        "label": "neuron " + identifier,
        "region": region,
        "activation": 0.25,
        "importance": 0.5,
        "uncertainty": 0.4,
        "exposures": 1,
        "aliases": [],
    }


def synapse(identifier: str, source: str, target: str):
    return {
        "id": identifier,
        "source_id": source,
        "target_id": target,
        "kind": "associates",
        "effective_weight": 1,
        "eligibility": 0.2,
        "plasticity": 1.0,
        "stability": 0.3,
        "uses": 1,
    }


class SubstrateQueryTests(unittest.TestCase):
    def test_concept_extraction_has_no_count_truncation_surface(self):
        text = " ".join("concept%d" % index for index in range(120))
        labels = NeuralSubstrate.extract_concepts(text)
        self.assertGreater(len(labels), 48)
        self.assertIn("concept119", labels)
        self.assertIn("concept117::concept118::concept119", labels)

    def test_concept_extraction_preserves_first_occurrence_at_large_scale(self):
        atoms = ["atom%d" % index for index in range(600)]
        text = " ".join(atoms + ["atom3", "atom1", "atom599"])
        labels = NeuralSubstrate.extract_concepts(text)

        self.assertEqual(labels[: len(atoms)], atoms)
        self.assertEqual(len(set(labels)), len(labels))
        self.assertIn("atom597::atom598::atom599", labels)

    def test_one_pass_statistical_learning_creates_live_ternary_pathways(self):
        memory = NeuralSubstrate(64, seed=31)
        labels = ["corpusatom%d" % index for index in range(160)]

        learned = memory.learn_statistical(" ".join(labels))

        effective = {
            int(record["effective_weight"])
            for record in memory.synapses.values()
        }
        self.assertTrue(effective.issubset({-1, 0, 1}))
        self.assertTrue(any(value != 0 for value in effective))
        signal, recalled = memory.recall_vector(
            memory.vector_for_labels(labels), workspace_slots=32
        )
        self.assertEqual(signal.shape, (64,))
        self.assertIn(
            learned["assembly_id"],
            {item["assembly_id"] for item in recalled},
        )
        self.assertGreater(memory._last_recall_audit["eligibleEdges"], 0)
        self.assertGreater(memory._last_recall_audit["activeNeuralNodes"], 1)
        self.assertTrue(memory._last_recall_audit["exactTernaryContribution"])

    def test_overlapping_experiences_organically_form_inhibitory_competition(self):
        memory = NeuralSubstrate(64, seed=47)
        first = memory.learn("shared anchor memory alpha")
        second = memory.learn("shared anchor memory beta")

        competitive = [
            record
            for record in memory.synapses.values()
            if record.get("kind") == "competes"
        ]
        self.assertTrue(competitive)
        self.assertTrue(
            all(int(record["effective_weight"]) == -1 for record in competitive)
        )
        cue = memory.space.weighted_bundle(
            [first["vector"], second["vector"]], [1.0, 1.0]
        )
        memory.recall_vector(cue, workspace_slots=32)
        self.assertGreater(memory._last_recall_audit["inhibitoryEdges"], 0)
        self.assertGreater(memory._last_recall_audit["inhibitorySignals"], 0)

    def test_live_mutation_invalidates_an_existing_cursor(self):
        memory = NeuralSubstrate(64, seed=41)
        memory.learn("cursor revision follows activation and synaptic state")
        brain = inspection_brain()
        brain.memory = memory
        brain.updated_at = "stable-fixture-time"

        first = brain.query_substrate(
            {"entity": "neurons", "zoom": 1, "pageSize": 1}
        )
        self.assertTrue(first["hasMore"])
        revision = first["revision"]

        memory.decay(0.01)
        with self.assertRaisesRegex(ValueError, "stale"):
            brain.query_substrate(
                {
                    "entity": "neurons",
                    "zoom": 1,
                    "pageSize": 1,
                    "cursor": first["nextCursor"],
                }
            )
        fresh = brain.query_substrate(
            {"entity": "neurons", "zoom": 1, "pageSize": 1}
        )
        self.assertNotEqual(fresh["revision"], revision)

    def test_inspection_rejects_a_fractional_live_synapse(self):
        brain = inspection_brain(
            neurons={"a": neuron("a"), "b": neuron("b")},
            synapses={"a>b": {**synapse("a>b", "a", "b"), "effective_weight": 0.5}},
        )
        with self.assertRaisesRegex(ValueError, "exact ternary"):
            brain.query_substrate(
                {"entity": "synapses", "zoom": 1, "pageSize": 1}
            )

    def test_page_request_traverses_beyond_former_five_thousand_limit(self):
        neurons = {
            "neuron-%05d" % index: neuron("neuron-%05d" % index)
            for index in range(5_007)
        }
        brain = inspection_brain(neurons=neurons)

        first = brain.query_substrate(
            {"entity": "neurons", "zoom": 1, "pageSize": 5_001}
        )
        self.assertEqual(first["returned"], 5_001)
        self.assertEqual(first["matched"], 5_007)
        self.assertTrue(first["hasMore"])
        self.assertFalse(first["transportLimited"])

        second = brain.query_substrate(
            {
                "entity": "neurons",
                "zoom": 1,
                "pageSize": 5_001,
                "cursor": first["nextCursor"],
            }
        )
        identifiers = [item["id"] for item in first["neurons"] + second["neurons"]]
        self.assertEqual(len(identifiers), 5_007)
        self.assertEqual(len(set(identifiers)), 5_007)
        self.assertFalse(second["hasMore"])

    def test_connected_synapse_cursor_exposes_every_matching_pathway(self):
        neurons = {
            "focus": neuron("focus", region="assembly"),
            **{
                "peer-%02d" % index: neuron("peer-%02d" % index)
                for index in range(23)
            },
        }
        synapses = {
            "connected-%02d" % index: synapse(
                "connected-%02d" % index,
                "focus" if index % 2 else "peer-%02d" % index,
                "peer-%02d" % index if index % 2 else "focus",
            )
            for index in range(17)
        }
        synapses.update(
            {
                "unrelated-%02d" % index: synapse(
                    "unrelated-%02d" % index,
                    "peer-%02d" % index,
                    "peer-%02d" % (index + 1),
                )
                for index in range(6)
            }
        )
        brain = inspection_brain(neurons=neurons, synapses=synapses)

        visited = []
        cursor = None
        while True:
            page = brain.query_substrate(
                {
                    "entity": "synapses",
                    "zoom": 1,
                    "connectedTo": "focus",
                    "pageSize": 4,
                    **({"cursor": cursor} if cursor else {}),
                }
            )
            self.assertEqual(page["matched"], 17)
            visited.extend(page["synapses"])
            if not page["hasMore"]:
                break
            cursor = page["nextCursor"]

        self.assertEqual(len(visited), 17)
        self.assertEqual(len({item["id"] for item in visited}), 17)
        self.assertTrue(
            all(
                "focus" in {item["sourceId"], item["targetId"]}
                for item in visited
            )
        )

    def test_transport_bytes_create_continuation_without_count_ceiling(self):
        neurons = {
            "neuron-%03d" % index: neuron("neuron-%03d" % index)
            for index in range(31)
        }
        brain = inspection_brain(neurons=neurons)
        visited = []
        cursor = None
        transport_limited = False
        with mock.patch("omni_core.brain.SUBSTRATE_INSPECTION_TRANSPORT_BYTES", 900):
            while True:
                page = brain.query_substrate(
                    {
                        "entity": "neurons",
                        "zoom": 1,
                        "pageSize": 1_000_000,
                        **({"cursor": cursor} if cursor else {}),
                    }
                )
                self.assertLessEqual(page["pageBytes"], 900)
                self.assertGreater(page["returned"], 0)
                visited.extend(item["id"] for item in page["neurons"])
                transport_limited = transport_limited or page["transportLimited"]
                if not page["hasMore"]:
                    break
                cursor = page["nextCursor"]

        self.assertTrue(transport_limited)
        self.assertEqual(len(visited), 31)
        self.assertEqual(len(set(visited)), 31)

    def test_large_assembly_relationships_move_to_synapse_cursor(self):
        members = ["member-%04d" % index for index in range(240)]
        children = ["child-%04d" % index for index in range(60)]
        assembly_id = "large-assembly"
        neurons = {
            assembly_id: neuron(assembly_id, region="assembly"),
            **{member: neuron(member) for member in members},
            **{child: neuron(child, region="assembly") for child in children},
        }
        assembly = {
            "id": assembly_id,
            "fingerprint": "f" * 64,
            "neuron_ids": members,
            "child_assembly_ids": children,
            "kind": "knowledge",
            "source": "fixture",
            "confidence": 0.5,
            "importance": 0.7,
            "rehearsals": 1,
            "source_label": "large fixture",
        }
        synapses = {
            "contains-%04d" % index: synapse(
                "contains-%04d" % index, assembly_id, member
            )
            for index, member in enumerate(members)
        }
        synapses.update(
            {
                "composes-%04d" % index: {
                    **synapse("composes-%04d" % index, assembly_id, child),
                    "kind": "composes",
                }
                for index, child in enumerate(children)
            }
        )
        brain = inspection_brain(
            neurons=neurons,
            assemblies=[assembly],
            synapses=synapses,
        )

        with mock.patch("omni_core.brain.SUBSTRATE_INSPECTION_TRANSPORT_BYTES", 900):
            page = brain.query_substrate(
                {"entity": "assemblies", "zoom": 1, "pageSize": 1}
            )
        inspected = page["assemblies"][0]
        self.assertTrue(inspected["relationshipsPaged"])
        self.assertEqual(inspected["neuronCount"], len(members))
        self.assertEqual(inspected["childAssemblyCount"], len(children))
        self.assertEqual(inspected["neuronIds"], [])
        self.assertEqual(inspected["childAssemblyIds"], [])

        cursor_page = brain.query_substrate(
            {
                "entity": "synapses",
                "zoom": 1,
                "connectedTo": assembly_id,
                "pageSize": len(members) + len(children),
            }
        )
        relationship_count = len(members) + len(children)
        self.assertEqual(cursor_page["matched"], relationship_count)
        self.assertEqual(len(cursor_page["synapses"]), relationship_count)


if __name__ == "__main__":
    unittest.main()
