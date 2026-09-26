import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.substrate_inspection import (
    PersistedSubstrateView,
    query_persisted_substrate,
)
from omni_core.vsa import NeuralSubstrate
from worker import Worker


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def neuron(identifier, region="semantic", activation=0.5):
    return {
        "id": identifier,
        "neuron_id": identifier,
        "label": "Neuron " + identifier,
        "region": region,
        "activation": activation,
        "importance": 0.7,
        "uncertainty": 0.2,
        "exposures": 2,
        "created_at": 1_700_000_000.0,
        "last_activated_at": 1_700_000_001.0,
        "aliases": [],
    }


def synapse(identifier, source, target, effective=1):
    return {
        "id": identifier,
        "source_id": source,
        "target_id": target,
        "kind": "associates",
        "effective_weight": effective,
        "eligibility": 0.4,
        "plasticity": 0.8,
        "uses": 3,
        "stability": 0.6,
        "last_updated_at": 1_700_000_002.0,
    }


class PersistedSubstrateInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-inspection-test-")
        self.brain = Path(self.temporary.name) / "brain"
        self.engine = self.brain / "engine"
        self.store = self.engine / "substrate"
        self.engine.mkdir(parents=True)
        self.brain_id = "inspection-brain"

    def tearDown(self):
        self.temporary.cleanup()

    def persist(self, substrate, records_per_shard=2):
        pointer = substrate.save_sharded(
            self.store,
            records_per_shard=records_per_shard,
        )
        metadata = {
            "brain_id": self.brain_id,
            "substrate": {
                "dimensions": substrate.space.dimensions,
                "seed": substrate.space.seed,
                "persistence": pointer,
            },
        }
        (self.engine / "brain.json").write_bytes(canonical(metadata))
        return pointer

    def remove_inline_summaries(self, pointer):
        old_path = self.store / pointer["generationManifest"]
        generation = json.loads(old_path.read_text("utf-8"))
        for shard in generation["shards"]:
            shard.pop("inspection", None)
        body = {key: value for key, value in generation.items() if key != "contentSha256"}
        generation_id = hashlib.sha256(canonical(body)).hexdigest()
        generation["contentSha256"] = generation_id
        payload = canonical(generation)
        relative = "generations/%s/manifest.json" % generation_id
        target = self.store / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        updated = {
            **pointer,
            "activeGeneration": generation_id,
            "contentSha256": generation_id,
            "generationManifest": relative,
            "generationManifestSha256": hashlib.sha256(payload).hexdigest(),
        }
        (self.store / "manifest.json").write_bytes(canonical(updated))
        metadata = json.loads((self.engine / "brain.json").read_text("utf-8"))
        metadata["substrate"]["persistence"] = updated
        (self.engine / "brain.json").write_bytes(canonical(metadata))
        return updated

    def set_attention_overlay(self, epoch, **overrides):
        metadata = json.loads((self.engine / "brain.json").read_text("utf-8"))
        metadata["fresh_attention_boundary"] = {"epoch": int(epoch)}
        metadata["attention_overlay"] = {
            "format": "omni-substrate-attention-overlay",
            "formatVersion": 1,
            "epoch": int(epoch),
            "legacyRawActive": False,
            "activeNeuronIds": [],
            "recalledAssemblyIds": [],
            "eligibleSynapseIds": [],
            **overrides,
        }
        (self.engine / "brain.json").write_bytes(canonical(metadata))

    def test_worker_returns_indexed_clusters_without_loading_adaptive_brain(self):
        substrate = NeuralSubstrate(16, seed=7)
        substrate.neurons = {
            "a": neuron("a", "semantic", 0.9),
            "b": neuron("b", "semantic", 0.3),
            "c": neuron("c", "assembly", 0.05),
        }
        substrate.synapses = {
            "a>b:associates": synapse("a>b:associates", "a", "b", 1),
            "c>a:associates": synapse("c>a:associates", "c", "a", -1),
        }
        pointer = self.persist(substrate)
        worker = Worker()

        with mock.patch.object(worker, "_get", side_effect=AssertionError("must not load")), mock.patch.object(
            PersistedSubstrateView,
            "records",
            side_effect=AssertionError("summarized overview must not read record shards"),
        ), mock.patch.object(
            PersistedSubstrateView,
            "tensors",
            side_effect=AssertionError("summarized overview must not read tensor shards"),
        ):
            page = worker.query_substrate(
                {
                    "brainId": self.brain_id,
                    "storagePath": str(self.brain),
                    "query": {"entity": "overview", "zoom": 0.72, "pageSize": 100},
                },
                "inspection-query",
            )

        self.assertEqual(page["totals"], {"assemblies": 0, "neurons": 3, "synapses": 2})
        self.assertEqual(page["matched"], 5)
        pathways = {
            item["id"]: item for item in page["clusters"] if item["kind"] == "pathway"
        }
        self.assertEqual(pathways["pathway:semantic>semantic"]["effectiveWeights"]["positive"], 1)
        self.assertEqual(pathways["pathway:assembly>semantic"]["effectiveWeights"]["negative"], 1)
        index_path = (
            self.store
            / "inspection"
            / "generations"
            / pointer["activeGeneration"]
            / "manifest.json"
        )
        self.assertTrue(index_path.is_file())
        self.assertEqual(worker.brains, {})

    def test_inspection_worker_exposes_no_mutating_rpc(self):
        with mock.patch.dict(os.environ, {"OMNI_WORKER_ROLE": "inspection"}):
            worker = Worker()
        self.assertEqual(
            set(worker.methods),
            {"health", "query_substrate", "cancel", "shutdown"},
        )
        self.assertFalse(worker.health({}, "health")["capabilities"]["neuralMutation"])

    def test_legacy_generation_backfills_atomically_from_bounded_shards(self):
        substrate = NeuralSubstrate(16, seed=11)
        substrate.neurons = {
            "left": neuron("left", "left", 0.7),
            "right": neuron("right", "right", 0.1),
        }
        substrate.synapses = {
            "left>right:associates": synapse(
                "left>right:associates", "left", "right", 0
            )
        }
        pointer = self.remove_inline_summaries(self.persist(substrate, 1))

        page = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "overview", "zoom": 0.72, "pageSize": 100},
        )

        self.assertEqual(page["matched"], 3)
        pathway = next(item for item in page["clusters"] if item["kind"] == "pathway")
        self.assertEqual(pathway["id"], "pathway:left>right")
        self.assertEqual(pathway["effectiveWeights"], {"negative": 0, "zero": 1, "positive": 0})
        index_path = (
            self.store
            / "inspection"
            / "generations"
            / pointer["activeGeneration"]
            / "manifest.json"
        )
        manifest = json.loads(index_path.read_text("utf-8"))
        body = {key: value for key, value in manifest.items() if key != "contentSha256"}
        self.assertEqual(manifest["contentSha256"], hashlib.sha256(canonical(body)).hexdigest())

        blob = self.store / "inspection" / manifest["shards"][0]["records"]["path"]
        blob.write_bytes(blob.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"):
            query_persisted_substrate(
                self.engine,
                self.brain_id,
                {"entity": "overview", "zoom": 0.72, "pageSize": 100},
            )

    def test_detail_cursor_reads_committed_shards_without_a_cardinality_ceiling(self):
        substrate = NeuralSubstrate(16, seed=17)
        substrate.neurons = {
            "node-%04d" % index: neuron(
                "node-%04d" % index,
                "region-%04d" % index,
                0.8,
            )
            for index in range(520)
        }
        substrate.synapses = {
            "node-%04d>node-%04d:associates" % (index, (index + 1) % 520): synapse(
                "node-%04d>node-%04d:associates" % (index, (index + 1) % 520),
                "node-%04d" % index,
                "node-%04d" % ((index + 1) % 520),
                1 if index % 2 else -1,
            )
            for index in range(520)
        }
        self.persist(substrate, 16)

        overview = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "overview", "zoom": 0.72, "pageSize": 2_000},
        )
        self.assertEqual(overview["matched"], 1_040)
        self.assertEqual(overview["returned"], 1_040)
        self.assertFalse(overview["hasMore"])

        visited = []
        cursor = None
        while True:
            page = query_persisted_substrate(
                self.engine,
                self.brain_id,
                {
                    "entity": "neurons",
                    "zoom": 1,
                    "pageSize": 137,
                    **({"cursor": cursor} if cursor else {}),
                },
            )
            visited.extend(item["id"] for item in page["neurons"])
            if not page["hasMore"]:
                break
            cursor = page["nextCursor"]
        self.assertEqual(len(visited), 520)
        self.assertEqual(len(set(visited)), 520)

    def test_connected_pathway_locator_has_full_zero_weight_parity_and_reads_only_candidates(self):
        substrate = NeuralSubstrate(16, seed=19)
        substrate.neurons = {
            "node-%03d" % index: neuron("node-%03d" % index)
            for index in range(80)
        }
        substrate.synapses = {
            "node-%03d>node-%03d:associates" % (index, (index + 1) % 80): synapse(
                "node-%03d>node-%03d:associates" % (index, (index + 1) % 80),
                "node-%03d" % index,
                "node-%03d" % ((index + 1) % 80),
                0 if index % 3 == 0 else (1 if index % 2 else -1),
            )
            for index in range(80)
        }
        pointer = self.persist(substrate, 2)
        all_synapses = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "synapses", "zoom": 1, "pageSize": 1_000},
        )["synapses"]
        connected_to = "node-037"
        expected = [
            value
            for value in all_synapses
            if connected_to in {value["sourceId"], value["targetId"]}
        ]

        first = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {
                "entity": "synapses",
                "connectedTo": connected_to,
                "zoom": 1,
                "pageSize": 100,
            },
        )
        self.assertEqual(first["synapses"], expected)
        self.assertTrue(
            any(value["effectiveWeight"] == 0 for value in first["synapses"])
        )
        adjacency_path = (
            self.store
            / "inspection"
            / "adjacency"
            / "generations"
            / (pointer["activeGeneration"] + ".json")
        )
        self.assertTrue(adjacency_path.is_file())
        manifest = json.loads(adjacency_path.read_text("utf-8"))
        self.assertEqual(manifest["sourceGeneration"], pointer["activeGeneration"])

        original = PersistedSubstrateView.records
        reads = []

        def counted(view, shard):
            reads.append((shard["bucket"], shard["part"]))
            return original(view, shard)

        with mock.patch.object(PersistedSubstrateView, "records", counted):
            second = query_persisted_substrate(
                self.engine,
                self.brain_id,
                {
                    "entity": "synapses",
                    "connectedTo": connected_to,
                    "zoom": 1,
                    "pageSize": 100,
                },
            )
        total_synapse_shards = sum(
            1
            for value in json.loads(
                (self.store / pointer["generationManifest"]).read_text("utf-8")
            )["shards"]
            if value["kind"] == "synapses"
        )
        self.assertEqual(second["synapses"], expected)
        self.assertLess(len(reads), total_synapse_shards)

    def test_attention_epoch_masks_stale_shard_activity_without_rewriting_it(self):
        substrate = NeuralSubstrate(16, seed=23)
        substrate.neurons = {
            "a": neuron("a", "semantic", 0.9),
            "b": neuron("b", "semantic", 0.3),
        }
        substrate.synapses = {
            "a>b:associates": synapse("a>b:associates", "a", "b", 1),
        }
        pointer = self.persist(substrate)
        pointer_bytes = (self.store / "manifest.json").read_bytes()

        legacy = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "neurons", "zoom": 1, "pageSize": 1},
        )
        stale_cursor = legacy["nextCursor"]
        query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "overview", "zoom": 0.72, "pageSize": 100},
        )
        self.set_attention_overlay(1)

        with mock.patch.object(
            PersistedSubstrateView,
            "records",
            side_effect=AssertionError(
                "an empty attention overlay must use the aggregate index"
            ),
        ):
            overview = query_persisted_substrate(
                self.engine,
                self.brain_id,
                {"entity": "overview", "zoom": 0.72, "pageSize": 100},
            )
        bands = [
            value
            for value in overview["clusters"]
            if value["kind"] == "activation-band"
        ]
        self.assertEqual(len(bands), 1)
        self.assertEqual(bands[0]["id"], "region:semantic:quiet")
        self.assertEqual(bands[0]["count"], 2)
        self.assertEqual(bands[0]["activeCount"], 0)
        self.assertEqual(bands[0]["meanActivation"], 0.0)
        self.assertEqual(bands[0]["maxActivation"], 0.0)
        self.assertEqual(overview["attentionEpoch"], 1)
        self.assertEqual(overview["substrateContentSha256"], pointer["contentSha256"])
        self.assertNotEqual(overview["revision"], pointer["activeGeneration"])
        self.assertEqual((self.store / "manifest.json").read_bytes(), pointer_bytes)

        neurons = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "neurons", "zoom": 1, "pageSize": 10},
        )["neurons"]
        self.assertTrue(all(value["activation"] == 0.0 for value in neurons))
        self.assertTrue(all(value["lastActivatedAt"] is None for value in neurons))
        synapses = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "synapses", "zoom": 1, "pageSize": 10},
        )["synapses"]
        self.assertEqual(synapses[0]["eligibility"], 0.0)
        with self.assertRaisesRegex(ValueError, "stale"):
            query_persisted_substrate(
                self.engine,
                self.brain_id,
                {
                    "entity": "neurons",
                    "zoom": 1,
                    "pageSize": 1,
                    "cursor": stale_cursor,
                },
            )

        self.set_attention_overlay(
            1,
            activeNeuronIds=["a"],
            eligibleSynapseIds=["a>b:associates"],
        )
        active = query_persisted_substrate(
            self.engine,
            self.brain_id,
            {"entity": "neurons", "zoom": 1, "pageSize": 10},
        )["neurons"]
        self.assertEqual(
            {value["id"]: value["activation"] for value in active},
            {"a": 0.9, "b": 0.0},
        )


if __name__ == "__main__":
    unittest.main()
