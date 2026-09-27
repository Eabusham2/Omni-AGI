"""Source contracts for optional bounded VSA assembly metadata paging.

The paged writer is deliberately fail-closed until v3 shard checkpointing is
bounded; these checks do not claim a durable paged brain or learned quality.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from omni_core.paged_assembly_index import PagedAssemblyIndex
from omni_core.paged_assembly_view import PagedAssemblyView
from omni_core.packed_vsa_vectors import PackedTernaryVectorView
from omni_core.vsa import NeuralSubstrate


class PagedVSAMetadataAdapterTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="omni-paged-vsa-adapter-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _index(self):
        return PagedAssemblyIndex(self.root / "assemblies.sqlite3")

    def test_bounded_migration_exact_lookup_and_paged_read_only_iteration(self):
        substrate = NeuralSubstrate(dimensions=16, seed=13)
        for index in range(7):
            identifier = "assembly-%04d" % index
            substrate.assemblies.append({
                "id": identifier,
                "assembly_neuron_id": identifier,
                "fingerprint": "fingerprint-%04d" % index,
                "neuron_ids": [],
                "kind": "knowledge",
                "source": "fixture",
                "importance": 0.5,
                "rehearsals": 1,
            })
            substrate.assembly_vectors[identifier] = torch.ones(16)
            substrate.neurons[identifier] = {"id": identifier, "region": "assembly"}
        index = self._index()
        result = substrate.enable_paged_assemblies(index, page_size=2)
        self.assertEqual(result, {"assemblyCount": 7, "alreadyPaged": False})
        self.assertIsInstance(substrate.assemblies, PagedAssemblyView)
        self.assertEqual(len(substrate.assembly_by_id), 7)
        self.assertEqual(
            substrate.assembly_by_id["assembly-0003"]["fingerprint"],
            "fingerprint-0003",
        )
        self.assertEqual(
            substrate.assembly_by_fingerprint["fingerprint-0005"]["id"],
            "assembly-0005",
        )
        self.assertEqual(
            [record["id"] for record in substrate.assemblies],
            ["assembly-%04d" % index for index in range(7)],
        )
        self.assertEqual(substrate.enable_paged_assemblies(index)["alreadyPaged"], True)
        self.assertEqual(len(substrate._assembly_by_id), 0)
        self.assertEqual(len(substrate._assembly_by_fingerprint), 0)
        with self.assertRaises(TypeError):
            substrate.assemblies[1:]
        with self.assertRaises(TypeError):
            substrate.assembly_by_id["assembly-0000"]["rehearsals"] = 2

    def test_migration_uses_bounded_sqlite_batch_windows(self):
        substrate = NeuralSubstrate(dimensions=16, seed=15)
        for number in range(300):
            identifier = "assembly-%04d" % number
            substrate.assemblies.append({
                "id": identifier,
                "fingerprint": "fingerprint-%04d" % number,
                "neuron_ids": [],
            })
            substrate.assembly_vectors[identifier] = torch.ones(16)
            substrate.neurons[identifier] = {"id": identifier}
        index = self._index()
        with mock.patch.object(index, "batch", wraps=index.batch) as batches:
            substrate.enable_paged_assemblies(index, page_size=31)
        self.assertEqual(batches.call_count, 2)
        self.assertTrue(all(
            call.kwargs == {"max_rows": 256, "max_payload_bytes": 8 * 1024 * 1024}
            for call in batches.call_args_list
        ))
        self.assertEqual(len(substrate.assemblies), 300)

    def test_existing_and_statistical_metadata_use_explicit_edit_transactions(self):
        substrate = NeuralSubstrate(dimensions=32, seed=17)
        text = "Copper lanterns crossed the violet harbor."
        learned = substrate.learn(text, retain_source_text=False)
        field = substrate.learn_statistical(
            "solar panels harvest sunlight above rooftops", source="file-a"
        )
        index = self._index()
        substrate.enable_paged_assemblies(index, page_size=2)
        before = index.get_by_id(learned["assembly_id"])["rehearsals"]

        with mock.patch.object(
            substrate.assemblies,
            "transaction",
            wraps=substrate.assemblies.transaction,
        ) as transaction:
            substrate.learn(text, retain_source_text=False)
            self.assertGreater(transaction.call_count, 0)
        self.assertEqual(
            index.get_by_id(learned["assembly_id"])["rehearsals"], before + 1
        )
        substrate.learn_statistical(
            "solar panels harvest sunlight above rooftops", source="file-b"
        )
        updated = index.get_by_id(field["assembly_id"])
        self.assertGreaterEqual(updated["statistical_experiences"], 2)
        self.assertIn("file-b", updated["source_provenance"])

        substrate.record_recall_activity([{
            "assembly_id": learned["assembly_id"], "score": 0.7
        }])
        self.assertIsNotNone(
            index.get_by_id(learned["assembly_id"])["last_recalled_at"]
        )
        self.assertEqual(
            substrate.neuron_vectors[learned["assembly_id"]].shape, (32,)
        )

    def test_paged_mode_rejects_raw_text_and_uses_bounded_sharded_writer(self):
        substrate = NeuralSubstrate(dimensions=16, seed=19)
        substrate.learn("A copper tram reached the violet harbor.")
        substrate.enable_paged_assemblies(self._index())
        count = len(substrate.assemblies)
        with self.assertRaisesRegex(ValueError, "raw source text"):
            substrate.learn("new raw text", retain_source_text=True)
        self.assertEqual(len(substrate.assemblies), count)
        with self.assertRaisesRegex(ValueError, "monolithic metadata"):
            substrate.metadata(include_records=True)
        with self.assertRaisesRegex(ValueError, "monolithic tensor"):
            substrate.tensor_state()
        checkpoint = self.root / "substrate"
        pointer = substrate.save_sharded(checkpoint, records_per_shard=2)
        self.assertEqual(pointer["formatVersion"], 3)
        self.assertTrue((checkpoint / pointer["generationManifest"]).is_file())

    def test_nonempty_or_vector_owning_index_cannot_be_silently_adopted(self):
        substrate = NeuralSubstrate(dimensions=16, seed=23)
        substrate.learn("Copper lanterns crossed the harbor.")
        index = self._index()
        index.upsert({"id": "prior", "fingerprint": "prior"})
        with self.assertRaisesRegex(ValueError, "empty index"):
            substrate.enable_paged_assemblies(index)
        self.assertIsInstance(substrate.assemblies, list)
        vector_index = PagedAssemblyIndex(
            self.root / "vectors.sqlite3", dimensions=16, seed=23
        )
        with self.assertRaisesRegex(ValueError, "must not own vector rows"):
            substrate.enable_paged_assemblies(vector_index)

    def test_shared_cache_attach_checks_committed_binding_and_exact_records(self):
        substrate = NeuralSubstrate(dimensions=16, seed=31)
        index = PagedAssemblyIndex(
            self.root / "shared.sqlite3", dimensions=16, seed=31
        )
        vectors = index._vectors
        assert vectors is not None
        substrate.neuron_vectors = vectors
        substrate.assembly_vectors = PackedTernaryVectorView(vectors)
        for number in range(5):
            identifier = "assembly-%04d" % number
            record = {
                "id": identifier,
                "fingerprint": "fingerprint-%04d" % number,
                "neuron_ids": [],
                "kind": "knowledge",
            }
            substrate.neurons[identifier] = {"id": identifier}
            vectors[identifier] = torch.ones(16)
            substrate.assembly_vectors.link(identifier)
            substrate.assemblies.append(record)
            index.upsert(record)
        generation = "a" * 64
        substrate.persistence_manifest = {"activeGeneration": generation}
        with self.assertRaisesRegex(ValueError, "not bound"):
            substrate.attach_verified_paged_assemblies(index, page_size=2)
        index.bind_committed_generation(generation)
        attached = substrate.attach_verified_paged_assemblies(index, page_size=2)
        self.assertEqual(attached["assemblyCount"], 5)
        self.assertIsInstance(substrate.assemblies, PagedAssemblyView)
        self.assertIs(substrate.assemblies.index._vectors, substrate.neuron_vectors)
        self.assertEqual(
            substrate.assembly_by_id["assembly-0004"]["fingerprint"],
            "fingerprint-0004",
        )


if __name__ == "__main__":
    unittest.main()
