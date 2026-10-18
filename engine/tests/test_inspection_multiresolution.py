"""Actual tiny JSON/SQLite index paths; no tensor/model/brain construction."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from omni_core.authenticated_paged_cache import canonical
from omni_core.inspection_resources import inspection_resources
from omni_core.inspection_multiresolution import aggregate_page
from omni_core.paged_inspection_queries import PagedInspectionIndex
from omni_core.substrate_inspection import query_persisted_substrate, _inspect_neuron, _inspect_assembly
from omni_core.whole_brain_inspection import query_whole_brain


class MultiresolutionInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-multiresolution-")
        self.engine = Path(self.temporary.name) / "engine"
        self.store = self.engine / "substrate"
        self.store.mkdir(parents=True)
        self.admissions = []
        self.neurons = [{"id": name, "neuron_id": name, "label": name, "region": region, "activation": activation,
            "importance": .5, "uncertainty": .2, "exposures": 1, "aliases": []}
            for name, region, activation in (("a", "sensory", .9), ("b", "semantic", .3), ("c", "semantic", .05),
                ("root-assembly", "assembly", .2), ("child", "assembly", .1))]
        self.assemblies = [{"id": "root-assembly", "assembly_neuron_id": "root-assembly", "source_label": "Actual fixture root",
            "neuron_ids": ["a" if number % 2 == 0 else "b" for number in range(5000)], "child_assembly_ids": ["child"],
            "confidence": .7, "importance": .5, "rehearsals": 1, "kind": "knowledge", "source": "fixture"},
            {"id": "child", "assembly_neuron_id": "child", "source_label": "Actual child", "neuron_ids": [],
             "child_assembly_ids": [], "confidence": .5, "importance": .3, "rehearsals": 0, "kind": "knowledge", "source": "fixture"}]
        self.contacts = [{"id": "contact-" + str(number), "sourceId": source, "targetId": target, "kind": "association", "effectiveWeight": sign,
            "connectionStrength": 1, "eligibility": 0., "plasticity": .2, "stability": .3, "uses": 1}
            for number, (source, target, sign) in enumerate((("a", "b", -1), ("a", "c", 0), ("a", "b", 1), ("b", "a", 1)))]
        self.shards = [self.shard("neurons", self.neurons), self.shard("assemblies", self.assemblies),
            {"kind": "synapses", "count": len(self.contacts), "records": {"bytes": 0}, "tensors": {"bytes": 0}}]
        self.view = SimpleNamespace(engine_directory=self.engine, store=self.store, brain_id="multiresolution-fixture",
            substrate_revision="b" * 64, generation_sha256="c" * 64, revision="d" * 64,
            generation={"formatVersion": 3}, counts={"neurons": 5, "assemblies": 2, "synapses": 4},
            attention_legacy_raw_active=True, attention_active_neuron_ids=frozenset(), attention_recalled_assembly_ids=frozenset(),
            attention_eligible_synapse_ids=frozenset(), attention_epoch=0,
            kind_shards=lambda kind: [row for row in self.shards if row["kind"] == kind],
            records=lambda _: (_ for _ in ()).throw(AssertionError("whole source records must not be decoded")),
            effective_activation=lambda row: row.get("activation", 0.), effective_eligibility=lambda _, value: value)
        saved = {"brain_id": self.view.brain_id, "mutable_state": {"activeGeneration": "a" * 64},
                 "substrate": {"persistence": {"activeGeneration": self.view.substrate_revision}}, "attention_overlay": {}}
        (self.engine / "brain.json").write_bytes(canonical(saved))

    def tearDown(self):
        self.temporary.cleanup()

    def shard(self, kind, records):
        ids = [row["id"] for row in records]
        payload = canonical({"kind": kind, "ids": ids, "vectorIds": ids, "records": records,
            **({"vectorStorage": "shared-neuron-packed"} if kind == "assemblies" else {"packedVectorState": {"ids": ids}})})
        sha = hashlib.sha256(payload).hexdigest()
        path = self.store / "blobs" / (sha + ".json")
        path.parent.mkdir(exist_ok=True); path.write_bytes(payload)
        return {"kind": kind, "count": len(records), "records": {"sha256": sha, "path": "blobs/" + sha + ".json", "bytes": len(payload)}, "tensors": None if kind == "assemblies" else {}}

    def index(self):
        with inspection_resources(lambda size, label: self.admissions.append((size, label))), \
                patch("omni_core.substrate_inspection._synapse_records", return_value=iter(self.contacts)):
            return PagedInspectionIndex(self.view)

    def test_streamed_large_member_arrays_and_real_ordinal_tail(self):
        original = json.loads
        def guarded(value, *args, **kwargs):
            text = value.decode("utf-8") if isinstance(value, bytes) else value
            self.assertFalse(text.startswith("["), "no unbounded membership array is JSON-decoded")
            self.assertNotIn('"neuron_ids":["a"', text)
            return original(value, *args, **kwargs)
        with patch("omni_core.inspection_source_stream.json.loads", side_effect=guarded):
            index = self.index()
        header, members, total = index.members("root-assembly", 4999, 2)
        self.assertEqual(total, 5001)
        self.assertEqual(header["neuron_ids"], [])
        self.assertEqual([(row["memberKind"], row["ordinal"], row["memberId"]) for row in members], [("neuron", 4999, "b"), ("assembly", 0, "child")])
        self.assertTrue(any("stream" in label for _, label in self.admissions))

    def test_actual_group_signs_and_exact_directed_pathway_contacts(self):
        index = self.index()
        rows, anchors, matched = aggregate_page(index, zoom=0, region="", search="", offset=0, page_size=1)
        self.assertEqual(matched, 5)
        pathway = rows[0]
        self.assertEqual(pathway["kind"], "pathway")
        self.assertEqual(pathway["effectiveWeights"], {"negative": 1, "zero": 1, "positive": 1})
        self.assertEqual({row["region"] for row in anchors}, {"sensory", "semantic"})
        self.assertEqual({row["region"]: row["count"] for row in anchors}, {"sensory": 1, "semantic": 2})
        contacts, total = index.page("synapses", region="", search="", connected_to="", offset=0, page_size=10,
                                     source_region="sensory", target_region="semantic")
        self.assertEqual(total, 3); self.assertEqual({row["id"] for row in contacts}, {"contact-0", "contact-1", "contact-2"})
        self.view.attention_legacy_raw_active = False
        self.view.attention_active_neuron_ids = frozenset({"a"})
        bands, _, _ = aggregate_page(index, zoom=.5, region="semantic", search="", offset=0, page_size=20)
        self.assertEqual([(row["count"], row["activeCount"]) for row in bands if row["kind"] == "activation-band"], [(2, 0)])

    def test_real_whole_map_member_cursor_actual_endpoints_and_snapshot_fence(self):
        self.index()
        with patch("omni_core.substrate_inspection.PersistedSubstrateView.open", return_value=self.view):
            query = {"entity": "substrate", "pageSize": 1, "substrate": {"entity": "assemblies", "zoom": 1, "membersOf": "root-assembly"}}
            admit = lambda size, label: self.admissions.append((size, label))
            first = query_whole_brain(self.engine, self.view.brain_id, query, admit=admit)
            self.assertEqual(first["substrate"]["membershipPage"]["members"][0]["ordinal"], 0)
            self.assertEqual({row["id"] for row in first["substrate"]["neurons"]}, {"a", "root-assembly"})
            second = query_whole_brain(self.engine, self.view.brain_id, {**query, "cursor": first["nextCursor"]}, admit=admit)
            self.assertEqual(second["substrate"]["membershipPage"]["members"][0]["ordinal"], 1)
            with self.assertRaisesRegex(ValueError, "generation/query changed"):
                query_whole_brain(self.engine, self.view.brain_id, {**query, "substrate": {**query["substrate"], "membersOf": "child"}, "cursor": first["nextCursor"]}, admit=admit)
            saved = json.loads((self.engine / "brain.json").read_bytes()); saved["updated_at"] = "changed"
            (self.engine / "brain.json").write_bytes(canonical(saved))
            with self.assertRaisesRegex(ValueError, "generation/query changed"):
                query_whole_brain(self.engine, self.view.brain_id, {**query, "cursor": second["nextCursor"]}, admit=admit)

    def test_cancelled_upgrade_does_not_publish_partial_index(self):
        checks = 0
        def cancelled():
            nonlocal checks
            checks += 1
            return checks >= 50
        with inspection_resources(lambda *_: None, cancelled), \
                patch("omni_core.substrate_inspection._synapse_records", return_value=iter(self.contacts)):
            with self.assertRaises(InterruptedError): PagedInspectionIndex(self.view)
        directory = self.store / "inspection" / "query-generations" / self.view.substrate_revision
        self.assertFalse((directory / "manifest.json").exists())
        self.assertEqual(list(directory.glob(".query-index-*")), [])

    def test_exact_activation_band_and_current_native_packed_ids(self):
        index = self.index()
        rows, count = index.page("neurons", region="semantic", search="", connected_to="", offset=0, page_size=2, activation_band="medium")
        self.assertEqual(count, 1); self.assertEqual([row["id"] for row in rows], ["b"])
        self.view.attention_legacy_raw_active = False
        self.view.attention_active_neuron_ids = frozenset({"a"})
        rows, count = index.page("neurons", region="semantic", search="", connected_to="", offset=0, page_size=2, activation_band="quiet")
        self.assertEqual(count, 2); self.assertEqual({row["id"] for row in rows}, {"b", "c"})
        bands, anchors, _ = aggregate_page(index, zoom=.5, region="semantic", search="", offset=0, page_size=8)
        self.assertEqual([(row["activationBand"], row["count"]) for row in bands if row["kind"] == "activation-band"], [("quiet", 2)])
        self.assertTrue(any(row["region"] == "semantic" and row["count"] == 2 for row in anchors))
        malformed = self.shard("neurons", self.neurons)
        path = self.store / malformed["records"]["path"]
        payload = json.loads(path.read_bytes()); payload["packedVectorState"]["ids"][0] = "wrong-packed-id"
        raw = canonical(payload); sha = hashlib.sha256(raw).hexdigest()
        (path.parent / (sha + ".json")).write_bytes(raw)
        malformed["records"] = {"path": "blobs/" + sha + ".json", "sha256": sha, "bytes": len(raw)}
        self.shards[0] = malformed
        self.view.substrate_revision = "e" * 64
        with self.assertRaisesRegex(ValueError, "streamed source IDs differ"): self.index()

    def test_checked_index_custody_rejects_changed_file_before_page_publication(self):
        index = self.index()
        with self.assertRaisesRegex(ValueError, "index changed during traversal"):
            with index.connection() as connection:
                self.assertEqual(connection.execute("SELECT count(*) FROM assembly_members").fetchone()[0], 5001)
                with index.path.open("ab") as handle: handle.write(b"changed derived index")


if __name__ == "__main__": unittest.main()
