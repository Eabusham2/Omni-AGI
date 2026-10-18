"""Tiny files/SQLite + inert objects. No tensor/model or native brain construction."""
import base64
import ast
from contextlib import contextmanager
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from omni_core.whole_brain_catalog import (InspectionScratch, canonical, digest, iter_catalog_files, iter_catalog_lineage_files,
    publish_whole_brain_catalog, read_owned_json)
from omni_core.whole_brain_inspection import query_whole_brain, _actual_sparse_endpoints, _stream_json_size


class Module:
    def __init__(self, **children):
        self._modules, self._buffers = children, {}


class Tensor:
    shape = (2, 1)


def packed():
    value = Module()
    value._buffers["_packed_forward_weight"] = Tensor()
    return value


def metadata():
    return {"brain_id": "whole-fixture", "mutable_state": {"activeGeneration": "a" * 64},
            "substrate": {"persistence": {"activeGeneration": "b" * 64}}, "attention_overlay": {}}


def brain():
    shared = packed()
    roots = {"decoder": Module(shared=shared), "memory_bridge": shared, "idea_adapter": Module(),
             "router": Module(), "liquid": Module(), "modalities": Module()}
    return SimpleNamespace(brain_id="whole-fixture", _ternary_export_roots=lambda: roots), roots


class WholeBrainInspectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-whole-graph-")
        self.engine = Path(self.temporary.name) / "engine"
        self.engine.mkdir()
        self.admissions = []
        self.admit = lambda size, label: self.admissions.append(size)
        self.disk = lambda size, label: None

    def tearDown(self):
        self.temporary.cleanup()

    def save(self, instance=None, saved=None, registry=None):
        instance = instance or brain()[0]
        saved = saved or metadata()
        saved["whole_brain_catalog"] = publish_whole_brain_catalog(instance, saved, self.engine,
            admit=self.admit, reserve_disk=self.disk, tensor_registry=registry)
        (self.engine / "brain.json").write_bytes(canonical(saved))
        return saved

    def rows(self):
        result, cursor = [], None
        while True:
            page = query_whole_brain(self.engine, "whole-fixture", {"catalogFacet": "all", "pageSize": 3, **({"cursor": cursor} if cursor else {})}, admit=self.admit)
            result.extend(page["records"])
            if not page["hasMore"]:
                return result
            cursor = page["nextCursor"]

    def test_real_alias_groups_stable_addresses_and_exact_saved_field_binding(self):
        first, roots = brain()
        registry = {"core": {"decoder.shared._packed_forward_weight": roots["memory_bridge"]._buffers["_packed_forward_weight"],
                              "memory_bridge._packed_forward_weight": roots["memory_bridge"]._buffers["_packed_forward_weight"]}, "plasticity": {}}
        saved = self.save(first, registry=registry)
        rows = self.rows()
        aliases = {row["alias"]: row["ownerId"] for row in rows if row["kind"] == "alias"}
        self.assertEqual(aliases["decoder.shared"], aliases["memory_bridge"])
        memberships = [row for row in rows if row["kind"] == "membership" and row["ownerId"] == aliases["memory_bridge"]]
        self.assertEqual(len(memberships), 2)
        bindings = [row for row in rows if row["kind"] == "inspection"]
        self.assertEqual({row["moduleId"] for row in bindings}, set(registry["core"]))
        self.assertTrue(all(row["tensorKey"] == row["moduleId"] and row["ownerId"] == aliases["memory_bridge"] for row in bindings))
        second, other = brain()
        second_registry = {"core": {key: other["memory_bridge"]._buffers["_packed_forward_weight"] for key in registry["core"]}, "plasticity": {}}
        self.save(second, registry=second_registry)
        self.assertEqual(saved["whole_brain_catalog"]["tailSha256"], self.save(second, registry=second_registry)["whole_brain_catalog"]["tailSha256"])
        self.assertEqual(rows, self.rows())

    def test_actual_connected_bridge_registration_not_name_similarity(self):
        instance, roots = brain()
        parents = []
        for index in range(2):
            value = Module(decoder=Module())
            value.identity = SimpleNamespace(region_id="r" + str(index), parent_id="parent" + str(index), generation_sha256=str(index + 1) * 64)
            parents.append(value)
        decoder = Module(regions=Module(**{str(i): region for i, region in enumerate(parents)}))
        decoder.regions = parents
        for field in ("workspace_exchange", "cortical_exchange"):
            exchange = Module(bridges=Module(r0_to_r1=packed(), r1_to_r0=packed()))
            exchange.bridges, exchange.widths = exchange._modules["bridges"], (2, 2)
            setattr(decoder, field, exchange); decoder._modules[field] = exchange
        # A similarly named unrelated module must not fabricate another edge.
        decoder._modules["r7_to_r9"] = packed()
        roots["decoder"] = decoder
        self.save(instance)
        rows = self.rows()
        self.assertEqual(len([row for row in rows if row["kind"] == "bridge"]), 4)
        regions = {row["id"] for row in rows if row["kind"] == "region" and row["regionKind"] == "connected-parent"}
        for row in rows:
            if row["kind"] == "bridge":
                self.assertIn(row["sourceRegionId"], regions); self.assertIn(row["targetRegionId"], regions)
                self.assertFalse(row["learnedNonzeroObserved"]); self.assertFalse(row["firingObserved"])

    def test_all_chained_pages_closure_and_forged_or_stale_cursors_refuse(self):
        instance, roots = brain()
        roots["decoder"]._modules.update({"module" + str(i): packed() for i in range(180)})
        saved = self.save(instance)
        self.assertGreater(saved["whole_brain_catalog"]["shards"], 1)
        self.assertEqual(len(self.rows()), saved["whole_brain_catalog"]["records"])
        closure = list(iter_catalog_files(self.engine, saved, admit=self.admit))
        self.assertEqual(len(closure), saved["whole_brain_catalog"]["shards"] + 1)
        first = query_whole_brain(self.engine, "whole-fixture", {"pageSize": 1}, admit=self.admit)
        cursor = json.loads(base64.urlsafe_b64decode(first["nextCursor"]))
        cursor["body"]["after"][0] += 1
        forged = base64.urlsafe_b64encode(canonical(cursor)).decode()
        with self.assertRaisesRegex(ValueError, "owned traversal"):
            query_whole_brain(self.engine, "whole-fixture", {"pageSize": 1, "cursor": forged}, admit=self.admit)
        changed = {**saved, "updated_at": "changed atomic metadata"}
        (self.engine / "brain.json").write_bytes(canonical(changed))
        with self.assertRaisesRegex(ValueError, "generation/query changed"):
            query_whole_brain(self.engine, "whole-fixture", {"pageSize": 1, "cursor": first["nextCursor"]}, admit=self.admit)

    def test_legacy_checkpoint_reports_missing_registry_without_fabrication_or_load(self):
        (self.engine / "brain.json").write_bytes(canonical(metadata()))
        page = query_whole_brain(self.engine, "whole-fixture", admit=self.admit)
        self.assertFalse(page["catalog"]["available"]); self.assertEqual(page["records"], [])
        self.assertIn("missing", page["catalog"]["reason"])
        self.assertFalse((self.engine / "whole-brain-catalog").exists())
        result = query_whole_brain(self.engine, "whole-fixture", {"entity": "cortex"}, admit=self.admit,
            cortex_query=lambda root, owner, query: {"brainId": owner, "hasMore": False, "records": []})
        self.assertEqual(result["cortex"]["brainId"], "whole-fixture")

    def test_regional_graph_resolves_only_bound_source_and_fences_after_delegate(self):
        instance, roots = brain()
        child = Module()
        child.identity = SimpleNamespace(region_id="parent-region", parent_id="actual-parent", generation_sha256="c" * 64)
        roots["decoder"]._modules["region"] = child
        states = {"format": "omni-connected-graph-states", "regions": {"parent-region": {
            "enginePath": "engine/connected-state/parent-region", "substrate": {"persistence": {"activeGeneration": "d" * 64}},
            "attentionOverlay": {}}}}
        saved = metadata(); saved["connected_graph_states"] = {**states, "sha256": digest(states)}
        self.save(instance, saved)
        region = next(row for row in self.rows() if row["kind"] == "region" and row["regionKind"] == "connected-parent")
        directory = self.engine / "connected-state" / "parent-region"; directory.mkdir(parents=True)
        calls = []
        def delegate(root, owner, query, *, _committed_source):
            value = _committed_source.resolve(root, owner)
            self.assertEqual(value["substrate"]["persistence"]["activeGeneration"], "d" * 64)
            calls.append((root, owner)); return {"brainId": owner, "hasMore": False, "records": []}
        page = query_whole_brain(self.engine, "whole-fixture", {"entity": "substrate", "regionId": region["id"]},
            admit=self.admit, substrate_query=delegate)
        self.assertEqual(calls, [(directory, "actual-parent")]); self.assertEqual(page["scope"]["nativeRegionPath"], ["parent-region"])
        def changed(root, owner, query, *, _committed_source):
            (self.engine / "brain.json").write_bytes(canonical({**saved, "updated_at": "changed during page"}))
            return {"brainId": owner, "hasMore": False}
        with self.assertRaisesRegex(ValueError, "changed during"):
            query_whole_brain(self.engine, "whole-fixture", {"entity": "substrate", "regionId": region["id"]}, admit=self.admit, substrate_query=changed)

    def test_tampered_closure_and_current_owner_or_cancellation_reject(self):
        saved = self.save()
        closure = list(iter_catalog_files(self.engine, saved, admit=self.admit))
        path = self.engine / closure[0][0]; path.write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError, "checksum|identity"):
            list(iter_catalog_files(self.engine, saved, admit=self.admit))
        with self.assertRaises(InterruptedError):
            query_whole_brain(self.engine, "whole-fixture", admit=self.admit, cancelled=lambda: True)
        with self.assertRaisesRegex(ValueError, "another identity"):
            query_whole_brain(self.engine, "other-owner", admit=self.admit)

    def test_high_water_resource_escrow_is_entered_and_fully_released(self):
        current, peak, entered = 0, 0, []
        @contextmanager
        def reserve(size, label):
            nonlocal current, peak
            entered.append(size); current += size; peak = max(peak, current)
            try: yield
            finally: current -= size
        with InspectionScratch(reserve) as scratch:
            scratch.admit(100, "first"); scratch.admit(40, "next page"); scratch.admit(150, "larger page")
            self.assertEqual(current, 150); self.assertEqual(entered, [100, 50])
        self.assertEqual(current, 0); self.assertEqual(peak, 150)

    def test_overview_and_region_scope_have_real_anchors_not_unrelated_dump(self):
        instance, roots = brain(); roots["modalities"]._modules["unrelated"] = packed()
        self.save(instance)
        first = query_whole_brain(self.engine, "whole-fixture", {"pageSize": 1}, admit=self.admit)
        self.assertEqual(first["records"][0]["regionKind"], "whole-brain")
        rows = self.rows(); decoder = next(row for row in rows if row["kind"] == "region" and row["label"] == "decoder")
        page = query_whole_brain(self.engine, "whole-fixture", {"regionId": decoder["id"], "pageSize": 256}, admit=self.admit)
        self.assertTrue(any(row["kind"] == "owner" for row in page["records"]))
        self.assertFalse(any(row.get("alias") == "modalities.unrelated" for row in page["records"]))
        indexed = {row["id"] for row in rows}
        self.assertTrue(all(anchor["id"] in indexed for anchor in page["anchors"]))

    def test_retained_original_lineage_closure_is_paged_and_never_current(self):
        saved = self.save(); pointer = saved["whole_brain_catalog"]
        key = digest({"sourceBrainId": "whole-fixture", "sourceMetadataSha256": digest(saved),
            "catalogBindingSha256": pointer["bindingSha256"], "tailSha256": pointer["tailSha256"], "indexSha256": pointer["index"]["sha256"]})
        body = {"entryKeySha256": key, "sourceBrainId": "whole-fixture", "sourceMetadataSha256": digest(saved),
            "catalogBindingSha256": pointer["bindingSha256"], "coreGeneration": "a" * 64, "substrateGeneration": "b" * 64,
            "closureRoot": "whole-brain-catalog/lineage/sources/" + key, "catalogPointer": pointer}
        entry = {**body, "entrySha256": digest(body)}
        source_files = list(iter_catalog_files(self.engine, saved, admit=self.admit))
        for relative, _, _ in source_files:
            destination = self.engine / body["closureRoot"] / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.engine / relative, destination)
        shard = {"format": "omni-whole-brain-catalog-lineage-shard", "version": 1, "previousSha256": None, "entries": [entry]}
        shard_sha = digest(shard); path = self.engine / "whole-brain-catalog/lineage/blobs" / (shard_sha + ".json")
        path.parent.mkdir(parents=True); path.write_bytes(canonical(shard))
        lineage = {"format": "omni-whole-brain-catalog-lineage", "version": 1, "tailSha256": shard_sha, "shards": 1, "entries": 1}
        copied = {**saved, "brain_id": "new-current-owner", "whole_brain_catalog_lineage": {**lineage, "manifestSha256": digest(lineage)}}
        del copied["whole_brain_catalog"]
        files = list(iter_catalog_lineage_files(self.engine, copied, admit=self.admit))
        self.assertEqual(len(files), len(source_files) + 1)
        self.assertTrue(all(relative.startswith("whole-brain-catalog/lineage/") for relative, _, _ in files))
        (self.engine / "brain.json").write_bytes(canonical(copied))
        page = query_whole_brain(self.engine, "new-current-owner", admit=self.admit)
        self.assertFalse(page["catalog"]["available"]); self.assertEqual(page["records"], [])

    def test_actual_worker_hook_is_load_free_and_save_hook_uses_existing_writer_registries(self):
        self.save()
        worker_source = Path(__file__).parents[1] / "worker.py"
        tree = ast.parse(worker_source.read_text())
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Worker")
        method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "query_whole_brain")
        for argument in (*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs): argument.annotation = None
        method.returns = None
        namespace = {"RpcFault": RuntimeError}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(worker_source), "exec"), namespace)
        reservations = []
        @contextmanager
        def reserve(size, label):
            reservations.append(size)
            yield
        fake = SimpleNamespace(_brain_id=lambda params: "whole-fixture", _storage=lambda params, owner: self.engine.parent,
            brains={}, reserve_stdio_memory=reserve, _request_cancel_check=lambda request: lambda: False,
            _get=lambda *args: self.fail("inspection must not load a brain"))
        result = namespace["query_whole_brain"](fake, {"query": {}}, "owned-source-inspection")
        self.assertEqual(result["records"][0]["regionKind"], "whole-brain"); self.assertTrue(reservations)
        source = (Path(__file__).parents[1] / "omni_core" / "brain.py").read_text()
        tree = ast.parse(source)
        method = next(node for owner in tree.body if isinstance(owner, ast.ClassDef) for node in owner.body
            if isinstance(node, ast.FunctionDef) and node.name == "_save_checkpoint_impl")
        call = next(node for node in ast.walk(method) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "publish_whole_brain_catalog")
        registry = next(keyword.value for keyword in call.keywords if keyword.arg == "tensor_registry")
        self.assertEqual([(key.value, value.id) for key, value in zip(registry.keys, registry.values)], [("core", "core"), ("plasticity", "plasticity")])

    def test_index_acquisition_failure_releases_both_owned_temporary_databases(self):
        import sqlite3
        original = sqlite3.connect
        calls = 0
        def connect(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2: raise OSError("Synthetic index initialization refusal")
            return original(*args, **kwargs)
        with patch("omni_core.whole_brain_catalog_index.sqlite3.connect", side_effect=connect):
            with self.assertRaisesRegex(OSError, "initialization refusal"):
                self.save()
        root = self.engine / "whole-brain-catalog"
        self.assertEqual(list(root.glob(".aliases-*")), [])
        self.assertEqual(list((root / "index").glob(".build-*")), [])
        self.assertFalse((self.engine / "brain.json").exists())

    def test_sparse_page_resolves_actual_endpoints_and_shared_assembly_rows_without_placeholders(self):
        import sqlite3
        database = sqlite3.connect(":memory:")
        database.execute("CREATE TABLE records(kind TEXT,id TEXT,payload BLOB,PRIMARY KEY(kind,id))")
        assemblies = [{"id": name, "assembly_neuron_id": name, "neuronIds": ["actual-member"], "childAssemblyIds": [], "neuronCount": 1,
                       "childAssemblyCount": 0} for name in ("assembly-a", "assembly-b")]
        for raw in [{"id": "actual-member"}, {"id": "assembly-a"}, {"id": "assembly-b"}]:
            database.execute("INSERT INTO records VALUES(?,?,?)", ("neurons", raw["id"], canonical(raw)))
        for raw in assemblies: database.execute("INSERT INTO records VALUES(?,?,?)", ("assemblies", raw["id"], canonical(raw)))
        @contextmanager
        def connection(): yield database
        view = SimpleNamespace(revision="checked-generation")
        factories = (lambda *args, **kwargs: view, lambda view: SimpleNamespace(connection=connection),
                     lambda view, raw: dict(raw), lambda view, raw: dict(raw))
        page = {"entity": "assemblies", "revision": "checked-generation", "neurons": [], "assemblies": assemblies, "synapses": []}
        try:
            example = {"text": "α\n\"🔬" * 3000, "nested": [True, None, 1.5, {"x": "\\"}]}
            self.assertEqual(_stream_json_size(example), len(canonical(example)))
            resolved, aliases = _actual_sparse_endpoints(page, self.engine, "whole-fixture", admit=self.admit, factories=factories)
            self.assertEqual({row["id"] for row in resolved["neurons"]}, {"actual-member", "assembly-a", "assembly-b"})
            self.assertEqual({row["assemblyId"] for row in aliases}, {"assembly-a", "assembly-b"})
            contacts, _ = _actual_sparse_endpoints({"entity": "synapses", "revision": "checked-generation", "neurons": [], "assemblies": [],
                "synapses": [{"sourceId": "actual-member", "targetId": "assembly-a"}]}, self.engine, "whole-fixture", admit=self.admit, factories=factories)
            self.assertEqual({row["id"] for row in contacts["neurons"]}, {"actual-member", "assembly-a"})
            database.execute("DELETE FROM records WHERE kind='neurons' AND id='actual-member'")
            with self.assertRaisesRegex(ValueError, "no actual neurons endpoint"):
                _actual_sparse_endpoints(page, self.engine, "whole-fixture", admit=self.admit, factories=factories)
        finally: database.close()

    def test_generic_selected_matrix_returns_exact_owner_memberships_without_prior_owner_visit(self):
        instance, roots = brain()
        field = "decoder.shared._packed_forward_weight"
        self.save(instance, registry={"core": {field: roots["memory_bridge"]._buffers["_packed_forward_weight"]}, "plasticity": {}})
        def query(root, owner, request):
            self.assertEqual(request["moduleId"], field)
            return {"brainId": owner, "hasMore": False, "selectedModule": {"id": field, "role": "core"}}
        selected = {"entity": "cortex", "cortex": {"entity": "links", "moduleId": field, "row": 0}, "pageSize": 64}
        result = query_whole_brain(self.engine, "whole-fixture", selected, admit=self.admit, cortex_query=query)
        binding = next(row for row in result["anchors"] if row["kind"] == "inspection")
        self.assertEqual((binding["role"], binding["moduleId"], binding["tensorKey"]), ("core", field, field))
        owner = next(row for row in result["anchors"] if row["kind"] == "owner")
        self.assertEqual(owner["id"], binding["ownerId"])
        memberships = [row for row in result["anchors"] if row["kind"] == "membership"]
        self.assertEqual(len(memberships), 2)
        regions = {row["id"] for row in result["anchors"] if row["kind"] == "region"}
        self.assertTrue(all(row["ownerId"] == owner["id"] and row["regionId"] in regions for row in memberships))
        bounded = query_whole_brain(self.engine, "whole-fixture", {**selected, "pageSize": 1}, admit=self.admit, cortex_query=query)
        self.assertEqual(len([row for row in bounded["anchors"] if row["kind"] == "membership"]), 1)
        self.assertTrue(any("paged owner details" in reason for reason in bounded["evidence"]["unavailable"]))


if __name__ == "__main__":
    unittest.main()
