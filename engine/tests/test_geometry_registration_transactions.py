"""Real quota/storage transactions only, no brain or model construction."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.shared_resource_ledger import SharedResourceLedger, SharedQuotaPause
from omni_core.registered_geometry_holdouts import register_geometry_holdouts, load_registered_geometry_holdouts, portable_geometry_reference
from omni_core.persistence import atomic_write_json, read_json, snapshot_files
from omni_core.paired_geometry_statistics import PairedLossWriter, paired_improvement_statistics
from omni_core.evolution import measured_process_comparison


class Policy:
    def __init__(self, directory, pool):
        self.ledger = SharedResourceLedger(Path(directory) / "quota.sqlite3")
        self.ledger.register_owner("fixture", pool)
    def reserve_spill(self, count, operation): return self.ledger.reserve("fixture", "spill", count)
    def status(self, **kwargs): return {"memoryPressure": False, "diskPressure": False}


class GeometryRegistrationTransactions(unittest.TestCase):
    def fixture(self, directory, pool=2**20):
        root = Path(directory); engine = root / "engine"; engine.mkdir()
        atomic_write_json(engine / "brain.json", {})
        declarations = {}
        for category in ("token", "tool", "modality"):
            source = root / (category + ".данные+!")
            source.write_bytes(b'{"text":"literal heldout"}\n')
            declarations[category] = ([{"path": str(source), "kind": "audio", "conditionText": "literal utterance"}]
                if category == "modality" else [{"path": str(source), "records": 1}])
        brain = SimpleNamespace(engine_path=engine, resource_policy=Policy(root, pool), geometry_holdout_registration=None)
        def save(): atomic_write_json(engine / "brain.json", {"geometry_holdout_registration": brain.geometry_holdout_registration})
        brain.save = save
        return brain, declarations

    def test_success_uses_safe_owned_names_original_provenance_and_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, declarations = self.fixture(directory)
            result = register_geometry_holdouts(brain, declarations)
            for category in result["categories"].values():
                for item in category["files"]:
                    self.assertEqual(item["path"], portable_geometry_reference(item["path"]))
                    self.assertIn("данные", item["sourceName"])
                    self.assertTrue((brain.engine_path / item["path"]).is_file())
            self.assertGreater(brain.resource_policy.ledger.status()["physicalSpillUsageBytes"], 0)
            self.assertEqual(brain.resource_policy.ledger.status()["pendingSpillBytes"], 0)

    def test_quota_denial_and_cancel_leave_no_new_orphan_or_pending_credit(self):
        for pool in (65536, 2**20):
            with self.subTest(pool=pool), tempfile.TemporaryDirectory() as directory:
                brain, declarations = self.fixture(directory, pool)
                calls = 0
                def cancelled():
                    nonlocal calls
                    calls += 1
                    return calls > 3
                with self.assertRaises((SharedQuotaPause, InterruptedError)):
                    register_geometry_holdouts(brain, declarations, cancelled=cancelled if pool > 65536 else None)
                self.assertEqual(list((brain.engine_path / "evaluation" / "data").iterdir()), [])
                status = brain.resource_policy.ledger.status()
                self.assertEqual(status["pendingSpillBytes"], 0)
                self.assertEqual(status["physicalSpillUsageBytes"], 0)
                self.assertIsNone(read_json(brain.engine_path / "brain.json").get("geometry_holdout_registration"))

    def test_precommit_failure_restores_previous_data_but_postcommit_error_never_rolls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, declarations = self.fixture(directory)
            previous = register_geometry_holdouts(brain, declarations)
            old_manifest = (brain.engine_path / "evaluation" / "geometry-holdouts.json").read_bytes()
            old_files = set((brain.engine_path / "evaluation" / "data").iterdir())
            def fail(): raise RuntimeError("precommit")
            brain.save = fail
            with self.assertRaisesRegex(RuntimeError, "precommit"): register_geometry_holdouts(brain, declarations)
            self.assertEqual((brain.engine_path / "evaluation" / "geometry-holdouts.json").read_bytes(), old_manifest)
            self.assertEqual(set((brain.engine_path / "evaluation" / "data").iterdir()), old_files)
            self.assertEqual(load_registered_geometry_holdouts(brain.engine_path)["benchmarkSha256"], previous["benchmarkSha256"])
            def committed_then_fail():
                atomic_write_json(brain.engine_path / "brain.json", {"geometry_holdout_registration": brain.geometry_holdout_registration})
                raise RuntimeError("after commit")
            brain.save = committed_then_fail
            with self.assertRaisesRegex(RuntimeError, "after commit"): register_geometry_holdouts(brain, declarations)
            self.assertNotEqual(load_registered_geometry_holdouts(brain.engine_path)["benchmarkSha256"], previous["benchmarkSha256"])
            self.assertTrue(old_files.issubset(set((brain.engine_path / "evaluation" / "data").iterdir())))

    def test_legacy_portable_alias_preserves_protected_manifest_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            brain, declarations = self.fixture(directory)
            result = register_geometry_holdouts(brain, declarations)
            manifest_path = brain.engine_path / "evaluation" / "geometry-holdouts.json"
            body = read_json(manifest_path)
            first = body["categories"]["token"][0]
            source = brain.engine_path / first["path"]
            legacy = "evaluation/data/" + source.name[:32] + ".данные+!"
            source.rename(brain.engine_path / portable_geometry_reference(legacy))
            first["path"] = legacy
            atomic_write_json(manifest_path, body)
            from omni_core.registered_geometry_holdouts import _file_sha
            brain.geometry_holdout_registration["manifestSha256"] = _file_sha(manifest_path); brain.save()
            before = manifest_path.read_bytes()
            loaded = load_registered_geometry_holdouts(brain.engine_path)
            self.assertEqual(loaded["benchmarkSha256"], brain.geometry_holdout_registration["manifestSha256"])
            self.assertEqual(before, manifest_path.read_bytes())
            self.assertEqual(first["path"], legacy)
            with self.assertRaises(ValueError): portable_geometry_reference("evaluation/data/../outside")

    def test_paired_supported_language_gain_tied_other_domains_and_no_fake_meta_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Policy(directory, 2**20)
            def scores(name, gain, count=20):
                writer = PairedLossWriter(Path(directory) / name, policy, count * 3)
                for category in ("token", "modality", "tool"):
                    for index in range(count):
                        writer.add(category, f"{index:064x}", 1 - gain if category == "token" else 1., 10)
                return writer.finish()
            old, new = scores("baseline.jsonl", 0), scores("candidate.jsonl", .1)
            stats = paired_improvement_statistics(Path(directory) / old["path"], Path(directory) / new["path"], old, new,
                objectives=["language-prediction"])
            self.assertTrue(stats["passed"]); self.assertEqual(stats["supportedObjectiveDomains"], ["token"])
            self.assertEqual(stats["domains"]["tool"]["ties"], 20)
            tied = scores("tied.jsonl", 0)
            self.assertFalse(paired_improvement_statistics(Path(directory) / old["path"], Path(directory) / tied["path"], old, tied,
                objectives=["language-prediction"])["passed"])
            small_old, small_new = scores("small-old.jsonl", 0, 2), scores("small-new.jsonl", .1, 2)
            self.assertFalse(paired_improvement_statistics(Path(directory) / small_old["path"], Path(directory) / small_new["path"], small_old, small_new,
                objectives=["language-prediction"])["passed"])
            wrong = dict(new)
            wrong["records"] -= 1
            with self.assertRaisesRegex(ValueError, "counts"):
                paired_improvement_statistics(Path(directory) / old["path"], Path(directory) / new["path"], old, wrong,
                    objectives=["language-prediction"])
            parent = measured_process_comparison(passed=True, wall_seconds=20, statistics=stats)
            child = measured_process_comparison(passed=True, wall_seconds=10, statistics=stats, parent=parent)
            self.assertTrue(child["gainPerCostImproved"])
            self.assertEqual(child["wallCostRatio"], .5)
            self.assertFalse(child["metaImprovementEstablished"])


if __name__ == "__main__": unittest.main()
