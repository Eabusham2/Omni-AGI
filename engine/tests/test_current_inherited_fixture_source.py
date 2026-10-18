"""Small custody-only fixtures. No native imports, model or training."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("current_inherited_fixture_source.py")
spec = importlib.util.spec_from_file_location("current_inherited_fixture_source", MODULE_PATH)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class FixtureSourceQualification(unittest.TestCase):
    def make_source(self, folder):
        source = Path(folder) / "source-a"
        engine = source / "engine"
        engine.mkdir(parents=True)
        roles = {}
        for name in ("core", "plasticity", "optimizer"):
            data = ("custody-schema-fixture-only:" + name).encode()
            sha = hashlib.sha256(data).hexdigest()
            path = engine / "state/blobs" / (sha + ".safetensors")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            if name != "optimizer":
                (engine / (name + ".safetensors")).write_bytes(data)
            roles[name] = {"path": "blobs/" + path.name, "bytes": len(data), "sha256": sha}
        pointers = {}
        for directory, body in (("state", {"format": "omni-mutable-state", "formatVersion": 1, "roles": roles}),
                                ("substrate", {"format": "omni-substrate-shards", "formatVersion": 3, "shards": []})):
            body["contentSha256"] = hashlib.sha256(fixture.canonical(body)).hexdigest()
            path = engine / directory / "generations" / body["contentSha256"] / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_bytes(fixture.canonical(body))
            pointers[directory] = {"activeGeneration": body["contentSha256"], "contentSha256": body["contentSha256"],
                                   "generationManifest": "generations/" + body["contentSha256"] + "/manifest.json",
                                   "generationManifestSha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (engine / "brain.json").write_text(json.dumps({"brain_id": source.name, "config": {"origin_kind": "ground-up"},
            "mutable_state": pointers["state"], "substrate": {"persistence": pointers["substrate"]}}))
        return source, roles

    def test_missing_origin_is_not_promoted_from_valid_current_bytes(self):
        with tempfile.TemporaryDirectory(prefix="omni-source-qualification-") as folder:
            source, _roles = self.make_source(folder)
            before = fixture.digest(source / "engine/brain.json")
            report = fixture.qualify_initialized_source(source)
            self.assertTrue(report["currentLearnedClosureVerified"])
            self.assertFalse(report["qualified"])
            self.assertIn("brain.json", report["missingOriginFiles"])
            self.assertEqual(before, fixture.digest(source / "engine/brain.json"))
            self.assertFalse((source / "engine/origin").exists())

    def test_missing_committed_blob_and_missing_origin_are_both_disclosed(self):
        with tempfile.TemporaryDirectory(prefix="omni-source-qualification-") as folder:
            source, roles = self.make_source(folder)
            (source / "engine/state" / roles["core"]["path"]).unlink()
            report = fixture.qualify_initialized_source(source)
            self.assertFalse(report["qualified"])
            self.assertFalse(report["currentLearnedClosureVerified"])
            self.assertIn("brain.json", report["missingOriginFiles"])
            self.assertIn(roles["core"]["path"], report["problems"][0])
            self.assertTrue((source / "engine/core.safetensors").is_file())

    def test_unsafe_paths_and_unqualified_copy_are_refused_without_target(self):
        with tempfile.TemporaryDirectory(prefix="omni-source-qualification-") as folder:
            source, _roles = self.make_source(folder)
            target = Path(folder) / "omni-connected-current-seal-owned" / source.name
            for relative in ("../outside", "/outside", "blobs\\outside", "blobs/./outside"):
                with self.assertRaises(ValueError):
                    fixture.inside(source, relative)
            with self.assertRaisesRegex(ValueError, "cannot be copied"):
                fixture.copy_qualified_initialized_source(source, target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
