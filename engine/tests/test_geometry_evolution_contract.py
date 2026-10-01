"""Metadata, frozen-file and fake candidate gates; no real neural construction."""

import copy
import hashlib
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded
from omni_core.evolution import (
    NeuralEvolutionManager, _architecture_signature, _bundle_checksum, _file_sha256,
    _geometry_holdout_checks, _geometry_holdout_metrics, _geometry_promotion_authorized,
    _geometry_training_complete,
)
from omni_core.architecture_migration import normalize_architecture_change
from omni_core.geometry_candidate_application import ROOTS, apply_isolated_geometry_candidate
from omni_core.persistence import atomic_write_json, read_json, snapshot_files
from omni_core.registered_geometry_holdouts import load_registered_geometry_holdouts
from omni_core.paired_geometry_statistics import PairedLossWriter
import test_geometry_candidate_application as application_fixtures
from test_geometry_candidate_application import Pager, Policy, roots


def register_data(engine):
    categories = {}
    for category in ("token", "modality", "tool"):
        path = engine / "evaluation" / "data" / (uuid.uuid4().hex + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"fixture": category, "heldout": True})
        categories[category] = [{"path": path.relative_to(engine).as_posix(), "sha256": _file_sha256(path), "records": 1}]
    manifest = engine / "evaluation" / "geometry-holdouts.json"
    atomic_write_json(manifest, {"format": "omni-registered-geometry-holdouts", "formatVersion": 1, "categories": categories})
    registration = {"format": "omni-geometry-holdout-registration", "formatVersion": 1,
        "manifestPath": "evaluation/geometry-holdouts.json", "manifestSha256": _file_sha256(manifest)}
    metadata = read_json(engine / "brain.json") if (engine / "brain.json").is_file() else {}
    atomic_write_json(engine / "brain.json", {**metadata, "geometry_holdout_registration": registration})
    return registration


def metrics(**kwargs):
    manifest = kwargs["registered_manifest"]
    policy = getattr(metrics, "policy")
    writer = PairedLossWriter(kwargs["score_path"], policy, sum(value["examples"] for value in manifest["categories"].values()))
    for category, values in manifest["categories"].items():
        for index in range(values["examples"]): writer.add(category, hashlib.sha256((category + str(index)).encode()).hexdigest(), 1., 1)
    paired = writer.finish()
    return {"format": "omni-native-geometry-holdouts", "formatVersion": 1, "benchmarkSha256": manifest["benchmarkSha256"],
        "pairedScores": paired,
        **{category: {"sourceSha256": values["sourceSha256"], "loss": 1.0, "examples": values["examples"],
            "heldOut": True, "trainingOverlapCount": 0} for category, values in manifest["categories"].items()},
        "resources": {"withinSelectedEnvelope": True, "peakManagedMemoryBytes": 4096, "peakAcceleratorMemoryBytes": 0}}


class EvolutionGeometryGates(unittest.TestCase):
    def test_training_receipt_must_cover_all_requested_windows_epochs_and_change_native_parameters(self):
        expected = {"epochs": 2, "fingerprints": [{"sha256": "c" * 64, "utf8Bytes": 18}], "latent_replay": False,
            "insertion_parameter_checksum": "a" * 64, "candidate_parameter_checksum": "b" * 64}
        valid = {"promoted": True, "epochs": 2, "samples": 1, "steps": 14, "trainingSequenceTokens": 4,
            "meanLoss": .1, "parameterChecksumBefore": "a" * 64, "parameterChecksumAfter": "b" * 64}
        self.assertTrue(_geometry_training_complete(valid, **expected))
        for change in ({"steps": 13}, {"epochs": 1}, {"samples": 0}, {"paused": True}, {"disabled": True},
            {"meanLoss": float("nan")}, {"parameterChecksumAfter": "a" * 64}, {"mode": "function-preserving-architecture-insertion"}):
            self.assertFalse(_geometry_training_complete({**valid, **change}, **expected))

    def test_callback_cannot_invent_success_without_protected_real_registered_data(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory); atomic_write_json(engine / "brain.json", {"config": {}})
            candidate = SimpleNamespace(engine_path=engine, evaluate_isolated_geometry_holdouts=metrics)
            with self.assertRaisesRegex(ValueError, "protected"):
                _geometry_holdout_metrics(candidate, [])
            register_data(engine)
            from test_geometry_registration_transactions import Policy as QuotaPolicy
            metrics.policy = QuotaPolicy(directory, 2**20)
            measured = _geometry_holdout_metrics(candidate, [], score_path=engine / "scores.jsonl")
            checks = _geometry_holdout_checks(measured, measured)
            self.assertTrue(checks["heldOutToken"]); self.assertFalse(checks["statisticallySupportedImprovement"])
            altered = copy.deepcopy(measured); altered["modality"]["loss"] = 2
            self.assertFalse(_geometry_holdout_checks(measured, altered)["heldOutModality"])
            def incomplete(**kwargs):
                result = metrics(**kwargs); result["token"]["examples"] = 0; return result
            candidate.evaluate_isolated_geometry_holdouts = incomplete
            with self.assertRaises(ValueError): _geometry_holdout_metrics(candidate, [], score_path=engine / "incomplete.jsonl")

    def test_registered_data_tamper_traversal_and_mid_measurement_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = Path(directory); atomic_write_json(engine / "brain.json", {})
            register_data(engine)
            token_path = engine / read_json(engine / "evaluation" / "geometry-holdouts.json")["categories"]["token"][0]["path"]
            from test_geometry_registration_transactions import Policy as QuotaPolicy
            metrics.policy = QuotaPolicy(directory, 2**20)
            def changing(**kwargs):
                result = metrics(**kwargs)
                atomic_write_json(token_path, {"different": True})
                return result
            with self.assertRaisesRegex(ValueError, "hash changed"):
                _geometry_holdout_metrics(SimpleNamespace(engine_path=engine, evaluate_isolated_geometry_holdouts=changing), [], score_path=engine / "changed.jsonl")
            with self.assertRaises(ValueError): load_registered_geometry_holdouts(engine)
            register_data(engine)
            body = read_json(engine / "evaluation" / "geometry-holdouts.json")
            body["categories"]["tool"][0]["path"] = "evaluation/../outside.jsonl"
            atomic_write_json(engine / "evaluation" / "geometry-holdouts.json", body)
            metadata = read_json(engine / "brain.json")
            metadata["geometry_holdout_registration"]["manifestSha256"] = _file_sha256(engine / "evaluation" / "geometry-holdouts.json")
            atomic_write_json(engine / "brain.json", metadata)
            with self.assertRaisesRegex(ValueError, "inside|owned"):
                load_registered_geometry_holdouts(engine)

    def test_geometry_authorization_is_candidate_evaluation_state_bound_and_default_deny(self):
        evaluated = {"evaluationSha256": "b" * 64}
        valid = {"mode": "ask", "approved": True, "candidateId": "abc", "evaluationSha256": "b" * 64,
            "candidateStateChecksum": "c" * 64}
        self.assertFalse(_geometry_promotion_authorized(None, "abc", evaluated, "c" * 64))
        self.assertTrue(_geometry_promotion_authorized(valid, "abc", evaluated, "c" * 64))
        self.assertTrue(_geometry_promotion_authorized({**valid, "mode": "full-authority"}, "abc", evaluated, "c" * 64))
        for change in ({"approved": False}, {"mode": "automatic"}, {"candidateId": "other"},
            {"evaluationSha256": "d" * 64}, {"candidateStateChecksum": "d" * 64}):
            self.assertFalse(_geometry_promotion_authorized({**valid, **change}, "abc", evaluated, "c" * 64))

    def test_manager_calls_only_isolated_fake_factory_and_persists_exact_cancelled_migration_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate, candidate_id = application_fixtures.ApplicationFixtures().fixture(directory)
            candidate_dir = candidate.engine_path.parent.parent
            def save():
                config = {key: value for key, value in vars(candidate.config).items() if not callable(value)}
                metadata = read_json(candidate.engine_path / "brain.json") if (candidate.engine_path / "brain.json").is_file() else {}
                atomic_write_json(candidate.engine_path / "brain.json", {**metadata, "config": config, "expert_count": 0})
                core = {name + "." + key: value for name in ROOTS if name != "router" for key, value in getattr(candidate, name).state_dict().items()}
                plastic = {"router." + key: value for key, value in candidate.router.state_dict().items()}
                plastic["state.liquid"] = candidate.liquid_state
                atomic_save_tensors_bounded(candidate.engine_path / "core.safetensors", core)
                atomic_save_tensors_bounded(candidate.engine_path / "plasticity.safetensors", plastic)
            candidate.save = save; save()
            snapshot_files(candidate.engine_path, candidate_dir / "stable")
            candidate.parameter_checksum = lambda: _file_sha256(candidate.engine_path / "core.safetensors")
            observed = []
            def apply(change, identity, **kwargs):
                observed.append((identity, kwargs["cancelled"]))
                return apply_isolated_geometry_candidate(candidate, change, identity, cancelled=kwargs["cancelled"],
                    module_factory=lambda config, source: roots(config.d_model, source.router.children["population"]), pager_factory=lambda *_: Pager())
            candidate.apply_isolated_geometry_candidate = apply
            manager = NeuralEvolutionManager.__new__(NeuralEvolutionManager)
            manager.engine_path = Path(directory) / "live" / "engine"
            updates = []
            manager.brain = SimpleNamespace(resource_policy=Policy(), _record_candidate=lambda *a, **kwargs: updates.append(kwargs))
            baseline = {"parentMetadataSha256": _file_sha256(candidate.engine_path / "brain.json"),
                "parentStateChecksum": _bundle_checksum(candidate.engine_path), "anchorTensorSha256": "a" * 64,
                "architecture": _architecture_signature(candidate.engine_path),
                "architectureMutation": normalize_architecture_change({"mutation": "resize-width", "dModel": 64}),
                "geometryRetentionPolicy": {"functionPreserved": False}}
            cancelled = lambda: False
            output = manager._prepare_geometry_candidate(candidate, candidate_id, candidate_dir, candidate.engine_path,
                baseline["architectureMutation"], baseline, Path(directory) / "baseline.json", cancel_check=cancelled)
            self.assertEqual(observed, [(candidate_id, cancelled)])
            self.assertFalse(output["functionPreserved"])
            self.assertTrue((candidate_dir / "geometry-migration.json").is_file())
            self.assertTrue((candidate_dir / "geometry-insertion" / "core.safetensors").is_file())
            record = {"id": candidate_id, "geometryMigration": baseline["geometryMigration"]}
            manager._load_geometry_migration(candidate_dir, record, baseline)
            atomic_save_tensors_bounded(candidate_dir / "geometry-insertion" / "core.safetensors", {"tampered": torch.tensor([0x55], dtype=torch.uint8)})
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                manager._load_geometry_migration(candidate_dir, record, baseline)
            with self.assertRaisesRegex(RuntimeError, "live parent"):
                manager._prepare_geometry_candidate(manager.brain, candidate_id, candidate_dir, manager.engine_path,
                    baseline["architectureMutation"], baseline, Path(directory) / "bad.json")


if __name__ == "__main__": unittest.main()
