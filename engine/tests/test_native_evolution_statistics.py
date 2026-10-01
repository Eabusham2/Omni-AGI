"""All native routes use real storage + stub measurements, no neural construction."""

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
from omni_core.evolution import NeuralEvolutionManager, _file_sha256
from omni_core.persistence import atomic_write_json
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded
from omni_core.paired_geometry_statistics import PairedLossWriter
from test_geometry_registration_transactions import Policy


class NativeStatisticsGates(unittest.TestCase):
    def fixture(self, directory, *, kind="neural", observations=40, improved=True):
        root = Path(directory)
        candidate_id = uuid.uuid4().hex
        candidate_dir = root / "candidates" / candidate_id
        engine = candidate_dir / "model" / "engine"
        categories = {}
        for category in ("token", "modality", "tool"):
            path = engine / "evaluation" / "data" / (uuid.uuid4().hex + ".jsonl")
            atomic_write_json(path, {"fixture": category})
            categories[category] = [{"path": path.relative_to(engine).as_posix(),
                "sha256": _file_sha256(path), "records": observations if category == "token" else 1}]
        manifest_path = engine / "evaluation" / "geometry-holdouts.json"
        atomic_write_json(manifest_path, {"format": "omni-registered-geometry-holdouts", "formatVersion": 1,
            "categories": categories})
        atomic_write_json(engine / "brain.json", {"geometry_holdout_registration": {
            "format": "omni-geometry-holdout-registration", "formatVersion": 1,
            "manifestPath": "evaluation/geometry-holdouts.json", "manifestSha256": _file_sha256(manifest_path)}})
        for filename in ("core.safetensors", "plasticity.safetensors"):
            atomic_save_tensors_bounded(engine / filename, {"fixture": torch.tensor([0x55], dtype=torch.uint8)})
        from omni_core.registered_geometry_holdouts import load_registered_geometry_holdouts
        registered = load_registered_geometry_holdouts(engine)
        policy = Policy(directory, 2**22)
        baselines = root / "evolution-baselines"
        def measure(path, *, after):
            writer = PairedLossWriter(path, policy, observations + 2)
            losses = {}
            for category, data in registered["categories"].items():
                loss = .8 if category == "token" and after and improved else 1.
                for index in range(data["examples"]):
                    writer.add(category, hashlib.sha256((category + str(index)).encode()).hexdigest(), loss, 1)
                losses[category] = {"sourceSha256": data["sourceSha256"], "loss": loss,
                    "examples": data["examples"], "heldOut": True, "trainingOverlapCount": 0}
            return {"format": "omni-native-geometry-holdouts", "formatVersion": 1,
                "benchmarkSha256": registered["benchmarkSha256"], **losses,
                "pairedScores": writer.finish(), "resources": {"withinSelectedEnvelope": True,
                    "peakManagedMemoryBytes": 4096, "peakAcceleratorMemoryBytes": 0}}
        baseline_scores = measure(baselines / (candidate_id + ".paired-baseline.jsonl"), after=False)
        mutation = {"mutation": "grow-experts", "addExperts": 1} if kind == "growth" else None
        baseline = {"architectureMutation": mutation, "objectiveTextFingerprints": [],
            "geometryHoldouts": baseline_scores, "objectives": ["language-prediction"],
            "architecture": {"expertCount": 0}, "resources": {"tensorBytes": 16},
            "baselineObjectiveLoss": 1., "baselineCapabilityLoss": 1., "baselineRetentionLoss": 1.,
            "parentStateChecksum": "a" * 64, "benchmarkSha256": "c" * 64}
        record = {"id": candidate_id, "status": "ready", "candidateType": kind,
            "candidateStateChecksum": "b" * 64, "candidateMetadataSha256": _file_sha256(engine / "brain.json"),
            "preliminaryMetrics": {"candidateObjectiveLoss": .9}}
        candidate = SimpleNamespace(engine_path=engine, device=torch.device("cpu"),
            parameter_checksum=lambda: "d" * 64, close=lambda: None,
            evaluate_isolated_geometry_holdouts=lambda **kwargs: measure(kwargs["score_path"], after=True),
            _resource_readings=lambda: {"diskFreeBytes": 2**35, "availableMemoryBytes": 2**35},
            _ternary_audit=lambda: {"coverage": 1., "violations": []})
        manager = NeuralEvolutionManager.__new__(NeuralEvolutionManager)
        manager.engine_path, manager.baselines_path = root / "live" / "engine", baselines
        manager.brain = SimpleNamespace(brain_id="fixture", device=torch.device("cpu"),
            _record_candidate=lambda _directory, **updates: record.update(updates),
            events=SimpleNamespace(append=lambda *_: None))
        manager._record = lambda _identity: (candidate_dir, record)
        manager._load_baseline = lambda *_: (baseline, [])
        manager._admit_isolated_load = lambda *_: 1
        manager._require_isolated_candidate = lambda *_: None
        manager._capability_loss = lambda *_: 1.
        manager._latent_loss = lambda *_: 1.
        return manager, candidate, candidate_id, candidate_dir, record, baseline

    def evaluate_stub(self, manager, candidate, candidate_id):
        with patch("omni_core.brain.AdaptiveBrain.load", return_value=candidate), \
            patch("omni_core.evolution._bundle_checksum", return_value="b" * 64), \
            patch("omni_core.evolution._architecture_signature", return_value={"expertCount": 0}), \
            patch("omni_core.evolution._architecture_compatible", return_value=True), \
            patch("omni_core.evolution._tensor_resources", return_value={"tensorBytes": 16}):
            return manager.evaluate(candidate_id)

    def test_every_native_kind_requires_and_seals_paired_supported_benefit(self):
        for kind in ("neural", "data", "substrate", "growth"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                manager, candidate, identity, location, record, baseline = self.fixture(directory, kind=kind)
                result = self.evaluate_stub(manager, candidate, identity)
                self.assertTrue(result["passed"])
                self.assertTrue(result["checks"]["statisticallySupportedImprovement"])
                self.assertTrue(result["processMeasurement"]["nativeQualityGainMeasured"])
                self.assertTrue(record["evaluationFileSha256"])
                manager._verify_native_promotion_evidence(identity, location, record, baseline)
                (manager.baselines_path / (identity + ".paired-candidate.jsonl")).write_bytes(b"altered")
                with self.assertRaisesRegex(ValueError, "identity"):
                    manager._verify_native_promotion_evidence(identity, location, record, baseline)
                # Exercise the real public promotion routing, stopping before
                # any live copy. Geometry is absent for all four stub kinds.
                with patch("omni_core.evolution.assert_architecture_quiescent"), self.assertRaisesRegex(ValueError, "identity"):
                    manager.promote(identity)

    def test_ties_insufficient_observations_and_missing_baseline_never_authorize_promotion(self):
        for observations, improved in ((40, False), (1, True)):
            with self.subTest(observations=observations, improved=improved), tempfile.TemporaryDirectory() as directory:
                manager, candidate, identity, _location, record, _baseline = self.fixture(directory,
                    observations=observations, improved=improved)
                result = self.evaluate_stub(manager, candidate, identity)
                self.assertFalse(result["passed"])
                self.assertEqual(record["status"], "rejected")
                with self.assertRaises(ValueError): manager._load_native_evaluation(identity, record)
        with tempfile.TemporaryDirectory() as directory:
            manager, candidate, identity, _location, record, baseline = self.fixture(directory)
            del baseline["geometryHoldouts"]
            with self.assertRaisesRegex(ValueError, "paired baseline"):
                self.evaluate_stub(manager, candidate, identity)
            self.assertEqual(record["status"], "rejected")

    def test_missing_registration_fails_before_candidate_save_load_or_training(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = NeuralEvolutionManager.__new__(NeuralEvolutionManager)
            manager.engine_path = Path(directory)
            atomic_write_json(manager.engine_path / "brain.json", {})
            manager.brain = SimpleNamespace(ingestion_checkpoints={}, distributed_training_seal=None)
            with self.assertRaisesRegex(ValueError, "protected"):
                manager.propose(latent_replay=True)


if __name__ == "__main__": unittest.main()
