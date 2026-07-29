import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from safetensors.torch import load_file


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.evolution import NeuralEvolutionManager, _bundle_checksum
from worker import Worker


class NeuralEvolutionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(113)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-neural-evolution-"
        )
        self.root = Path(self.temporary.name)
        self.brain = AdaptiveBrain.create(
            "evolution-brain",
            self.root,
            OmniConfig.micro(
                origin_kind="blank",
                parallel_thoughts=1,
                max_seq_len=40,
                learn_from_own_messages=False,
            ),
        )
        self.manager = NeuralEvolutionManager(self.brain)

    def tearDown(self):
        try:
            self.brain.events.close()
        except Exception:
            pass
        self.temporary.cleanup()

    def propose(self):
        return self.manager.propose(
            texts=[
                "A neural candidate learns this exact causal amber sentence."
            ],
            epochs=2,
            learning_rate=0.003,
            objectives=["language-prediction", "retention", "capability"],
            provenance={
                "reason": "focused test",
                "source": "project-authored fixture",
            },
        )

    def test_proposal_is_isolated_and_uses_only_safe_tensor_checkpoints(self):
        before_parameter = self.brain.parameter_checksum()
        before_state = _bundle_checksum(self.brain.engine_path)
        result = self.propose()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.brain.parameter_checksum(), before_parameter)
        self.assertEqual(
            _bundle_checksum(self.brain.engine_path),
            before_state,
        )
        candidate = (
            self.brain.engine_path / "candidates" / result["id"] / "model" / "engine"
        )
        self.assertTrue(load_file(str(candidate / "core.safetensors")))
        self.assertTrue(load_file(str(candidate / "plasticity.safetensors")))
        unsafe = {
            ".ckpt",
            ".joblib",
            ".pkl",
            ".pickle",
            ".pt",
            ".pth",
        }
        self.assertFalse(
            [
                path
                for path in (self.brain.engine_path / "candidates").rglob("*")
                if path.is_file() and path.suffix.lower() in unsafe
            ]
        )
        self.assertEqual(
            result["parentParameterChecksum"], before_parameter
        )
        self.assertNotEqual(
            result["parentStateChecksum"], result["candidateStateChecksum"]
        )
        self.assertTrue(result["candidateDiffSha256"])
        self.assertIn("resources", result)
        self.assertEqual(
            result["architectureCandidate"]["supported"], False
        )

    def test_improving_candidate_evaluates_promotes_and_rolls_back(self):
        before = self.brain.parameter_checksum()
        result = self.propose()
        evaluation = self.manager.evaluate(result["id"])
        self.assertTrue(evaluation["passed"])
        self.assertTrue(all(evaluation["checks"].values()))
        promotion = self.manager.promote(result["id"])
        self.assertTrue(promotion["promoted"])
        promoted = AdaptiveBrain.load(
            self.root, expected_brain_id=self.brain.brain_id
        )
        self.assertNotEqual(promoted.parameter_checksum(), before)
        promoted.events.close()

        rollback = self.manager.rollback(result["id"])
        self.assertTrue(rollback["rolledBack"])
        restored = AdaptiveBrain.load(
            self.root, expected_brain_id=self.brain.brain_id
        )
        self.assertEqual(restored.parameter_checksum(), before)
        restored.events.close()
        record = json.loads(
            (
                self.brain.engine_path
                / "candidates"
                / result["id"]
                / "candidate.json"
            ).read_text("utf-8")
        )
        self.assertEqual(record["status"], "rolled-back")

    def test_regression_is_rejected_by_immutable_capability_gate(self):
        result = self.propose()
        baseline_path = (
            self.brain.engine_path
            / "evolution-baselines"
            / (result["id"] + ".json")
        )
        baseline = json.loads(baseline_path.read_text("utf-8"))
        regressed = baseline["baselineCapabilityLoss"] * 100.0 + 1.0
        with mock.patch.object(
            self.manager, "_capability_loss", return_value=regressed
        ):
            evaluation = self.manager.evaluate(result["id"])
        self.assertFalse(evaluation["passed"])
        self.assertFalse(evaluation["checks"]["capabilityRetention"])
        self.assertIn("capabilityRetention", evaluation["failures"])
        with self.assertRaisesRegex(ValueError, "pass evaluation"):
            self.manager.promote(result["id"])

    def test_promotion_rejects_a_stale_live_baseline(self):
        result = self.propose()
        self.assertTrue(self.manager.evaluate(result["id"])["passed"])
        with torch.no_grad():
            next(self.brain.decoder.parameters()).add_(0.125)
        self.brain.save()
        changed = self.brain.parameter_checksum()
        with self.assertRaisesRegex(ValueError, "baseline is stale"):
            self.manager.promote(result["id"])
        self.assertEqual(self.brain.parameter_checksum(), changed)
        record = json.loads(
            (
                self.brain.engine_path
                / "candidates"
                / result["id"]
                / "candidate.json"
            ).read_text("utf-8")
        )
        self.assertEqual(record["status"], "stale")

    def test_interrupted_promotion_recovers_complete_parent_checkpoint(self):
        parent_checksum = self.brain.parameter_checksum()
        result = self.propose()
        self.assertTrue(self.manager.evaluate(result["id"])["passed"])
        candidate_dir = (
            self.brain.engine_path / "candidates" / result["id"]
        )
        self.brain._record_candidate(
            candidate_dir,
            status="promoting",
            operation="promote",
        )
        # Simulate process death after only the first of the three atomic
        # checkpoint replacements.
        shutil.copy2(
            str(candidate_dir / "model" / "engine" / "core.safetensors"),
            str(self.brain.engine_path / "core.safetensors"),
        )
        self.brain.events.close()
        recovered = AdaptiveBrain.load(
            self.root, expected_brain_id=self.brain.brain_id
        )
        self.assertEqual(recovered.parameter_checksum(), parent_checksum)
        record = json.loads(
            (candidate_dir / "candidate.json").read_text("utf-8")
        )
        self.assertEqual(record["status"], "interrupted")
        self.assertTrue(record["stableCheckpointRestored"])
        recovered.events.close()

    def test_compatible_expert_growth_architecture_promotes_and_rolls_back(self):
        before_checksum = self.brain.parameter_checksum()
        before_experts = self.brain.decoder.expert_count
        result = self.manager.propose(
            texts=[
                "A compatible residual expert learns this amber architecture fact."
            ],
            epochs=2,
            architecture_change={
                "mutation": "grow-experts",
                "addExperts": 1,
            },
            provenance={"reason": "focused compatible architecture test"},
        )
        self.assertEqual(result["candidateType"], "architecture")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(self.brain.decoder.expert_count, before_experts)
        mutation = result["architectureMutation"]
        self.assertEqual(mutation["mutation"], "grow-experts")
        self.assertEqual(mutation["expertCountBefore"], before_experts)
        self.assertEqual(mutation["expertCountAfter"], before_experts + 1)
        self.assertGreater(mutation["estimatedGrowthBytes"], 0)

        evaluation = self.manager.evaluate(result["id"])
        self.assertTrue(evaluation["passed"], evaluation["failures"])
        self.assertTrue(evaluation["checks"]["architectureCompatible"])
        self.assertEqual(
            evaluation["metrics"]["expertCountAfter"], before_experts + 1
        )
        promotion = self.manager.promote(result["id"])
        self.assertEqual(promotion["candidateType"], "architecture")
        self.assertTrue(promotion["promoted"])
        promoted = AdaptiveBrain.load(
            self.root, expected_brain_id=self.brain.brain_id
        )
        self.assertEqual(promoted.decoder.expert_count, before_experts + 1)
        self.assertNotEqual(promoted.parameter_checksum(), before_checksum)
        promoted.events.close()

        rollback = self.manager.rollback(result["id"])
        self.assertTrue(rollback["rolledBack"])
        restored = AdaptiveBrain.load(
            self.root, expected_brain_id=self.brain.brain_id
        )
        self.assertEqual(restored.decoder.expert_count, before_experts)
        self.assertEqual(restored.parameter_checksum(), before_checksum)
        restored.events.close()

    def test_arbitrary_architecture_shape_mutation_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, "only the compatible grow-experts"
        ):
            self.manager.propose(
                texts=["unsafe width change"],
                architecture_change={
                    "mutation": "change-width",
                    "dModel": 1024,
                },
            )

    def test_blank_brain_can_add_function_preserving_expert_without_replay(self):
        self.assertEqual(len(self.brain.replay), 0)
        result = self.manager.propose(
            architecture_change={
                "mutation": "grow-experts",
                "addExperts": 1,
            }
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(
            result["training"]["mode"],
            "function-preserving-architecture-insertion",
        )
        evaluation = self.manager.evaluate(result["id"])
        self.assertTrue(evaluation["passed"], evaluation["failures"])

    def test_worker_exposes_complete_neural_candidate_rpc_surface(self):
        worker = Worker()
        worker.brains[self.brain.brain_id] = self.brain
        canonical = {
            "evolution.propose",
            "evolution.evaluate",
            "evolution.list",
            "evolution.promote",
            "evolution.reject",
            "evolution.rollback",
        }
        self.assertTrue(canonical.issubset(worker.methods))
        listed = worker.evolution_list(
            {
                "brainId": self.brain.brain_id,
                "storagePath": str(self.root),
            },
            "list-neural-candidates",
        )
        self.assertEqual(listed["candidates"], [])
        with contextlib.redirect_stdout(io.StringIO()):
            architecture = worker.evolution_propose(
                {
                    "brainId": self.brain.brain_id,
                    "storagePath": str(self.root),
                    "texts": [
                        "The worker safely grows one compatible ternary expert."
                    ],
                    "epochs": 1,
                    "architectureChange": {
                        "mutation": "grow-experts",
                        "addExperts": 1,
                    },
                },
                "compatible-architecture-candidate",
            )
        self.assertEqual(architecture["candidateType"], "architecture")
        self.assertEqual(
            architecture["architectureMutation"]["mutation"],
            "grow-experts",
        )


if __name__ == "__main__":
    unittest.main()
