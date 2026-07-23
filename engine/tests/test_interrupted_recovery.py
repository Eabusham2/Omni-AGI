import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig


class InterruptedCandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-interrupted-candidate-"
        )
        self.root = Path(self.temporary.name) / "brain"
        config = OmniConfig.micro(
            parallel_thoughts=1,
            max_seq_len=40,
            learn_from_own_messages=False,
        )
        brain = AdaptiveBrain.create("interrupted-brain", self.root, config)
        self.stable_checksum = brain.parameter_checksum()
        self.stable_steps = brain.counters["training_steps"]
        brain.events.close()

    def tearDown(self):
        self.temporary.cleanup()

    def test_killed_json_rpc_worker_quarantines_candidate_and_reloads_stable_state(
        self,
    ):
        request = {
            "jsonrpc": "2.0",
            "id": "kill-during-training",
            "method": "train",
            "params": {
                "brainId": "interrupted-brain",
                "storagePath": str(self.root),
                "texts": [
                    "Interrupted candidate fixture repeats long enough to keep "
                    "the supervised worker inside slow neural optimization."
                ],
                "epochs": 10000,
                "jobId": "interrupted-job",
            },
        }
        process = subprocess.Popen(
            [sys.executable, "-u", str(ENGINE / "worker.py")],
            cwd=str(ENGINE),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdin is not None
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            deadline = time.monotonic() + 60.0
            candidate_record = None
            while time.monotonic() < deadline:
                records = list(
                    (self.root / "engine" / "candidates").glob(
                        "*/candidate.json"
                    )
                )
                if records:
                    record = json.loads(records[0].read_text("utf-8"))
                    if record.get("status") == "training":
                        candidate_record = records[0]
                        break
                if process.poll() is not None:
                    stderr = process.stderr.read() if process.stderr else ""
                    self.fail(
                        "worker exited before candidate training began: %s"
                        % stderr[-2000:]
                    )
                time.sleep(0.025)
            self.assertIsNotNone(
                candidate_record, "worker did not persist a candidate phase"
            )
            time.sleep(0.1)
            self.assertIsNone(process.poll())
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            if process.stdin is not None:
                process.stdin.close()
            if process.stderr is not None:
                process.stderr.close()

        recovered = AdaptiveBrain.load(
            self.root, expected_brain_id="interrupted-brain"
        )
        self.assertEqual(recovered.parameter_checksum(), self.stable_checksum)
        self.assertEqual(
            recovered.counters["training_steps"], self.stable_steps
        )
        assert candidate_record is not None
        record = json.loads(candidate_record.read_text("utf-8"))
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(record["previousStatus"], "training")
        self.assertFalse(record["stableCheckpointRestored"])
        recovery_events = [
            event
            for event in recovered.events.recent(50)
            if event["kind"] == "candidate-recovered"
        ]
        self.assertEqual(len(recovery_events), 1)
        self.assertEqual(
            recovery_events[0]["payload"]["candidateId"],
            candidate_record.parent.name,
        )
        unsafe_extensions = {".pt", ".pth", ".pkl", ".pickle"}
        self.assertFalse(
            [
                path
                for path in (self.root / "engine").rglob("*")
                if path.suffix.lower() in unsafe_extensions
            ]
        )
        recovered.events.close()

    def test_interrupted_promotion_restores_complete_pre_candidate_checkpoint(
        self,
    ):
        brain = AdaptiveBrain.load(
            self.root, expected_brain_id="interrupted-brain"
        )
        candidate_id, candidate_dir = brain._begin_candidate("test-promotion")
        brain._record_candidate(candidate_dir, status="promoting")
        with torch.no_grad():
            next(brain.decoder.parameters()).add_(25.0)
        brain.save()
        self.assertNotEqual(brain.parameter_checksum(), self.stable_checksum)
        brain.events.close()

        recovered = AdaptiveBrain.load(
            self.root, expected_brain_id="interrupted-brain"
        )
        self.assertEqual(recovered.parameter_checksum(), self.stable_checksum)
        record = json.loads(
            (
                self.root
                / "engine"
                / "candidates"
                / candidate_id
                / "candidate.json"
            ).read_text("utf-8")
        )
        self.assertEqual(record["status"], "interrupted")
        self.assertEqual(record["previousStatus"], "promoting")
        self.assertTrue(record["stableCheckpointRestored"])
        recovered.events.close()


if __name__ == "__main__":
    unittest.main()
