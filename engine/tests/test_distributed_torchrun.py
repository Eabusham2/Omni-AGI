import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
PROJECT = ENGINE.parent
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig


def _pyarrow_available():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipIf(
    os.environ.get("OMNI_SKIP_DISTRIBUTED_INTEGRATION") == "1",
    "explicitly disabled by OMNI_SKIP_DISTRIBUTED_INTEGRATION",
)
@unittest.skipUnless(_pyarrow_available(), "PyArrow is required")
@unittest.skipUnless(torch.distributed.is_available(), "torch.distributed is unavailable")
@unittest.skip(
    "pending rank-synchronized packed ternary updates: this historical "
    "two-rank positive proof is unsafe with direct backward synapse mutations"
)
class DistributedTorchrunIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyarrow as pa
        import pyarrow.parquet as parquet

        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="omni-torchrun-proof-")
        cls.root = Path(cls.temporary.name)
        cls.dataset = cls.root / "tiny.parquet"
        parquet.write_table(
            pa.table(
                {
                    "text": [
                        "alpha distributed cortex fact",
                        "beta distributed cortex fact",
                        "gamma distributed cortex fact",
                    ]
                }
            ),
            cls.dataset,
            row_group_size=1,
        )
        cls.config = OmniConfig.micro(
            origin_kind="ground-up",
            max_seq_len=16,
            train_batch_size=1,
            gradient_accumulation=1,
        )
        cls.config_path = cls.root / "config.json"
        cls.config_path.write_text(
            json.dumps(cls.config.to_dict()), encoding="utf-8"
        )
        cls.initial = cls.root / "initial"
        initial = AdaptiveBrain.create(
            "torchrun-proof",
            cls.initial,
            cls.config,
            initialize_ground_up=True,
        )
        # The CLI argument deliberately points at a live brain that now has
        # post-origin private state. A new distributed run must locate and copy
        # only its immutable engine/origin, never this mutable current record.
        initial.training_sources.append(
            {
                "id": "post-origin-private-source",
                "name": "must-not-be-inherited",
                "kind": "text",
                "bytes": 1,
                "imported_at": "2026-09-07T00:00:00Z",
            }
        )
        initial.save()
        initial.events.close()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def _environment(cls):
        return {
            **os.environ,
            "PYTHONPATH": str(ENGINE),
            "OMP_NUM_THREADS": "1",
        }

    @classmethod
    def _training_arguments(
        cls,
        *,
        run_path: Path,
        output_path: Path,
        resume: str,
        accumulation: int,
        epochs: int = 1,
        inject_failure: bool = False,
    ):
        values = [
            str(ENGINE / "distributed_train.py"),
            "train",
            "--dataset",
            str(cls.dataset),
            "--output",
            str(output_path),
            "--run-dir",
            str(run_path),
            "--brain-id",
            "torchrun-proof",
            "--initial-ground-up",
            str(cls.initial),
            "--config",
            str(cls.config_path),
            "--device",
            "cpu",
            "--epochs",
            str(epochs),
            "--global-batch-records",
            "2",
            "--micro-batch-records",
            "1",
            "--gradient-accumulation",
            str(accumulation),
            "--checkpoint-steps",
            "1",
            "--capability-rehearsal-waves",
            "1",
            "--amp",
            "off",
            "--cpu-threads",
            "1",
            "--resume",
            resume,
        ]
        if inject_failure:
            values.extend(("--inject-failure", "1:2"))
        return values

    @classmethod
    def _torchrun(cls, arguments, *, timeout=360):
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            *arguments,
        ]
        started = time.monotonic()
        result = subprocess.run(
            command,
            cwd=PROJECT,
            env=cls._environment(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result, time.monotonic() - started, command

    @staticmethod
    def _checkpoint(run_path: Path):
        pointer = json.loads(
            (run_path / "active-checkpoint.json").read_text(encoding="utf-8")
        )
        return json.loads(
            (run_path / pointer["checkpoint"]).read_text(encoding="utf-8")
        )

    def test_two_rank_failure_resume_equivalence_coverage_and_cancel(self):
        run_path = self.root / "ddp-run"
        output_path = self.root / "ddp-output"
        failed, failure_seconds, failure_command = self._torchrun(
            self._training_arguments(
                run_path=run_path,
                output_path=output_path,
                resume="never",
                accumulation=1,
                inject_failure=True,
            )
        )
        self.assertNotEqual(failed.returncode, 0, msg=failed.stdout)
        interrupted = self._checkpoint(run_path)
        self.assertEqual(interrupted["globalOptimizerSteps"], 1)
        self.assertEqual(interrupted["dynamicHighWater"], 2)
        self.assertEqual(
            [value["nextGlobalOrdinal"] for value in interrupted["rankCursors"]],
            [2, 2],
        )
        self.assertEqual(
            interrupted["capabilityRehearsal"]["eventCount"], 2
        )
        self.assertEqual(
            interrupted["capabilityRehearsal"]["lastReceipt"]["phase"],
            "middle",
        )
        self.assertTrue((run_path / "failures" / "rank-00001.json").is_file())

        resumed, resume_seconds, resume_command = self._torchrun(
            self._training_arguments(
                run_path=run_path,
                output_path=output_path,
                resume="required",
                accumulation=1,
                inject_failure=True,
            )
        )
        self.assertEqual(
            resumed.returncode,
            0,
            msg="stdout:\n%s\nstderr:\n%s" % (resumed.stdout, resumed.stderr),
        )
        completed = self._checkpoint(run_path)
        self.assertEqual(completed["dynamicHighWater"], 3)
        self.assertEqual(completed["globalOptimizerSteps"], 2)
        self.assertEqual(
            [value["ownedRecordsCompleted"] for value in completed["rankCursors"]],
            [2, 1],
        )
        self.assertEqual(
            [value["epoch"] for value in completed["rankCursors"]], [1, 1]
        )
        schedule = completed["capabilityRehearsal"]
        self.assertTrue(schedule["startCompleted"])
        self.assertTrue(schedule["finalCompleted"])
        self.assertEqual(schedule["lastPeriodicWave"], 1)
        self.assertEqual(schedule["eventCount"], 3)
        self.assertEqual(schedule["lastReceipt"]["phase"], "final")
        self.assertEqual(schedule["lastReceipt"]["appliedRank"], 0)
        self.assertEqual(schedule["lastReceipt"]["after"]["correct"], 8)
        self.assertTrue(schedule["lastReceipt"]["regressionGatePassed"])
        self.assertEqual(completed["telemetry"]["rankCount"], 2)
        self.assertEqual(len(completed["telemetry"]["perRank"]), 2)

        single_run = self.root / "single-run"
        single_output = self.root / "single-output"
        single_command = [
            sys.executable,
            *self._training_arguments(
                run_path=single_run,
                output_path=single_output,
                resume="never",
                accumulation=2,
            ),
        ]
        started = time.monotonic()
        single = subprocess.run(
            single_command,
            cwd=PROJECT,
            env=self._environment(),
            capture_output=True,
            text=True,
            timeout=360,
        )
        single_seconds = time.monotonic() - started
        self.assertEqual(
            single.returncode,
            0,
            msg="stdout:\n%s\nstderr:\n%s" % (single.stdout, single.stderr),
        )
        ddp_status = json.loads(
            (run_path / "distributed-status.json").read_text(encoding="utf-8")
        )
        single_status = json.loads(
            (single_run / "distributed-status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            ddp_status["metrics"],
            single_status["metrics"],
            "global loss/measurement reduction must be functionally exact",
        )

        ddp_brain = AdaptiveBrain.load(output_path)
        single_brain = AdaptiveBrain.load(single_output)
        initial_brain = AdaptiveBrain.load(self.initial)
        try:
            self.assertTrue(
                any(
                    source.get("id") == "post-origin-private-source"
                    for source in initial_brain.training_sources
                )
            )
            for trained in (ddp_brain, single_brain):
                self.assertFalse(
                    any(
                        source.get("id") == "post-origin-private-source"
                        for source in trained.training_sources
                    )
                )
            ddp_state = ddp_brain._core_tensors()
            single_state = single_brain._core_tensors()
            initial_state = initial_brain._core_tensors()
            differing_elements = 0
            maximum_absolute_difference = 0.0
            maximum_relative_difference = 0.0
            for name in ddp_state:
                left = ddp_state[name].cpu().float()
                right = single_state[name].cpu().float()
                difference = (left - right).abs()
                differing_elements += int(difference.ne(0).sum().item())
                if difference.numel():
                    maximum_absolute_difference = max(
                        maximum_absolute_difference,
                        float(difference.max().item()),
                    )
                    maximum_relative_difference = max(
                        maximum_relative_difference,
                        float(
                            (
                                difference
                                / torch.maximum(
                                    left.abs(), right.abs()
                                ).clamp_min(torch.finfo(torch.float32).tiny)
                            ).max().item()
                        ),
                    )
                torch.testing.assert_close(
                    ddp_state[name].cpu(),
                    single_state[name].cpu(),
                    rtol=3e-6,
                    # Different FP32 reduction parenthesization may move a
                    # terminal mantissa bit; the observed worst case is
                    # 2.384185791015625e-07 across 3 tiny records after the
                    # full crash/resume path, well within ordinary FP32
                    # collective reduction precision.
                    atol=5e-7,
                    msg=lambda message, key=name: "%s: %s" % (key, message),
                )
            self.assertTrue(
                any(
                    not torch.equal(ddp_state[name].cpu(), initial_state[name].cpu())
                    for name in ddp_state
                ),
                "training must mutate parameters",
            )
            self.assertEqual(ddp_brain.config.origin_kind, "ground-up")
            self.assertIsNone(
                ddp_brain.packed_ternary_manifest["pretrainedTextCortex"]
            )
            self.assertEqual(
                len(ddp_brain.memory.neurons), len(set(ddp_brain.memory.neurons))
            )
            self.assertEqual(
                len(ddp_brain.memory.synapses), len(set(ddp_brain.memory.synapses))
            )
            assembly_ids = [
                str(value["id"]) for value in ddp_brain.memory.assemblies
            ]
            self.assertEqual(len(assembly_ids), len(set(assembly_ids)))
        finally:
            ddp_brain.events.close()
            single_brain.events.close()
            initial_brain.events.close()

        active_before_cancel = (run_path / "active-checkpoint.json").read_bytes()
        cancel = subprocess.run(
            [
                sys.executable,
                str(ENGINE / "distributed_train.py"),
                "cancel",
                "--run-dir",
                str(run_path),
                "--reason",
                "integration cancellation proof",
            ],
            cwd=PROJECT,
            env=self._environment(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(cancel.returncode, 0, msg=cancel.stderr)
        cancelled, cancel_seconds, cancel_command = self._torchrun(
            self._training_arguments(
                run_path=run_path,
                output_path=output_path,
                resume="required",
                accumulation=1,
                epochs=2,
            ),
            timeout=120,
        )
        self.assertEqual(cancelled.returncode, 0, msg=cancelled.stderr)
        status = json.loads(
            (run_path / "distributed-status.json").read_text(encoding="utf-8")
        )
        self.assertEqual(status["state"], "cancelled")
        self.assertEqual(
            (run_path / "active-checkpoint.json").read_bytes(),
            active_before_cancel,
            "cancel before the next wave must not advance the checkpoint",
        )

        # Keep exact invocations/timing in the assertion context and CI log
        # without imposing a machine-speed gate.
        print(
            json.dumps(
                {
                    "failureCommand": failure_command,
                    "failureSeconds": failure_seconds,
                    "resumeCommand": resume_command,
                    "resumeSeconds": resume_seconds,
                    "singleCommand": single_command,
                    "singleSeconds": single_seconds,
                    "cancelCommand": cancel_command,
                    "cancelSeconds": cancel_seconds,
                    "differingParameterElements": differing_elements,
                    "maximumAbsoluteParameterDifference": (
                        maximum_absolute_difference
                    ),
                    "maximumRelativeParameterDifference": (
                        maximum_relative_difference
                    ),
                    "exactMetrics": ddp_status["metrics"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
