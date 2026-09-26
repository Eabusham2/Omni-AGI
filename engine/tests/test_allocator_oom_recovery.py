import contextlib
import errno
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.brain import is_allocator_oom_error
from omni_core.offload import NeuralStateResourcePause
from worker import RpcFault, Worker


class AllocatorOomRecoveryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(409)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-oom-")
        self.root = Path(self.temporary.name) / "brain"

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, **overrides):
        return AdaptiveBrain(
            "oom-brain",
            self.root,
            OmniConfig.micro(
                learn_from_own_messages=False,
                **overrides,
            ),
        )

    @staticmethod
    def measured():
        return {
            "loss": 1.0,
            "language_loss": 1.0,
            "idea_loss": 0.0,
            "workspace_loss": 0.0,
            "stability_loss": 0.0,
        }

    def test_classifier_covers_allocator_backends_but_not_unrelated_errors(self):
        self.assertTrue(is_allocator_oom_error(MemoryError()))
        self.assertTrue(
            is_allocator_oom_error(
                RuntimeError("MPS backend out of memory (MPS allocated: 9 GB)")
            )
        )
        self.assertTrue(
            is_allocator_oom_error(
                RuntimeError("DefaultCPUAllocator: can't allocate memory")
            )
        )
        self.assertTrue(
            is_allocator_oom_error(OSError(errno.ENOMEM, "localized failure"))
        )
        self.assertFalse(
            is_allocator_oom_error(RuntimeError("dataset schema is invalid"))
        )
        self.assertFalse(
            is_allocator_oom_error(
                ValueError("the 'out of memory' dataset label is invalid")
            )
        )
        self.assertFalse(
            is_allocator_oom_error(
                RuntimeError("dataset allocation failed validation")
            )
        )
        self.assertFalse(
            is_allocator_oom_error(
                RuntimeError("DefaultCPUAllocator configuration is invalid")
            )
        )
        try:
            try:
                raise MemoryError("discarded allocator context")
            except MemoryError:
                raise RuntimeError("dataset schema is invalid") from None
        except RuntimeError as suppressed:
            self.assertFalse(is_allocator_oom_error(suppressed))
        try:
            raise RuntimeError("wrapped training failure") from MemoryError()
        except RuntimeError as wrapped:
            self.assertTrue(is_allocator_oom_error(wrapped))

    def test_uncommitted_gradient_batch_retries_with_smaller_ram_microbatch(self):
        brain = self.make_brain(
            max_seq_len=64,
            train_batch_size=4,
            gradient_accumulation=2,
        )
        trainable = next(
            parameter
            for parameter in brain.decoder.parameters()
            if parameter.requires_grad
        )
        batch_sizes = []

        def synthetic_loss(encoded, vectors):
            del vectors
            batch_sizes.append(len(encoded))
            if len(batch_sizes) == 1:
                raise RuntimeError("CUDA out of memory")
            loss = trainable.reshape(-1)[0] * 0.0 + 1.0
            return loss, self.measured()

        experiences = [
            (
                "record-%d" % index,
                torch.ones(brain.config.vsa_dim),
            )
            for index in range(4)
        ]
        with patch.object(
            brain,
            "_experience_ids_batch_loss",
            side_effect=synthetic_loss,
        ), patch.object(
            brain,
            "_maintain_neural_state_resources",
            return_value={},
        ):
            result = brain._optimize_streaming_experience_batch(experiences)

        self.assertEqual(batch_sizes, [4, 2, 2])
        self.assertEqual(brain._runtime_train_batch_size, 2)
        self.assertEqual(result["records"], 4.0)
        self.assertEqual(result["optimizer_steps"], 1.0)
        brain.events.close()

    def test_scheduled_retry_selects_a_valid_lower_divisor_before_mutation(self):
        brain = self.make_brain(
            max_seq_len=64,
            train_batch_size=5,
            gradient_accumulation=1,
            training_resource_mode="manual",
        )
        trainable = next(
            parameter
            for parameter in brain.decoder.parameters()
            if parameter.requires_grad
        )
        initial = trainable.detach().clone()
        cases = (
            (
                "manual-prime-target",
                "manual",
                5,
                1,
                5,
                1,
                5,
                1,
                5,
                [5, 1, 1, 1, 1, 1],
            ),
            (
                "auto-composite-target",
                "auto",
                2,
                3,
                3,
                2,
                6,
                2,
                3,
                [3, 2, 2, 2],
            ),
        )

        for case_index, (
            label,
            resource_mode,
            configured_batch,
            configured_accumulation,
            initial_physical,
            accumulation,
            record_count,
            expected_physical,
            expected_accumulation,
            expected_batch_sizes,
        ) in enumerate(cases):
            with self.subTest(policy=label):
                brain.config.training_resource_mode = resource_mode
                brain.config.train_batch_size = configured_batch
                brain.config.gradient_accumulation = (
                    configured_accumulation
                )
                with torch.no_grad():
                    trainable.copy_(initial)
                brain._optimizer = torch.optim.SGD([trainable], lr=0.01)
                batch_sizes = []
                draws = []
                processed = []
                parameter_snapshots = []
                allocator_failed = False

                def synthetic_loss(encoded, vectors):
                    nonlocal allocator_failed
                    batch_sizes.append(len(encoded))
                    draws.append(float(torch.rand(()).item()))
                    parameter_snapshots.append(trainable.detach().clone())
                    if not allocator_failed:
                        allocator_failed = True
                        raise RuntimeError("CUDA out of memory")
                    processed.extend(int(vector[0].item()) for vector in vectors)
                    mean_value = torch.stack(
                        [vector[0] for vector in vectors]
                    ).to(trainable.device).mean()
                    return (
                        trainable.reshape(-1)[0] * mean_value,
                        self.measured(),
                    )

                experiences = [
                    (
                        "record-%d" % value,
                        torch.full(
                            (brain.config.vsa_dim,), float(value)
                        ),
                    )
                    for value in range(1, record_count + 1)
                ]
                schedule = {
                    "physicalBatchRecords": initial_physical,
                    "gradientAccumulation": accumulation,
                    "trainingSequenceTokens": 64,
                }
                torch.manual_seed(1_900 + case_index)
                original_step = brain._optimizer.step
                with patch.object(
                    brain,
                    "_training_resource_plan",
                    return_value={"pauseBeforeStep": False},
                ), patch.object(
                    brain,
                    "_experience_ids_batch_loss",
                    side_effect=synthetic_loss,
                ), patch.object(
                    brain,
                    "_accumulate_slow_importance",
                ), patch.object(
                    brain,
                    "_commit_slow_anchors",
                ), patch.object(
                    brain,
                    "_canonicalize_streaming_learning_state",
                    return_value=0.0,
                ), patch.object(
                    brain,
                    "_maintain_neural_state_resources",
                    return_value={},
                ), patch.object(
                    brain._optimizer,
                    "step",
                    wraps=original_step,
                ) as optimizer_step:
                    report = brain._optimize_streaming_experience_batch(
                        experiences,
                        learning_schedule=schedule,
                        schedule_locked=False,
                    )

                logical_target = initial_physical * accumulation
                self.assertEqual(
                    configured_batch * configured_accumulation,
                    logical_target,
                )
                self.assertEqual(logical_target, record_count)
                self.assertEqual(
                    logical_target % int(report["physical_batch_records"]),
                    0,
                )
                self.assertEqual(
                    int(report["physical_batch_records"]),
                    expected_physical,
                )
                recomputed_accumulation = (
                    logical_target
                    // int(report["physical_batch_records"])
                )
                self.assertEqual(
                    recomputed_accumulation,
                    expected_accumulation,
                )
                self.assertEqual(
                    int(report["physical_batch_records"])
                    * recomputed_accumulation,
                    logical_target,
                )
                self.assertEqual(batch_sizes, expected_batch_sizes)
                self.assertEqual(processed, list(range(1, record_count + 1)))
                self.assertEqual(draws[0], draws[1])
                self.assertTrue(
                    all(
                        torch.equal(snapshot, initial)
                        for snapshot in parameter_snapshots
                    )
                )
                optimizer_step.assert_called_once_with()
                self.assertFalse(torch.equal(trainable.detach(), initial))

        brain.events.close()

    def test_uneven_microbatches_normalize_by_windows_not_batch_count(self):
        brain = self.make_brain(
            max_seq_len=64,
            train_batch_size=4,
            gradient_accumulation=1,
        )
        trainable = next(
            parameter
            for parameter in brain.decoder.parameters()
            if parameter.requires_grad
        )
        initial = trainable.detach().clone()
        experiences = [
            (
                "record-%d" % index,
                torch.full((brain.config.vsa_dim,), float(index)),
            )
            for index in range(1, 6)
        ]

        def synthetic_loss(encoded, vectors):
            del encoded
            mean_value = torch.stack(
                [vector[0] for vector in vectors]
            ).to(trainable.device).mean()
            loss = trainable.reshape(-1)[0] * mean_value
            measured = self.measured()
            measured["loss"] = float(mean_value)
            measured["language_loss"] = float(mean_value)
            return loss, measured

        def run(physical_batch):
            with torch.no_grad():
                trainable.copy_(initial)
            brain._runtime_train_batch_size = physical_batch
            brain._optimizer = torch.optim.SGD([trainable], lr=1.0)
            with patch.object(
                brain,
                "_training_resource_plan",
                return_value={"pauseBeforeStep": False},
            ), patch.object(
                brain,
                "_experience_ids_batch_loss",
                side_effect=synthetic_loss,
            ), patch.object(
                brain,
                "_accumulate_slow_importance",
            ), patch.object(
                brain,
                "_commit_slow_anchors",
            ), patch.object(
                brain,
                "_maintain_neural_state_resources",
                return_value={},
            ), patch(
                "torch.nn.utils.clip_grad_norm_",
                return_value=torch.tensor(0.0),
            ):
                report = brain._optimize_streaming_experience_batch(experiences)
            update = float(
                initial.reshape(-1)[0] - trainable.detach().reshape(-1)[0]
            )
            return update, report

        full_update, full_report = run(5)
        uneven_update, uneven_report = run(4)
        split_update, split_report = run(2)
        for update in (full_update, uneven_update, split_update):
            self.assertAlmostEqual(update, 3.0, places=5)
        for report in (full_report, uneven_report, split_report):
            self.assertAlmostEqual(report["loss"], 3.0, places=6)
        brain.events.close()

    def test_forward_retry_restores_rng_and_replays_every_resized_window(self):
        brain = self.make_brain(
            max_seq_len=16,
            train_batch_size=1,
            gradient_accumulation=1,
        )
        trainable = next(
            parameter
            for parameter in brain.decoder.parameters()
            if parameter.requires_grad
        )
        text = "abcdefghijklmnopqrstuvwxyz0123456789"
        draws = []
        replayed_text = []

        def synthetic_loss(encoded, vectors):
            del vectors
            draws.append(float(torch.rand(()).item()))
            if len(draws) == 1:
                raise RuntimeError("CUDA out of memory")
            replayed_text.extend(
                brain.tokenizer.decode(ids.tolist()) for ids in encoded
            )
            loss = trainable.reshape(-1)[0] * 0.0 + 1.0
            return loss, self.measured()

        with patch.object(
            brain,
            "_training_resource_plan",
            return_value={"pauseBeforeStep": False},
        ), patch.object(
            brain,
            "_experience_ids_batch_loss",
            side_effect=synthetic_loss,
        ), patch.object(
            brain,
            "_maintain_neural_state_resources",
            return_value={},
        ):
            result = brain._optimize_streaming_experience_batch(
                [(text, torch.ones(brain.config.vsa_dim))]
            )

        self.assertEqual(draws[0], draws[1])
        self.assertEqual("".join(replayed_text), text)
        self.assertEqual(brain._runtime_training_max_seq_len, 8)
        self.assertEqual(result["records"], 1.0)
        brain.events.close()

    def test_successful_retry_clears_only_allocator_pause(self):
        brain = self.make_brain(max_seq_len=8, train_batch_size=1)
        allocator_pause = {
            "mode": "allocator-oom-checkpoint-recovery",
            "oomCount": 1,
            "recoveryPlan": {
                "physicalBatchSize": 1,
                "sequenceTokens": 8,
            },
        }
        brain.apply_allocator_oom_downgrade(allocator_pause)
        self.assertIsNotNone(brain.resource_pause)
        trainable = next(
            parameter
            for parameter in brain.decoder.parameters()
            if parameter.requires_grad
        )

        def synthetic_loss(encoded, vectors):
            del encoded, vectors
            return trainable.reshape(-1)[0] * 0.0 + 1.0, self.measured()

        with patch.object(
            brain,
            "_training_resource_plan",
            return_value={"pauseBeforeStep": False},
        ), patch.object(
            brain,
            "_experience_ids_batch_loss",
            side_effect=synthetic_loss,
        ), patch.object(
            brain,
            "_maintain_neural_state_resources",
            return_value={},
        ):
            brain._optimize_streaming_experience_batch(
                [("one", torch.ones(brain.config.vsa_dim))]
            )
        self.assertIsNone(brain.resource_pause)

        disk_pause = {
            "reason": "disk reserve",
            "readings": {"mode": "transactional-disk-backed"},
        }
        brain.resource_pause = disk_pause
        brain._clear_allocator_recovery_pause()
        self.assertIs(brain.resource_pause, disk_pause)
        brain.events.close()

    def test_minimum_microbatch_raises_recoverable_checkpoint_pause(self):
        brain = self.make_brain(max_seq_len=8, train_batch_size=1)
        before_checksum = brain.parameter_checksum()
        before_steps = brain.counters["training_steps"]
        with patch.object(
            brain,
            "_experience_ids_batch_loss",
            side_effect=RuntimeError("MPS backend out of memory"),
        ):
            with self.assertRaises(NeuralStateResourcePause) as raised:
                brain._optimize_streaming_experience_batch(
                    [("one", torch.ones(brain.config.vsa_dim))]
                )

        status = raised.exception.status
        self.assertTrue(status["allocatorOutOfMemory"])
        self.assertTrue(status["recoverable"])
        self.assertTrue(status["rollbackRequired"])
        self.assertTrue(status["resumeFromLastCheckpoint"])
        self.assertFalse(status["sourceRecordsSkipped"])
        self.assertFalse(status["scratchWriteAttempted"])
        self.assertEqual(status["recoveryPlan"]["physicalBatchSize"], 1)
        self.assertEqual(status["recoveryPlan"]["sequenceTokens"], 8)
        self.assertEqual(brain.parameter_checksum(), before_checksum)
        self.assertEqual(brain.counters["training_steps"], before_steps)
        brain.events.close()

    def test_worker_restores_atomic_generation_and_marks_oom_job_resumable(self):
        worker = Worker()
        partial = MagicMock()
        partial.brain_id = "oom-rollback"
        partial.storage_path = Path("/tmp/omni-oom-rollback")
        partial.events = MagicMock()
        status = {
            "allocatorOutOfMemory": True,
            "recoverable": True,
            "rollbackRequired": True,
            "resumeFromLastCheckpoint": True,
            "oomCount": 1,
            "recoveryPlan": {
                "physicalBatchSize": 1,
                "sequenceTokens": 32,
            },
        }
        partial.ingest.side_effect = NeuralStateResourcePause(
            "allocator paused", status
        )
        restored = MagicMock()
        restored.brain_id = partial.brain_id
        restored.storage_path = partial.storage_path
        restored.events = MagicMock()
        restored.apply_allocator_oom_downgrade.return_value = {
            "physicalBatchSize": 1,
            "sequenceTokens": 32,
        }
        restored.ingestion_checkpoints = {
            "source": {
                "transactionId": "a" * 64,
                "contentHash": "b" * 64,
                "epoch": 0,
                "policy": "pretrain",
                "committedRecords": 512,
                "visitedRecords": 512,
                "commitSequence": 1,
            }
        }
        worker.brains[partial.brain_id] = partial

        with patch.object(worker, "_get", return_value=partial), patch(
            "worker.AdaptiveBrain.load", return_value=restored
        ) as load, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RpcFault) as raised:
                worker.ingest(
                    {
                        "brainId": partial.brain_id,
                        "text": "uncommitted suffix",
                        "jobId": "oom-job",
                    },
                    "oom-request",
                )

        self.assertEqual(raised.exception.code, -32020)
        self.assertTrue(raised.exception.data["recoverable"])
        self.assertEqual(
            raised.exception.data["activeCheckpoints"][0]["committedRecords"],
            512,
        )
        # Worker eviction owns the complete brain lifecycle. ``close()``
        # releases both the neural conversation ledger and the event log;
        # asserting the old events-only call would miss the conversation DB.
        partial.close.assert_called_once()
        load.assert_called_once_with(
            partial.storage_path,
            expected_brain_id=partial.brain_id,
        )
        restored.apply_allocator_oom_downgrade.assert_called_once_with(status)
        self.assertIs(worker.brains[partial.brain_id], restored)
        worker.shutdown({}, "shutdown")


if __name__ == "__main__":
    unittest.main()
