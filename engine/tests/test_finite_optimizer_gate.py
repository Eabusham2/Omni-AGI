"""Finite and rollback gates for the packed-authoritative brain.

The production cortex has no resident FP32 weight masters or Adam moments.
Tests of numerical control utilities use temporary tensors only; learned
state assertions inspect packed synapses and the worker's atomic generation.
"""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.brain import STREAMING_CANONICAL_CHUNK_ELEMENTS
from omni_core.model import PackedAdaptiveBitLinear
from omni_core.offload import ResourcePolicy
from omni_core.optimizers import PackedOnlyOptimizer
from worker import Worker


class FiniteOptimizerGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(991)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-finite-packed-")
        self.root = Path(self.temporary.name) / "brain"

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, *, persisted_origin=False) -> AdaptiveBrain:
        constructor = AdaptiveBrain.create if persisted_origin else AdaptiveBrain
        return constructor(
            "finite-packed-brain",
            self.root,
            OmniConfig.micro(
                learn_from_own_messages=False,
                gradient_checkpointing=False,
                max_seq_len=64,
                dropout=0.0,
            ),
            **({"initialize_ground_up": True} if persisted_origin else {}),
        )

    @staticmethod
    def finite_measurements():
        return {
            "loss": 0.0,
            "language_loss": 0.0,
            "idea_loss": 0.0,
            "workspace_loss": 0.0,
            "stability_loss": 0.0,
        }

    def optimize_with_loss(self, brain, loss):
        experiences = [
            (
                "finite packed gate probe",
                torch.linspace(-1.0, 1.0, brain.config.vsa_dim),
            )
        ]
        with patch.object(
            brain,
            "_training_resource_plan",
            return_value={"pauseBeforeStep": False},
        ), patch.object(
            brain,
            "_experience_ids_batch_loss",
            side_effect=loss,
        ), patch.object(
            brain,
            "_maintain_neural_state_resources",
            return_value={},
        ):
            return brain._optimize_streaming_experience_batch(experiences)

    def test_loss_component_diagnostic_names_the_failed_component(self):
        brain = self.make_brain()
        try:
            encoded = torch.tensor(
                brain.tokenizer.encode(
                    "component-specific finite diagnostic",
                    add_bos=True,
                    add_eos=True,
                ),
                dtype=torch.long,
                device=brain.device,
            )
            vector = torch.linspace(-1.0, 1.0, brain.config.vsa_dim)
            with patch.object(
                brain,
                "_stability_penalty",
                return_value=torch.tensor(float("nan"), device=brain.device),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"non-finite batch training loss component: stability_loss$",
                ):
                    brain._experience_ids_batch_loss([encoded], [vector])
        finally:
            brain.events.close()

    def test_nonfinite_local_gradient_cannot_mutate_packed_synapses(self):
        brain = self.make_brain()
        try:
            projection = brain.memory_bridge
            self.assertIsInstance(projection, PackedAdaptiveBitLinear)
            before = tuple(
                tensor.clone() for tensor in projection.authoritative_packed_tensors()
            )
            stability_before = projection._row_stability.clone()
            with self.assertRaisesRegex(ValueError, "finite"):
                projection.learn_from_gradient(
                    torch.ones((1, projection.in_features), device=brain.device),
                    torch.full(
                        (1, projection.out_features),
                        float("nan"),
                        device=brain.device,
                    ),
                    projection.online_learning_rate,
                )
            for original, current in zip(before, projection.authoritative_packed_tensors()):
                self.assertTrue(torch.equal(original, current))
            self.assertTrue(torch.equal(stability_before, projection._row_stability))
            self.assertIsInstance(brain._optimizer, PackedOnlyOptimizer)
            self.assertEqual(brain._optimizer.state, {})
        finally:
            brain.events.close()

    def test_nonfinite_clip_gate_prevents_optimizer_commit(self):
        brain = self.make_brain()
        try:
            projection = brain.memory_bridge
            before = tuple(
                tensor.clone() for tensor in projection.authoritative_packed_tensors()
            )
            stability_before = projection._row_stability.clone()

            def packed_loss(_encoded, _vectors):
                activity = torch.ones(
                    (1, projection.in_features), device=brain.device
                )
                return projection(activity).sum(), self.finite_measurements()

            with patch(
                "torch.nn.utils.clip_grad_norm_",
                return_value=torch.tensor(float("inf")),
            ) as clip, patch.object(brain._optimizer, "step") as step:
                with self.assertRaisesRegex(
                    RuntimeError, "non-finite total gradient norm after clipping"
                ):
                    self.optimize_with_loss(brain, packed_loss)
            clip.assert_called_once()
            self.assertTrue(clip.call_args.kwargs["error_if_nonfinite"])
            self.assertEqual(clip.call_args.args[1], brain.config.grad_clip)
            step.assert_not_called()
            # Direct backward may have changed codes before the failed clip.
            # The logical-batch snapshot must restore them in memory as well
            # as preventing a later durable commit or retry from seeing them.
            for original, current in zip(before, projection.authoritative_packed_tensors()):
                self.assertTrue(torch.equal(original, current))
            self.assertTrue(torch.equal(stability_before, projection._row_stability))
        finally:
            brain.events.close()

    def test_packed_only_optimizer_rejects_dense_moment_state(self):
        brain = self.make_brain()
        try:
            self.assertEqual(tuple(brain._named_slow_parameters()), ())
            self.assertIsInstance(brain._optimizer, PackedOnlyOptimizer)
            self.assertEqual(
                brain._optimizer.state_dict(),
                {"state": {}, "param_groups": []},
            )
            with self.assertRaisesRegex(ValueError, "no dense optimizer state"):
                brain._optimizer.load_state_dict(
                    {"state": {"fake-master": {"exp_avg": 1}}, "param_groups": []}
                )
        finally:
            brain.events.close()

    def test_corrupt_packed_codes_and_scale_fail_closed(self):
        brain = self.make_brain()
        try:
            projection = brain.memory_bridge
            packed = projection._packed_forward_weight
            original = packed.clone()
            with torch.no_grad():
                packed[0, 0] = (packed[0, 0] & 0xFC) | 0x03
            with self.assertRaisesRegex(ValueError, "reserved code"):
                projection.packed_forward_weight()
            with torch.no_grad():
                packed.copy_(original)
            projection.packed_forward_weight()

            scale = projection._packed_forward_scale
            original_scale = scale.clone()
            with torch.no_grad():
                scale.fill_(float("nan"))
            with self.assertRaisesRegex(ValueError, "scale"):
                projection.packed_forward_weight()
            with torch.no_grad():
                scale.copy_(original_scale)
            projection.packed_forward_weight()
        finally:
            brain.events.close()

    def test_worker_reloads_exact_packed_generation_after_failed_ingest(self):
        brain = self.make_brain(persisted_origin=True)
        worker = Worker()
        restored = None
        try:
            brain.save()
            expected_checksum = brain.parameter_checksum()
            expected_optimizer = brain._optimizer.state_dict()
            expected_codes = brain.memory_bridge._packed_forward_weight.clone()
            expected_counters = dict(brain.counters)

            def mutate_then_fail(**_kwargs):
                levels = brain.memory_bridge.effective_weight().clone()
                levels[0, 0] = 0 if int(levels[0, 0]) != 0 else 1
                brain.memory_bridge.set_ternary_weight_(levels)
                self.assertNotEqual(brain.parameter_checksum(), expected_checksum)
                raise RuntimeError("uncommitted packed update")

            worker.brains[brain.brain_id] = brain
            with patch.object(brain, "ingest", side_effect=mutate_then_fail), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "uncommitted packed update"):
                    worker.ingest(
                        {
                            "brainId": brain.brain_id,
                            "storagePath": str(brain.storage_path),
                            "text": "uncommitted packed update",
                            "jobId": "packed-rollback",
                        },
                        "packed-rollback-request",
                    )

            restored = worker.brains[brain.brain_id]
            self.assertIsNot(restored, brain)
            self.assertEqual(restored.parameter_checksum(), expected_checksum)
            self.assertTrue(
                torch.equal(restored.memory_bridge._packed_forward_weight, expected_codes)
            )
            self.assertEqual(restored._optimizer.state_dict(), expected_optimizer)
            self.assertEqual(restored.counters, expected_counters)
            rollback_event = next(
                event
                for event in reversed(restored.events.recent(20))
                if event["kind"] == "ingestion-rollback"
            )
            self.assertFalse(
                rollback_event["payload"]["uncommittedLearningRepresentedAsCommitted"]
            )
        finally:
            if restored is not None:
                restored.events.close()
            elif brain.brain_id in worker.brains:
                worker.brains[brain.brain_id].events.close()
            else:
                brain.events.close()

    def test_binary_canonicalization_preserves_tiny_and_maximum_finite_values(self):
        smallest_normal = torch.finfo(torch.float32).tiny
        smallest_subnormal = torch.nextafter(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(1.0, dtype=torch.float32),
        ).item()
        maximum = torch.finfo(torch.float32).max
        next_normal = torch.nextafter(
            torch.tensor(smallest_normal, dtype=torch.float32),
            torch.tensor(float("inf"), dtype=torch.float32),
        ).item()
        last_first_bin = torch.nextafter(
            torch.tensor(2.0 * smallest_normal, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        ).item()
        values = torch.tensor(
            [
                maximum, -maximum, smallest_normal, -smallest_normal,
                next_normal, 1.5 * smallest_normal, last_first_bin,
                2.0 * smallest_normal, smallest_subnormal, -smallest_subnormal,
                1.4451447615393347e-31, -1.4451447615393347e-31,
                1.0, -1.0, 0.0,
            ],
            dtype=torch.float32,
        )
        raw_values = values.clone()
        subnormal_before = values[8:10].clone()
        AdaptiveBrain._canonicalize_streaming_float_tensor(values)
        self.assertTrue(bool(torch.isfinite(values).all()))
        self.assertEqual(float(values[0]), maximum)
        self.assertEqual(float(values[1]), -maximum)
        self.assertEqual(float(values[2]), smallest_normal)
        self.assertEqual(float(values[3]), -smallest_normal)
        self.assertTrue(bool(torch.isfinite(values[4:8]).all()))
        self.assertTrue(torch.equal(values[8:10], subnormal_before))
        self.assertNotEqual(float(values[10]), 0.0)
        self.assertNotEqual(float(values[11]), 0.0)
        finite_normal = values.abs().ge(smallest_normal) & values.abs().lt(maximum)
        normal_bits = values[finite_normal].abs().view(torch.int32)
        self.assertTrue(bool(normal_bits.bitwise_and(0xF).eq(0).all()))
        once = values.clone()
        second_delta = AdaptiveBrain._canonicalize_streaming_float_tensor(values)
        self.assertEqual(second_delta, 0.0)
        self.assertTrue(torch.equal(values, once))
        fallback_values = raw_values.clone()
        with patch.object(
            AdaptiveBrain,
            "_round_streaming_ieee_bits_on_device",
            side_effect=RuntimeError("aten::view.dtype is not implemented for test backend"),
        ):
            AdaptiveBrain._canonicalize_streaming_float_tensor(fallback_values)
        self.assertTrue(torch.equal(fallback_values, once))

    def test_streaming_canonicalization_never_rewrites_packed_synapses(self):
        brain = self.make_brain()
        try:
            before = brain.parameter_checksum()
            self.assertEqual(tuple(brain._named_slow_parameters()), ())
            self.assertEqual(brain._canonicalize_streaming_learning_state(), 0.0)
            self.assertEqual(brain.parameter_checksum(), before)
            self.assertEqual(brain._optimizer.state, {})
        finally:
            brain.events.close()

    def test_canonicalization_bounds_each_device_chunk(self):
        values = torch.full(
            (STREAMING_CANONICAL_CHUNK_ELEMENTS + 17,),
            1.4451447615393347e-31,
            dtype=torch.float32,
        )
        device_round = AdaptiveBrain._round_streaming_ieee_bits_on_device
        with patch.object(
            AdaptiveBrain,
            "_round_streaming_ieee_bits_on_device",
            wraps=device_round,
        ) as rounded:
            AdaptiveBrain._canonicalize_streaming_float_tensor(values)
        self.assertEqual(
            [call.args[0].numel() for call in rounded.call_args_list],
            [STREAMING_CANONICAL_CHUNK_ELEMENTS, 17],
        )
        self.assertTrue(bool(torch.isfinite(values).all()))

    def test_repeated_corpus_steps_preserve_unrelated_packed_modules(self):
        reserve = patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)
        brain = self.make_brain()
        try:
            brain._replace_optimizer()
            self.assertIsInstance(brain._optimizer, PackedOnlyOptimizer)
            unrelated = {
                "action": brain.decoder.action_policy,
                "internal_action": brain.decoder.internal_action_policy,
                "image": brain.modalities.image,
                "audio": brain.modalities.audio,
                "video": brain.modalities.video,
            }
            before_unrelated = {
                name: {key: value.clone() for key, value in module.state_dict().items()}
                for name, module in unrelated.items()
            }
            bridge = brain.memory_bridge
            bridge.online_learning_rate = 100.0
            bridge_before = bridge._packed_forward_weight.clone()
            reports = []
            with patch.object(
                brain,
                "_training_resource_plan",
                return_value={
                    "pauseBeforeStep": False,
                    "physicalBatchRecords": 1,
                    "windowTokens": 64,
                },
            ), patch.object(
                brain,
                "_maintain_neural_state_resources",
                return_value={},
            ):
                for index in range(5):
                    reports.append(
                        brain._optimize_streaming_experience_batch(
                            [
                                (
                                    "corpus record %d keeps every token" % index,
                                    torch.linspace(-1.0, 1.0, brain.config.vsa_dim),
                                )
                            ]
                        )
                    )
            self.assertTrue(all(report["records"] == 1.0 for report in reports))
            self.assertFalse(torch.equal(bridge_before, bridge._packed_forward_weight))
            self.assertEqual(brain._optimizer.state, {})
            for name, module in unrelated.items():
                for key, original in before_unrelated[name].items():
                    self.assertTrue(torch.equal(module.state_dict()[key], original), f"{name}.{key}")
        finally:
            brain.events.close()


if __name__ == "__main__":
    unittest.main()
