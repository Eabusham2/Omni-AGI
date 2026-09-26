import contextlib
import copy
import io
import math
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
from omni_core.offload import ResourcePolicy
from worker import Worker


class FiniteOptimizerGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(991)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-finite-optimizer-"
        )
        self.root = Path(self.temporary.name) / "brain"

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self, *, persisted_origin=False) -> AdaptiveBrain:
        constructor = AdaptiveBrain.create if persisted_origin else AdaptiveBrain
        return constructor(
            "finite-optimizer-brain",
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
    def active_parameter(brain):
        return next(
            (name, parameter)
            for name, parameter in brain._named_slow_parameters().items()
            if parameter.requires_grad and parameter.numel() > 0
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

    def synthetic_loss(self, parameter, *, nonfinite_gradient=False):
        def calculate(_encoded, _vectors):
            loss = parameter.reshape(-1)[0] * 0.0
            if nonfinite_gradient:
                loss.register_hook(
                    lambda gradient: torch.full_like(gradient, float("nan"))
                )
            else:
                loss = loss + parameter.reshape(-1)[0]
            return loss, self.finite_measurements()

        return calculate

    def optimize_with_loss(self, brain, loss):
        experiences = [
            (
                "finite optimizer gate probe",
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

    def assert_nested_exact(self, expected, actual, path="state"):
        if isinstance(expected, torch.Tensor):
            self.assertIsInstance(actual, torch.Tensor, path)
            self.assertEqual(expected.dtype, actual.dtype, path)
            self.assertEqual(expected.device.type, actual.device.type, path)
            self.assertTrue(torch.equal(expected, actual), path)
            return
        if isinstance(expected, dict):
            self.assertIsInstance(actual, dict, path)
            self.assertEqual(set(expected), set(actual), path)
            for key in expected:
                self.assert_nested_exact(
                    expected[key], actual[key], "%s.%s" % (path, key)
                )
            return
        if isinstance(expected, (list, tuple)):
            self.assertIsInstance(actual, type(expected), path)
            self.assertEqual(len(expected), len(actual), path)
            for index, (left, right) in enumerate(zip(expected, actual)):
                self.assert_nested_exact(
                    left, right, "%s[%d]" % (path, index)
                )
            return
        self.assertEqual(expected, actual, path)

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

    def test_nonfinite_gradient_fails_before_slow_importance_mutates(self):
        brain = self.make_brain()
        try:
            parameter_name, parameter = self.active_parameter(brain)
            brain._sync_stability_state()
            importance_before = {
                name: value.clone()
                for name, value in brain.slow_importance.items()
            }
            metaplastic_updates_before = brain.counters["metaplastic_updates"]

            with self.assertRaisesRegex(
                RuntimeError,
                "non-finite gradient before slow-importance: %s$"
                % parameter_name.replace(".", r"\."),
            ):
                self.optimize_with_loss(
                    brain,
                    self.synthetic_loss(
                        parameter, nonfinite_gradient=True
                    ),
                )

            self.assertEqual(
                brain.counters["metaplastic_updates"],
                metaplastic_updates_before,
            )
            self.assertEqual(set(brain.slow_importance), set(importance_before))
            for name, value in importance_before.items():
                self.assertTrue(torch.equal(brain.slow_importance[name], value))
            self.assertEqual(len(brain._optimizer.state), 0)
            self.assertTrue(
                all(
                    candidate.grad is None
                    for group in brain._optimizer.param_groups
                    for candidate in group["params"]
                )
            )
        finally:
            brain.events.close()

    def test_clip_gate_requests_error_and_checks_returned_norm(self):
        brain = self.make_brain()
        try:
            _, parameter = self.active_parameter(brain)
            original_step = brain._optimizer.step
            with patch(
                "torch.nn.utils.clip_grad_norm_",
                return_value=torch.tensor(float("inf")),
            ) as clip, patch.object(
                brain._optimizer, "step", wraps=original_step
            ) as step:
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"non-finite total gradient norm after clipping$",
                ):
                    self.optimize_with_loss(
                        brain, self.synthetic_loss(parameter)
                    )

            step.assert_not_called()
            clip.assert_called_once()
            self.assertTrue(clip.call_args.kwargs["error_if_nonfinite"])
            self.assertEqual(
                clip.call_args.args[1], brain.config.grad_clip
            )
        finally:
            brain.events.close()

    def test_pre_step_adam_invariants_reject_corrupt_persisted_state(self):
        brain = self.make_brain()
        try:
            parameter_name, parameter = self.active_parameter(brain)
            state = brain._optimizer.state[parameter]
            valid_state = {
                "step": torch.tensor(27.0),
                "exp_avg": torch.zeros_like(parameter),
                "exp_avg_sq": torch.zeros_like(parameter),
            }
            brain._sync_stability_state()
            importance_before = {
                name: value.clone()
                for name, value in brain.slow_importance.items()
            }
            updates_before = brain.counters["metaplastic_updates"]
            cases = (
                (
                    "nonfinite-step",
                    {**valid_state, "step": torch.tensor(float("nan"))},
                    "invalid Adam optimizer step before optimizer step: "
                    "%s.step$" % parameter_name.replace(".", r"\."),
                ),
                (
                    "negative-step",
                    {**valid_state, "step": torch.tensor(-1.0)},
                    "invalid Adam optimizer step before optimizer step: "
                    "%s.step$" % parameter_name.replace(".", r"\."),
                ),
                (
                    "fractional-step",
                    {**valid_state, "step": torch.tensor(1.5)},
                    "invalid Adam optimizer step before optimizer step: "
                    "%s.step$" % parameter_name.replace(".", r"\."),
                ),
                (
                    "nonfinite-first-moment",
                    {
                        **valid_state,
                        "exp_avg": torch.full_like(parameter, float("nan")),
                    },
                    "non-finite Adam optimizer state before optimizer step: "
                    "%s.exp_avg$" % parameter_name.replace(".", r"\."),
                ),
                (
                    "nonfinite-second-moment",
                    {
                        **valid_state,
                        "exp_avg_sq": torch.full_like(
                            parameter, float("inf")
                        ),
                    },
                    "non-finite Adam optimizer state before optimizer step: "
                    "%s.exp_avg_sq$"
                    % parameter_name.replace(".", r"\."),
                ),
                (
                    "negative-second-moment",
                    {
                        **valid_state,
                        "exp_avg_sq": torch.full_like(parameter, -1.0),
                    },
                    "negative Adam optimizer second moment before optimizer "
                    "step: %s.exp_avg_sq$"
                    % parameter_name.replace(".", r"\."),
                ),
            )
            original_step = brain._optimizer.step
            with patch.object(
                brain._optimizer, "step", wraps=original_step
            ) as optimizer_step:
                for label, corrupt_state, expected in cases:
                    with self.subTest(state=label):
                        state.clear()
                        state.update(
                            {
                                key: value.clone()
                                for key, value in corrupt_state.items()
                            }
                        )
                        with self.assertRaisesRegex(RuntimeError, expected):
                            self.optimize_with_loss(
                                brain, self.synthetic_loss(parameter)
                            )
            optimizer_step.assert_not_called()
            self.assertEqual(
                brain.counters["metaplastic_updates"], updates_before
            )
            for name, value in importance_before.items():
                self.assertTrue(torch.equal(brain.slow_importance[name], value))
        finally:
            brain.events.close()

    def test_post_step_nonfinite_parameter_fails_without_sanitizing_it(self):
        brain = self.make_brain()
        try:
            parameter_name, parameter = self.active_parameter(brain)
            original_step = brain._optimizer.step

            def corrupt_parameter(*args, **kwargs):
                result = original_step(*args, **kwargs)
                with torch.no_grad():
                    parameter.reshape(-1)[0] = float("nan")
                return result

            with patch.object(
                brain._optimizer,
                "step",
                side_effect=corrupt_parameter,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "non-finite parameter after optimizer step: %s$"
                    % parameter_name.replace(".", r"\."),
                ):
                    self.optimize_with_loss(
                        brain, self.synthetic_loss(parameter)
                    )

            self.assertTrue(
                math.isnan(float(parameter.detach().reshape(-1)[0]))
            )
        finally:
            brain.events.close()

    def test_post_step_nonfinite_adam_moment_names_parameter_and_moment(self):
        brain = self.make_brain()
        try:
            parameter_name, parameter = self.active_parameter(brain)
            original_step = brain._optimizer.step

            def corrupt_moment(*args, **kwargs):
                result = original_step(*args, **kwargs)
                brain._optimizer.state[parameter]["exp_avg"].reshape(-1)[
                    0
                ] = float("nan")
                return result

            with patch.object(
                brain._optimizer,
                "step",
                side_effect=corrupt_moment,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "non-finite Adam optimizer state after optimizer step: "
                    "%s.exp_avg$"
                    % parameter_name.replace(".", r"\."),
                ):
                    self.optimize_with_loss(
                        brain, self.synthetic_loss(parameter)
                    )

            self.assertTrue(
                torch.isnan(
                    brain._optimizer.state[parameter]["exp_avg"]
                ).any()
            )
        finally:
            brain.events.close()

    def test_worker_reloads_exact_atomic_state_after_post_step_failure(self):
        brain = self.make_brain(persisted_origin=True)
        worker = Worker()
        restored = None
        try:
            _, parameter = self.active_parameter(brain)
            self.optimize_with_loss(brain, self.synthetic_loss(parameter))
            brain.save()
            expected_checksum = brain.parameter_checksum()
            expected_optimizer = copy.deepcopy(brain._optimizer.state_dict())
            expected_anchors = {
                name: value.clone() for name, value in brain.slow_anchors.items()
            }
            expected_importance = {
                name: value.clone()
                for name, value in brain.slow_importance.items()
            }
            expected_counters = dict(brain.counters)
            original_step = brain._optimizer.step

            def corrupt_moment(*args, **kwargs):
                result = original_step(*args, **kwargs)
                brain._optimizer.state[parameter]["exp_avg_sq"].reshape(-1)[
                    0
                ] = float("nan")
                return result

            def run_failed_batch(**_kwargs):
                return self.optimize_with_loss(
                    brain, self.synthetic_loss(parameter)
                )

            worker.brains[brain.brain_id] = brain
            with patch.object(
                brain._optimizer,
                "step",
                side_effect=corrupt_moment,
            ), patch.object(
                brain, "ingest", side_effect=run_failed_batch
            ), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    RuntimeError,
                    r"non-finite Adam optimizer state after optimizer step: "
                    r".*\.exp_avg_sq$",
                ):
                    worker.ingest(
                        {
                            "brainId": brain.brain_id,
                            "storagePath": str(brain.storage_path),
                            "text": "uncommitted finite-gate batch",
                            "jobId": "finite-gate-rollback",
                        },
                        "finite-gate-request",
                    )

            restored = worker.brains[brain.brain_id]
            self.assertIsNot(restored, brain)
            self.assertEqual(restored.parameter_checksum(), expected_checksum)
            self.assert_nested_exact(
                expected_optimizer,
                restored._optimizer.state_dict(),
                "optimizer",
            )
            self.assertEqual(set(restored.slow_anchors), set(expected_anchors))
            self.assertEqual(
                set(restored.slow_importance), set(expected_importance)
            )
            for name, value in expected_anchors.items():
                self.assertTrue(torch.equal(restored.slow_anchors[name], value))
            for name, value in expected_importance.items():
                self.assertTrue(
                    torch.equal(restored.slow_importance[name], value)
                )
            self.assertEqual(restored.counters, expected_counters)
            self.assertTrue(
                all(
                    bool(torch.isfinite(value).all())
                    for state in restored._optimizer.state.values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                )
            )
            rollback_event = next(
                event
                for event in reversed(restored.events.recent(20))
                if event["kind"] == "ingestion-rollback"
            )
            self.assertEqual(rollback_event["payload"]["activeCheckpoints"], [])
            self.assertFalse(
                rollback_event["payload"][
                    "uncommittedLearningRepresentedAsCommitted"
                ]
            )
        finally:
            if restored is not None:
                restored.events.close()
            elif brain.brain_id in worker.brains:
                worker.brains[brain.brain_id].events.close()
            else:
                try:
                    brain.events.close()
                except Exception:
                    pass

    def test_binary_canonicalization_preserves_tiny_and_maximum_finite_values(
        self,
    ):
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
                maximum,
                -maximum,
                smallest_normal,
                -smallest_normal,
                next_normal,
                1.5 * smallest_normal,
                last_first_bin,
                2.0 * smallest_normal,
                smallest_subnormal,
                -smallest_subnormal,
                1.4451447615393347e-31,
                -1.4451447615393347e-31,
                1.0,
                -1.0,
                0.0,
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
        finite_normal = (
            values.abs().ge(smallest_normal)
            & values.abs().lt(maximum)
        )
        normal_bits = values[finite_normal].abs().view(torch.int32)
        self.assertTrue(
            bool(normal_bits.bitwise_and(0xF).eq(0).all())
        )
        once = values.clone()
        second_delta = AdaptiveBrain._canonicalize_streaming_float_tensor(
            values
        )
        self.assertEqual(second_delta, 0.0)
        self.assertTrue(torch.equal(values, once))
        fallback_values = raw_values.clone()
        with patch.object(
            AdaptiveBrain,
            "_round_streaming_ieee_bits_on_device",
            side_effect=RuntimeError(
                "aten::view.dtype is not implemented for test backend"
            ),
        ):
            AdaptiveBrain._canonicalize_streaming_float_tensor(
                fallback_values
            )
        self.assertTrue(torch.equal(fallback_values, once))

    def test_post_canonicalization_gate_names_each_mutated_state_kind(self):
        brain = self.make_brain()
        try:
            parameter_name = "decoder.action_policy.hidden.weight"
            parameter = brain._named_slow_parameters()[parameter_name]
            state = brain._optimizer.state[parameter]
            exact_large_step = float((1 << 20) + 1)
            state["step"] = torch.tensor(exact_large_step)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
            targets = (
                (
                    "parameter.%s" % parameter_name,
                    parameter,
                ),
                (
                    "optimizer.%s.exp_avg_sq" % parameter_name,
                    state["exp_avg_sq"],
                ),
                (
                    "slow-anchor.%s" % parameter_name,
                    brain.slow_anchors[parameter_name],
                ),
                (
                    "slow-importance.%s" % parameter_name,
                    brain.slow_importance[parameter_name],
                ),
            )
            for expected_label, tensor in targets:
                with self.subTest(state=expected_label):
                    original = tensor.reshape(-1)[0].detach().clone()
                    with torch.no_grad():
                        tensor.reshape(-1)[0] = float("nan")
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "non-finite streaming learning state after "
                        "canonicalization: %s$"
                        % expected_label.replace(".", r"\."),
                    ):
                        brain._canonicalize_streaming_learning_state()
                    with torch.no_grad():
                        tensor.reshape(-1)[0] = original
            self.assertEqual(float(state["step"]), exact_large_step)
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

    def test_repeated_corpus_steps_preserve_unrelated_learned_modules(self):
        # This isolated micro fixture writes only a tiny checkpoint. The
        # production reserve remains unchanged even when the host is below
        # its normal 20 GiB free-space requirement.
        reserve = patch.object(
            ResourcePolicy, "_adaptive_disk_reserve", return_value=1024 * 1024
        )
        reserve.start()
        self.addCleanup(reserve.stop)
        brain = self.make_brain()
        try:
            brain._replace_optimizer()
            brain._sync_stability_state()
            unrelated = {
                name: parameter
                for name, parameter in brain._named_slow_parameters().items()
                if name.startswith(
                    (
                        "decoder.action_policy.",
                        "decoder.internal_action_policy.",
                        "modalities.",
                    )
                )
            }
            self.assertTrue(unrelated)
            self.assertTrue(
                any(name.startswith("modalities.") for name in unrelated)
            )
            self.assertTrue(
                any(name.startswith("decoder.action_policy.") for name in unrelated)
            )
            connected_ids = {
                id(parameter)
                for parameter in brain._streaming_experience_parameters()
            }
            self.assertEqual(tuple(brain.decoder.workspace_strength.parameters()), ())
            self.assertEqual(tuple(brain.decoder.memory_strength.parameters()), ())
            self.assertTrue(
                all(
                    id(parameter) not in connected_ids
                    for parameter in unrelated.values()
                )
            )
            for name, parameter in unrelated.items():
                brain.slow_anchors[name].copy_(parameter.detach().cpu())
                brain.slow_importance[name].fill_(0.5)

            action_name = "decoder.action_policy.hidden.weight"
            action_parameter = unrelated[action_name]
            action_state = brain._optimizer.state[action_parameter]
            action_state["step"] = torch.tensor(27.0)
            action_state["exp_avg"] = torch.zeros_like(action_parameter)
            action_state["exp_avg_sq"] = torch.zeros_like(action_parameter)
            parameter_before = {
                name: parameter.detach().clone()
                for name, parameter in unrelated.items()
            }
            anchor_before = {
                name: brain.slow_anchors[name].clone() for name in unrelated
            }
            importance_before = {
                name: brain.slow_importance[name].clone()
                for name in unrelated
            }
            action_state_before = {
                key: value.clone() for key, value in action_state.items()
            }
            connected_parameter = brain.memory_bridge.weight
            connected_before = connected_parameter.detach().clone()
            traces = []
            original_step = brain._optimizer.step

            def traced_step(*args, **kwargs):
                trace = {
                    "unrelatedGradientsNone": all(
                        parameter.grad is None
                        for parameter in unrelated.values()
                    ),
                    "connectedGradientFinite": (
                        connected_parameter.grad is not None
                        and bool(
                            torch.isfinite(connected_parameter.grad).all()
                        )
                    ),
                    "connectedGradientNonzero": (
                        connected_parameter.grad is not None
                        and bool(connected_parameter.grad.ne(0).any())
                    ),
                }
                result = original_step(*args, **kwargs)
                trace.update(
                    {
                        "actionStep": float(action_state["step"]),
                        "actionMomentMaximum": float(
                            action_state["exp_avg"].abs().max()
                        ),
                        "actionSecondMomentMaximum": float(
                            action_state["exp_avg_sq"].abs().max()
                        ),
                    }
                )
                traces.append(trace)
                return result

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
            ), patch.object(
                brain._optimizer,
                "step",
                side_effect=traced_step,
            ):
                for index in range(5):
                    reports.append(
                        brain._optimize_streaming_experience_batch(
                            [
                                (
                                    "corpus record %d keeps every token"
                                    % index,
                                    torch.linspace(
                                        -1.0,
                                        1.0,
                                        brain.config.vsa_dim,
                                    ),
                                )
                            ]
                        )
                    )

            self.assertEqual(len(traces), 5)
            self.assertTrue(
                all(
                    trace["unrelatedGradientsNone"]
                    and trace["connectedGradientFinite"]
                    and trace["connectedGradientNonzero"]
                    and trace["actionStep"] == 27.0
                    and trace["actionMomentMaximum"] == 0.0
                    and trace["actionSecondMomentMaximum"] == 0.0
                    for trace in traces
                ),
                traces,
            )
            self.assertTrue(
                all(
                    report["records"] == 1.0
                    and report["optimizer_steps"] == 1.0
                    for report in reports
                )
            )
            self.assertFalse(
                torch.equal(connected_parameter.detach(), connected_before)
            )
            for name, parameter in unrelated.items():
                self.assertTrue(
                    torch.equal(parameter.detach(), parameter_before[name]),
                    name,
                )
                self.assertTrue(
                    torch.equal(brain.slow_anchors[name], anchor_before[name]),
                    name,
                )
                self.assertTrue(
                    torch.equal(
                        brain.slow_importance[name], importance_before[name]
                    ),
                    name,
                )
                if name != action_name:
                    self.assertNotIn(parameter, brain._optimizer.state, name)
            for key, value in action_state_before.items():
                self.assertTrue(torch.equal(action_state[key], value), key)
        finally:
            brain.events.close()


if __name__ == "__main__":
    unittest.main()
