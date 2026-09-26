import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.model import GlobalWorkspace


class StreamingBatchNumericPathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(733)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-streaming-numeric-"
        )
        self.root = Path(self.temporary.name) / "brain"

    def tearDown(self):
        self.temporary.cleanup()

    def make_brain(self) -> AdaptiveBrain:
        return AdaptiveBrain(
            "streaming-numeric-brain",
            self.root,
            OmniConfig.micro(
                learn_from_own_messages=False,
                gradient_checkpointing=False,
                max_seq_len=64,
                dropout=0.0,
            ),
        )

    def test_workspace_summary_chunks_match_full_outputs_and_gradients(self):
        reference = GlobalWorkspace(
            dimensions=12, slots=11, iterations=3
        ).eval()
        chunked = GlobalWorkspace(
            dimensions=12, slots=11, iterations=3
        ).train()
        chunked.load_state_dict(reference.state_dict(), strict=True)
        reference.query_chunk_slots = reference.slots
        chunked.query_chunk_slots = 3
        self.assertEqual(
            tuple(reference.state_dict()), tuple(chunked.state_dict())
        )
        self.assertEqual(
            tuple(chunked.state_dict()),
            (
                "latents",
                "query.weight",
                "key.weight",
                "value.weight",
                "update.weight",
                "broadcast.weight",
                "norm.scale",
            ),
        )

        base_inputs = torch.randn(2, 7, 12)
        reference_inputs = base_inputs.clone().requires_grad_(True)
        chunked_inputs = base_inputs.clone().requires_grad_(True)
        attention_mask = torch.tensor(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, False, False, False],
            ]
        )
        summary_probe = torch.randn(2, 12)

        _, reference_summary = reference(
            reference_inputs, attention_mask=attention_mask
        )
        chunked_summary = chunked.summarize(
            chunked_inputs, attention_mask=attention_mask
        )
        torch.testing.assert_close(
            chunked_summary,
            reference_summary,
            rtol=1e-6,
            atol=1e-7,
        )

        (reference_summary * summary_probe).sum().backward()
        (chunked_summary * summary_probe).sum().backward()
        torch.testing.assert_close(
            chunked_inputs.grad,
            reference_inputs.grad,
            rtol=1e-5,
            atol=1e-6,
        )
        for (reference_name, reference_parameter), (
            chunked_name,
            chunked_parameter,
        ) in zip(reference.named_parameters(), chunked.named_parameters()):
            self.assertEqual(chunked_name, reference_name)
            self.assertIsNotNone(reference_parameter.grad)
            self.assertIsNotNone(chunked_parameter.grad)
            torch.testing.assert_close(
                chunked_parameter.grad,
                reference_parameter.grad,
                rtol=1e-4,
                atol=3e-5,
                msg=lambda message, name=reference_name: "%s: %s"
                % (name, message),
            )

    def test_workspace_sequential_backward_is_none_safe_and_side_effect_free(
        self,
    ):
        workspace = GlobalWorkspace(
            dimensions=8, slots=10, iterations=2
        ).train()
        workspace.query_chunk_slots = 3
        workspace.query.weight.requires_grad_(False)
        inputs = torch.randn(2, 6, 8)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
            ]
        )

        summary = workspace.summarize(inputs, attention_mask=mask)
        active_parameters = tuple(
            parameter
            for parameter in workspace.parameters()
            if parameter.requires_grad
        )
        gradients = torch.autograd.grad(
            summary.square().mean(),
            active_parameters,
            allow_unused=True,
        )
        self.assertTrue(gradients)
        self.assertTrue(
            all(
                gradient is not None
                and bool(torch.isfinite(gradient).all())
                for gradient in gradients
            )
        )
        self.assertIsNone(workspace.query.weight.grad)
        self.assertTrue(
            all(parameter.grad is None for parameter in active_parameters)
        )

    def test_padded_batch_trains_every_non_padding_target(self):
        brain = self.make_brain()
        rows = [
            torch.tensor(
                brain.tokenizer.encode(
                    "short", add_bos=True, add_eos=True
                ),
                dtype=torch.long,
                device=brain.device,
            ),
            torch.tensor(
                brain.tokenizer.encode(
                    "a deliberately longer row",
                    add_bos=True,
                    add_eos=True,
                ),
                dtype=torch.long,
                device=brain.device,
            ),
        ]
        vectors = [
            torch.linspace(-1.0, 1.0, brain.config.vsa_dim),
            torch.linspace(1.0, -1.0, brain.config.vsa_dim),
        ]
        captured = {}

        def capture_logits(_module, _inputs, output):
            output.retain_grad()
            captured["logits"] = output

        handle = brain.decoder.language_head.register_forward_hook(
            capture_logits
        )
        try:
            loss, measured = brain._experience_ids_batch_loss(rows, vectors)
            logits = captured["logits"]
            expected_language = torch.stack(
                [
                    F.cross_entropy(
                        logits[index, : row.numel() - 1],
                        row[1:],
                    )
                    for index, row in enumerate(rows)
                ]
            ).mean()
            self.assertAlmostEqual(
                measured["language_loss"],
                float(expected_language.detach()),
                places=6,
            )

            loss.backward()
            self.assertIsNotNone(logits.grad)
            active_predictions = logits.grad.abs().sum(dim=-1).gt(0)
            expected_predictions = torch.zeros_like(
                active_predictions, dtype=torch.bool
            )
            for index, row in enumerate(rows):
                expected_predictions[index, : row.numel() - 1] = True

            self.assertTrue(
                torch.equal(active_predictions, expected_predictions)
            )
            self.assertEqual(
                int(active_predictions.sum()),
                sum(int(row.numel()) - 1 for row in rows),
            )
            self.assertEqual(
                int(brain.decoder.embedding.weight.grad[brain.tokenizer.pad_id]
                    .count_nonzero()),
                0,
            )
        finally:
            handle.remove()
            brain.events.close()

    def test_mps_cache_release_brackets_each_microbatch_backward(self):
        brain = self.make_brain()
        brain._runtime_train_batch_size = 1
        brain.config.grad_clip = 100.0
        parameter = next(
            value
            for value in brain.decoder.parameters()
            if value.requires_grad
        )
        initial = parameter.detach().clone()
        experiences = [
            ("first", torch.full((brain.config.vsa_dim,), 2.0)),
            ("second", torch.full((brain.config.vsa_dim,), 4.0)),
        ]

        def run(backend):
            events = []
            with torch.no_grad():
                parameter.copy_(initial)
            brain._optimizer = torch.optim.SGD([parameter], lr=0.01)
            brain.device_backend = backend

            def synthetic_loss(_encoded, vectors):
                value = float(vectors[0][0])
                events.append("forward:%d" % value)
                loss = parameter.reshape(-1)[0] * value
                loss.register_hook(
                    lambda _gradient, label=value: events.append(
                        "backward:%d" % label
                    )
                )
                return loss, {
                    "loss": value,
                    "language_loss": value,
                    "idea_loss": 0.0,
                    "workspace_loss": 0.0,
                    "stability_loss": 0.0,
                }

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
                "_release_training_allocator_cache",
                side_effect=lambda: events.append("cache"),
            ), patch.object(
                brain,
                "_maintain_neural_state_resources",
                return_value={},
            ), patch.object(
                torch,
                "mps",
                SimpleNamespace(
                    get_rng_state=lambda: torch.get_rng_state().clone()
                ),
            ):
                report = brain._optimize_streaming_experience_batch(
                    experiences
                )
            return parameter.detach().clone(), report, events

        try:
            cpu_parameter, cpu_report, cpu_events = run("cpu")
            mps_parameter, mps_report, mps_events = run("mps")
            self.assertTrue(torch.equal(mps_parameter, cpu_parameter))
            self.assertEqual(mps_report["loss"], cpu_report["loss"])
            self.assertEqual(
                mps_report["language_loss"],
                cpu_report["language_loss"],
            )
            self.assertEqual(
                cpu_events,
                [
                    "forward:2",
                    "backward:2",
                    "forward:4",
                    "backward:4",
                ],
            )
            self.assertEqual(
                mps_events,
                [
                    "forward:2",
                    "cache",
                    "backward:2",
                    "cache",
                    "forward:4",
                    "cache",
                    "backward:4",
                    "cache",
                ],
            )
        finally:
            brain.device_backend = "cpu"
            brain.events.close()


if __name__ == "__main__":
    unittest.main()
