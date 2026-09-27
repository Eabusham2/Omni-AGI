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
from omni_core.model import (
    GlobalWorkspace,
    PACKED_AUTHORITATIVE_PROJECTION_TYPES,
)


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
        # Compare exact input derivatives at a fixed packed synaptic state.
        # Online packed plasticity is verified separately from this numerical
        # chunking equivalence test.
        for workspace in (reference, chunked):
            for module in workspace.modules():
                if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES):
                    module.online_learning_rate = 0.0
        reference.query_chunk_slots = reference.slots
        chunked.query_chunk_slots = 3
        self.assertEqual(
            tuple(reference.state_dict()), tuple(chunked.state_dict())
        )
        state = chunked.state_dict()
        for name in ("latent_table", "query", "key", "value", "update", "broadcast"):
            self.assertEqual(state[f"{name}._packed_forward_weight"].dtype, torch.uint8)
            self.assertNotIn(f"{name}.weight", state)
        self.assertIn("norm.scale_delta._packed_forward_weight", state)
        self.assertNotIn("norm.scale", state)
        self.assertEqual(tuple(chunked.named_parameters()), ())

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
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, chunked.state_dict()[name]), name)

    def test_workspace_sequential_backward_is_none_safe_and_side_effect_free(
        self,
    ):
        workspace = GlobalWorkspace(
            dimensions=8, slots=10, iterations=2
        ).train()
        workspace.query_chunk_slots = 3
        for module in workspace.modules():
            if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES):
                module.online_learning_rate = 0.0
        packed_before = {
            name: value.clone()
            for name, value in workspace.state_dict().items()
            if name.endswith("._packed_forward_weight")
        }
        inputs = torch.randn(2, 6, 8, requires_grad=True)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, False, False, False],
            ]
        )

        summary = workspace.summarize(inputs, attention_mask=mask)
        (input_gradient,) = torch.autograd.grad(summary.square().mean(), inputs)
        self.assertTrue(bool(torch.isfinite(input_gradient).all()))
        self.assertIsNone(inputs.grad)
        self.assertEqual(tuple(workspace.named_parameters()), ())
        for name, before in packed_before.items():
            self.assertTrue(torch.equal(before, workspace.state_dict()[name]), name)

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
        padding_before = brain.decoder.embedding.effective_weight()[
            brain.tokenizer.pad_id
        ].clone()

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
            self.assertTrue(
                torch.equal(
                    brain.decoder.embedding.effective_weight()[brain.tokenizer.pad_id],
                    padding_before,
                )
            )
        finally:
            handle.remove()
            brain.events.close()

    def test_mps_cache_release_brackets_each_microbatch_backward(self):
        brain = self.make_brain()
        brain._runtime_train_batch_size = 1
        brain.config.grad_clip = 100.0
        # A test-only scalar probes allocator/cache ordering; it is not an
        # FP32 shadow of any production packed synapse.
        probe = torch.nn.Parameter(torch.tensor(1.0))
        initial = probe.detach().clone()
        experiences = [
            ("first", torch.full((brain.config.vsa_dim,), 2.0)),
            ("second", torch.full((brain.config.vsa_dim,), 4.0)),
        ]

        def run(backend):
            events = []
            with torch.no_grad():
                probe.copy_(initial)
            brain._optimizer = torch.optim.SGD([probe], lr=0.01)
            brain.device_backend = backend

            def synthetic_loss(_encoded, vectors):
                value = float(vectors[0][0])
                events.append("forward:%d" % value)
                loss = probe * value
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
            return probe.detach().clone(), report, events

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
