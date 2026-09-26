import sys
import unittest
from pathlib import Path

import torch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import omni_core
from omni_core import model as model_module
from omni_core.config import OmniConfig
from omni_core.model import (
    OmniDecoder,
    PACKED_AUTHORITATIVE_PROJECTION_TYPES,
    PackedAdaptiveBitConv2d,
    PackedAdaptiveBitConvTranspose2d,
    RMSNorm,
    packed_runtime_status,
    require_packed_runtime_complete,
)
from omni_core.tokenizer import ByteTokenizer
from omni_core.optimizers import adamw_for_remaining_parameters
from omni_core.ternary_packing import inspect_module_ternary_layout


class TernaryDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_float_master_projection_classes_are_absent(self):
        self.assertFalse(hasattr(omni_core, "BitLinear"))
        for name in (
            "BitLinear",
            "BitConv1d",
            "BitConv2d",
            "BitConv3d",
            "BitConvTranspose1d",
            "BitConvTranspose2d",
            "BitConvTranspose3d",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(model_module, name))

    def test_modality_convolutions_use_exact_ternary_forward_weights(self):
        inputs = torch.randn(1, 3, 8, 8, requires_grad=True)
        encoder = PackedAdaptiveBitConv2d(
            3, 4, 3, padding=1, online_learning_rate=100.0
        )
        decoder = PackedAdaptiveBitConvTranspose2d(
            4, 3, 3, padding=1, online_learning_rate=100.0
        )
        before = [
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors())
            for module in (encoder, decoder)
        ]
        encoded = encoder(inputs)
        output = decoder(encoded)
        output.square().mean().backward()
        for module, snapshot in zip((encoder, decoder), before):
            self.assertTrue(
                set(module.effective_weight().reshape(-1).tolist()).issubset(
                    {-1, 0, 1}
                )
            )
            self.assertEqual(tuple(module.parameters()), ())
            self.assertTrue(
                any(
                    not torch.equal(previous, current)
                    for previous, current in zip(
                        snapshot, module.authoritative_packed_tensors()
                    )
                )
            )
            restored = type(module)(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                padding=module.padding,
            )
            restored.load_state_dict(module.state_dict())
            self.assertTrue(
                torch.equal(module.effective_weight(), restored.effective_weight())
            )

    def test_final_packed_gate_accepts_linear_and_convolution_kernels(self):
        decoder = OmniDecoder(OmniConfig.micro(dropout=0.0))
        self.assertTrue(packed_runtime_status(decoder)["complete"])
        convolution = PackedAdaptiveBitConv2d(3, 4, 3, padding=1)
        status = packed_runtime_status(convolution)
        self.assertTrue(status["complete"])
        self.assertEqual(status["denseConvolutionBlockers"], [])
        self.assertEqual(status["packedConvolutionModules"], 1)
        self.assertTrue(require_packed_runtime_complete(convolution)["complete"])

    def test_attention_is_causal_and_finite(self):
        config = OmniConfig.micro(dropout=0.0)
        model = OmniDecoder(config).eval()
        first = torch.tensor([[1, 10, 11, 12, 13]], dtype=torch.long)
        second = first.clone()
        second[0, -1] = 99
        with torch.no_grad():
            left = model(first)["logits"]
            right = model(second)["logits"]
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.allclose(left[:, :-1], right[:, :-1], atol=1e-6))

    def test_context_capacity_does_not_eagerly_allocate_future_position_or_kv_state(self):
        config = OmniConfig.micro(dropout=0.0, max_seq_len=1_000_000)
        model = OmniDecoder(config).eval()
        rotary = model.blocks[0].attention.rotary
        self.assertEqual(rotary.max_seq_len, 1_000_000)
        self.assertEqual(rotary.cached_seq_len, 0)

        with torch.no_grad():
            model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))

        self.assertGreaterEqual(rotary.cached_seq_len, 5)
        self.assertLess(rotary.cached_seq_len, 1_000_000)
        before = rotary.cached_seq_len
        rotary.configure_max_seq_len(2_000_000)
        self.assertEqual(rotary.max_seq_len, 2_000_000)
        self.assertEqual(rotary.cached_seq_len, before)

    def test_bounded_query_chunks_preserve_exact_causal_attention(self):
        config = OmniConfig.micro(dropout=0.0, max_seq_len=32)
        reference = OmniDecoder(config).eval()
        chunked = OmniDecoder(config).eval()
        chunked.load_state_dict(reference.state_dict())
        for block in chunked.blocks:
            block.attention.query_chunk_tokens = 3
        ids = torch.tensor([[1, 259, 10, 11, 12, 13, 14, 260]])

        with torch.no_grad():
            expected = reference(ids)["logits"]
            actual = chunked(ids)["logits"]

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_global_workspace_latents_expand_past_the_old_fixed_ceiling(self):
        config = OmniConfig.micro(working_memory_slots=512)
        model = OmniDecoder(config)
        self.assertEqual(model.global_workspace.latents.shape[0], 128)

    def test_decoder_controls_and_grown_expert_routes_are_packed_authoritative(self):
        config = OmniConfig.micro(dropout=0.0)
        model = OmniDecoder(config)
        model.grow_expert(torch.linspace(-1.0, 1.0, config.d_model))
        self.assertEqual(tuple(model.workspace_strength.parameters()), ())
        self.assertEqual(tuple(model.memory_strength.parameters()), ())
        self.assertEqual(tuple(model.final_norm.parameters()), ())
        self.assertEqual(tuple(model.expert_prototypes.parameters()), ())
        for table in (
            model.workspace_strength.levels,
            model.memory_strength.levels,
            model.final_norm.scale_delta,
        ):
            self.assertEqual(table._packed_forward_weight.dtype, torch.uint8)
            self.assertTrue(
                set(table.effective_weight().reshape(-1).tolist()).issubset(
                    {-1, 0, 1}
                )
            )
        route = model.expert_prototypes[0]
        self.assertEqual(route._packed_forward_weight.dtype, torch.uint8)
        self.assertTrue(
            set(route.effective_weight().reshape(-1).tolist()).issubset(
                {-1, 0, 1}
            )
        )
        state = model.state_dict()
        self.assertNotIn("workspace_strength", state)
        self.assertNotIn("memory_strength", state)
        self.assertNotIn("final_norm.scale", state)
        self.assertNotIn("expert_prototypes.0", state)
        self.assertIn("workspace_strength.levels._packed_forward_weight", state)
        self.assertIn("memory_strength.levels._packed_forward_weight", state)
        self.assertIn("final_norm.scale_delta._packed_forward_weight", state)
        self.assertIn("expert_prototypes.0._packed_forward_weight", state)
        packed_layout = inspect_module_ternary_layout(
            {"decoder": model}, dynamic_synapses={}
        )
        for name in (
            "decoder.workspace_strength.levels.weight",
            "decoder.memory_strength.levels.weight",
            "decoder.final_norm.scale_delta.weight",
            "decoder.expert_prototypes.0.weight",
        ):
            self.assertIn(name, packed_layout)

        restored = OmniDecoder(config)
        restored.grow_expert()
        restored.load_state_dict(state, strict=True)
        self.assertEqual(state.keys(), restored.state_dict().keys())
        for name, value in state.items():
            self.assertTrue(torch.equal(value, restored.state_dict()[name]), name)

    def test_packed_residual_gain_learns_without_float_master(self):
        model = OmniDecoder(OmniConfig.micro(dropout=0.0))
        gain = model.workspace_strength
        gain.levels.online_learning_rate = 1000.0
        before = gain.levels._packed_forward_weight.clone()
        (-100.0 * gain()).backward()
        self.assertFalse(torch.equal(before, gain.levels._packed_forward_weight))
        self.assertEqual(tuple(gain.parameters()), ())
        self.assertTrue(
            set(gain.levels.effective_weight().reshape(-1).tolist()).issubset(
                {-1, 0, 1}
            )
        )

    def test_rmsnorm_gain_is_one_packed_row_with_channelwise_learning(self):
        norm = RMSNorm(5)
        self.assertEqual(norm.scale_delta._packed_forward_weight.shape, (1, 2))
        self.assertEqual(norm.scale_delta.effective_weight().shape, (1, 5))
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        expected = inputs * torch.rsqrt(inputs.square().mean(dim=-1, keepdim=True) + 1e-6)
        actual = norm(inputs)
        self.assertTrue(torch.allclose(actual, expected))
        norm.scale_delta.online_learning_rate = 1000.0
        before = norm.scale_delta._packed_forward_weight.clone()
        (-100.0 * actual.sum()).backward()
        self.assertFalse(torch.equal(before, norm.scale_delta._packed_forward_weight))
        self.assertEqual(tuple(norm.parameters()), ())

    def test_right_padded_batch_matches_individual_workspace_and_expert_routes(self):
        config = OmniConfig.micro(dropout=0.0, max_seq_len=24)
        model = OmniDecoder(config).eval()
        model.grow_expert(torch.linspace(-1.0, 1.0, config.d_model))
        model.grow_expert(torch.linspace(1.0, -1.0, config.d_model))
        rows = [
            torch.tensor([1, 259, 40, 41, 260], dtype=torch.long),
            torch.tensor([1, 259, 88, 260], dtype=torch.long),
        ]
        memory = torch.randn(2, config.idea_dim)
        input_ids = torch.zeros((2, 5), dtype=torch.long)
        attention_mask = torch.zeros((2, 5), dtype=torch.bool)
        for index, row in enumerate(rows):
            input_ids[index, : row.numel()] = row
            attention_mask[index, : row.numel()] = True

        with torch.no_grad():
            individual = [
                model(
                    row.unsqueeze(0),
                    memory_bias=memory[index : index + 1],
                    use_global_workspace=True,
                )
                for index, row in enumerate(rows)
            ]
            batched = model(
                input_ids,
                memory_bias=memory,
                use_global_workspace=True,
                attention_mask=attention_mask,
            )

        for index, row in enumerate(rows):
            final = int(row.numel()) - 1
            self.assertTrue(
                torch.allclose(
                    batched["hidden"][index, final],
                    individual[index]["hidden"][0, -1],
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    batched["workspace"][index],
                    individual[index]["workspace"][0],
                    atol=1e-6,
                )
            )
            self.assertTrue(
                torch.allclose(
                    batched["expert_routing"][index],
                    individual[index]["expert_routing"][0],
                    atol=1e-6,
                )
            )

        with self.assertRaisesRegex(ValueError, "right padding"):
            model(
                input_ids,
                attention_mask=torch.tensor(
                    [[True, False, True, False, False]] * 2
                ),
            )

    def test_tiny_ternary_decoder_can_overfit(self):
        config = OmniConfig.micro(
            dropout=0.0,
            learning_rate=0.01,
            max_seq_len=32,
        )
        model = OmniDecoder(config)
        tokenizer = ByteTokenizer()
        ids = torch.tensor(
            [tokenizer.dialogue("hi", "hello", complete=True)], dtype=torch.long
        )
        optimizer = adamw_for_remaining_parameters(
            model.parameters(), lr=0.01, weight_decay=0.01
        )
        packed = [
            module
            for module in model.modules()
            if isinstance(module, PACKED_AUTHORITATIVE_PROJECTION_TYPES)
        ]
        before = [
            tuple(tensor.clone() for tensor in module.authoritative_packed_tensors())
            for module in packed
        ]
        with torch.no_grad():
            initial = float(model(ids, labels=ids)["loss"])
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, labels=ids)["loss"]
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = float(model(ids, labels=ids)["loss"])
        self.assertTrue(
            any(
                not torch.equal(previous, current)
                for module, snapshot in zip(packed, before)
                for previous, current in zip(
                    snapshot, module.authoritative_packed_tensors()
                )
            )
        )
        self.assertLess(final, initial)

    def test_continuing_dialogue_training_strengthens_a_slang_response(self):
        config = OmniConfig.micro(
            dropout=0.0,
            learning_rate=0.01,
            max_seq_len=32,
        )
        model = OmniDecoder(config)
        tokenizer = ByteTokenizer()
        prompt = tokenizer.dialogue("hey", "", complete=False)
        sequence = torch.tensor(
            [tokenizer.dialogue("hey", "yo fam", complete=True)], dtype=torch.long
        )
        next_token = tokenizer.encode("y")[0]

        def probability() -> float:
            with torch.no_grad():
                logits = model(torch.tensor([prompt], dtype=torch.long))["logits"]
                return float(torch.softmax(logits[0, -1], dim=-1)[next_token])

        initial = probability()
        optimizer = adamw_for_remaining_parameters(
            model.parameters(), lr=0.01, weight_decay=0.01
        )
        for _ in range(60):
            optimizer.zero_grad(set_to_none=True)
            loss = model(sequence, labels=sequence)["loss"]
            loss.backward()
            optimizer.step()
        final = probability()
        self.assertGreater(final, initial)

    def test_role_boundaries_are_non_text_special_tokens(self):
        tokenizer = ByteTokenizer()
        ids = tokenizer.dialogue("human", "brain", complete=True)
        self.assertIn(tokenizer.human_id, ids)
        self.assertIn(tokenizer.brain_id, ids)
        self.assertEqual(tokenizer.decode(ids), "humanbrain")

    def test_training_windows_cover_every_utf8_byte_without_truncation(self):
        tokenizer = ByteTokenizer()
        text = ("whole document 🧠 " * 40) + "tail-sentinel"
        windows = list(tokenizer.windows(text, max_length=24))
        payload = [
            token
            for window in windows
            for token in window
            if tokenizer.byte_offset <= token < tokenizer.human_id
        ]
        self.assertGreater(len(windows), 1)
        self.assertEqual(
            bytes(token - tokenizer.byte_offset for token in payload),
            text.encode("utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
