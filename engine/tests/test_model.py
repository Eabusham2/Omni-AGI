import sys
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.config import OmniConfig
from omni_core.model import BitLinear, OmniDecoder
from omni_core.tokenizer import ByteTokenizer


class TernaryDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        torch.set_num_threads(1)

    def test_bitlinear_uses_three_effective_levels_and_ste_gradient(self):
        layer = BitLinear(5, 4)
        values = set(layer.effective_weight().reshape(-1).tolist())
        self.assertTrue(values.issubset({-1, 0, 1}))
        output = layer(torch.randn(2, 5)).sum()
        output.backward()
        self.assertIsNotNone(layer.weight.grad)
        self.assertGreater(float(layer.weight.grad.abs().sum()), 0.0)

    def test_dense_recipe_bypasses_ternary_forward(self):
        layer = BitLinear(3, 2)
        layer.ternary = False
        inputs = torch.randn(2, 3)
        expected = torch.nn.functional.linear(inputs, layer.weight, layer.bias)
        self.assertTrue(torch.allclose(layer(inputs), expected))

    def test_attention_is_causal_and_finite(self):
        config = OmniConfig.micro(dropout=0.0, parallel_thoughts=1)
        model = OmniDecoder(config).eval()
        first = torch.tensor([[1, 10, 11, 12, 13]], dtype=torch.long)
        second = first.clone()
        second[0, -1] = 99
        with torch.no_grad():
            left = model(first)["logits"]
            right = model(second)["logits"]
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.allclose(left[:, :-1], right[:, :-1], atol=1e-6))

    def test_tiny_ternary_decoder_can_overfit(self):
        config = OmniConfig.micro(
            dropout=0.0,
            parallel_thoughts=1,
            learning_rate=0.01,
            max_seq_len=32,
        )
        model = OmniDecoder(config)
        tokenizer = ByteTokenizer()
        ids = torch.tensor(
            [tokenizer.dialogue("hi", "hello", complete=True)], dtype=torch.long
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        with torch.no_grad():
            initial = float(model(ids, labels=ids)["loss"])
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, labels=ids)["loss"]
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            final = float(model(ids, labels=ids)["loss"])
        self.assertLess(final, initial)

    def test_continuing_dialogue_training_strengthens_a_slang_response(self):
        config = OmniConfig.micro(
            dropout=0.0,
            parallel_thoughts=1,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
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


if __name__ == "__main__":
    unittest.main()
