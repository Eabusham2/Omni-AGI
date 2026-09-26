"""Focused tests for authoritative packed token/symbol embeddings."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.model import (  # noqa: E402
    OmniDecoder,
    PackedAdaptiveTernaryEmbedding,
    pack_ternary_weight,
    packed_runtime_status,
)
from omni_core.config import OmniConfig  # noqa: E402
from omni_core.ternary_packing import (  # noqa: E402
    export_module_ternary_shards,
    inspect_module_ternary_layout,
    verify_ternary_shards,
)


class PackedAdaptiveEmbeddingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(83)
        torch.set_num_threads(1)

    def test_backward_updates_only_touched_nonpadding_rows_and_reloads(self):
        layer = PackedAdaptiveTernaryEmbedding(
            9, 5, padding_idx=0, scale=1.0, online_learning_rate=100.0
        )
        self.assertEqual(tuple(layer.parameters()), ())
        self.assertEqual(layer.logical_ternary_parameter_count, 45)
        self.assertEqual(layer.authoritative_packed_tensors()[0].numel(), 18)
        self.assertEqual(packed_runtime_status(layer)["packedAuthoritativeEmbeddingModules"], 1)
        self.assertEqual(
            set(layer.state_dict()),
            {"_packed_forward_weight", "_packed_forward_scale", "_online_learning_rate"},
        )
        with torch.no_grad():
            layer._packed_forward_weight[3:4].copy_(
                pack_ternary_weight(torch.zeros((1, 5), dtype=torch.int8))
            )
        before = layer.effective_weight().clone()
        indices = torch.tensor([[0, 3, 3, 4]], dtype=torch.long)
        output = layer(indices)
        self.assertEqual(tuple(output.shape), (1, 4, 5))
        self.assertTrue(output.requires_grad)
        loss = F.mse_loss(output[:, 1:3], torch.ones((1, 2, 5)))
        loss.backward()
        after = layer.effective_weight()
        self.assertTrue(bool((after[3] == 1).all()))
        self.assertTrue(torch.equal(after[0], before[0]))
        self.assertTrue(torch.equal(after[4], before[4]))
        self.assertTrue(bool((after[0] == 0).all()))
        restored = PackedAdaptiveTernaryEmbedding(9, 5, padding_idx=0)
        restored.load_state_dict(layer.state_dict())
        self.assertTrue(torch.equal(restored.effective_weight(), after))
        self.assertTrue(torch.equal(restored.eval()(indices), layer.eval()(indices)))
        self.assertEqual(tuple(restored(torch.empty((0,), dtype=torch.long)).shape), (0, 5))

    def test_layout_export_and_corrupt_checkpoint_fail_closed(self):
        layer = PackedAdaptiveTernaryEmbedding(7, 3, padding_idx=0)
        self.assertEqual(
            inspect_module_ternary_layout({"token": layer}, dynamic_synapses={}),
            {"token.weight": ((7, 3), "projection")},
        )
        with tempfile.TemporaryDirectory() as temporary:
            export_module_ternary_shards(
                Path(temporary), {"token": layer}, expected_names={"token.weight"}
            )
            verified = verify_ternary_shards(
                Path(temporary), expected_names={"token.weight"}
            )
            self.assertTrue(
                torch.equal(verified.tensors["token.weight"], layer.effective_weight())
            )
        corrupted = {name: value.clone() for name, value in layer.state_dict().items()}
        corrupted["_packed_forward_weight"][1, 0] = 0xFF
        layer.load_state_dict(corrupted)
        with self.assertRaisesRegex(ValueError, "reserved code"):
            layer(torch.tensor([1]))

    def test_native_decoder_exports_both_packed_embedding_tables(self):
        decoder = OmniDecoder(OmniConfig.micro(dropout=0.0))
        self.assertIsInstance(decoder.embedding, PackedAdaptiveTernaryEmbedding)
        self.assertIsInstance(
            decoder.action_argument_head.token_embedding,
            PackedAdaptiveTernaryEmbedding,
        )
        runtime = packed_runtime_status(decoder)
        self.assertTrue(runtime["complete"])
        self.assertEqual(runtime["packedAuthoritativeEmbeddingModules"], 3)
        self.assertEqual(runtime["denseEmbeddingBlockers"], [])
        layout = inspect_module_ternary_layout(
            {"decoder": decoder}, dynamic_synapses={}
        )
        self.assertEqual(
            layout["decoder.embedding.weight"],
            ((decoder.config.vocab_size, decoder.config.d_model), "projection"),
        )
        self.assertEqual(
            layout["decoder.action_argument_head.token_embedding.weight"],
            ((258, decoder.config.d_model), "projection"),
        )
        self.assertEqual(
            layout["decoder.global_workspace.latent_table.weight"],
            (
                (decoder.global_workspace.slots, decoder.config.d_model),
                "projection",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            export_module_ternary_shards(
                Path(temporary), {"decoder": decoder}, expected_names=set(layout)
            )
            verified = verify_ternary_shards(
                Path(temporary), expected_names=set(layout)
            )
            self.assertTrue(
                torch.equal(
                    verified.tensors["decoder.embedding.weight"],
                    decoder.embedding.effective_weight(),
                )
            )


if __name__ == "__main__":
    unittest.main()
