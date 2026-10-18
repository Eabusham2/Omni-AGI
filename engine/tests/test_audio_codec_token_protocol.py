"""Tiny native residual-code prediction checks; no brain or corpus training."""

import unittest

import torch

from omni_core.modalities import (
    PackedTernaryTable,
    _codec_token_logits,
    _codec_token_loss,
    _sample_codec_ids,
)


class AudioCodecTokenProtocolTests(unittest.TestCase):
    def codebook(self, values):
        levels = torch.tensor(values, dtype=torch.int8)
        table = PackedTernaryTable(*levels.shape, scale=1.)
        table.projection.set_ternary_weight_(levels.t().contiguous())
        return table

    def test_logits_identify_actual_nearest_code_ids(self):
        predicted = torch.tensor([[[0.9, 0.0, -0.9], [0.1, 0.9, 0.1]]])
        codebook = self.codebook([[1, 0], [0, 1], [-1, 0]])
        logits = _codec_token_logits(predicted, codebook)
        self.assertEqual(tuple(logits.shape), (1, 3, 3))
        self.assertEqual(logits.argmax(dim=-1).tolist(), [[0, 1, 2]])

    def test_discrete_objective_only_teaches_hidden_target_positions(self):
        predicted = torch.tensor([[[0.1, 0.2], [0.1, -0.2]]], requires_grad=True)
        first = torch.zeros_like(predicted)
        first_ids = torch.tensor([[0, 1]])
        residual_ids = torch.tensor([[1, 0]])
        unknown = torch.tensor([[False, True]])
        first_codebook = self.codebook([[1, 0], [0, 1]])
        residual_codebook = self.codebook([[-1, 0], [0, -1]])
        before = [{name: value.clone() for name, value in item.state_dict().items()}
                  for item in (first_codebook, residual_codebook)]
        loss = _codec_token_loss(predicted, first_ids, residual_ids, first,
                                 first_codebook, residual_codebook, unknown)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertEqual(predicted.grad[:, :, 0].tolist(), [[0.0, 0.0]])
        self.assertGreater(float(predicted.grad[:, :, 1].abs().sum()), 0.0)
        for snapshot, table in zip(before, (first_codebook, residual_codebook)):
            for name, value in table.state_dict().items():
                self.assertTrue(torch.equal(value, snapshot[name]), name)

    def test_sampling_is_seeded_and_returns_codebook_ids(self):
        logits = torch.tensor([[[50.0, -50.0], [-50.0, 50.0]]])
        first = torch.Generator(device="cpu").manual_seed(19)
        second = torch.Generator(device="cpu").manual_seed(19)
        ids = _sample_codec_ids(logits, first)
        self.assertEqual(ids.tolist(), [[0, 1]])
        self.assertTrue(torch.equal(ids, _sample_codec_ids(logits, second)))


if __name__ == "__main__":
    unittest.main()
