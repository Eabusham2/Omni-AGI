import tempfile
import unittest
from pathlib import Path

import torch

from omni_core.ternary_packing import (
    ConnectionStateSource, TernaryTensorSpec, TernaryPackingError,
    export_ternary_shards, inspect_module_ternary_layout, verify_ternary_shards,
)


class StreamedConnectionTests(unittest.TestCase):
    def spec(self, pages, shape=(1, 6)):
        return TernaryTensorSpec(name="substrate.dynamic_synapses.weights",
            values=torch.tensor([-1, 0, 1, 1, -1, 0], dtype=torch.int8),
            kind="dynamic-synapse", connection_state=ConnectionStateSource(shape, pages))

    def test_bounded_pages_round_trip_and_layout_without_resident_matrix(self):
        spec = self.spec(lambda: iter((torch.tensor([0, 15], dtype=torch.uint8),
                                      torch.tensor([128, 255, 14, 8], dtype=torch.uint8))))
        self.assertEqual(inspect_module_ternary_layout({}, dynamic_synapses={spec.name: spec}),
                         {spec.name: ((6,), "dynamic-synapse")})
        with tempfile.TemporaryDirectory() as temporary:
            export_ternary_shards(Path(temporary), [spec])
            loaded = verify_ternary_shards(Path(temporary))
            self.assertTrue(torch.equal(loaded.connection_states[spec.name],
                torch.tensor([[0, 15, 128, 255, 14, 8]], dtype=torch.uint8)))

    def test_rejects_missing_extra_float_and_oversized_pages(self):
        for page in (torch.zeros(5, dtype=torch.uint8), torch.zeros(7, dtype=torch.uint8),
                     torch.zeros(6), torch.zeros((1 << 20) + 1, dtype=torch.uint8)):
            with self.subTest(shape=page.shape, dtype=page.dtype), tempfile.TemporaryDirectory() as temporary:
                spec = self.spec(lambda: iter((page,)))
                with self.assertRaises(TernaryPackingError):
                    export_ternary_shards(Path(temporary), [spec])
                self.assertFalse(Path(temporary, "manifest.json").exists())

    def test_rejects_same_count_wrong_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(TernaryPackingError, "matching"):
                export_ternary_shards(Path(temporary), [self.spec(
                    lambda: iter((torch.zeros(6, dtype=torch.uint8),)), (2, 3))])

    def test_source_generation_change_aborts_before_manifest_publication(self):
        def pages():
            yield torch.zeros(3, dtype=torch.uint8)
            raise RuntimeError("graph generation changed")
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "generation changed"):
                export_ternary_shards(Path(temporary), [self.spec(pages)])
            self.assertFalse(Path(temporary, "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
