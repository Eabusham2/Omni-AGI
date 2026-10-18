"""Global execution RNG is distinct from the private token sampler."""
import unittest
from tempfile import TemporaryDirectory
import torch
from omni_core.execution_rng import capture_execution_rng, restore_execution_rng
from omni_core.execution_state_codec import ExecutionStateReader, ExecutionStateWriter


class ExecutionRngTests(unittest.TestCase):
    def test_restore_keeps_the_exact_next_global_draw_not_a_sampler_reseed(self):
        original = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, original)
        torch.manual_seed(713)
        saved = capture_execution_rng("cpu")
        expected = torch.randn(17)
        torch.randn(31)
        restore_execution_rng(saved, "cpu")
        self.assertTrue(torch.equal(expected, torch.randn(17)))

    def test_bad_owner_or_geometry_is_rejected_before_global_state_changes(self):
        saved = capture_execution_rng("cpu")
        for value in ({**saved, "device": "mps"}, {**saved, "cpu": torch.zeros(3, dtype=torch.uint8)}, {**saved, "accelerator": torch.zeros(4, dtype=torch.uint8)}):
            with self.assertRaises(ValueError): restore_execution_rng(value, "cpu")
            self.assertTrue(torch.equal(saved["cpu"], torch.get_rng_state()))

    def test_current_execution_codec_round_trip_restores_global_rng_without_model_or_reseed(self):
        original = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, original)
        torch.manual_seed(729)
        with TemporaryDirectory(prefix="omni-execution-rng-") as directory:
            writer = ExecutionStateWriter(directory)
            tree = writer.encode({"globalRng": capture_execution_rng("cpu")})
            expected = torch.randn(23)
            torch.randn(37)
            reader = ExecutionStateReader(directory, writer.files, pager=None, device="cpu")
            try:
                restored = reader.decode(tree)
                restore_execution_rng(restored["globalRng"], "cpu")
                self.assertTrue(torch.equal(expected, torch.randn(23)))
                self.assertEqual(next(iter(writer.files.values()))["dtype"], "torch.uint8")
            finally:
                reader.close()


if __name__ == "__main__": unittest.main()
