"""Constructor-free saved-tensor fixtures; no brain, model, or training job."""

import gc
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.offload import NeuralStateResourcePause
from omni_core.working_attention_paging import (
    SavedActivityTensor,
    WorkingAttentionCancelled,
    WorkingAttentionPager,
)


class SavedActivationMappingFixtures(unittest.TestCase):
    def pager(self, directory: str, *, tile: int = 256) -> WorkingAttentionPager:
        return WorkingAttentionPager(
            Path(directory), resident_budget_bytes=0,
            scratch_budget_bytes=8 * 1024 * 1024,
            device_tile_budget_bytes=tile,
        )

    def images(self, directory: str):
        return list(Path(directory).glob("*.saved-activation"))

    def test_large_noncontiguous_cpu_restore_has_no_whole_anonymous_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            source = torch.arange(609, dtype=torch.float64).reshape(7, 29, 3).transpose(0, 1)
            saved = pager.save_activation(source)
            self.assertGreater(source.numel() * source.element_size(), pager.device_tile_budget_bytes)
            with patch("torch.empty", side_effect=AssertionError("anonymous restore allocation")), patch(
                "torch.cat", side_effect=AssertionError("whole restore concatenation")
            ):
                restored = saved.restore()
            self.assertEqual(restored.shape, source.shape)
            self.assertEqual(restored.dtype, source.dtype)
            self.assertTrue(torch.equal(restored, source))
            self.assertEqual(restored.device.type, "cpu")
            self.assertEqual(len(self.images(directory)), 1)
            self.assertLessEqual(pager.status()["largestReadBytes"], pager.device_tile_budget_bytes // 4)
            self.assertLessEqual(pager.status()["peakComputeTileBytes"], pager.device_tile_budget_bytes)
            saved.close()
            del restored
            gc.collect()
            pager.close()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_private_restore_mutation_cannot_rewrite_saved_bytes_or_other_restores(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            source = torch.arange(129, dtype=torch.float32)
            saved = pager.save_activation(source)
            first, second = saved.restore(), saved.restore()
            image = self.images(directory)[0]
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            first[0] = -999
            self.assertEqual(second[0].item(), source[0].item())
            third = saved.restore()
            self.assertTrue(torch.equal(third, source))
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), digest)
            self.assertEqual(len(self.images(directory)), 1)
            saved.close()
            del first, second, third
            gc.collect()
            pager.close()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_tensor_views_own_backing_file_beyond_saved_hook_and_pager_close(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            source = torch.arange(257, dtype=torch.float64)
            saved = pager.save_activation(source)
            restored = saved.restore()
            surviving_view = restored[1::2]
            image = self.images(directory)[0]
            saved.close()
            pager.close()
            del restored
            gc.collect()
            self.assertTrue(image.is_file())
            self.assertGreater(sum(pager._owned_spill_leases.values()), 0)
            self.assertTrue(torch.equal(surviving_view, source[1::2]))
            del surviving_view
            gc.collect()
            self.assertFalse(image.exists())
            self.assertEqual(pager._owned_spill_leases, {})
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_exact_dtype_shape_and_complex_conjugate_round_trips(self):
        for dtype in (
            torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16,
            torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128,
        ) + tuple(
            getattr(torch, name) for name in (
                "uint16", "uint32", "uint64", "float8_e4m3fn", "float8_e5m2",
            ) if hasattr(torch, name)
        ):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                pager = self.pager(directory)
                source = torch.arange(65, dtype=torch.float32).reshape(5, 13)
                if dtype in (torch.complex64, torch.complex128):
                    source = (source + source * 1j).to(dtype).conj()
                elif dtype == torch.bool:
                    source = source.remainder(2).to(dtype)
                else:
                    source = source.to(dtype)
                source = source.transpose(0, 1)
                saved = pager.save_activation(source)
                restored = saved.restore()
                self.assertEqual(restored.dtype, dtype)
                self.assertEqual(restored.shape, source.shape)
                self.assertTrue(torch.equal(
                    restored.view(torch.uint8),
                    source.resolve_conj().resolve_neg().contiguous().view(torch.uint8),
                ))
                saved.close()
                del restored
                gc.collect()
                pager.close()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_scalar_and_zero_element_tensors_preserve_shape(self):
        for source in (torch.tensor(3.25, dtype=torch.float64), torch.empty(2, 0, 3)):
            with self.subTest(shape=source.shape), tempfile.TemporaryDirectory() as directory:
                pager = self.pager(directory)
                saved = pager.save_activation(source)
                restored = saved.restore()
                self.assertEqual(restored.shape, source.shape)
                self.assertTrue(torch.equal(restored, source))
                self.assertEqual(len(self.images(directory)), int(source.numel() > 0))
                saved.close()
                del restored
                gc.collect()
                pager.close()

    def test_changed_image_checksum_is_revalidated_before_a_new_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = pager.save_activation(torch.arange(129, dtype=torch.float32))
            restored = saved.restore()
            with self.images(directory)[0].open("r+b") as handle:
                handle.write(b"bad!")
            with self.assertRaisesRegex(ValueError, "checksum"):
                saved.restore()
            saved.close()
            del restored
            gc.collect()
            pager.close()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_corrupt_source_page_aborts_image_without_leaking_its_file(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = pager.save_activation(torch.arange(129, dtype=torch.float32))
            page = pager._pages[saved.pages[-1]]
            with page.path.open("r+b") as handle:
                handle.write(b"bad!")
            before = pager.status()["spillBytes"]
            with self.assertRaisesRegex(ValueError, "checksum"):
                saved.restore()
            self.assertEqual(self.images(directory), [])
            self.assertEqual(pager.status()["spillBytes"], before)
            saved.close()
            pager.close()

    def test_missing_source_coverage_never_returns_uninitialized_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = pager.save_activation(torch.arange(129, dtype=torch.float32))
            saved.pages.pop()
            with self.assertRaisesRegex(ValueError, "exact tensor size"):
                saved.restore()
            self.assertEqual(self.images(directory), [])
            saved.close()
            pager.close()

    def test_cancellation_during_bounded_image_transfer_cleans_partial_image(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = pager.save_activation(torch.arange(129, dtype=torch.float32))
            count = 0

            def cancelled():
                nonlocal count
                count += 1
                return count > 7

            with self.assertRaises(WorkingAttentionCancelled), pager.activate(cancelled):
                saved.restore()
            self.assertEqual(self.images(directory), [])
            saved.close()
            pager.close()

    def test_accelerator_restore_retains_full_tensor_minimum_without_allocating(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = pager.save_activation(torch.arange(129, dtype=torch.float32))
            with patch("torch.empty", side_effect=AssertionError("unadmitted accelerator allocation")):
                with self.assertRaisesRegex(NeuralStateResourcePause, "complete accelerator tensor minimum"):
                    saved.restore(torch.device("cuda"))
            self.assertEqual(self.images(directory), [])
            saved.close()
            pager.close()

    def test_explicitly_released_saved_activation_is_not_restorable(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = SavedActivityTensor(pager, torch.arange(129, dtype=torch.float32))
            saved.close()
            with self.assertRaisesRegex(RuntimeError, "released saved activation"):
                saved.restore()
            pager.close()

    def test_restore_image_is_admitted_against_already_occupied_spill_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            saved = pager.save_activation(torch.arange(129, dtype=torch.float32))
            # A settings change leaves exactly the existing source allocation;
            # a second physical image cannot be silently overcommitted.
            pager.scratch_budget_bytes = pager.status()["spillBytes"]
            with self.assertRaisesRegex(NeuralStateResourcePause, "spill pool is full"):
                saved.restore()
            self.assertEqual(self.images(directory), [])
            self.assertEqual(pager._owned_spill_leases, {})
            saved.close()
            pager.close()

    def test_saved_hook_pure_autograd_expression_retains_exact_first_derivative(self):
        # This is a fixed scalar tensor expression, not a model forward,
        # optimizer update, dataset job, or learned capability test.
        with tempfile.TemporaryDirectory() as directory:
            pager = self.pager(directory)
            source = torch.arange(321, dtype=torch.float64, requires_grad=True)
            with pager.activate(), pager.saved_activation_hooks():
                expression = source.square().sum()
            expression.backward()
            self.assertTrue(torch.equal(source.grad, 2 * source.detach()))
            self.assertGreater(pager.status()["cpuSavedActivationMappingCount"], 0)
            del expression
            gc.collect()
            pager.close()
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
