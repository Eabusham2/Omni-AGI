"""Tiny direct-packed floating-activity checks; no brain or corpus training."""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from omni_core.model import pack_ternary_weight
from omni_core.offload import NeuralStateResourcePause
from omni_core.packed_native_kernel import native_packed_kernel_available, packed_forward_float
from omni_core.router_state_paging import RouterStatePager
from omni_core.spiking import STDPSynapses
from .test_router_state_paging import owner as dense_owner
from .test_sparse_router_state import sparse_owner


@contextmanager
def forbid_weight_expansion():
    """Fail if recurrence reintroduces any former Python weight expansion."""
    with ExitStack() as stack:
        for name in (
            "omni_core.spiking.unpack_tile",
            "omni_core.spiking.unpack_ternary_weight_rows",
            "omni_core.sparse_router_state.unpack_tile",
            "omni_core.sparse_router_state.weighted_ternary_levels",
            "omni_core.router_state_paging.unpack_tile",
            "omni_core.model.unpack_ternary_weight_rows",
            "omni_core.packed_connections.weighted_ternary_levels",
            "torch.mv",
        ):
            stack.enter_context(patch(name, side_effect=AssertionError("expanded weight path: " + name)))
        stack.enter_context(patch.object(STDPSynapses, "iter_effective_tiles",
            side_effect=AssertionError("decoded tile traversal")))
        yield


def repeated_fixture(pager=None, *, reverse=False):
    """Known matrices, including border blocks and noncontiguous layer IDs."""
    value = sparse_owner(pager, rows=67, columns=69)
    matrix = torch.zeros(67, 69, dtype=torch.int64)
    positions = [(1, 1, 0), (0, 1, 0), (0, 0, 12), (0, 0, 2), (0, 0, 0)]
    if reverse:
        positions.reverse()
    with value._operation(), value._sparse_state.transaction():
        for row, column, layer in positions:
            block = value._sparse_state.ensure(row, column, layer=layer)
            generator = torch.Generator().manual_seed(1300 + row * 100 + column * 20 + layer)
            signs = torch.randint(-1, 2, (block.rows, block.columns), dtype=torch.int8, generator=generator)
            states = torch.randint(0, 256, signs.shape, dtype=torch.uint8, generator=generator)
            block._packed_weights.copy_(pack_ternary_weight(signs))
            block._connection_state.copy_(states)
            matrix[row*64:row*64+block.rows, column*64:column*64+block.columns].add_(
                signs.to(torch.int64) * ((states.to(torch.int64) & 15) + 1))
    return value, matrix


@unittest.skipUnless(native_packed_kernel_available(), "build native CPU library first")
class PackedFloatCPUFixtures(unittest.TestCase):
    def test_fractional_weighted_odd_widths_are_exact_without_expansion(self):
        generator = torch.Generator().manual_seed(813)
        for width in (1, 3, 4, 5, 65, 129):
            with self.subTest(width=width):
                signs = torch.randint(-1, 2, (7, width), dtype=torch.int8, generator=generator)
                states = torch.randint(0, 256, signs.shape, dtype=torch.uint8, generator=generator)
                activity = torch.randint(-63, 64, (3, width), generator=generator).float() / 32
                expected = activity.double() @ (signs.double() * ((states & 15).double() + 1)).t()
                packed = pack_ternary_weight(signs)
                with forbid_weight_expansion():
                    actual = packed_forward_float(activity, packed, states, width, 7)
                    resistance_changed = packed_forward_float(activity, packed, states ^ 0xf0, width, 7)
                self.assertTrue(torch.equal(actual, expected.float()))
                self.assertTrue(torch.equal(actual, resistance_changed))

    def test_double_preserves_sub_float32_and_sub_int8_activity(self):
        signs = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.int8)
        activity = torch.tensor([[1 + 2**-40, 2**-200, -1.2345678901234567]], dtype=torch.float64)
        states = torch.tensor([[0, 15, 255], [15, 2, 0], [0, 0, 15]], dtype=torch.uint8)
        with forbid_weight_expansion():
            result = packed_forward_float(activity, pack_ternary_weight(signs), states, 3, 3)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result[0, 0].item(), 1 + 2**-40)
        self.assertEqual(result[0, 1].item(), -3 * 2**-200)
        self.assertEqual(result[0, 2].item(), activity[0, 2].item() * 16)

    def test_unit_gain_half_and_bfloat16_preserve_input_and_output_dtype(self):
        signs = torch.tensor([[1, -1, 0], [0, 1, 1]], dtype=torch.int8)
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype), forbid_weight_expansion():
                activity = torch.tensor([[.125, -.375, 1.75]], dtype=dtype)
                result = packed_forward_float(activity, pack_ternary_weight(signs), None, 3, 2)
                self.assertEqual(result.dtype, dtype)
                self.assertTrue(torch.equal(result, torch.tensor([[.5, 1.375]], dtype=dtype)))

    def test_bad_activity_buffers_and_reserved_padding_fail_closed(self):
        signs = torch.tensor([[0x55]], dtype=torch.uint8)
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite"):
                packed_forward_float(torch.tensor([[bad, 0., 1.]]), signs, None, 3, 1)
        with self.assertRaisesRegex(ValueError, "floating activity"):
            packed_forward_float(torch.ones(1, 3, dtype=torch.int8), signs, None, 3, 1)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            packed_forward_float(torch.ones(1, 6)[:, ::2], signs, None, 3, 1)
        with self.assertRaisesRegex(ValueError, "connection state"):
            packed_forward_float(torch.ones(1, 3), signs, torch.zeros(1, 3), 3, 1)
        for reserved in (0x57, 0xd5):  # Logical first lane and unused fourth lane.
            with self.subTest(reserved=reserved), self.assertRaisesRegex(ValueError, "reserved"):
                packed_forward_float(torch.ones(1, 3), torch.tensor([[reserved]], dtype=torch.uint8), None, 3, 1)

    def test_empty_batch_and_zero_width(self):
        self.assertEqual(tuple(packed_forward_float(torch.empty(0, 3),
            torch.tensor([[0x55]], dtype=torch.uint8), None, 3, 1).shape), (0, 1))
        result = packed_forward_float(torch.empty(2, 0), torch.empty(3, 0, dtype=torch.uint8), None, 0, 3)
        self.assertTrue(torch.equal(result, torch.zeros(2, 3)))


@unittest.skipUnless(torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader"), "Metal compiler unavailable")
class PackedFloatMetalFixtures(unittest.TestCase):
    def test_dyadic_weighted_activity_matches_exact_reference(self):
        generator = torch.Generator().manual_seed(73)
        for width in (1, 7, 65, 129):
            with self.subTest(width=width):
                signs = torch.randint(-1, 2, (9, width), dtype=torch.int8, generator=generator)
                states = torch.randint(0, 256, signs.shape, dtype=torch.uint8, generator=generator)
                activity = torch.randint(-31, 32, (4, width), generator=generator).float() / 32
                packed = pack_ternary_weight(signs)
                expected = activity.double() @ (signs.double() * ((states & 15).double() + 1)).t()
                with forbid_weight_expansion():
                    actual = packed_forward_float(activity.to("mps"), packed.to("mps"), states.to("mps"), width, 9).cpu()
                    changed = packed_forward_float(activity.to("mps"), packed.to("mps"), (states ^ 0xf0).to("mps"), width, 9).cpu()
                self.assertTrue(torch.equal(actual, expected.float()))
                self.assertTrue(torch.equal(actual, changed))

    def test_general_f32_activity_has_bounded_rounding_difference(self):
        generator = torch.Generator().manual_seed(92)
        signs = torch.randint(-1, 2, (7, 257), dtype=torch.int8, generator=generator)
        states = torch.randint(0, 256, signs.shape, dtype=torch.uint8, generator=generator)
        activity = torch.randn(3, 257, generator=generator)
        packed = pack_ternary_weight(signs)
        cpu = packed_forward_float(activity, packed, states, 257, 7)
        with forbid_weight_expansion():
            metal = packed_forward_float(activity.to("mps"), packed.to("mps"), states.to("mps"), 257, 7).cpu()
        torch.testing.assert_close(metal, cpu, rtol=2e-5, atol=1e-4)

    def test_unit_gain_half_and_reserved_padding(self):
        signs = pack_ternary_weight(torch.tensor([[1, -1, 0]], dtype=torch.int8)).to("mps")
        activity = torch.tensor([[.125, -.375, 1.75]], dtype=torch.float16, device="mps")
        with forbid_weight_expansion():
            actual = packed_forward_float(activity, signs, None, 3, 1).cpu()
        self.assertEqual(actual.dtype, torch.float16)
        self.assertEqual(actual.item(), .5)
        with self.assertRaisesRegex(ValueError, "reserved"):
            packed_forward_float(activity, torch.tensor([[0xd5]], dtype=torch.uint8, device="mps"), None, 3, 1)


@unittest.skipUnless(native_packed_kernel_available(), "build native CPU library first")
class PackedRouterRecurrenceFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="packed-float-router-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def pager(self, *, hot=0):
        pager = RouterStatePager(self.root, hot_bytes=hot, tile_bytes=2048, journal_ram_bytes=0)
        self.addCleanup(pager.close)
        return pager

    def test_sparse_repeated_contacts_hot_and_mapped_have_canonical_exact_result(self):
        activity = (torch.arange(69).float() - 34) / 32
        for hot in (0, 1_000_000):
            with self.subTest(hot=hot):
                pager = self.pager(hot=hot)
                value, matrix = repeated_fixture(pager)
                before = value.checksum_with_controls()
                revision = pager.mutation_revision
                expected = matrix.double() @ activity.double()
                identity = {id(block): key for key, block in value.blocks.items()}
                expected_order = [(key, region) for key in sorted(value.blocks)
                    for region in value._sparse_state.regions(value.blocks[key])]
                released = []
                original_release = pager.release_tile

                def release(block, *region):
                    released.append((identity[id(block)], region))
                    return original_release(block, *region)

                def native(activity, signs, states, width, rows):
                    self.assertEqual(signs.dtype, torch.uint8)
                    self.assertEqual(states.dtype, torch.uint8)
                    self.assertEqual(signs.numel(), rows * ((width + 3) // 4))
                    return packed_forward_float(activity, signs, states, width, rows)

                with forbid_weight_expansion(), patch.object(pager, "release_tile", side_effect=release), \
                        patch("omni_core.spiking.packed_forward_float", side_effect=native) as dispatch:
                    actual = value.recurrent(activity)
                self.assertTrue(torch.equal(actual, expected.float()))
                self.assertEqual(released, expected_order)
                self.assertEqual(dispatch.call_count, len(expected_order))
                self.assertEqual(value.checksum_with_controls(), before)
                self.assertEqual(pager.mutation_revision, revision)
                self.assertEqual(pager._active, 0)
                self.assertGreater(pager.status()["mappedLogicalBytes" if hot == 0 else "cpuHeapBytes"], 0)

    def test_insertion_order_cannot_change_repeated_layer_traversal(self):
        left, matrix = repeated_fixture()
        right, reversed_matrix = repeated_fixture(reverse=True)
        self.assertNotEqual(list(left.blocks), list(right.blocks))
        self.assertTrue(torch.equal(matrix, reversed_matrix))
        activity = torch.randn(69, generator=torch.Generator().manual_seed(8))
        with forbid_weight_expansion():
            self.assertTrue(torch.equal(left.recurrent(activity), right.recurrent(activity)))

    def test_unallocated_connectivity_does_not_scan_potential_blocks(self):
        value = sparse_owner(rows=65, columns=100_001)
        with forbid_weight_expansion(), \
                patch("omni_core.sparse_router_state.tile_ranges", side_effect=AssertionError("virtual block scan")), \
                patch("omni_core.spiking.packed_forward_float", side_effect=AssertionError("no allocated block")):
            actual = value.recurrent(torch.full((100_001,), .125))
        self.assertTrue(torch.equal(actual, torch.zeros(65)))
        self.assertEqual(len(value.blocks), 0)

    def test_dense_compatibility_preserves_fractional_double_noncontiguous_activity(self):
        value = dense_owner(rows=3, columns=5)
        levels = torch.tensor([[1, 0, 0, 0, 0], [0, -1, 0, 0, 0], [-1, 1, 0, 1, -1]], dtype=torch.int8)
        value.set_effective_weights(levels)
        activity = torch.tensor([1+2**-40, 9, 2**-200, 9, .125, 9, -.375, 9, 1.25, 9], dtype=torch.float64)[::2]
        self.assertFalse(activity.is_contiguous())
        expected = levels.double() @ activity
        with forbid_weight_expansion():
            actual = value.recurrent(activity)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual.dtype, torch.float64)
        self.assertEqual(actual[0].item(), 1+2**-40)
        self.assertEqual(actual[1].item(), -2**-200)

    def test_kernel_failure_releases_active_mapped_tile_without_mutation(self):
        for sparse in (False, True):
            with self.subTest(sparse=sparse):
                pager = self.pager()
                value = sparse_owner(pager) if sparse else dense_owner(pager)
                value.set_effective_weights(torch.ones(3, 5, dtype=torch.int8))
                before = value.checksum_with_controls()
                with forbid_weight_expansion(), \
                        patch("omni_core.spiking.packed_forward_float", side_effect=RuntimeError("injected native failure")), \
                        patch.object(pager, "release_tile", wraps=pager.release_tile) as release:
                    with self.assertRaisesRegex(RuntimeError, "injected native failure"):
                        value.recurrent(torch.full((5,), .125))
                self.assertEqual(release.call_count, 1)
                self.assertEqual(pager._active, 0)
                self.assertEqual(value.checksum_with_controls(), before)

    def test_cancellation_releases_completed_tile_and_stops_before_next_dispatch(self):
        pager = self.pager()
        value, _ = repeated_fixture(pager)
        before = value.checksum_with_controls()
        cancelled = False
        pager.cancelled = lambda: cancelled

        def native(*args):
            nonlocal cancelled
            result = packed_forward_float(*args)
            cancelled = True
            return result

        with forbid_weight_expansion(), patch("omni_core.spiking.packed_forward_float", side_effect=native) as dispatch, \
                patch.object(pager, "release_tile", wraps=pager.release_tile) as release:
            with self.assertRaisesRegex(InterruptedError, "tile boundary"):
                value.recurrent(torch.full((69,), .125))
        cancelled = False
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(release.call_count, 1)
        self.assertEqual(pager._active, 0)
        self.assertEqual(value.checksum_with_controls(), before)

    def test_tile_admission_refusal_precedes_native_dispatch_and_releases(self):
        pager = self.pager()
        value = sparse_owner(pager)
        value.set_effective_weights(torch.ones(3, 5, dtype=torch.int8))
        before = value.checksum_with_controls()
        original_ram = pager.ram
        active = []

        @contextmanager
        def ram(amount, operation):
            if operation == "allocated router packed sign/gain tile":
                raise NeuralStateResourcePause("injected tile refusal", {})
            with original_ram(amount, operation):
                active.append(operation)
                try:
                    yield
                finally:
                    active.remove(operation)

        with forbid_weight_expansion(), patch.object(pager, "ram", side_effect=ram), \
                patch.object(pager, "release_tile", wraps=pager.release_tile) as release, \
                patch("omni_core.spiking.packed_forward_float") as dispatch:
            with self.assertRaisesRegex(NeuralStateResourcePause, "injected tile refusal"):
                value.recurrent(torch.full((5,), .125))
        dispatch.assert_not_called()
        self.assertEqual(release.call_count, 1)
        self.assertEqual(active, [])
        self.assertEqual(pager._active, 0)
        self.assertEqual(value.checksum_with_controls(), before)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_live_vectors_retain_existing_cpu_router_placement(self):
        value = sparse_owner()
        value.set_effective_weights(torch.tensor([[1, -1, 0, 0, 1]] * 3, dtype=torch.int8))
        activity = torch.tensor([.125, .25, .5, 1., 2.], device="mps")

        def native(activity, signs, states, width, rows):
            self.assertEqual(activity.device.type, "cpu")
            self.assertEqual(signs.device.type, "cpu")
            self.assertEqual(states.device.type, "cpu")
            return packed_forward_float(activity, signs, states, width, rows)

        with forbid_weight_expansion(), patch("omni_core.spiking.packed_forward_float", side_effect=native):
            actual = value.recurrent(activity)
        self.assertEqual(actual.device.type, "mps")
        self.assertTrue(torch.equal(actual.cpu(), torch.full((3,), 1.875)))


if __name__ == "__main__":
    unittest.main()
