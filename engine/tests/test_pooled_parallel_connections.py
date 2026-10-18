"""Bounded contact pages, stable identities, and direct packed computation."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from omni_core.model import PackedAdaptiveBitLinear, pack_ternary_weight
from omni_core.packed_connections import CONTACT_PAGE_CAPACITY, PooledParallelConnections
from omni_core.native_core_paging import NativeCorePager
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded, BoundedTensorFile, load_module_bounded
from omni_core.ternary_packing import collect_module_ternary_tensors, export_module_ternary_shards, verify_ternary_shards, ParallelConnectionSource


class PooledParallelConnectionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(187)

    def test_empty_contact_range_does_not_sample_system_or_reserve_cursors(self):
        layer = PackedAdaptiveBitLinear(3, 2)
        pool = layer.parallel_connections
        def unexpected(*args):
            self.fail("empty contact traversal attempted a resource probe")
        pool._native_core_pager = SimpleNamespace(shared_resource_policy=SimpleNamespace(reserve_ram=unexpected))
        self.assertEqual(list(pool.iter_records()), [])

    def test_thousands_of_contacts_use_pages_and_roundtrip_without_per_contact_modules(self):
        model = PackedAdaptiveBitLinear(3, 2, scale=1.).eval()
        model.fill_ternary_(0)
        for index in range(CONTACT_PAGE_CAPACITY + 3):
            model.add_parallel_connection_(index % 2, index % 3, sign=1 if index % 2 else -1, strength=index % 16 + 1)
        pool = model.parallel_connections
        self.assertEqual(len(pool), CONTACT_PAGE_CAPACITY + 3)
        self.assertEqual(len(pool.pages), 2)
        self.assertLess(len(tuple(model.modules())), 8)
        self.assertEqual(pool.storage_status()['parallelConnectionAllocatedTensorBytes'], 8 + 2 * CONTACT_PAGE_CAPACITY * 45 // 4)
        self.assertNotIn('_sort_order', ' '.join(model.state_dict()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.safetensors'
            atomic_save_tensors_bounded(path, model.state_dict())
            pager = NativeCorePager(Path(directory) / 'pages', cpu_hot_bytes=0, accelerator_hot_bytes=0)
            with pager.construction(from_checkpoint=True):
                restored = PackedAdaptiveBitLinear(3, 2, scale=1.).eval()
            load_module_bounded(restored, BoundedTensorFile(path), '')
            pager.finish_load()
            self.assertEqual(restored.parallel_connection_records(), model.parallel_connection_records())
            self.assertTrue(torch.equal(model(torch.ones(2, 3)), restored(torch.ones(2, 3))))
            self.assertEqual(pager.status()['registeredPackedOwners'], 3)
            pager.close()

    def test_deleted_contact_ids_never_alias_later_births(self):
        layer = PackedAdaptiveBitLinear(1, 1, scale=1.).eval()
        layer.fill_ternary_(0)
        first = layer.add_parallel_connection_(0, 0, sign=1)
        second = layer.add_parallel_connection_(0, 0, sign=1)
        layer.restore_parallel_connections_([layer.parallel_connection_records()[1]])
        third = layer.add_parallel_connection_(0, 0, sign=-1)
        self.assertEqual((first, second, third), ('w_r0_c0_n0', 'w_r0_c0_n1', 'w_r0_c0_n2'))
        with self.assertRaisesRegex(ValueError, 'changed endpoints'):
            layer.parallel_connections.update(first, 0x56, 0)
        restored = PackedAdaptiveBitLinear(1, 1, scale=1.).eval()
        restored.load_state_dict(layer.state_dict())
        self.assertEqual(restored.add_parallel_connection_(0, 0, sign=1), 'w_r0_c0_n3')

    def test_fresh_wider_region_copy_preserves_weight_and_bias_contact_ids(self):
        source = PackedAdaptiveBitLinear(2, 2, bias=True, scale=1.).eval()
        source.add_parallel_connection_(1, 1, sign=-1, strength=4)
        source.add_parallel_connection_(0, 1, sign=1, strength=7, bias=True)
        target = PackedAdaptiveBitLinear(3, 4, bias=True, scale=1.).eval()
        target.copy_parallel_connections_from_(source)
        self.assertEqual(target.parallel_connection_records(), source.parallel_connection_records())
        self.assertEqual(target.parallel_connection_records(bias=True), source.parallel_connection_records(bias=True))
        for key, value in source.parallel_connections.state_dict().items():
            self.assertTrue(torch.equal(target.parallel_connections.state_dict()[key], value))
        with self.assertRaisesRegex(ValueError, 'fresh nonshrinking'):
            target.copy_parallel_connections_from_(source)

    def test_export_is_streamed_and_rejects_mutation_between_collection_and_write(self):
        model = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
        model.add_parallel_connection_(0, 1, sign=-1, strength=5)
        spec = collect_module_ternary_tensors({'cortex': model})[0]
        self.assertIsInstance(spec.parallel_connections, ParallelConnectionSource)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(model, 'parallel_connection_records', side_effect=AssertionError('full contact list is forbidden')):
                export_module_ternary_shards(Path(directory), {'cortex': model})
            self.assertEqual(verify_ternary_shards(Path(directory)).parallel_connections['cortex.weight'], ((0, 1, 0, 0x54, 4),))
        model.add_parallel_connection_(1, 1, sign=1)
        with self.assertRaisesRegex(ValueError, 'changed during streamed export'):
            list(spec.parallel_connections.records())

    def test_forward_and_input_gradient_never_expand_weight_rows(self):
        model = PackedAdaptiveBitLinear(3, 2, scale=1.).eval()
        model._packed_forward_weight.copy_(pack_ternary_weight(torch.tensor([[1, -1, 0], [-1, 0, 1]], dtype=torch.int8)))
        model.set_connection_strengths_(torch.tensor([[2, 3, 1], [4, 1, 2]], dtype=torch.int16))
        model.add_parallel_connection_(0, 2, sign=1, strength=5)
        model.add_parallel_connection_(1, 0, sign=1, strength=3)
        inputs = torch.ones(1, 3, requires_grad=True)
        with patch('omni_core.model.unpack_ternary_weight_rows', side_effect=AssertionError('weight expansion is forbidden')):
            output = model(inputs)
            output.sum().backward()
        self.assertEqual(output.detach().tolist(), [[4., 1.]])
        self.assertEqual(inputs.grad.tolist(), [[1., -3., 7.]])

    def test_paged_deepcopy_preserves_contacts_without_live_pager_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            pager = NativeCorePager(Path(directory), cpu_hot_bytes=0, accelerator_hot_bytes=0)
            with pager.construction():
                layer = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
            layer.add_parallel_connection_(1, 0, sign=-1, strength=2)
            target = copy.deepcopy(layer)
            self.assertIsNone(target._native_core_pager)
            self.assertIsNone(target.parallel_connections._native_core_pager)
            self.assertEqual(target.parallel_connection_records(), layer.parallel_connection_records())
            target.add_parallel_connection_(0, 0, sign=1)
            self.assertEqual(len(layer.parallel_connections), 1)
            pager.close()

    def test_rewired_page_has_a_distinct_authoritative_checksum(self):
        from omni_core.persistence import tensor_checksum
        layer = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
        layer.add_parallel_connection_(0, 0, sign=1, strength=2)
        before = tensor_checksum(value for module in layer.modules() if callable(getattr(module, 'authoritative_packed_tensors', None)) for value in module.authoritative_packed_tensors())
        page = next(iter(layer.parallel_connections.pages.values()))
        page._record_meta[0, 2] = 1
        after = tensor_checksum(value for module in layer.modules() if callable(getattr(module, 'authoritative_packed_tensors', None)) for value in module.authoritative_packed_tensors())
        self.assertNotEqual(before, after)

    def test_nonalphabetical_page_birth_order_has_same_checksum_after_reload(self):
        from omni_core.persistence import tensor_checksum
        from omni_core.sparse_router_state import canonical_module_walk
        layer = PackedAdaptiveBitLinear(2, 64, scale=1.).eval()
        layer.add_parallel_connection_(40, 1, sign=1, strength=3)
        layer.add_parallel_connection_(0, 0, sign=-1, strength=2)
        target = PackedAdaptiveBitLinear(2, 64, scale=1.).eval()
        target.load_state_dict(layer.state_dict())
        def checksum(model):
            return tensor_checksum(value for _name, module in canonical_module_walk(model)
                if callable(getattr(module, 'authoritative_packed_tensors', None))
                for value in module.authoritative_packed_tensors())
        self.assertEqual(checksum(layer), checksum(target))


if __name__ == '__main__':
    unittest.main()
