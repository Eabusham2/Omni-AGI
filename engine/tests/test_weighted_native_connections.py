"""Small native contact fixtures: exact gains, discrete learning and reload."""
import sys
import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.model import (PackedAdaptiveBitLinear, PackedAdaptiveTernaryEmbedding,
    PackedAdaptiveBitConv1d, PackedAdaptiveBitConvTranspose1d, pack_ternary_weight,
    PackedAdaptiveBitConv2d, PackedAdaptiveBitConv3d,
    PackedAdaptiveBitConvTranspose2d, PackedAdaptiveBitConvTranspose3d, packed_online_step)
from omni_core.ternary_packing import export_module_ternary_shards, verify_ternary_shards
from omni_core.bounded_tensor_io import atomic_save_tensors_bounded, BoundedTensorFile, load_module_bounded
from omni_core.native_core_paging import NativeCorePager
from omni_core.optimizers import PackedMutationSnapshot
from omni_core.packed_collective import PackedCollectiveController
from omni_core.packed_collective_hooks import packed_derivative_sink, packed_row_owner
from omni_core.parameter_diagnostics import ParameterDeltaJournal
from types import SimpleNamespace


def weighted_collective_rank(rank, size, root, initial_contacts=1):
    """Tiny real two-rank fixture; no decoder or saved brain is built."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(root) / "weighted-rendezvous"), rank=rank, world_size=size)
    try:
        layer = PackedAdaptiveBitLinear(1, 1, bias=True, scale=1.0).eval()
        layer.fill_ternary_(1)
        layer.set_connection_strengths_(torch.tensor([[16]], dtype=torch.int16))
        for _ in range(initial_contacts):
            layer.add_parallel_connection_(0, 0, sign=-1, strength=2)
        context = SimpleNamespace(rank=rank, world_size=size, distributed=True, backend="gloo", device=torch.device("cpu"), is_rank_zero=rank == 0)
        from omni_core.model import _apply_packed_gradient_rows
        collective = PackedCollectiveController({"layer": layer}, context, Path(root) / str(rank), _apply_packed_gradient_rows)
        before = collective.identity()
        collective.begin("weighted-shared-step")
        with packed_derivative_sink(collective):
            layer.learn_from_gradient(torch.ones(1, 1), torch.tensor([[-2. if rank else 1.]]), 1000.)
        collective.commit(retain_rollback=True)
        learned = collective.identity()
        records = layer.parallel_connection_records()
        rollback_chunks = collective.database.execute('SELECT COUNT(*) FROM parallel_rollback').fetchone()[0]
        collective.rollback()
        Path(root, "weighted-result-%d.json" % rank).write_text(json.dumps({
            "learned": learned, "records": records, "restored": before == collective.identity(),
            "rollbackChunks": rollback_chunks,
            "state": layer._connection_state.tolist(), "output": layer(torch.ones(1, 1)).detach().tolist(),
        }))
        collective.close()
    finally:
        dist.destroy_process_group()


class WeightedNativeConnectionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_unit_native_state_migrates_with_identical_output(self):
        source = PackedAdaptiveBitLinear(7, 3, bias=True).eval()
        inputs = torch.randn(2, 7)
        old_state = {key: value.clone() for key, value in source.state_dict().items()
                     if "connection_state" not in key}
        target = PackedAdaptiveBitLinear(7, 3, bias=True).eval()
        target.load_state_dict(old_state)
        self.assertTrue(torch.equal(target(inputs), source(inputs)))
        self.assertEqual(tuple(target.parameters()), ())
        self.assertTrue(torch.equal(target.connection_strengths(), torch.ones(3, 7, dtype=torch.int16)))

    def test_individual_strengths_change_forward_and_input_gradient(self):
        layer = PackedAdaptiveBitLinear(2, 1, scale=1.0).eval()
        layer.set_ternary_weight_(torch.tensor([[1, -1]], dtype=torch.int8))
        layer.set_connection_strengths_(torch.tensor([[2, 3]], dtype=torch.int16))
        inputs = torch.tensor([[1.0, 1.0]], requires_grad=True)
        self.assertEqual(float(layer(inputs).detach()), -1.0)
        layer(inputs).sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.tensor([[2.0, -3.0]])))
        self.assertEqual(layer.effective_weight().tolist(), [[1, -1]])
        self.assertEqual(layer.packed_forward_status()["connectionStateBytes"], 2)

    def test_discrete_learning_strengthens_then_grows_independent_contact(self):
        layer = PackedAdaptiveBitLinear(1, 1, scale=1.0).eval()
        layer.fill_ternary_(1)
        for _ in range(16):
            layer.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 1), 100.0)
        self.assertEqual(int(layer.connection_strengths()[0, 0]), 16)
        self.assertEqual(len(layer.parallel_connections), 1)
        self.assertEqual(float(layer(torch.ones(1, 1))), 17.0)
        self.assertEqual(layer.logical_ternary_parameter_count, 2)
        self.assertEqual(tuple(layer.parameters()), ())
        restored = PackedAdaptiveBitLinear(1, 1, scale=1.0).eval()
        restored.load_state_dict(layer.state_dict())
        self.assertEqual(float(restored(torch.ones(1, 1))), 17.0)
        self.assertEqual(restored.parallel_connection_records(), layer.parallel_connection_records())

    def test_parallel_contacts_have_independent_signs_strengths_and_derivatives(self):
        layer = PackedAdaptiveBitLinear(2, 1, scale=1.0).eval()
        layer.fill_ternary_(0)
        layer.add_parallel_connection_(0, 0, sign=1, strength=3)
        layer.add_parallel_connection_(0, 0, sign=-1, strength=1)
        layer.add_parallel_connection_(0, 1, sign=-1, strength=4)
        inputs = torch.ones(1, 2, requires_grad=True)
        self.assertEqual(float(layer(inputs).detach()), -2.0)
        layer(inputs).sum().backward()
        self.assertEqual(inputs.grad.tolist(), [[2.0, -4.0]])
        layer.fill_ternary_(0)
        self.assertEqual(float(layer(torch.ones(1, 2))), 0.0)
        self.assertEqual(len(layer.parallel_connections), 0)

    def test_embedding_and_both_convolution_directions_use_connection_state(self):
        embedding = PackedAdaptiveTernaryEmbedding(3, 2, initial_level=1, scale=1).eval()
        embedding.set_connection_strengths_(torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int16))
        embedding.add_parallel_connection_(1, 0, sign=-1, strength=2)
        self.assertEqual(embedding(torch.tensor([1])).tolist(), [[1.0, 4.0]])
        for factory in (PackedAdaptiveBitConv1d, PackedAdaptiveBitConvTranspose1d):
            layer = factory(1, 1, 1, bias=True, scale=1.0).eval()
            layer._packed_forward_weight.copy_(pack_ternary_weight(torch.ones(1, 1, dtype=torch.int8)))
            layer.set_connection_strengths_(torch.tensor([[3]], dtype=torch.int16))
            layer.add_parallel_connection_(0, 0, sign=1, strength=2)
            layer.add_parallel_connection_(0, 0, sign=-1, strength=1, bias=True)
            inputs = torch.ones(1, 1, 2, requires_grad=True)
            self.assertEqual(layer(inputs).tolist(), [[[4.0, 4.0]]])
            layer(inputs).sum().backward()
            self.assertEqual(inputs.grad.tolist(), [[[5.0, 5.0]]])

    def test_bounded_saved_state_and_inference_shards_preserve_strength(self):
        layer = PackedAdaptiveBitLinear(2, 1, scale=1.0).eval()
        layer.fill_ternary_(1)
        layer.set_connection_strengths_(torch.tensor([[8, 16]], dtype=torch.int16))
        layer.add_parallel_connection_(0, 1, sign=-1, strength=2)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            atomic_save_tensors_bounded(path / "weights.safetensors", layer.state_dict())
            restored = PackedAdaptiveBitLinear(2, 1, scale=1.0).eval()
            load_module_bounded(restored, BoundedTensorFile(path / "weights.safetensors"), "")
            self.assertTrue(torch.equal(layer(torch.ones(1, 2)), restored(torch.ones(1, 2))))
            export_module_ternary_shards(path / "inference", {"cortex": layer})
            verified = verify_ternary_shards(path / "inference")
            self.assertTrue(torch.equal(verified.connection_states["cortex.weight"], layer._connection_state))
            self.assertEqual(verified.manifest["formatVersion"], 2)
            self.assertEqual(verified.parallel_connections["cortex.weight"], ((0, 1, 0, 0x54, 1),))
            entry = verified.manifest["tensors"][0]["connectionState"]
            (path / "inference" / entry["shard"]).write_bytes(b"\xff\xff")
            with self.assertRaisesRegex(ValueError, "connection-state checksum"):
                verify_ternary_shards(path / "inference", retain_names=())

    def test_collective_grows_and_rolls_back_strength_and_contacts(self):
        layer = PackedAdaptiveBitLinear(1, 1, scale=1.0).eval()
        layer.fill_ternary_(1)
        layer.set_connection_strengths_(torch.tensor([[16]], dtype=torch.int16))
        with tempfile.TemporaryDirectory() as folder:
            context = SimpleNamespace(distributed=False, world_size=1, backend="gloo", device=torch.device("cpu"), is_rank_zero=True)
            collective = PackedCollectiveController({"layer": layer}, context, folder,
                __import__("omni_core.model", fromlist=["_apply_packed_gradient_rows"])._apply_packed_gradient_rows)
            before = collective.identity()
            collective.begin("parallel-birth")
            with packed_derivative_sink(collective), packed_row_owner(layer):
                layer.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 1), 100.0)
            self.assertEqual(len(layer.parallel_connections), 0)
            collective.commit(retain_rollback=True)
            self.assertEqual(float(layer(torch.ones(1, 1))), 17.0)
            collective.rollback()
            self.assertEqual(collective.identity(), before)
            self.assertEqual(len(layer.parallel_connections), 0)
            collective.close()

    def test_mutation_snapshot_and_changed_byte_journal_remove_new_contacts(self):
        layer = PackedAdaptiveBitLinear(1, 1, scale=1.0).eval()
        layer.fill_ternary_(1)
        layer.set_connection_strengths_(torch.tensor([[16]], dtype=torch.int16))
        snapshot = PackedMutationSnapshot.capture([layer])
        layer.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 1), 100.0)
        snapshot.restore()
        self.assertEqual(float(layer(torch.ones(1, 1))), 16.0)
        self.assertEqual(len(layer.parallel_connections), 0)
        snapshot.close()
        with tempfile.TemporaryDirectory() as folder:
            journal = ParameterDeltaJournal((("cortex", layer),), directory=folder,
                reserve_ram=lambda *_: None, reserve_disk=lambda *_: None, include_state=True)
            layer.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 1), 100.0)
            self.assertGreater(journal.delta_norm(), 0)
            journal.restore()
            self.assertEqual(len(layer.parallel_connections), 0)
            self.assertEqual(float(layer(torch.ones(1, 1))), 16.0)
            journal.close()

    def test_paging_spills_strength_and_old_checkpoint_defaults_without_master(self):
        with tempfile.TemporaryDirectory() as folder:
            pager = NativeCorePager(Path(folder) / "pool", cpu_hot_bytes=0, accelerator_hot_bytes=0)
            with pager.construction():
                layer = PackedAdaptiveBitLinear(4, 2, scale=1.0).eval()
            layer.set_ternary_weight_(torch.ones(2, 4, dtype=torch.int8))
            layer.set_connection_strengths_(torch.full((2, 4), 3, dtype=torch.int16))
            layer.add_parallel_connection_(0, 0, sign=1, strength=2)
            before = layer(torch.ones(1, 4)).detach().clone()
            pager.flush()
            pager.cool_to_budget(0)
            self.assertTrue(torch.equal(layer(torch.ones(1, 4)), before))
            self.assertEqual(tuple(layer.parameters()), ())
            pager.close()

    def test_loss_backward_learns_strength_in_authoritative_bytes(self):
        layer = PackedAdaptiveBitLinear(1, 1, scale=1., online_learning_rate=100.).train()
        layer.fill_ternary_(1)
        losses = []
        for _ in range(3):
            output = layer(torch.ones(1, 1))
            loss = (output - 3).square().mean()
            losses.append(float(loss.detach()))
            loss.backward()
        self.assertEqual(losses, [4., 1., 0.])
        self.assertEqual(int(layer._connection_state[0, 0] & 15), 2)
        self.assertEqual(tuple(layer.named_parameters()), ())

    def test_repeated_contact_learning_is_seed_replayable_after_load(self):
        layer = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
        layer.fill_ternary_(0)
        for index in range(12):
            layer.add_parallel_connection_(1, 0, sign=1, strength=1 + index % 4)
        layer.add_parallel_connection_(0, 1, sign=-1, strength=3)
        target = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
        target.load_state_dict(layer.state_dict())
        for model in (layer, target):
            model.learn_from_gradient(torch.tensor([[1., 2.]]), torch.tensor([[.3, -.2]]), .7,
                generator=torch.Generator().manual_seed(710))
        self.assertEqual(target.parallel_connection_records(), layer.parallel_connection_records())
        self.assertTrue(torch.equal(target._connection_state, layer._connection_state))

    def test_loaded_contacts_require_complete_state_before_allocation(self):
        layer = PackedAdaptiveBitLinear(1, 1, bias=True, scale=1.).eval()
        layer.add_parallel_connection_(0, 0, sign=1)
        state = layer.state_dict()
        target = PackedAdaptiveBitLinear(1, 1, bias=True, scale=1.).eval()
        malformed = {key: value for key, value in state.items() if not (key.startswith("parallel_connections.pages.") and key.endswith("_connection_state"))}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            target.load_state_dict(malformed)
        self.assertEqual(len(target.parallel_connections), 0)
        malformed = {key: value for key, value in state.items() if key not in {"_connection_state", "_bias_connection_state"}}
        with self.assertRaisesRegex(ValueError, "require saved base"):
            target.load_state_dict(malformed)
        self.assertEqual(len(target.parallel_connections), 0)

    def test_parallel_restore_validates_endpoints_before_mutation(self):
        layer = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
        layer.add_parallel_connection_(0, 0, sign=1)
        before = layer.parallel_connection_records()
        for records in ([('w_r1_c1_n0', 0, 1, 0x56, 0)],
                        [('w_r1_c1_n0', 1, 1, 0x57, 0)],
                        [('w_r1_c1_n0', 1, 1, 0x56, -1)],
                        [('w_r1_c1_n0', 1, 1, 0x56, 0)] * 2):
            with self.assertRaises(ValueError):
                layer.restore_parallel_connections_(records)
            self.assertEqual(layer.parallel_connection_records(), before)

    def test_growth_and_replica_install_respect_admission_without_partial_birth(self):
        with tempfile.TemporaryDirectory() as folder:
            pager = NativeCorePager(Path(folder), cpu_hot_bytes=0, accelerator_hot_bytes=0)
            with pager.construction():
                layer = PackedAdaptiveBitLinear(1, 1, scale=1.).eval()
            files = set(Path(folder).iterdir())
            def deny(_amount, _device):
                raise MemoryError("contact registry full")
            pager.reserve_admission = deny
            with self.assertRaises(MemoryError):
                layer.add_parallel_connection_(0, 0, sign=1)
            with self.assertRaises(MemoryError):
                layer.restore_parallel_connections_([('w_r0_c0_n0', 0, 0, 0x56, 0)])
            self.assertEqual(len(layer.parallel_connections), 0)
            self.assertEqual(set(Path(folder).iterdir()), files)
            pager.close()

    def test_disk_rollback_retires_new_contact_spill_files_and_preserves_old(self):
        with tempfile.TemporaryDirectory() as folder:
            pager = NativeCorePager(Path(folder) / "pages", cpu_hot_bytes=0, accelerator_hot_bytes=0)
            with pager.construction():
                layer = PackedAdaptiveBitLinear(1, 33, scale=1.).eval()
            layer.fill_ternary_(1)
            layer.set_connection_strengths_(torch.full((33, 1), 16, dtype=torch.int16))
            layer.add_parallel_connection_(0, 0, sign=-1, strength=2)
            before_files = set((Path(folder) / "pages").iterdir())
            before = {key: value.clone() for key, value in layer.state_dict().items()}
            from omni_core.native_core_paging import BoundedPackedRollback
            snapshot = BoundedPackedRollback.capture([layer], directory=Path(folder) / "rollback")
            layer.add_parallel_connection_(32, 0, sign=1, strength=1)
            layer.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 33), 1000.)
            self.assertGreater(len(set((Path(folder) / "pages").iterdir())), len(before_files))
            snapshot.restore()
            self.assertEqual(set((Path(folder) / "pages").iterdir()), before_files)
            self.assertEqual(set(layer.state_dict()), set(before))
            self.assertTrue(all(torch.equal(layer.state_dict()[key], value) for key, value in before.items()))
            snapshot.close()
            pager.close()

    def test_parallel_births_during_learning_are_not_revisited(self):
        layer = PackedAdaptiveBitLinear(2, 1, scale=1.).eval()
        layer.fill_ternary_(0)
        layer.add_parallel_connection_(0, 0, sign=1, strength=16)
        layer.add_parallel_connection_(0, 1, sign=1, strength=2)
        layer.learn_from_gradient(torch.ones(1, 2), -torch.ones(1, 1), 1000.)
        records = layer.parallel_connection_records()
        self.assertEqual(len(records), 3)
        self.assertEqual([(r[2], (r[4] & 15) + 1) for r in records], [(0, 16), (0, 1), (1, 3)])

    def test_paged_strict_state_load_activates_new_contacts(self):
        source = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
        source.fill_ternary_(1)
        source.add_parallel_connection_(1, 1, sign=-1, strength=3)
        with tempfile.TemporaryDirectory() as folder:
            pager = NativeCorePager(Path(folder), cpu_hot_bytes=0, accelerator_hot_bytes=0)
            with pager.construction():
                target = PackedAdaptiveBitLinear(2, 2, scale=1.).eval()
            target.load_state_dict(source.state_dict())
            self.assertTrue(torch.equal(source(torch.ones(1, 2)), target(torch.ones(1, 2))))
            pager.close()

    def test_small_head_transaction_rolls_back_growth_when_later_owner_fails(self):
        first = PackedAdaptiveBitLinear(1, 1, scale=1.).eval()
        second = PackedAdaptiveBitLinear(1, 1, scale=1.).eval()
        first.fill_ternary_(1)
        first.set_connection_strengths_(torch.tensor([[16]], dtype=torch.int16))
        first.add_parallel_connection_(0, 0, sign=-1, strength=2)
        before = {key: value.clone() for key, value in first.state_dict().items()}
        with patch.object(second, 'commit_online_step', side_effect=RuntimeError('second owner failed')):
            with self.assertRaisesRegex(RuntimeError, 'second owner failed'):
                with packed_online_step([first, second]):
                    first.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 1), 1000.)
                    second.learn_from_gradient(torch.ones(1, 1), -torch.ones(1, 1), 1000.)
        self.assertEqual(set(first.state_dict()), set(before))
        self.assertTrue(all(torch.equal(first.state_dict()[key], value) for key, value in before.items()))
        self.assertIsNone(first._online_transaction)
        self.assertIsNone(second._online_transaction)

    def test_strength_and_contacts_match_dense_spatial_gradients_in_all_dimensions(self):
        from torch.nn import functional as F
        cases = (
            (PackedAdaptiveBitConv1d, F.conv1d, 1),
            (PackedAdaptiveBitConv2d, F.conv2d, 2),
            (PackedAdaptiveBitConv3d, F.conv3d, 3),
            (PackedAdaptiveBitConvTranspose1d, F.conv_transpose1d, 1),
            (PackedAdaptiveBitConvTranspose2d, F.conv_transpose2d, 2),
            (PackedAdaptiveBitConvTranspose3d, F.conv_transpose3d, 3),
        )
        for constructor, dense, dims in cases:
            with self.subTest(constructor=constructor.__name__):
                layer = constructor(2, 4, 2, groups=2, scale=1., bias=True).eval()
                layer.set_connection_strengths_(torch.full_like(layer._connection_state, 3, dtype=torch.int16))
                layer.add_parallel_connection_(layer._matrix_rows - 1, layer._matrix_width - 1, sign=1, strength=2)
                layer.add_parallel_connection_(0, 0, sign=-1, strength=2, bias=True)
                weight = layer.effective_weight().float() * 3
                index = (layer.in_channels - 1, layer.out_channels // layer.groups - 1) if layer.transposed else (layer.out_channels - 1, layer.in_channels // layer.groups - 1)
                weight[index + (1,) * dims] += 2
                values = torch.randint(-1, 2, (1, 2, *((3,) * dims))).float()
                actual_input = values.clone().requires_grad_()
                expected_input = values.clone().requires_grad_()
                actual = layer(actual_input)
                expected = dense(expected_input, weight, layer.effective_weighted_bias().float(), groups=2)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
                actual.sum().backward()
                expected.sum().backward()
                self.assertTrue(torch.equal(actual_input.grad, expected_input.grad))

    def test_transposed_convolution_export_authenticates_internal_matrix_order(self):
        layer = PackedAdaptiveBitConvTranspose1d(4, 6, 3, groups=2, bias=False, scale=1.).eval()
        values = torch.arange(layer._matrix_rows * layer._matrix_width).reshape(layer._matrix_rows, layer._matrix_width).remainder(3).to(torch.int8) - 1
        layer._packed_forward_weight.copy_(pack_ternary_weight(values))
        with tempfile.TemporaryDirectory() as folder:
            export_module_ternary_shards(Path(folder), {"conv": layer})
            result = verify_ternary_shards(Path(folder))
            self.assertEqual(result.manifest["tensors"][0]["packedRowLayout"], {"kind": "transpose-convolution", "groups": 2})
            self.assertTrue(torch.equal(result.tensors['conv.weight'].reshape(values.shape), values))
            self.assertEqual(tuple(result.connection_states['conv.weight'].shape), (18, 2))
            manifest = result.manifest
            del manifest['tensors'][0]['packedRowLayout']
            unsigned = {key: value for key, value in manifest.items() if key != 'contentSha256'}
            canonical = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
            manifest['contentSha256'] = hashlib.sha256(canonical(unsigned)).hexdigest()
            encoded = canonical(manifest)
            (Path(folder) / 'manifest.json').write_bytes(encoded)
            (Path(folder) / 'manifest.sha256').write_text(hashlib.sha256(encoded).hexdigest() + '\n')
            with self.assertRaisesRegex(ValueError, 'connection-state shape'):
                verify_ternary_shards(Path(folder), retain_names=())

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_weighted_strength_contacts_and_rollback_match_across_ranks(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(weighted_collective_rank, args=(2, folder), nprocs=2, join=True)
            values = [json.loads(Path(folder, 'weighted-result-%d.json' % rank).read_text()) for rank in range(2)]
            self.assertEqual(values[0], values[1])
            self.assertTrue(values[0]['restored'])
            self.assertEqual(len(values[0]['records']), 2)
            self.assertEqual(values[0]['output'], [[14.]])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_collective_contact_state_streams_multiple_bounded_batches(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.spawn(weighted_collective_rank, args=(2, folder, 135), nprocs=2, join=True)
            values = [json.loads(Path(folder, 'weighted-result-%d.json' % rank).read_text()) for rank in range(2)]
            self.assertEqual(values[0], values[1])
            self.assertTrue(values[0]['restored'])
            self.assertEqual(values[0]['rollbackChunks'], 2)
            self.assertEqual(len(values[0]['records']), 136)
            self.assertEqual(values[0]['output'], [[-254.]])


if __name__ == "__main__":
    unittest.main()
