"""Tiny exact code-column learning/collective fixtures, never a user brain."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file

from omni_core.modalities import PackedTernaryTable
from omni_core.model import _apply_packed_gradient_rows, unpack_ternary_weight_rows
from omni_core.native_core_paging import NativeCorePager
from omni_core.offload import NeuralStateResourcePause
from omni_core.packed_collective import PackedCollectiveController
from omni_core.packed_collective_hooks import packed_derivative_sink, packed_row_owner
from omni_core.packed_gradient_tiles import apply_packed_gradient_tile
from omni_core.packed_connections import encode_connection_state
from omni_core.working_attention_paging import WorkingAttentionPager


def context(rank=0, size=1):
    return SimpleNamespace(rank=rank, world_size=size, distributed=size > 1,
        backend="gloo", device=torch.device("cpu"), is_rank_zero=rank == 0)


def fixed_table(codes=137, dimensions=3):
    torch.manual_seed(14)
    table = PackedTernaryTable(codes, dimensions, scale=1.)
    p = table.projection
    p.fill_ternary_(1)
    p._packed_stability_strength = .5
    p._row_stability.fill_(3)
    p.online_learning_rate = 1000.
    p._connection_state[:, 64] = 15  # Saturated base contacts grow real repeats.
    p.add_parallel_connection_(0, 64, sign=1, strength=2)
    p.add_parallel_connection_(1, 129, sign=-1, strength=3)
    return table


def reference_step(table, ids, gradient):
    p = table.projection
    derivative = torch.zeros(table.dimensions, table.rows)
    derivative.index_add_(1, ids.reshape(-1), gradient.reshape(-1, table.dimensions).t())
    with packed_row_owner(p):
        changed = _apply_packed_gradient_rows(p.packed_forward_weight(), table.rows, 0, derivative,
            p.online_learning_rate, p._packed_forward_scale, row_stability=p._row_stability,
            stability_strength=p._packed_stability_strength, connection_state=p._connection_state)
    return derivative, changed


def contacts(table):
    return sorted(record[1:] for record in table.projection.iter_parallel_connection_records())


def collective_rank(rank, size, root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + str(Path(root) / "rendezvous"), rank=rank, world_size=size)
    try:
        table = fixed_table()
        c = PackedCollectiveController({"table": table.projection}, context(rank, size), Path(root) / str(rank), _apply_packed_gradient_rows)
        before = c.identity()
        c.begin("same-global-column-step")
        ids = torch.tensor([1, 64, 64]) if rank == 0 else torch.tensor([1, 129, 136])
        values = -torch.ones(3, 3) if rank == 0 else torch.tensor([[1., 1., 1.], [-1., -1., -1.], [1., 1., 1.]])
        with packed_derivative_sink(c):
            # Deliberately different local batching, no private divisor.
            if rank:
                for start in range(3):
                    table.learn_selected(ids[start:start+1], values[start:start+1])
            else:
                table.learn_selected(ids, values)
        if c.identity() != before:
            raise AssertionError("collective backward privately changed packed state")
        c.commit(retain_rollback=True)
        learned = c.identity()
        result = {"identity": learned, "contacts": contacts(table),
            "codes": table(torch.tensor([1, 64, 129, 136])).detach().tolist(),
            "rowStability": table.projection._row_stability.tolist()}
        c.rollback()
        result["rollbackExact"] = c.identity() == before
        Path(root, "rank-%d.json" % rank).write_text(json.dumps(result))
        c.close()
    finally:
        dist.destroy_process_group()


class SelectedModalityTilingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory(prefix="selected-code-tiles-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def activity_pager(self, budget):
        pager = WorkingAttentionPager(self.root / "activity", resident_budget_bytes=0,
            scratch_budget_bytes=1024*1024, device_tile_budget_bytes=budget)
        self.addCleanup(pager.close)
        return pager

    def test_duplicate_sums_and_discrete_state_match_full_width_reference(self):
        table = fixed_table()
        reference = copy.deepcopy(table)
        ids = torch.tensor([136, 64, 1, 64, 129, 1])
        gradient = torch.tensor([[-1., 1., -1.], [-1., -1., -1.], [.5, -.5, 1.],
                                 [-1., -1., -1.], [1., -1., 1.], [.25, -.25, -.5]])
        old_contacts = {item[0]: item for item in table.projection.iter_parallel_connection_records()}
        derivative, expected_changed = reference_step(reference, ids, gradient)
        captured = torch.zeros_like(derivative)

        def apply(owner, packed, width, start, column, values, *args, **kwargs):
            self.assertLessEqual(values.shape[0], 64)
            self.assertLessEqual(values.shape[1], 64)
            captured[start:start+values.shape[0], column:column+values.shape[1]].add_(values)
            return apply_packed_gradient_tile(owner, packed, width, start, column, values, *args, **kwargs)

        with patch("omni_core.modalities.apply_packed_gradient_tile", side_effect=apply):
            changed = table.learn_selected(ids, gradient)
        self.assertTrue(torch.equal(captured, derivative))
        self.assertEqual(changed, expected_changed)
        for name in ("_packed_forward_weight", "_connection_state", "_row_stability"):
            self.assertTrue(torch.equal(getattr(table.projection, name), getattr(reference.projection, name)), name)
        self.assertEqual(contacts(table), contacts(reference))
        actual = {item[0]: item for item in table.projection.iter_parallel_connection_records()}
        expected = {item[0]: item for item in reference.projection.iter_parallel_connection_records()}
        for key in old_contacts:
            self.assertEqual(actual[key], expected[key])  # Existing owner identity is unchanged.
        self.assertEqual(table.projection._pending_stability_events, 1)
        self.assertEqual(table.projection._row_stability.tolist(), [4, 4, 4])

    def test_untouched_zero_sign_retains_its_gain_resistance_and_packed_neighbors(self):
        table = PackedTernaryTable(69, 2, scale=1.)
        p = table.projection
        p.fill_ternary_(0)
        p._connection_state.fill_(0xe9)
        p.online_learning_rate = 1000.
        before = p._connection_state.clone()
        table.learn_selected(torch.tensor([0, 68]), -torch.ones(2, 2))
        self.assertTrue(torch.equal(p._connection_state[:, 1:68], before[:, 1:68]))
        levels = unpack_ternary_weight_rows(p._packed_forward_weight, 69)
        self.assertTrue(bool(levels[:, 1:68].eq(0).all()))
        self.assertTrue(bool(levels[:, [0, 68]].eq(1).all()))
        self.assertTrue(bool(((p._packed_forward_weight[:, -1] >> 2) == 0x15).all()))

    def test_once_per_row_stability_and_per_contact_resistance_across_tiles(self):
        table = PackedTernaryTable(133, 2, scale=1.)
        p = table.projection
        p.fill_ternary_(0)
        p._packed_stability_strength = 1.
        p._row_stability[:] = torch.tensor([0, 15], dtype=torch.uint8)
        p._connection_state[0, 64] = 0xf0
        p.online_learning_rate = .1
        with patch("torch.rand", side_effect=lambda shape, **kw: torch.full(shape, .07,
                **{key: value for key, value in kw.items() if key != "generator"})):
            table.learn_selected(torch.tensor([0, 64, 132]), torch.ones(3, 2))
        levels = unpack_ternary_weight_rows(p._packed_forward_weight, 133)
        self.assertEqual(levels[0, [0, 64, 132]].tolist(), [-1, 0, -1])
        self.assertTrue(bool(levels[1].eq(0).all()))
        self.assertEqual(p._row_stability.tolist(), [1, 15])

    def test_low_ram_large_codebook_uses_both_bounded_axes_and_mapped_release(self):
        pager = NativeCorePager(self.root / "native", cpu_hot_bytes=0, chunk_bytes=4096)
        self.addCleanup(pager.close)
        with pager.construction():
            table = PackedTernaryTable(4097, 65, scale=1.)
        table.projection.online_learning_rate = 1000.
        ids = torch.tensor([0, 63, 64, 4096, 64])
        activity = self.activity_pager(32*1024)
        seen = []
        original = unpack_ternary_weight_rows

        def decode(packed, width, *args, **kwargs):
            self.assertLessEqual(width, 64)
            self.assertLessEqual(packed.shape[0], 64)
            seen.append(tuple(packed.shape))
            return original(packed, width, *args, **kwargs)

        with activity.activate(), patch("omni_core.model.unpack_ternary_weight_rows", side_effect=decode), \
                patch.object(pager, "release_chunk", wraps=pager.release_chunk) as release:
            table.learn_selected(ids, -torch.ones(5, 65))
        self.assertTrue(seen)
        self.assertGreater(release.call_count, 0)
        self.assertGreater(pager.status()["cpuMappedLogicalBytes"], 0)
        self.assertLessEqual(activity.status()["peakComputeTileBytes"], 32*1024)
        self.assertEqual(table.projection._connection_state.shape, (65, 4097))
        self.assertEqual(tuple(table.parameters()), ())

    def test_admission_and_late_invalid_input_fail_before_mutation(self):
        table = fixed_table()
        before = copy.deepcopy(table.state_dict())
        activity = self.activity_pager(512)
        with activity.activate(), self.assertRaises(NeuralStateResourcePause):
            table.learn_selected(torch.tensor([1, 136]), torch.ones(2, 3))
        invalid = torch.ones(2, 3)
        invalid[-1, -1] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid"):
            table.learn_selected(torch.tensor([1, 136]), invalid)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, table.state_dict()[key]), key)

    def test_collective_tiled_and_full_derivatives_sum_before_one_update_and_rollback(self):
        table = fixed_table()
        c = PackedCollectiveController({"table": table.projection}, context(), self.root / "collective", _apply_packed_gradient_rows)
        self.addCleanup(c.close)
        before = c.identity()
        c.begin("mixed-table-derivatives")
        ids = torch.tensor([1, 64, 64, 136])
        values = -torch.ones(4, 3)
        full = torch.zeros(137, 3)
        full[1] = 1.  # Exact cancellation, not two independent updates.
        with packed_derivative_sink(c):
            table.learn_selected(ids[:1], values[:1])
            table.learn_selected(ids[1:], values[1:])
            table.learn_from_gradient(full)
        self.assertEqual(c.identity(), before)
        self.assertEqual(c.database.execute("SELECT COUNT(*) FROM derivatives").fetchone()[0], 0)
        self.assertLessEqual(c.database.execute("SELECT MAX(length(gradient)) FROM column_derivatives").fetchone()[0], 64*4)
        c.commit(retain_rollback=True)
        levels = table(torch.tensor([1, 64, 136])).detach()
        self.assertTrue(bool(levels[0].eq(1).all()))
        self.assertTrue(bool(levels[-1].eq(2).all()))
        self.assertEqual(table.projection._row_stability.tolist(), [4, 4, 4])
        c.rollback()
        self.assertEqual(c.identity(), before)

    def test_collective_late_tile_failure_rolls_back_base_repeats_and_counters(self):
        table = fixed_table()
        c = PackedCollectiveController({"table": table.projection}, context(), self.root / "rollback", _apply_packed_gradient_rows)
        self.addCleanup(c.close)
        before = c.identity()
        c.begin("fail-after-two-tiles")
        with packed_derivative_sink(c):
            table.learn_selected(torch.tensor([64, 136]), -torch.ones(2, 3))
        count = 0

        def fail(*args, **kwargs):
            nonlocal count
            result = apply_packed_gradient_tile(*args, **kwargs)
            count += 1
            if count == 2:
                raise RuntimeError("injected late tile failure")
            return result

        with patch("omni_core.packed_collective_tiles.apply_packed_gradient_tile", side_effect=fail), \
                self.assertRaisesRegex(RuntimeError, "injected late tile failure"):
            c.commit()
        self.assertEqual(c.identity(), before)
        self.assertEqual(c.step_id, "")

    def test_checkpoint_orientation_and_native_unit_migration_after_tiled_learning(self):
        table = fixed_table()
        table.learn_selected(torch.tensor([64, 136]), -torch.ones(2, 3))
        path = self.root / "table.safetensors"
        save_file({key: value.contiguous() for key, value in table.state_dict().items()}, str(path))
        restored = PackedTernaryTable(137, 3, scale=1.)
        restored.load_state_dict(load_file(str(path)))
        self.assertEqual(tuple(restored.projection._packed_forward_weight.shape), (3, 35))
        self.assertTrue(torch.equal(restored(torch.arange(137)), table(torch.arange(137))))
        self.assertEqual(contacts(restored), contacts(table))
        plain = PackedTernaryTable(137, 3, scale=1.)
        plain.projection.fill_ternary_(1)
        saved = {key: value for key, value in plain.state_dict().items() if key != "projection._connection_state"}
        migrated = PackedTernaryTable(137, 3, scale=1.)
        migrated.load_state_dict(saved)
        self.assertTrue(bool(migrated.projection._connection_state.eq(0).all()))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_actual_metal_selected_update_matches_deterministic_cpu_state(self):
        table = fixed_table()
        metal = copy.deepcopy(table).to("mps")
        ids, gradient = torch.tensor([1, 64, 136, 64]), -torch.ones(4, 3)
        table.learn_selected(ids, gradient)
        metal.learn_selected(ids.to("mps"), gradient.to("mps"))
        self.assertTrue(torch.equal(table.projection._packed_forward_weight, metal.projection._packed_forward_weight.cpu()))
        self.assertTrue(torch.equal(table.projection._connection_state, metal.projection._connection_state.cpu()))
        self.assertEqual(contacts(table), contacts(metal))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_two_ranks_sum_disjoint_duplicate_and_cancelled_code_tiles(self):
        mp.spawn(collective_rank, args=(2, str(self.root)), nprocs=2, join=True)
        results = [json.loads((self.root / ("rank-%d.json" % rank)).read_text()) for rank in range(2)]
        self.assertEqual(results[0], results[1])
        self.assertTrue(results[0]["rollbackExact"])
        self.assertEqual(results[0]["rowStability"], [4, 4, 4])
        self.assertEqual(results[0]["codes"][0], [1., 1., 1.])
        self.assertEqual(results[0]["codes"][-1], [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
