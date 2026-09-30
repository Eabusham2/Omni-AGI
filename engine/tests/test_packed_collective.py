"""Primitive packed collectives only: no neural modules/brains are built."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omni_core.model import _apply_packed_gradient_rows, unpack_ternary_weight_rows
from omni_core.packed_collective import PackedCollectiveController
from omni_core.packed_collective_hooks import defer_packed_rows, packed_derivative_sink, packed_row_owner
from omni_core.collective_controls import synchronize_control_state
from omni_core.window_wave_buffer import PreparedWindowWave


def owner():
    return SimpleNamespace(_packed_forward_weight=torch.full((1, 1), 0x55, dtype=torch.uint8),
        _packed_forward_bias=None, _packed_forward_scale=torch.tensor(1.0),
        _row_stability=torch.zeros(1, dtype=torch.uint8), _bias_row_stability=None,
        _packed_stability_strength=1.0, _pending_stability_events=0)


def primitive_rank(rank, size, root):
    dist.init_process_group("gloo", init_method="file://" + str(Path(root) / "rendezvous"), rank=rank, world_size=size)
    try:
        model_owner = owner()
        context = SimpleNamespace(rank=rank, world_size=size, distributed=True, device=torch.device("cpu"), backend="gloo", is_rank_zero=rank == 0)
        controller = PackedCollectiveController({"projection": model_owner}, context, Path(root) / ("rank-%d" % rank), _apply_packed_gradient_rows)
        controller.begin("same-native-step-0")
        gradient = torch.tensor([[1.0, -2.0, 0.0, 0.0]]) if rank == 0 else torch.tensor([[-3.0, 1.0, 0.0, 0.0]])
        wave = PreparedWindowWave(physical_batch=3 if rank == 0 else 1, ram_budget=0,
            directory=Path(root) / ("input-%d" % rank), reserve=lambda **_: None)
        splits = 2 if rank == 0 else 3
        for index in range(splits):
            # Different microbatch cardinality/short target counts, same
            # globally weighted aggregate mathematical derivative.
            wave.append(torch.tensor([1, 8 + index, 2]), gradient[0] / splits, torch.tensor([0.]))
        with packed_derivative_sink(controller), packed_row_owner(model_owner):
            for batch in wave.batches():
                accumulated = sum((cue for _ids, cue, _noise in batch), torch.zeros(4)).reshape(1, 4)
                staged = defer_packed_rows(model_owner._packed_forward_weight, 4, 0, accumulated, 100.0,
                    model_owner._packed_forward_scale, model_owner._row_stability, 1.0)
        wave.close()
        if staged != 0 or int(model_owner._packed_forward_weight.item()) != 0x55:
            raise AssertionError("rank privately mutated packed bytes during backward deferral")
        controller.commit()
        values = unpack_ternary_weight_rows(model_owner._packed_forward_weight, 4, 0, 1).tolist()
        # Existing scalar controls/moments are primitive tensors, not a
        # constructed neural module or a floating projection master.
        control = torch.tensor(0.25 if rank == 0 else 99.0)
        module = SimpleNamespace(named_parameters=lambda: [("gain", control)])
        optimizer = SimpleNamespace(state={control: {"step": torch.tensor(3. if rank == 0 else 91.),
            "exp_avg": torch.tensor([0.125 if rank == 0 else 50.])}})
        brain = SimpleNamespace(slow_anchors={"gain": torch.tensor(0.2 if rank == 0 else 70.)},
            slow_importance={"gain": torch.tensor(0.3 if rank == 0 else 80.)},
            counters={"training_steps": 29 if rank == 0 else 99})
        synchronize_control_state(controller, module, optimizer, brain)
        shared_identity = controller.identity()
        model_owner._packed_forward_scale.fill_(2. if rank else 1.)
        try:
            controller.begin("different-scale-must-fail")
        except RuntimeError as error:
            mismatch_rejected = "one packed neural identity" in str(error)
        else:
            raise AssertionError("different packed scales were silently accepted")
        model_owner._packed_forward_scale.fill_(1.)
        controller.begin("rollback-canonical-apply")
        with packed_derivative_sink(controller), packed_row_owner(model_owner):
            defer_packed_rows(model_owner._packed_forward_weight, 4, 0, torch.tensor([[1., 1., 0., 0.]]),
                100., model_owner._packed_forward_scale, model_owner._row_stability, 1.)
        def failing_apply(*args):
            _apply_packed_gradient_rows(*args)
            raise RuntimeError("injected after canonical packed mutation")
        controller.apply_rows = failing_apply
        try:
            controller.commit()
        except RuntimeError:
            rollback_exact = controller.identity() == shared_identity
        else:
            raise AssertionError("injected canonical mutation failure was ignored")
        Path(root, "result-%d.json" % rank).write_text(json.dumps({"levels": values, "identity": shared_identity,
            "control": control.item(), "moment": optimizer.state[control]["exp_avg"].item(),
            "step": optimizer.state[control]["step"].item(), "counters": brain.counters,
            "mismatchRejected": mismatch_rejected, "rollbackExact": rollback_exact}))
        controller.close()
    finally:
        dist.destroy_process_group()


class PackedCollectivePrimitiveTests(unittest.TestCase):
    def test_no_sink_preserves_normal_direct_path_and_missing_owner_fails_closed(self):
        self.assertIsNone(defer_packed_rows(None, 1, 0, None, 1, 1))
        with packed_derivative_sink(SimpleNamespace(stage=lambda *_: 0)):
            with self.assertRaisesRegex(RuntimeError, "registered owner"):
                defer_packed_rows(None, 1, 0, None, 1, 1)

    def test_single_rank_collective_stages_derivatives_without_weight_master_and_can_rollback(self):
        with tempfile.TemporaryDirectory() as root:
            model_owner = owner()
            context = SimpleNamespace(rank=0, world_size=1, distributed=False, device=torch.device("cpu"), backend="gloo", is_rank_zero=True)
            controller = PackedCollectiveController({"projection": model_owner}, context, root, _apply_packed_gradient_rows)
            try:
                initial = controller.identity()
                controller.begin("step-0")
                with packed_derivative_sink(controller), packed_row_owner(model_owner):
                    defer_packed_rows(model_owner._packed_forward_weight, 4, 0, torch.tensor([[-1., 1., 0., 0.]]),
                        100., model_owner._packed_forward_scale, model_owner._row_stability, 1.)
                self.assertEqual(controller.identity(), initial)
                controller.commit(retain_rollback=True)
                self.assertNotEqual(controller.identity(), initial)
                controller.rollback()
                self.assertEqual(controller.identity(), initial)
                self.assertEqual(model_owner._packed_forward_weight.dtype, torch.uint8)
            finally:
                controller.close()

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo primitive infrastructure is unavailable")
    def test_two_ranks_reduce_derivatives_then_broadcast_one_identical_packed_update(self):
        with tempfile.TemporaryDirectory() as root:
            mp.spawn(primitive_rank, args=(2, root), nprocs=2, join=True)
            results = [json.loads(Path(root, "result-%d.json" % rank).read_text()) for rank in range(2)]
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["levels"], [[1, 1, 0, 0]])
            self.assertEqual(results[0]["control"], 0.25)
            self.assertEqual(results[0]["moment"], 0.125)
            self.assertEqual(results[0]["step"], 3.)
            self.assertEqual(results[0]["counters"], {"training_steps": 29})
            self.assertTrue(results[0]["mismatchRejected"])
            self.assertTrue(results[0]["rollbackExact"])


if __name__ == "__main__":
    unittest.main()
