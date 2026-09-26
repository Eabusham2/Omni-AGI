import sys
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.liquid import CfCCell, LTCCell, LiquidController
from omni_core.model import PackedAdaptiveBitLinear
from omni_core.spiking import LIFPopulation, STDPSynapses


class SpikingAndLiquidTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)

    def test_lif_leaks_fires_and_resets(self):
        population = LIFPopulation(2, leak=0.5, threshold=0.6)
        first, _ = population.step(torch.tensor([0.4, 0.0]))
        second, _ = population.step(torch.tensor([0.4, 0.0]))
        self.assertEqual(float(first[0]), 0.0)
        self.assertEqual(float(second[0]), 1.0)
        self.assertLess(float(population.membrane[0]), 0.6)

    def test_stdp_causal_potentiation_and_anti_causal_depression(self):
        causal = STDPSynapses(1, 1, learning_rate=0.1)
        causal.step(torch.tensor([1.0]), torch.tensor([0.0]))
        causal.step(torch.tensor([0.0]), torch.tensor([1.0]))
        self.assertGreater(float(causal.weights[0, 0]), 0.0)

        anti = STDPSynapses(1, 1, learning_rate=0.1)
        anti.step(torch.tensor([0.0]), torch.tensor([1.0]))
        anti.step(torch.tensor([1.0]), torch.tensor([0.0]))
        self.assertLess(float(anti.weights[0, 0]), 0.0)

    def test_metaplasticity_reduces_repeated_update(self):
        synapses = STDPSynapses(
            1, 1, learning_rate=0.1, metaplasticity_rate=1.0
        )
        synapses.step(torch.tensor([1.0]), torch.tensor([0.0]))
        first = synapses.step(
            torch.tensor([0.0]), torch.tensor([1.0])
        ).abs().item()
        synapses.reset_activity()
        synapses.step(torch.tensor([1.0]), torch.tensor([0.0]))
        second = synapses.step(
            torch.tensor([0.0]), torch.tensor([1.0])
        ).abs().item()
        self.assertLess(second, first)

    def test_stdp_synapses_are_packed_authoritative_without_float_weight_mirror(self):
        synapses = STDPSynapses(3, 2)
        synapses.set_effective_weights(
            torch.tensor(
                [[-1, 0, 0], [1, 0, 1]],
                dtype=torch.int8,
            )
        )
        effective = synapses.effective_weight()
        self.assertEqual(effective.dtype, torch.int8)
        self.assertEqual(
            set(effective.reshape(-1).tolist()),
            {-1, 0, 1},
        )
        self.assertEqual(synapses._packed_weights.dtype, torch.uint8)
        self.assertEqual(synapses._packed_weights.numel(), 2)
        self.assertEqual(synapses.logical_ternary_parameter_count, 6)
        self.assertEqual(len(synapses.authoritative_packed_tensors()), 1)
        self.assertNotIn("weights", synapses.state_dict())

    def test_packed_stdp_checkpoint_preserves_levels_and_timing_state(self):
        source = STDPSynapses(3, 2, learning_rate=0.1)
        source.step(torch.tensor([1.0, 0.0, 0.0]), torch.zeros(2))
        source.step(torch.zeros(3), torch.tensor([1.0, 0.0]))
        restored = STDPSynapses(3, 2, learning_rate=0.1)
        restored.load_state_dict(source.state_dict())
        self.assertTrue(
            torch.equal(source.effective_weight(), restored.effective_weight())
        )
        self.assertTrue(torch.equal(source.pre_trace, restored.pre_trace))
        self.assertTrue(torch.equal(source.stability, restored.stability))
        self.assertTrue(
            torch.equal(
                source.eligibility_accumulator,
                restored.eligibility_accumulator,
            )
        )

    def test_packed_stdp_migrates_native_float_weights_and_rejects_reserved_codes(self):
        synapses = STDPSynapses(3, 2)
        obsolete = synapses.state_dict()
        del obsolete["_packed_weights"]
        del obsolete["eligibility_accumulator"]
        del obsolete["decay_cycles"]
        obsolete["weights"] = torch.tensor(
            [[-0.8, -0.1, 0.0], [0.2, 0.3, 0.9]], dtype=torch.float32
        )
        synapses.load_state_dict(obsolete)
        self.assertTrue(
            torch.equal(
                synapses.effective_weight(),
                torch.tensor([[-1, 0, 0], [0, 1, 1]], dtype=torch.int8),
            )
        )
        self.assertNotIn("weights", synapses.state_dict())
        malformed = synapses.state_dict()
        malformed["_packed_weights"] = malformed["_packed_weights"].clone()
        malformed["_packed_weights"][0, 0] = 0xFF
        with self.assertRaisesRegex(RuntimeError, "reserved code"):
            synapses.load_state_dict(malformed)

    def test_cfc_and_ltc_are_trainable_and_stable(self):
        for cell in (CfCCell(8, 8), LTCCell(8, 8, solver_steps=4)):
            inputs = torch.randn(3, 8, requires_grad=True)
            projections = [
                module
                for module in cell.modules()
                if isinstance(module, PackedAdaptiveBitLinear)
            ]
            self.assertTrue(projections)
            for projection in projections:
                projection.online_learning_rate = 100.0
            before = [
                tuple(tensor.clone() for tensor in projection.authoritative_packed_tensors())
                for projection in projections
            ]
            state = cell(inputs, elapsed=0.5)
            self.assertEqual(tuple(state.shape), (3, 8))
            self.assertTrue(torch.isfinite(state).all())
            state.pow(2).mean().backward()
            self.assertIsNotNone(inputs.grad)
            self.assertTrue(
                any(
                    not torch.equal(previous, current)
                    for projection, snapshot in zip(projections, before)
                    for previous, current in zip(
                        snapshot, projection.authoritative_packed_tensors()
                    )
                )
            )

    def test_ltc_time_controls_are_packed_ternary_and_reload(self):
        cell = LTCCell(3, 3, solver_steps=3)
        self.assertEqual(tuple(cell.parameters()), ())
        tonic = torch.ones(1, 1)
        self.assertTrue(
            torch.allclose(cell.leak_logit(tonic), torch.full((1, 3), -0.25))
        )
        self.assertTrue(
            torch.allclose(cell.capacitance_logit(tonic), torch.zeros(1, 3))
        )
        for projection in (cell.leak_logit, cell.capacitance_logit):
            projection.online_learning_rate = 100.0
        leak_before = cell.leak_logit._packed_forward_weight.clone()
        capacitance_before = cell.capacitance_logit._packed_forward_weight.clone()
        (
            -cell.leak_logit(tonic).sum()
            + cell.capacitance_logit(tonic).sum()
        ).backward()
        self.assertFalse(
            torch.equal(leak_before, cell.leak_logit._packed_forward_weight)
        )
        self.assertFalse(
            torch.equal(
                capacitance_before,
                cell.capacitance_logit._packed_forward_weight,
            )
        )
        inputs = torch.randn(2, 3)
        output = cell(inputs, elapsed=0.5)
        self.assertTrue(torch.isfinite(output).all())
        reloaded = LTCCell(3, 3, solver_steps=3)
        reloaded.load_state_dict(cell.state_dict())
        self.assertTrue(torch.equal(output, reloaded(inputs, elapsed=0.5)))

    def test_controller_exposes_bounded_controls_for_both_modes(self):
        inputs = torch.randn(1, 8)
        for mode in ("cfc", "ltc"):
            controller = LiquidController(8, mode=mode)
            state, controls = controller(inputs)
            self.assertEqual(tuple(state.shape), (1, 8))
            self.assertEqual(
                set(controls),
                {"retention", "threshold_offset", "noise_scale", "ponder_scale"},
            )
            self.assertTrue(all(torch.isfinite(value).all() for value in controls.values()))


if __name__ == "__main__":
    unittest.main()
