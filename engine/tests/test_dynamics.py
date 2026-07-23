import sys
import unittest
from pathlib import Path

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.liquid import CfCCell, LTCCell, LiquidController
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

    def test_cfc_and_ltc_are_trainable_and_stable(self):
        inputs = torch.randn(3, 8, requires_grad=True)
        for cell in (CfCCell(8, 8), LTCCell(8, 8, solver_steps=4)):
            state = cell(inputs, elapsed=0.5)
            self.assertEqual(tuple(state.shape), (3, 8))
            self.assertTrue(torch.isfinite(state).all())
            state.pow(2).mean().backward(retain_graph=True)
            gradients = [
                parameter.grad
                for parameter in cell.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)

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
