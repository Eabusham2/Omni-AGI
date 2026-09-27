"""The host reserve must be remeasured after a successful scratch spill."""

import unittest
from types import SimpleNamespace

from omni_core.brain import AdaptiveBrain


class GrowthPressureRecheckTest(unittest.TestCase):
    def test_growth_unblocks_when_offload_releases_memory(self) -> None:
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        samples = iter((
            {"diskPressure": False, "memoryPressure": True},
            {"diskPressure": False, "memoryPressure": False},
        ))
        calls = []

        def status(**estimate):
            calls.append(estimate)
            return next(samples)

        brain.resource_policy = SimpleNamespace(status=status)
        brain._resource_readings = lambda: {"observed": True}
        brain._maintain_neural_state_resources = lambda: {"offloaded": True}
        brain.growth_pause = {"reason": "old pressure"}

        self.assertTrue(brain._allow_substrate_growth(4096))
        self.assertIsNone(brain.growth_pause)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], calls[1])


if __name__ == "__main__":
    unittest.main()
