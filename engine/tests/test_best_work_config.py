import unittest
from omni_core.config import OmniConfig


class BestWorkConfigTests(unittest.TestCase):
    def test_default_is_off_and_enabled_setting_roundtrips_without_geometry_change(self):
        default = OmniConfig.micro()
        default.validate()
        self.assertFalse(default.best_work_enabled)
        enabled = OmniConfig.from_dict({**default.to_dict(), 'best_work_enabled': True, 'best_work_exploration_branches': 4})
        self.assertTrue(enabled.best_work_enabled)
        self.assertEqual(enabled.best_work_exploration_branches, 4)
        for field in ('d_model', 'n_layers', 'd_ff', 'vsa_dim', 'router_neurons', 'working_memory_slots'):
            self.assertEqual(getattr(default, field), getattr(enabled, field))
        loaded = OmniConfig.from_dict(enabled.to_dict())
        self.assertTrue(loaded.best_work_enabled)

    def test_external_contract_is_typed_and_respects_explicit_exploration_limit(self):
        enabled = OmniConfig.from_external({'hardwareTier': 'micro', 'bestWorkEnabled': True, 'bestWorkExplorationBranches': 8})
        self.assertTrue(enabled.best_work_enabled)
        self.assertEqual(enabled.best_work_exploration_branches, 8)
        for update in ({'bestWorkEnabled': 'true'}, {'bestWorkExplorationBranches': True},
                       {'bestWorkExplorationBranches': 0}, {'bestWorkExplorationBranches': 17}):
            with self.assertRaises(ValueError):
                OmniConfig.from_external({'hardwareTier': 'micro', **update})

    def test_requested_ceiling_does_not_silently_change_ordinary_temperature(self):
        normal = OmniConfig.micro(temperature=1.2)
        normal.validate()
        self.assertEqual(normal.temperature, 1.2)
        normal.best_work_enabled = True
        with self.assertRaisesRegex(ValueError, '0.95'):
            normal.validate()


if __name__ == '__main__':
    unittest.main()
