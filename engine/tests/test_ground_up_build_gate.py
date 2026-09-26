"""No-Build structural checks for the native initialization gate.

The next authorized trained-brain acceptance run must validate actual loss,
manifest promotion, and restart on a freshly built instance.
"""

import inspect
import unittest

from omni_core.brain import AdaptiveBrain
from omni_core.ground_up import (
    GROUND_UP_ACTION_EXAMPLES,
    GROUND_UP_TOOL_NEGATIVE_EXAMPLES,
    GROUND_UP_TOOL_TRAJECTORIES,
    GROUND_UP_V3_SOURCE_RECORDS,
    ground_up_curriculum_manifest,
)


class GroundUpBuildGateTests(unittest.TestCase):
    def test_current_curriculum_is_native_and_complete(self):
        manifest = ground_up_curriculum_manifest()
        self.assertEqual(manifest["formatVersion"], 3)
        self.assertEqual(manifest["sourceKind"], "ordinary-training-examples")
        self.assertFalse(manifest["pretrainedWeights"])
        self.assertIsNone(manifest["externalFoundation"])
        self.assertEqual(
            GROUND_UP_V3_SOURCE_RECORDS,
            len(GROUND_UP_ACTION_EXAMPLES)
            + len(GROUND_UP_TOOL_TRAJECTORIES)
            + len(GROUND_UP_TOOL_NEGATIVE_EXAMPLES),
        )

    def test_creation_requires_explicit_local_curriculum(self):
        source = inspect.getsource(AdaptiveBrain.create)
        self.assertIn("if not initialize_ground_up:", source)
        self.assertIn("brain._train_ground_up_curriculum(", source)
        self.assertIn("brain._finalize_ground_up_creation(", source)
        self.assertNotIn("_train_bundled_starter", source)
        self.assertNotIn("foundation_cortex", source)

    def test_restart_authenticates_exact_native_receipt(self):
        source = inspect.getsource(AdaptiveBrain._validate_ground_up_training_manifest)
        self.assertIn("_validate_ground_up_v3_training_manifest", source)
        self.assertIn("resolve_ground_up_curriculum_manifest", source)
        self.assertNotIn("_v2", source)
        self.assertNotIn("starter", source)


if __name__ == "__main__":
    unittest.main()
