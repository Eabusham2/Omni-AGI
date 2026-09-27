"""Pure fail-closed checks for delegating live inspection to committed shards."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.substrate_inspection import require_current_substrate_generation


class CurrentGenerationSeamTests(unittest.TestCase):
    def setUp(self):
        self.overlay = {
            "format": "omni-substrate-attention-overlay",
            "formatVersion": 1,
            "epoch": 2,
            "legacyRawActive": False,
            "activeNeuronIds": ["active"],
            "recalledAssemblyIds": [],
            "eligibleSynapseIds": [],
        }
        self.view = SimpleNamespace(
            generation={"formatVersion": 3, "stateRevision": 7},
            pointer={"activeGeneration": "a" * 64},
            counts={"neurons": 10, "assemblies": 2, "synapses": 15},
            attention_overlay=self.overlay,
        )
        self.expected = {
            "activeGeneration": "a" * 64,
            "stateRevision": 7,
            "counts": dict(self.view.counts),
            "attentionOverlay": dict(self.overlay),
        }

    def test_matching_bounded_identity_is_accepted(self):
        require_current_substrate_generation(self.view, self.expected)

    def test_changed_live_metadata_is_rejected(self):
        for field, value in (
            ("activeGeneration", "b" * 64),
            ("stateRevision", 8),
            ("counts", {**self.view.counts, "synapses": 16}),
            ("attentionOverlay", {**self.overlay, "activeNeuronIds": []}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "differs from the committed"):
                    require_current_substrate_generation(
                        self.view, {**self.expected, field: value}
                    )

    def test_missing_identity_and_old_generation_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "identity is invalid"):
            require_current_substrate_generation(
                self.view, {"stateRevision": 7}
            )
        self.view.generation["formatVersion"] = 2
        with self.assertRaisesRegex(ValueError, "differs from the committed"):
            require_current_substrate_generation(self.view, self.expected)


if __name__ == "__main__":
    unittest.main()
