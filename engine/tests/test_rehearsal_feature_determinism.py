import copy
import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.capability_rehearsal import (
    REHEARSAL_FEATURE_LATTICE,
    canonicalize_rehearsal_features,
)


class RehearsalFeatureDeterminismTests(unittest.TestCase):
    @staticmethod
    def _updated_state(
        initial: torch.nn.Module,
        features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        model = copy.deepcopy(initial)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(features), torch.tensor([0, 1], dtype=torch.long)
        )
        loss.backward()
        optimizer.step()
        return {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def test_sub_lattice_jitter_converges_but_meaningful_drift_remains(self):
        lattice = REHEARSAL_FEATURE_LATTICE
        base = torch.tensor(
            [[3.0, -5.0, 7.0], [-2.0, 4.0, -8.0]],
            dtype=torch.float32,
            requires_grad=True,
        ) * lattice
        sub_lattice_jitter = torch.tensor(
            [[0.49, -0.49, 0.25], [-0.25, 0.49, -0.49]],
            dtype=torch.float32,
        ) * lattice
        meaningful_drift = torch.tensor(
            [[0.75, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ) * lattice

        canonical_base = canonicalize_rehearsal_features(base)
        canonical_jitter = canonicalize_rehearsal_features(
            base + sub_lattice_jitter
        )
        canonical_drift = canonicalize_rehearsal_features(
            base + meaningful_drift
        )
        self.assertFalse(canonical_base.requires_grad)
        self.assertTrue(torch.equal(canonical_base, canonical_jitter))
        self.assertFalse(torch.equal(canonical_base, canonical_drift))

        torch.manual_seed(197)
        initial = torch.nn.Linear(3, 2)
        base_state = self._updated_state(initial, canonical_base)
        jitter_state = self._updated_state(initial, canonical_jitter)
        drift_state = self._updated_state(initial, canonical_drift)
        self.assertTrue(
            all(
                torch.equal(base_state[name], jitter_state[name])
                for name in base_state
            )
        )
        self.assertTrue(
            any(
                not torch.equal(base_state[name], drift_state[name])
                for name in base_state
            )
        )


if __name__ == "__main__":
    unittest.main()
