"""Exactness and bounded traversal checks for continuous memory settling."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.memory_lifecycle import OrganicMemoryLifecycle
from omni_core.vsa import NeuralSubstrate


class _WithoutAssemblyIndex:
    """Run the pre-index full traversal against the same substrate semantics."""

    def __init__(self, substrate):
        self.substrate = substrate

    def __getattr__(self, name):
        if name == "assembly_by_id":
            raise AttributeError(name)
        return getattr(self.substrate, name)


class _CountingAssemblies(list):
    traversals = 0

    def __iter__(self):
        self.traversals += 1
        return super().__iter__()


class _RouterSynapses:
    def __init__(self):
        self.decays = []

    def decay_unused(self, amount):
        self.decays.append(amount)


def _substrate(count=12):
    substrate = NeuralSubstrate(dimensions=16, seed=41)
    substrate.assemblies = [
        {
            "id": "assembly-%05d" % index,
            "rehearsals": 1 + (index % 5),
            "importance": 0.2 + 0.05 * (index % 5),
        }
        for index in range(count)
    ]
    substrate.assembly_vectors.update({
        record["id"]: torch.ones(16) for record in substrate.assemblies
    })
    substrate.neurons = {
        record["id"]: {
            "id": record["id"],
            "activation": 0.15 + 0.11 * (index % 6),
            "uncertainty": 0.25,
            "exposures": 1 + index,
        }
        for index, record in enumerate(substrate.assemblies)
    }
    substrate.attention_active_neuron_ids = {
        "assembly-%05d" % index for index in (0, 1, 3, 5)
    }
    substrate.synapses = {
        "edge-%d" % index: {
            "id": "edge-%d" % index,
            "source_id": "assembly-%05d" % source,
            "target_id": "assembly-%05d" % target,
            "effective_weight": 1,
            "stability": 0.3 + 0.1 * index,
            "uses": 2 + index,
            "eligibility": 0.4,
        }
        for index, (source, target) in enumerate(((0, 1), (1, 0), (0, 3), (5, 0)))
    }
    substrate._last_recall_audit = {
        "eligibleEdges": 4,
        "inhibitorySignals": 1,
        "activeNeuralNodes": 4,
        "suppressedAssemblies": 1,
    }
    return substrate


def _settle(
    lifecycle, substrate, router, identifier, *, salience=0.65, forgetting_rate=0.04
):
    return lifecycle.settle(
        memory=substrate,
        router=router,
        vector=torch.linspace(-1.0, 1.0, 16),
        assembly_id=identifier,
        source="fixture",
        salience=salience,
        novelty=0.42,
        prediction_error=0.24,
        importance=0.78,
        spike_rate=0.35,
        forgetting_rate=forgetting_rate,
        long_term_threshold=0.5,
    )


class MemoryLifecycleScalingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_indexed_settle_matches_full_traversal_for_focus_afterimages_and_decay(self):
        indexed = _substrate()
        baseline = _substrate()
        indexed_lifecycle = OrganicMemoryLifecycle()
        baseline_lifecycle = OrganicMemoryLifecycle()
        indexed_router = SimpleNamespace(synapses=_RouterSynapses())
        baseline_router = SimpleNamespace(synapses=_RouterSynapses())
        with mock.patch(
            "omni_core.memory_lifecycle._iso_now", return_value="2026-01-01T00:00:00Z"
        ), mock.patch(
            "omni_core.memory_lifecycle.uuid.uuid4",
            return_value=SimpleNamespace(hex="fixed-afterimage-id"),
        ):
            for identifier, salience in (
                ("assembly-00000", 0.72),
                ("assembly-00001", 0.28),
                ("assembly-00000", 0.86),
                ("assembly-00002", 0.47),
            ):
                fast = _settle(
                    indexed_lifecycle, indexed, indexed_router, identifier,
                    salience=salience,
                )
                slow = _settle(
                    baseline_lifecycle, _WithoutAssemblyIndex(baseline),
                    baseline_router, identifier, salience=salience,
                )
                self.assertEqual(fast, slow)
                self.assertEqual(indexed_lifecycle.active_focus, baseline_lifecycle.active_focus)
                self.assertEqual(indexed_lifecycle.metadata(), baseline_lifecycle.metadata())
                self.assertEqual(indexed.assemblies, baseline.assemblies)
                self.assertEqual(indexed.neurons, baseline.neurons)
                self.assertEqual(indexed.synapses, baseline.synapses)
                self.assertEqual(indexed_router.synapses.decays, baseline_router.synapses.decays)
                for left, right in zip(
                    indexed_lifecycle.afterimage_vectors,
                    baseline_lifecycle.afterimage_vectors,
                ):
                    self.assertTrue(torch.equal(left, right))

    def test_warm_tracked_settle_does_not_traverse_cold_assemblies(self):
        memory = _substrate(4_096)
        self.assertFalse(memory.attention_legacy_raw_active)
        memory.assemblies = _CountingAssemblies(memory.assemblies)
        lifecycle = OrganicMemoryLifecycle()
        router = SimpleNamespace(synapses=_RouterSynapses())
        # First cycle builds the exact index and removes any legacy labels.
        _settle(lifecycle, memory, router, "assembly-00000")
        memory.assemblies.traversals = 0
        _settle(lifecycle, memory, router, "assembly-00001")
        self.assertEqual(memory.assemblies.traversals, 0)

    def test_legacy_raw_epoch_still_considers_cold_raw_activations(self):
        memory = _substrate()
        memory.attention_legacy_raw_active = True
        memory.neurons["assembly-00011"]["activation"] = 0.99
        lifecycle = OrganicMemoryLifecycle()
        lifecycle._update_focus(memory, "assembly-00000", 0.3)
        self.assertIn(
            "assembly-00011",
            [item["assemblyId"] for item in lifecycle.active_focus],
        )

    def test_legacy_labels_are_migrated_on_first_settle_and_new_append(self):
        memory = _substrate()
        memory.assemblies[2]["memory_stage"] = "lasting"
        memory.neurons["assembly-00002"]["memory_stage"] = "lasting"
        lifecycle = OrganicMemoryLifecycle()
        router = SimpleNamespace(synapses=_RouterSynapses())
        _settle(lifecycle, memory, router, "assembly-00000")
        self.assertNotIn("memory_stage", memory.assemblies[2])
        self.assertNotIn("memory_stage", memory.neurons["assembly-00002"])
        memory.assemblies.append({"id": "new-assembly", "rehearsals": 1, "memory_stage": "working"})
        memory.assembly_vectors["new-assembly"] = torch.ones(16)
        memory.neurons["new-assembly"] = {
            "id": "new-assembly", "activation": 0.2, "uncertainty": 0.3, "exposures": 1,
        }
        memory.attention_active_neuron_ids.add("new-assembly")
        _settle(lifecycle, memory, router, "new-assembly")
        self.assertNotIn("memory_stage", memory.assemblies[-1])


if __name__ == "__main__":
    unittest.main()
