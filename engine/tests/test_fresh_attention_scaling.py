import hashlib
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import torch


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core import AdaptiveBrain, OmniConfig
from omni_core.vsa import NeuralSubstrate


class _TraversalForbiddenDict(dict):
    def __iter__(self):
        raise AssertionError("fresh attention traversed a substrate mapping")

    def items(self):
        raise AssertionError("fresh attention traversed substrate items")

    def values(self):
        raise AssertionError("fresh attention traversed substrate values")


class _TraversalForbiddenList(list):
    def __iter__(self):
        raise AssertionError("fresh attention traversed substrate assemblies")


class FreshAttentionScalingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(109)
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="omni-fresh-attention-scale-"
        )
        self.root = Path(self.temporary.name) / "brain"

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _tree_hashes(root: Path):
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    def test_clear_is_constant_in_virtual_substrate_cardinality(self):
        memory = NeuralSubstrate(16, seed=31)

        memory.mark_attention_neuron("legacy-neuron")
        memory.mark_attention_assembly("legacy-assembly")
        memory.mark_attention_synapse("legacy-synapse")
        self.assertEqual(memory.attention_active_neuron_ids, {"legacy-neuron"})
        self.assertEqual(memory.attention_recalled_assembly_ids, {"legacy-assembly"})
        self.assertEqual(memory.attention_eligible_synapse_ids, {"legacy-synapse"})

        class VirtualLargeMapping(_TraversalForbiddenDict):
            def __len__(self):
                return 10_000_000

        memory.neurons = VirtualLargeMapping()
        memory.synapses = VirtualLargeMapping()
        memory.assemblies = _TraversalForbiddenList()
        memory.attention_epoch = 8
        memory.attention_legacy_raw_active = False
        memory.attention_active_neuron_ids = {"active-a", "active-b"}
        memory.attention_recalled_assembly_ids = {"assembly-a"}
        memory.attention_eligible_synapse_ids = {"edge-a", "edge-b", "edge-c"}

        started = time.perf_counter()
        cleared = memory.clear_attention_activity(9)
        elapsed = time.perf_counter() - started

        self.assertEqual(
            cleared,
            {
                "activatedNeurons": 2,
                "recalledAssemblies": 1,
                "eligibilityTraces": 3,
                "legacyRawActive": 0,
            },
        )
        self.assertEqual(memory.attention_epoch, 9)
        self.assertFalse(memory.attention_active_neuron_ids)
        self.assertFalse(memory.attention_recalled_assembly_ids)
        self.assertFalse(memory.attention_eligible_synapse_ids)
        # This is deliberately generous for loaded CI hosts; an accidental
        # ten-million-record traversal cannot complete beneath the envelope.
        self.assertLess(elapsed, 0.5)

    def test_large_committed_reset_never_reads_or_rewrites_substrate_shards(self):
        # A synthetic substrate/reset fixture; it does not assert that Build
        # trained or promoted an origin.
        brain = AdaptiveBrain(
            "fresh-scale-brain",
            self.root,
            OmniConfig.micro(
                max_seq_len=40,
                learn_from_own_messages=False,
                online_learning=False,
            ),
        )
        memory = brain.memory
        dimensions = memory.space.dimensions
        shared_vector = torch.ones(dimensions, dtype=torch.float32)
        cardinality = 8_192
        memory.neurons = {}
        memory.assembly_vectors.clear()
        memory.neuron_vectors.clear()
        memory.synapses = {}
        memory.assemblies = []
        for index in range(cardinality):
            identifier = "scale-neuron-%05d" % index
            memory.neurons[identifier] = {
                "id": identifier,
                "neuron_id": identifier,
                "label": identifier,
                "region": "scale",
                "activation": 0.75,
                "importance": 0.5,
                "uncertainty": 0.25,
                "exposures": 1,
                "created_at": 1_700_000_000.0,
                "last_activated_at": 1_700_000_001.0,
                "aliases": [],
            }
            memory.neuron_vectors[identifier] = shared_vector
            target = "scale-neuron-%05d" % ((index + 1) % cardinality)
            edge = "%s>%s:scale" % (identifier, target)
            memory.synapses[edge] = {
                "id": edge,
                "source_id": identifier,
                "target_id": target,
                "kind": "scale",
                "effective_weight": 1,
                "eligibility": 0.6,
                "plasticity": 0.8,
                "uses": 1,
                "stability": 0.5,
                "last_updated_at": 1_700_000_002.0,
            }
        memory.growth_events = cardinality * 2
        memory.state_revision += 1
        memory.attention_legacy_raw_active = False
        memory.attention_active_neuron_ids = {
            "scale-neuron-00000",
            "scale-neuron-00001",
        }
        memory.attention_eligible_synapse_ids = {
            next(iter(memory.synapses)),
        }
        brain.save()

        substrate_root = brain.engine_path / "substrate"
        pointer_before = (substrate_root / "manifest.json").read_bytes()
        tree_before = self._tree_hashes(substrate_root)
        content_before = memory.persistence_manifest["contentSha256"]
        memory.neurons = _TraversalForbiddenDict(memory.neurons)
        memory.synapses = _TraversalForbiddenDict(memory.synapses)
        memory.assemblies = _TraversalForbiddenList(memory.assemblies)

        def forbid_growth(_estimated_bytes):
            raise AssertionError("fresh attention requested substrate growth reserve")

        memory.growth_guard = forbid_growth
        with mock.patch.object(
            memory,
            "save_sharded",
            side_effect=AssertionError("fresh attention rewrote substrate shards"),
        ), mock.patch.object(
            memory,
            "prune_unreferenced",
            side_effect=AssertionError("fresh attention pruned substrate shards"),
        ), mock.patch.object(
            brain,
            "runtime_card",
            side_effect=AssertionError("fresh attention rescanned runtime diagnostics"),
        ), mock.patch.object(
            brain,
            "_maintain_neural_state_resources",
            side_effect=AssertionError("fresh attention rescanned residency"),
        ), mock.patch.object(
            brain.resource_policy,
            "require_disk",
            wraps=brain.resource_policy.require_disk,
        ) as reserve:
            result = brain.start_fresh_attention("large-substrate-boundary")
            reserve_calls_after_commit = reserve.call_count
            repeated = brain.start_fresh_attention("large-substrate-boundary")

        self.assertTrue(result["committed"])
        self.assertFalse(result["idempotent"])
        self.assertTrue(repeated["idempotent"])
        self.assertEqual(reserve.call_count, reserve_calls_after_commit)
        self.assertEqual(reserve_calls_after_commit, 1)
        self.assertEqual(reserve.call_args.args[1], "neural-state checkpoint")
        self.assertEqual(result["boundary"]["cleared"]["activatedNeurons"], 2)
        self.assertEqual(
            result["boundary"]["cleared"]["substrateEligibilityTraces"], 1
        )
        self.assertEqual(result["synapsesPreserved"], cardinality)
        self.assertEqual(result["substrateContentSha256"], content_before)
        self.assertEqual((substrate_root / "manifest.json").read_bytes(), pointer_before)
        self.assertEqual(self._tree_hashes(substrate_root), tree_before)
        brain.close()

        reloaded = AdaptiveBrain.load(self.root, "fresh-scale-brain")
        self.assertEqual(len(reloaded.memory.neurons), cardinality)
        self.assertEqual(len(reloaded.memory.synapses), cardinality)
        self.assertEqual(
            reloaded.memory.persistence_manifest["contentSha256"], content_before
        )
        self.assertTrue(
            all(
                reloaded.memory.effective_activation(value) == 0.0
                for value in reloaded.memory.neurons.values()
            )
        )
        reloaded.close()

    def test_cfc_and_ltc_transient_state_reset_and_reload_without_weight_drift(self):
        for mode in ("cfc", "ltc"):
            with self.subTest(liquid_mode=mode):
                root = Path(self.temporary.name) / ("liquid-" + mode)
                brain_id = "fresh-liquid-" + mode
                brain = AdaptiveBrain(
                    brain_id,
                    root,
                    OmniConfig.micro(
                        max_seq_len=40,
                        learn_from_own_messages=False,
                        online_learning=False,
                        liquid_mode=mode,
                    ),
                )
                with torch.no_grad():
                    brain.liquid_state.fill_(0.625)
                brain.save()
                checksum = brain.parameter_checksum()
                nonzero_units = int(
                    torch.count_nonzero(brain.liquid_state).item()
                )
                self.assertGreater(nonzero_units, 0)

                result = brain.start_fresh_attention("fresh-liquid-" + mode)

                self.assertEqual(
                    result["boundary"]["cleared"]["liquidStateUnits"],
                    nonzero_units,
                )
                self.assertEqual(brain.parameter_checksum(), checksum)
                self.assertEqual(
                    int(torch.count_nonzero(brain.liquid_state).item()), 0
                )
                brain.close()

                reloaded = AdaptiveBrain.load(root, brain_id)
                self.assertEqual(reloaded.config.liquid_mode, mode)
                self.assertEqual(reloaded.parameter_checksum(), checksum)
                self.assertEqual(
                    int(torch.count_nonzero(reloaded.liquid_state).item()), 0
                )
                reloaded.close()


if __name__ == "__main__":
    unittest.main()
