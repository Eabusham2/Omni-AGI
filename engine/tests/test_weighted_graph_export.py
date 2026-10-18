import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.vsa import NeuralSubstrate
from omni_core.ternary_packing import export_module_ternary_shards, verify_ternary_shards


class WeightedGraphExportTests(unittest.TestCase):
    def graph(self):
        memory = NeuralSubstrate(16, seed=211)
        source = memory._ensure_neuron("first", 1.)
        target = memory._ensure_neuron("second", 1.)
        for moment in range(18):
            memory._strengthen_synapse(source, target, float(moment + 2), kind="association", amount=10.)
        return memory

    def owner(self, memory):
        brain = AdaptiveBrain.__new__(AdaptiveBrain)
        brain.memory = memory
        brain.resource_policy = SimpleNamespace(require_disk=lambda *_: None)
        brain._allow_substrate_growth = lambda *_: True
        return brain

    def test_graph_strengths_and_repeated_contact_rows_survive_inference_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = self.graph()
            memory.save_sharded(root / "substrate", records_per_shard=2)
            for ordinal, source in enumerate((memory, NeuralSubstrate.load_sharded(
                root / "substrate", memory.metadata(include_records=False), lazy_synapses=True))):
                brain = self.owner(source)
                specs = brain._dynamic_ternary_tensors()
                output = root / str(ordinal)
                export_module_ternary_shards(output, {}, dynamic_synapses=specs)
                verified = verify_ternary_shards(output)
                name = "substrate.dynamic_synapses.weights"
                signs = verified.tensors[name].reshape(-1).to(torch.int64)
                gains = (verified.connection_states[name].reshape(-1).to(torch.int64) & 15) + 1
                self.assertEqual(sorted((signs * gains).tolist()), [2, 16])

    def test_unmatched_sign_and_strength_order_is_rejected(self):
        brain = self.owner(self.graph())
        values, count, order, basis = brain._dynamic_synapse_pack_state()
        with self.assertRaisesRegex(ValueError, "different ordering"):
            brain._dynamic_ternary_tensors(values, substrate_order=(count, "0" * 64, basis))

    def test_resident_graph_mutation_invalidates_pending_companion(self):
        brain = self.owner(self.graph())
        spec = next(iter(brain._dynamic_ternary_tensors().values()))
        key = next(iter(brain.memory.synapses))
        brain.memory.synapses[key]["connection_strength"] = 4
        with self.assertRaisesRegex(ValueError, "changed"):
            list(spec.connection_state.pages())


if __name__ == "__main__":
    unittest.main()
