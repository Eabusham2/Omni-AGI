"""Exact rollback of a failed connected regional admission."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.connected_transaction import ConnectedExperienceTransaction
from omni_core.full_merge import connected_descriptor, merged_config
from omni_core.offload import ResourcePolicy, ResourceReading
from omni_core.persistence import tensor_checksum
from omni_core.shared_resource_ledger import SharedResourceLedger
from tests.test_full_merge_regions import config, region, snapshot


def graph_state(loaded):
    result = {}
    for identifier, memory in loaded.iter_memories():
        vectors = {key: (memory.neuron_vectors.row_payload(key), memory.neuron_vectors.update_count(key))
                   for key in memory.neuron_vectors}
        result[identifier] = {
            "neurons": {key: dict(value) for key, value in memory.neurons.items()},
            "assemblies": [dict(value) for value in memory.assemblies],
            "synapses": {key: copy.deepcopy(dict(value)) for key, value in memory.synapses.items()},
            "vectors": vectors,
            "revision": memory.state_revision,
            "growth": (memory.growth_events, memory.growth_pauses),
            "attention": copy.deepcopy(memory.attention_overlay_metadata()),
        }
    return result


class ConnectedTransactionsTests(unittest.TestCase):
    def test_failed_second_region_restores_first_region_without_full_graph_copy(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parents = [snapshot(root, region("a")), snapshot(root, region("b", 16))]
            descriptor = connected_descriptor(parents)
            settings = merged_config(descriptor, settings=config())
            quota = SharedResourceLedger(root / "quota.sqlite3")
            def policy(path, **kwargs):
                kwargs.pop("device_resource_settings", None)
                return ResourcePolicy(path, **kwargs, shared_ledger=quota,
                    reading_provider=lambda: ResourceReading(total_memory_bytes=8 * 1024**3,
                        available_memory_bytes=6 * 1024**3, process_memory_bytes=256 * 1024**2,
                        disk_total_bytes=128 * 1024**3, disk_free_bytes=100 * 1024**3,
                        managed_process_memory_bytes=256 * 1024**2, managed_memory_verified=True))
            with patch("omni_core.brain.ResourcePolicy", policy):
                brain = AdaptiveBrain("merged", root, settings, _connected_descriptor=descriptor)
                try:
                    graph_before = graph_state(brain._connected_regions)
                    core_before = tensor_checksum(brain.decoder.state_dict().values())
                    blocks_before = [set(part.router.synapses.blocks) for part in brain.decoder.regions]
                    with patch.object(brain.decoder.regions[1].router, "route", side_effect=RuntimeError("second-region fault")):
                        with self.assertRaisesRegex(RuntimeError, "second-region fault"):
                            with ConnectedExperienceTransaction(brain):
                                brain.decoder.observe_experience(torch.ones(1, 24), event_key="failed-neural-event")
                    self.assertEqual(graph_state(brain._connected_regions), graph_before)
                    self.assertEqual(tensor_checksum(brain.decoder.state_dict().values()), core_before)
                    self.assertEqual([set(part.router.synapses.blocks) for part in brain.decoder.regions], blocks_before)
                    self.assertEqual(list((brain._live_paging_cache_directory / "connected-admission").iterdir()), [])
                    # The same valid update commits after a fresh recovery
                    # boundary; a later Stop must not erase this admission.
                    brain._connected_recovery_required = False
                    with ConnectedExperienceTransaction(brain):
                        brain.decoder.observe_experience(torch.ones(1, 24), event_key="committed-neural-event")
                    graph_committed = graph_state(brain._connected_regions)
                    self.assertNotEqual(graph_committed, graph_before)
                    with self.assertRaisesRegex(InterruptedError, "later admission"):
                        with ConnectedExperienceTransaction(brain):
                            brain.decoder.observe_experience(torch.zeros(1, 24), event_key="later-failed-event")
                            raise InterruptedError("later admission stopped")
                    self.assertEqual(graph_state(brain._connected_regions), graph_committed)
                finally:
                    brain.close()


if __name__ == "__main__":
    unittest.main()
