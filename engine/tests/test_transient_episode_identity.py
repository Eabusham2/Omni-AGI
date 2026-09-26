"""Pure transient episode checks without building or training a brain."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from omni_core.brain import AdaptiveBrain
from omni_core.memory_lifecycle import OrganicMemoryLifecycle
from omni_core.offload import PagedWorkingMemory


class _DiskReserve:
    def __init__(self):
        self.allow = True

    def require_disk(self, _estimated_bytes, _label):
        if not self.allow:
            raise RuntimeError("disk reserve denied")


class TransientEpisodeIdentityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory(prefix="omni-episode-identity-")
        self.addCleanup(self.temporary.cleanup)
        self.policy = _DiskReserve()
        self.pages = PagedWorkingMemory(
            Path(self.temporary.name) / "working.sqlite3", self.policy
        )
        self.page_events = []
        self.brain = SimpleNamespace(
            paged_working_memory=self.pages,
            hot_state_residency=SimpleNamespace(
                note_access=lambda ids, **kwargs: self.page_events.append(
                    (tuple(ids), kwargs)
                )
            ),
            memory_lifecycle=OrganicMemoryLifecycle(),
            config=SimpleNamespace(working_memory_slots=4, memory_resident_items=3),
            working_memory=[],
            workspace_items=[],
            counters={"workspace_rehearsals": 0, "workspace_evictions": 0},
        )

    def append(self, vector, assembly_id="shared-field"):
        AdaptiveBrain._append_working_memory(
            self.brain,
            vector,
            assembly_id=assembly_id,
            source="test",
            salience=0.7,
        )

    @staticmethod
    def vector(index):
        value = torch.zeros(16)
        value[index] = 1.0
        return value

    def test_distinct_same_field_episodes_remain_separate_and_repeats_merge(self):
        first = self.vector(0)
        distinct = self.vector(1)
        self.append(first)
        self.append(distinct)
        self.assertEqual(len(self.brain.working_memory), 2)
        self.assertTrue(torch.equal(self.brain.working_memory[0], first))
        self.assertTrue(torch.equal(self.brain.working_memory[1], distinct))

        near_repeat = first + self.vector(2) * 0.01
        self.append(near_repeat)
        self.assertEqual(len(self.brain.working_memory), 2)
        self.assertEqual(self.brain.counters["workspace_rehearsals"], 1)
        self.assertTrue(torch.equal(self.brain.working_memory[1], distinct))

    def test_distinct_cold_page_is_never_consumed_even_under_disk_pressure(self):
        cold = self.vector(0)
        cold_id = self.pages.append(
            cold,
            {"assemblyId": "shared-field", "salience": 0.9, "rehearsals": 2},
        )
        self.policy.allow = False
        preview = self.pages.peek_hot(hot_assembly_ids=["shared-field"])
        self.assertEqual(preview[0][1]["pageId"], cold_id)
        self.assertEqual(self.pages.page_ids(), (cold_id,))

        different = self.vector(1)
        self.append(different)
        self.assertEqual(self.pages.page_ids(), (cold_id,))
        self.assertTrue(torch.equal(self.pages.read(cold_id)[0], cold))
        self.assertEqual(len(self.brain.working_memory), 1)
        self.assertTrue(torch.equal(self.brain.working_memory[0], different))
        self.assertFalse(self.page_events[-1][1]["page_in"])

    def test_near_identical_cold_page_moves_into_one_resident_episode(self):
        cold = self.vector(0)
        cold_id = self.pages.append(
            cold,
            {"assemblyId": "shared-field", "salience": 0.8, "rehearsals": 2},
        )
        self.append(cold + self.vector(1) * 0.01)
        self.assertEqual(self.pages.count(), 0)
        self.assertEqual(len(self.brain.working_memory), 1)
        self.assertEqual(self.brain.workspace_items[0]["rehearsals"], 3)
        self.assertTrue(self.page_events[-1][1]["page_in"])
        with self.assertRaises(KeyError):
            self.pages.read(cold_id)

    def test_afterimages_separate_merge_by_similarity_then_fade_and_reactivate(self):
        lifecycle = OrganicMemoryLifecycle()
        first = self.vector(0)
        distinct = self.vector(1)

        def admit(vector):
            lifecycle._admit_afterimage(
                vector,
                assembly_id="shared-field",
                source="test",
                strength=0.25,
                salience=0.2,
                novelty=0.0,
                prediction_error=0.0,
                rehearsals=1,
                scores={},
                signals={},
            )

        admit(first)
        admit(distinct)
        self.assertEqual(len(lifecycle.afterimage_items), 2)
        self.assertTrue(torch.equal(lifecycle.afterimage_vectors[1], distinct))
        admit(first + self.vector(2) * 0.01)
        self.assertEqual(len(lifecycle.afterimage_items), 2)
        self.assertTrue(torch.equal(lifecycle.afterimage_vectors[1], distinct))

        lifecycle.cycle = 30
        with mock.patch.object(
            lifecycle, "_measure_signals", return_value=({}, 1, ())
        ):
            lifecycle._rescore_afterimages(
                SimpleNamespace(neurons={}),
                active_assembly_id="shared-field",
                active_vector=first,
                forgetting_rate=0.0,
                record_index={},
                synapse_index={},
                focus_index={"shared-field": {"currentlyFiring": True}},
            )
        self.assertEqual(lifecycle.afterimage_items[0]["lastActiveCycle"], 30)
        self.assertEqual(lifecycle.afterimage_items[1]["lastActiveCycle"], 0)
        self.assertEqual(lifecycle.afterimage_items[1]["inactiveCycles"], 30)

        lifecycle.cycle = 60
        for item in lifecycle.afterimage_items:
            item.update({
                "introducedCycle": 0,
                "lastActiveCycle": 0,
                "strength": 0.001,
                "salience": 0.01,
                "unfinishedScore": 0.0,
            })
        with mock.patch.object(
            lifecycle, "_measure_signals", return_value=({}, 1, ())
        ):
            lifecycle._rescore_afterimages(
                SimpleNamespace(neurons={}),
                active_assembly_id="unrelated",
                forgetting_rate=0.1,
                record_index={},
                synapse_index={},
                focus_index={},
            )
        self.assertEqual(lifecycle.afterimage_items, [])
        self.assertEqual(lifecycle.expired_afterimages, 2)
        admit(first)
        self.assertEqual(len(lifecycle.afterimage_items), 1)


if __name__ == "__main__":
    unittest.main()
