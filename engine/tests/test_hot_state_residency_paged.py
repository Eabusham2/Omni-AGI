"""Pure residency-planner checks that must not traverse paged populations."""

import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.offload import HotStateResidencyPlanner


class PagedNeurons(Mapping[str, Mapping[str, Any]]):
    def __init__(self) -> None:
        self.records = {
            "active": {"activation": 1.0, "exposures": 1},
            "page": {"activation": 0.0, "exposures": 0},
        }

    def status(self) -> dict[str, int]:
        return {"rowCount": 1_000_000}

    def iter_pages(self):
        raise AssertionError("paged neurons must not be traversed")

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        return self.records[key]

    def __iter__(self):
        raise AssertionError("paged neurons must not be traversed")

    def __len__(self) -> int:
        raise AssertionError("planner should use the exact status count")


class PagedAssemblies(Sequence[Mapping[str, Any]]):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        raise AssertionError("paged assemblies must not be traversed")

    def get_by_id(self, record_id: str):
        return {"id": record_id, "importance": 0.0} if record_id in {"active", "page"} else None


class PagedSynapses(Mapping[str, Mapping[str, Any]]):
    persisted_cold_count = 1

    def __init__(self) -> None:
        self.resident = {"syn": {"uses": 0, "stability": 0.0}}

    def resident_items(self):
        return self.resident.items()

    def __contains__(self, key: object) -> bool:
        return key in {"syn", "cold-syn"}

    def __getitem__(self, key: str) -> Mapping[str, Any]:
        raise AssertionError("cold synapses must not be restored")

    def __iter__(self):
        raise AssertionError("paged synapses must not be traversed")

    def __len__(self) -> int:
        return 2


class PagedHotResidencyTests(unittest.TestCase):
    def test_counts_and_page_transitions_without_corpus_traversal(self):
        planner = HotStateResidencyPlanner()
        inputs = {
            "neurons": PagedNeurons(),
            "assemblies": PagedAssemblies(),
            "synapses": PagedSynapses(),
            "unfinished_ids": ["active"],
            "attention_active_ids": ["active"],
            "resident_budget": 2,
            "paged_assembly_ids": ["page"],
        }
        first = planner.update(**inputs)
        self.assertEqual(first["totalEntities"], 1_000_002)
        self.assertEqual(first["hotEntities"], 2)
        self.assertEqual(first["coldEntities"], 1_000_000)
        self.assertEqual(first["persistedColdSynapses"], 1)
        self.assertFalse(first["coldIdSetComplete"])
        self.assertEqual(planner.hot_ids, frozenset({"active", "syn"}))

        planner.note_access(["page", "page"])
        second = planner.update(**inputs)
        self.assertEqual(second["totalEntities"], 1_000_002)
        self.assertEqual(second["hotEntities"] + second["coldEntities"], 1_000_002)
        self.assertEqual(planner.hot_ids, frozenset({"active", "page"}))
        self.assertEqual(planner.became_cold_ids, frozenset({"syn"}))
        self.assertEqual(planner.page_in_candidate_ids, frozenset({"page"}))
        self.assertEqual(planner.page_out_candidate_ids, frozenset({"syn"}))
        self.assertEqual(second["pageInCandidates"], 1)
        self.assertEqual(second["pageOutCandidates"], 1)


if __name__ == "__main__":
    unittest.main()
