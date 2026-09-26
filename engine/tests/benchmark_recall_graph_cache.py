"""No-Build timing comparison for repeated exact substrate recall.

Run: python engine/tests/benchmark_recall_graph_cache.py
"""

import json
import statistics
import sys
import time
from pathlib import Path

import torch

ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.vsa import (
    LazyPersistedSynapses,
    NeuralSubstrate,
    _unpack_ternary_level,
)


def make_substrate(
    edge_count: int = 12_000, live_every: int = 1
) -> tuple[NeuralSubstrate, torch.Tensor]:
    memory = NeuralSubstrate(16, seed=13)
    cue = torch.tensor([1.0] + [0.0] * 15)
    for identifier in ("seed", "neighbor"):
        memory.neurons[identifier] = {"id": identifier}
        memory.assemblies.append({"id": identifier, "neuron_ids": []})
    memory.assembly_vectors["seed"] = cue
    memory.assembly_vectors["neighbor"] = torch.tensor(
        [0.8, 0.2] + [0.0] * 14
    )
    for index in range(edge_count):
        source = f"cold-{index % 2_000}"
        target = f"cold-{(index + 1) % 2_000}"
        memory.neurons.setdefault(source, {"id": source})
        memory.neurons.setdefault(target, {"id": target})
        identifier = f"{source}>{target}:bench-{index}"
        memory.synapses[identifier] = {
            "id": identifier,
            "source_id": source,
            "target_id": target,
            "effective_weight": (
                (1 if index % 3 else -1) if index % live_every == 0 else 0
            ),
        }
    memory.synapses["seed>neighbor:bench"] = {
        "id": "seed>neighbor:bench",
        "source_id": "seed",
        "target_id": "neighbor",
        "effective_weight": 1,
    }
    return memory, cue


def timed_recall(memory: NeuralSubstrate, cue: torch.Tensor, count: int) -> float:
    samples = []
    for _ in range(count):
        start = time.perf_counter()
        memory.recall_vector(cue, workspace_slots=8, record_activity=False)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def timed_mutated_recall(
    memory: NeuralSubstrate, cue: torch.Tensor, count: int
) -> float:
    samples = []
    record = memory.synapses["seed>neighbor:bench"]
    for _ in range(count):
        record["effective_weight"] = -int(record["effective_weight"])
        start = time.perf_counter()
        memory.recall_vector(cue, workspace_slots=8, record_activity=False)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def legacy_lazy_edge_scan(lazy: LazyPersistedSynapses) -> list:
    """Pre-optimization scan shape, retained here only as a timing reference."""

    result = []
    for key in lazy._ordered_keys:
        replacements = {
            index: lazy._dirty[record_id]
            for record_id, (location_key, index) in lazy._dirty_locations.items()
            if location_key == key and record_id in lazy._dirty
        }
        deleted = {
            index
            for _record_id, (location_key, index) in lazy._deleted_locations.items()
            if location_key == key
        }
        structures, packed = lazy._forward_by_shard.get(key, ([], b""))
        base = {
            index: (record_id, source, target, _unpack_ternary_level(packed, offset))
            for offset, (index, record_id, source, target) in enumerate(structures)
        }
        for index in sorted(set(base).union(replacements)):
            if index in deleted:
                continue
            replacement = replacements.get(index)
            if replacement is not None:
                weight = NeuralSubstrate.exact_effective_weight(
                    replacement.get("effective_weight", 0)
                )
                if weight:
                    result.append(
                        (
                            str(replacement["source_id"]),
                            str(replacement["target_id"]),
                            weight,
                        )
                    )
                continue
            _record_id, source, target, weight = base[index]
            result.append((source, target, weight))
    for record_id, record in lazy._dirty.items():
        if record_id in lazy._new_ids:
            weight = NeuralSubstrate.exact_effective_weight(
                record.get("effective_weight", 0)
            )
            if weight:
                result.append(
                    (str(record["source_id"]), str(record["target_id"]), weight)
                )
    return result


def measure_lazy_overlay_scan() -> dict:
    lazy = LazyPersistedSynapses(Path.cwd(), [], 0, 1)
    lazy._ordered_keys = [(f"{index % 16:x}", index) for index in range(5_000)]
    lazy._forward_by_shard = {
        key: ([(0, f"edge-{index}", f"source-{index}", f"target-{index}")], b"\x56")
        for index, key in enumerate(lazy._ordered_keys)
    }
    for index, key in enumerate(lazy._ordered_keys[:100]):
        identifier = f"edge-{index}"
        lazy._dirty_locations[identifier] = (key, 0)
        lazy._dirty[identifier] = {
            "source_id": f"source-{index}",
            "target_id": f"target-{index}",
            "effective_weight": -1,
        }
    assert list(lazy.iter_effective_edges()) == legacy_lazy_edge_scan(lazy)

    def median(call):
        samples = []
        for _ in range(8):
            start = time.perf_counter()
            call()
            samples.append(time.perf_counter() - start)
        return statistics.median(samples)

    current = median(lambda: list(lazy.iter_effective_edges()))
    legacy = median(lambda: legacy_lazy_edge_scan(lazy))
    return {
        "shards": len(lazy._ordered_keys),
        "dirtyShards": 100,
        "currentMedianMs": round(current * 1000, 3),
        "legacyMedianMs": round(legacy * 1000, 3),
        "speedup": round(legacy / current, 2),
        "scope": "synthetic indexed lazy scan after sparse mutation; no disk I/O",
    }


def measure(live_every: int) -> dict:
    memory, cue = make_substrate(live_every=live_every)
    cached_result = memory.recall_vector(
        cue, workspace_slots=8, record_activity=False
    )
    cached_audit = dict(memory._last_recall_audit)
    cached = timed_recall(memory, cue, 12)
    mutated = timed_mutated_recall(memory, cue, 12)
    original_nodes = memory.neurons
    memory.neurons = dict(original_nodes)
    try:
        uncached_result = memory.recall_vector(
            cue, workspace_slots=8, record_activity=False
        )
        assert torch.equal(cached_result[0], uncached_result[0])
        assert cached_result[1] == uncached_result[1]
        assert cached_audit == memory._last_recall_audit
        uncached = timed_recall(memory, cue, 12)
    finally:
        memory.neurons = original_nodes
    return {
        "nodes": len(memory.neurons),
        "synapses": len(memory.synapses),
        "liveEdges": 1 + sum(
            int(record["effective_weight"] != 0)
            for record in memory.synapses.values()
            if record["id"] != "seed>neighbor:bench"
        ),
        "cachedMedianMs": round(cached * 1000, 3),
        "uncachedMedianMs": round(uncached * 1000, 3),
        "speedup": round(uncached / cached, 2),
        "mutationBetweenRecallsMedianMs": round(mutated * 1000, 3),
        "scope": "one cue, exact eager substrate recall, no Build",
    }


def main() -> None:
    print(
        json.dumps(
            {
                "allLive": measure(1),
                "mostlyDormant": measure(20),
                "lazyOverlayScan": measure_lazy_overlay_scan(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
