"""Manual, no-brain-Build timing of warm memory settling against full scan.

Run: .venv/bin/python engine/tests/benchmark_memory_lifecycle_settle.py
This creates only in-memory NeuralSubstrate fixtures and writes no user data.
"""

import json
import time
from types import SimpleNamespace

from test_memory_lifecycle_scaling import (
    _RouterSynapses,
    _WithoutAssemblyIndex,
    _settle,
    _substrate,
)

from omni_core.memory_lifecycle import OrganicMemoryLifecycle


def _time_cycles(count, *, indexed, forgetting_rate, cycles=12):
    memory = _substrate(count)
    substrate = memory if indexed else _WithoutAssemblyIndex(memory)
    lifecycle = OrganicMemoryLifecycle()
    router = SimpleNamespace(synapses=_RouterSynapses())
    # Exclude the initial exact-index build and one-time legacy-label migration.
    _settle(
        lifecycle, substrate, router, "assembly-00000",
        forgetting_rate=forgetting_rate,
    )
    started = time.perf_counter()
    for index in range(cycles):
        _settle(
            lifecycle, substrate, router,
            "assembly-%05d" % (index % 2),
            forgetting_rate=forgetting_rate,
        )
    return (time.perf_counter() - started) / cycles


if __name__ == "__main__":
    observations = []
    for count in (1_024, 8_192, 32_768):
        for forgetting_rate in (0.0, 0.04):
            full_scan = _time_cycles(
                count, indexed=False, forgetting_rate=forgetting_rate,
            )
            indexed = _time_cycles(
                count, indexed=True, forgetting_rate=forgetting_rate,
            )
            observations.append(
                {
                    "assemblies": count,
                    "forgettingRate": forgetting_rate,
                    "fullScanMs": round(full_scan * 1000, 3),
                    "indexedMs": round(indexed * 1000, 3),
                    "speedup": round(full_scan / indexed, 2),
                }
            )
    print(json.dumps(observations, indent=2))
