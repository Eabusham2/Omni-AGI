"""Benchmark a generation-bound Brain Map overview without loading a brain.

Usage:
  PYTHONPATH=engine python engine/tests/benchmark_persisted_substrate_inspection.py \
    /path/to/brain/engine brain-id
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path


ENGINE = Path(__file__).resolve().parents[1]
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.substrate_inspection import (
    PersistedSubstrateView,
    query_persisted_substrate,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_directory", type=Path)
    parser.add_argument("brain_id")
    arguments = parser.parse_args()
    view = PersistedSubstrateView.open(
        arguments.engine_directory,
        arguments.brain_id,
    )
    index_path = (
        view.store
        / "inspection"
        / "generations"
        / view.revision
        / "manifest.json"
    )
    was_cached = index_path.is_file()
    started = time.perf_counter()
    page = query_persisted_substrate(
        arguments.engine_directory,
        arguments.brain_id,
        {"entity": "overview", "zoom": 0.72, "pageSize": 1_000_000},
    )
    elapsed = time.perf_counter() - started
    maximum_resident = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        maximum_resident *= 1024
    print(
        json.dumps(
            {
                "brainId": arguments.brain_id,
                "sourceGeneration": view.revision,
                "cacheHit": was_cached,
                "elapsedSeconds": elapsed,
                "maximumResidentBytes": maximum_resident,
                "totals": page["totals"],
                "clusters": page["matched"],
                "indexBytes": sum(
                    path.stat().st_size
                    for path in (view.store / "inspection").rglob("*")
                    if path.is_file()
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
