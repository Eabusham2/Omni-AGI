#!/usr/bin/env python3
"""Build or verify one generation-bound OmniCortex forward index.

The first run against a legacy generation intentionally scans every synapse
shard once. The index is published atomically only after the complete scan.
Later runs prove that indexed cold loading opens zero synapse shards.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from omni_core.vsa import LazyPersistedSynapses, NeuralSubstrate


def emit(value: Dict[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "engine_directory",
        type=Path,
        help="Brain engine directory containing brain.json and substrate/",
    )
    parser.add_argument(
        "--scrub",
        action="store_true",
        help="After cold-load proof, verify every persisted synapse shard",
    )
    parser.add_argument("--progress-every", type=int, default=256)
    arguments = parser.parse_args()

    engine = arguments.engine_directory.expanduser().resolve()
    metadata = json.loads((engine / "brain.json").read_text("utf-8"))
    substrate_metadata = metadata.get("substrate")
    if not isinstance(substrate_metadata, dict):
        raise ValueError("engine metadata has no persisted neural substrate")
    pointer = substrate_metadata.get("persistence")
    if not isinstance(pointer, dict):
        raise ValueError("engine metadata has no sharded substrate generation")
    generation = str(pointer.get("activeGeneration", ""))
    index_path = (
        engine
        / "substrate"
        / "forward-index"
        / "generations"
        / (generation + ".json")
    )
    cached = index_path.is_file()
    opened = 0
    progress_every = max(1, int(arguments.progress_every))
    original_read = LazyPersistedSynapses._read_shard

    def counted_read(
        self: LazyPersistedSynapses,
        key: Tuple[str, int],
        *,
        validate: bool = False,
    ):
        nonlocal opened
        value = original_read(self, key, validate=validate)
        opened += 1
        if opened % progress_every == 0:
            emit(
                {
                    "event": "legacy-index-scan-progress",
                    "generation": generation,
                    "openedSynapseShards": opened,
                }
            )
        return value

    LazyPersistedSynapses._read_shard = counted_read
    try:
        started = time.perf_counter()
        memory = NeuralSubstrate.load_sharded(
            engine / "substrate",
            substrate_metadata,
            lazy_synapses=True,
        )
        build_elapsed = time.perf_counter() - started
        build_opens = opened
        if not index_path.is_file():
            raise RuntimeError("forward index was not published")

        opened = 0
        warm_started = time.perf_counter()
        warm = NeuralSubstrate.load_sharded(
            engine / "substrate",
            substrate_metadata,
            lazy_synapses=True,
        )
        warm_elapsed = time.perf_counter() - warm_started
        warm_opens = opened
    finally:
        LazyPersistedSynapses._read_shard = original_read

    status = warm.synapses.paging_status()
    emit(
        {
            "event": "forward-index-ready",
            "generation": generation,
            "cacheHit": cached,
            "synapses": len(memory.synapses),
            "shards": status["shards"],
            "indexBytes": index_path.stat().st_size,
            "indexSha256": sha256(index_path),
            "buildSeconds": build_elapsed,
            "buildShardOpens": build_opens,
            "indexedColdLoadSeconds": warm_elapsed,
            "indexedColdLoadShardOpens": warm_opens,
        }
    )
    if warm_opens != 0:
        raise RuntimeError("indexed cold load unexpectedly opened synapse shards")

    if arguments.scrub:
        def scrub_progress(done: int, total: int, _key: Tuple[str, int]) -> None:
            if done % progress_every == 0 or done == total:
                emit(
                    {
                        "event": "forward-index-scrub-progress",
                        "generation": generation,
                        "verifiedShards": done,
                        "totalShards": total,
                    }
                )

        emit(
            {
                "event": "forward-index-scrub-complete",
                **warm.synapses.scrub_persisted_shards(scrub_progress),
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
