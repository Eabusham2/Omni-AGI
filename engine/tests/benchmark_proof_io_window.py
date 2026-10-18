"""Paired SQLite proof I/O timings; no model construction or training."""
import hashlib
import json
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

from omni_core.authenticated_paged_cache import cache_session
from omni_core.paged_assembly_index import PagedAssemblyIndex


def measure(count=250, runs=3):
    source = Path(__file__).resolve().parents[1] / "omni_core" / "authenticated_paged_cache.py"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    samples = {"baseline": [], "window": []}
    with tempfile.TemporaryDirectory(prefix="omni-proof-io-benchmark-") as folder:
        root = Path(folder)
        for trial in range(runs):
            for label in ("baseline", "window") if trial % 2 == 0 else ("window", "baseline"):
                owner = PagedAssemblyIndex(root / ("%s-%d.sqlite3" % (label, trial)))
                session = cache_session(owner)
                session._ensure_proofs()
                identities = [(1, index + 1, 4096, index + 1000, index + 2000) for index in range(count)]
                started = time.perf_counter_ns()
                with session.proof_io_window() if label == "window" else nullcontext():
                    for index, identity in enumerate(identities):
                        path = root / ("blob-%05d" % index)
                        checksum = hashlib.sha256(str(index).encode()).hexdigest()
                        if not session.remember_blob(path, checksum, identity) or session.lookup_blob(path, identity) != checksum:
                            raise RuntimeError("proof benchmark verification failed")
                samples[label].append((time.perf_counter_ns() - started) / 1e6)
                if session.proof_writes != count or session.proof_hits != count:
                    raise RuntimeError("proof benchmark coverage changed")
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("benchmark source changed during measurements")
    return {"scope": "proof SQLite write+authenticated lookup, including final close",
            "rowsPerRun": count, "runs": runs, "sourceSha256": source_hash,
            "milliseconds": samples,
            "medianMilliseconds": {key: statistics.median(values) for key, values in samples.items()},
            "wholeBrainSpeedMeasured": False}


if __name__ == "__main__":
    print(json.dumps(measure(), indent=2))
