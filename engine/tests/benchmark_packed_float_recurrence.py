"""Opt-in fixed tiny recurrence comparison, never training or a speed gate.

Run: PYTHONPATH=engine .venv/bin/python -m engine.tests.benchmark_packed_float_recurrence
The decoded path below is the former implementation, retained ONLY as a
benchmark baseline. Production recurrence must not call it.
"""
from contextlib import closing
import hashlib
import json
from pathlib import Path
import platform
from statistics import median
import tempfile
from time import perf_counter_ns

import torch

from omni_core.router_state_paging import RouterStatePager
from .test_packed_float_recurrence import repeated_fixture
from .test_router_state_paging import owner as dense_owner


@torch.no_grad()
def decoded_baseline(owner, activity):
    with owner._operation(), owner._ram(
            (owner.pre_neurons + owner.post_neurons) * activity.element_size() * 3,
            "benchmark old recurrent live vectors"):
        vector = activity.detach().reshape(-1).to(device="cpu").contiguous()
        output = torch.zeros(owner.post_neurons, dtype=vector.dtype)
        with closing(owner.iter_effective_tiles()) as tiles:
            for (r0, r1, c0, c1), levels in tiles:
                output[r0:r1].add_(torch.mv(levels.to(vector.dtype), vector[c0:c1]))
        return output.to(activity.device)


def milliseconds(call, repetitions):
    start = perf_counter_ns()
    for _ in range(repetitions):
        call()
    return (perf_counter_ns() - start) / repetitions / 1_000_000


def measure(root, *, label, hot, tile, dense=False):
    pager = RouterStatePager(root, hot_bytes=hot, tile_bytes=tile, journal_ram_bytes=0)
    try:
        if dense:
            owner = dense_owner(pager, rows=67, columns=69)
            generator = torch.Generator().manual_seed(114)
            owner.set_effective_weights(torch.randint(-1, 2, (67, 69), dtype=torch.int8, generator=generator))
        else:
            owner, _ = repeated_fixture(pager)
        activity = (torch.arange(owner.pre_neurons).float() - 34) / 32
        before = owner.checksum_with_controls()
        expected = decoded_baseline(owner, activity)
        actual = owner.recurrent(activity)
        if not torch.equal(actual, expected):
            raise AssertionError("fixed dyadic recurrence fixture must match exactly")
        tile_count = raw_bytes = old_float_bytes = 0
        with closing(owner.iter_packed_tiles()) as tiles:
            for (r0, r1, c0, c1), signs, states in tiles:
                tile_count += 1
                raw_bytes = max(raw_bytes, signs.numel() + (0 if states is None else states.numel()))
                old_float_bytes = max(old_float_bytes, (r1-r0)*(c1-c0)*activity.element_size())
        old, new = lambda: decoded_baseline(owner, activity), lambda: owner.recurrent(activity)
        for _ in range(2):
            old()
            new()
        samples = {"decoded": [], "packed": []}
        for index in range(7):
            calls = (("decoded", old), ("packed", new)) if index % 2 == 0 else (("packed", new), ("decoded", old))
            for name, call in calls:
                samples[name].append(milliseconds(call, 5))
        if owner.checksum_with_controls() != before:
            raise AssertionError("read-only recurrence changed learned state")
        status = pager.status()
        return {
            "case": label, "rows": owner.post_neurons, "columns": owner.pre_neurons,
            "allocatedBlocks": len(owner.blocks) if hasattr(owner, "blocks") else None,
            "allocatedContactSlots": owner.logical_ternary_parameter_count,
            "tileBudgetBytes": tile, "tilesPerCall": tile_count,
            "residentStateBytes": status["cpuHeapBytes"], "mappedLogicalStateBytes": status["mappedLogicalBytes"],
            "maximumRawSignGainPayloadBytes": raw_bytes,
            "formerMaximumFloatWeightTileBytesOnly": old_float_bytes,
            "exactFixedDyadicOutput": True, "learnedStateUnchanged": True,
            "samplesMillisecondsPerCall": samples,
            "decodedMedianMilliseconds": median(samples["decoded"]),
            "packedMedianMilliseconds": median(samples["packed"]),
            "decodedOverPackedRatio": median(samples["decoded"]) / median(samples["packed"]),
        }
    finally:
        pager.close()


def main():
    torch.set_num_threads(1)
    core = Path(__file__).resolve().parents[1] / "omni_core"
    with tempfile.TemporaryDirectory(prefix="packed-float-comparison-") as directory:
        root = Path(directory)
        cases = [measure(root / str(index), **configuration) for index, configuration in enumerate((
            {"label": "sparse-repeated-hot-64KiB", "hot": 1_000_000, "tile": 65536},
            {"label": "sparse-repeated-mapped-64KiB", "hot": 0, "tile": 65536},
            {"label": "sparse-repeated-hot-1MiB", "hot": 1_000_000, "tile": 1048576},
            {"label": "dense-compatible-hot-64KiB", "hot": 1_000_000, "tile": 65536, "dense": True},
        ))]
    print(json.dumps({
        "scope": "fixed constructor-free CPU recurrence only; no brain, learning, corpus, or accelerator timing",
        "platform": platform.platform(), "architecture": platform.machine(), "torch": torch.__version__,
        "threads": torch.get_num_threads(), "pairedRounds": 7, "callsPerSample": 5,
        "nativeManifest": json.loads((core / "native/manifest.json").read_text()),
        "sourceSha256": {name: hashlib.sha256((core / name).read_bytes()).hexdigest() for name in
            ("spiking.py", "sparse_router_state.py", "packed_native_kernel.py", "kernels/weighted_ternary.cpp", "kernels/weighted_ternary.metal")},
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()
