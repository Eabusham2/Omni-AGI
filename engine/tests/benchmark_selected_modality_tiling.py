"""Bounded CPU comparison, not a media-quality or whole-process RSS benchmark.

Run directly with PYTHONPATH=engine. The reference reproduces the former
selected-derivative shape using actual packed learning, never float weights.
"""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

import torch

from omni_core.modalities import PackedTernaryTable
from omni_core.model import _apply_packed_gradient_rows
from omni_core.packed_collective_hooks import packed_row_owner
from omni_core.persistence import tensor_checksum
from omni_core.working_attention_paging import WorkingAttentionPager


def full_width_reference(table, ids, gradient):
    """Former production selected path: 64 dimensions, all code columns."""
    projection = table.projection
    with torch.no_grad(), packed_row_owner(projection):
        for start in range(0, table.dimensions, 64):
            end = min(table.dimensions, start + 64)
            derivative = torch.zeros((end - start, table.rows), dtype=torch.float32)
            for sample in range(0, ids.numel(), 256):
                derivative.index_add_(1, ids[sample:sample + 256],
                    gradient[sample:sample + 256, start:end].transpose(0, 1))
            _apply_packed_gradient_rows(projection.packed_forward_weight(), table.rows, start,
                derivative, projection.online_learning_rate, projection._packed_forward_scale,
                row_stability=projection._row_stability,
                stability_strength=projection._packed_stability_strength,
                connection_state=projection._connection_state)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    codes, dimensions, trials = 8193, 65, 5
    base = PackedTernaryTable(codes, dimensions, scale=1.)
    base.projection.fill_ternary_(1)
    base.projection.online_learning_rate = 1000.
    base.projection._packed_stability_strength = .5
    base.projection._row_stability.fill_(3)
    base.projection.add_parallel_connection_(0, 64, sign=1, strength=2)
    ids = torch.tensor([0, 63, 64, codes - 1, 64])
    gradient = -torch.ones(ids.numel(), dimensions)
    before = tensor_checksum(base.state_dict().values())
    timings = {'fullWidth': [], 'twoAxisNormal': [], 'twoAxisPressure': []}
    shapes, peaks = {}, {'twoAxisNormal': 0, 'twoAxisPressure': 0}
    with tempfile.TemporaryDirectory(prefix='omni-selected-learning-comparison-') as directory:
        for index in range(trials):
            tables = {name: copy.deepcopy(base) for name in timings}
            pagers = {name: WorkingAttentionPager(Path(directory) / (str(index) + name), resident_budget_bytes=0,
                scratch_budget_bytes=1024 * 1024, device_tile_budget_bytes=budget)
                for name, budget in (('twoAxisNormal', 1024 * 1024), ('twoAxisPressure', 32 * 1024))}
            try:
                # Alternate order; construction/copies/checksums are excluded.
                modes = tuple(timings) if index % 2 == 0 else tuple(reversed(timings))
                for mode in modes:
                    started = time.perf_counter_ns()
                    if mode == 'fullWidth':
                        full_width_reference(tables[mode], ids, gradient)
                    else:
                        with pagers[mode].activate():
                            shapes[mode] = tables[mode]._learning_tile_shape(ids.numel())
                            tables[mode].learn_selected(ids, gradient)
                    timings[mode].append((time.perf_counter_ns() - started) / 1e6)
                for mode, pager in pagers.items():
                    peaks[mode] = max(peaks[mode], pager.status()['peakComputeTileBytes'])
                checksums = {tensor_checksum(table.state_dict().values()) for table in tables.values()}
                if len(checksums) != 1:
                    raise AssertionError('actual native state diverged from full-width selected derivative reference')
            finally:
                for pager in pagers.values():
                    pager.close()
    if before != tensor_checksum(base.state_dict().values()):
        raise AssertionError('comparison altered its immutable fixture source')
    root = Path(__file__).resolve().parents[1] / 'omni_core'
    sources = ('modalities.py', 'packed_gradient_tiles.py', 'packed_collective_tiles.py')
    digest = hashlib.sha256()
    for name in sources:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    print(json.dumps({'format': 'omni-selected-learning-comparison-1', 'backend': 'cpu',
        'sourceSha256': digest.hexdigest(), 'codes': codes, 'dimensions': dimensions,
        'selectedOccurrences': ids.numel(), 'trials': trials, 'milliseconds': timings,
        'medianMilliseconds': {key: statistics.median(values) for key, values in timings.items()},
        'oldDerivativeFloatBytes': min(dimensions, 64) * codes * 4,
        'newDerivativeFloatBytes': {key: shape[0] * shape[1] * 4 for key, shape in shapes.items()},
        'newAdmittedLearningPeakBytes': peaks,
        'twoAxisTile': {key: {'dimensions': shape[0], 'codes': shape[1], 'samples': shape[2]}
            for key, shape in shapes.items()},
        'nativeStateMatched': True, 'sourceUnchanged': True,
        'scope': 'selected-code CPU update only; not whole-model throughput or resident memory'}, indent=2))


if __name__ == '__main__':
    main()
