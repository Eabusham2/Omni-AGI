"""One bounded paired native graph.load IO comparison; no brain/model run."""
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.paged_read_session import ReadSession
from omni_core.paged_recall_graph import PagedRecallGraph
from test_paged_read_session import neuron


def run():
    with tempfile.TemporaryDirectory(prefix='omni-paged-read-io-') as folder:
        root = Path(folder); nodes = PagedNeuronMetadata(root / 'nodes.sqlite3')
        nodes.import_page([neuron(i) for i in range(32)], max_rows=32)
        edges = [('e%d' % i, 'n%d' % (i % 33), 'n%d' % ((i + 3) % 33), (-1 if i % 3 else 1) * (1 + i % 16)) for i in range(384)]
        timings, hashes, opens = {'original': [], 'keeper': []}, [], []
        for repetition in range(3):
            for name in (('original', 'keeper') if repetition % 2 == 0 else ('keeper', 'original')):
                graph = PagedRecallGraph(root / ('run-%d-%s' % (repetition, name)), reserve=lambda *_args: True)
                started = time.perf_counter_ns()
                try:
                    if name == 'keeper':
                        with ReadSession([nodes]) as scope:
                            graph.load(iter(edges), node_exists=lambda key: key in nodes,
                                synapse_revision=1, neuron_revision=nodes.graph_revision)
                            opens.append(scope.opened_connections)
                    else:
                        graph.load(iter(edges), node_exists=lambda key: key in nodes,
                            synapse_revision=1, neuron_revision=nodes.graph_revision)
                    timings[name].append((time.perf_counter_ns() - started) / 1e6)
                    raw = list(graph.connection.execute('SELECT * FROM raw_edges ORDER BY ordinal'))
                    hashes.append(hashlib.sha256(json.dumps(raw, separators=(',', ':')).encode()).hexdigest())
                finally:
                    graph.close()
        if len(set(hashes)) != 1 or opens != [1, 1, 1]:
            raise AssertionError('paired real graph traversal differs or keeper was not bounded')
        return {'format': 'omni-paged-read-io-comparison', 'version': 1, 'componentOnly': True,
            'brainConstructed': False, 'edgesPerTraversal': len(edges), 'pairedRepetitions': 3,
            'milliseconds': timings, 'medianMilliseconds': {key: statistics.median(value) for key, value in timings.items()},
            'speedRatio': statistics.median(timings['original']) / statistics.median(timings['keeper']),
            'exactOrderedGraphSha256': hashes[0], 'keeperConnectionsPerTraversal': opens,
            'keeperCacheRamEnvelopeBytes': 320 * 1024, 'wholeProcessRssMeasured': False,
            'modelLoadSpeedClaim': False}


if __name__ == '__main__': print(json.dumps(run(), sort_keys=True, indent=2))
