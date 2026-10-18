"""Bounded paired SQL-component comparison; never constructs a model/brain.

Both variants use the existing admitted ReadSession. This isolates endpoint
membership batching rather than repeating the earlier keeper comparison.
"""

import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time
from unittest.mock import patch

from omni_core.paged_neuron_metadata import PagedNeuronMetadata
from omni_core.paged_recall_graph import PagedRecallGraph


def compare():
    deadline = time.monotonic() + 30.0
    with tempfile.TemporaryDirectory(prefix="omni-membership-io-") as folder:
        root = Path(folder)
        nodes = PagedNeuronMetadata(root / "nodes.sqlite3")
        nodes.import_page([{"id": "n%d" % index, "neuron_id": "n%d" % index,
                            "label": "node %d" % index, "region": "semantic"}
                           for index in range(32)], max_rows=32)
        revision = nodes.graph_revision
        source_sha = hashlib.sha256(nodes.path.read_bytes()).hexdigest()
        edges = [("e%d" % index, "n%d" % (index % 33), "n%d" % ((index + 1) % 33),
                  (-1 if index % 3 else 1) * (1 + index % 16)) for index in range(384)]
        samples = {"single_membership_ms": [], "batched_membership_ms": []}
        query_counts = {"single_membership_ms": [], "batched_membership_ms": []}
        expected_hash = None
        for pair in range(3):
            order = (False, True) if pair % 2 == 0 else (True, False)
            for batched in order:
                if time.monotonic() >= deadline:
                    raise TimeoutError("bounded membership comparison exceeded its deadline")
                graph = PagedRecallGraph(root / ("pair-%d-%s" % (pair, batched)),
                                         reserve=lambda *_args: True)
                observed = []
                connect = nodes._connect

                def traced_connection():
                    connection = connect()
                    connection.set_trace_callback(observed.append)
                    return connection

                try:
                    with patch.object(nodes, "_connect", traced_connection):
                        start = time.perf_counter()
                        graph.load(iter(edges), node_exists=lambda value: value in nodes,
                                   synapse_revision=1, neuron_revision=revision, read_stores=(nodes,),
                                   node_exists_many=(lambda values: nodes.contains_many(
                                       values, expected_graph_revision=revision)) if batched else None)
                        elapsed = 1000.0 * (time.perf_counter() - start)
                    rows = list(graph.connection.execute("SELECT * FROM raw_edges ORDER BY ordinal"))
                    graph_sha = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
                    if expected_hash is not None and graph_sha != expected_hash:
                        raise AssertionError("endpoint batching changed exact graph bytes/order")
                    expected_hash = graph_sha
                    key = "batched_membership_ms" if batched else "single_membership_ms"
                    samples[key].append(round(elapsed, 6))
                    query_counts[key].append(sum("FROM paged_neuron_records WHERE neuron_id" in sql
                                                 for sql in observed))
                finally:
                    graph.close()
                if hashlib.sha256(nodes.path.read_bytes()).hexdigest() != source_sha:
                    raise AssertionError("membership comparison changed its source database")
        medians = {key: statistics.median(values) for key, values in samples.items()}
        return {"scope": "paged recall graph endpoint SQL only; ReadSession enabled in both",
                "modelConstructed": False, "edges": len(edges), "knownNodes": 32,
                "pairs": 3, "pageEdges": 64, "maxMembershipIds": 128,
                "samples": samples, "medianMs": medians, "membershipQueries": query_counts,
                "componentSpeedup": round(medians["single_membership_ms"] /
                                           medians["batched_membership_ms"], 6),
                "sourceSha256BeforeAndAfter": source_sha, "exactOrderedGraphSha256": expected_hash,
                "wholeLoadMeasured": False, "processRssMeasured": False}


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
