"""Small paired CPU cue comparison, not brain throughput/quality training."""
import hashlib
import json
import statistics
import time

import torch

from engine.tests.test_streamed_vsa_activity import reference_cue, semantic_id
from omni_core.streamed_vsa_activity import activity_block_rows, activity_reduction_scratch_bytes
from omni_core.vsa import NeuralSubstrate


def main():
    torch.set_num_threads(1)
    results = []
    for dimensions, count in ((64, 32), (64, 1025), (256, 4097)):
        memory = NeuralSubstrate(dimensions, seed=947)
        for index in range(32):
            label = "learned-%d" % index
            memory.neuron_vectors[semantic_id(label)] = memory.space.symbol("source:" + label)
        labels = ["learned-%d" % (index % 32) for index in range(count)]
        digest = lambda: hashlib.sha256(b"".join(memory.neuron_vectors.row_payload(key) for key in memory.neuron_vectors)).hexdigest()
        before = digest()
        timings = {"reference": [], "streamed": []}
        expected = actual = None
        for _ in range(3):
            for name, operation in (("reference", lambda: reference_cue(memory, labels)), ("streamed", lambda: memory.vector_for_labels(labels))):
                started = time.perf_counter()
                value = operation()
                timings[name].append((time.perf_counter() - started) * 1000)
                if name == "reference":
                    expected = value
                else:
                    actual = value
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
        assert digest() == before
        baseline = statistics.median(timings["reference"])
        streamed = statistics.median(timings["streamed"])
        results.append({"dimensions": dimensions, "labelsVisited": count, "blockRows": activity_block_rows(dimensions),
                        "referenceMedianMs": round(baseline, 6), "streamedMedianMs": round(streamed, 6),
                        "throughputRatio": round(baseline / streamed, 6),
                        "maxAbsoluteCueDifference": float((actual - expected).abs().max()),
                        "bitIdenticalCue": bool(torch.equal(actual, expected)),
                        "referenceRetainedAndStackTensorBytes": 2 * count * dimensions * 4,
                        "streamedReductionScratchEstimateBytes": activity_reduction_scratch_bytes(dimensions, count),
                        "packedStateUnchanged": True})
    print(json.dumps({"seed": 947, "backend": "cpu", "threads": 1, "pairsPerProfile": 3,
                      "scope": "VSA cue composition only; calculated tensor scratch is not measured RSS or whole-brain speed", "results": results}, indent=2))


if __name__ == "__main__":
    main()
