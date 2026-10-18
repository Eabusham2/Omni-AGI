# Cold-Load versus first-recall graph reuse

Source-only investigation, October 11. No model or private native fixture was
loaded, initialized, trained or generated. No production source was changed
for this investigation; the completed endpoint-membership slice stays closed.

## What the current architecture actually does

The current Load path already avoids eagerly rebuilding the disposable recall
edge graph:

- `AdaptiveBrain._load_impl` restores the checkpoint and explicitly clears the
  connected training-only activation caches instead of running a startup
  training/feature forward. The inspected Load/constructor path does not invoke
  `memory.recall_vector` or `PagedRecallGraph.load`.
- `LazyPersistedSynapses` installs a generation-bound, checked compact forward
  topology. `iter_weighted_edge_records` traverses its bounded nonzero groups
  plus genuine mutable overlays; it does not reconstruct every cold zero-edge
  record as a Python object.
- `_recall_vector_paged` returns immediately when there is no positive match.
  The first positive recall creates `PagedRecallGraph` only if no compatible
  graph is already present. Later recalls use `refresh`, exact source revisions
  and mutation observers, applying notified edge and membership deltas rather
  than rebuilding the whole graph.
- `release_paged_recall_scratch` closes that graph at runtime shutdown, and its
  `TemporaryDirectory` is removed. A later process/owner therefore builds a new
  disposable graph on its first positive recall, not merely because Load
  reads the checkpoint.

These are inspected source contracts, not a new whole-Load trace or a measured
latency claim. The old retained stack observations do not establish that the
latest Load path still executes that same graph-building call.

## Why arbitrary persisted reuse is not safe yet

There is no committed neural-generation field that authenticates a completed
recall graph's exact signed/strength-weighted edges, order, missing-endpoint
eligibility and incoming normalization. A derived file claiming the right
generation plus a hash of itself is not such an anchor: modified edges and a
recomputed receipt could falsely change recall without changing the actual
brain checkpoint.

`AuthenticatedCacheSession` deliberately keeps its proof secret only in the
currently verified process. Persisting that secret or accepting a prior
process's self-authenticated rows would weaken the existing cold-recovery
contract. `PreparedPagedCache` builds a new private cache from committed blobs
and establishes fresh proofs rather than trusting an old working DB.

The existing complete endpoint index is useful but is not a substitute for a
ready recall graph. It authenticates endpoint-to-source-group membership,
including zero-valued synapses. Its rows do not themselves establish every
weighted contribution, eligible edge ordinal or fan-in denominator. Likewise,
a cache's clean revision marker establishes bookkeeping, not cryptographic
equivalence to learned source bytes.

## Disposition

No persisted-graph adoption was added. The requested eager-Load avoidance is
already present; cross-process first-recall reuse remains a separate potential
optimization, not a verified speedup or a blocker newly fixed here. The current
generation/resource/cancellation/rollback behavior was not weakened.

A future lazy source-backed graph would have to obtain complete incident
groups from freshly authenticated source proofs, preserve exact ordering and
all incoming normalization, handle live overlays and rollback, and retain
truthful graph-wide audit counts. Alternatively, persisted adoption would need
an independently trusted source-derived graph commitment and checked schema,
identity, content and publication rules. Merely attaching an old DB, retaining
its receipt, or rehashing its own bytes is insufficient. Neither alternative
is implemented or claimed by this source-only finding.

Relevant inspected sources:

- `engine/omni_core/brain.py`: `_load_impl` and connected startup-cache handling.
- `engine/omni_core/vsa.py`: `_recall_vector_paged`,
  `release_paged_recall_scratch`, `LazyPersistedSynapses` and
  `iter_effective_edge_records`.
- `engine/omni_core/paged_recall_graph.py`: `load`, `refresh`, `close`.
- `engine/omni_core/paged_recurrent_spreading.py`: graph attachment and complete
  incoming normalization during synchronous settling.
- `engine/omni_core/authenticated_paged_cache.py`: process-private proof key.
- `engine/omni_core/committed_paged_cache.py`: fresh checked cache construction.
- `engine/omni_core/paged_synapse_endpoints.py`: checked endpoint groups/root.

No additional benchmark or already-passed test was rerun for a documentation
finding. The earlier whole-Load/inherited-Worker latency gate remains open.
