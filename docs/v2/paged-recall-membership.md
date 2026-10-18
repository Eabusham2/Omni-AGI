# Bounded cold-recall endpoint membership

## Next observed SQL component

The retained cold-Load stack evidence described in [paged-read-session.md](paged-read-session.md)
identified actual `PagedRecallGraph.load` endpoint validation. The completed
read keeper removed connection/PRAGMA/schema setup per lookup. Source inspection
then found the remaining one-query-per-endpoint path: `_replace` called the
paged neuron's single-ID `__contains__` up to twice for each nonzero edge, even
when a small group of source nodes was reused. This is an identified component,
not a claim that it alone explains all whole-Load latency.

## Implementation

- `PagedNeuronMetadata.contains_many` checks at most 128 requested structural
  IDs in one admitted read transaction. It checks the exact expected structural
  revision in the same SQLite snapshot as the membership query. It returns
  only a frozen set of present IDs, never learned vectors, labels, answers or
  a retained corpus-sized mirror.
- `PagedRecallGraph.load` optionally retains at most 64 edge references for a
  resource-admitted window, obtains their exact membership once, and applies
  them in their original order. It verifies the source revision before and
  after query/application. These are transaction/page windows, not neuron,
  synapse, idea or recall caps; traversal continues through the full input.
- The real paged recall entry uses this path only for its actual
  `PagedNeuronMetadata` source. Other/custom sources keep the original
  single-ID callback. Existing `ReadSession`, per-edge resource checks,
  cancellation, missing-endpoint behavior, weighted signs, repeated edge
  handling and mutation observer/revision publication remain in place.
- A primary connected learning transaction still owns the real connection and
  savepoints. The read keeper cannot hide its writes or convert it into a
  read-only transaction. No brain, worker, persistence, neural equation or
  runtime-seal requirement was replaced.

## Focused evidence

Six new constructor-free pager/graph cases passed in separate affected runs:

- Five cases, **0.228 s**: exact ordered graph/incoming-degree/sign/absence/
  duplicate/removal parity; source-file hash unchanged; SQL reduction and
  bounded pages; changed revision rejected before publication; cancellation;
  RAM refusal before SQL; invalid callback IDs and page-resource refusal.
- One additional real connected-writer/savepoint/rollback case, **0.029 s**:
  the bulk read sees the uncommitted owner mutation, opens no read keeper,
  and leaves no row after the primary transaction rolls back.

A short paired comparison used 384 edges, 32 known source nodes and three
alternating-order pairs. **Both variants already use the completed read
keeper**; this is not a repeat of the keeper-versus-open comparison. Membership
queries fell from **757 to 6**. Median graph-load component time was
**283.750 ms to 16.167 ms (17.551x)**. The ordered graph SHA and complete source
database SHA matched before/after every sample. Raw measurements are in
[paged-recall-membership-comparison.json](paged-recall-membership-comparison.json),
with a bounded reproducible helper in
`engine/tests/compare_paged_recall_membership.py`.

The benchmark uses no model, generation, learning, external effects or private
native fixture. Its no-op graph resource callback deliberately isolates SQL;
the production per-edge resource sampler is unchanged and may dominate real
Load. This is **not** whole-Load, TPS, process-RSS, Metal/CUDA or trained-quality
evidence, and ratios from separate component benchmarks must not be multiplied
into a claimed whole-Load speedup. No fresh current-seal Worker continuation was
run, and the earlier 17-minute capture and 113-second cancelled Load remain
failed/open latency evidence. All retained native fixtures/references are
untouched.
