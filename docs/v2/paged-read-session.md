# Native read traversal IO

## Observed issue

The inherited execution fixture required 1,023.892 seconds to load and capture
one new token frontier. A later bounded Worker attempt stopped safely at
113.192 seconds during its first explicit Load. Neither result establishes
acceptable latency. One live native stack sample showed recurring SQLite
connect/PRAGMA/schema parsing. Cold recall's actual `graph.load` validates both
endpoints of every edge through paged neuron metadata; the previous path opened
and configured a fresh connection for each lookup. Native vector-export
generation checks had the same repeated connection setup pattern.

## Narrow implementation

`paged_read_session.py` retains admitted, operation-local connections only for
declared native read stores. SQLite uses `query_only`, a write/control-denying
authorizer, a 256 KiB cache and no mmap; the retained connection envelope is
320 KiB, not a claim about total process RSS. App `ResourcePolicy` holds an
actual RAM lease until close. Automatic retention skips boolean-only custom
row-budget callbacks because those cannot escrow a retained cache; their
original unkept read path remains functional.

Existing transactions remain short per-operation read savepoints. This is not
one keeper-long frozen snapshot that could mask a later committed revision.
Existing generation checks remain, file replacement invalidates the keeper,
and cancellation is checked at every borrow/read. The original connected
learning transaction always takes precedence and retains its real write
savepoints and rollback journals. Read reuse cannot silently turn learning
into a read transaction or discard primary writes.

Adoption is limited to these affected paths:

- Native `PagedRecallGraph.load` receives explicit source-node stores and
  verifies their generation before and after endpoint validation.
- `recall_vector(record_activity=False)` retains the declared paged stores
  during ordinary read-only recall. Mutable activity recording stays outside
  this read-only wrapper.
- Native vector-export traversals retain their exact native row order,
  signs, strength/resistance bytes and repeated-contact ordinals. Existing
  page admission occurs before fetching BLOBs; a refused page retries the
  same cursor at smaller size, down to one row without skipping records.
  The read ContextVar resets BEFORE a generator yields to its consumer, so
  later genuine mutations still follow their actual writer and are caught
  by the original next-generation check.

This does not add a prompt, answer lookup, foreign decoder, floating learned
master, capped idea set, or corpus-sized RAM mirror. It changes IO ownership,
not neural equations or trained routing.

## Focused evidence

- Initial three cases passed in **0.519 s**: actual graph endpoint/edge-order
  parity; stale generation, cancellation, replacement and write control; and
  primary connected-write precedence with real rollback.
- Authorizer/RAM-lease follow-up cases passed **0.079 s**, including release
  after an exception and refusal to disable read authority through a SQL
  comment.
- Actual ordinary native paged recall plus native packed export parity passed
  **3.893 s** before default wiring: bit-exact output vector, recalled rows,
  measured audit and inference manifest. No brain/model was constructed.
- After wiring, affected recall/cancellation, native export yield-scope/stale
  mutation, and low-buffer exact native page traversal passed **3 cases /
  2.318 s**. These are targeted affected checks, not broad rerun suites.
- The explicit actual graph-load entry rejected an independently changed
  source-node generation before publication (**1 case / 0.029 s**).
- One bounded paired actual graph-load comparison used 384 edges, 32 known
  source nodes and three alternating-order pairs. Median original was
  **2,949.680 ms**, keeper **264.254 ms** (**11.162x** for this component).
  Ordered graph hashes matched exactly; one keeper connection served each
  traversal. Raw measurements are in
  [paged-read-session-comparison.json](paged-read-session-comparison.json).

The comparison is CPU/SQLite IO only. It is not a trained large-model, whole
Load, tokens-per-second, Metal/CUDA or whole-process-memory benchmark. The
17-minute capture used the earlier exporter import and must NOT be relabeled
as an optimized result. No new model run or initial training was performed
after the bounded Worker failure; integrated latency/continuation qualification
remains open until explicitly authorized and observed.

The next distinct endpoint-SQL component now has bounded membership pages on
top of this keeper, documented in [paged-recall-membership.md](paged-recall-membership.md).
Its comparison enables the keeper in both variants and does not replace or
rerun the earlier keeper comparison or any native model fixture.
