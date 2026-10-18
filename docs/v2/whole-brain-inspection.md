# Whole-brain inspection source contract

Additive combined layers, aggregate zoom and exact membership source/evidence:
[whole-brain-multiresolution.md](whole-brain-multiresolution.md).

2026-10-15. Source/synthetic evidence only; no native brain construction, Load,
training, decoder execution or hardware operation was performed in this lane.
Root owns the unified primary canvas, desktop integration and visual evidence.
The ordinary public documentation corpus remains version 20 until root updates
it separately.

## One graph, distinct evidence

The adapter composes actual registered cortical owners, sparse neurons,
assemblies, contacts and regional bridges into one brain identity. Its source
facets are paged inspection inputs, not three independent models or primary
maps. Node kinds remain distinct. A tensor axis, assembly, semantic neuron and
LIF channel are not coalesced to make the picture appear connected.

Existing `OmniDecoder.forward` already feeds shared workspace/memory activity
through one hidden stream. Its learned expert residuals receive that same
stream. Connected decoders retain exact parent owners and exchange causal
activity through actual packed `RegionExchange` bridges. Sensory encoders feed
the same idea space. This work does not rewrite that neural architecture.
Neither those source paths nor this map prove biological overlap, human
equivalence or trained competence.

The new catalog walks the actual six trainable-role references and registered
`_modules`, groups only identical registered objects, and records authenticated
parent lineage from actual region identities. Object IDs are temporary SQLite
grouping keys only. Published IDs derive from stable aliases/lineage, never
process addresses. Multiple actual aliases/memberships retain one owner.
Containment and alias membership are explicitly **not** learned synapses or a
semantic-neuron-overlap claim.

Actual `RegionExchange` registration supplies source/target parent regions and
the packed bridge owner. Similar-looking module names do not establish edges.
A registered bridge does not claim observed nonzero weights or firing; its
selected tensor contacts remain separately inspectable. Exact weight targets
come only from the real save-role/tensor registry: `inspection` records bind
an owner to `role`, `moduleId`/`tensorKey` and `field`. No guessed path-prefix
rewrite or arbitrary first module is used for drill-down.

Existing cortical contact coordinates/signs/strengths and substrate synapses
remain typed learned connections. Assembly member IDs create actual membership
relationships; shared members remain the same sparse identities. Tokens are
not mapped one-to-one to neurons. A semantic-neuron ↔ cortical-axis ↔ router
channel mapping is not recorded, so the response reports it unavailable.

## Persistence and bounded traversal

`publish_whole_brain_catalog(brain, fresh_metadata, engine_path, *, admit,
reserve_disk, cancelled=None, tensor_registry=...)` is a save hook, after current
mutable/substrate/connected pointers are staged and before atomic `brain.json`.
Root passes the existing `core` and `plasticity` writer registries, not a new
state-dict copy. `InspectionScratch` enters and retains incremental RAM escrow
up to the maximum live scratch footprint; streamed pages do not accumulate
leases for every past page. The writer charges scratch SQLite/immutable writes
to the caller's disk admission and checks cancellation.

The constant-size `whole_brain_catalog` pointer binds source generations,
attention overlay and connected descriptors. It references a hash-addressed
chain of bounded JSON catalog shards plus an immutable SQLite location index.
The persistent index contains stable IDs/locations, not temporary object keys.
Current-owner roots/regions come first in indexed overview pages. Exact region
members, shared owners, owner details and exhaustive traversal use indexed
`overview`, `members`, `owner` and `all` facets. Endpoint anchors are actual
indexed records, never nodes fabricated from endpoint strings. Display page
sizes do not cap model or alias cardinality.

`iter_catalog_files(engine, metadata, *, admit, cancelled=None)` yields only
the exact current `(relativePath, bytes, sha256)` index and shard closure under
`whole-brain-catalog/index/` and `whole-brain-catalog/blobs/`; it never globs old
generations. Referenced hashes, sizes, FD/path identity and chain counts are
checked. Hash verification of unchanged immutable index files is cached by
exact physical identity in a small operational metadata cache. Immutable orphan
files after an interrupted uncommitted publication are not a neural commit.

New-ID copies retain original catalog bytes/IDs as provenance, not as a current
destination registry. The separate constant-size `whole_brain_catalog_lineage`
pointer names paged lineage entries and exact source closures. Native
`iter_catalog_lineage_files` mirrors the main-owned entryKey/entry hash schema
and walks those exact files. It never promotes historical lineage to current
ownership. A new current registry requires a genuine destination-owner save.
Same-ID snapshot copies preserve the current pointer. Main portability wiring
and its distinct filesystem evidence are owned by the portability lane.

## Query, regional sources and projection

`query_whole_brain(engine, brain_id, query, *, admit, cancelled=None, ...)
` never constructs or Loads a brain. Public DTOs live in
`src/shared/wholeBrainInspection.ts`; main validates source data in
`wholeBrainInspectionProtocol.ts`. Queries accept typed facets/region/owner IDs,
not paths, raw metadata or renderer topology. Root exposes
`window.omni.wholeBrainInspection.query`.

The snapshot fence includes current metadata SHA, core/substrate generations,
attention overlay, connected descriptors and catalog hash. Source metadata is
checked before and after delegated pages. Process-local authenticated cursors
bind that exact snapshot and request, and cannot be edited to read an unrelated
old shard. Restart or changed source requires refresh, not silent cross-snapshot
joining. Historical checkpoints without a catalog report missing registry;
inspection does not auto-create one.

Regional sparse pages resolve exact catalog region IDs through checksum-bound
`connected_graph_states`. Only an internal `VerifiedCommittedSubstrateSource`
can supply the selected store's committed pointer/attention metadata to the
existing substrate reader. Public normal queries retain their default path;
RPC JSON cannot supply that descriptor. Region paths, actual source parent IDs
and current root metadata are fenced around the delegated read. Regional
cortical owners remain in the root joint tensor inventory and use their exact
inspection bindings rather than inventing independent cortical files.

`projectWholeBrainInspection` in `src/shared/wholeBrainProjection.ts` composes
typed pages, exact identities and real anchors. Contradictory duplicate record,
node or edge IDs under the same snapshot throw instead of silently replacing
data. Different snapshots/brain owners cannot be combined.

Live LIF capture remains an independently labeled capture/epoch scope. Actual
hard-spike frames may mark a LIF channel as observed firing; stored contact
weights do not mark synapses or postsynaptic neurons as firing. Live router
channels are not silently joined to committed semantic/cortical units. Existing
selected cortical live observers similarly need a genuine owner/time join before
the primary graph can claim matching all-region live activity.

## Exact focused evidence

`engine/tests/test_whole_brain_inspection.py`: **11 distinct constructor-free
cases passed across scoped runs**. Initial run: 7 passed, one test expectation
failed because tampered index size/identity was rejected before its expected
JSON checksum error. That affected expectation plus two new lineage/actual
worker-save-hook cases passed **3/3 in 0.462 s**; the previous seven were not
rerun. One new SQLite acquisition-failure cleanup case passed **1/1 in 0.017 s**.
Fixtures use inert registered objects and tiny actual JSON/SQLite files, not
neural checkpoints. Coverage includes stable shared aliases, actual registered
bridges, exhaustive shard pages, authenticated/stale cursors, missing legacy
registry, exact regional source fences, real endpoint anchors, high-water escrow,
lineage-only copied custody and loaded-free production hook delegation.

`tests/wholeBrainInspection.test.ts`: **7/7, 21 ms** pure protocol/projector
fixtures. Covers shared owners vs structural edges, exact saved field targets,
actual sparse kinds/assembly overlap, incompatible snapshots, contradictory
same-ID data, independent observed LIF scope and source-authority rejection.

These are mechanism/source checks, not actual loaded-model catalog publication,
all-region activity, trained connectivity/recall quality, packaged platform/UI
acceptance, hardware throughput or neural architecture qualification. Root owns
coordinated compiler and UI validation; no broad old suites were run here.

## Bounded endpoint and UI-fixture follow-up

The whole-brain adapter now resolves returned sparse contact/assembly endpoints
through the same verified committed view and authenticated paged query index.
It admits exact payload lengths before decoding, uses bounded string sizing
instead of encoding a whole duplicate page before admission, preserves the
source revision/fence and refuses a missing actual endpoint. Ordinary substrate
reader behavior is unchanged. Added child-assembly anchors keep their other
relationships paged; no recursive whole-graph expansion or placeholder neuron
is invented.

The actual VSA source stores `assembly_neuron_id` and requires an assembly's
vector to alias its neuron row. The adapter reports this explicit source alias;
the graph preserves distinct typed assembly/neuron records, connected by a
structural alias rather than another learned contact. Verified connected-store
scope or an actual whole-brain root anchor supplies structural sparse ownership.
A registered-role selection cannot assign the root sparse store to that role.
Tensor axes likewise receive structural ownership from their exact selected
matrix, not a fabricated synapse. Explicit delegated facet/module requests are
checked before the main process returns them to the view.

Only the new affected cases were run for this follow-up: native endpoint case
**1/1 in 0.004 s** and pure protocol/projector case **1/1 in 6 ms**, with seven
older TS cases skipped. Earlier checks were not rerun. The engine package may
import Torch transitively; the tests construct no tensor/model/native brain and
execute no decoder or learning. Four owned TS/TSX entries syntax-transpiled and
the adapter AST parsed; these are not a coordinated typecheck.

`tests/e2e/whole-brain-fixture.{html,tsx}` mounts production `WholeBrainMap` with
an explicitly synthetic inert API, shared-owner/bridge/weight pages, overlapping
sparse assemblies, exact backing-row aliases, second pages, stale/wrong-owner
controls, theme/compact controls and visible query receipts. It respects the
ordinary reader's coarse-versus-detailed zoom rule instead of secretly supplying
detail when zoom is omitted. This lane did not launch Electron or claim actual
neural/runtime effects. Root owns visual execution and renderer corrections.

The generic Parameters → selected matrix path now supplies its actual indexed
inspection binding, owner, membership records and region endpoints without a
prior Owner-details visit. The join uses the exact saved role/tensor key; no
module-name prefix inference is used. Membership anchors are viewport-bounded;
additional memberships remain explicitly available through paged Owner details.
One new constructor-free adapter case passed **1/1 in 0.027 s** for both complete
and bounded membership variants. No other tests or neural work were run for it.
