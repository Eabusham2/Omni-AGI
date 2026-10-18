# Combined whole-brain viewport and multiresolution traversal

This source slice implements the latest verbatim request in AMENDMENTS.md:
parameters, ideas, neurons and connections belong together in one primary map.
It does not claim biological anatomy or trained understanding, or equate a
tensor axis with a semantic neuron.

## Concurrent windows and custody

WholeBrainMap retains independently bounded current parameter inventory/detail,
sparse aggregates, ideas, sparse detail and exact-member windows. Paging changes
only its window; every source remains traversable. Sizes are transport/RAM
windows, not model cardinality ceilings. Refresh reads catalog, actual packed
modules, sparse aggregates and a small idea page sequentially: concurrent reads
from one sender would supersede one another under the existing cancellation
contract. It installs the complete new viewport only after matching the whole
owner/snapshot and validating the combined projection. Failure keeps the old
consistent graph. Instance change/dispose/supersession fences late results.
The React checklist informed this lifecycle and bounded-state composition.

Unified layer controls show parameters/owners, ideas/neurons and connections
together. Actual matrix records show logical-parameter counts separately from
neuron and idea counts. They do not expand every weight or imply these counts
are interchangeable. Exact parameter row/contact drill-down and Next/Previous
remain available. Source structure is distinct from individual learned contacts.

## Actual aggregate zoom and pathways

Source detail exposes region totals, activation groups and exact neuron pages.
The authenticated index supplies real region totals and off-page region anchors.
Aggregate pathways connect those actual groups with contact count and complete
negative/zero/positive sign counts, not a representative sign or weight. If a
real group endpoint is absent, an explicitly unavailable pathway summary is
shown instead of a fabricated endpoint. Means/active counts are committed state
summaries, not hard-spike events. Live router captures remain independently
fenced; semantic/cortical/router identities are not silently joined.

Pathway drill-down filters the exact ordered sourceRegion/targetRegion pair,
not either endpoint broadly. Main validates both the receipt and actual returned
endpoint regions. Activation-band drill-down uses the explicit actual
activationBand field, never guesses from labels; returned effective activations
must match the selected band.

## Real assembly member cursor

Previously relationshipsPaged could omit an oversized list without a real exact
member cursor. membersOf now traverses the original neuron and child-assembly
arrays. Its receipt binds parent, total, global offset, member kind/ID and the
original per-array ordinal. Two indexed ordinal seeks avoid walking all prior
members on each page. The whole-map HMAC cursor authenticates owner, snapshot
and query; changed parent/filter/generation refuses continuation. Actual typed
endpoints/backing-row aliases accompany the window. Missing endpoints refuse,
never become placeholder neurons.

Projection separates immutable assembly identity metadata from partial member
windows, retaining exact ordinal relationships without duplicating learned
identities or treating two partial arrays as contradictory full state.

## Derived streaming index and resources

PagedInspectionIndex's derived cache is version 4; saved neural/catalog/lineage
schemas are unchanged. Source FDs are streamed through SHA verification, parsed
incrementally and identity-checked afterward. Assembly member arrays go directly
to SQLite, never a whole Python JSON list. Arrayless headers retain full counts.
Source ids/vectorIds and native packed-vector IDs are checked without whole-ID
list materialization. Existing optional explicit backing IDs/shared storage keep
their original meaning; absent provenance is not fabricated.

The caller's entered high-water escrow admits held root metadata plus
source/header/window scratch. SQL uses a small cache and disk-backed groups;
disk admission/cancellation runs around streaming/SQL/publication. Free-space
inspection uses portable disk_usage rather than statvfs. Page reads recheck
authenticated index custody/file identity; failed or cancelled upgrades do not
publish partial cache descriptors.

These are conservative scratch admissions, not an OS-enforced whole-process
RSS limit. Source generation manifests still need admitted metadata RAM.
The first process-authenticated index build traverses the selected generation
and can take significant time. Synapse tensors retain their existing admitted
complete-shard decoding path. Large-state throughput, first-index latency,
all-region live joins and all-platform/package behavior remain unproved here.

## Exact bounded evidence

New engine/tests/test_inspection_multiresolution.py has six distinct
constructor-free JSON/SQLite cases. Four passed across initial/affected runs:
actual groups/sign counts and directed contacts; real member cursor/endpoints
and snapshot/filter fence; cancellation cleanup; and 5,001-member streaming.
The last guard initially had a test-only bytes/string spy error; corrected,
it passed alone in 0.161 s. A native packed-ID/exact-band case passed alone,
then its explicit-band variant plus a new changed-index-custody case passed
2/2 in 0.177 s. No tensor/model/native brain was constructed; package imports
may import Torch/class definitions, which is not neural execution evidence.

New tests/wholeBrainMultiresolution.test.tsx passed four pure combined-view,
aggregate/member, serial hydration and owner/generation cases in 22 ms. One
additional exact band/ordered-pathway receipt case passed alone in 3 ms, with
the first four skipped. Renaming this renderer-facing file from .ts to .tsx
corrected its Node/no-JSX compilation placement, not product behavior.

The inert production-map fixture now has concurrent parameter/idea/aggregate
layers, source-band/pathway controls and three exact one-record member pages.
Root owns visual inspection; this lane did not launch Electron or claim the
new flows visually passed. Syntax/scoped whitespace checks passed. Coordinated
compiler and large-state/all-region/platform/package acceptance stay separate.
Public corpus pack 20 is unchanged. No model Load/creation/training, provider,
hardware bench, build, CI, Git mutation or old broad suite was performed.
