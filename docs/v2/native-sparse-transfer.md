# Native sparse substrate transfer

Internal contract and evidence note, October 15, 2026. This describes the
current `brain.substrate` boundary for committed native sparse neurons,
assemblies and synapses. It is not a whole-cortex/model importer, foreign
foundation loader, arbitrary database editor, or proof of useful transfer.
No public manual or documentation curriculum is changed by this note.

## Supported scope

The exact scope marker is `native-sparse-assemblies`. Supported rows are:

- sparse neuron rows and their packed payload/update counters;
- assembly metadata and closed referenced child/neuron dependencies;
- sparse contacts, including ternary sign, strength, resistance, timing and
  repeated-contact state represented by the current native row format.

The format explicitly excludes dense cortex parameters, complete identity,
origin/evaluator replacement, candidate source changes, hidden prompt memory,
tool authority and an imported running process. Export manifests assert
`cortexStateIncluded: false` and `sourceAuthorityIncluded: false`. No operation
averages whole identities or silently converts foreign model weights.

`brain.substrate` is Ask by default and currently exposes `inspect`, `export`,
`register-bundle`, `read`, `prepare-import`, `prepare-rollback`, `apply`,
`status`, `list` and `cancel`. Inspection reads committed saved state and must
not call Load, instantiate a brain or inspect mutable live-only activity.

## Export and selected-task registration

`export` accepts an explicit all/neuron/assembly/synapse selection. Native
closure expansion includes required referenced sparse rows rather than emitting
dangling selected fragments. The app owns the output path, streams rows into an
immutable SQLite bundle, and records exact bytes, SHA-256, manifest bytes/hash,
source brain/generation, row count and `verified-owned-export` provenance.

`register-bundle` accepts only a main-owned attachment ID from the current task
turn. It copies the exact selected file through guarded FD/path identity checks,
rehashes every byte, asks the native validator to inspect the bundle, and then
rechecks the task selection. Because supplied bytes are external evidence, its
provenance is `selected-native-shape-source-unverified`; registration does not
assert that another brain generated it. Neither route executes repository setup
scripts or treats a caller-authored filesystem path as authority.

`read` returns bounded base64 pages with page and total hashes after complete
bundle verification. Its small verified-file cache is keyed by current owner,
bundle identity, SHA and filesystem identity; any path/FD change invalidates it.
This page interface is inspection/custody, not a neural update.

## Immutable preview: append and replace-exact

`prepare-import` requires a registered bundle and an explicit mode:

- `append` admits only additions whose identifiers/fingerprints do not collide
  with current committed sparse state. A collision is reported, not overwritten.
- `replace-exact` may replace selected existing rows only after recording their
  exact preimages. It does not mean replace the whole brain.

Preparation creates another immutable SQLite artifact; it does not mutate the
target. The plan binds target generation, bundle SHA, selected-preimage SHA,
changes SHA, additions/replacements/identical/conflict counts and every expected
before-image. Fingerprint, geometry, packed-byte/counter or selected-preimage
changes fail the compare-and-swap boundary. The target generation is audited,
but unrelated native changes are preserved and do not fail merely because the
whole generation hash advanced. `apply` requires the exact plan ID plus its
reviewed plan SHA and refuses any plan with unresolved conflicts.

## Detached apply and truthful state

An accepted apply returns a tracked job; `queued` does **not** mean applied. The
job enters the ordinary per-brain writer FIFO and waits for the same instance's
explicit loaded boundary. It retains a main-only origin authority in RAM. Main
rechecks owner, tool permission and runtime/package/capability revision before
FIFO acquisition and before invoking native apply. After dispatch, host
revocation drives the cancellation signal; native performs its own resource,
cancellation and selected-preimage guards during traversal and immediately before
atomic publication. Native does not invoke a fresh JavaScript permission callback
inside that publication boundary. No callback, approval token, runtime session
or live grant is stored in SQLite; only the audited authority revision is recorded.

Native apply revalidates the plan/preimages, performs the sparse row update and
returns one exact atomic receipt binding job/request/plan/bundle/preimages/
changes/generation. `weightsAveraged`, `cortexParametersChanged` and
`originChanged` must all be false. The host archives the previous and new exact
native heads. A matching retry may return a duplicate receipt; it must not apply
the same mutation again.

Stop before dispatch can be `cancelled`. If dispatch may have started without a
terminal atomic receipt, the durable state is `uncertain`, not a fabricated
rollback or safe retry. A native `cancelled-unapplied` boundary is the narrower
case that proves no commit occurred. Permission/runtime changes retire queued
authority and require review again. Restart, unload, copy and import never revive
the in-memory grant; queued/boundary work becomes dormant and applying work
becomes uncertain. `status` and paged `list` remain the tool-facing interfaces.
Studio now mounts owner-bound asynchronous job cards, paged plan
inspection, trusted exact-job Stop and chronological chat correlation. Its
subscription registers before taking an owner-validated initial page and
coalesces newer revisions across that fence. See
[sparse-jobs-ui.md](sparse-jobs-ui.md) for the precise UI/source evidence and
its synthetic-transport limitations.

## Reviewed inverse and its limit

`prepare-rollback` requires one complete applied job, its exact saved plan and
its authentic atomic receipt. It creates a new reviewed `replace-exact` plan;
rollback is never an implicit side effect of Stop.

The inverse can restore rows that the original operation **replaced**, because
their exact before-images are present in the original plan. It cannot yet prove
that rows introduced by append are safe to delete: current active references,
dependencies and later reuse are not fully proven. Those proposed removals are
reported as explicit conflicts/`unprovenRemovalRows`, so the inverse cannot be
applied until a future complete live-reference proof resolves them. This is a
narrow replacement rollback, not arbitrary undo or full-brain history reversal.

## Complete portability and lineage

Current custody lives under `substrate-transfers/`: a strict SQLite workspace,
immutable `bundles/` and `plans/`, plus isolated lineage archives. Copy/import
first snapshots and validates every current SQL row and artifact path/size/hash.
The untouched source workspace—including raw requests, artifact descriptors,
job payloads, history and archived heads—and exact bundle/plan files is retained
under:

`substrate-transfers/lineage/<originalBrainId>/<workspaceSha256>/`

The lineage manifest binds database owner, database bytes/hash and every artifact
path/size/hash. A new current owner receives only deterministic historical job
projections: source `applying` becomes `uncertain`; every other source job is
dormant. `originalBrainId`, plan/bundle hashes and original receipts remain
visible, while request rows, plans, heads and authority are not made current.
A new apply therefore requires a fresh current preview and review.

Full merge copies every pre-existing nested lineage archive first, byte for byte,
then archives the parent's current projection separately. Parent ID and workspace
hash isolate collisions; exact-identical paths deduplicate, while any differing
bytes fail. No first parent wins and no old plan is rewritten to look valid for
the merged brain. The exported lineage resolver is inspection/source custody,
not execution authority.

Root wired exact-byte capture/adoption into learning-workspace copy/import,
per-parent merge into full-merge assembly, and preserved-root recovery. One new
shared clone-hook fixture with an inert SQLite bundle passed in 165 ms; it did
not create or train a core/brain.

## Scoped evidence

These are reported focused runs, not a whole-suite or release claim:

- `tests/sparseSubstrateTransfer.test.ts`: the first 9/9 host cases passed in
  385 ms. One later `cancelled-unapplied` case passed in 91 ms with the previous
  nine skipped: 10 distinct Node cases total. They use tiny owned SQLite and a
  synthetic native bridge, not a real neural owner.
- `engine/tests/test_sparse_substrate_transfer.py`: an initial run exposed a
  genuine list-index versus paged-index dispatch error (4 passed, 3 errored).
  The affected three passed in 69 ms after repair. A stricter native metadata/
  fingerprint guard plus one new case later produced five affected passes in
  82 ms. Across those scoped runs there are eight distinct constructor-free
  cases; three untouched initial cases were not rerun.
- `tests/sparseSubstrateTransferPortability.test.ts`: three initial private-
  workspace cases passed in 269 ms. One later nested adoption-to-merge lineage
  case passed in 179 ms with the original three skipped.
- Root reports the separate shared clone-hook case above and coordinated Node,
  renderer and Python syntax checks passing. No CI/package run is inferred.

Python tests use `RawVectors`/snapshot fixtures and synthetic atomic publication
metadata. There is no actual `AdaptiveBrain`, decoder, Load, training, saved
production identity or quality evaluation in this evidence.

## Remaining acceptance

- Deferred by the current no-new-brain instruction: if later authorized, run a
  bounded actual loaded native owner through preview, reviewed detached apply,
  checkpoint/restart and inverse receipt verification. Its absence is an
  evidence boundary, not a blocker this documentation task should execute.
- Prove trained usefulness and retention without claiming whole-cortex transfer.
- Add complete live dependency/reference evidence before allowing deletion of
  rows introduced by append.
- Exercise the mounted job/progress/review presentation through the real
  preload/IPC/service path and an actual instance; current Electron observations
  use synthetic transport and perform no native effect.
- Qualify cancellation, recovery, portability and artifact downloads in packaged
  Windows/macOS/Linux builds and under meaningful resource pressure.
- Keep whole-instance merge and broader executable/plugin integration outside
  this narrow sparse-transfer claim. The explicit evaluator-backed native
  candidate proposal route is tracked in
  [native-candidate-tools.md](native-candidate-tools.md); source wiring is not
  actual native execution. No requirement is inferred for an arbitrary tensor
  or cortex editor.

Primary implementation is in `src/shared/sparseSubstrateTransfer.ts`,
`src/main/sparseSubstrateTransfer{Store,Service,Portability}.ts`,
`engine/omni_core/sparse_substrate_transfer.py`, and the supervised worker bridge.
