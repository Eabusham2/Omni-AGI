# Windows portable selected-instance workspace

Status: source implemented; focused host/file fixtures passed. This is R037's
portable-folder implementation, not a Windows runtime, neural continuity,
packaging, signing, release or hardware acceptance claim.

## Transfer contract

The existing Library flow can create a new Windows portable folder from a
selected **Unloaded**, ready native instance and a main-process-selected clean
Windows application directory. Trusted native dialogs select the runtime and
new output directory. Renderer requests contain the instance/request identity
and validated ordinary presentation settings, never arbitrary filesystem paths.

The result contains the selected EXE, its required Electron/resources/runtime
files, a hash-bound workspace marker and manifest, and the existing full
`private-archive` `.omni` export. It is not a weights-only export. The source
archive machinery carries the saved learned state, chat/history, working
continuity, immutable origin, pending learning progress, source custody and
supported host workspaces. The portable implementation reuses that archive
contract; the fixtures below do not independently prove every native archive
clause. The archive remains unsanitized private data, not an anonymized share.

Typed ordinary app settings copied by this path are appearance, workspace
arrangement/profiles, notifications, the exact selected theme dependency, and
tool approval timeout. Instance configuration and supported saved per-instance
workspaces follow the private archive. Raw browser LocalStorage/session state,
keychain data, device credentials, private reviewer settings, MCP authentication,
host-access leases and source-drive reservations are not copied as authority.
An installed theme package is a dependency, not an executable activation grant.

`resources/portable-runtime.json` is produced by the Windows `afterPack` hook.
It records actual desktop/worker PE architecture and the packaged ASAR hash.
The hook inspects the packaged main entry for the portable protocol and rejects
a stale compiled main. These checks are a source-presence and byte-integrity
receipt, not signed attestation or a claim that the application/worker ran.

## First start, relocation and destination authority

Only a packaged Windows executable with the fixed adjacent
`omni-portable.json` marker selects portable mode. Without that marker the
normal installer paths and behavior remain unchanged. There is no environment
override or directory-wide search. The marker is a small exact-FD checked
record; Windows filesystem identity uses bigint device/inode and nanosecond
timestamps rather than lossy numeric file identifiers.

Portable learned data and ordinary settings live under the folder's `data`
directory. Device-owned credentials, policy, host access and resource ledgers
live outside that transferable folder, bound to the workspace and canonical
destination on the current device. Copying or moving the folder therefore
requires current-destination preparation and resource review; it cannot inherit
the old device's live grants or physical-drive leases.

Startup reads marker/manifest/settings and the destination receipt without
importing, loading or opening the saved instance through cache-mutating
repository hydration. An explicit `prepare({requestId})` uses the existing
storage-operation import progress/Pause/Resume/Stop channels. Its private
main-owned read scope permits only the preparation path to hydrate saved data
while ordinary reads are held. Runtime admission/effects remain closed.

For a fresh folder the existing importer prepares the selected archive in an
isolated staging repository. With that empty target the original identity and
immutable origin must remain exact; an unexpected importer rekey is rejected.
Repository-wide content-addressed source blobs are preserved as well as the
selected instance directory. An already-used relocated folder keeps its newer
saved chats/state/progress. The preserved original cargo is a backup and is
never automatically reimported over that newer data. A completed adoption
receipt with missing/wrong selected data blocks visibly instead of reviving
stale cargo.

Existing per-workspace copy/adoption adapters retire copied pending execution
authority and keep its saved work/cursors. Learning tasks pause; schedules,
automation, MCP App processing, runtime plugins, external advice, executable
panel effects, attachment continuation grants and sparse-transfer work become
dormant or otherwise retain their existing explicit copied-state semantics.
No past completed external action is deliberately replayed. Mutable hardlinks
are rejected before retiring a copy, so another live source is not rewritten.
The native saved state is not repacked/retrained by this host authority step.
The retirement is checkpointed per workspace behind the closed admission gate,
not presented as one global multi-database filesystem transaction.

The current destination uses a separate host-policy store. Copied permission
projections cannot bootstrap Full grants through `get`, recovery, `save` or
permission snapshots. Existing current-device policy/defaults win. Destination
approval is a fit/resource approval, not a blanket tool, plugin, admin,
microphone or credential grant. Per-tool access remains separately controlled.

Resource review runs the existing planner against the actual saved context and
working-memory requirements, without hidden Load or silently shrinking them.
Approval rechecks the exact reviewed configuration and current fit, updates the
destination-local logical shared pool and leaves source-drive selections
discarded. It does not allocate or ship a padded 20 GB file. The normal 20 GiB
free-space floor/write admission remains in force.

## Publication, progress and recovery

Creation and preparation reuse `BrainStorageOperationSession` for real phases,
file/byte progress, Pause/Resume/Stop and disk-write admission. Runtime copying
and hashing use cancellable bounded buffers. The selected archive uses the
existing archive writer/cursor machinery; no new dataset ingestion occurs.
There is no premature 100% completion before publication.

Creation records request operands and intent before long copying/hashing. A
request ID with changed operands is rejected. Terminal failed/cancelled requests
are not silently re-executed. A complete lost-ACK request resolves to the same
verified publication instead of creating another folder.

Publishing reserves a new destination with exclusive `mkdir`, persists its exact
directory identity, and moves the runnable EXE or instance `brain.json` last.
An existing or replaced directory is never overwritten. Interrupted split
publication resumes only when the exact staged-plus-published tree matches its
prepared digest and retained owner; duplicated/changed files fail visibly.
Adoption reconciles this receipt **before** repository hydration can refresh
history caches. No second import is needed for a publication-before-ACK loss.
Late Stop after a durable completed adoption returns that actual state rather
than claiming the applied work did not happen.

Streaming reads reject symlinks and verify the exact opened FD/path identities
before and after the transfer. JSON metadata is admitted before decoding and
hash-checked. Mutable receipt writes fsync their file before rename. These are
application crash/retry boundaries, not a universal power-loss/hostile-OS
transaction guarantee. Cleanup targets only exact generated staging/output
identities; source runtime, source instance and retained cargo are not deleted.

## Stable host API and implementation

`window.omni.portable` exposes `create`, `prepare`, `status`,
`reviewDestination` and `approveDestination`. The renderer create request is
`{brainId, requestId, presentation}`; selected native paths stay main-owned.
Status distinguishes preparation-required, preparing, review-required, ready
and blocked. Review/approval bind the exact review ID/hash and current planner
result. Existing storage-operation subscriptions/control methods serve both
long operations.

Owned implementation:

- `src/shared/portableWorkspace.ts`
- `src/main/portableBootstrap.ts`
- `src/main/portableWorkspaceFiles.ts`
- `src/main/portableWorkspaceReadScope.ts`
- `src/main/portableWorkspaceAuthority.ts`
- `src/main/portableWorkspaceService.ts`
- Optional `BrainRepository` policy/read guard and archive snapshot guard;
  early main-process portable composition in `src/main/index.ts`.
- `scripts/portable-runtime-after-pack.cjs`,
  `scripts/portable-runtime-receipt.mjs`,
  `scripts/verify-portable-runtime.mjs` and the Windows smoke receipt hook.

Root-owned typed IPC/preload/UI, runtime admission guards and package hook are
integrated separately. This lane did not change the product version or publish
anything.

## Exact validation boundary

Fourteen distinct new focused host/file cases passed across scoped runs:
12 in `tests/windowsPortableWorkspace.test.ts`, plus 2 in
`tests/windowsPortablePackaging.test.ts`. They use tiny inert PE/ASAR files,
synthetic archive adapters, actual local SQLite/file operations, the actual
host-only repository policy/read code and existing host dormancy adapters.
They never construct a native brain or execute the packaged application.

After the final read/adoption/policy/publication changes, only the 9 affected
workspace cases were run: **9 passed, 3 unchanged skipped**, 708 ms test time,
2.23 s wall. These include two new cases for real host-only get/save read scope
and copied-Full rejection, and exclusive/split-tree publication. The previously
passed unchanged host dormancy/policy-snapshot/build-receipt and 2 packaging
cases were not rerun. Root owns coordinated final Node/Web compilation.

Still not proven by these fixtures: actual Windows x64/ARM64 package execution,
installer/ZIP/relocation across real drives, actual private neural archive
import/Load/resume on Windows, every saved-workspace cross-device continuation,
code signing/notarization, full packaged UI acceptance and release publication.
The runtime verifier explicitly reports `selectedInstanceTested:false`,
`destinationReservationTested:false` and `neuralAcceptanceTested:false`.
Root removed the separate release verifier's major-v1-only assumption without
weakening exact package/tag or immutable-tag binding. One new stable-v2 policy
case passed (7 ms). The app version remains1.1.1 at this source point; no tag,
branch, remote CI, package or asset was published by this work.
