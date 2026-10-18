# Actual multi-drive ephemeral spill

## Data path

The native packed-core, mutable router/control and working-attention pagers now
place real backing files in selected drive pools. This includes saved-activation restoration
images. Placement chooses the highest measured write rate that has capacity,
then tries the next selected pool. Unknown rates do not masquerade as measured
speeds. Only generated session directories and generated filenames are used.

The existing single-drive path is retained when no additional directories are
selected. Authoritative checkpoints, snapshots, origins and journals stay in
their existing instance locations. Native export streams the actual mapped
tensor contents; it does not export private backing paths as portable state.

## Authority

`catalog.selectSpillPools(brainId?)` opens the native directory picker. Main
observes the volumes, creates private `.omni-spill-*` backing roots, and records
root/backing device-and-inode identities in the installation-local shared
ledger. Existing-instance picks grant that instance only. Build picks return
`CreateBrainRequest.spillPoolSelectionId`, a one-use token claimed for the new
instance before worker creation. An installation-local claim receipt makes an
exact same-token/same-owner/same-directory retry idempotent; it cannot grant a
different owner or restore revoked access. The token is not a `BrainConfig` field.

`config.deviceResources.storageDirectories` remains a preference list, not
filesystem authority. Imported bundles, copied configuration, source files and
model output cannot create local drive grants. On another installation, the
user must select destination directories. The host `shared-resources.sqlite3`
grant tables are not part of instance exports.

Grant identities are rechecked before allocation and cleanup. Replaced,
symlinked, unverified or detached roots cannot silently redirect writes. A
changed physical root may require choosing a new empty folder; old backing
data is not overwritten to repair that choice.

## Quotas and file lifetime

One SQLite transaction admits each spill against both:

1. The largest configured shared pool across instances, not the sum of their
   reservations.
2. The largest selected allowance for the verified physical drive group,
   without adding several directories in the same group.

Fresh drive-free space must also remain above its reserve after pending writes
and unmaterialized sparse-file promises. Selected percentages remain fixed
through transient occupancy. Automatic increases require both the per-drive
and storage-wide increase choices. A manual percentage change is a new explicit
selection, not an automatic escalation.

Accounting distinguishes charged bytes, last-observed allocated bytes,
unmaterialized promises, and pending allocations. Hardlink aliases retain one
physical charge and preserve the original drive group. Native mapped pages
remain charged after unlink while exported tensor views still hold them.
Saved activation images remain valid after their pager closes.
Disabling overflow also disables the old default-directory fallback. CPU saved
activations can instead restore in admitted RAM; insufficient RAM pauses rather
than secretly creating a disk image.

Failed staging, mmap creation, cancellation and short writes unwind the exact
pending file/lease. At first placement in a new runtime manager, bounded batches
reclaim only registered generated backings from demonstrably dead jobs. Live
aliases, reused live PIDs, changed identities, unknown files and OS-held files
are preserved. A detached drive is not evidence that its files were deleted;
those charges remain until availability and removal can be verified.

## Verification

- 15 two-volume fixture cases write and read real tiny native/router/attention files,
  restore saved activation images, stream portable safetensors, enforce shared
  quotas/reserves, reject ungranted paths, and exercise failure/cancel/dead-job/
  detached-drive cleanup, disabled spill and RAM-only restore. Volumes and capacities are injected; total writes are
  small and no user brain is created.
- 10 existing shared-quota cases and 42 existing native-core/attention paging
  cases pass, covering the retained single-drive path.
- 8 main authority cases cover staged/idempotent grants, existing-owner grants,
  canonical root identity, imported-path rejection, changed backing roots,
  pinned physical-group quota snapshots and Python/TypeScript signatures.
  14 existing main registry cases pass.
- TypeScript checking and changed Python compilation pass. No CI or package
  build was run for this slice.

## Deliberate boundaries

Build and Device settings expose the picker and reviewed per-device controls.
The context planner now consumes authorized physical-group quotas and the
global largest-pool limit, with separate live pressure. Primary checkpoints,
training scratch and recovery/growth space remain separately reserved. The
working pager no longer clamps these selected-drive allowances to the primary
volume's free bytes; every actual allocation still passes transactional quota
and physical-reserve checks. RAM-first Auto remains distinct from manual max.

Additional persistent checkpoint placement, multi-GPU scheduling, and physical
multi-drive/cross-platform performance qualification remain separate work.
Synthetic volumes prove placement/accounting logic, not hardware throughput.
