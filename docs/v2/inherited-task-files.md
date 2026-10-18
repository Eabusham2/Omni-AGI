# Reviewed files for inherited tasks

This is a distinct extension of [warm task-file scopes](task-continuation-attachments.md). It adds a current-main review and retired-custody reader without changing retired attachments to bound, rewriting accepted audio/input hashes, copying live permissions, or replaying completed effects.

## Exact source and review

An inherited task can have initial bound files without any warm-Steer scope. Its source set is therefore derived from the **sealed original parent's** portable workspace, not from a whole-history query or only the destination's warm-scope tables.

The actual main-native presentation is authenticated before review. A private WeakMap keeps that source object separate from renderer-shaped data. The source context binds the current registered root execution/turn, exact native frontier and input SHA, source brain/turn, source receipt epoch, destination epoch, registered region path, source manifest and native binding.

`inheritedTaskFileClosure` follows registered parent regions through their sealed parent indexes. Each generation, source metadata and complete portable inventory must agree. It verifies the original attachment-registry blob, selects only that original task's bound files and its checked warm inherited references, verifies original file bytes, and matches the deterministic destination retired aliases. Nested merges compose the original alias chain; direct-parent unrelated history and unsent selections are excluded. Metadata allocations are admitted against physical/headroom limits rather than silently truncating the task set.

The public typed API is on `brain`:

- `prepareInheritedTaskFiles({brainId, executionId})` returns `InheritedTaskFileReview | null`. It performs no Load, model generation, learning or permission change. Preparation does not write review/grant rows; pending review state is process-private.
- `approveInheritedTaskFiles({brainId, reviewId, reviewSha256})` is an explicit trusted-user action. It rechecks the actual native context, closure, exact disclosed files, epochs and current `task.attachments` policy revision, then atomically saves an immutable destination-local adoption receipt. An exact lost approval ACK recovers the same receipt.
- `revokeInheritedTaskFiles({brainId, reviewId, reviewSha256})` dismisses a pending review or revokes the exact saved approval and its active lease.

The disclosure contains safe file names, types, lengths, hashes and original binding identities, not host paths or raw private source JSON. Approval covers those exact files for that exact inherited task, not parent credentials, source tools, camera/microphone access or a broader history grant. A changed checkpoint/source/input/file set/epoch/policy needs a new review. Existing matching approval is the only approval bypass.

The October17 user decision allows explicit review of the exact saved copies
even when `task.attachments = Off`. The approval applies only to this inherited
task and its disclosed bytes. Declining review leaves it paused. No general
attachment policy, `system.files` setting or original computer-path access is
enabled by that review.

## Dedicated read boundary

The ordinary base attachment binder/reader still refuses retired files. `InheritedTaskFiles` instead receives an authenticated main-only resume context and an exact current approval, then creates a process-private task lease.

Reads use the existing exact-FD SHA verifier over the current owner's content-addressed retired bytes. Before and after awaited work they check:

- The same current main inherited task tuple, signal, lifecycle and input SHA.
- Destination attention epoch and current file-permission revision.
- The exact still-approved adoption receipt.
- Immutable original closure identities and original blob guard.
- Exact current retired registry row/manifest/private-source SHA and owned paths.

These checks do **not** issue a native worker RPC while the decoder is awaiting
a tool result. ToolExecutor rechecks ordinary policy plus the current exact
saved-file review before dispatch. When general attachment policy is Off,
only `task.attachments` list/read for the reviewed active task receives that
scoped exception; neither a model-supplied turn/review token nor a host-file
action can use it. Its authority revision includes both the current policy and
the exact adoption receipt. Revocation, Fresh, changed policy and ended task
retire access. The runtime schema discloses only the exact retained attachment
IDs; it does not enable the tool installation-wide. An explicit saved-file
review is not a microphone, execution or computer-files grant.

`InheritedTaskFileReference` explicitly says `custody: "retired"`; it does not invent a new bound turn. Read results retain the original binding separately. Safe message projection preserves the original accepted human turn/input SHA while attaching destination-owned file IDs. Original native audio source/cursor/ledger identities remain unchanged.

Trusted-user historical file-chip lookup can read renderer-safe retired **metadata only**. It does not expose raw paths or bytes, activate an execution, bind a file, or become an AI tool. This allows receipts/history to display correctly without weakening retired file access.

## Portability

Review records and immutable receipts are stored in prefixed tables in the existing attachment SQLite registry. Validation covers their hashes, owner, review fingerprint and receipt. Copy, import, same-owner recovery and full merge retain audit records as **dormant**. No process-private execution lease or approval authority is copied. Full merge retains distinct source review lineage without replaying it; destination review is separate.

These are unsanitized operational records, including file associations. A future privacy-cleaned copy must inspect these records together with attachment/scope custody; it must not assume that removing one live registry row anonymizes all historical associations.

## Focused evidence and limits

On 2026-10-13, `tests/inheritedTaskFiles.test.ts` passed **13 distinct new host cases** across three targeted runs:

- First ten: 1.918 seconds of tests, 2.83 seconds total.
- Two additional nested-merge and metadata-only cases: 485 milliseconds, 1.29 seconds total; the previous ten were skipped.
- One additional production main-owner rejection hook: 174 milliseconds, 1.85 seconds total; the previous twelve were skipped.

The tests use actual temporary files, sealed workspace inventories and SQLite registries with explicitly **synthetic native-context callbacks**. They cover initial/warm/nested exact file sets, readonly prepare/no grant, original bytes and unchanged retired bindings, current ownership/policy/epoch, lost ACK, revoke during awaited reads, corruption, original-turn projection, dormant portability and metadata-only historical lookup. The earlier warm-scope twenty cases were not rerun.

Final shared Node typecheck completed successfully after the Steer fixture typing fixes; this lane reused that result rather than starting a duplicate compiler. Its final Web typecheck and whitespace check also passed. None of those checks exercises a live neural decoder or constitutes CI/package acceptance.

Not performed: real neural inherited-cursor/file-tool end-to-end use, model loading/training/generation, device capture, provider calls, packaged-platform acceptance, CI, commit or release. The coordinator owns the App review panel and its actual UI check. Source/host evidence does not claim that native/model or UI acceptance is complete.

October17 policy extension adds two distinct real temporary-files/SQLite host
cases. One verifies exact reviewed reads while Off and unchanged global policy;
one passes through real ToolExecutor policy/intent/dispatch fences, rejects
model argument tokens and host-file access, and checks revocation. They passed
in160ms and206ms respectively across selected runs, without rerunning the13
closed cases. Two affected static review-render cases passed34ms. The first
ToolExecutor fixture attempt failed because its synthetic directory basename
did not match the declared owner; the fixture was corrected to the production
owner invariant, with no production intent guard weakened. These are not a
live native decoder or screenshot-based UI proof.
