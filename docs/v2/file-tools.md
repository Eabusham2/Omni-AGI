# Bounded file reading, search, and task handoff

Internal source contract and focused evidence, October 15, 2026. This implements
the file-tool portion of R053/R163, including the request to “search tool for file
or read file all or segment or use imported file in cmd terminal.” It is not a
model-training, native-learning, packaged-platform, UI, or performance result.

## Source and integration seams

The implementation is in `src/main/fileToolService.ts`, with typed results in
`src/shared/fileTools.ts`, source-custody helpers in
`src/main/fileToolHandoffCustody.ts`, declared teaching in
`src/main/fileCapabilityPack.ts`, and the `system.files` catalog plus narrow
`ToolExecutor` dispatch/cancellation hooks. `windows.files` uses the same
permissioned dispatch path. The service never starts or loads a neural worker.

The main composition supplies the existing current-task attachment reader,
current attachment policy, exact continued-task source validation, physical
resource admission, brain workspace root, and disk reserve checks. These are
trusted callbacks, not renderer arguments. Runtime installation, lifecycle
retirement, and snapshot/adopt/full-merge/recovery composition are coordinated
by the root lane; the focused tests below exercise the service and closure
helpers directly, not the assembled Electron application.

## Read and search actions

| Action | Exact boundary |
| --- | --- |
| `read`, `read-all` | One bounded first page, then `read-next` until complete. Neither returns one giant full-file string. |
| `read-segment` | Offset and maximum byte length, with optional checked `expectedFileIdentity`. It returns `nextOffset` if needed but does not retain a cursor. |
| `read-next` | Only the existing opaque cursor and optional identical original path. It cannot change source or filters. |
| `list`, `list-next` | Streamed `opendir` traversal in host directory order. `totalKnown` is false until the actual end. No whole-directory `readdir` allocation. |
| `search`, `search-next` | Metadata, case-insensitive name substring, or exact literal UTF-8-byte content search. A bounded empty page can still have more source bytes/entries. |
| `close` | Retires the current read/search/list cursor and owned handles. |
| `materialize-task` | Copies a selected current-task attachment to an exact main-owned path for a separate shell/code action. It never executes that action. |
| `write` | Existing bounded text write, with streamed expected-SHA verification and a final current-authority check before atomic publication. |

Reads default to 64 KiB and admit up to 1 MiB per page; a larger request is
explicitly refused, not silently clamped. Every page reports its exact offset,
byte length, UTF-8 or base64 transport encoding, page SHA-256, and checked source
identity. UTF-8-shortened pages from the existing attachment reader advance by
their actual byte count. `wholeSha256` appears only after a complete sequential
read from offset zero. A page checksum never claims to cover an unread tail.

Search retains at most the current 64 KiB source block, boundary-pattern carry,
and admitted result viewport. The default result page is 64 matches, up to 256;
the default scan budget is 1 MiB, up to 16 MiB per action. Content patterns are
literal UTF-8 bytes, not regular expressions or case-folded content. Matches
can overlap and cross source-block boundaries. Returned offsets and match
checksums refer to exact bytes. Recursive traversal is streamed, skips link
targets, and exposes no arbitrary tree-cardinality ceiling. Oversized transport
results fail visibly instead of reporting an unvisited tail complete.

## Current authority and source custody

`FileToolContext` is `{chatTurnId?, policyRevision, assertCurrent}`. The turn is
the actual main-owned task relation. The stable policy revision includes current
permission/package revisions, not the first action's temporary origin lease.
Each new page is authorized using that new call's `assertCurrent`; retirement of
the first action does not invalidate a legitimately continued page. A changed
policy, source, task, or owner does invalidate the old cursor. `Off` is not
bypassed by attachment selection, cursor possession, copied history, or a prior
approval. `Ask`, reviewer, module-origin and nested-tool checks still run through
ordinary `ToolExecutor` admission for every action.

Host sources require canonical absolute paths and regular files. Leaf links and
symbolic ancestors are refused. Opened file descriptors are bound to
device/inode/size/mtime/ctime/mode and checked against the current path before and
after awaited reads. POSIX opens use `O_NOFOLLOW`. Directory identities are
checked during traversal. This is explicitly checked-open-file consistency,
**not an atomic whole-filesystem snapshot**.

Task sources use the existing continued-task reader and exact original
attachment hash, manifest, size, and source identity. Warm/inherited task files
retain original owning brain/turn identities; those are not rewritten to pretend
the successor initially selected them. No parallel attachment registry,
dataset ingestion, or neural mutation is introduced.

Opaque HMAC-bound cursors belong to the current host, brain, task and page
generation. They expire after five minutes of inactivity and are retired by
Stop/lifecycle cleanup. A current session retains one previous page result for
the same request ID and exact operands; conflicting reuse refuses. Restart,
copy/import and full merge never import a live cursor or its authority.

## Imported-file handoff and portability

`materialize-task` accepts only an attachment ID in the actual current task. It
does not accept a supplied destination path. The returned path names a full,
hash-verified copy under `attachments/handoffs/<requestSha256>.data`; command and
code tools can use that exact path only through their separate current
permissions. Copying does not start a shell, grant code access, or create a
permission carveout.

Durable intent binds brain, task, request ID and exact source before copying.
Publication is exclusive and does not replace an existing destination. Exact
duplicate requests validate and reuse the same copy; changed source/operands
refuse. A failure after actual copy publication retains an explicit
`FileToolEffectError` receipt with `copyPublished: true` and
`executionStarted: false`. The copied file is not deleted or described as
unapplied merely because a later Stop or acknowledgement check failed.

`withFileHandoffSnapshot` serializes handoff publication against closure capture.
`prepareFileHandoffCustody` validates exact intent/receipt/data and writes only
dormant current-owner aliases. `validateFileHandoffCustody` performs the same
source checks read-only. `rekeyFileHandoffCustody` preserves immutable original
source IDs/receipts/bytes, archives old alias bytes, and creates current-owner
historical aliases. `mergeFileHandoffCustody` validates its parent source
**without writing or annotating it**, copies exact closure bytes, and rejects
conflicting same-path content. It runs inside the caller's existing staged merge
transaction, not a claimed standalone atomic directory replacement.

Raw intent, complete receipt, full data, source indexes and alias history remain
in the closure. Published-but-unacknowledged and interrupted-unpublished states
are distinct. Partial `.next` files are retained with exact byte/hash notes and
`usableAsCompletedHandoff: false`; an unprovable partial file is explicitly
interrupted-unattributed, not assigned a fabricated task. Orphan data, receipts,
custody/interrupted aliases, unsafe paths, changed raw bytes and corrupt history
refuse closure validation. Every alias is `historical-not-live`; no old absolute
path, open handle, approval, credential or execution grant is copied as live.

## Declared teaching and evidence

`declaredFileCapabilityPack()` produces ordinary project-authored pack
`bounded-file-tools-1`, version 1, with the actual catalog action schemas and 13
declared examples. Every new read/list/search/close/handoff action has a record.
Placeholder cursors/paths are explicitly examples, not grants or successful-run
receipts. Codepoint-sorted canonical hashing matches the existing declared-pack
contract. `executedHostActions`, preference labels and reward-model claims remain
false. The public manual corpus version is unchanged by this slice.

Nine distinct tiny owned-filesystem cases in `tests/fileToolService.test.ts`
passed across two focused commands:

- Eight baseline cases passed in 459 ms: complete/segmented reads and checksums;
  a new current action after first-origin retirement; changed task/policy and
  source refusal; cross-block/overlapping content and name search; symlink,
  forged-cursor/path and revoke-during-await refusal; exact continued-task
  shortened UTF-8 reads; idempotent nonexecuting handoff; late-publication Stop
  receipts; dormant copy/rekey/merge with unchanged parent source; orphan
  custody refusal; explicit partial-stage retention; declared capabilities.
- One new affected case passed in 12 ms with the prior eight skipped: orphan
  interrupted metadata refusal, read-only preservation, all new action records,
  no fabricated results, and exact codepoint canonical pack hash.

These are real small temporary-file operations with injected permission/resource
callbacks, not real user-file access, shell execution, actual Electron tool
delivery, Windows filesystem acceptance, package verification, model behavior,
or a throughput benchmark. The existing attachment verifier may still need a
full first-source verification; these tests do not turn that cost into a speed
claim. Broader assembled-host/platform acceptance remains separate.

## Root composition follow-up

Main installs the service using the current continued-task reader, exact visible
source hashes/size, current attachment permission revision, reviewed family-RAM
envelope and disk admission. Permission changes, suspend, recovery, action Stop
and app shutdown retire owned live cursors/work. Initial Build and later
declared-capability preparation install the new pack.

The shared learning workspace captures attachments under the same handoff gate
and prepares exact dormant custody; adoption rekeys aliases, per-parent full
merge retains original handoff bytes, and recovery preserves later history.
One new actual shared snapshot → adopt → recapture → merge case passed in386ms
in `tests/fileToolWorkspaceWiring.test.ts`. Its first incomplete fixture was
correctly rejected for missing authoritative attachment registry; the corrected
fixture creates that registry rather than weakening validation. This remains a
small real-filesystem/source-hook check, not real user-file/shell/brain or
packaged-platform acceptance.
