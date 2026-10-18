# Saved MCP task custody

MCP task records are source-original operational history, not neural parameters
and not a second execution queue. They keep original task/reference/source
identities, real creation ACKs, observed task states, lost-outcome records,
immutable beforeimages and exact decoded final payloads. Their fixed authority
is `historical-not-live`. Device credentials, session grants and permission
leases are not serialized. Saved text remains unsanitized as requested.

## Coherent snapshots and lifecycle

`captureMcpTaskCustody` runs a short canonical per-source task-store lease and
copies the complete validated folder into caller-owned private staging. Model,
archive and merge work consume those stable paths after the lease releases.
Repository snapshots declare exact task component paths and hashes. Clone
preserves current and immutable-origin task bytes with independent mutable
files; source IDs are never rewritten to grant the copy authority. Restore
selects the saved current closure and retains later exact observations as
content-addressed dormant lineage rather than deleting or resending them.

Read admission uses the same selected app-family metadata guard as the catalog,
with physical/heap fallback when no host callback exists. Source identity,
canonical paths, file size, payload checksums, cancellation and disk admission
remain enforced. Ephemeral repository capture/validation staging is removed on
success or error. A paused merge may retain its coherent private copy for
resume; once all content-addressed blobs commit, its task-capture staging is
discarded. No model is loaded or started to preserve task records.

## Archives and merge

Format-2 archives containing any task receipts require `mcp-task-custody-v1`.
Current/origin task namespaces, historical snapshot components and task folders
inside authenticated opaque connected-parent closures are validated before
identity reservation. Format-1 task payloads and task namespaces missing their
feature token are rejected. A missing payload or valid-checksum JSON carrying
an unknown live-grant field is not accepted as an executable task.

Full merge skips live task folders during its ordinary source walk. It instead
inventories original logical paths from coherent private captures, including
current/origin, ancestor and recovery scopes, then copies from those captures.
Parent seals likewise hash coherent short captures. Complete parent groups
flatten into source-original dormant lineage; identical complete groups may
deduplicate, but distinct source/content history never collapses by task ID.
Existing conflicting-content validators remain strict. No provider polling,
task cancellation, execution replay or permission adoption occurs here.

The complete merged lineage is also part of the new starting saved origin,
when that origin exists in the unpublished assembly. A non-serializable,
coordinator-owned permit is valid only inside its actual assembly callback;
distinct destination identity and that permit are required before origin
custody is seeded. A pathname, imported boolean or later direct helper call
cannot authorize changing an existing saved origin. No parent origin or neural
metadata/tensor/manifest is rewritten or resealed.

Known interrupted-publication `.json.<UUID>.next` fragments are retained as raw
inert bytes through capture, copy, history, merge and archives. They are never
parsed or promoted into a completed task. Unknown partial filenames still fail
closed instead of being silently discarded.

## Evidence boundary

New focused repository cases use JSON/SQLite/ZIP and inert saved fixtures only.
They cover snapshot/copy/recovery, current/origin/history, format-2 collision
import/re-export, raw final outcomes, dormant source ownership, malformed and
undeclared task payload refusal, metadata admission and cancellation. Three
base cases passed in 3.07 s with the other 76 file cases skipped. A fourth
historical-only archive case passed alone in 1.02 s with the other 79 skipped;
it proves feature declaration and invalid-history refusal before identity
reservation even when current/origin contain no tasks. The first base case was
then extended and passed on the affected path with exact inert interrupted
fragment custody. These are file and
host custody proofs, not real MCP network/server, learned tool choice, neural
execution, large-state, UI, platform or package qualification.

The new full-merge task fixture checks staged-copy routing and all actual
parent/current/origin/ancestor/recovery sources with a non-neural coordinator;
its deliberately inert assembler never constructs or loads a model. The
separate embedded-source case exercises complete task closure inside opaque
connected-parent logical paths and passed in the initial scoped run. The
affected all-scope merge case finally passed alone in 1.97 s with its sibling
skipped, including exact new-origin custody, rejection of an out-of-assembly
attempt to seed a saved origin, unchanged original parent origins and removal
of consumed private task-capture staging. The fixture selects exact outcome
versions rather than assuming stable task IDs are globally unique across
different historical source versions. All six distinct new custody cases have
passed across scoped/affected runs; this is not a rerun of the already-closed
archive or broader repository fixtures. TypeScript, complete runtime, network,
cross-platform CI and package qualification remain separate gates.
