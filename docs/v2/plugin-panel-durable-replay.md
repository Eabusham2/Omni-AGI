# Existing scripted panels: durable replay instead of a lifetime quota

This is a separate correction to `PluginInteractivePanelService`, not a claim
that the new lifecycle runtime removed the older panel limit automatically.
`MAX_REQUESTS_PER_VIEW=256` and its close/reopen requirement are removed from the
existing transport. Static panels, opaque scripted documents, declared tool
selection, current permission/revision checks, and child response semantics stay
in place.

## Physical resources and exact effects

The service now requires host composition for the exact brain root, real disk
reserve checks, and transient shared-family transfer admission. There is no
production in-memory fallback. Only in-flight requests and unresolved Ask
reviews remain in the live request map. Each owns an admitted transfer lease;
terminal durable commits remove their request/host IDs and release the lease.
Pending private approvals consume their admitted memory until reviewed or closed.
A large lifetime history does not produce a growing main-process request map.

`plugin-panels/workspace.sqlite3` preserves nonce-free view/source/revision
metadata, exact original operands, host request identity, prepared/dispatched
states, actual host results, final child receipts, review dispositions, and
hashed complete event history. Per-write disk checks cover transaction/journal
headroom. A full disk or RAM ceiling returns visible pressure; it does not reset
the view, evict historical receipts, or repeat an effect.

An intent and dispatch checkpoint commit before ToolExecutor runs. Current
owner/nonce/package hash and access revisions are checked again after awaited
admission and immediately before dispatch. Exact old-ID retries consult checked
durable receipts; changed operands under the same ID fail. A prepared/dispatched
request without a terminal acknowledgment returns a stable **uncertain** receipt,
never another attempt. A new request that failed before any durable preparation
may be retried when resources recover because no effect occurred.

The existing 1 MiB input and 2 MiB child output limits remain transport frames,
not a lifetime history or source-data limit. Full actual ToolExecutor results
are admitted before serialization and retained in history even when the child
projection is too large. History has explicit metadata cursors and checked byte
pages, including oversized records. SQLite integrity/hash scanning is bounded in
the JavaScript transfer buffers; this is not a constant-time I/O claim or a
large-database/native-allocation benchmark.

One-use approval tokens and channel nonces never enter portable rows/history.
Ask retains the token only in the current host request, and rechecks authority
before a reviewed effect. Terminal review retries read their recorded outcome;
late approval responses after close are invalidated. Closing aborts pending
pre-dispatch admission, cancels exact active host requests, waits for actual
terminal custody, and closes the writer. Trusted `closeOwner` also awaits a
writer already closing outside the active-view map, preventing recovery from
racing that flush.

## Complete, dormant custody

The existing learning-workspace export/fork/import envelope now captures the
current panel database and checked SHA-named source archives. Isolated SQLite
snapshots include committed journal state; copied owner metadata is rekeyed and
saved views become dormant. Exact original payloads and all actual result history
are preserved. No nonce, pending token, view activation, or external operation
is restored or auto-run.

Full merge retains each complete parent database as an immutable namespaced
source, deduplicating only identical archive bytes. Recovery retains later real
panel effects as history without replaying them. Neither merge nor import
recreates an active iframe or grants a new view access to a parent's request IDs.
Historical source reads check exact SHA, owner and file identity again before
publishing their data. The API exposes `history`, `historyBytes` and paged
`historySources`; these are human-selected history inspection, not module grants.

## Focused evidence

Only new fake-host/small-SQLite fixtures ran. No model/brain, training, provider,
remote service, device, personal source, or real host effect executed.

- Six new cases passed in **2.165 s**. One existing view completed **270** unique
  cheap fake calls, retained zero terminal live request/host-ID entries, replayed
  request 0 without another effect, and refused changed operands. Other cases
  cover RAM/disk pressure before work, lost acknowledgment without replay,
  cancellation during pending disk admission, private Ask/final review replay,
  complete over-frame output/history byte pages, and real export-envelope
  copy/merge with dormant state and unchanged original file checksum.
- One distinct already-closing owner-boundary case passed in **11 ms**; the six
  completed cases were skipped, not rerun.
- Owned-file syntax transpilation and scoped tracked-file whitespace checks
  passed. Existing old fixtures received only mechanical temporary-store/fake
  admission constructor injection; their previous suite was not rerun in this lane.

Root subsequently wired required resource/disk/root options, suspend/recovery
closure, typed main/preload/IPC history operations and the lazy history UI in the
existing Plugin Manager. Two new history UI cases, one affected preload history
case, and coordinated main/renderer TypeScript compilation passed. Actual native
panel-history and packaged acceptance are not inferred from those checks.
These focused source cases do
not prove every platform, arbitrary third-party panel quality, perfect OS memory
enforcement, or trained native understanding of tools.
