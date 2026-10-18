# External assistant and action-review custody

## Scope and authority

R236's optional assistant and R094's optional pre-action reviewer share the
brain-managed `external-advice/workspace.sqlite3`. This is an advisory history,
not Omni neural memory, a reward model, or permission authority. Root's separate
device-private credential/grant store is never part of this workspace.

Custody retains accepted source prompt, visible selected context, response,
errors, provider/model selection, settings, source IDs/revisions, timestamps and
parsed review receipts. It does not sanitize that recorded history or silently
turn enabled preferences off. Structural credential IDs, keys, approval tokens,
live controllers and unknown executable/schema objects are rejected rather
than exported. Ordinary quoted human text is not treated as an operational
field. Service-level pre-provider credential handling remains a separate
boundary; this component does not reconstruct text that was never recorded.

An external import, new merged identity or duplicate has no copied device grant
or in-flight call. It therefore remains dormant even when its retained portable
settings say enabled. An explicit current-device activation is separate from
reading history. Historical `allow`, revisions and observed evidence cannot
become a one-use approval token or override existing tool permissions.

## Committed snapshots and exact source records

Only the exact workspace database path is admitted by the complete learning
workspace inventory. The existing isolated SQLite online snapshot captures one
committed transaction, including a truthful queued/streaming prefix; live WAL,
SHM and journal files are not copied to a new identity. The captured CAS blob is
then checked by the same backend file-level schema/hash/row validator before the
inventory claims completion. Files stream through the existing hash/copy path;
history is not collected into one growing JSON array.

Rekey writes the source request/settings/meta archives and the new presentation
in one FULL-sync transaction. `inherited_requests` preserves the canonical
original request payload and SHA; `inherited_settings` preserves each complete
channel configuration; `inherited_meta` retains its original owner, settings
revision and update date. Source request/context/preview hashes remain bound to
the source—not falsely recomputed as authorization for the new owner.

The target request presentation may differ only in its current owner, checked
ancestry-derived ID, ancestry pointer, and queued/streaming→cancelled state.
Every prompt/context/response/time/error/provider/authority-audit field remains
source-exact. The original active-state record remains archived. No provider
call is resumed and no successful/complete result is fabricated from a prefix.

Parsed `ActionReviewerApprovalReview` receipts have their own `reviews` table,
including timeout/unavailable results with no provider request. Their original
payload, source owner, review ID, proposal hash, exact invocations, selected
evidence, disposition and timing remain unchanged. Only current row ownership
and a checked inherited row ID change. When recovery returns to the source
owner itself, its original review ID remains valid; that is not a fresh review.

## Duplicate, full merge and recovery

Trusted duplicate/fork and complete `.omni` export/import use the existing
learning-workspace snapshot/adoption path. All advice history travels; no
installation-private grants or credentials do. Mutable destination files have
independent inodes, and unexpected hardlinked writable databases are refused.

Full merge preserves every parent's source records, configurations, settings
metadata and parsed reviews. A verified original source owner/ID/SHA controls
ancestry deduplication. Matching prompts, plausible content, model names or
independent identical-looking replies are not enough. Different content is
retained, not averaged. The first retained current configuration stays chosen;
other parents' complete configurations remain in checked history. Parents
are read/snapshotted, never mutated by this helper.

Recovery keeps the snapshot's chosen settings but imports later real advice
and review receipts as historical branches. An earlier prefix and later final
for the same request are both retained under their distinct source SHAs. The
repository performs this before host-workspace promotion and includes the
advice directory only when a staged owned database exists. It does not undo an
external effect or resurrect a pending call. Main lifecycle integration must
retire the old active owner epoch before same-ID promotion so a late provider
callback cannot write into the restored identity.

Cancellation/resource refusal rolls back the private transaction. Per-row SQL
iteration, original rowid frontiers and bounded SQLite cache settings avoid a
history-count-sized memory collection. Actual files/rows still consume disk and
transfer RAM; this is not unlimited physical capacity or a new learned store.

## Evidence and remaining boundary — October 13

`tests/externalAdvicePortability.test.ts` uses only small valid synthetic records
and real SQLite/portability functions—no model, provider, key, network or app:

- Six new cases passed in 429 ms (2.36 s runner): complete archive/adoption,
  independent copies, exact source archives, retired streaming prefixes,
  preserved enabled settings, repeated-copy ancestry, distinct-parent merge,
  shared-ancestry dedup, standalone parsed timeout receipts, structural-field/
  missing-archive refusal and cancellation rollback.
- A seventh, distinct same-ID recovery case exposed an incorrect inherited
  review-ID choice. The helper was fixed to retain the source review ID when
  owner and source are the same; the strict validator was not weakened. Only
  this affected case reran and passed in 57 ms, with the earlier six skipped. It
  checks later prefix/final/settings/review retention and a static repository
  hook/promotion-order contract, not a live full-repository recovery.

Actual provider streaming, current-device activation/epoch retirement, mounted
assistant/reviewer UI, live approval/late-callback handling, complete native
brain restore, packaged platforms and public release remain separate gates.
No old contract suite, neural fixture, broad compiler, CI, commit or publication
was run by this portability slice.
