# Explicit Ponder episodes and reply chronology

## Implemented source contract

`NativeActionEmissionLedger` reserves the Ponder worker identity before its
actual invocation. `NativePonderEpisode` emits an operational `ponder-episode`
frame with `state: started`; its measured `finished` frame comes from the
native loop's cleanup. Main receives these as `chat-ponder-episode` frames and
updates the same visible action, without scheduling a second idle cycle.
Completion reports measured convergence or an existing learned action-head
transition. Cancellation, Steer, suspension, resource pressure and failure do
not become fabricated successful Ponder results.
If setup/cleanup fails before final pass measurement, the failure reports
unknown pass/delta totals explicitly, not an invented zero or successful work.

Inside that active invocation, a non-Ponder transition from the existing
internal action head may use the existing typed schema/argument selector.
Private and visible transitions perform the same selector computation and
local proposal bookkeeping, including a genuinely selected native Stop.
Only the visible invocation can publish the resulting selected action. Its
top-level `parentPonderActionId` is assigned by the active emission ledger;
model/tool argument fields cannot create that relationship. Repeated effects
retain their existing exactly-once identity even if another episode considers
the same effect.
A genuinely selected typed Stop retains the native decoder's existing Stop
boundary; it is not merely displayed as a later host-side stop card.

Canonical pre-speech computation has its own ledger before private Best Work
branches begin. Private branches copy that completed ledger and have no
executable emitter. The selected visible decode/replay reuses canonical
identities. Mid-generation episodes use the same wrapper. An active Ponder
invocation cannot be saved as a token-suspension frontier; it first reports its
actual stopped/completed boundary, and completed provenance survives resume.

Main accepts a child link only when its exact native parent is currently
active, has actual kind Ponder, and belongs to this same brain and chat turn.
It derives durable `ActionEvent.chatTurnId` and
`parentPonderActionEventId` using host identities. The relationship means
**selected during Ponder**; a permitted host effect may finish later.
Permissions, approval, result retention and cancellation remain owned by the
existing host executor. Neighboring events, timestamps, human actions and
unknown historical events do not establish ancestry.

## Exact public-reply anchors

Main incrementally measures only actual public `chat-token` deltas. An action
received on that stream can carry a live `replyAnchor` containing `turnId`,
`utf16Offset`, and `prefixSha256`. Ponder self-reports are a different channel
and are not part of this public prefix. Final-only historical proposals do not
receive an inferred position.

The existing native public-text canonicalization keeps its whitespace-strip
behavior. At that exact boundary it produces an
`omni-public-reply-text-binding` receipt with original/canonical SHA-256
digests, UTF-16 lengths, and measured removed prefix/suffix lengths.
Interrupted text uses identity normalization. Main verifies the receipt against
its actual public stream and canonical same-turn brain message, then translates
the offsets deterministically and binds the exact `messageId`. There is no
substring guessing, token-position heuristic or stored prefix plaintext per
action. Canonical prefix digests are computed incrementally in offset order.

Unrelated text changes, missing owners, malformed receipts or tokenizer/public
stream divergence leave the affected entries unanchored. They are not evidence
that a particular reply boundary was observed. The conversation ledger refuses
unbound anchors, validates bound prefixes against the canonical message, and
includes turn, parent, native identity, selection metadata and anchor in its
idempotent projection comparison.

Mid-generation approval, decline and other action updates save a synchronous
immutable snapshot without an unfinished live anchor. The live event retains
its exact prefix measurement. Final canonical binding appends a separately
timestamped revision; an earlier saved action row is never rewritten or
accepted through a relaxed idempotency check.

## Copy, merge, archive and recovery inspection

Source inspection found no additional new-field identity-remapping gap in the
covered history paths. `ConversationLedger.clone` preserves host action,
message and turn IDs while rebinding the brain owner and hash chain. Local
copy, snapshot and archive import use that path; recovery stages the exact
same-owner SQLite ledger. Thus a copied reply anchor still names its copied
canonical message and a child still names its copied Ponder event.

Full-merge host adoption first completes the per-parent identity map, then
recursively projects ID fields. That includes `chatTurnId`,
`parentPonderActionEventId`, and `replyAnchor.turnId`/`messageId`. Original
reply content and prefix digests are unchanged. This is a source inspection,
not a new full-merge, `.omni`, new-brain or recovery runtime qualification.

## Bounded evidence and limits

- `engine/tests/test_ponder_episode_provenance.py`: 14 passing stdlib, AST and
  inert-callback cases; no neural-library import or brain construction.
- `tests/ponderEpisodeProvenance.test.ts`: 16 original focused cases passed;
  two subsequently added mid-generation approve/decline cases passed using the
  actual service snapshot method and tiny real SQLite ledger. The latter run
  selected only those two cases, not the earlier sixteen. Coverage includes
  exact public-prefix/normalization, permission results, Stop/resume and
  immutable terminal-to-canonical action revisions.

These are source/mechanism checks. They do not prove learned natural use,
quality of a Ponder self-report, a successful real provider/tool action, a
trained-model capability, packaged runtime behavior, or release/CI status.
Renderer projection and its no-engine UI evidence are tracked separately.
