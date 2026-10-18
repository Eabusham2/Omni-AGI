# Sparse substrate jobs in Studio

Internal UI contract and evidence note, October 15, 2026. This note covers the
current Studio presentation for the narrow native sparse-transfer boundary in
[native-sparse-transfer.md](native-sparse-transfer.md). It does not expand that
boundary into whole-cortex editing, prove a useful neural transfer, or claim
that the synthetic Electron walkthrough performed a native mutation.

## Current product contract

Studio now mounts sparse-transfer job cards in the existing Tools surface and
binds a job to its chronological `ChatActionCard` when the main-owned
`actionEventId` matches the action event and `originalBrainId === brainId`.
Typed execution output can supply the exact job ID when there is no valid
current-owner action correlation. Historical copied correlation remains
provenance; it is not used to place a copied job into a new owner's chat.

The dedicated `window.omni.substrate` API exposes:

- paged current-owner job summaries, one exact job, and one exact plan preview;
- owner-filtered job revision events;
- trusted Studio Stop for one existing job.

It exposes no raw receipt JSON, authorization revision, callback, approval
token, native path, or mutation method. Full unsanitized receipts remain in
main-owned portable custody. Apply and inverse apply still go through
`brain.substrate` and the ordinary exact ToolExecutor Ask review. Studio Stop
only cancels the selected existing operation; it does not create authority,
undo a commit, or turn an uncertain dispatch into a safe retry.

## Subscription and snapshot fence

The preload listener is installed before the subscribe request. Main registers
the exact sender/brain/subscription identity, then takes an owner-validated
initial page of up to 40 jobs. A failed owner/read check removes that
subscription and rejects the acknowledgement.

The acknowledgement carries the page and its observed timestamp. Preload
validates brain ownership and positive revisions, delivers the page, and then
delivers only the highest buffered pre-acknowledgement revision for each job.
Later events are revision-deduplicated. A dispose before acknowledgement
removes the listener and exact main subscription; a late acknowledgement cannot
revive it. The normal paged read remains available for older jobs and its
cursor, so this initial page is a race fence rather than an all-history load.

## Review and state presentation

Cards use durable job states instead of interpreting a queue acknowledgement as
success. The current presentation covers queued/updating, stopped, historical,
review-required, failed, uncertain, and complete results. Full identifiers and
hashes remain in foldable details. Preview cards show the exact operation,
append or replace-exact mode, row/addition/replacement/identical/conflict
counts, and wrapped hashes. Apply is a separate exact-SHA Ask after preview.
Rollback is also a separate reviewed inverse plan; Stop is not rollback.

The current inverse limitation remains visible: replacements with saved
before-images can be previewed, while deletion of introduced rows stays an
explicit conflict until complete live-reference proof exists.

## Numbered synthetic Electron walkthrough

Root reports the following accepted observations in
`test-results/sparse-jobs-ui-2026-10-15/`. The fixture used the real mounted
React presentation with synthetic transport and state. It did not use the
production preload/IPC path, create or Load a brain, invoke a provider, or
perform a real native effect.

1. `01-existing-controls.jpg` records the existing Tools controls before the
   sparse cards, as a baseline rather than evidence of a changed product layout.
2. `03-stop-light.jpg` shows one synthetic queued job changing through Updating
   to Stopped while an earlier completed job remains visible; timestamps align.
   `02` was superseded because its timestamp presentation was stale.
3. `04-narrow-dark-top.jpg` shows the top of the dark compact surface without
   horizontal clipping. The measured content width was 360 px inside an
   1,100-px native fixture window; this is not a claim that the app window itself
   ran at 360 px.
4. `05-narrow-dark-review.jpg` shows the lower review state with wrapped long
   SHA values, Deny and Approve, and an exact replacement inverse preview of
   zero additions and 48 replacements.
5. `06-approval-keyboard-focus.jpg` records actual Tab focus on the approval
   control.
6. `08-page-bottom.jpg` shows the page bottom after Return approved a new
   synthetic queued job, with no bottom clipping.
7. `07` was rejected because it cropped the relevant content. Rejected or
   superseded captures are not acceptance evidence.

The initial inverse Deny made no change. A fresh preview Ask followed by Approve
created only a preview. Apply required a separate exact-SHA Ask and produced a
synthetic queued job; it did not perform native apply. The owned fixture window
was closed through the UI and exited with status 0.

## Focused source evidence

No checks were rerun for this documentation update. Reported current evidence is:

- four new sparse job UI/controller cases passed in 77 ms;
- six distinct sparse UI service/preload/main-subscription cases cover safe
  projection, revision delivery, owner routing, listener-first acknowledgement,
  snapshot/live coalescing and trusted exact-job Stop;
- three structured-experience guard cases passed in 5 ms;
- the native/store/portability mechanism evidence remains separately enumerated
  in [native-sparse-transfer.md](native-sparse-transfer.md).

These are scoped source and synthetic-presentation checks, not a whole-suite,
real IPC walkthrough, actual loaded-native mutation, release, or package result.

## Strengths and remaining boundaries

The current implementation keeps the existing app, locates each current-owner
job chronologically, preserves completed history, uses truthful durable states,
wraps long evidence, provides keyboard-reachable review controls, and separates
preview, apply, Stop, and inverse semantics.

Still open are a real service/preload/IPC walkthrough with an actual instance,
broader theme/window/accessibility qualification, packaged Windows/macOS/Linux
acceptance, meaningful resource-pressure behavior, and trained usefulness.
Actual native execution is deferred under the current no-new-brain instruction;
that evidence boundary must not be filled with synthetic output.

The source-wired native candidate proposal route is documented separately in
[native-candidate-tools.md](native-candidate-tools.md). It connects an explicit
`source.self-modify` candidate kind to the existing evaluator, but its focused
source tests do not prove native candidate execution. Whole-instance merge
remains its own reviewed feature. Neither requirement implies inventing an
arbitrary tensor/cortex editor.
