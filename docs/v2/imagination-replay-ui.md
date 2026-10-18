# Replay controls in saved imagination recordings

`ImaginationReplayControls` is inside the existing selected-recording details
of `ImaginationCaptureControls`, in Imagination and the chat Record drawer.
It does not replace the app, load a model or generate new imagination.

Prepare a fixed recording to review a delivered-timeline plan, then choose
Make replay. Auto comes from recorded media and resource inspection; optional
width/height/FPS and historical timing overrides remain explicit. Audio-only
plans say WAV rather than displaying a fictitious video canvas. An active
recording must be stopped before preparing a fixed version.

Replay history is paged and filtered by the exact selected recording in main.
Completed available WAV/MP4 files expose normal playback controls. Original
fragments remain separately accessible. Stop cancels replay encoding, not the
recording or its saved source bytes. Only dormant copied replay jobs expose
Resume; failed/stopped jobs require a new explicit prepared request.

Lost creation acknowledgements keep the same request and plan IDs for retry.
Preparation has its own exact request ID, real runtime/source/spooling progress
and Stop acknowledgement. A missing preparation response retries the same
request rather than starting another runtime/download/inspection operation.
Stop remains available while preparation awaits its result; only a typed
terminal receipt ends its waiting state. A finished plan is not described as
cancelled when it wins the Stop race.
Renderer reads/control responses are fenced to the instance/recording owner;
older event revisions cannot overwrite a newer receipt. Expanded receipt
inspection asks for a bounded manifest summary/counts/hash, not an unlimited
array of fragments, tracks or spans. Timing is labelled presentation timing,
not proof of original neural synchronization.

## Focused evidence

Four initial presentation cases passed in9ms: exact Auto/override semantics,
millisecond timing without floating-roundoff rejection, invalid/duplicate
offset rejection, revision/owner filtering, inactive copied/stopped states,
and no effects or fabricated media on initial markup. Renderer TypeScript
passed after correcting a React19 optional-ref initialization error. The
earlier capture/model/media tests were not rerun. A separate copied-recording
alias case passed2ms with the original4 skipped; it uses only main's current
recording alias while preserving the immutable original recording identity.
One further preparation-ownership/progress case passed17ms, with the earlier5
skipped. It rejects foreign/older progress and does not invent0/0 totals.

Computer use exercised the actual component against a synthetic main API:

1. Narrow light entry: no automatic plan, encoder or playback; existing dormant
   history is visible. `01-light-entry.jpg` is accepted.
2. Explicit audio plan and a lost creation response: retry used the identical
   request and plan IDs; the synthetic ledger showed one unique job. A terminal
   synthetic WAV exposed playback controls and a manifest receipt.
   `02-dark-playback-receipt.jpg` is accepted, including below-fold content.
   Playback was not started;0:00/0:00 is an unloaded `preload=none` control, not
   measured file duration or a successful codec/playback proof.
3. Dormant Resume followed by Stop retained source history. Older paging
   preserved owner scope. Width0 failed before another plan call. Expanded
   narrow dark fields and focused Prepare remained readable in
   `03-dark-options-validation.jpg`.
4. Switching instances removed old state and media; the new owner's explicitly
   prepared audio plan had no invented FPS/canvas. Desktop dark view is
   `04-desktop-isolated-plan.jpg`.
5. The newly added preparation flow held a synthetic runtime operation with
   actual-shaped10,485,760/20,971,520-byte progress. Its Stop remained available,
   acknowledged terminal cancellation and reenabled Prepare without deleting
   the saved dormant recording. `05-preparation-progress.jpg` and
   `06-preparation-stopped.jpg` are accepted. A later one-line presentation
   guard suppresses the duplicate short terminal label when the explanatory
   kept-fragments notice is already shown; this did not change a host effect.

Images are under `test-results/imagination-replay-ui-manual/`. These are
presentation/transport fixtures, not a real neural recording, encoder run,
native-quality test, complete theme audit or packaged-platform acceptance.
The real replay backend, actual runtime-download cancellation, and codec/custody
integration remain separate gates from this synthetic transport proof.
