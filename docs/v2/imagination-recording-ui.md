# Imagination recording in the existing app

## Mounted behavior

`ImaginationCaptureControls` is mounted in the existing Imagination controls
and a compact Record drawer beside Call in chat. It does not replace either
workspace or send a prompt, create a model, or choose an imagination action.

- Record now or look back 15/30/60/180/300 seconds, with a custom lookback.
- The rolling window uses those presets or a custom value up to 300 seconds.
- Display actual available lookback, saved-fragment count and recorded gaps;
  requested lookback is never presented as recovered content when absent.
- Stop keeps committed fragments. There is no delete/undo disguised as Stop.
- Always-save starts off. Copied preferences remain visibly dormant until
  explicitly activated; a stopped automatic recording can be deliberately
  rearmed through Apply. Saved dormant/resource-paused recordings expose Resume.
- Browse recording and fragment pages without a total-library display cap.
  Individual verified delivered images/audio/video use normal media controls;
  missing media is marked unavailable. There is no fabricated completed MP4.

Async reads and writes are fenced to their mounted instance. Event refreshes
coalesce and never overwrite unsaved settings during unchanged revisions.
Controls apply acknowledged state immediately rather than waiting for a timer.
A recording request keeps its ID after a lost acknowledgement, so Retry does
not create another recording. Stop uses the control revision, not the changing
frame revision. Switching instances never submits the previous instance's
pending action to the new owner.

## Scope of proof

Four new presentation cases passed in 24 ms: duration validation, owner/revision
page merging, fragment isolation and no effects/media during server rendering.
Renderer TypeScript passed at this source point. These are not neural tests.

The actual component was exercised through computer use with an explicitly
synthetic host at `tests/e2e/imagination-capture-fixture.html`:

1. A 30-second requested lookback reported only 12 seconds available; Stop kept
   its three fixture fragments. Unavailable bytes were not represented as media.
2. Lost recording acknowledgement retained the exact request; retry returned
   the one committed record and immediately exposed Stop. The synthetic call
   ledger showed one create plus one same-request retry per deliberate recording.
3. Switching to a dormant copy reset the view to that owner's empty recording
   history, unchecked always-save and explicit activation. Narrow dark controls
   fit without clipping.
4. A custom 45.125-second window applied. A 60-second lookback was rejected
   before a new synthetic recording call.

Captures are under `test-results/imagination-capture-ui-manual/`.
`01-lookback-stop-narrow.jpg` is diagnostic before a subsequently fixed excess
label-height rule; `02-idempotent-retry.jpg` and
`03-dormant-copy-dark.jpg` capture the corrected controls. A later fixture-only
scroll-container correction allows explicit below-fold inspection; it changes
neither the app nor the capture contract.

No actual brain, decoder, user file, device recording, target-process access,
permission grant, provider, training corpus, CI or package ran in this UI slice.
Real media throughput, portable packaged UI and learned spontaneous/user intent
remain separate gates. Ordinary native chat actions without trusted main intent
stay unclassified and retained, not silently keyword-classified and erased.
