# Optional MacBook Touch Bar controls

Omni AGI Studio exposes a small native Touch Bar mirror on macOS systems where
Electron and the hardware support it. It does not replace or hide the existing
composer, keyboard, Stop, or Call controls. Windows, Linux, and Macs without a
Touch Bar continue to use the same visible controls; the native adapter is
inert there.

## Current controls

Only the currently visible conversation can publish controls:

- **Send** mirrors the enabled idle composer action.
- **Queue** replaces Send while conversation work is active and uses the same
  draft, attachment-preparation, context-fit, Load, and inherited-file-review
  checks as the visible composer.
- **Stop** exists only for one exact loaded, active text turn. It is removed
  while cancellation is already under way and is not a saved-frontier Stop
  control during Unload. Live voice retains its separate **End Call** action.
- **Call / End Call** exists only when the actual live-voice controller is
  available. Starting a call still requires a loaded runtime; ending an active
  call remains available through the normal call controller.

The Touch Bar state contains only instance identity, a monotonic revision,
control modes/readiness, and the exact active turn ID needed by Stop. It does
not contain the draft, prompt, response, credentials, files, or a model-written
executable payload.

## Ownership and stale-action fences

`MacTouchBarController` accepts updates only through Studio's trusted main-frame
IPC boundary. The main process validates the exact state shape, binds button
callbacks to the focused window and accepted revision, clears ownership on
top-level navigation/window close, and removes the bar while the window is not
focused.

The renderer caches a proposed state only after the main process acknowledges
that exact current revision. Before dispatching any callback it checks the
current brain, the accepted state, the latest semantic readiness, and the exact
live turn or voice-controller state again. This second check closes the interval
between a state/ref change and the next native update: for example, an old Stop
callback cannot cancel a newer turn through the generic cancellation function.
Hidden, replaced, unmounted, or unloaded conversation workspaces clear native
ownership.

## Recorded evidence and limits

- Three injected main-controller cases previously passed for strict payload
  validation, focus/revision ownership, stale Stop removal, navigation reset,
  and non-macOS inert behavior.
- One preload case previously passed for the typed update/listener bridge.
- Two initial renderer presentation cases previously passed for Send/Queue,
  exact Stop, call readiness, hidden state, and stale operand rejection.
- After the final App integration, one new focused race/readiness case passed;
  renderer TypeScript and scoped whitespace checks also passed.

No physical Touch Bar was available for this implementation lane, so button
appearance, actual hardware presses, accessibility behavior on Touch Bar
hardware, packaged macOS behavior, and cross-version Electron/macOS acceptance
remain unverified. These focused checks are not CI, signing, notarization,
packaging, or release evidence.
