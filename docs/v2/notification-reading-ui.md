# Readable notifications in the existing app

The existing corner notification remains in place. Each notice now has its own
identity, even when two notices contain identical text. Its UI-only reading
timer uses monotonic elapsed time, pauses while hovered or keyboard-focused,
and resumes from the remaining duration. A retired timer cannot dismiss a
newer notice or erase a resumed timer.

Short notices receive four seconds and longer notices up to eighteen seconds
before automatic dismissal when not being read. These are display durations,
not model deadlines, job timeouts, ETAs or cancellation. The message region
wraps, scrolls when needed and is keyboard-focusable. Dismiss remains explicit
and keyboard-accessible. A neutral information icon no longer represents every
error as a successful operation.

The existing concise-message policy is retained; this change does not add a
full diagnostic viewer or alter complete operational logs. It does not change
the application's screens, layout system or theme selection.

## Focused evidence

Four initial timer/presentation cases passed in 7 ms. A distinct late-callback
case passed in 1 ms with the earlier four skipped. These cover monotonic
remaining time, no multiplied timer, retired-owner isolation, longer reading
time, accessible markup, and same-owner pause/resume callback retirement.

Computer use inspected the actual component with synthetic notices:

- `01-light-narrow-focused.jpg` and `02-dark-narrow-focused.jpg` show readable
  wrapping and retained keyboard focus at a 360-pixel document width. Measured
  toast bounds were left 10, right 350, width 340 pixels.
- The focused notice remained present for more than 28 seconds, beyond its
  ordinary maximum reading timer, while other work continued. An unfocused
  notice later expired; no blocking wait or model work was involved.
- Identical notices received separate owner identities. Tab exposed the
  Dismiss focus outline in `03-dark-dismiss-focus.jpg`; Enter removed the
  notice.

Images are under `test-results/toast-notice-ui-manual/`. This is bounded
light/dark component evidence with synthetic transport, not full Electron,
all-theme/viewport, accessibility-conformance or real-job acceptance. No model,
provider, credential, training, job cancellation or permission action ran.
