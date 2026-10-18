# Time and schedules — bounded interface check

October11 native Electron/computer-use check of the real
`ClockWorkspacePanel`. The transport in `tests/e2e/clock-workspace-fixture.tsx`
is explicitly synthetic: no real schedule, notification, permission grant,
model Load, neural update or external action was created. Backend timing,
delivery and portability evidence belongs to the clock service's separate
fixtures; this document does not replace it.

1. **Entry and ordinary controls: inspected.** The dark full-width panel shows
   time/zone, reminder/task entry, stopwatches, saved schedules, wakeup receipts
   and expandable rate history. Initial capture:
   `test-results/clock-ui-manual/01-initial-dark.png`.
2. **Task submission: passed at the synthetic UI boundary.** A recurring
   ten-minute wakeup with explicit Load checked produced a review of its exact
   saved task, interval, delay and `loadWhenDue:true`; approving it added exactly
   one fixture schedule. This is not real automatic-Load acceptance.
3. **Approval placement: corrected.** Saving from lower on the page originally
   placed review above the viewport without a visible response (capture02).
   User-triggered approval/error now focuses and reveals its own region;
   background events do not force scroll. A lower-page saved-task review was
   checked again after the final start-alignment correction: capture06 shows
   its complete heading, operation and readable Approve/Decline controls.
   Decline removes the review and says “Declined; no action ran.”
4. **Validation and stopwatch controls: passed at the UI boundary.** Invalid
   zero duration revealed the actual validation message without dispatch
   (capture03). Start, lap, lap disclosure and Pause operated through the
   synthetic approval flow; the paused stopwatch retained its measured value
   and disabled only Lap. Hot reload during implementation recreates this
   in-memory fixture; none of these checks claims persistence through restart.
5. **Narrow light reflow and lower controls: corrected and inspected.** A320px
   panel within a wide window originally kept two columns, clipping selectors
   and overflowing the stopwatch input. Container-based reflow now stacks
   those columns and paired fields. Captures04 and05 show the corrected top
   and fully scrolled bottom with wrapped labels and accessible action rows.
   Hover/focus text remained readable in the inspected states. This does not
   establish every theme, physical device or assistive-technology behavior.

All accepted captures were inspected as the actual selected fixture window;
capture02 is before-fix evidence, not acceptance. No unrelated window was
operated. The actual App-host fixture later replaced this window.

Eight duration presentation cases pass, including minute/hour carry and
unknown-duration handling. Six affected navigation cases and renderer
TypeScript pass. Source navigation now exposes the schedule page while busy
or unloaded; no whole-production navigation/model-wakeup claim follows from
this isolated presentation run.
