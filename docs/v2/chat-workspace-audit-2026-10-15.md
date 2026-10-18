# Chat workspace audit — October 15, 2026

Internal current-source UI audit. This records one bounded native Electron
inspection of the existing Chat workspace and attachment menu. It is not a
redesign, broad accessibility audit, backend acceptance run, or trained-brain
evaluation.

## Audit boundary

The harness used the existing `responsive-main.cjs` no-preload Electron path,
an isolated profile at `/tmp/omni-v2-chat-audit.qgsvbx`, and the compiled current
production renderer. It did not create or Load a brain, train anything, call a
provider, learn a file, or perform a real chat/tool effect. Fixture state was
synthetic and existed only to inspect the mounted interface.

macOS **Window → Move & Resize → Top Left** produced saved screenshots measuring
1,444 × 962 pixels. That is the captured image size, not a claim about CSS
viewport dimensions or every compact window size. The owned window was closed
with Command-W; the Electron process exited 0.

## Confirmed defects before the correction

1. The Chat header exposed **Pause** and **Delete** shortcuts. Pause duplicated
   narrower task/job controls, while Delete placed an identity-destructive
   action beside ordinary conversation controls.
2. A preview-only **fast learning active** badge asserted a runtime condition
   that the fixture had not measured.
3. The attachment menu surface was transparent enough for page text to bleed
   through, reducing readability.
4. In a short window, the attachment menu was constrained by its parent
   scroller and clipped later choices. Compact runtime copy also overflowed the
   available header width.

## Implemented narrow correction

- The Chat header no longer contains Pause or Delete. Instance deletion remains
  available through the existing Library confirmation dialog. Dataset/task
  Pause remains on its actual scoped work surface.
- The unsupported fast-learning preview badge was removed rather than replaced
  with another decorative status.
- `ChatAttachmentMenu` now uses the native `auto` popover top layer, an opaque
  theme-aware surface, and trigger/viewport-derived geometry. It is no longer
  clipped by a parent element's positioning or overflow. This matches the
  platform behavior documented by [MDN's `popover` reference](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/popover):
  open popovers enter the top layer, and `auto` popovers support light dismiss
  and Escape.
- The component closes after a choice, Escape, or workspace navigation and
  returns focus to its invoker. Native toggle state is synchronized through the
  component event path; React's general DOM event reference documents the
  [`onToggle` event prop](https://react.dev/reference/react-dom/components/common).
- Compact runtime text is hidden at widths at or below 870 px instead of
  overflowing. The attachment section has its own root so generic composer
  `div` rules do not compress its buttons.

These changes preserve the existing app structure, composer, navigation,
Library deletion flow, and scoped learning controls.

## Accepted final observations

All accepted captures below were saved and reopened before inclusion.

1. **Short dark, first choices visible.**
   [`07-short-dark-final.jpg`](../../test-results/chat-workspace-audit-2026-10-15/07-short-dark-final.jpg)
   shows the final opaque attachment surface in the short dark layout with its
   first choices visible and no parent-scroller clipping.
2. **Short dark, final choices reachable.**
   [`08-short-dark-bottom.jpg`](../../test-results/chat-workspace-audit-2026-10-15/08-short-dark-bottom.jpg)
   records actual popup scrolling to the final folder/video choices. This is
   evidence that the menu's own scroll range reaches the bottom, not a claim
   about file selection or ingestion.
3. **Keyboard focus.**
   [`09-final-dark-focus.jpg`](../../test-results/chat-workspace-audit-2026-10-15/09-final-dark-focus.jpg)
   records actual Tab focus in the final dark menu.
4. **Navigation closes the menu.**
   [`10-popup-navigation.jpg`](../../test-results/chat-workspace-audit-2026-10-15/10-popup-navigation.jpg)
   shows the menu absent after navigating to **Data & training**. It establishes
   navigation cleanup only; the destination page itself was not audited.
5. **Auto light readability.**
   [`11-short-auto-light-final.jpg`](../../test-results/chat-workspace-audit-2026-10-15/11-short-auto-light-final.jpg)
   shows the restored Auto light appearance with an opaque readable menu.

## Diagnostic and rejected captures

- [`01-chat-before.jpg`](../../test-results/chat-workspace-audit-2026-10-15/01-chat-before.jpg)
  and [`02-attach-menu-before.jpg`](../../test-results/chat-workspace-audit-2026-10-15/02-attach-menu-before.jpg)
  are reopened before-state diagnostics. They are useful for the original
  header/badge and menu problems, not final acceptance.
- [`03-light-corrected.jpg`](../../test-results/chat-workspace-audit-2026-10-15/03-light-corrected.jpg)
  is an intermediate light/header/focus capture from before the native top-layer
  correction.
- `04-menu-keyboard-focus.jpg` was inspected in the image-view tool only and was
  not reopened from its saved file; it is not accepted evidence.
- [`05-short-dark-menu.jpg`](../../test-results/chat-workspace-audit-2026-10-15/05-short-dark-menu.jpg)
  records the rejected parent-clipped menu and overflowing compact runtime text.
- [`06-short-dark-corrected.jpg`](../../test-results/chat-workspace-audit-2026-10-15/06-short-dark-corrected.jpg)
  records the rejected first popover iteration, where generic composer `div`
  styling compressed the buttons. The dedicated section root fixed that issue.

Rejected and intermediate screenshots are retained as diagnostic chronology;
they do not add to the accepted final set.

## Focused source evidence reported by root

- One new header-source case initially used a guessed dialog name, was corrected
  to the actual identifier, and then passed in 4 ms.
- The attachment-source case passed initially in 5 ms and its affected final
  form passed in 6 ms; those are two runs of one case, not two distinct cases.
- Two new pure placement cases passed in 5 ms.
- Total: four distinct UI/source cases. This is not a whole UI suite.
- Renderer TypeScript passed at the first popover source point. The later
  dedicated-section change compiled successfully into the Electron harness;
  the final coordinated compiler remained pending in another resource/
  portability lane at this audit point.

No tests or applications were run while writing this document.

## Strengths

- Destructive identity deletion is separated from normal Chat controls and
  retains its Library confirmation path.
- Pause remains attached to the dataset/task it actually controls.
- The UI no longer asserts unmeasured fast-learning activity.
- The attachment menu is visually opaque in the inspected dark and Auto light
  states, escapes parent clipping through the native top layer, retains an
  internal scroll range, closes on expected transitions, and returns focus.
- The compact header removes low-priority runtime text instead of allowing it to
  collide with primary controls.

## Remaining risks and limits

- The accepted walkthrough used synthetic renderer state and no preload. It did
  not verify production IPC, native picker results, attachment hashing,
  ingestion, learning, chat send, or actual backend cancellation.
- Only the inspected dark and Auto light states are evidenced. This is not an
  all-theme, all-density, high-contrast, localization, zoom, screen-reader, or
  OS-wide accessibility claim.
- The saved image dimensions do not prove behavior at every CSS viewport,
  display scale, or operating-system window-manager size.
- Native popover behavior depends on the packaged Electron/browser capability;
  packaged Windows/macOS/Linux acceptance remains separate.
- The Data & training navigation capture proves menu closure only. It is not a
  review of that workspace.
- Final coordinated compilation, broader regression coverage, and packaged
  application verification remain outside this bounded audit.
