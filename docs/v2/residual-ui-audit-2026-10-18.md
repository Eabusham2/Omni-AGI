# Residual current-App UI audit — October 18, 2026

Status: bounded residual walkthrough complete. This is a finite no-engine
appearance/interaction audit of the existing App, not a redesign or a full U1
completion claim.

## Scope and exclusions

The existing `desktop-current-main.cjs` fixture mounts the production React App
without a preload. Its library, conversation and counters are labelled design
preview data. No saved brain, model, Load, training, crawler, provider, device
permission, real tool or filesystem operation is authorized by this audit.

Already accepted Data/runtime, questions, MCP, compact-header, attachment/composer,
Library and Brain Map checks are excluded. CI owns the automatic theme/geometry
matrix. Older evidence is used only to avoid repetition, not as evidence for a
newly inspected state.

## Finite residual walkthrough

1. Learn and Fact Check empty/unavailable states and lower content. **Learn
   accepted:** 01 shows Default/dark entry and honest no-desktop-runtime notice;
   02 shows the actual expanded scope/date/robots controls and final empty Tasks
   region after scrolling; 03 records Tab focus reaching the allow-site field.
   The Learn/Add actions stay disabled. No data or source was submitted.
   **Fact Check accepted:** 04 shows its distinct claims/source inputs; 05
   reaches the final empty Tasks region. No checking request ran.
2. Time & schedules entry, lower content and unavailable controls. **Accepted
   with corrections:** 06 is the old misleading status; 07 verifies the honest
   unavailable status. 08 records the failed scroll and is not bottom proof.
   09 reaches the actual Saved schedules, Wakeup activity and final automation
   disclosure after the local-scroll fix; 10 opens that disclosure and shows
   its complete unavailable/empty state. No timer/action file was created.
3. Trace & journal actual mounted tabs and lower content. **Accepted:** 11 shows
   the selected sample trace,12 its actual lower cards/footer,13 the switched
   Inner journal tab with sample entry and distinction notice. These remain
   design-preview data, not evidence of actual neural activity or learning.
4. Imagination entry, selected controls and lower content in Classic/light.
   **Accepted with corrections:** 18 shows pale hard-coded selected Image/Auto
   text;19 verifies existing theme-ink foregrounds.20 reaches the final gallery
   row and explicit “DEMO PREVIEW · NO MODEL RAN” label, but also exposed a dark
   caption inheriting the light theme's dark text on the always-dark sample
   canvas.21 verifies the corrected canvas-local caption. No media generation ran.
5. Tools & permissions lower integration/plugin surfaces in Classic/light.
   **Accepted with corrections:** 14 is the before header-spacing state;15
   verifies the padded header.16 reaches the page bottom and demonstrates
   generic input/grid rules splitting native MCP checkboxes from their labels.
   17 verifies scoped inline checkbox labels and actual Server-id field focus.
   No checkbox was toggled, credential entered, connection attempted or host
   action run. Plugin management honestly reports unavailable without preload.
6. Forks & agents and Evolution entry/lower states. **Accepted with corrections:**
   22 records faint lineage labels/buttons in Classic/light;23 reaches the final
   import field and verifies theme-ink labels/buttons.24 records unpadded
   Evolution card headings;25 verifies card-header padding.26 reaches the actual
   final protected-evaluation, recursive checkbox, Off policy and disabled Start
   control. A single affected follow-up in31 verifies the readable success badge
   and truthful recovery-origin caption. No fork, merge, import, export, refresh
   or improvement action ran.
7. Optional Assistant/reviewer overlay. **Accepted in the disconnected state:**
   27 opens Assistant in Classic/light;28 reaches its final prompt and disabled
   Preview/Send controls.29 switches to Action reviewer;30 reaches its final
   explanation. No enable checkbox, provider, key, shared-context option or
   saved configuration was changed. The overlay closes normally. The isolated
   preview's appearance was restored to Liquid Glass + Auto (AX confirmed
   “Following the OS · currently light”) before quitting; its process exited 0.

Every accepted screenshot below was saved from the exact CUA capture and
reopened. No synthetic callback counts as a live backend result. The local
profile was `/tmp/omni-residual-ui.SmLcgz`; the existing native
`tests/e2e/desktop-current-main.cjs` preview was used at its 1180×850 window
configuration. Captures are 2360×1700 device pixels; resized display previews
are not CSS-pixel geometry measurements. The shared Vite server was left alone.

## Findings

Learn entry and its expanded lower scope form are readable in the inspected
Default/dark state, with no overlapping fields or unreachable bottom actions.
Keyboard focus reaches the lower field. This is neither a complete accessibility
claim nor evidence that learning/crawling ran.

Time entry screenshot 06 reproduced a misleading permanent “Reading host time…”
status beside the explicit unavailable desktop connection. The no-connection
branch now says host time is unavailable; it does not change clock/tool behavior
or infer a running request. Screenshot07 is the accepted affected correction.

Screenshot08 is a **rejected bottom capture**: actual wheel scrolling did not
move the Time workspace, and its lower schedules/automation controls remained
clipped. The bounded workspace grid lacked a local scroll owner. Only
`.time-workspace` now gets `min-height:0`, `overflow:auto` and overscroll
containment; the existing layout and actions are unchanged. Its corrected
lower-content correction is accepted in09. The expanded action-file panel
repeated the same stale “Reading…” copy despite its unavailable connection;
its no-connection branch is corrected in10. This second copy issue was
observed in the live accessibility state before correction, not captured in
a before PNG.

Classic/light Tools screenshot14 shows its Authority matrix heading and
description pressed against the card border. The narrow
`.permission-surface > .surface-title` selector now has card-header padding and
wrapping; other card styles and permission behavior are unchanged. Its affected
final capture is accepted in15.

Screenshot16 also reproduced a current-App composition defect outside the
already accepted MCP protocol/App fixtures: generic integration-label/input
rules made access checkboxes full-width and detached their labels. Only
`.mcp-access-controls` and `.mcp-add-scope` checkbox layout are scoped back to
native-sized inline controls. Screenshot17 accepts the corrected appearance
and focus border without changing any access choice.

Both absent-connection status fixes also consume their existing actual error
state, so a present API with a failed null-snapshot read does not perpetually
say “Reading…”. Three new pure/render presentation cases pass; they are not
evidence of a successful real clock or automation read. No owner hooks,
requests, timers or tool policies were changed.

Imagination selected Image/Auto foregrounds were hard-coded light lavender and
became hard to read on the Classic/light surfaces (18). Only the two selected
button foregrounds now use the existing theme `--text-1` token (19). The sample
canvas's label now uses a canvas-local light foreground instead of inheriting
light-theme dark ink; its fixed dark background is unchanged (21). This is a
scoped visual-readability correction, not a measured full contrast matrix.

Forks screenshot22 exposed faint hard-coded lavender labels and controls in
Classic/light. Only the small lineage tags, Review merge, LAB badge and Export
origin foregrounds now use existing theme text tokens (23). Labels, actions,
lineage data and merge semantics were not changed. Root review then selected one
real residual issue: the semantic green badge was visibly pale on light. Only
the shared `.healthy-pill`, `.lineage-pill`, `.pack-status` foreground changed
from `--green` to the existing `--success` token; dark keeps the existing green,
while light uses `#176b48`. The single affected Classic/light recapture31 was
saved and reopened. Source-token alpha compositing on an assumed white card
estimates contrast 1.742:1 before and 6.255:1 after; that is not a runtime
computed-style or complete contrast audit.

Source inspection also found the former “Origin verified” header unconditional,
and its lower “Verified” caption based only on checksum presence. That semantic
issue was reported separately before changing words. The root lane then changed
the header to “Recovery origin” and the lower caption to a stored-checksum label
or “No origin checksum recorded,” without adding or claiming verification.
Capture31 shows that final no-checksum branch. The owned follow-up preview was
restored to Liquid Glass + Auto and exited0.

Evolution screenshot24 exposed headings touching both card borders. Only the
compose/archive direct `.surface-title` headers now receive the same padding
and divider used by adjacent cards (25). Its form, routes, evaluator custody,
permissions and actions are unchanged. Actual lower content is reachable (26).

The optional Assistant/reviewer overlay was readable and independently scrollable
in its inspected Classic/light state (27–30), with unavailable-service text and
disabled saves/sends. Switching the tab changed visible labels correctly. The
native accessibility snapshot retained the prior checkbox name “Enable assistant
on this device” after the Reviewer switch although the visible text correctly
said pre-action review. The existing source uses a wrapping label with dynamic
text, not a hard-coded accessible name. This observation is retained as an
unresolved assistive-technology/cache risk, not promoted to a confirmed source
defect or screen-reader pass.

## Accepted current screenshot index

All files are under `test-results/residual-ui-audit-2026-10-18/` in this checkout.
These are current-run screenshots, not reused older audit evidence.

| Step | Exact captures | Observed scope |
| --- | --- | --- |
| 1 | `01-learn-default-dark.png`, `02-learn-scope-bottom-dark.png`, `03-learn-scope-keyboard-focus.png`, `04-fact-check-default-dark.png`, `05-fact-check-bottom-dark.png` | Entry, scope disclosure, actual lower Tasks content and one keyboard-focus transition |
| 2 | `06-time-schedules-default-dark.png`, `07-time-status-corrected.png`, `09-time-bottom-scroll-corrected.png`, `10-time-action-files-unavailable.png` | Before/corrected status, actual lower scroll and action-file disclosure |
| 3 | `11-trace-default-dark.png`, `12-trace-bottom-default-dark.png`, `13-inner-journal-default-dark.png` | Sample trace entry/bottom and journal tab |
| 4 | `18-imagination-classic-light.png`, `19-imagination-selected-labels-corrected.png`, `20-imagination-bottom-classic-light.png`, `21-imagination-caption-corrected.png` | Before/corrected selected text and lower sample gallery/caption |
| 5 | `14-tools-classic-light.png`, `15-tools-header-spacing-corrected.png`, `16-tools-integrations-bottom-classic-light.png`, `17-tools-checkboxes-focus-corrected.png` | Before/corrected header and lower inline checkbox composition/focus |
| 6 | `22-forks-agents-classic-light.png`, `23-forks-bottom-labels-corrected.png`, `24-evolution-classic-light.png`, `25-evolution-header-corrected.png`, `26-evolution-bottom-classic-light.png`, `31-forks-success-ink-corrected.png` | Entry, corrected labels/headers, actual final lower controls and one affected final badge follow-up |
| 7 | `27-assistant-overlay-classic-light.png`, `28-assistant-bottom-classic-light.png`, `29-reviewer-channel-classic-light.png`, `30-reviewer-bottom-classic-light.png` | Overlay entry, actual independent lower scroll, reviewer tab and final copy |

`08-time-bottom-default-dark.png` is deliberately **not** in the accepted list:
it is diagnostic evidence of the failed pre-fix scroll, not bottom coverage.

Representative corrected states:

![Time final lower controls](../../test-results/residual-ui-audit-2026-10-18/09-time-bottom-scroll-corrected.png)
![Tools corrected inline labels](../../test-results/residual-ui-audit-2026-10-18/17-tools-checkboxes-focus-corrected.png)
![Imagination corrected sample caption](../../test-results/residual-ui-audit-2026-10-18/21-imagination-caption-corrected.png)
![Forks corrected lower view](../../test-results/residual-ui-audit-2026-10-18/23-forks-bottom-labels-corrected.png)
![Forks readable recovery badge](../../test-results/residual-ui-audit-2026-10-18/31-forks-success-ink-corrected.png)
![Evolution corrected card headers](../../test-results/residual-ui-audit-2026-10-18/25-evolution-header-corrected.png)
![Assistant reachable lower controls](../../test-results/residual-ui-audit-2026-10-18/28-assistant-bottom-classic-light.png)

## Remaining evidence

The seven finite residual steps above are accepted only within their named
no-engine states. Real backend transitions, actual generation/learning, all
permission/error/loading states, full hover/disabled/keyboard combinations,
screen-reader operation, zoom/localization and a universal all-theme/all-platform
accessibility claim remain outside this walkthrough. Default/dark and
Classic/light are different inspected contexts, not an exhaustive theme matrix.
CI remains the owner of the automatic theme/geometry matrix; no local Playwright
run or repeated accepted Data/runtime/MCP/map/header audit was performed.

The three new `workspaceReadStatus.test.tsx` pure/render cases passed with bundled
Node24. They check disconnected, genuine initial pending and connected-read-error
copy, not real service success or hook timing. The owned-file whitespace check
passed. No model/brain was created or loaded; no data was trained, no crawler or
provider was contacted, and no permissions were widened. The preview exited 0.
