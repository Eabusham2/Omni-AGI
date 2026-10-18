# Exact saved-copy review UI check — October17

Scope: existing `InheritedTaskFilesReview` with the new user-approved saved-copy
exception while general file access is Off. Native Electron displays the real
component, existing styles and synthetic owner/file data. Callbacks update an
explicitly simulated receipt only. No file bytes, model, permission, provider,
native resume, training or packaged application were exercised.

All accepted screenshots below were saved and reopened through computer use.
The1000×850 window uses a360px compact content container for narrow checks;
this is not a claim that every screen was tested in a360px-wide window.

| Step | Result | Accepted capture |
| --- | --- | --- |
|1. Off review, light | Primary approval remains available; copy-only scope and original-path denial are readable; long name fits. | `test-results/saved-file-review-ui-2026-10-17/03-styled-off-light.png` |
|2. Compact dark | Filename and controls wrap without overlap; approval and Leave paused remain visible. | `test-results/saved-file-review-ui-2026-10-17/04-compact-dark.png` |
|3. Busy and disclosure, bottom | Expanded access identity wraps; actual bottom/receipt visible after scrolling; both actions disabled while verifying. | `test-results/saved-file-review-ui-2026-10-17/05-busy-dark-bottom.png` |
|4. Keyboard | Tab from access disclosure reaches the visible primary focus ring. Return produces the synthetic approval receipt in the accessibility state. | `test-results/saved-file-review-ui-2026-10-17/06-keyboard-approve-focus.png` |
|5. Leave paused | Approval is removed; only Review again remains. Synthetic prior receipt remains visibly labeled. | `test-results/saved-file-review-ui-2026-10-17/07-paused-no-approval.png` |

Captures01/02 are rejected as styled-product evidence: the new isolated fixture
initially omitted `pluginManager.css`/`maps.css`. The fixture imports were
corrected to match the existing app; production CSS was not changed to make a
fixture pass. The accepted03–07 captures are from the corrected source.

The inspected states have readable hierarchy, wrapped identifiers, separated
buttons and visible keyboard focus. No new visual defect was confirmed in
this bounded flow. Screen-reader behavior, zoom/forced colors, every theme,
actual host review+resume and all packaged platforms remain unverified. The
separate real-files/SQLite/ToolExecutor tests establish only their host scope.

Owned preview session22643 was closed through computer use and exited0. The
existing Vite5189 server was reused; no second engine process was launched.
