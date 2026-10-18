# Initial-training action review: bounded visual check

October 15, 2026. Existing production approval components and styles in a
private Electron UI fixture, using an explicitly synthetic transport. No model,
training, package script, real host operation, provider or saved instance ran.
The deferred native merged-resume brain was not created.

## Observed steps

1. **Pending, light — readable.** Two exact synthetic action proposals appear
   with separate Deny/Approve controls. Operational IDs stay under Request
   details. Full operands are available, not abbreviated summaries.
2. **360px content, dark — readable.** Text and controls wrap inside the surface;
   the app window itself was not resized to 360px. Top fixture controls are
   fixture-only, not a replacement application design.
3. **Deny, expanded details, scroll to bottom — correct.** Denying the first
   proposal removes only that proposal. The second remains, its complete long
   source path wraps, IDs expand, and the bottom controls remain reachable.
   Native Tab focus visibly marks Deny without changing its text contrast.
4. **Keyboard approval — correct.** Tab from Deny to Approve, then Return,
   approves the remaining synthetic proposal. Both pending cards and the
   waiting status disappear; an exact-action-reviewed status remains.

Each screenshot was saved and reopened before acceptance:

- `test-results/training-review-ui-2026-10-15/01-pending-light.jpg`
- `test-results/training-review-ui-2026-10-15/02-narrow-dark.jpg`
- `test-results/training-review-ui-2026-10-15/03-one-denied-details-bottom.jpg`
- `test-results/training-review-ui-2026-10-15/04-reviewed-no-pending.jpg`

All captures are 2200×1740 physical pixels. The native check window was closed
through its actual Close control and exited successfully afterward.

## Implementation and source checks

`PluginRuntimeApprovalCard` is shared with the existing Plugin Manager, keeping
the same exact operands and buttons. `RuntimeApprovalController` serializes and
coalesces pending-list reads, ignores late results after teardown, rejects
foreign-owner data, permits independent reviews without duplicate submission,
and retains genuine failed/uncertain action results after refreshing the list.

`PluginRuntimeApprovalQueue` is mounted in the existing first-Build and resumed
initialization surfaces. Navigating to Library while build progress is active
does not hide that overlay. The renderer does not create a brain or load a
worker to inspect or review these host proposals.

Four distinct new controller/presentation cases passed in scoped runs. The
new presentation first exposed a case-insensitive `.ts`/`.tsx` import collision;
the controller received a distinct basename and only the affected case reran.
A required React ref initializer was corrected before renderer compilation
passed. The later genuine-action-error case and its affected concurrent-review
case passed together; closed other cases were skipped.

## Limits

This verifies this component's synthetic native presentation and wiring, not
real initial training, parser execution, real permission grants/effects, all
themes, the full fixed-overlay stack, every keyboard/screen-reader interaction,
or packaged cross-platform acceptance. The fixture uses the production styles
but a relative test surface; whole-app overlay stacking remains a separate gate.
Screenshot readability and one visible focus state do not establish WCAG
compliance. Real runtime Stop/process release has separate evidence in
`plugin-runtime-host-evidence.md`.
