# Build timing, output details, and plugin themes

## Scope

October 13 source and native Electron checks of the existing production
`BuildProgressStatus`, `StorageOperationTiming`, `ResponseOutputCounter`,
`PluginThemeHost`, and `PluginThemeChooser` components. The isolated fixture
supplies labeled synthetic events/catalog data. It does not create, load, train,
quiz or replace a brain, call a provider, execute a tool, or access private files.
The user's new-brain merged-resume deferral remains in force.

## Implemented contract

- Build duration comes from producer-measured monotonic time. Weighted stage
  percentages are not used to invent an ETA. The renderer advances an explicit
  remaining-time sample and shows the current estimating phase when it expires
  or is absent. Estimated total appears only with elapsed plus an authoritative
  ETA; completed total is actual measured elapsed.
- Engine core completion maps to host readiness at 99%. Only the awaited host
  initialization coordinator's saved-ready result publishes complete/100%.
  This applies to create, retry and startup recovery. Initial-material progress
  remains separate and cannot impersonate that durable host completion.
- Storage timing uses a monotonic clock and freezes renderer extrapolation for
  paused/terminal events. Unknown/finalizing estimates are readable text, not a
  bare dash or a clipped essential message.
- Basic Build no longer displays the changed-checksum box. Diagnostic/checksum
  data remains available internally and the authoritative parameter total stays
  in the runtime inspector.
- The reply counter keeps compact output/TPS text and exposes its exact raw
  value through `title` and its accessible name. This display does not generate
  output tokens or alter inference cadence.
- Plugin themes now have explicit Apply/Clear controls and a mounted Studio
  consumer. Selection is local to the installation and binds plugin ID, exact
  immutable package hash and theme ID. Catalog disable/version mismatch restores
  prior owned inline values; late async replies cannot apply a stale selection.
  These are data-only tokens, not arbitrary executable feature modules.

## Focused source evidence

- Four original pure timing cases plus one new estimated-total case passed;
  one storage backward-wall-clock regression passed.
- One new constructor-free host terminal adapter case passed (2ms).
- Three rendered component/source checks passed on final root changes (10ms).
- The directly affected existing keyboard/counter source case passed (3ms),
  twelve unrelated cases skipped. Its obsolete verbose-composer assertions were
  updated to the already accepted compact input counter and submit component;
  production UI was not reverted to satisfy the old test.
- Plugin theme helper/controller/render tests: four passed (9ms).
- React review: fixed component boundaries, native controls, cleaned event and
  interval subscriptions, versioned local state, async owner fencing. No new
  framework, app shell or browser UI was substituted.

## Native visual evidence

Saved exact captures are under `test-results/basic-controls-ui-2026-10-13/`.
Each was reopened before accepting the noted evidence. Window was 1100×1000
logical pixels; the narrow **content surface** was 360px, not a physical 360px
application window. Fixture-only sections use production styles with static
placement and their own scrolling container, so full-app overlay/focus/stacking
is not proved here.

1. `01-initial-light.jpg`: initial 99%/estimating and compact output visible.
   Found cramped timing columns; fixed auto-fit and container-relative width.
2. `02-light-narrow.jpg`: timing wraps; found storage remaining text ellipsis.
   Fixed only timing cells to wrap instead of clipping essential status.
3. `03-narrow-estimate.jpg`: measured remaining and estimated total readable;
   storage finalizing text wraps after the fix.
4. `04-dark-durable-paused.jpg`: supplied durable-complete event shows 100%,
   total and completion label; paused timing readable, no countdown.
5. `05-theme-applied.jpg`: actual Apply control visibly selects the hash-bound
   inert theme and changes Studio tokens, with Clear available.
6. `06-disabled-theme-restored.jpg`: disabling synthetic catalog entry restores
   Studio appearance and disables Apply with a readable reason.
7. `07-cleared-narrow-lower.jpg`: Clear removes selection; lower theme controls
   and entire disabled explanation inspected after scrolling down.
8. `08-output-target.jpg`: compact counter and exact accessible target observed;
   an OS hover bubble was not captured, so this is not visual bubble proof.

## Limits

This is bounded component UI evidence, not actual initial training, storage
operation execution, current merged-native resume, full-app/all-theme/accessibility
qualification, paid-provider acceptance or packaged/release acceptance. The first
two captures retain their pre-fix defects rather than being counted as post-fix
passes. Arbitrary plugin feature modules, source MIME selection, their host
integration and cross-platform packages remain separate current work.
