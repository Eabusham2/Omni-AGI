# Bounded desktop appearance and library audit — October 13

The existing Electron renderer was compiled and opened through the existing
`tests/e2e/responsive-main.cjs` harness, using the isolated user-data directory
`/private/tmp/omni-v2-interface-audit.UKuTRG`. That harness has no preload or
neural/provider connection. Its named library entries and numbers are explicitly
labelled interface fixtures, not recreated saved brains or learned-state proof.
No model, training, provider/key, user brain or training folder was accessed.

All UI interactions used computer use on the actual native Electron window.
The existing interface, layouts, profiles, palettes and controls were retained;
no browser replacement, new application or redesign was created. Exact saved
JPEGs were reopened before using them as evidence.

## Flow and findings

1. **Library entry — healthy within this view.** `01-library.jpg` shows the
   fresh Apple Liquid Glass/Auto-light default and the existing desktop library.
   This is not evidence that its synthetic counters describe a real brain.
2. **Appearance top and bottom — inspected, with a confirmed surface issue.**
   `02-appearance-auto-light.jpg`, `03-appearance-lower-light.jpg` and
   `04-appearance-lower-dark.jpg` show explicit mode changes and scrolling to
   layout/notification controls. These are pre-fix diagnostics, not a final
   all-theme pass: underlying page text was later confirmed through the panel.
   No OS theme or notification permission was changed.
3. **Library status — confirmed misleading copy, corrected.**
   `05-library-status-diagnostic.jpg` is an intermediate scroll and does not
   establish the footer. `06-library-footer-diagnostic.jpg` clearly shows
   “Compute is ready” while the fixture also says no trained brain exists.
   Source also showed “Engine online” based only on preload presence, not a
   worker receipt. `libraryRuntimePresentation` now labels previews honestly
   and uses neutral “Local workspace” copy for production, without claiming
   a loaded/online worker or measured hardware. Two new pure presentation
   cases passed in 3 ms; renderer TypeScript and desktop compilation passed.
   `07-library-footer-corrected.jpg` verifies the preview branch in Electron.
   The production branch has source/presentation evidence, not a real worker
   availability test.
4. **Narrower desktop window — responsive reflow observed.** Native Window
   sizing produced `08-window-sizing-check.jpg`, with the library reflowing
   from four cards across to two. Saved captures are 2880×1740 and 2080×1740
   image pixels respectively; these are not claimed CSS viewport dimensions
   or phone-size coverage.
5. **Appearance readability — failure then targeted correction.**
   `09-classic-light-narrower.jpg` and `10-colorful-dark-diagnostic.jpg` show
   legible library text bleeding through the popup. Only `.appearance-panel`
   changed: existing `--surface-solid`, paint isolation and no backdrop blur.
   Dimensions, padding, layout, palettes, radii, responsive rules, global
   materials and Apple default were not changed. Recompiled native views
   `11-classic-light-corrected.jpg` and `12-colorful-dark-corrected.jpg` verify
   that underlying text no longer shows through, including the selected and
   pointer/focus states visible in those frames. This is not every independent
   hover/keyboard/disabled interaction.
6. **Default and lower content — restored and healthy in this view.**
   `13-glass-auto-lower-corrected.jpg` confirms Liquid Glass plus Auto, reading
   the current light OS appearance, and the corrected panel's lower layout
   and notification controls. Dynamic OS-theme switching was not exercised.
   The exact isolated preview window was then closed; no saved brain was
   deleted or unloaded. The user-data scratch directory remains available.

Screenshots are under `test-results/desktop-ui-audit-2026-10-13/`.

## Verification boundary

Desktop code compilation succeeded after each affected source change. The
existing static/dynamic import chunking warning is non-blocking and unchanged;
this is not a startup-speed measurement. No old theme/model suites were rerun.
The temporary app had no backend and cannot establish live IPC, permissions,
training, learning, generation, native capability, package or release success.

Still open: the rest of the requested workspaces, Default and remaining
theme/layout combinations, smaller/shorter viewports, overlays and all scroll
positions, distinct hover/focus/disabled/error states, screen-reader and zoom
behavior, real chat Queue/Steer/Stop/off-page work, and required packaged
platform acceptance. These bounded corrections do not close the full UI audit
or the v2 goal.
