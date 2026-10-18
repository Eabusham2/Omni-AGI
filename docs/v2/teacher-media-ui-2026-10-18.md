# Media teacher choices — focused UI audit

Scope: the new output choices in the existing `ApiTeacherPanel`, using the
existing isolated provider fixture and actual application styles. The fixture
provides synthetic settings/model-list callbacks. No credential, provider,
brain, neural training or paid request was used. Earlier discovery/job-control
checks were not repeated.

The1100×900 Electron window uses the fixture's380px compact content container.
All accepted screenshots were saved and reopened through computer use under
`test-results/teacher-media-ui-2026-10-18/`.

| Step | Result | Accepted capture |
| --- | --- | --- |
|1. Supported choices | Gemini shows image/audio/video choices. All can be selected; Tab and Space activate Video with a visible focus indicator. Compact light labels stay beside their boxes and the bottom action remains visible. |02-fixed-compact-light.png |
|2. Theme | The same checked choices and bottom action remain readable in compact dark. No overlap or clipped choice labels observed. |03-fixed-compact-dark.png |
|3. Provider change | Local Ollama removes unsupported media choices and the key field; only a synthetic metadata-list callback occurs. No model or lesson is started. |04-local-provider-no-media-claims.png |

One real visual defect was found: the generic integration input/label rules
made checkboxes full-width and put their labels on separate rows. Scoped media
fieldset rules now keep native checkboxes and labels together, preserve theme
colors and wrap the legend. The first correction needed higher specificity
than the later generic integration rule; only the affected visible states
were revisited. Capture01 shows the original defect and is retained as failed
evidence, not a passing styled state.

This closes the focused new-choice inspection. It does not establish provider
capability, account/model compatibility, native media learning, every theme,
physical small-window layout or packaged-platform acceptance. The separate
source/storage tests and provider-contract fixtures retain their own limits.
Owned preview33847 was closed through computer use and exited0.
