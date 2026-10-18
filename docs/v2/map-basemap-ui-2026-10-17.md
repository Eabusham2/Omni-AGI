# Optional map controls: bounded native UI check

The existing Electron application components were rendered in an isolated
Electron window using `map-basemap-fixture.tsx`. Its settings API is synthetic:
no real profile, credential, permission, provider, model or training was used.
This checks presentation and callback/state behavior, not service delivery.
The production host service has separate tiny HTTP/SQLite tests.

## Checked flow

| Step | Observed result | Health |
| --- | --- | --- |
| 1. Closed optional section | No callback until expanded; offline default remains visible. | Passed |
| 2. New keyless form | Existing theme/form controls; labels and template readable in wide/light. | Passed |
| 3. Save gating | Save disabled without review/reference; enabled after both. Synthetic save retained a disabled profile, without a provider request. | Passed |
| 4. Saved profile, compact dark | Source, revision, disabled status and actions wrap within 360px content. | Passed |
| 5. Keyed provider setup | Mapbox-specific owner/style/credential fields and contract limitation appear; no real key entered or agreement accepted. | Passed |
| 6. Bottom and keyboard | Tab reaches timing, checks and Discard; focused control, disabled Save and long review text remain readable at the compact bottom. | Passed |
| 7. Pending API-owner replacement | Replacing the API while a list is held releases busy controls. Delivering the retired reply does not show its retired profile; current Refresh logs owner2. | Passed |
| 8. Retained raster display | A real one-pixel local PNG passes SHA/dimension decoding and reaches rendered status, with provider-fit caption and required attribution. This pixel is not geographic imagery. | Passed |
| 9. Corrupt retained raster | Wrong SHA produces an explicit checksum error, not a map/provider fetch. Compact dark attribution, source details and keyboard link focus remain visible. | Passed |

## Accepted screenshots

Each exact CUA capture was saved, reopened and inspected before acceptance:

- `test-results/map-basemap-ui-2026-10-17/01-collapsed-light.png`
- `02-form-light.png`
- `03-bottom-save-light.png`
- `04-compact-dark.png`
- `06-keyboard-compact-bottom.png`
- `07-retained-pixel-provenance.png`
- `08-checksum-dark.png`

`05-mapbox-compact-bottom.png` is rejected as bottom evidence: the requested
scroll did not reach the bottom. Step6 uses the actual keyboard-focused bottom
capture instead. The initially borrowed synthetic PNG could not decode; it was
replaced with a valid CRC/zlib PNG in the fixture, then only its affected local
render/error path was retried. No production media code was changed to hide it.
An HMR update reset the isolated fixture between checks; saved earlier captures
are scoped to their visible state, not proof of state surviving a reload.

## Scope and remaining evidence

No new spacing, clipping, contrast or owner-flight defect was reproduced in
these checked states. This is not full accessibility/theme/platform acceptance,
actual map quality, provider account/license verification, IPC delivery, native
tool selection or learned map understanding. External links were not opened.
The preview window was closed through CUA; its process exited0. The fixture
server is separate from the normal application and no neural worker was started.

Root bridge checks cover one preload case and two actual ToolExecutor policy
cases. Their first run exposed a missing test-service method; after correcting
that mock, the revocation case explicitly requires entry into dispatch and the
current-policy error, avoiding a false positive from an earlier unrelated error.
Both cases passed in22ms; the preload case passed in27ms. The separate catalog
RAM helper has3 new cases passing in3ms. These checks do not establish real
provider success or a whole-product green suite.
