# Inline chat imagination visual evidence

This is bounded presentation evidence for R028. The fixture imports the actual
`ChatImaginationMedia` component and supplies one clearly labelled synthetic,
validated image action. It does not run a brain, decoder, provider, file tool,
device, permission, or media generator.

## Accepted observations

Root operated the fixture through its visible React controls rather than DOM
mutation:

- `test-results/chat-imagination-inline-ui-manual/02-light-inline-stable.jpg`
  shows the settled 360px light view with media visible while action audit is
  collapsed.
- `test-results/chat-imagination-inline-ui-manual/03-dark-inline-stable.jpg`
  shows the corresponding settled narrow dark view.
- `test-results/chat-imagination-inline-ui-manual/04-light-desktop-audit-open.jpg`
  shows the 960px light view with the audit explicitly expanded beneath the
  still-visible media.

At 360px, the measured document width was 360px, media surface 273px, and
image 251px. In the desktop expanded state there was exactly one media image
outside the audit and zero images inside the audit. This supports the narrow
structural claim: validated imagination media is a chronological sibling of,
not a child hidden inside, the default-collapsed audit.

`01-dark-inline-collapsed.jpg` captured a stale responsive transition and had
an incorrect filename/theme relationship. It is diagnostic only and is not
accepted evidence of current layout or clipping.

## Evidence boundary

These observations do not establish actual model generation, decoder quality,
audio/video playback, APNG timing, retention, provider behavior, packaged-app
behavior, or full-theme accessibility. The later
`ephemeral-spontaneous` retention-marker case is source-tested separately; the
screenshots are not evidence for its Temporary/Record copy.
