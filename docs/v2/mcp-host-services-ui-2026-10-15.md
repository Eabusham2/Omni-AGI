# MCP host-services panel — bounded Electron UI evidence

Date: October 15, 2026.

## Audit scope

This combined UX/accessibility check rendered the production
`McpHostServicesPanel` inside a fixture-only Electron window. The fixture used
typed in-memory `McpRootsApi` and `McpDiagnosticsApi` implementations. It had no
preload, native picker, filesystem grant, MCP process, network, provider, brain,
worker, model, training, or user-file selection.

Fixture sources:

- `tests/e2e/mcp-host-services-fixture.html`
- `tests/e2e/mcp-host-services-fixture.tsx`
- `tests/e2e/mcp-host-services-main.cjs`

The existing Vite server on port 5189 served current renderer source. Electron
used the isolated profile `/tmp/omni-mcp-host-ui.FI4XkX`. Every accepted image
was captured from the owned Electron window, saved locally, and reopened before
acceptance. All owned fixture windows exited with code 0.

## Numbered flow

1. **Collapsed, no calls — healthy.** The final post-fix render keeps the panel
   collapsed and shows roots 0, diagnostics 0, picker 0, and network 0. Merely
   mounting the production component does not invoke either API.
   [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/post-fix/01-final-collapsed-zero-calls-light.png).
2. **Explicit open and selection — healthy after fix.** Opening the disclosure
   invokes one status read. Choosing folders invokes only the synthetic typed
   callback and produces `Selected; fresh disclosure review required`; the
   fixture records that no native picker opened. The long name and URI now
   occupy separate lines and wrap without horizontal clipping.
   [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/05-fixed-owner-and-uri-light.png).
3. **Review to shared — healthy.** Review moves the selected root to `Shared
   with the current connection`; it does not imply file access or a sandbox.
   [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/post-fix/shared/01-final-shared-long-uri-light.png).
4. **Clear — healthy.** Clear returns to `No filesystem roots shared` and
   disables Review/Clear while leaving explicit Choose/Refresh/Diagnostics
   controls available.
5. **Stale owner and overlapping flight — healthy after fix.** A delayed
   selection from API owner 1 was held, the API object changed, and a second
   delayed selection started under owner 2. Releasing owner 1 did not display
   its result or clear owner 2's busy state. Releasing owner 2 alone published
   the current review-required status.
6. **Wrong-owner refusal — healthy.** A response with another brain ID was not
   adopted; the component displayed `Filesystem disclosure returned another
   source connection.` The message remains readable on the compact dark surface.
   [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/13-compact-dark-wrong-owner-final.png).
7. **Diagnostic page one — healthy.** The panel displays accepted/rejected/
   rate-limited/evicted counters, expandable records and the exact untrusted
   synthetic payload. The long JSON URI wraps rather than forcing horizontal
   scrolling.
   [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/06-diagnostics-page-one-light.png).
8. **Next diagnostic page — healthy.** `Next log page` requests the declared
   cursor and replaces the viewport with the second bounded page; the final page
   has no false Next control.
   [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/07-diagnostics-page-two-light.png).
9. **Compact dark reflow and bottom — healthy.** A 360px-wide content surface
   inside the 1100px native test window reflows the controls and diagnostic
   summary without horizontal clipping. Scrolling reaches the complete fixture
   log and zero-effect footer.
   [Top](../../test-results/mcp-host-services-ui-2026-10-15/08-compact-dark-top.png) ·
   [bottom](../../test-results/mcp-host-services-ui-2026-10-15/09-compact-dark-bottom.png).
10. **Keyboard focus — healthy.** Native Tab navigation reaches the disclosure
    summary and shows a visible focus outline.
    [Accepted screenshot](../../test-results/mcp-host-services-ui-2026-10-15/10-compact-dark-keyboard-focus.png).

## Findings and repairs

The audit found two production defects and root repaired both during this run:

1. Replacing `roots`/`diagnostics` during an in-flight request retired the old
   result but left `busy` true, permanently disabling Choose, Refresh, and Logs.
   The panel now retires the owner flight and uses a per-flight object token, so
   an old `finally` cannot clear or strand a newer request.
2. A root name and URI were rendered inline as `namefile:///…`. The URI is now a
   separate block while retaining `overflow-wrap:anywhere`.

The pre-fix evidence remains diagnostic, not acceptance:

- `02-review-required-long-uri-light.png` and `03-shared-light.png` show the old
  inline URI presentation.
- `04-stale-owner-stuck-before-fix.png` shows every production action disabled
  after the stale response returned.
- `11-rejected-wrong-scroll-position.png` did not show its intended refusal
  message and was rejected; the later accepted refusal capture replaces it.

No additional defect was observed in the bounded flow after those repairs.

## Strengths

- The collapsed default avoids background disclosure or diagnostic reads.
- Copy explains that roots are names/URIs only and do not grant access or
  sandbox a local server.
- Selection, review, shared state and clear are distinct visible steps.
- Current owner checks refuse foreign responses, and the fixed async fencing
  survives an overlapping replacement request.
- Diagnostic retention is explicitly bounded/current-process and paged.
- Long untrusted values wrap in both root and JSON presentations.
- Native summary/button semantics provide a usable keyboard path and visible
  focus.

## Evidence limits

This is production-component presentation and callback evidence only. It does
not prove preload/IPC composition, a real native folder picker, root URI
canonicalization, file access, notification delivery to a live MCP server,
diagnostic redaction against real secrets, authentication, provider/account
interoperability, packaged Windows/macOS/Linux behavior, screen-reader output,
or full WCAG compliance. The compact result is a measured 360px **content
surface**, not a claim that the native window itself was resized to 360px. Only
the shown light and dark states were inspected; this is not an all-theme audit.
