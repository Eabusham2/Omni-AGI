# Remote task controls — bounded Electron UI audit

The existing production `McpTasksPanel`, `McpTaskActionView` and styles were
inspected through computer use in isolated Electron fixtures on October17–18.
All callbacks were synthetic and labeled in the visible screen. No MCP server,
remote account, credential, permission grant, brain, neural learning or external
effect was exercised. Host/protocol/storage evidence is recorded separately in
mcp-durable-tasks.md and mcp-task-portability.md.

Each accepted screenshot was saved and reopened before acceptance. Paths below
are under `test-results/mcp-tasks-ui-2026-10-17/`. The fixture window is1100×860;
compact checks use a360px content container inside it. They do not establish a
360px physical-window, all-theme, forced-colors or screen-reader pass.

| Step | Observed result and health | Accepted capture |
| --- | --- | --- |
|1. Closed history | Zero local-page/dispatch callbacks before expansion. Chat ACK says remote work in progress. Healthy. |01-local-zero-start.png |
|2. Local page and Ask | Expansion reads one local page; remote result uses normal exact Ask. Decline enables the original controls and records no remote request. Healthy. |02-ask-result.png |
|3. Stop waiting | Approved held result exposes local Stop. Stop retains an unknown request outcome and current status/result controls; visible text states remote work was not cancelled. Healthy. |03-stop-local-only.png |
|4. Result | Released typed result shows source receipt and received data; unrelated lost-ACK history stays unknown. Healthy after fixture correction below. |04-result-and-unknown-custody.png |
|5. Inherited page | Next/Previous page work; inherited terminal history retains its source owner and offers no management controls. Healthy. |05-inherited-no-management.png |
|6. Retired API | API owner changes while metadata waits; old reply is released and does not repopulate rows. Refresh remains available. Healthy. |06-retired-api-reply.png |
|7. Selected brain | A held result from the previous brain is released after selection changes. Its result card is not adopted by the newly selected brain; the fresh page stays working. Healthy. |13-retired-brain-result.png, with exact accessibility-state receipts |
|8. Light/keyboard | Compact light controls and long source JSON wrap. Shift+Tab focuses Cancel remote task; Return opens its specific Ask. Healthy. |08-light-keyboard-cancel.png |
|9. Remote cancel | Exact approval yields the server cancellation status and removes another Cancel control. One confusing acceptance paragraph was corrected to terminal-status wording, then visually checked. Healthy after correction. |10-corrected-cancel-receipt.png |
|10. Failure | Typed RPC failure shows Remote task failed, original result receipt and returned error; source history stays distinct. Full card and controls fit in compact light. Healthy. |12-full-failure-card.png |

The native accessibility state also confirmed normal approval/decline labels,
busy-disabled actions, separate local Stop, unchanged original tool/action
identity, retained final-result metadata, and current selected brain/API IDs.
No duplicate network effect or actual result correctness is inferred from those
synthetic counters.

## Corrections and rejected evidence

The isolated lost-ACK fixture row originally spread a different task's later
`result`/`lastOperation` fields. Clearing those fields fixed that fixture data;
production validators were not weakened. Only the affected result display was
revisited after its refresh. An initial stale accessibility index failed after
that refresh; a new full state was obtained before further actions.

Capture07 is dark, lower-content-only, despite its old light filename; it is
not accepted as the light/control proof. Capture09 predates the corrected
cancel paragraph;10 replaces it. Capture11 clips the failure card's upper
context;12 replaces it. These files remain for audit history. Earlier captures
still support their unchanged specific states.

The terminal paragraph now follows actual `cancelled`, `failed`, `completed`,
`input_required` or saved-result state instead of always describing initial
acceptance. The generic action execution receipt continues to preserve its own
completion/failure independently from remote task status.

## Limits and closure

This closes the bounded new-control audit above. The wider U1 matrix, real
provider/session interoperability, actual main/preload-to-server delivery,
native learned use, platform packages and assistive-technology behavior remain
unverified. No new product redesign resulted from this audit. The owned final
preview session47225 was closed through computer use and exited0. The old
preview28040 and Vite server had already disappeared after interruption; only
the remaining UI states were resumed in the new isolated profile.
