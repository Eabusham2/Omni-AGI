# Portable workspace — bounded component UI check

The production `PortableWorkspaceControls` was inspected in native Electron,
using a clearly labeled synthetic typed API and the existing application
styles. No real picker, filesystem operation, preload/IPC, Windows runtime,
brain, Load, model, training or provider ran.

1. **Preparation entry — healthy:** only Prepare was offered before destination
   review. Its callback changed the state to Preparing and removed premature
   Review/Approve controls. Capture `01-preparation-required.jpg` records entry;
   the Preparing transition was checked in the accessibility tree.
2. **Blocked fit — healthy:** RAM, shared pool and context values were readable;
   the shortage was visible and Approve disabled. Warnings and exact review
   details expanded without covering other controls. See `02-blocked-review.jpg`.
3. **Allowed fit and keyboard approval — healthy:** after a new synthetic fit
   review, Tab reached Approve and Return invoked it. Ready appeared without
   any automatic Load. The fixture log explicitly records no runtime Load.
4. **Selected copy and error recovery — healthy:** the action was disabled until
   an instance was selected. Success invoked the matching created callback;
   a cancelled picker result stayed quiet; an error remained inline and bounded.
   Refresh cleared that error. See `03-ready-create-error-light-compact.jpg`
   for bottom receipts and `04-ready-copy-light-compact.jpg` for Ready/copy/error.
5. **Compact light and bottom scroll — healthy for this surface:** native
   Top Left resizing produced 1444×962-pixel captures, not a claimed CSS
   viewport. The actual component's controls/text remained readable and its
   bottom could be reached. This does not inspect the entire Library layout.

All four screenshots were saved and reopened under
`test-results/portable-workspace-audit-2026-10-15/` before acceptance. The owned
window closed normally and the fixture process exited 0.

Three new source component cases passed (43 ms): API-owner state clearing and
stale status/create ACK fencing; preparation/review/approval states; quiet
cancellation/current error. They use injected React hooks, not a real backend.
The production owner-change effect now clears old status/review/error as well
as busy state, so another API owner cannot inherit a stale fit display.

This is not clean-Windows portable startup, a package smoke test, actual
resource-fit measurement, native continuation or full accessibility/all-theme
acceptance. Those remain separate from the fourteen scoped portable host/file
cases recorded in `windows-portable-workspace.md`.
