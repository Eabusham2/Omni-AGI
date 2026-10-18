# MCP account controls — bounded visual check

October 8. Product Design audit method: inspect the actual product component,
capture the visible state, compare accessibility/behavior and fix observed
issues. The in-app browser displayed `tests/e2e/mcp-auth-fixture.html`, using
real `McpAuthenticationPanel` and application styles with explicitly labeled
synthetic authorization. No neural model, account or external service was used.

1. **Entry, dark, narrow: accepted.** Optional client configuration is collapsed;
   the scope and Sign in action are readable. Screenshot
   `test-results/mcp-auth-manual/01-required-dark.jpg`.
2. **Expanded configuration and validation: accepted after correction.** An
   invalid port disables login. Added a visible alert explaining the allowed
   port range/automatic alternative, rather than leaving only a disabled button.
   `02-invalid-port-dark.jpg` records this target while scrolled to the fields.
3. **Pending login, wide light: failed then fixed.** The pre-fix shared `.button`
   hover style rendered white text on a pale surface (`03-authorizing-light.jpg`).
   Changed the shared hover fill/text/border to theme tokens. The affected
   button now has measured `rgb(23,21,28)` foreground on a light tinted surface,
   and Stop is clearly readable (`04-pending-light-fixed.jpg`). Stop returned
   the fixture to its required-login state.
4. **Signed in, light, 390px: accepted.** Session-only storage is visible;
   sign-in/disconnect wrap without overlapping. Page horizontal overflow was
   zero. Disconnect returned to Not signed in. Screenshot
   `05-authorized-light-narrow.jpg`. The temporary viewport override was reset.

Labels, expanded disclosure, keyboard-focused inputs, visible validation and
status announcements have source/accessibility evidence. This is not a full
WCAG audit, all-palette app audit, or a real provider consent-flow test. The
overall app, disabled/hover states beyond this component, and other layouts
remain in the full v2 visual acceptance gate. Before-fix screenshot 03 is
failure evidence, not an accepted final UI.

## October 11 credential replacement continuation

The existing isolated Electron window displayed the real `McpCredentialPanel`
and account component using labeled synthetic transport. No real credentials,
account login, subprocess or remote request were used.

1. **Local replacement validation: passed.** Non-text JSON values display a
   concise validation message. Valid synthetic variables save, clear the input,
   and update only the visible configured variable names. Stored values are
   never loaded into the field.
2. **Clear confirmation: passed.** Keep credentials dismisses confirmation;
   Clear now produces a cleared acknowledgement and an empty-variable receipt.
   The inspected 320px light surface wraps the disclosure, explanation and
   actions without clipping. Saved/inspected screenshot:
   `test-results/mcp-auth-manual/06-credential-confirmation-narrow-light.png`.
3. **Remote replacement: passed.** The synthetic bearer input is masked and
   emptied after success; account status changes to Not signed in rather than
   retaining stale account-login status. The message requires explicit Refresh
   to reconnect. Receipt output contains only the operation, keys, value-present
   boolean and `remoteEffects:false`, not the synthetic value. Saved/inspected
   screenshot: `07-remote-replacement-narrow-light.png`.

These are component/UI checks, separate from the real service/client transaction
fixtures. Production provider interoperability and full application theme,
keyboard, screen-reader and overlay coverage remain open. The older October 8
JPGs are recorded historical evidence; those files were absent in the current
checkout and were not used as October 11 screenshot evidence or recreated.
