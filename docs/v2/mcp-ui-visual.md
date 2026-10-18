# Connected-app reviews and real Electron App sandbox

October11 bounded native-computer-use continuation. These checks complement,
not replace, the MCP owner's protocol/permission fixtures.

## Review controls

The actual `McpInteractionPanel` ran against explicitly labeled synthetic
requests in `tests/e2e/mcp-review-fixture.html`. No model, browser navigation,
real credential, service call or external effect ran in that fixture.

1. **Instance filtering: passed.** Four selected-instance records were shown;
   an unrelated instance's pending request stayed excluded.
2. **Typed form: passed.** A rejected one-selection response displayed the
   transport validation error and retained the form. Selecting two values and
   accepting removed just that completed request. This tests UI error handling;
   production schema validation has separate host tests.
3. **Cancel and URL decline: passed.** Cancellation and declining the full,
   visibly punycode destination removed their own records, with no browser
   opening and no accepted content. The original form was not accepted again
   after the connection interruption; only its separate cancellation path was
   exercised in the reopened fixture.
4. **Native input/output review: passed.** Edited task text remained visible;
   explicitly allowing it removed only its input request. Editing the response
   before sharing sent that edited value through the callback. Server persona
   text was visibly labeled omitted, not inserted in the reviewed task messages.
   This fixture did not perform neural inference; the actual native text/image/
   audio implementation has separate evidence in native-mcp-sampling.md.
5. **Presentation: corrected and inspected.** Initial controls used square
   native gray styling; the owner applied the application's themed buttons and
   fields. The accepted320px light surface wraps text/actions, preserves visible
   focus and exposes the lower controls by scrolling. Before/final captures:
   `test-results/mcp-review-manual/01-form-before-polish-dark.png` and
   `02-reviewed-output-narrow-light.png`. Each saved capture was opened and
   inspected; the before capture is not final acceptance.

## Real App host and double sandbox

The isolated Electron entry `mcp-app-host-main.ts` uses the real `McpAppService`,
custom protocol, CSP documents, request/origin/frame helpers, trusted-main-frame
IPC check, and actual `McpAppView`. Only the original completed call, its HTML
resource and final tool dispatch are synthetic. No brain is loaded or trained;
this is not a real-provider or production ToolExecutor permission check.

6. **Sandbox/handshake: passed on this Mac/Electron runtime.** Explicit Open
   loads the real proxy and opaque inner frame. Initial Connecting resolves to
   an actual initialized handshake. The authored probe evaluates `typeof`
   inside that frame and reports Node `require`/`process` and Studio bridge all
   undefined. An undeclared fetch is blocked by the configured sandbox network
   boundary. The App remains visible and operable without receiving a preload
   bridge. This is bounded evidence, not a universal security certification.
7. **RPC deduplication and explicit Close/reopen: passed.** Two clicks issuing
   the same RPC ID/body retained dispatch count1. Close removes the frame;
   explicit reopening rereads its UI and completes a new handshake without
   rerunning the original captured tool call. The synthetic tool is only called
   by its separate button, not automatically during reopening. Capture:
   `test-results/mcp-app-manual/01-native-sandbox-and-dedup-dark.png`.
8. **Fresh per-open handles: affected live smoke passed.** After durable
   artifact IDs were separated from active view handles, the actual Electron
   fixture was rebuilt as a tiny shell only. Open/Close/reopen initialized
   two different protocol origins (`2d6b…` then `0995…`) for the same artifact.
   The host trace contains only UI resource reads and initialize notifications;
   no synthetic tool dispatch ran during this affected check. The earlier RPC
   test was not repeated. Capture:
   `test-results/mcp-app-manual/02-fresh-handle-reopen-dark.png`.

## Product integration and remaining gates

- Run mounts reviews persistently outside ordinary task questions and inert
  off-page regions. While requests exist, the workspace gives them an inline,
  capped scroll area instead of covering the chat with an overlay.
- Completed matching MCP action cards mount `McpAppView` from an opaque host
  hint; the main-owned artifact/view lookup remains authoritative. Six scoped
  presentation cases pass, and renderer TypeScript passed after mounting.
  Copied artifacts use a separately verified current-owner projection without
  rewriting original execution output or opaque user IDs. The MCP owner's
  durable artifact/real host fixtures have separate evidence; the UI smoke
  above is not itself a restart/fork/merge proof.
- Current scope does not prove full production chat/App placement across every
  theme/layout, real provider interoperability, full UI-driven portability,
  newly added permission-review variants, complete keyboard/screen-
  reader accessibility, or all-platform sandbox behavior. Those gates remain.
- Browser/App source is untrusted task data. No timeout grants permission,
  hidden persona prompt, foreign model, completed-effect replay, or claim of
  broad learned capability was introduced by these UI checks.
