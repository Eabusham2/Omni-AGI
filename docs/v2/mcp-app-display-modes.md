# MCP App display modes

Source added October 18, 2026, using the stable
[Apps 2026-01-26 display contract](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx#display-modes).
This extends the existing Electron `McpAppView`; it does not create another app
or browser window.

## Negotiation and actual presentation

The installed presentation relay advertises `inline` and `fullscreen` in
`hostContext.availableDisplayModes`. Without a relay, the host advertises only
inline. The View's declared `appCapabilities.availableDisplayModes` is validated
at initialization. An unavailable mode, including PIP, returns the current
mode; missing declarations do not silently acquire fullscreen support.

`ui/request-display-mode` retains the existing ready-view, exact owner, private
view token, connection policy, cancellation and request-ID deduplication checks.
Main sends a scoped presentation command with a unique operation identity.
The trusted Studio renderer acknowledges the mode actually displayed before
the original request resolves. A refused dialog operation returns inline.
The host sends `ui/notifications/host-context-changed` with the resulting mode.
One live App presentation can occupy fullscreen at a time.

The App surface is a native HTML dialog in the same renderer document. Entering
the top layer and returning inline preserve the existing iframe DOM node,
resource URL, token, App state and connection. The host header remains visible
with Return to chat and Close. Escape returns an App that supports inline to
the chat; a view without an inline return path can be closed. Host Expand is
offered only after a successful handshake declaring both modes. Main remains
responsible for requests made directly by the App.

No native OS fullscreen, PIP window, raw camera/microphone/location/clipboard
grant, App-exposed execution tool or alternate Tasks protocol is added. Existing
tool, resource, data, network and link permissions remain their own boundaries.
No presentation request runs a model, learns content or replays a completed
server operation.

## Ownership and interruption

Commands and context events carry exact brain/view/token identity and a
monotonic host sequence. Old or foreign events cannot alter a replacement view.
Acknowledgements require the current ready view and its pending operation.
Close, connection/account invalidation and host disposal retire presentation
leases, leave fullscreen and discard old acknowledgements. These transient
leases are not saved or exported.

Stopping a pending presentation request restores its previous mode. A missing
renderer acknowledgement returns the previous mode after five seconds and
sends a sequenced restoration event; a later acknowledgement is refused.
This is a presentation deadline, not a tool timeout, automatic approval or
remote cancellation. The renderer's sequence check prevents a failed old
acknowledgement from reversing a newer restoration.

## Recorded evidence

Nine distinct new pure contracts passed across scoped runs: five host cases,
three renderer-helper cases and one preload case. They cover capability
negotiation, missing/unsupported modes, actual acknowledgement gating, other
fullscreen owners, wrong or late acknowledgement identity, timeout, Stop,
retirement and sequence filtering. The production `McpAppService` fixture
performs its actual handshake with a synthetic permitted resource result and
proves the mode change adds no tool execution. Its initial fixture was missing
the existing required resource provenance; only that failed case was corrected
and repeated, while the production guard remained intact.

Renderer tests use an inert dialog object; they prove helper decisions and
node-preserving operations, not actual Electron rendering, focus or screen-reader
behavior. No old roots/logging/tasks/App suites were repeated. No provider,
credential, native brain, Load, training, hardware or package test ran. Direct
Electron Open/Expand/App-request/Escape, missing/late acknowledgement, access
retirement and explicit reopening now have bounded evidence in
[the October18 display check](mcp-display-ui-2026-10-18.md). Full
theme/layout/accessibility coverage and packaged OS acceptance remain separate.

## Existing Electron fixture composition

`tests/e2e/mcp-app-host-main.ts` and its existing preload/page now carry the
same presentation event and acknowledgement path. The synthetic App declares
inline/fullscreen and has its own Request fullscreen/Request inline buttons.
It displays a per-document instance marker, resulting host-context mode and
actual RPC response, so a UI audit can distinguish resizing from reloading.
The existing synthetic tool remains separate and is not run by these controls.

The fixture page offers Normal, Hold, Refuse and explicit access retirement
three seconds after a fullscreen acknowledgement. Held acknowledgements can be
released after the five-second restoration to inspect late-receipt refusal.
Retirement updates the fixture access control and the production App component;
Restore enables only this synthetic source for an explicit fresh Open. Mode
events and acknowledgements enter its visible host trace without private view
tokens. No PIP/device or other unsupported capability is declared.

The tiny main entry was bundled successfully (183 KiB, 24 ms) and the preload
passed Node's syntax check. It was not launched by this lane. Root owns the
current desktop audit and the following bounded computer-use check.

From the repository root, with the existing Vite fixture server on port 5189:

```sh
./node_modules/.bin/esbuild tests/e2e/mcp-app-host-main.ts --bundle --platform=node --format=cjs --external:electron --outfile=.runtime/mcp-app-host-ui/main.cjs
```

Set `OMNI_MCP_APP_UI_ORIGIN=http://127.0.0.1:5189` and
`OMNI_MCP_APP_UI_DATA` to an existing dedicated temporary profile, then launch
`./node_modules/.bin/electron .runtime/mcp-app-host-ui/main.cjs`. Without an
origin override the existing fixture defaults to port 5192. Neither profile is
the user's production Omni instance directory.
