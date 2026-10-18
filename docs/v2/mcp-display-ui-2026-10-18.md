# Connected-App display — bounded Electron check

The production MCP App host/proxy/renderer ran in the existing isolated
`mcp-app-host` fixture. Its completed tool result and HTML resource are synthetic;
no account, model, Load, training, external provider or real host tool ran.
This checks actual Electron presentation and IPC, not remote interoperability.

## Steps

1. **Inline Open — healthy.** `01-inline-ready.png` shows the completed App
   handshake, sandbox boundary and page marker `dtq2ejl61g`. The first AX read
   preceded readiness; the accepted saved screenshot shows the ready state.
2. **Host Expand — healthy.** `02-host-fullscreen.png` shows the real dialog
   top layer and available Return/Close actions. The proxy/resource identity
   and page marker are unchanged. This is Studio fullscreen, not OS fullscreen.
3. **App-requested inline — healthy.** `03-app-inline-preserved.png` shows
   the original marker and actual `{"mode":"inline"}` RPC response. The
   original source tool was not called again.
4. **App-requested fullscreen and Escape — healthy.** The exact App request
   returns fullscreen in `04-app-fullscreen-response.png`. Escape returns to
   inline in `05-escape-inline.png`, preserving the same marker. The App's
   displayed last RPC response remains the earlier fullscreen response;
   its separate current-mode field correctly changes to inline.
5. **Missing acknowledgement — healthy restoration.** In the fixture's Hold
   setting, `06-held-presentation.png` shows pending presentation with Return
   disabled and Close still available. The host does not tell the App that
   fullscreen was confirmed. After the real five-second deadline,
   `07-timeout-inline-restored.png` shows inline and the actual inline response.
   Independent compiler/documentation work ran during that deadline; no
   arbitrary delay was inserted.
6. **Late reply — refused.** Releasing the held acknowledgement did not restore
   fullscreen (`08-late-ack-stays-inline.png`). The actual isolated host logged
   `This app display acknowledgement expired.` The process did not crash.
   Reading the visible host trace was diagnostic only; the fixture tool and
   static-plugin buttons were not invoked.
7. **Access retirement and explicit reopen — healthy.**
   `09-retirement-pending-fullscreen.png` records the permitted displayed App.
   The fixture retires access after three seconds. In
   `10-access-retired-exits-fullscreen.png`, the frame is gone and the host
   requires explicit reopening. Restoring access alone did not recreate it.
   Explicit Open creates a fresh proxy and marker `54fet6oqn7a` in
   `11-explicit-reopen.png`, as expected after closing a retired view.

Accepted screenshots were saved and reopened under
`test-results/mcp-display-ui-2026-10-18/`. Dark and light host appearance were
observed. The App intentionally supplies its own light content; this is not a
claim that Studio recolors arbitrary third-party pages. Keyboard Escape and
visible focus were checked. The initial stale AX tree did not expose Expand,
so its observed screenshot coordinates were used; later AX trees exposed the
ready controls normally. The owned Electron process84394 exited0 after Command-W.

## Limits

The synthetic source, permission control and delayed acknowledgement are
declared fixtures. They do not establish account authorization, native brain
learning, every theme/size, screen-reader compliance, Windows/Linux packaging,
native OS fullscreen or PIP. The separate nine source contracts cover exact
identity, negotiation, missing/late/wrong acknowledgements and teardown. No
broader closed MCP task/root/logging suite was rerun for this presentation work.
