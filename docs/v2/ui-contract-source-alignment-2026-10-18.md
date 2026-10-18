# UI contract source alignment — October 18

## Verified boundary

This is a static reconciliation of the five existing `test:ui:contract`
specifications with the current Electron renderer. No Electron application,
browser, Playwright session, native model, Load operation, training job, device
capture, provider, or external action was started for this pass.

A scoped TypeScript program containing these five specifications completed
with **zero diagnostics**. This establishes source/type consistency only; the
coordinated GitHub UI run remains the runtime gate.

## Confirmed stale assumptions corrected

| Specification | Former assumption | Current contract |
| --- | --- | --- |
| `chat-ui-contract.spec.ts` | Preview Brain Map exposes a nonzero legacy `.graph-legend` neuron count. | The actual `WholeBrainMap` exposes its unified layers and an honest unavailable state when no inspection API exists. The test requires disabled refresh, no substitute graph, and a functioning Back to chat control. |
| `chat-ui-contract.spec.ts` | Composer shape was not explicitly checked. | Idle preview has one submit control in Message controls, plus the existing attachment control. Disabled Send and Enter preserve the draft and cannot create synthetic messages. The draft also survives the actual map round trip. |
| `responsive.spec.ts` | Workspace navigation contains nine actions. | All twelve current routes are enumerated by their accessible names and retain per-route viewport reachability checks. |
| `responsive.spec.ts` | Start live voice is always visible and acts as a pressed toggle. | The visible Call toolbar button controls the existing Call controls region. Opening it reveals Start live voice and Camera and screen sharing; closing it hides those controls. The test does not start voice or camera capture. |
| `theme-matrix.spec.ts` | Hand-injected action markup represents the current component, and Live Perception is an exposed standalone product panel. | The inert CSS sample uses the current expandable action `details`/`summary` structure and the Camera and screen section structure. Every injected sample explicitly identifies itself as CSS-only, with no action/capture or measured resource claim. |
| `theme-matrix.spec.ts` | Old `.map-page`, `.map-inspector`, `.graph-status`, and `.map-viewport-controls` are the primary Brain Map. | Contrast targets now reference the actual whole-map layers, selection/detail headings, source-navigation text, and no-engine status. An explicitly synthetic SVG label preserves isolated graph-label CSS coverage without inventing an inspection API or firing evidence. |
| `theme-matrix.spec.ts` | Changing a surface pack must override the separately selected chat/inspector arrangement. | The surface-pack matrix pins the independent split arrangement. Existing four-arrangement tests still require side-by-side, full-focus, stacked, and mixed geometry. Pack-specific proportions, inset, radius, rhythm, and typography assertions remain. |
| `imagination-gallery.spec.ts` | Reaching an empty preview gallery might be described as persisted-media evidence. | The test explicitly identifies the offline preview, checks the no-local-files message, and requires no persisted artifact items or media elements. Open, Escape, reopen, close, and return-to-chat assertions remain. |

## Working checks retained

- `appearance.spec.ts` already matches the current mode, palette, layout,
  persistence, fieldset names, and appearance-pack attributes. It was inspected
  and left unchanged.
- System appearance still follows live emulated light/dark changes. Fresh
  Apple defaults and explicit saved-preference precedence remain checked.
- The contrast minima are unchanged: **7:1** body text, **4.5:1** ordinary
  representative text, and **3:1** the existing icon/disabled targets.
- Both matrix viewport sizes and all four appearance packs remain. The six
  responsive viewport sizes and keyboard/inspector checks remain.
- The theme matrix now scrolls to the actual lower Appearance notification
  switch, preserving bottom-of-panel coverage rather than checking its top only.
- Existing no-fake-chat assertions, draft preservation, disabled Send,
  off-chat inactivity, gallery dismissal, root overflow, button labels, and
  RAM-settings reachability checks remain.

## Not established by this pass

These specs still use the existing no-preload Electron design-preview shell.
They do not establish native inference, learned recall, real queue/Steer/Stop
delivery, live Brain Map records, a device or provider connection, retained
media bytes, packaged-runtime delivery, or performance. Inert injected action,
meter, camera, builder, and SVG samples establish CSS behavior only when the UI
runner actually executes them. Measured contrast and geometry are pending that
run; no visual pass is inferred from source inspection or TypeScript success.

No production UI, contrast threshold, model architecture, or user data was
changed by this reconciliation. Full training and learned-capability testing
remain deferred under the current task agreement.
