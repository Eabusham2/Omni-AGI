# Desktop preload and renderer startup boundary

## Observed defect and scope

Root's actual application check found a sandboxed Studio preload failure:
`module not found: node:crypto`. The renderer then displayed synthetic design-
preview identities because the bridge was missing. Root replaced the preload
UUID import with browser `globalThis.crypto.randomUUID`; sandboxing and
`nodeIntegration: false` remain intact. Root separately corrected the normal-
quit destroyed-window access. Neither root fix was changed by this source lane.

## Complete preload-entry contract

`tests/preloadSandboxEntry.test.ts` bundles the full `src/preload/index.ts`
entry with the installed esbuild in browser/CJS mode. Electron is its only
external import and its only permitted runtime `require`. The entire bundle
executes in a small VM with browser WebCrypto and Electron's main-frame flag,
without Node filesystem, Buffer, crypto imports, or a Node-polyfill fallback.
The host test's WebCrypto object is supplied by its test runtime, not bundled
into the shipped preload.

The cases require main-frame `omni` exposure, typed platform/brain IPC, no eager
work, no raw IPC/process/require exposure, browser UUID generation, listener-
before-subscribe ordering, exact subscription/brain identity, buffered initial
events, revision filtering, cleanup, fresh subsequent identities, and no API
exposure to a subframe. The Electron module and IPC responses are inert test
adapters: this is complete-entry source execution, not a claim that real
Electron context-bridge cloning or the application backend ran in that VM.

## Renderer behavior

The existing desktop `main.tsx` now wraps `App` with `RendererStartupGate`.
Normal startup requires the exposed desktop bridge with callable platform and
brain-list endpoints. A missing or malformed bridge renders a truthful desktop-
connection error and Reload window control. It does not render the synthetic
App branch, create a replacement bridge, or attempt an external service.

A no-bridge design fixture may opt in using the single exact query
`?omni-preview=1`. A working bridge still takes precedence; a malformed partial
bridge cannot use the flag to hide its failure. Duplicate or different values
are refused. This is an explicit no-authority presentation choice, not a
permission, credentials, IPC access, or alternate trusted Studio origin.

Only the existing no-preload `tests/e2e/responsive-main.cjs` helper needed that
query when loading the compiled renderer. The real main-process URL, navigation,
sender/frame checks and media/IPC authority are unchanged. Direct `App` fixtures
remain explicitly isolated no-engine fixtures by construction.

Android uses native Activity/XML and iOS uses native SwiftUI entry points;
their current packaging does not reuse this renderer or asynchronously install
`window.omni`. No companion bootstrap was changed.

## Evidence and limits

The three new preload-entry cases and three new renderer gate cases passed
**6/6**, in805ms total, with the supported bundled Node runtime and one worker.
The gate checks render the actual failure component, suppress synthetic child
rendering, and exercise only its supplied Reload callback. The preload checks
invoke only inert typed IPC adapters. No old suites were rerun.

This lane did not launch an app, Playwright or browser; construct/load/train a
brain; invoke a provider; change real-main authority; push; or dispatch CI.
Root owns the actual-main bridge/platform/no-preview recheck, normal shutdown
and packaged smoke. Source tests and an explicit fixture flag cannot stand in
for those real runtime results.

## Actual compiled-main recheck

Root rebuilt only the application shell after the fixes (main/preload/renderer
bundles exited0). The actual `electron .` application then ran against the
isolated `/tmp/omni-source-shell-fixed.bzdZwC` data/profile directories, with
background learning suspended and no preview query. Native computer-use
inspection showed the real Brain Library with **0 independent identities**,
not the three synthetic preview entries. Clicking Build a new brain opened the
actual Identity form. Root stopped before Continue/Create; no brain was created,
loaded, queried or trained.

Ordinary Cmd+Q completed with process exit0. The process-handle result and a
subsequent process check showed no surviving owned app/worker and no destroyed-
`webContents` JavaScript alert. Two AST cases separately exercise the actual
closed callback with a throwing destroyed-contents getter, checking both the
matching and newer trusted-owner IDs.

This closes the observed source-shell preload, hidden-preview and normal-quit
regressions at the local macOS entry point. It is not packaged-platform proof.
The unlocked local Python environment still lacks `jsonschema` and reports its
worker initialization failure honestly; this check does not claim working
neural inference. Locked packaged-worker health and clean artifact startup
remain CI gates. Both TypeScript projects pass; the integrated fixed non-neural
Python selector passes93 cases, and the affected seven-file desktop selection
passes20 cases. No Playwright runner or model test was launched locally.
