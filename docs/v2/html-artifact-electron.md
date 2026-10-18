# Saved HTML artifact Electron fixture

Updated 2026-10-13. This file deliberately separates source/bundle readiness
from observations made in an actual Electron window.

## Source and bundle evidence

The bounded fixture uses:

- the actual `HtmlArtifactStore` and a task-owned temporary SQLite registry;
- the actual `HtmlArtifactService`, `omni-html-artifact:` protocol document,
  CSP/header/request/navigation helpers, and `HtmlArtifactView`;
- one enabled artifact and one exact copied/rekeyed artifact whose script and
  bridge begin dormant;
- a local counter/canvas, Node/Studio-boundary text, and an undeclared network
  probe inside the opaque child;
- one explicitly synthetic `fixture.readonly.inspect` callback. It has no file,
  network, provider, credential, model, or other external effect and always
  requires the real artifact host's visible Ask flow before returning a labeled
  readonly fixture receipt.

The fixture sources are `tests/e2e/html-artifact-host-main.ts`,
`html-artifact-host-preload.cjs`, `html-artifact-host-fixture.html`, and
`html-artifact-host-fixture.tsx`. Node TypeScript, the main/renderer fixture
bundles, and the scoped whitespace check passed. These are source/build checks,
not runtime observations.

The prepared main owns an ephemeral loopback server with exactly three routes:
`/index.html`, `/renderer.js`, and `/renderer.css`. It does not expose an
arbitrary path or filesystem router. From the repository root, the one-window
launch seam is:

```text
./node_modules/.bin/electron .runtime/html-artifact-host-ui/main.cjs
```

Closing the fixture window disposes live views, closes that loopback server,
and deletes its temporary SQLite workspace.

## Actual Electron observations

Root completed a bounded CUA-controlled run on macOS26.3, Electron43.2.0,
ARM64. No brain, model, provider, private source file or credential was loaded.
Two initial blank harness windows were rejected, not counted as passed UI
evidence: Vite's injected preamble conflicted with the strict outer CSP, and
the first standalone bundle used classic JSX rather than the app's automatic
JSX runtime. The harness now serves precompiled assets through its exact
loopback routes, and its renderer bundle uses `--jsx=automatic`. The production
artifact CSP was not weakened to make the fixture pass.

Accepted screenshots were saved and reopened from their exact PNG bytes:

1. **Enabled document/local canvas — pass.** The real custom-scheme document
   opened, Node/Studio globals were unavailable, undeclared network was blocked,
   and Increment changed only the local counter. Screenshot01 shows initial
   counter0; the later approved result shows its actual counter1.
   ![Enabled isolated artifact](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/html-artifact-ui-manual/01-loaded-enabled.png)
2. **Exact host Ask review — pass.** The actual host bridge produced a Studio
   review for only `fixture.readonly.inspect`, with the exact `{value:1,note}`
   operands and saved artifact identity. The child remained waiting.
   ![Exact host review](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/html-artifact-ui-manual/02-exact-host-review.png)
3. **Reviewed readonly result — pass.** Approve returned an explicitly labeled
   synthetic readonly receipt with observedCounter1. No real tool effect ran.
   ![Reviewed fixture result](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/html-artifact-ui-manual/03-reviewed-result.png)
4. **Origin/CSP and fresh reopen — pass.** Visible host trace showed origin
   `null`, a script-disabled copied view, hash-only script policy for enabled
   content and connect-src none. Close/Reopen changed the actual view handle
   from863e6c… to501234…; local state restarted and no prior tool was replayed.
   ![Actual protocol policy trace](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/html-artifact-ui-manual/04-actual-csp-trace.png)
5. **Copied dormancy/explicit activation — pass.** Clicking Increment while
   dormant did not run script. Explicit activation created a new view; its
   local Increment then changed the counter. No parent's host grant transferred.
   ![Explicit copied interaction](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/html-artifact-ui-manual/05-activated-copy.png)
6. **Light host appearance/scrolling — pass for checked desktop size.** The
   actual parent theme changed while artifact content retained its own supplied
   styling. Controls/text were readable, and both lower copied content and
   the protocol trace were inspected by scrolling. This is not all-theme,
   responsive, hover or accessibility certification.
   ![Light parent and reopened artifact](/Users/eyadabushama/.codex/worktrees/9eaa/Omni-AGI/test-results/html-artifact-ui-manual/06-light-host-and-reopened.png)

The fixture was closed afterward; process inspection confirmed its Electron
main was no longer running. Its task-owned temporary workspace/server were
closed by the fixture lifecycle. The existing Omni app was not redesigned.

### Cleanup finding

The actual exit log exposed a late Close IPC request after activation had
already invalidated that handle. Main cleanup is now idempotent for an absent
view, never recreates it, and still rejects a foreign owner/nonce for a live
view. One new actual small-file host regression passed35ms (the three earlier
host cases skipped). This **source fix was not rerun through another full
Electron flow**; final packaged/lifecycle acceptance remains separate.

The bounded checklist above covers:

1. enabled artifact document and exact CSP/current protocol origin;
2. local counter/canvas activity without a host trace effect;
3. synthetic readonly request, visible exact Ask arguments, approval, and final
   labeled receipt;
4. close/reopen with a different view handle and no automatic tool replay;
5. copied presentation initially script-disabled and explicit activation of
   that exact revision;
6. screenshot path and the precise runtime/OS observed.

Even a successful run will not establish packaged-release behavior, other OS
runtimes, hostile-content certification, trained natural artifact generation,
or useful visualization quality.
