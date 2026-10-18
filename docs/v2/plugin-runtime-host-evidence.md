# Owned plugin runtime: actual Electron boundary check

October 15, 2026. This is an app integration fixture, **not a brain**: a
synthetic plugin catalog, a private temporary operational store, and one owned
JavaScript module. No BrainService, neural worker, model, corpus, provider,
personal source, real ToolExecutor effect, or saved instance was run or changed.
The newly requested native merged-brain fixture remains user-deferred.

## Failed discovery and affected correction

`work/plugin-runtime-host-2026-10-15/attempt-1.json` remains failed. It reached
checked readiness, persisted isolation state/contribution, and a typed pure
tool call, then Stop failed with `Object has been destroyed`. The process
factory reread a destroyed BrowserWindow's WebContents getter during release.
It now retains the exact owned WebContents and sender identity before teardown.

Source review also found that aborting before `deactivate` prevented reliable
local cleanup. Stop now refuses new work immediately, cancels already admitted
host requests before grace, delivers a fresh main-owned shutdown challenge, and
allows up to 100 ms for local cleanup. The public SDK is already closed during
cleanup. A matching private ACK reports callback outcomes; it grants no new
capability. Timeout and observer/disk errors still terminate/release the child.
The ACK is bootstrap self-report, not hostile-code attestation.

## One affected actual retry

`work/plugin-runtime-host-2026-10-15/attempt-2.json` passed in **892.770 ms**:

- Separate owned renderer PID and checked readiness: 514.444 ms from fixture start.
- Persisted observations: Node `require`, `process`, and Studio `omni` globals
  were unavailable; the declared UI contribution was present.
- Typed pure addition returned 5; a string operand was rejected without coercion.
- Stop persisted an acknowledged cleanup diagnostic: 1 callback, 1 completed,
  0 failed. The measured ACK wait was 0.860 ms within the 100 ms grace.
- Contributions retired and no live WebContents remained after Stop.
- Peak **sampled** app-family RSS was 360,579,072 bytes (~343.9 MiB), under the
  fixture's 768 MiB monitored envelope. This is not an OS-enforced hard cap or
  a proof that no allocation existed between samples.

Exact built fixtures:

| File | SHA-256 |
| --- | --- |
| `.runtime/plugin-runtime-host/main.cjs` | `c3b56df59a32f007c25c16e61f6a20bb48be1b6742413fcf7ce435f3c32787c6` |
| `.runtime/plugin-runtime-host/preload.cjs` | `d2dc3bdd88d8606ba28d0863dfdd5683db1bc68a3e68d3a0326b1db691e96301` |

The current source at this check passed coordinated main and renderer TypeScript
compilation and tracked whitespace checks. Three distinct new fake-host shutdown
cases covered correlation/closed SDK/host cancellation, timeout and refused
diagnostic persistence, and a throwing observer. The observer case ran alone
(27 ms; the earlier two skipped); the earlier affected pair passed in 127 ms.
No completed broad suite was rerun.

## Boundaries

This checks owned JavaScript lifecycle and the production Electron isolation
composition on this Mac. It does not qualify arbitrary hostile plugins, every
host capability, all operating systems, packaged release behavior, learned tool
understanding, custom media parsers, or the deferred merged-native resume proof.
UI bindings have separate evidence in `plugin-runtime-ui.md`; synthetic UI Stop
alone was not counted as process-release proof.
