# Runtime-stream parser: original attempt and approved affected retry

October 15, 2026. This is an app-integration fixture, **not a brain**. It imports
the production runtime service, stream-parser adapter, restricted preload,
Electron process factory and resource admission, with a synthetic exact catalog.
No BrainService, EngineSupervisor, neural worker, PyTorch, model, curriculum,
corpus, provider, personal file or real ToolExecutor effect is used.

The planned input is 8,193 deterministically generated raw bytes, source SHA-256
`10a7c2838bcc0427685211a564df54d35c7fb3dcc5ff1d420fc16c41cae7df73`.
The actual normalized owned package hash is
`7fdcf27823a96bbc3ae5dbc706dfb71dd1db2a4b6ff08d8c8daf03351f914aa9`;
the owned parser module hash is
`b14f868b394c13c48c6780fc2f7dcda5f722ead806b02123d690229bcb202a2f`.
The fixture would use an 8 KiB transport frame inside its physically admitted
lease to exercise multiple push/drain pages and one explicitly synthetic
Ask/review pause. This is not a real permission-enforcement or learned-tool test.

## Terminal result: FAILED / resource admission refused

The single authorized command was:

```sh
node scripts/run-owned-plugin-parser-proof.mjs
```

It stopped at production startup resource admission, before creating the parser
renderer or sending a source page. The recorded message is that selected app RAM
or physical transfer headroom was insufficient, with no next effect dispatched.
It is **not** a successful registerParser/push/drain/Ask/provenance proof.

- App elapsed: **156.675 ms**. Outside watchdog elapsed: **700.595 ms**.
- Peak sampled app-family RSS: **154,304,512 bytes**. Outside monitored family,
  including the watchdog process: **266,813,440 bytes**, under 768 MiB.
- The 20 GiB free-disk floor held; sampled minimum was **33,545,043,968 bytes**.
- No watchdog kill, parser steps, Ask requests, synthetic approvals or live
  WebContents occurred. Child exited with code 1; a post-run process inventory
  found no Electron, Python or neural worker.
- Every recorded production source and compiled-fixture hash was unchanged
  before/after the run.

The original private profile/reports remain at:

`/var/folders/l1/3bjvcv150hv_srvsrj66t8jr0000gn/T/omni-runtime-parser-owned-hEtcgv`

Content-preserving JSON copies (with one trailing newline added by the patch
writer) are retained in the work evidence directory:
[app report](../../work/plugin-runtime-parser-owned-2026-10-15/attempt-1.app.json)
and [outside watchdog](../../work/plugin-runtime-parser-owned-2026-10-15/attempt-1.watchdog.json).
The exact original app/watchdog file SHA-256 values are, respectively,
`7594cdf801f18688e15aae2b60bd0fb94534c1e991ecda1d669b7a4858abfc5c`
and `990a490c88245ed93caeb458948ee5c01270b37368c9119d033c907040e012a3`.
Failed original evidence was not overwritten or relabeled.

## Source-only interpretation and next boundary

The RSS envelope was not the only admission condition: production admission
also checks physical free bytes, heap availability, startup estimate and existing
transfer reservations. The attempt did not record the exact failing readings,
so its low sampled RSS does **not** establish a false-positive guard or tell us
which headroom condition failed. A later read-only OS observation showed about
99.9 MB free; it is an after-attempt observation, not the original failed reading.

Source inspection confirms that the fixture's 8 KiB frame clamp is applied to
the returned lease **after** production admission; it does not reduce the
production reservation calculation. That fact alone does not prove the cause
of this refusal and is not permission to weaken safety or retry.

The existing reviewed availability path in `resourcePlanner.ts` includes
macOS inactive/purgeable pages and an independent `memory_pressure -Q` reading,
then exposes `availableSafeRamBytes` with an adaptive OS reserve. The production
runtime composition already obtains that reviewed envelope but separately
passes bare Node `freemem()` to runtime admission. This is a source-level
composition inconsistency, not retrospective measurement of the failed attempt.
A precise proposal was sent to root to reuse that already-reviewed available
pool while retaining the current ceiling, RSS, heap, reservation and additional
headroom checks. No production change or retry was made by this lane.

The new fixture passed its focused TypeScript entry check (with the existing
Markdown ambient declaration), esbuild compilation and Node syntax check.
There were no production source edits or old generic-module fixture reruns.
At that point the actual parser integration gate remained unverified. Root then
approved the precise reviewed-availability composition correction and **one**
affected retry, described below. No new brain/model test was authorized.

## Approved affected retry: PASSED, same limits

Root changed the production composition to reuse the already-computed
`envelope.availableSafeRamBytes`. The RSS/selected ceiling, heap checks,
reservations, startup estimate and additional headroom checks were unchanged.
The owned harness now calls `ResourcePlanner.reviewedRamEnvelope(DEFAULT_CONFIG)`
for synthetic configuration measurement only; it records `modelFitEvaluated:false`
and never creates a brain, performs model fitting or runs a hardware benchmark.
Actual reservation snapshots and async-context-labeled admission readings were
added as fixture-only diagnostics.

The single separately approved command was:

```sh
node scripts/run-owned-plugin-parser-proof.mjs --reviewed-availability-retry
```

It passed in **2,039.119 ms** app time / **2,550.455 ms** outside-watchdog time:

- The actual restricted Electron SDK registered the parser, processed four raw
  source pages plus a final push (**5 pushes**) and **8 drains**.
- All **8,193 bytes** reconstructed exactly to the original source hash above.
  Complete JSONL contained **11,253 bytes / 5 records**, with output SHA-256
  `85bb9c8b676143675f41a91f47fcec1240d55b160f2517fd1fc9a0806a75d85b`.
- Exactly one clearly synthetic Ask paused at **2,304 input bytes**, remained
  paused during the controlled 40 ms review hold, accepted only its exact current
  review identity, and continued to complete progress. One fake approved callback
  occurred; no real ToolExecutor effect or permission-enforcement claim is made.
- The parser session stopped, **0 live WebContents** remained, and the post-run
  process inventory showed no Electron/Python/neural worker.
- Outside peak sampled family RSS was **478,920,704 bytes** (~456.7 MiB), below
  the unchanged **768 MiB** envelope. App-only sampled peak was **371,179,520
  bytes**. Minimum sampled free disk was **33,533,538,304 bytes**, above the
  unchanged 20 GiB reserve. No watchdog kill occurred; exit code was 0.
- All recorded production-source/build hashes stayed unchanged during this run.

All **80** recorded admission readings passed the RSS, reviewed availability and
heap checks. The first startup reading was:

| Field | Actual bytes |
| --- | ---: |
| Resident / selected fixture ceiling | 152,879,104 / 805,306,368 |
| Reviewed safe available RAM / bare free comparison | 5,798,205,849 / 95,174,656 |
| Available JS heap | 4,389,614,080 |
| Prior reserved / attempted reservation | 0 / 128,284,360 |
| Additional headroom | 67,108,864 |

The unchanged startup estimate was 67,108,864 bytes; the actual module script was
1,305 bytes. The minimum reviewed available pool across the run was
5,626,407,157 bytes. These new measurements support the corrected current
composition; they do **not** retrospectively prove the numerical cause of the
original refusal, whose detailed readings were absent.

The fresh private directory is:

`/var/folders/l1/3bjvcv150hv_srvsrj66t8jr0000gn/T/omni-runtime-parser-owned-V3PB5t`

Complete content-preserving JSON copies (one added trailing newline) are retained
as [retry app report](../../work/plugin-runtime-parser-owned-2026-10-15/attempt-2.app.json)
and [retry watchdog](../../work/plugin-runtime-parser-owned-2026-10-15/attempt-2.watchdog.json).
Original file SHA-256 values are `ac63fd8c3cae2d3bebb3f7bde800015d40f4ed405e72b6ec900ad735d9d17db7`
and `8175ce19b2bc5b4a9da483c3f6cd71e93a4ffe87f771c372ed68292f8bb29379`.
The failed attempt and its compiled artifacts were left untouched; the retry
used a distinct build directory.

Retry main/preload SHA-256:

- `.runtime/plugin-runtime-parser-owned-retry/main.cjs`:
  `c1af16cf50119191b35c7b81aeb7e493c06a21342e759f0ae2223b9d8046de28`
- `.runtime/plugin-runtime-parser-owned-retry/preload.cjs`:
  `d2dc3bdd88d8606ba28d0863dfdd5683db1bc68a3e68d3a0326b1db691e96301`

## Acceptance boundary

This closes the bounded owned-byte parser **app integration** case on this Mac.
It does not establish real tool permission enforcement (the host was synthetic),
large/private corpus coverage, arbitrary third-party parser correctness,
neural learning, trained comprehension, all operating systems or packaged release
quality. The 768 MiB figure is a sampled monitored envelope, not an OS-enforced
hard memory sandbox or proof that no between-sample allocation existed. No closed
generic module fixture or source suite was rerun, and no further retry is pending.
