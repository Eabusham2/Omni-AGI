# Manual regional execution — main, preload and bounded UI evidence

October15 source integration; this is not complete R1, actual neural/hardware
execution or package acceptance. The native consumer is documented separately
in `manual-regional-execution.md`.

## Main-owned contract

`RegionalExecutionService` exposes Prepare, Status, Activate and Deactivate.
The renderer supplies only the instance/page, an opaque review token, or an
existing epoch. It cannot authorize a topology, policy hash, registry hash,
filesystem path, worker Start or neural Load.

Main observes topology through `DeviceResourceService.topologyForLoadedOwner`
and `requestLoadedRegionalExecution`. Discovery and native review use the same
already-loaded worker and ordinary serialized boundary, not general discovery
that may start another worker. The shared serial helper preserves the existing
native-candidate semantics. Exact owner/config/origin/path and the process-local
generation are rechecked after acquiring the slot. Current native policy,
registry/object geometry and physical topology bind the actual activation.

Review custody is process-local and one-use. A lost activation ACK is inspected
with Status rather than automatically dispatched again. Suspend, recovery and
worker close retire main review tokens. Saved/copy/import route preferences
never arm a live lease. Backend cache/pin/transaction quiescence remains required
for changes and does not clear active context to force a fit.

The advanced control is mounted in the existing Device & runtime workspace.
Route drafts go through its ordinary settings Apply control. Batch edits retain
descriptor inspection while invalidating activation authority; Apply alone does
not restore that authority. New RAM-route allocation is normalized to the
separately chosen global RAM mode/ceiling rather than silently raising it.
CUDA allocation controls become editable only for a matching acknowledged
CPU-tile route; unsupported native CUDA remains visibly unsupported.

## Scoped source results

- Main owner/review service:3 new fake-host cases,14ms; affected status-shape
  recheck also3/3,14ms. These are the same3 cases, not6 independent proofs.
- Loaded supervisor:1 new constructor-free fake-pipe case,4ms;2 affected prior
  native-candidate helper cases,5ms after extracting the shared serial helper.
- Preload:1 new bridge case,31ms. Loaded topology:1 new injected case,7ms.
- CUDA allocation presentation:1 new render case,25ms after fixing the test's
  missing React import; native support is never relabeled by the fixture.
- Controls:4 initial new cases,22ms;1 separate new batch-edit/owner case,25ms
  with the prior4 skipped;2 later new presentation/default cases,23ms with the
  prior5 skipped. Seven distinct cases, no old-suite rerun.
- Current Node and renderer TypeScript checks pass. Initial diagnostics were
  optional engine/test fields and a `flatMap` union inference; those exact
  diagnostics were corrected, not hidden by weakening TypeScript.

Separate transfer-rate identity checks retain measurements only for the same
physical interface/route and never overwrite fresher readings. They do not
prove a device's speed. The corrected int64/contact-state accounting and native
consumer evidence remain in `device-source-readiness.md`.

## Numbered native Electron UI flow

The production component ran in an explicitly synthetic no-preload fixture,
with no brain, probe, Load, model, training, real activation, transfer or neural
effect. Do not read its synthetic counters as measured hardware activity.

1. **Inert mount/open — healthy:** no API operation happened until explicit
   Prepare. The surface is collapsed by default.
2. **Batch edits — healthy:** one CPU route and one explicitly acknowledged CUDA
   fallback route were drafted from the same retained page. The Add action was
   disabled before CUDA acknowledgment. Unsupported NPU was absent from choices.
3. **Apply versus activation — healthy:** Apply made the settings clean but left
   Activate disabled until fresh Prepare. See `01-saved-routes-stale-review.jpg`.
4. **Paging and explicit activation — healthy:** Next exposed the third exact
   descriptor; Activate consumed the current synthetic review token and displayed
   two routes. It did not create or run a neural model.
5. **Busy boundary — healthy:** Prepare/Deactivate disabled during the synthetic
   foreground reply while Status remained usable. The compact light metrics and
   route list did not overlap. See `02-active-busy-light-compact.jpg`.
6. **Deactivate and owner retirement — healthy:** after the boundary, exact epoch
   deactivation worked; changing API owner cleared operational state while saved
   preferences remained. Bottom scrolling exposed the actual synthetic callback
   receipts. See `03-new-owner-bottom-receipts.jpg`.
7. **Fresh target default — fixed and spot-checked:** a fresh Prepare had shown
   CPU for an already saved CUDA route. The selector now resolves draft choice,
   then exact saved target, then the first eligible target. Its saved fallback
   acknowledgement is preserved. The affected-only fixture and screenshot
   `05-saved-cuda-corrected.jpg` verify the corrected fresh state.
8. **Invalid lease label — fixed and spot-checked:** an armed invalid lease now
   says Armed but invalid, not Active manual lease. The separate synthetic
   spot-state is in `04-invalid-lease-corrected.jpg`; no activation was attempted.

All five accepted screenshots were saved and reopened under
`test-results/regional-execution-audit-2026-10-15/`. Native compact captures are
1444×962 image pixels, not claimed CSS viewports. Both owned fixture windows
closed normally; both processes exited0. The second run checked only the two
changed presentation states rather than repeating the original flow.

These screenshots/AX checks do not establish all-theme accessibility, the full
Device & runtime parent screen, actual main→worker→neural behavior, automatic
whole-path placement, regional training, GPU-backed context, hardware speed,
multi-GPU qualification or packaged-platform acceptance. Those remain explicit
requirements, not replaced by this manual decoder-only slice.
