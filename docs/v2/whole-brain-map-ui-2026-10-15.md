# Whole-brain map — desktop integration

## Scope

The latest user clarification is preserved in AMENDMENTS.md. Brain Map now
mounts one `WholeBrainMap` in the existing Electron application. Registered
regions, shared owners, sparse records and selected packed contacts are source
facets of that view, not separate primary brains. The app layout was not remade.

The source contract is in whole-brain-inspection.md. This document records the
root application bridge and UI evidence separately from the source adapter.
No brain construction, Load, training or native forward was run for this slice.

## Implemented root boundaries

- Typed preload/IPC delegates to `BrainService.queryWholeBrain`; cached metadata
  is read with `repo.get(id, false)`. There is no implicit Load, runtime selection,
  startup preflight or neural-writer claim. A separate inspection worker may
  read the committed files; that is not a second neural model.
- Main validates the normalized query against the exact response facet, root
  owner or selected connected-region lineage. Renderer-supplied paths and
  topology are not accepted. Each sender's replacement request aborts only its
  prior inspection, not chat generation.
- The primary canvas retains overview and selected region/owner source context
  during parameter drill-down. Previous/Next records and exact matrix-row
  controls revisit the source without storing every prior tensor page in RAM.
  Paging is a transport/display mechanism, not a model-size cap. Large-scale
  multiresolution/viewport qualification is still open.
- Refresh retires the previous committed snapshot. A changed generation fence
  or foreign/late owner response is rejected instead of joined into the graph.
- Structural memberships and registered bridges are dashed and explicitly not
  observed learned weights. Actual contacts retain sign, strength, repeat
  identity, eligibility, plasticity, stability and usage when reported.
- Connections are selectable by keyboard/canvas or the expandable connection
  list. Exact values and provenance are available without filling the ordinary
  screen with raw IDs. Unknown values are not replaced with measured zeros.
- Optional live LIF observations feed the same canvas under their own capture
  scope. They are not silently equated with committed semantic neurons or
  tensor axes. Empty transport polls do not erase an actual last observation.

## Focused root evidence

- `wholeBrainAppBoundary.test.ts`: 2 fake-host/preload cases passed (55 ms).
  These call the actual prototype with inert dependencies; no worker was run.
- `wholeBrainMapPresentation.test.tsx`: the 3 affected existing renderer cases
  passed (16 ms). One new repeated-contact/plasticity case then passed alone
  (6 ms); the previous three were skipped, not counted again.
- `wholeBrainConnectionSelection.test.tsx`: one new static-render case passed
  (17 ms). Its initial assertion incorrectly required adjacent HTML attributes;
  it was corrected to inspect the exact contact element without depending on
  attribute order. No product behavior was relaxed.
- The independent host-boundary lane adds 3 new fake cases (11 ms), covering
  cross-facet payloads, foreign root ownership and connected-lineage anchors.
- The fixed Node CI contract selector now names 99 distinct existing files.
  Path presence is checked; this is not a fresh 99-file suite pass.

## Visual acceptance

The production component was inspected through native computer use in one
isolated Electron preview using `tests/e2e/whole-brain-fixture.html`. There was
no preload/native engine. The preview closed normally with exit0. The existing
Vite fixture server was reused; no second model or neural worker was started.

1. **Overview and shared-owner drill-through — passed, synthetic.** One primary
   graph retained the two regions and their shared owner. Region → Owner →
   weight inspection preserved their actual fixture memberships. The final
   generic Parameters → matrix route also returned exact binding/owner anchors:
   8 typed nodes,11 relationships,0 off-page endpoints.
2. **Paging, rows and connection selection — passed with a boundary.** Modules
   and matrix contacts reached their second page and Previous returned. Manual
   row1 appeared in the actual query receipt. The HTML connection list selected
   the exact repeated edge and displayed +1/strength3. A native AX press on an
   SVG path produced no change; do not report direct SVG mouse/AX selection as
   proven. The explicit accessible list remains available.
3. **Sparse detail and overlap — passed, synthetic.** Detailed requests emitted
   `zoom:1`, avoiding the original aggregate-default blank detail. Two typed
   assemblies retained one common member and separate backing-neuron aliases:
   9 nodes,12 relationships,0 off-page endpoints. This did not execute learning.
4. **Wrong owner, stale cursor and mixed snapshot — passed, synthetic.** Wrong
   owner and old cursor responses were refused. A fresh generation2 page was
   also rejected by the production snapshot fence while generation1 remained
   displayed. Refresh installed one generation2 overview and cleared the error.
5. **Light/wide and dark/360px pane — targeted pass.** The audit found and fixed
   a two-column layout tied to window rather than pane width, an unthemed row
   input/adjacent label, and weak structural-line contrast. Details now stack
   by available pane width, the row control uses theme tokens and spacing, and
   structural links remain visibly distinct. Longer explanatory copy moved to
   a collapsed legend. The 3D control and projected-layout state were inspected;
   this is not a full all-theme, OS-resize, touch or screen-reader audit.

Accepted saved/reopened captures under `test-results/whole-brain-ui-2026-10-15/`:
`05-compact-dark-reflow-fixed.png` (lower controls),
`08-connected-parameters-final-light.png` (complete connected parameter graph),
`10-compact-dark-contrast-final.png` (final compact3D).
The initial01/02 and07 are partial/diagnostic states, not a complete graph
acceptance.04 shows the cramped-layout finding.06 captured a development hot
reload and the wrong state; reject it.09 predates the contrast fix. Development
fixture reloads were not a production chat/context-reset test.

The affected controls-only LIF component additionally reached contacts129–140
of140 with Next disabled/Previous enabled, and exact Stop returned Observation
off. Its compact dark capture was saved/reopened at
`test-results/router-contact-audit-2026-10-15/09-combined-controls-contact-page.png`.
An in-flight contact-page revision fence prevents a late previous page from
replacing the new viewport. No LIF step or neural transmission was executed.

No screenshot or successful source check is being reported as actual neural
publication, biological overlap, live all-region firing, trained recall,
hardware performance or packaged-platform acceptance.

The current `.omni` format decision remains separate: catalog snapshot/fork/
recovery closure is source-tested, while export/import format changes await
the user's answer. Do not call complete export of the new catalog accepted.
