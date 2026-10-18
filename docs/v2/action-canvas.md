# Action-file canvas

## User workflow

The **Automation** workspace keeps its existing action-file create/edit flow and
adds two views of the same ordered steps:

- **Grid** automatically lays out every tool or wait step.
- **Freeform** retains explicit node positions. Drag a node, enter its X/Y
  coordinates, or use Alt + arrow keys; Shift + Alt + arrow moves farther.
- **Add tool** and **Add wait** append typed steps. **Move earlier**, **Move
  later**, and **Remove step** change the underlying sequence.

Connections are a visual trace of array order, not executable Bézier edges,
parallel branches, inferred dependencies, or evidence that a tool ran. Editing
is inert. **Save** and **Run** remain separate host-owned operations.

## Saved format and authority boundary

`AutomationProgram` may include this optional presentation field:

```json
{
  "canvas": {
    "version": 1,
    "mode": "freeform",
    "positions": [
      { "stepId": "inspect", "x": 24, "y": 24 }
    ]
  }
}
```

`automationFormat.ts` remains the authoritative main-process admission point.
It accepts only `version`, `mode`, and `positions`; every position must refer to
a distinct existing step, and coordinates must be finite, nonnegative, safe
numbers. Unknown fields such as an execution flag are rejected rather than
interpreted or repaired.

The ordered `steps` array remains the complete runtime sequence. Canvas
coordinates never become tool arguments, permissions, trigger activation, or
control flow. Reordering or removing a step is refused when it would silently
break a prior-step reference. Literal data that merely resembles a reference is
not rewritten.

## Persistence, copies, and execution

Saving validates and serializes the complete action file into the existing
immutable, content-addressed `.omni-action.json` store. Grid/freeform switching
and node movement do not mutate executable fields. Existing workspace custody
copies the exact revision bytes; copied definitions remain dormant and no run
is synthesized. Activation, manual Run, triggers, nested Off/Ask/Auto/Full
checks, current resource admission, effect receipts, Stop, and recovery keep
the separate [host-automation contract](host-automations.md).

The editor supplies pointer movement, numeric coordinate inputs, keyboard node
movement, ordinary buttons for sequence edits, selected-state semantics, and a
text reminder that editing runs no tools. Complete screen-reader, forced-color,
zoom, all-theme, compact-window, and packaged-platform acceptance still require
their own UI evidence.

## Evidence boundary

Five focused cases in `tests/actionCanvas.test.ts` passed in 80 ms. They cover
preservation of executable fields; typed add/reorder/remove behavior; refusal of
broken dependencies; rejection of forged, duplicate, nonfinite, negative, and
unknown layout data; and byte-identical content-addressed storage/copy with zero
runs and dormant copied activation. The renderer TypeScript check also passed.

One subsequent distinct 2,001-step projection case passed4ms, explicitly
forbidding repeated array searches and checking the final position and unchanged
steps. The original five cases were skipped, not rerun. Whole-canvas positions
are now indexed once per projection instead of scanned once per node. This is
a source-complexity correction, not a measured whole-app render-speed claim.

A bounded native Electron fixture checked Grid and Freeform in light mode, then
Freeform and Grid on a dark narrow surface. The narrow content surface measured
at most 360 px with 16 px padding inside a 1,100 px native window; it was not a
360 px native-window proof. Alt+Right moved X from 296 to 306, and the typed Y
field accepted 150. Entering invalid `{` arguments disabled Save, JSON switching,
and node selection in the fresh accessibility tree; valid
`{"note":"canvas-only"}` restored them. One explicit Save produced exactly one
synthetic `studio.automation/save` receipt and no Run. The separate five source
cases establish real stored/copied revision bytes, not that synthetic GUI
handler.

Captures 01–05 supplied the interaction and narrow-node evidence before the
selected-view visibility correction. Accepted capture 06, after that correction,
shows the Canvas and Grid selected states visibly in the dark view. Hot reload
deliberately remounted a source-changed fixture draft; it did not create another
Save, and the original single synthetic receipt remained the only Save receipt.
No model was loaded.

Those accepted fixture views cover the stated light/dark, grid/freeform,
keyboard/typed-coordinate, validation, and Save presentation only. Pointer
drag, complete screen-reader operation, forced colors, zoom, every theme/window
size, packaged operating systems, trained natural use, and actual tool effects
from a saved file remain unproved. The source tests loaded no model and performed
no host tool effect.

### Captured flow

All screenshots are under `test-results/action-canvas-ui-2026-10-13/`. Each
exact JPEG was saved, reopened and visually inspected in this run.

| Step | View | Observed health and limits |
| --- | --- | --- |
| 1 | `01-grid-light.jpg` — open Create action file | Readable existing editor, named tool node and native controls. Lower fields continue below the initial viewport. This precedes the selected-view CSS correction. |
| 2 | `02-freeform-light.jpg` — add Wait, select Freeform, keyboard-move | Nodes, sequence line, coordinate controls and lower Save are readable. AltRight changed the selected X by10. Separate typed input accepted Y150; native AX setValue alone produced0, so that failed automation input is not the typing proof. |
| 3 | `03-freeform-dark-narrow.jpg` — switch theme and narrow surface | Text and edit controls remain readable. Freeform intentionally scrolls horizontally to distant nodes rather than shrinking labels. A curved line reflects the different Y positions. |
| 4 | `04-grid-dark-narrow.jpg` — scroll to lower controls/receipt | Save and arguments fit the narrow surface; synthetic receipt shows the Save operation. The grid itself is above this capture; this image is lower-control evidence, not a grid-node image. |
| 5 | `05-grid-dark-narrow-nodes.jpg` — scroll back to grid | Nodes stack in one column, controls wrap, labels remain readable. Canvas/Grid pressed states were semantic but not visually differentiated; corrected next. |
| 6 | `06-selected-view-dark.jpg` — affected selected-state correction | Canvas and Grid have visible theme-aware selected styling. Hot reload intentionally remounted the test editor during source editing; the original synthetic receipt remained. This is not an app restart/learned-state preservation test. |

Fresh native accessibility observations separately checked invalid argument
draft→disabled Save/view/selection, valid draft→enabled controls, the exact one
`studio.automation/save` request, and zero Run activity. No screenshot is claimed
for the temporary invalid-draft state. The isolated window was closed and its
Electron process exited0; user settings and saved brains were not accessed.
