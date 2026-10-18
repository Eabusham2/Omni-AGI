# Brain Map: exact weighted records and observed activity

## Confirmed source defects corrected

- Cortical request results and selected-module metadata are now keyed to the
  brain, document revision and complete query. A rejected/new request cannot
  retain the prior brain's selection or trits. Cursor recovery stays paged.
- The DOM scroll position resets with a new viewport, not just the virtual-row
  index. Role, bias, strength, resistance, contribution and repeated ordinals
  remain visible in labels/hover evidence.
- A repeated-contact page is storage, not an independently firing projection.
  Its exact forward owner is supplied by the inspection inventory. The actual
  loaded projection describes its output axis and dimension. Embedding,
  projection and convolution viewports are not inferred from filename regexes.
  Unknown/storage-only owners report unavailable instead of available-but-empty.
- Read-only activity captures use independent IDs. Cleanup of an old capture
  cannot disable the newer capture or an unrelated owner. Polls are serialized
  and stale responses are not displayed. Only actual existing forward hooks
  supply values; inspection never creates a neural model.
- Weight coordinates are described as tensor columns/rows. A stored embedding
  or flattened convolution matrix does not justify a generic causal
  input-neuron→output-neuron explanation. Bias links correctly represent a
  constant-one contribution to the indexed output. Selected link codes now
  truthfully report validation, just like selected element codes.
- Known base parameter slots remain visible when repeated contacts exist.
  Allocated repeated-page slots and page counts are reported separately; no
  whole-payload scan is performed to invent a total of live repeated contacts.
  The complete-logical-inventory flag stays false when that total is unknown.
- Malformed selected token boundaries fail explicitly rather than returning an
  empty page with the same continuation offset. Declared idea/token owner
  relationships now list only owners actually present in the inventory.

## Sparse 2D/3D graph seam

`BrainMapSubstrateGraph` reuses the existing graph theme/layout conventions and
accepts only the current real paged node/contact records. It retains every
passed node; page size is an IO/render envelope, not a total model limit.
Independent repeated contacts and self-contact records receive separate curves.
Original neuronal endpoints remain beside assembly/cluster layout aliases.
Aggregate sign counts are shown as aggregates, not as one learned ternary sign.
Hover/focus evidence includes exact reported strength, contribution, plasticity,
stability, eligibility and repeated-contact identity. Missing fields remain
unreported. Off-page endpoints are not connected to made-up nodes.

The 3D placement is a deterministic visualization layout, not anatomical
location, semantic depth or learned importance. Rotation supports pointer drag,
arrow keys and buttons. Geometry does not use activity to invent firing.
Measured activation requires its explicit observed flag. A scalar activation is
not classified as a spike. Recorded spike-event highlighting requires a genuine
event source; current missing synapse events remain unobserved.

The root integration lane mounts this component in `App.tsx`, preserves the
existing paging/filter/zoom controls, and supplies actual native metadata.
Multiple activation-band clusters in one region are not collapsed last-wins or
connected by invented band-to-band pathways.

## New scoped verification only

- `tests/brainMapEvidencePresentation.test.tsx`: **7 cases / 84 ms**
  (793 ms runner). Server-rendered 2D/3D data includes all 173 provided nodes,
  separate repeated/self contacts, exact weighted metadata, stable/orbiting
  geometry, aggregate aliases, no guessed firing, exact descriptor-driven
  cortical ranges, separated repeated capacity and result-key invalidation.
- `engine.tests.test_cortical_viewer_evidence`: **5 cases / 25 ms**. Tiny
  safe-tensor files and non-neural fake owners establish bias endpoint identity,
  link validation, known-base/repeated capacity reporting, invalid-token paging
  rejection and capture-ID/actual-axis behavior.
- The new component/helpers/test and both affected viewer components passed
  scoped TypeScript compilation. A temporary config explicitly supplied this
  repository's Node type roots after the first temporary-config invocation
  could not resolve `node:crypto`; that was not a product source failure.
  Python compilation of the affected inspection/descriptor/pager/worker files
  also passed.

No model constructor, neural forward, initial training, user corpus, device
capture, closed suite, load benchmark, CI or release ran in this slice. These
checks establish source/render-data contracts, not real trained activity,
whole-model latency, screenshot acceptance or all-platform operation.

## Specific CUA states for the root lane

The root can serve `work/brain-map-evidence/viewport.html` with its existing
local Vite server. Query options are `theme=light|dark`, `width=320` and `dim=3d`.
The fixture is explicitly synthetic and never claims trained activity.

1. Sparse map: select one real node, switch 2D→3D, drag and use arrow/reset
   buttons, return to 2D; selection and external page controls stay usable.
2. Repeated weighted contacts: hover/focus each independent curve and confirm
   original endpoint IDs, sign, strength, plasticity and ordinals. Aggregate
   records must not say their display sign is an individual learned sign.
3. At 320 px width and in light/dark themes, check the dimension/orbit toolbar,
   explanatory scope line, paging controls and selected-node text for overlap.
4. Cortical map: select weight and bias owners, then search/switch groups,
   switch brain or provoke an inspection failure. Old selection/count/trits
   must not remain as if they belonged to the new query.
5. Navigate to a far paged/slot offset, then a new module. Scroll starts at the
   new viewport; tombstone-only repeated pages still allow Next.
6. Observe an already loaded real forward owner, switch viewport/toggle quickly,
   and confirm actual output-axis labels. An unloaded/storage-only owner stays
   unavailable. No brain is loaded solely to inspect it.
7. Until real per-contact spike identity exists, every such edge says firing
   unobserved; nonzero signs or strong weights must never create a firing flash.

Actual LIF/router event capture is a separate follow-through. It must preserve
real owner/channel identity and never assign spikes to semantic UUIDs by
similarity, confidence or an arbitrary activation threshold.

## Focused visual follow-through from the root lane

The root's CUA inspection confirmed an actual min-content sizing defect:
the 320 px fixture grew to 676.921875 px and clipped its toolbar. Removing the
aspect-ratio/min-height width expansion and measuring the actual SVG meet-scale
now keeps the graph within its parent, preserves readable selected labels and
44 screen-px hit targets, and hides only optional narrow unselected text.
Two new sizing-helper cases passed in 2 ms; the earlier 12 cases were not rerun.

At the same narrow state, root observed an actual 318×440 SVG inside the 320 px
fixture and 14 px measured label height; selection and rotation worked. The
desktop dark/right-rotated state then revealed a distinct selected-label/cluster
text collision. Screen-space label/count suppression now reserves selected text
first and retains every real node/contact/title/inspector record. Two new
collision cases passed in 15 ms. Root retested only that affected desktop state:
the selected label was readable, optional overlapping text was suppressed and
all four nodes/six contacts retained accessibility records.

Screenshots are retained in `test-results/brain-map-ui-manual`, including
`01-narrow-before-sizing.jpg`, the accepted 02/03 narrow states,
`04-desktop-dark.jpg` and `05-desktop-label-collision-fixed.jpg`. These are
synthetic real-component visual checks, not an actual model firing or an
all-theme/all-platform UI acceptance claim.
