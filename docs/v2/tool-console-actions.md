# Registered tool actions in the existing console

The existing Tools & permissions page now reads registered action descriptions
instead of leaving newly added tools with an empty default action. This is an
additive read-only metadata endpoint, `tool.descriptions(brainId)`, not a new
tool execution path, permission grant, worker request or hidden model input.

Builtin and currently registered external actions are described even when the
tool is Off. A retained permission for an unregistered tool is labelled as such;
it cannot gain a working runner merely by appearing in the list. Returned
schemas/actions are isolated copies, and actual execution always uses the
ordinary current-policy/approval checks.

Known actions appear in a select. When available, the default is a real
no-operand inspection action such as capabilities; otherwise a declared action
is offered, never an invented operation. Required inputs are shown in an
expandable schema. They are not filled with pretend data. Selecting/inspecting
does not run anything, change permissions or overwrite the arguments draft.
Large schema text is materialized only after expansion.

The existing console field styling now includes select controls. Its input,
textarea, select and field-label text use 12 px rather than the prior smaller
console sizing; this is a scoped readability fix, not a workspace redesign.

## Exact evidence

- Three new metadata/presentation cases passed in 8 ms: Off descriptions do not
  mutate authority, copies do not alias host schemas, defaults choose declared
  inspection, and closed schema details avoid eager materialization.
- One preload case passed in 4 ms: only the owner ID goes to the read endpoint,
  never an execute/grant invocation.
- One constructor-free actual BrainService method case passed in 3 ms: its
  permission-metadata reader is used once, with no engine/provider request or
  grant projection.
- Renderer TypeScript passed at this source point. The two then-in-flight
  clipboard child-stream typing issues were later repaired in the separate
  clipboard lane; this historical picker check does not relabel a later
  whole-tree Node compile as green.

Computer use exercised the actual picker with synthetic registry/button
receipts only: the new VM action default was capabilities; create exposed its
real-shaped required fields without invoking anything; status closed old
schema details and kept the typed JSON draft. Only the explicit synthetic Run
increased its counter. Dark 360 px controls and below-fold content remained
readable. The first light diagnostic showed the unstyled select; the scoped
style correction is visible in `02-dark-corrected-select.jpg` and
`03-action-change-draft.jpg` under `test-results/tool-action-picker-ui-manual/`.

Full live app registration/permission churn, packaged cross-platform UI and
actual learned tool choice remain separate acceptance. No tool, VM, clipboard,
model, provider or user file ran or was accessed in this UI fixture.
