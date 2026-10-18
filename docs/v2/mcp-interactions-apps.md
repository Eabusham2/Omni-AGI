# MCP session, human-review and App boundary report

This is a scoped implementation report, not universal provider or whole-product
acceptance. No account login, paid API, corpus training, CI or release was run by
this lane.

October 15 standards continuation: selected-root disclosure, standard server and
App diagnostics, negotiated logging/setLevel, and exact ready-view tool/resource
list-change forwarding now have installed host paths. See
[the bounded source report](mcp-standard-host-services.md) for sixteen new cases,
current-session/permission/redaction fences and remaining UI/provider evidence.
This slice does not claim raw-device grants or broader sampling support. The
later [durable-task report](mcp-durable-tasks.md) separately implements outbound
server-owned tasks for the exact negotiated 2025-11-25 protocol, not the
incompatible 2026 Tasks extension draft or a client-receiver task service.

## Implemented source

- `mcpEventStream.ts`, `mcpStream.ts`, `mcpLegacySse.ts`, `mcpRequestDeadline.ts`:
  bounded SSE frames, opaque event IDs, background GET, GET-only result recovery,
  optional 405 reporting, same-origin legacy endpoint negotiation and explicit
  session DELETE receipts. Unknown tool effects and subscriptions are not replayed.
- `mcpInteractions.ts`, `mcpInteractionProtocol.ts`, `McpInteractionPanel.tsx`:
  private durable review receipts; explicit accept/decline/cancel; flat typed
  forms; validated enum multi-select; full-URL consent; known-epoch completion
  notifications; URL-required errors without original-effect retry.
  Main-owned App action/network/link approvals use this same scrollable review
  surface, with exact arguments/origins/URL, rather than oversized native alerts.
- Same-brain sampling uses the dedicated `EngineSupervisor.sampleNativeMcp`
  adapter and exact reviewed JSON/hash. Input/output reviews remain independent.
  Image/audio blocks preserve exact bytes and order during text edits. Review
  shows MIME, source bytes and canonical-payload fingerprint without decoding or
  autoplaying untrusted media. Native acknowledgements must cover every supplied
  block and match the raw-source hash, owner and private/no-mutation receipt.
  Native decoder/platform evidence is maintained in `native-mcp-sampling.md`.
- `mcpApps.ts`, `mcpAppMetadata.ts`, `mcpAppProxy.ts`, `McpAppView.tsx`:
  a main-owned completed-call descriptor, exact permitted `ui://` resource read,
  strict HTML-profile MIME and original URI/provenance, double iframe relay,
  initialization/initialized ordering, input-before-result delivery, same-server
  tool/resource mediation, explicit exact-origin network/link review and close.
- App-only tools do not enter model schema registration or the capability
  curriculum. Apps can call them only through a valid main-owned view and the
  existing `ToolExecutor` permissions, argument checks, actual-outcome retention
  and audit. Model-only tools remain unavailable to Apps. Remote result metadata
  cannot fabricate the reserved client App descriptor.
- Requests bind brain/server/connection epoch/view token/phase. Duplicate IDs
  share the original receipt; changed payloads with the same ID fail. Reopening
  an explicitly closed view rereads its UI resource, not its original completed
  tool. Disconnect, credential changes, disable/removal and actual Unload revoke
  the old view; no automatic subscription or uncertain-effect replay occurs.
- `mcpAppArtifacts.ts`, `mcpAppPresentation.ts`, `fullMergeMcpApps.ts`:
  brain-owned inert, content-hashed original call/result/HTML artifacts survive
  the existing workspace fork/export/import/snapshot/recovery paths. Full merge
  reads the lease-consistent sealed workspace, dedups identical content/shared
  ancestry, and retains conflicting source/resource versions for explicit
  mapping review. Recovery keeps later real App evidence without replaying it.
  Original tool outputs, including opaque nested user `*Id` values, stay verbatim;
  a separately verified `ActionEvent.mcpAppView` projects the current owner.
  Durable artifact IDs are distinct from fresh per-open URL handles and tokens.
  Old document/network handles expire on Close, Unload or connection loss.
- `mcpAppTaskData.ts` and the separate `app-data` review retain exact proposed
  user task/content/structured data, resource URIs and media order/bytes.
  Root system/persona/owner/approval fields and forged assistant messages fail.
  Approved data enters a bounded pending-byte queue with immutable source/review
  hashes and an ordered predecessor chain. Later current-context updates
  supersede pending older versions without erasing their audit evidence.
  Copies/merges/recovery make pending work dormant rather than transferring a
  grant or replaying it. A visible App card labels the initial queue receipt
  `neuralAdmission: not-yet-admitted`; that is **not** a neural-learning receipt.
  Genuine text/structured admission now updates it only after the loaded serial
  native ordinary-experience/save boundary returns an exact hash-bound ACK.
- Main registers only `omni-mcp-app:` as an additional Studio frame source.
  The outer trusted relay uses a separate origin and `allow-scripts
  allow-same-origin`; the untrusted inner view uses `allow-scripts` only, an
  opaque origin and a main-owned CSP. Neither gets the Studio preload. IPC checks
  the actual trusted WebContents, exact main-frame object and renderer document.
  App-origin network traffic strips ambient cookie/auth/referrer headers; response
  cookies/site-clearing headers are stripped. Request identity remains tracked
  even after the originating frame disappears. Self-navigation back into the
  trusted relay and unapproved nested frames are denied.

## Root renderer integration contract

Main retains actual completed call evidence in the ordinary tool result without
rewriting it. For a completed matching `ActionEvent`, prefer the verified
current-owner `event.mcpAppView`; the raw current-instance hint remains at:

```ts
event.execution?.output?._meta?.["omni/mcpApp"]
```

The root-owned action card now renders:

```tsx
<McpAppView brainId={event.brainId} viewId={descriptor.id} />
```

The component has stable props `{ brainId, viewId, api? }`. Its API is
`window.omni.mcp.apps.describe/open/request/close`; only the trusted Studio renderer has
that API. Opening is an explicit user action, not an iframe gaining dispatch.
Do not trust a descriptor supplied in a renderer request: main looks it up in
its own completed-call artifact registry. `describe` is read-only; a source
conflict requires explicit selected-capture review. Reopen does not recreate a
live session or old network/tool grant from stored bytes. The current installed
source must match the original endpoint; reconfiguration/authentication remains
explicit. Initialization sends the stable `ui/notifications/tool-input` before
`ui/notifications/tool-result` only after the App's initialized notification.

The root already placed `McpInteractionPanel` outside inert chat/view regions.
The review panel now uses Studio `.button` variants and themed rounded form
controls. The existing `tests/e2e/mcp-review-fixture.html` is synthetic UI only;
its screenshots are not a real provider/native-competence claim.

Completed matching action cards also mount `McpAppView` in current source.
Root's bounded real Electron host check passed the double-sandbox handshake,
blocked undeclared network access, same-ID RPC deduplication, explicit Close,
and fresh per-open handle/origin reopen without replaying the original tool.
Durable artifact custody separately covers current-owner projection and inert
fork/export/import/snapshot/recovery/full-merge closure. These closed slices do
not establish every provider, platform, theme, accessibility state or a full
UI-driven portability round trip.

## Ordinary App data and genuine follow-ups

`McpAppObservationDelivery` runs event-driven work outside inline MCP execution,
under the existing brain writer. It never starts/replaces a worker, implicitly
Loads, reenters a chat's held writer or uses fake `ChatToolObservation` receipts.
`app_observation_frontier` and `admit_app_observation` execute on the native serial
dispatcher, not its flags-only stdin reader. Exact raw record/data JSON hashes,
FIFO predecessor/ordinal and compare-and-swap against one bounded saved native
head bind every admission. Superseded rows advance progress without claiming
learning; an exact current-head lost-ACK retry never learns twice.

The native adapter calls the genuine `learn_experience(...,kind="experience",
source="mcp-app",steps=0)` working/adaptive-memory path and atomically saves its
head with the neural checkpoint. Text, structured values and embedded text
resources use that ordinary path. The source now also integrates ordered inline
image/audio blocks and image/audio binary resources through the same brain's
sensory regions before one ordinary experience admission. Links remain literal
references, not automatic fetches; unsupported binary formats fail explicitly.
Neither base64 text nor private media *sampling* is mislabeled media learning.
Actual packed ordinary-media admission is still a separate native evidence gate;
constructor-free encoder/codec fixtures and host checks do not close it. See
`native-app-observation.md` for exact native fixture and save boundaries.

The one later actual packed media attempt persisted genuine sensory-state
growth, one experience and complete image/audio audit coverage, but returned
reloadRequired and failed the full live adapter/duplicate assertion. Its saved
files were inspected without a model; no retry ran. This is partial mechanism
evidence, not a full integration pass or slow language-weight update. Exact
boundaries are in `native-app-observation-media.md`.

Mixed sensory/reference admissions have one bounded native summary and a
content-addressed current-ACK JSONL audit. Each row binds the real source block,
raw bytes/hash, order and actual frame/window/sample coverage. Main streams and
verifies those rows against the exact approved data, copies the raw audit into
the immutable brain-owned App ledger, and only then acknowledges admission.
Before replacing that native head, the next claim must supply its durably
archived audit hash. An exact lost-ACK retry returns the existing audit without
decoding or learning again. Current native audit custody also survives
snapshot/fork/export/import/recovery; historical audits remain in the immutable
App ledger and full-merge parent closure. Owner rebinding preserves original
source IDs/bytes and never restores live handles or grants.

A reviewed text `ui/message` then starts one ordinary chat follow-up using its
actual body, not an invented prefix or hidden prompt. Main derives the internal
`NativeAppChatInput` from the owned approved ledger and actual native ACK. Native
verifies exact record/data/capture bytes and the current head before accepting
the task, reuses its already-admitted experience/afterimage, and records truthful
App source metadata rather than a second fast human-authored experience.
Canonical decoder user-role tokens are retained; `ChatMessage.inputProvenance`
and the input receipt identify `serverLabel · App` in the root UI. No renderer,
remote tool schema or model argument can submit that internal authority proof.

Follow-up dispatch intent is durable before generation. Unknown outcomes are not
resubmitted automatically. A real committed controller receipt can reconcile
the original turn without replaying effects. Text-only follow-up scope is
explicit: an empty/media/resource task is not replaced by a fabricated prompt.
Native save/cleanup owner release invalidates the current instance's admission
and old App handles; subsequent work waits for explicit Load, not preflight
autoload. Read-only native receipt events update the visible App card.
If disk/archive verification fails after a worker-confirmed release, that owner
is still invalidated; an incomplete host ACK never fabricates a ready instance.
Foreground preemption cooperatively cancels only the native read phase. An
entered ordinary mutation/save settles and returns its applied receipt instead
of killing the warm worker or falsely reporting that nothing happened. A
correlated safe pre-mutation cancellation keeps its exact claim pending until a
later readiness event, without a timer retry loop.

## Focused verification

Protocol fixtures exercise the production MCP client, ToolExecutor and private
review service with labeled synthetic transport/native outputs. They establish
host permission/order/ownership behavior, not trained-model competence.

- `mcpSessionTransport.test.ts`: background notifications, dedup, cancellation,
  DELETE, UTF-8 event IDs, GET resume, resumed-session expiry without POST replay,
  malformed streams, old SSE and cross-origin credential refusal.
- `mcpHostRequests.test.ts`: actual server-origin form/sampling requests,
  internal turn binding, continuing cancellation reads and URL-required errors.
- `mcpInteractions.test.ts`: schema validation, secrets/URL rejection, private
  receipts, no navigation retry after receipt failure, restart cancellation,
  explicit input/output review, reviewed media immutability and timeout leases.
- `mcpApps.test.ts`: exact completed provenance/resource permissions; App-only
  schema exclusion; before-ready/owner/token/visibility denial; same-ID effect
  dedup; network/link consent; CSP/resource checks; iframe/self-navigation and
  main-frame IPC; ambient credential stripping; explicit resource-only reopen;
  actual production-client/ToolExecutor restart and copied-owner reopen with
  distinct URL/token identities, plus real App data review/cancellation and an
  honest pending receipt with no neural-learning call.
- `mcpAppArtifacts.test.ts`: exact original evidence, owner-only copy rebinding,
  existing workspace closure after original removal, conflicting lineage,
  verified projections, sealed-parent merge, corrupt-blob/schema rejection,
  queue hash chains, supersession, dedup/backpressure and dormant copied work.
- `mcpAppTaskData.test.ts`: exact media/resource/structured-value retention and
  persona/authority/role/schema rejection without fetching or adopting content.
- Node and renderer TypeScript are checked separately. Neither compilation nor
  these fixtures substitutes for final packaged/Electron browser acceptance.

Latest scoped results: 111 tests across 13 MCP/integration files passed in
6.58 seconds before the final App-review addition. The affected 36 tests across
five files then passed in 5.86 seconds, including the new App-authority review
and static renderer fixtures. These runs overlap and must not be summed. Node
and renderer TypeScript and the owned-path whitespace diff check passed. Static
renderer output proves structure/escaping, not readable colors or browser focus.

Durability continuation: nine App-host cases passed before the separate data
review addition; seven artifact-closure cases, four existing workspace cases
(isolated, unchanged timeouts), and one real full-merge host opaque-output case
passed. The new data slice then passed four parser/queue cases and two affected
production-host/static-panel cases. These overlapping scoped runs are not an
aggregate whole-product count. Node TypeScript passed; renderer checking is
tracked separately. Root's bounded actual Electron Open/Close/reopen smoke
confirmed distinct per-open origins with only UI resource rereads, not completed
tool replay. See `mcp-ui-visual.md` for its precise visual/runtime scope.

The main integration subsequently passed fourteen affected artifact/delivery
cases, eight loaded-only engine/delivery cases, and five affected provenance/
owner-release cases (overlapping runs, not an aggregate). These use clearly
labeled synthetic native gateways/processes and verify host protocol decisions,
not a learned model's intelligence. The native owner's actual19.069-second
ordinary-admission/source/save-reload fixture passed. A subsequent strengthened
packed final-reload check reached accepted source state but timed out at30s;
it is **not** a full pass and was not rerun. Compilation and actual native scope
are reported separately from broad packaged/provider acceptance.

The current-audit continuation passed sixteen affected host archive/delivery/
supervisor cases in 4.81s, including safe read preemption versus a late applied
ACK and owner release despite archive failure. A further unknown-publication
outcome case passed without retry or a false "nothing applied" claim. Five
audit source/coverage cases and five static review/card cases passed in 1.80s;
channel/hash checks are tracked separately. One real
filesystem `BrainRepository` snapshot→fork→export/import→recovery fixture passed
in 1.729s, including missing-current-audit rejection; it performs no model
training, decoding or chat. These results overlap earlier runs and are not an
aggregate competence claim. The main audit fixtures explicitly use synthetic
native measurements; native media encoding/admission evidence is separate.

## Still open; do not silently call these supported

1. Full production chat/App placement across every theme/layout, direct allowed
   network/provider compatibility, complete accessibility, all-platform sandbox
   behavior and a UI-driven portability round trip. Root placement and bounded
   real double-sandbox/Close/reopen checks are done; they are not every-provider
   or every-platform acceptance.
2. Actual packed ordinary image/audio App observation learning, non-text task
   follow-ups, and full packaged native chat/UI portability acceptance remain
   open. Ordinary sensory integration and recoverable per-block audit custody
   are source-wired with scoped codec/host/filesystem checks; text/structured
   admission and source-bound text follow-up have the distinct proofs above.
   This is not all-modal/all-platform trained-model competence.
3. PIP, raw camera/microphone/location/clipboard grants and view-exposed tools
   are not advertised. Negotiated inline/fullscreen now uses the current
   Electron window and the same iframe, with actual renderer acknowledgements
   and exact-view host context events. Its nine scoped source contracts are
   recorded in [the display report](mcp-app-display-modes.md); direct Electron,
   theme, keyboard and packaged-platform acceptance remain separate UI gates.
   App logging and negotiated tool/resource change forwarding have bounded
   source evidence, not every-provider/packaged/UI qualification. Wildcard domain requests fail with
   an explicit exact-origin requirement, not an invisible broad grant.
4. Sampling tool calls, server/all-server context and output-media sampling are
   not advertised. User input/output review and basic text/media-input private
   native inference are distinct from those optional capabilities.
5. The 2026 Tasks extension draft, client-receiver tasks and provider-specific
   nonstandard authentication adapters remain outside this supported boundary.
   Dated 2025-11-25 outbound server-owned tasks now have their own genuine
   negotiated implementation and scoped evidence in mcp-durable-tasks.md;
   real provider/account and packaged task acceptance remain separate.
   Standard roots and
   logging now have exact reviewed/operational host paths described above. Basic
   explicit bearer, stdio environment credentials and the generic OAuth/client
   lifecycle are implemented with scoped evidence; that does not prove every
   MCP server or packaged OS/codec combination works.

## Primary references

- [MCP 2025-11-25 transports](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
- [MCP 2025-11-25 elicitation](https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation)
- [MCP 2025-11-25 sampling](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling)
- [Official MCP Apps 2026-01-26 specification](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx)
- [Electron frame-navigation events](https://www.electronjs.org/docs/latest/api/web-contents#event-will-frame-navigate)

These implementations follow protocol ideas; they do not vendor the SDK or claim
that external providers, permissions or user intent are established by metadata.
