# MCP registry, access and transport status

This page records the MCP work implemented for v2 so far. It does not mark the
complete integrations workstream finished.

## Current reconciliation — October 13

Later focused slices supersede older wording below that treated Apps, account
controls, or presets as source-only future work:

- Completed matching chat actions now mount `McpAppView` through the verified
  current-owner projection. `McpInteractionPanel` is mounted outside inert
  workspace regions, and the existing Tools surface mounts authentication,
  credential-replacement and inert preset controls.
- A bounded real Electron check passed the dedicated App protocol, opaque inner
  sandbox, no-Node/no-preload boundary, blocked undeclared network request,
  same-ID RPC deduplication, explicit Close and fresh-handle reopen without
  replaying the original tool. This is not every-provider/platform/theme or a
  universal sandbox certification.
- Brain-owned App capture/resource artifacts and reviewed data have focused
  fork/export/import/snapshot/recovery/full-merge custody evidence. Restored
  artifacts remain inert: credentials, sessions, handles, approvals and
  completed effects are not revived or replayed.
- Reviewed text/structured App observations have a genuine bounded native
  ordinary-experience/save boundary and source-bound text follow-up mechanism.
  Image/audio sensory admission has partial actual native evidence with a
  truthful `reloadRequired` failure, not a completed live-adapter proof.

Historical test chronology later on this page is preserved as chronology. The
current remaining scope is stated under **Still open now** and in
[the interaction/App report](mcp-interactions-apps.md).

## October 15 standard host continuation

Explicit device-local root selection/review and roots/list/root-change
notifications now have a real host path. No home, brain or application directory
is automatic; disclosure is exact-session/current-policy bound, with Off
fail-closed and fresh review after retirement. Negotiated logging/setLevel,
bounded/redacted operational server/App diagnostics and exact ready-view tool/
resource list-change forwarding are source-wired. Logs are not chat, learning,
hidden prompts or exported artifacts; notifications never invoke a tool.
See [the standards evidence boundary](mcp-standard-host-services.md) for sixteen
new scoped cases and remaining UI/provider/platform evidence. Fullscreen/PIP,
raw devices and broader sampling remain unsupported. The later task slice below
supersedes the earlier absence of a durable-task consumer.

## October 17 dated durable-task continuation

Outbound server-owned tasks now have a genuine negotiated **2025-11-25**
consumer: required/optional/forbidden tool semantics, original remote tool IDs
and grants, committed creation/unknown/ACK custody, task status/result/cancel,
and account-scoped list data without adopting foreign task IDs. There is no
2026 Tasks extension draft or client task sampling/elicitation receiver.

The existing Tools screen mounts explicit local metadata paging with dormant
versus current-session labels. Remote management uses ordinary `tool.execute`
and exact Ask/reviewer/current-account checks. Stop waiting aborts only the local
wait; remote Cancel is separate. Lost ACKs do not trigger resends or unproved
rebinds. Chat labels distinguish accepted work from final results, including
main-owned typed custody on a failed/unknown wait. Final source receipts remain
stable while distinct explicit reads retain ordinary separate experience IDs.

Protocol/store/client/owned-stdio and presentation/failure-host checks have
scoped source evidence. Native interactive task-UI acceptance is parent-owned
and in flight; this does not establish real provider/account, model-learning,
all-platform or package success. See [the exact task contract](mcp-durable-tasks.md)
and [source-original portable custody](mcp-task-portability.md).

## October 18 App display modes

The existing Electron App card now supports negotiated inline/fullscreen
presentation using the stable Apps `2026-01-26` protocol. Fullscreen expands the
same iframe inside the current window, with Return to chat and Close controls;
no separate App window or device permission is created. Main returns the actual
renderer-acknowledged mode and forwards context changes to that exact view.
PIP remains unavailable. This supersedes earlier inline-only wording; see
[display mode contracts and evidence](mcp-app-display-modes.md).

## Implemented

- One installation registry stores server definitions. New connections default
  to all instances; an instance-only registration remains owned by that instance.
  Summaries expose the scope and owner explicitly.
- Installation, server, instance and per-server instance switches determine
  effective access. A global Off wins over a local On. New instance selections
  inherit the last configured instance's shared-server overrides; otherwise they
  use server defaults. Explicit exclusions and disabled defaults persist.
- Access is enforced at schema registration and immediately before transport
  dispatch, even when an old tool permission remains Auto or Full. The production
  tool executor supplies the calling instance, and cross-instance use is rejected.
- Disabling or removing a connection aborts its pending requests, stops its local
  server and removes available schemas. Registry updates do not wait behind
  remote requests or neural learning. Re-enabling reconnects when needed.
- Existing per-instance registry entries migrate as instance-only connections,
  keeping tool identities and credential references. They do not become shared
  credentials through inheritance.
- HTTP bearer credentials use the existing separate secret store. Credential
  values are excluded from the registry and capability-learning documents.
- Local stdio servers can receive explicitly configured environment variables.
  Encrypted/session-only values live outside every brain; the registry and
  summaries retain variable names only. Launch does not inherit unrelated app
  credentials. Missing/inconsistent stored variables refuse startup rather than
  silently dropping authentication. Removal and failed registration clear the
  credential record. The add form has a collapsed JSON environment field.
- Stdio and Streamable HTTP calls negotiate supported protocol revisions:
  `2025-11-25`, `2025-06-18`, `2025-03-26` and `2024-11-05`. A revision being
  recognized does not mean every optional feature of that revision is supported.
- HTTP handles JSON and split SSE responses, matching request IDs without waiting
  for an indefinitely open stream. Bounds, cancellation and malformed responses
  release their readers. Sessions require valid visible-ASCII identifiers.
- An explicit expired-session HTTP 404 triggers one new handshake and fresh tool
  validation. Unknown network outcomes are not automatically repeated. Stop sends
  a best-effort cancellation notification; it cannot guarantee a remote service
  has already stopped an effect.
- Stdio handles tool-list change notifications, ping and explicit unsupported
  server-request responses. Cancellation is sent to the local server. Exited
  processes reconnect through a new handshake; concurrent callers share startup.
- Tool discovery preserves input/output schemas, descriptions, titles,
  annotations and metadata. Input arguments are validated against supported JSON
  Schema dialects before dispatch, including after schema changes.

## Capability learning in the real application

`McpClientService.attachBrain()` binds available schemas and permissions without
starting a neural worker. `learnCapabilities()` commits the typed capability
document through `BrainService.learnStructuredExperience()` and saves its schema
hash only after that learning call succeeds. Concurrent callers share one write;
changed schemas require a new successful receipt.

The desktop startup in `src/main/index.ts` installs
`ChatActionController.setCapabilityPreparation()`. It runs before `service.chat`
takes the neural writer, covering desktop chat, mobile chat and child agents that
use that controller. Queued structured learning receives the cancellation signal.

Failure leaves `schemaLearning: pending`. Chat still opens without a readiness
quiz. The controller publishes compact host diagnostics through
`ChatStateStreamEvent.notices` and `ChatResult.capabilityWarnings`. Those messages
are not inserted into the model's prompt. Disabled schemas are not learned.

These receipts prove that the learning path was invoked and durably acknowledged;
they do not prove a trained model understands or correctly uses every tool.

Discovery triggered by an inline tool call now registers changed schemas without
waiting for another neural writer or a whole-brain permission write. Waiting
there could deadlock the chat writer that owns the tool call. Unknown new tool
IDs remain Off until a safe boundary persists their default Ask permission. The
learning receipt remains pending until `learnCapabilities()` preparation commits
the actual new document. Explicit registration and manual refresh still perform
their normal permission and schema-learning calls.

## Negotiated resources, prompts and completion

- Declared `resources`, `prompts` and `completions` capabilities generate typed,
  brain-callable wrappers for `resources/list`, `resources/templates/list`,
  `resources/read`, `prompts/list`, `prompts/get` and `completion/complete`.
  Subscribe/unsubscribe wrappers require the separate `resources.subscribe`
  declaration. A resource-only server need not implement `tools/list`.
- These are client-owned RPC operations, not fabricated remote tools. Their
  internal hash identity contains a reserved NUL prefix, which every accepted
  remote tool name forbids, including the supported older revisions. Display
  names remain ordinary strings; the reserved seed is never sent remotely.
  A separate `.client.` identity segment also makes collision separation
  structural rather than relying only on a truncated hash.
  Server-supplied `clientOperation` fields are ignored. The capability curriculum
  records the distinction between client operations and remote tools.
- Both the model and future resource/prompt browser call the existing
  `tool.execute` route using the wrapper's ID. Ask/Off/Auto/Full, argument
  validation, calling-instance ownership, registry switches, cancellation,
  actual-outcome learning and audit use the same `ToolExecutor` boundary. No
  direct-dispatch preload API was added.
- Resource contents and prompt messages return a typed
  `omni-mcp-client-result` envelope with server/instance/transport/protocol,
  requested URI or prompt, observation time, `untrustedTaskData: true` and
  `hiddenPromptUsed: false`. Text, binary blobs, image/audio blocks, embedded
  resources, resource links, original URIs and user/assistant roles remain data.
  A remote message never acquires a system/persona role. Reading a `file:` URI
  through its server does not grant access to that host filesystem path.
- List calls return one validated page, not an assertion that the entire catalog
  was visited. Cursors stay opaque, operation-specific and connection-local;
  repeated cycles, list-change races, reused old-session cursors and malformed
  pages fail explicitly. Cursors are not persisted as recovery tokens.
- Completion validates prompt/resource references and related capabilities,
  includes prior argument context, applies cancellable per-connection pacing,
  and enforces the protocol's maximum 100 suggestions per response.
- Declared list changes and updates for acknowledged subscriptions produce
  read-only `mcp.onCapabilityEvent` observations. Updates received before a
  matching subscribe acknowledgement are held until that acknowledgement;
  unrequested URIs do not become subscriptions. Notifications arrive through
  stdio, HTTP POST response streams and the separately negotiated background
  HTTP GET stream; optional GET 405 is reported without inventing support.
- Credential replacement, disable/removal, connection loss and acknowledged
  native Unload invalidate subscription/session state. The actual app installs
  `installMcpUnloadBoundary` only after the native worker confirms Unload.
  Lifecycle reconnect requirements remain visible. A fresh permitted call may
  establish a session; resource subscriptions require an explicit new request.
  Neither completed effects nor subscriptions are automatically replayed.

Persisted capability declarations are catalog metadata, not continuing remote
authority. A new live session revalidates them before dispatch and removes
wrappers no longer supported. Model competence, a rendered resource browser and
real-provider interoperability are not established by these host fixtures.

## Current authorization and remaining interoperability

Authorization protocol helpers now exist in `mcpOAuthProtocol.ts`: challenge
parsing, resource/issuer discovery URI construction, resource and exact issuer
binding, advertised S256 checks, secure PKCE/state generation, and authorization/
code-exchange parameters. Five focused cases pass. These follow the official
[2025-11-25 authorization specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization).
The `McpOAuthService` now wires these primitives into the real MCP client,
preload and account controls. Protected-resource and OAuth/OIDC discovery use
the advertised challenge/path fallback order, exact resource/issuer checks,
S256, verified network addresses and no credential-carrying redirects.
An explicit login uses an OS browser plus an exclusively loopback callback,
constant-time state validation, single-use code exchange and resource binding.
Static public/confidential clients, optional real hosted metadata documents,
and advertised dynamic public-client registration are supported. The product
does not invent a hosted client metadata URL or a registered client ID.

Tokens/client secrets are in the separate encrypted store, or process-only
storage when no secure OS backend exists. Expiring credentials share a refresh
transaction, persist rotated refresh tokens, and stay bound to their resource.
Disconnect removes local authorization first, then reports remote revocation
failure honestly. Stop/disabling/removal closes pending login listeners.
Authentication-required registration is retained for user sign-in rather than
deleted. A 401/scope challenge does not retry a tool effect or grant access.
The initiating instance's switches still apply; another disabled instance
cannot cancel an otherwise permitted initiating login.

Nine distinct service/production-client cases passed across the affected runs
(the initial seven, production-client case after a fixture URL filter fix,
and a new late-refresh/disconnect race). They include actual
loopback callbacks, malformed state, dynamic registration, scope guidance,
single-flight/rotated refresh, confidential credentials, encrypted storage,
revocation, disabled access and no effect retry. Two rendering cases and the
23 existing registry cases passed. No real provider/account login or paid API
request was made. Provider-specific interoperability remains to be checked;
this is not a claim of universal authentication support.

The account controls were inspected manually through the in-app browser using
the actual React component and a labeled synthetic transport fixture, not a
brain or real login. See [the bounded visual report](mcp-auth-visual.md).

The following remain required work, not unsupported success claims:

- Real provider interoperability for wired OAuth flows, hosted client metadata
  and any service-specific nonstandard authentication adapters. The generic
  protocol, token lifecycle and account controls have the scoped evidence above.
- Provider-specific non-bearer adapters beyond OAuth/client credentials.
- Full App coverage across supported providers, every theme/layout/platform,
  complete accessibility and a UI-driven portability round trip. Baseline chat
  placement, bounded real Electron Open/Close/reopen and durable inert artifact
  custody are already recorded; they are not relabelled open.
- Optional App context updates, media/resource task follow-ups and additional
  standard MCP capabilities listed in
  [the interaction/App boundary report](mcp-interactions-apps.md). Reviewed
  text `ui/message` follow-up is already source-wired and is not relabelled open.
- Richer connection diagnostics and the final visual acceptance audit of all
  registry states. Reviewed inert presets and current authentication/credential
  controls are mounted; selecting a preset still does not install, authenticate,
  connect, grant permission, discover schemas or call a provider.

The production client now advertises form/URL elicitation and reviewed
same-brain text/image/audio-input sampling only when their real main-process
handlers are installed. Sampling tools, video/output media and hidden
cross-server context are not advertised. The App extension is
installed by main composition with a dedicated sandbox/proxy host; completed
matching action cards mount its renderer component. Bounded real Electron
sandbox/Open/Close/fresh-reopen evidence passed, while provider-wide, packaged,
all-platform and full-accessibility acceptance remain separate gates. See
[the interaction/App boundary report](mcp-interactions-apps.md),
[the bounded Electron report](mcp-ui-visual.md), and
[native sampling evidence](native-mcp-sampling.md).

## Session transports and reviewed host interactions

Streamable HTTP background GET, bounded event parsing, event-ID deduplication,
GET-only response resumption, explicit session DELETE and the older separate
HTTP/SSE endpoint transport are implemented. Resumption never resends the
original POST. A lost resumed session reports an unknown outcome instead of
repeating a tool. Older endpoint events must remain same-origin; cross-origin
endpoints require explicit reconfiguration/re-authentication and do not receive
credentials automatically. Reader cancellation and terminal ownership prevent
late stream/session receipts from overwriting a replacement connection.

Elicitation has a separate explicit-review surface, not task-choice assumptions.
Flat primitive forms and string-enum multi-selects validate before sending.
Sensitive credential fields require URL elicitation. Full URLs are shown before
an OS-browser navigation; there is no prefetch, credential inspection, timeout
approval or claim that opening a URL completes authorization. Error `-32042`
also surfaces reviewed URL requests, without automatically retrying the original
effect. Private review receipts live outside brains/exports; submitted form
values are recorded only by hash. Pending reviews cancel after restart rather
than replaying navigation or sampling.

Sampling reviews exact current-task messages before native inference and reviews
the generated text again before returning it to the server. Server systemPrompt
and model preferences are visibly separate review data: they do not become a
hidden persona or select another model. Internal controller turn ownership—not
server arguments—binds an inline sample to the actual task. A real review holds
the network-wait deadline lease; elapsed review time never grants authority.

## Verification

Focused command:

```sh
./node_modules/.bin/vitest run tests/mcpRegistryAccess.test.ts tests/integrationLearning.test.ts --maxWorkers=1
```

The fixtures exercise access inheritance/revocation, stale Full grants,
cross-instance calls, schema-learning failure/retry/deduplication, production
controller preparation, atomic preferences, credential exclusion, protocol
negotiation, SSE cancellation, expired sessions, uncertain effects and real local
stdio subprocess lifecycle. The existing integration file also contains teacher
and secret-store fixtures; its entire count must not be described as MCP-only.

No native brain is constructed or trained by these fixtures. They use host
stubs and synthetic JSON-RPC services. Provider OAuth and real learned tool
competence are not covered by these tests.

`tests/mcpEnvironment.test.ts` adds three cases: strict name/value/record-budget
validation, an actual local Node JSON-RPC subprocess verifying its explicit
credential and the absence of an unrelated parent secret, and failed/wrong-
transport registration cleanup. All three passed in83ms. Values were absent
from registry and neural schema calls. No real service was contacted.

The resource/prompt/completion fixtures are in `tests/mcpCapabilities.test.ts`
and `tests/mcpCapabilityPreload.test.ts`. They exercise actual local stdio
JSON-RPC, HTTP/SSE protocol data, the production `ToolExecutor` approval and
outcome path, multimodal preservation, reserved identity separation, pagination,
restart capability revalidation, subscription/unload/reconnect, cancellation,
and deferred writer-safe schema learning. They use a stubbed neural learner;
no native brain is constructed or trained. A general resource/prompt browser's
visual states and real service accounts remain separate acceptance work.

The current focused slice has 15 capability cases and one preload observer case.
The other affected files contain 52 existing registry, environment, integration,
OAuth service and OAuth protocol cases. In particular, the deferred-writer case
adds a genuinely new tool, independently blocks permission persistence and
schema learning, proves inline dispatch waits on neither, then requires the safe
preparation path to persist Ask and acknowledge the actual changed schema.

Credential replacement now has a typed preload operation and collapsed controls
for existing servers. It replaces/clears the complete explicit environment or
HTTP bearer/account login without re-adding registration, changing tool IDs,
enabling access or dispatching remote calls. Active transports stop and require
explicit reconnection. Failed registry publication restores prior secret values;
if restoration itself fails, the user is told to re-enter them. Secrets are
never returned to the renderer. Two additional stdio replacement/rollback cases
passed (279ms), one OAuth/bearer rollback-and-clear case passed (148ms), and one
new rendering case passed. The later bounded actual-component check covered
local/remote replacement and narrow light wrapping; real provider behavior,
every state/theme and complete accessibility remain in the full visual gate.

## Still open now

- Real provider/account interoperability for generic OAuth, client metadata,
  dynamic registration, bearer/environment credentials and service-specific
  nonstandard authentication. No focused synthetic or local-server fixture
  proves a third-party account works.
- Full production App behavior across providers, packaged operating systems,
  every theme/layout, keyboard/screen-reader accessibility, allowed external
  network origins and a UI-driven portability round trip.
- PIP, raw device grants, view-exposed tools, the 2026 Tasks extension
  draft, client-receiver tasks and sampling tools,
  server/all-server context, video sampling and generated-media output. These
  are not advertised as hidden fallbacks.
- Real provider/App behavior for the newly source-wired roots/logging/list-change
  surfaces. Their bounded host evidence is separate from actual remote service,
  Electron relay/UI, account and packaged-platform acceptance.
- Actual packed ordinary image/audio App learning through the complete live
  adapter, non-text App task follow-ups, slow-language learning and broadly
  useful native tool/App competence. Existing mechanism evidence is bounded as
  described in `native-app-observation*.md`.

## Protocol references

- [MCP transports and session lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
- [MCP authorization](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
- [MCP initialization and capability negotiation](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)
- [MCP resources](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)
- [MCP prompts](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts)
- [MCP completion](https://modelcontextprotocol.io/specification/2025-11-25/server/utilities/completion)
- [MCP pagination](https://modelcontextprotocol.io/specification/2025-11-25/server/utilities/pagination)
- [MCP cancellation](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/cancellation)
- [MCP dated server-owned tasks](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/tasks)
