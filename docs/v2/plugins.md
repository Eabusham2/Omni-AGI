# Omni AGI Studio plugin and custom-tool SDK

Updated 2026-10-18. Omni plugins combine immutable declarations with optional
isolated panels, feature-runtime modules, themes, tools and source parsers.
This SDK describes supported interfaces in current source. It does not grant
arbitrary main-process/native execution or certify third-party packages,
all-platform isolation, trained competence or every possible extension hook.

## Package and catalog

An Omni plugin is UTF-8 JSON with:

- `format: "omni-plugin-package"` and `formatVersion: 1`;
- a lowercase package ID and semantic version;
- name, author, license, documentation, and optional HTTPS homepage;
- exact nested capability declarations;
- zero or more custom tools, static panels, isolated interactive panels,
  `runtimeModules`, theme-token sets, and training-format declarations;
- an optional parent SHA-256 when an edit creates a new version.

The installation-private catalog is stored outside every brain and outside the
automatic file-tool scope. Canonical package bytes are content addressed by
SHA-256. Reusing one package ID/version with different content is rejected.
Editing never mutates an installed version: it creates a new semantic version
whose parent hash names the reviewed base. Installing a new version does not
activate it. Activating a new content hash turns the global toggle off so that
the exact new version must be reviewed before use. A global toggle submission
also compare-and-sets the displayed active hash and revision, so a stale Enable
cannot authorize a concurrently activated version. Merely storing an inactive
version does not revoke the unchanged active parser/learning lease.

Import never runs a repository setup script, package manager, binary, Node
module, or Python module. Export publishes the canonical declarative package;
an explicitly confirmed replacement uses a private staged file and rollback.
Credential-like immutable argument keys are rejected during package
normalization; secrets belong to separately reviewed installation credential
stores/integrations and are never embedded in a plugin package.

## Author an executable module in the app

Open **Tools & permissions → Plugins → Plugin maker**. Its draft editor offers:

- **Feature & menu**: editable JavaScript with lifecycle/event callbacks and
  declared buttons, menus and panels;
- **App styling**: a recognized CSS draft plus its complete JavaScript envelope;
- **Custom tool**: an editable typed runtime-tool implementation;
- **Training format**: a streaming UTF-8-to-JSONL example and a bound format
  declaration, not a pre-authorized parser.

You can add multiple modules, select one to edit, change package/module metadata,
and expand **Events, placements & tools** or **Complete package JSON**. Host
capabilities, parser bindings and unrelated package sections remain in that JSON.
The CSS field edits only a recognized data envelope without evaluating its
JavaScript. Other module bodies remain editable as JavaScript.

**Add package** and **Save new version** use the existing host canonical
validation, immutable version/hash and edit-lineage rules. Editing, adding a
starter or saving does not run it. Removing a module from the draft does not
stop an installed active version. Review package/instance access and explicitly
activate the exact runtime module before execution; **Stop** retires that
session. The maker does not supply raw Node, filesystem, network or credentials.
Module host calls still require declared capabilities and current nested
permissions. An editable example is not a tested third-party feature or proof
of hostile-code security on every packaged platform.

## Custom tools and nested permissions

A v1 custom action delegates to exactly one existing typed Studio tool/action.
Its argument binding can forward all input fields or an explicit top-level
allowlist, then apply immutable package constants. Recursive `plugin.*`
delegation is rejected.

Execution requires all of these to remain current:

1. the globally enabled package and active immutable hash;
2. an explicit enablement for this brain and exact content hash;
3. an explicit enabled selection for the declared nested capability;
4. the custom tool's ordinary Off/Ask/Auto/Full setting;
5. the nested built-in or MCP tool's own current permission.

The stricter result wins. Outer Full never bypasses nested Off or Ask. Risky
nested Auto operations—such as shell/device activity or file access outside
the instance workspace—still ask. An Ask card carries the resolved nested tool,
action, exact resolved arguments, package hash, and both permission levels.
The opaque one-use approval authority additionally binds catalog, instance,
outer-permission and nested-permission revisions without exposing them as
grant-bearing UI data. Changing the package, global/instance toggle, arguments,
or either permission invalidates the approval.

Global activation, global enable/disable, and per-instance access publish a
synchronous mutation epoch before touching the catalog. Format leases register
before their asynchronous validation and must retain both the global and
instance epoch; an in-flight or newly started mutation either refuses or aborts
the lease. This closes the validation/register gap at the cooperative parser
boundary.

Changing instance access is a serialized staged review, not a claimed atomic
transaction across the catalog and policy database. The resolver is disabled
first. Each policy write is compare-and-set against the current host revision.
On failure, rollback touches only the exact revision written by that operation;
a newer human Off or another edit is never overwritten. Failed stages remain
visible and disabled until reviewed again.

ToolExecutor tracks both the outer and nested capability IDs while validation
is in flight. A relevant policy change aborts the shared signal. If an effect
already started, `dispatchStarted` stays true and the actual or partial outcome
uses the normal durable receipt path; it is not rewritten as “never ran.”

## Presentation and data declarations

Packages can store:

- static HTML panels;
- panel CSS without imports or URL loads;
- named CSS custom-property theme tokens;
- training-format extension/media-type metadata; a custom filename extension
  may be inert, bound to one existing built-in streaming parser, or use the
  reviewed pure WASM-to-JSONL or exact Runtime JS stream boundary described
  below.

Static HTML rejects scripts, event handlers, navigation/resource attributes,
forms, embedded frames/objects, and active metadata. `PluginStaticPanel` loads
only `omni-plugin-panel://<content-hash>/<package>/<panel>`; main revalidates the
registered immutable package/panel and serves it through a dedicated protocol.
The iframe has an empty `sandbox` attribute and the response/header CSP is
`default-src 'none'`; only inline style is allowed. Script, connections,
images/media/fonts, frames, workers, objects, forms, base URLs, navigation,
preload, credentials, filesystem, devices, and main-process bridges are absent.
Regex validation is defense in depth, not the isolation boundary. This is not
the separate MCP App sandbox and not a general JavaScript sandbox.

Training-format entries may remain `declaration-only`, or select
`{ "kind": "builtin-stream", "format": ... }`. The latter can map a custom
filename extension to Omni's existing text, JSON/JSONL, CSV/TSV,
Parquet/Arrow, SQLite, archive/manifest, PDF, image, audio, or video stream. It
does not install parser code. The dataset manifest binds the package hash,
format ID, selected parser, catalog revision, instance revision, selected
source checksum (or the normal transactional SQLite snapshot), and normal
exhaustive cursor/coverage. More than one enabled
match is rejected. A package/access change aborts the active cooperative
request so the uncommitted suffix remains paused at its durable record/batch
boundary; it is not called 100% complete or silently skipped.
Duplicate admission binds both the raw source checksum and the immutable
plugin/parser identity. The same bytes under a different reviewed parser run
that parser instead of borrowing an older content-only receipt; the same bytes
under the same parser may reuse the matching completed source receipt.

A format may now instead select an immutable import-free
`{ "kind": "wasm-stream", "moduleId": ..., "output": "jsonl" }` module.
Its exact package must declare and the instance must enable the reserved
`data.training-parser` capability for that format. Studio streams manifest-owned
bytes through a killable resource-admitted Worker with one batch in flight,
validates complete JSONL output, and binds source/module/output provenance into
the durable receipt before native ingestion. The module receives no filesystem,
network, Node, credential, or tool bridge. See
[Executable training-format parsers](executable-training-parsers.md).

A format may also select one exact package `runtimeModules` entry:

```json
{
  "kind": "runtime-stream",
  "moduleId": "source-transformer",
  "output": "jsonl"
}
```

For typed media output, replace `"jsonl"` with
`{ "kind": "media-assets", "mediaTypes": ["image/png", "audio/wav", "video/mp4"] }`
using only supported declared MIME types. Runtime parser modules use
`globalThis.omniRuntime.registerParser({ push, drain })` during a fresh exact
training-parser session. The host, not module code, opens the selected source
and sends bounded ordered byte pages. JSONL handlers return bounded output
pages; media handlers return ordered asset start/chunk/end records whose raw
bytes, kind, MIME, size, container signature, and SHA are verified before a
media-only native archive is committed. Media is not converted into captions
or base64 text examples. A complete zero-asset result explicitly schedules no
modality update.

The training-parser session cannot mount UI, publish a dynamic tool, use general
module state, see raw paths, or reuse an ordinary activated module session. A
parser may call only the package capabilities enabled for this exact instance;
every call enters current nested ToolExecutor policy. Ask waits in the
Studio-owned review surface, including before first Chat. The module never
receives an approval token and cannot approve itself. Host call identities are transaction/ordinal-bound,
so a restart cannot silently repeat an uncertain dispatched effect. Parser
preparation itself does not require or implicitly Load the neural worker.

**Training & Evolution** uses automatic format detection; it has no next-file
parser picker. Automatic preserves built-in detection for known formats; otherwise it may
use one uniquely matched enabled custom extension. A MIME-only declaration
does not automatically opt a source into plugin execution. To use a MIME-only
format or override a known built-in format, an explicit typed request must name
the exact reviewed plugin/parser entry. That selection binds
`{pluginId, contentSha256, formatId}`; it is not a
free-form parser name. Extension or observed MIME must still match that
declaration. Local signature evidence and HTTP Content-Type retain their
actual provenance, and source bytes/MIME/access are rechecked at start/resume.
The manifest and typed web/crawl contracts still support that explicit parser
selection where required; removing its UI picker does not grant parser access.
Ambiguity, a stale hash,
missing access or a mismatched MIME fails rather than choosing a different
parser. No parser choice itself starts learning.

After an exact package is enabled for a brain, Studio also records a durable,
content-hash-bound ordinary-learning job for its public documentation, typed
tool/action schemas, declared capabilities, format declarations, and the
interactive panel's public id/title/tool allowlist. HTML, CSS, and executable
script bytes are not inserted into neural context or treated as tool results. The
enable operation does not wait behind a live neural writer. Learning starts
only when that same instance is already loaded, never by implicitly loading or
spawning a model, and is complete only after the native learner returns the
matching training-source receipt. It contains no fabricated tool execution or
result, persona, behavioral prompt, or hidden prompt. This proves the queue and
receipt contract, not that a small or untrained brain will use the new tool
reliably.

The semantic row is inserted in the same catalog transaction that publishes
per-instance executable access. Global Enable atomically creates any missing
rows for every already-enabled matching instance before it commits. Startup
also reconciles installations created by older builds. A process exit therefore
cannot leave callable access without a discoverable pending job; lost neural
acknowledgements retry by content hash and complete from the actual source
receipt rather than inventing success.

An optional `interactivePanels` entry adds local HTML controls or a canvas,
CSS, one JavaScript program, and an exact allowlist of package-local
tool/actions. It does not turn the static-panel path into an active surface.
Main creates a fresh 128-bit view handle and 256-bit channel nonce for every
Open and binds them to the brain, immutable package hash, catalog revision,
instance revision, and panel. Neither value is saved in the package or brain.

```json
{
  "id": "viewer",
  "title": "Local viewer",
  "html": "<button id=\"read\">Read</button><canvas id=\"view\"></canvas>",
  "css": "canvas { width: 100%; }",
  "script": "document.querySelector('#read').addEventListener('click', async () => { const value = await omniPlugin.call('reader', 'read', { path: 'note.txt' }); document.querySelector('#view').dataset.state = value ? 'complete' : 'empty'; });",
  "tools": [{ "toolId": "reader", "actions": ["read"] }]
}
```

`toolId` is package-local and must name a tool/action already declared in the
same immutable manifest. `arguments` must be structured JSON and is bounded by
the panel transport. The returned promise resolves with the real completed
tool output, rejects on denial/failure, and waits while trusted Studio displays
an Ask review.

Interactive panels load from `omni-plugin-app://<view-id>/index.html` inside an
iframe with `sandbox="allow-scripts"` and no `allow-same-origin`. Main serves
only the registered document. Its CSP denies default/network/image/media/font/
frame/worker/object/form/navigation access, permits no dynamic evaluation, and
authorizes the one inserted program by SHA-256. Package HTML still rejects
scripts, event attributes, resources, frames, and forms. Script source with a
closing `script` element is rejected before hashing, so package text cannot
break out of the authorized element. The child has no preload, Node APIs,
Studio DOM, filesystem, credentials, devices, or main-process object.

The child receives a frozen `window.omniPlugin` API. `call(localToolId,
action, arguments)` posts structured data to the trusted Studio relay. Studio
requires the exact iframe `event.source`, opaque `null` origin, channel nonce,
schema, and request identity; main repeats those checks and retains every
request ID for the lifetime of the view. An exact replay receives the retained
receipt and changed operands are rejected. Main resolves only an action in the
panel allowlist and sends the resulting `plugin.<package>.<tool>` invocation
through ordinary `ToolExecutor`, including outer and nested permission,
risky-Auto, intent, cancellation, audit, and learning paths.

If ToolExecutor returns Ask, the child receives only `approval-required`.
The reusable approval token, exact nested operands, and review identity stay in
trusted Studio. A separate human Approve/Deny control is bound to the child
request, package, and current revisions; the child cannot echo a token to
approve itself. Editing/disabling the package, changing instance access, or
closing/reloading the view aborts work, invalidates pending approval, and makes
the handle stale. An effect that already started retains its truthful ordinary
result or partial receipt; it is not rewritten as success or rerun.

Interactive panels are one browser-UI extension boundary. They are separate
from the feature runtime below and from both executable training-parser
sessions; none grants direct main-process or native-code access.

## Feature-runtime modules

An optional `runtimeModules` entry runs an immutable JavaScript body in its own
hidden sandboxed Electron renderer. It can perform local computation, register
typed tools, receive selected host events, keep application state and publish
reviewed UI contributions. It does not run inside the brain, replace its
learned model, or receive Node, a shell, direct keys, filesystem handles or the
Studio DOM. Network, navigation, new windows, downloads and device permissions
are denied. Host effects are available only through declared, currently
enabled tools.

Importing or globally enabling a package does not execute a module. In the
Plugins manager, review the package and instance access, then choose
**Feature runtime → Activate** for that exact module. Activation binds the
immutable content hash, current catalog/instance revisions and current
resource admission. Changing package/access or choosing **Stop** retires its
tools and contributions, cancels pending approvals and closes the owned
renderer. Brain Unload/recovery also stops live modules. A later activation
starts a new JavaScript session; it does not resume a captured JS stack or
replay unfinished host effects.

The module declaration is:

```ts
runtimeModules?: Array<{
  id: string;
  title: string;
  description: string;
  script: string;
  events: string[];
  ui: { kinds: Array<"style" | "menu" | "button" | "panel">; slots: string[] };
  dynamicTools: boolean;
}>;
```

For example, this entry adds an inert panel after explicit activation:

```json
{
  "id": "text-helper",
  "title": "Text helper",
  "description": "A local text utility.",
  "script": "globalThis.omniRuntime.onActivate(() => globalThis.omniRuntime.ui.put({id:'status',kind:'panel',slot:'chat.after',title:'Text helper',html:'<p>Text helper is active.</p>'}));",
  "events": ["brain.lifecycle"],
  "ui": { "kinds": ["panel"], "slots": ["chat.after"] },
  "dynamicTools": false
}
```

The frozen `globalThis.omniRuntime` API is the module's only host-facing SDK:

| API | Purpose |
| --- | --- |
| `context` | Read-only brain/package/content-hash/module identity; no credential or approval token |
| `onActivate(fn)` / `onDeactivate(fn)` | Register lifecycle callbacks; return an unsubscribe function |
| `on(topic, fn)` | Subscribe to an explicitly declared host topic or this module's `ui:<event>` callback |
| `call(toolId, action, args, requestId?)` | Request a declared current host capability through ordinary ToolExecutor |
| `registerTool(definition, handlers)` | Publish typed local handlers only when `dynamicTools: true` |
| `ui.put(contribution)` / `ui.remove(id)` | Register/remove declared styles, menus, buttons or inert panels in allowed slots |
| `state.get(key)` / `state.read(key, offset, length)` | Read current application state or bounded UTF-8 pages |
| `state.put(key, value, expectedRevision)` | Save structured JSON with compare-and-set revision checking |
| `receipts.read(id, offset, length)` | Read a saved response in bounded pages when it exceeds one transfer frame |

The generated bootstrap reports ready only after top-level declarations.
Activation waits for that acknowledgement; a top-level exception or missing
ready fails the bounded startup rather than marking a broken module active.
Modules do not send their own ready message through the public SDK.

Register callbacks at top level and put effects in `onActivate`; startup host
calls are rejected. Deactivation is best-effort cleanup, not a final durable
transaction—save needed state while active. Current host topics are
`brain.lifecycle` and `chat.action`; their data is observed lifecycle/action
metadata, not conversation contents or an invented view of private thoughts.
A UI contribution's `event` is delivered as `ui:<event>` to its own module.
Current mounted slots are `chat.toolbar`, `chat.before`, `chat.after` and `tools`; declaring
an unknown slot does not create a new placement. Panel HTML remains inert and
style contributions cannot load resources or execute code.

### Dynamic typed tools and nested permission

Set `dynamicTools: true` before using a registration such as this module body:

```js
const runtime = globalThis.omniRuntime;
runtime.onActivate(async () => {
  await runtime.registerTool({
    id: "text-stats",
    title: "Text statistics",
    description: "Count the code points in provided text.",
    actions: [{
      id: "measure",
      description: "Measure the supplied text without host access.",
      inputSchema: {
        type: "object",
        properties: { text: { type: "string" } },
        required: ["text"],
        additionalProperties: false
      }
    }]
  }, {
    measure: ({ text }) => ({ characters: Array.from(text).length })
  });
});
```

The host publishes a source-bound `extension.<sha256>` identity rather than
letting a module impersonate a built-in tool. New missing policy begins at
Ask; an existing Off is never raised by registration. Current argument schemas
apply before the module handler runs. Use self-contained local schemas; remote
schema references are not fetched. A handler's `runtime.call(...)` needs the
package's enabled nested capability and the nested tool's current policy too.
Nested Off blocks; nested Ask and risky Auto still require the exact host
review. Outer Full is not a bypass. The module never receives an approval
token and cannot resolve its own Ask. Changing operands or relevant revisions
invalidates old approval.

Runtime `call` does not recursively invoke other `plugin.*` or `extension.*`
tools. Declare the required built-in or MCP capability directly instead.

Reusing an exact request identity preserves its actual receipt; changed
operands are rejected. An uncertain dispatched effect is not automatically
repeated. A delivered event is not proof that an async handler finished, so
use revisioned state and inspect real receipts rather than treating a callback
as an exactly-once transaction.

Large completed outputs return `{receipt, state}` instead of a truncated
value. Use `receipts.read(receipt.id, offset, length)` to traverse the saved
response. Schemas use supported local MCP-style JSON Schema dialects without
remote references, async validation, automatic defaults or type coercion;
invalid input rejects before the local handler is dispatched.

### Resources, state, history and portable copies

The renderer and existing app family participate in resource admission. Frame
sizes, state/history reads, persistent writes and disk reserve are bounded;
large saved data uses pages or receipt descriptors instead of truncation.
Resource pressure can refuse activation or stop a module. Safe forceful stop
requires exclusive current renderer PID/creation-time ownership; inability to
verify release is reported, not hidden by killing another process.

`window.omni.pluginRuntime` is the trusted Studio API for Activate/Stop, paged
sessions/history, exact history-event bytes, inherited-state sources/conflicts,
state inspection/selection, contributions and host-owned approval review. It
is not exposed as a general preload to the module. Runtime history and state
are complete unsanitized application records and may contain private task
data; they are not a replacement for native neural memory or a transcript of
unobservable thought.

Copy/export/import/recovery/full merge retain exact supported runtime state,
history, source hashes and completed receipts, but never a live renderer,
nonce, approval, process handle or automatic host-effect replay. Sessions start
dormant, including those from a trusted local copy. Distinct merged source
states remain attributed separately. If sources disagree on a key, Stop the
module, choose the inherited source, **Inspect exact source**, then **Use
reviewed source**. Human selection binds that exact source and current key
revision; the module cannot select a parent state or combine grants for itself.
Review package/access again and explicitly Activate when ready.

## Studio themes

An enabled immutable package's **Studio themes** section provides **Apply**
and **Clear plugin theme**. Apply stores the exact package hash and theme ID
for this device, verifies the currently enabled version, and applies validated
CSS tokens. Clear removes that selection and restores the owned appearance
values without overwriting newer changes by another style owner. A changed or
disabled package does not silently redirect the selection to a different hash.
Theme application runs no package script and grants no instance/tool access.

## Brain portability and lineage

Current, private-archive, and referenced `.omni` exports carry a declared
plugin section when the brain has plugin selections. That section contains the
exact immutable packages, public documentation and declarations, reviewed
selection records, semantic-learning receipt state, install provenance, and
content hashes. It never contains credential-store entries, bearer/OAuth
material, one-use approval tokens, live sessions, catalog database paths, or
executable setup code. The outer `.omni` checksum table and the inner canonical
package/content hashes are both verified before an imported identity is
published. Undeclared files, changed packages, malformed selections, and
credential-like immutable delegate constants are rejected.

Portability intentionally separates history from authority:

- **Trusted local Duplicate/Fork** preserves the exact active selection and
  tool levels in the new identity, with independent host-policy revisions. A
  completed ordinary-learning receipt is copied before access is enabled so
  the already-cloned neural state is not taught the same declaration again.
- **External `.omni` import** installs missing packages as inactive catalog
  dependencies and retains every selection as a dormant inspection record.
  It creates no per-instance access and transfers no credentials or approval
  authority. If the exact package was already enabled for some other local
  brain, that unrelated installation-wide state is not rewritten; the newly
  imported brain still cannot call it until separately reviewed.
- **Full merge** keeps each parent's package/selection/semantic provenance as
  a distinct dormant record. It does not combine grants or make a parent
  plugin callable in the merged identity.
- **Origin export** omits later mutable plugin selections. It remains the
  immutable origin rather than silently absorbing current integrations.

Removing a brain removes its instance and dormant-selection rows, not shared
immutable packages still used by other identities. A failed external import
may leave a verified newly installed package version in the installation
catalog, but it remains inactive and confers no brain/tool authority.

## Typed preload API

`window.omni.plugins` exposes:

| Operation | Result |
| --- | --- |
| `list()` | Versioned installation catalog summaries |
| `package(pluginId, hash?)` | Verified active or selected immutable package |
| `installFile()` / `installPackage(value, sourceLabel?)` | Reviewed file import or plugin-maker install |
| `edit(pluginId, baseHash, value)` | New version linked to an installed parent hash |
| `activate(pluginId, hash)` | Select version and disable it pending review |
| `setEnabled(pluginId, enabled, expectedHash, expectedRevision)` | Compare-and-set installation-wide master toggle for the reviewed active version |
| `exportFile(pluginId, hash?)` | Reviewed canonical package export |
| `instance(brainId, pluginId)` | Exact per-instance selection plus pending/running/complete ordinary-learning receipt, or `null` |
| `setInstanceAccess(request)` | Explicit content hash, capability choices, and per-tool levels |
| `portable(brainId)` | Read-only active and dormant portability/lineage records; no credentials or grant token |
| `openPanel(brainId, pluginId, hash, panelId)` | Fresh brain/hash/revision-bound scripted-panel launch; no portable authority |
| `panelRequest(request)` | Main-validated child request routed through a declared plugin tool and ordinary ToolExecutor |
| `reviewPanelRequest(request)` | Trusted Studio Approve/Deny for one exact pending child request; approval tokens never enter the child |
| `closePanel(brainId, viewId, nonce)` | Revoke the view, cancel in-flight requests, and invalidate retained approvals |
| `onEvent(listener)` | Catalog/instance/semantic status only, with exact revisions and package hash, delivered only to the trusted Studio contents; no source text, credentials, paths, or authority keys |

The renderer never receives the catalog database, blob paths, ToolExecutor,
filesystem handles, or credentials. Interactive calls accept only main-issued
per-open handles; IPC again checks that the caller is the trusted main Studio
frame. The untrusted child has no preload method at all.

`PluginManager` is mounted in Tools & permissions. `PluginApprovalReviewView`
is mounted before the chat approval button, and the manual console retains the
real `ToolExecutionResult` for `PluginToolApprovalReviewView`. Studio CSP admits
only the dedicated plugin-panel scheme—never generic `about:` or `blob:`
frames—and main blocks plugin descendant navigation/network. The Ask parser
additionally requires an actual approval-required `plugin.*`
execution whose event/execution/review identities and package hash match, so
untrusted MCP/shell output cannot fabricate the review. Exact nested arguments
render in a bounded scrollable `pre`.

`PluginInteractivePanel` is mounted for declared scripted panels. It uses a
second dedicated scheme and an opaque script-only iframe. The host-owned Ask
card remains outside the child and shows the resolved nested operation.

## Limits

The examples in this SDK are declarations, not recorded successful tool effects
or neural training. A schema/document learning receipt does not prove that a
brain understands or will naturally select the plugin. Browser isolation and
host validation reduce authority; they do not certify arbitrary hostile
third-party packages or every packaged OS. Review package provenance, declared
capabilities, nested permissions, stored unsanitized data, and exact version
hash before enabling executable surfaces.

Saved inline output is now a separate, narrower boundary: an actual completed
`studio.artifact.create` action can retain a local HTML/SVG/canvas presentation
in chat. Its raw content stays main-owned, its optional script runs under a
fresh opaque hash-CSP frame, and every declared host call re-enters the normal
permission executor. New-owner copies/imports/merges keep presentation bytes
but start dormant. This does not make package panels themselves durable chat
artifacts; see [Saved HTML artifacts](html-artifacts.md).

Still open:

- complete real-account/package Electron interaction and
  visual/accessibility review of the manager, all themes, and both Ask-review
  placements;
- real Electron/packaged, accessibility, resize, and hostile-package acceptance
  for the scripted-panel host;
- arbitrary Node/native modules, unreviewed custom decoder hooks, unsupported
  binary/media codecs, and unrestricted direct application/decoder internals;
- real dedicated-renderer startup/isolation/termination and hostile-package
  qualification on every packaged platform; injected host fixtures are not
  that acceptance;
- learned natural selection of newly installed tools and full corpus quality;
- packaged cross-platform acceptance for the wider plugin manager/tool flow.
