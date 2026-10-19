# Integrations and application interfaces

Omni AGI Studio keeps the renderer isolated from filesystem, process,
credential, and neural-worker authority. The desktop main process exposes a
typed `window.omni` preload bridge and validates every call again at the host
boundary.

This document describes interfaces present in current source. Source and
focused tests are not universal service interoperability, trained competence,
all-platform packaging, or release evidence.

The current primary routes are **Conversation**, **Training & Evolution**,
**Brain map**, **Tools & permissions** and **Settings**. Home owns brain
selection/import/export/duplicate/merge management. Clock, schedules, operational
history, fact checking and imagination keep their existing chat/tool/API paths;
they are not separate primary pages. A UI destination is not an execution grant.

## Desktop bridge

The authoritative public types are in `src/shared/types.ts` and the focused
shared modules it references. IPC channel names are in `src/shared/ipc.ts`; the
only renderer exposure is assembled in `src/preload/index.ts`.

| Group | Current responsibility |
| --- | --- |
| `window` | Native window controls, external navigation, data-folder reveal and appearance |
| `brain` | Build/library state, explicit lifecycle, duplicate/fork, full merge, storage operations, snapshots, export/import, inspection and Fresh attention |
| `portable` | Selected-instance Windows folder creation, observable saved-file preparation and exact destination resource review |
| `regionalExecution` | Loaded-owner manual decoder-route review, explicit activation/deactivation and operational status |
| `nativeOperationAuto` | Loaded-owner review/Arm/Stop and paged qualified-operation metadata; no implicit model operation |
| `chat` | Send/cancel, action approval/cancellation, delivery receipts, current-task attachments, paged conversation, feedback and stream events |
| `learning` | Specific Learn and selected-claim Fact Check lifecycle |
| `questions` | Durable structured questions, answers, cancellation and timeout preferences |
| `data` | Host-owned source selection, preview, manifests, ingestion, crawl, coverage and source ledger |
| `teacher` | Explicit hosted/local teacher configuration, bounded model discovery and durable attributable training jobs |
| `localTeacher` | Explicit runtime/GGUF selection, resource planning, attributable lessons and durable local-teacher jobs |
| `mcp` | Registry/access, authentication, capability events, review interactions, connected Apps and local durable-task custody/Stop waiting |
| `maps` | Private search connections and optional explicitly configured retained-raster profiles; no automatic public source |
| `tool` | Tool policies, exact execution, cancellation and approval preferences |
| `automation` | Inert action-file authoring, immutable revisions, explicit activation/run/review and durable run state |
| `train` | Runtime training jobs and events |
| `modality` | Native generation, artifacts, speech, capture and observation controls |
| `artifacts` | Saved chat HTML artifact open, explicit dormant activation, declared tool relay/review and close |
| `plugins` | Immutable package/catalog/access, scripted panels and non-authorizing dependency lineage |
| `pluginRuntime` | Explicit module activation/Stop, runtime contributions, host review and paged state/history/conflict inspection |
| `substrate` | Current-owner sparse job/plan pages, exact-job Stop and revision events; apply/rollback remain permissioned tool actions |
| `voice` | Instance-owned input/output routing, live/buffered delivery and pace; revision-checked settings with committed native parameter-count evidence |
| `agent` | Isolated forks and reviewed overlay merges |
| `evolution` | Candidate, evaluation, promotion and rollback lifecycle |
| `trace` | Operational trace queries |
| `mobile` | Same-desktop-brain pairing, status and device revocation |

Long-running calls return a task, job, operation, question, interaction, or view
ID and publish typed events. An ID is an ownership/correlation value, not a
display label or permission token.

## Initial documentation corpus

`documentationCurriculumPack.ts` imports the exact bundled
[`USER_GUIDE.md`](USER_GUIDE.md), this file, [`plugins.md`](plugins.md),
[`html-artifacts.md`](html-artifacts.md),
[`executable-training-parsers.md`](executable-training-parsers.md), and
[`managed-environments.md`](managed-environments.md),
[`clipboard.md`](clipboard.md), and
[`api-teacher-model-discovery.md`](api-teacher-model-discovery.md) as raw build
inputs. Pack version 24
declares a separate ordinary-document corpus
containing record names, paths, bytes, content hashes, provenance URLs,
license, and a canonical pack hash.

`BuildInitializationCoordinator` invokes
`BrainService.teachInitialDocumentation()` only after all user-selected initial
sources have completed and before action-head teaching and readiness. Each
document traverses the ordinary native text-ingestion path and produces a real
`TrainingSource` receipt. After all records settle, one durable journal receipt
binds the pack ID, version, and canonical pack hash. Content-hash deduplication
and marker idempotence make retry after a lost acknowledgement safe.

The documentation pack does not edit the frozen origin-v3 curriculum, define an
action target, claim a tool executed successfully, create a persona/policy
prompt, or receive preference/reward labels. Its text is learned state, not a
runtime hidden prompt. Changing any bundled manual changes its record/pack hash and
requires an explicit pack-version review.

## Neural terminology boundary

Native learned state uses packed ternary signs, discrete integer
strength/resistance, explicit repeated contacts, recurrent LIF-style activity,
timing eligibility, working/liquid state and typed modality routes. These are
software contracts. They do not serialize biological cells, ion channels,
neurotransmitters, metabolism, anatomy or subjective experience. Exact stored
contact and activity inspection is not a decoder for hidden thoughts or a
guarantee that a learned representation means what a person would mean.

The two-bit sign plane is only one component of storage. Connection-state
bytes, repeated-contact endpoints/pages, timing controls, indexes, caches,
transient activity and host storage remain additional. No API reports a
human-brain-equivalent bit count, consciousness score, perfect-memory guarantee,
or biological-equivalence verdict. The user-facing comparison and primary
research anchors are in
[Brain-inspired software and biological boundaries](brain-versus-biology.md).

## Portable Windows workspace API

`window.omni.portable` exposes the typed `PortableWorkspaceApi` from
`src/shared/portableWorkspace.ts`:

This retained API/packaging contract is not a separate current navigation panel.
Home's brain-export menu creates `.omni` archives/local references, not a Windows
application folder. Portable Windows delivery remains a distinct requirement
whose current packaged behavior must be verified separately.

```ts
create({ brainId, requestId, presentation })
prepare({ requestId })
status()
reviewDestination()
approveDestination({ reviewId, reviewSha256 })
```

`create` requires an **Unloaded**, ready selected native instance. The renderer
supplies typed ordinary presentation preferences, not paths, credentials or
browser-storage dumps. Main-process native dialogs select a clean compatible
Windows application directory and a new output directory. Creation verifies
the runtime's portable-protocol receipt, actual packaged ASAR hash and PE
architecture, and copies the runtime plus a full **unsanitized private archive**.
It never overwrites an existing output folder or silently loads a model.

The selected archive carries saved learned state/history, working continuity,
sources/artifacts, supported pending progress and instance configuration.
Separate ordinary app settings are appearance, workspace profiles/arrangement,
notifications, the selected theme dependency and tool approval timeout. They are
validated typed data, not a raw LocalStorage/keychain transfer. A theme package
does not grant its executable modules activation or tool access.

`create` and `prepare` use the existing `brain` storage-operation progress and
Pause/Resume/Stop controls, correlated by the exact request ID. Status exposes
`preparation-required`, `preparing`, `review-required`, `ready` or `blocked`.
On the destination, **Prepare saved files** adopts the archive before normal
saved-data access or neural admission. Only then may **Review destination**
produce an exact review ID/hash for **Approve this resource fit**. Approval
rechecks current settings and planner fit; stale reviews require a new review.
Tool access is separate in **Tools & permissions**.

Learned data and ordinary settings remain folder-local. Credential stores,
reviewer/host-access authority and resource ledgers stay device-owned outside
the transferable folder. Copied tool/plugin grants, live MCP/device/process
handles, Auto-load authority and source-drive leases are not revived. Supported
pending work is preserved with copied external authority dormant, not replayed
as newly authorized effects. Moving an already-used folder preserves its newer
saved state and requires current-destination review; it does not reimport stale
initial cargo over that state.

The resource plan establishes a destination-local **logical shared pool** with
normal free-space/write safeguards. No padded 20 GB asset is shipped or
allocated just to represent the reservation. Without the fixed adjacent
portable marker, ordinary installer data locations and startup behavior remain
unchanged. A runtime receipt proves only its declared byte/source contract; it
does not prove a Windows launch, cross-hardware neural compatibility, trained
quality, signing or release publication.

## Automatic operation placement API

`window.omni.nativeOperationAuto` exposes the loaded-only `NativeOperationAutoApi`:

```ts
prepare({ brainId, acknowledgeCudaCpuTileFallback? })
arm({ brainId, reviewId, samplesPerCandidate? })
status(brainId)
plans({ brainId, after?, limit? })
stop({ brainId, operationId? })
```

Internal Review/Status version2 declares exact `stageCoverageByWorkKind` for
thirteen individual kinds: three perception embeddings, ordinary/connected
cached steps, three media producers, one prepared text-learning wave,
ordinary/connected full-prefix sampling, and ordinary/connected uncached
full forwards. Each review declares its actually supported subset.
Preparation/Arm do not Load or run a model. Main binds one-use review metadata
to the exact loaded process, owner, registry, resources and policy. A supported
future real operation qualifies candidates with admitted private state and
restoration, then computes each subsequent input anew.

Perception requires exact embedding equality. Cached steps and full prefixes
require matching logical caches, connected activity, neural output, token,
sampler diagnostics and RNG. Full forwards additionally compare their complete
workspace and action outputs without inventing a sampling stage. Media compares
neural state, artifacts and publication effects; prepared waves compare updated
native state, RNG and effects before committing one genuine update.
Cached `contextRange` metadata derives from actual page/attention-tile
transitions. Reuse is explicitly an inference for unseen inputs/frontiers,
not proof they were previously measured. Plans contain no prior output cache
or source text and are not portable grants. Metadata pages default to32 and
allow1..128; cursors bind the current process and revision, not a lifetime
model-size cap.

The actual Worker request owns the common fixed-memory scheduler, reply
planner and outer-learning orchestrator. Fixed memory/STDP, action/Ponder,
mutable media/replay/evolution and publication phases remain on their real
owners and execute once. An already observed fixed prefix may enter the
complete-operation cost basis; a saved summary cannot recreate the live
request, review, candidate or effect authority.

Six existing text kinds additionally admit `native-mixed-sequential` candidates:
ordinary/connected retained cached steps, complete prefixes and uncached full
forwards. The route binds exact registered decoder blocks, reviewed devices and
actual participants. Real regional block calls and transfers occur inside the
existing complete-operation comparison/equality gate. Perception, media and
training do not advertise this mixed mode. No new work kind, concurrent device
execution or distributed collective is implied. See
[mixed-region complete operations](mixed-region-complete-operations.md).

Stop immediately retires policy authority via flags and serial cleanup; it
does not kill/reload a worker or cancel chat. A private failed comparison
restores original owners before a single ordinary fallback. A genuine unsafe
restore pauses for explicit Unload instead. Prefix, forward and wave decisions
apply to that prepared input only. Actual reply and outer-learning orchestration
now consume the supported operations, but do not privately replay or relocate
an entire mutable conversation/training pipeline. Concurrent/cross-rank
execution, retained device-cache/context migration, unsupported direct kernels
and arbitrary whole-brain placement are not established by this API. Real
end-to-end device speed/pressure and packaged behavior require their own proof.

## Manual regional execution API

`window.omni.regionalExecution` exposes the typed `RegionalExecutionApi` from
`src/shared/regionalExecution.ts`:

```ts
prepare({ brainId, start, count })
status(brainId)
activate({ brainId, reviewId })
deactivate({ brainId, epoch })
```

The optional `start`/`count` page the exact registered decoder blocks. An already
loaded native owner is required; these calls never implicitly Load or select an
inspection worker. Main observes topology and binds the settings, owner,
registry and device identities. The renderer receives inspection data plus an
opaque current review ID, not authority to submit its own topology, paths or
policy hashes. Activation consumes that review; use Status after an uncertain
acknowledgment rather than automatically activating again.

`DeviceResourceSettings.regionalExecution` stores versioned manual preferences,
not a live grant. In the advanced resource controls in **Settings**, **Prepare routes** exposes descriptors
for drafting multiple saved routes. Apply them with **Save memory & storage**,
then obtain a fresh **Prepare routes** review and explicitly **Activate reviewed
routes**. A descriptor page can remain visible after editing while its old
activation review is stale. Selecting a route enables its explicit allocation
without replacing the separately selected global RAM policy/ceiling.

The process-local current-owner lease is never exported, copied, or automatically
armed by saved settings, Load, restart, import or recovery. Current capacity is
checked at dispatch. A stale identity/policy/target refuses execution rather than
silently remapping; Deactivate retires the lease at a safe boundary without
clearing working context. Status is readable during a reply, while Prepare,
Activate and Deactivate wait for the supported reply boundary.

The consumer runs the same existing block and packed owners, transfers caller
activity to the selected region and back, and retains the ordinary cached
attention object. It does not substitute another model or an expanded floating
learned master. Supported CPU/Metal routes are direct native packed execution.
CUDA requires explicit `packed-cpu-tile-fallback` acknowledgment and keeps
`nativeComputeSupported:false`: the selected GPU may hold activity/cache while
admitted CPU packed tiles perform the projection. This is not native CUDA
compute, direct GPU-peer pipelining or a speed claim.

Status exposes lease validity, completed calls, observed transfer bytes/counts,
known link estimates and unknown costs. It explicitly reports no regional
training, automatic default, credited GPU context capacity or measured
whole-path performance. The separate automatic-operation consumers, actual
request/reply/outer-learning scheduler and six-text-kind mixed sequential
routes are described above; manual preferences do not arm them. This manual
lease still does not implement regional training, retained VRAM-backed context,
concurrent/cross-rank execution or unsupported direct XPU/NPU kernels. Broader
all-region placement and real hardware performance qualification remain open.

## Action-file canvas metadata

The Automation create/edit surface can project an `AutomationProgram` as a
grid or freeform canvas. Its optional saved field is:

```ts
canvas?: {
  version: 1;
  mode: "grid" | "freeform";
  positions: Array<{ stepId: string; x: number; y: number }>;
}
```

The main-process action-file validator admits only the exact version/mode/
positions schema. Each position must name a distinct existing step and contain
finite, nonnegative, safe-number coordinates. The field is inert presentation
metadata: the `steps` array remains the only execution order, and a displayed
curve does not express parallelism, data flow, activation, or authority.

Saving stores the complete validated program as its immutable content-addressed
revision; it does not invoke a tool. Existing action-file portability preserves
the exact canvas bytes while keeping copied definitions dormant. Actual Run,
trigger activation, nested permission checks, receipts, and recovery retain the
contracts in [Independent host action files](host-automations.md). The editor,
schema, accessibility controls, and current evidence are detailed in
[Action-file canvas](action-canvas.md).

## Current-task attachment API

`window.omni.chat.attachments` exposes:

| Operation | Result |
| --- | --- |
| `stage({brainId, requestId, selection})` | Native-picker selection copied by streaming hash into the app-owned task store; no ingestion |
| `stageDropped({brainId, requestId}, File[])` | Preload converts real `File` objects with Electron's trusted `getPathForFile`; renderer strings are not accepted as paths |
| `cancelStage(brainId, requestId)` | Cooperative cancellation of that exact staging stream |
| `list(brainId)` | Active renderer-safe staged/bound handles without host paths |
| `forTurn(brainId, turnId)` | Lazy exact handles for one history/current turn, avoiding an all-history materialization |
| `byIds(brainId, attachmentIds)` | Exact staged/bound handles named by a queued/delivery receipt, preserving queue display without binding early |
| `remove(brainId, attachmentId)` | Deletes only an unsent staged handle and unreferenced bytes; bound history is immutable |

Normal `chat.send` keeps its positional API and carries attachments in existing
typed turn metadata: `{kind: "task", source: "human", createdAt,
attachmentIds}`. Main atomically binds the exact set before the neural call.
Authoritative human messages and accepted-input/delivery receipts retain stable
attachment IDs; the renderer resolves current handles only when displaying
that turn. The main/native-only resolved descriptor adds one brain-relative
content path for verified sensory/tool reads, never an arbitrary renderer path.
Queued receipts retain IDs but do not bind them until their awaited pending-send
boundary, so queue removal cannot race a hidden consume/rebind.

`stage` accepts the existing file/media selections and a native folder selection.
A folder is sealed into one immutable main-owned container with its manifest,
members, source hashes and traversal counts. It is one task attachment, not an
automatic dataset or neural-learning operation. The original selected directory
is not reopened by later task reads.

`task.attachments` has `list` and paged
`read({attachmentId, offset?, maxBytes?, entryPath?})`. Without `entryPath`, a
folder read pages manifest entries using ordinal offsets, reported as
`folder.cursorUnit: "entries"`. With an explicit safe relative `entryPath`, it
reads that exact saved member in byte pages and returns `cursorUnit: "bytes"`
with member hash/MIME metadata. `entryPath` is not a host pathname or a new grant;
passing it for a non-folder attachment is refused. The container/member/owner
checks and current-turn or exact reviewed-saved-copy fences still apply.

Its Auto default is limited to
the exact brain/turn binding; Off still wins, and a manual Tools-console call
without the trusted chat-turn context fails. Staging itself never calls the
dataset learner. Ordinary visible tool-result learning remains unchanged.

Paged reads verify and read one exact opened file descriptor. A bounded
process-private proof cache avoids repeated whole-file hashing only for qualified
unchanged local files; current owner/turn/manifest/source/file checks still run
on every page. Windows and unknown filesystems fully hash every read. See
[the integrity, performance and platform boundaries](task-attachment-read-io.md).

Raw audio chat reuses this registry and additionally binds a current microphone
session/utterance/sequence and permission revision. Native code rehashes and
traverses the complete raw audio source without using a platform transcript.
See [Native assembly](native-assembly.md) and
[MCP presets and maps](maps-presets.md) for adjacent tool contracts.

## Permissioned file tool

`system.files` is invoked through the existing `window.omni.tool.execute`
boundary; it does not expose a renderer filesystem API. The historical
`windows.files` name normalizes to the same tool and policy rather than adding a
second grant. Current actions are:

| Action | Result boundary |
| --- | --- |
| `list`, `list-next` | Streamed host-order directory pages; total count is known only at EOF |
| `read`, `read-all`, `read-next` | Bounded sequential pages with exact source/page hashes; only zero-to-EOF completion has `wholeSha256` |
| `read-segment` | One offset/length page with optional exact file-identity check and no retained cursor |
| `search`, `search-next` | Metadata, case-insensitive filename substring, or exact literal UTF-8 byte search with bounded scan/result windows |
| `close` | Retires the exact live cursor and owned handles |
| `materialize-task` | Copies one current-task attachment to a main-owned path for a later, separately permissioned shell/code action |
| `write` | The existing bounded text write with expected-source verification and atomic publication |

`FileToolPage`, `FileToolSearchPage`, and `TaskFileHandoff` are defined in
`src/shared/fileTools.ts`. Reads default to 64 KiB and refuse requests above
1 MiB rather than silently clamping. Every page re-enters ToolExecutor and
rechecks current permission, owner/task, source identity, and resource
admission. Cursors are opaque, HMAC-bound live host state with five-minute idle
expiry; they are not exported or revived after Stop, restart, copy, merge, or
policy change.

Host reads use canonical absolute regular-file paths, refuse symbolic leaves or
ancestors, and bind checked file descriptors to current filesystem identity.
Directory traversal does not follow links and is explicitly not an atomic tree
snapshot. Attachment reads reuse the existing current/continued-task custody,
not a dataset-ingestion shortcut. `materialize-task` accepts no renderer-chosen
destination, performs no execution, and returns historical-not-live custody in
copies. See [Bounded file tools](file-tools.md).

## Clipboard tool adapters

`studio.clipboard` and `system.clipboard` are distinct catalog tools routed
through the normal ToolExecutor policy/revision boundary. The first owns only
the portable `clipboard/workspace.sqlite3` ledger and content-addressed blobs;
the second owns only the currently permitted OS text-clipboard call. Their
required `scope` values cannot be substituted.

The main-process host bridge launches a fixed trusted platform command with no
shell, streams one bounded UTF-8 read page, and monitors the helper PID. Host
mutations use persist-before-dispatch effect rows and terminal or uncertain
receipts, so lost acknowledgements never cause an automatic second write.
Installed Linux Wayland/X11 routes use an exact tracked foreground holder;
writes require bounded count-plus-SHA-256 readback equality before success.
Only Studio-owned children are stopped on replacement, Clear, resource failure,
or shutdown, and live holder authority is never portable.
Complete learning-workspace snapshots carry the unsanitized instance
clipboard and historical effect receipts but never an OS grant or handle. See
[Instance and host clipboards](clipboard.md). Focused source tests do not prove
actual desktop clipboard access or compositor/X-server behavior.

## Managed environment tool adapter

`runtime.sandbox` and `runtime.vm` are normal catalog tools dispatched through
`window.omni.tool.execute`; they do not add renderer filesystem/process access
or a hidden preload authority. Main binds each mutation to the calling brain,
exact action/arguments, request ID, current permission revision, and optional
current chat turn used to resolve a VM image attachment.

`EnvironmentService` owns versioned environment/operation identities and a
FULL-sync SQLite ledger at `environments/workspace.sqlite3`. The external Lima
runtime lives in an installation-private brain/environment directory with a
clean `HOME` and `LIMA_HOME`; it is not part of the portable learned brain.
Complete workspace snapshots stream the ledger and sandbox workspace files.
Rekey/import/full merge clears live adapter identity, credential-store
forwarding, authority and request replay state. It retains dormant provenance,
workspace bytes and exact completed receipts without sanitizing program-created
content; private exports may therefore contain sensitive task data.

The adapter API and platform evidence are documented in
[Managed sandboxes and virtual machines](managed-environments.md). The current
Lima tests prove command/configuration construction against injected bounded
process results, not a real guest boot. Actual macOS sandbox evidence is one
small controlled fixture. Windows Sandbox source wiring uses an exact owned
UUID, hardened no-network/no-device/no-clipboard configuration, refuses an
unrelated current-user session, and marks guest output unavailable because the
Microsoft CLI exposes only an exit code. Seven injected protocol cases do not
constitute an actual Windows guest boot or packaged-platform proof. No
cross-platform or hostile-code certification is
implied.

## Selected web-image tool route

`web.search.search` accepts `query`, optional `limit` and optional `images`.
The image-category option applies to a configured SearxNG-compatible provider;
the public RSS fallback can return only page references. Search and page fetch
results expose observed `imageReferences` with source SHA, URL, originating
brain/turn and `fetched: false`. Literal HTML image metadata is extracted without
running page scripts.

`web.fetch.inspect-image` accepts one current-task `referenceId` and an optional
`maxBytes` stop size. The existing web permission applies; references are not
grants. The host resolves an actually registered reference, checks every HTTPS
redirect and current learning scope, streams bytes under current RAM/disk
admission, verifies MIME/signature/complete source bytes and stages an exact CAS
attachment. A separate action-owned `webimg-…` scope preserves the original chat
turn's immutable attachment set. Another turn or copied identity cannot reuse
that scope as historical file authority.

The structured `WebImageSelection` descriptor carries exact reference, download,
task-manifest and source hashes; pixels/base64 are not a text prompt.
`omni-web-image-perception` is a read-only same-brain pixel/window receipt.
`omni-web-image-learning` is returned only by the ordinary queued native media
ingestion path after its actual transaction commits. It binds the same source
SHA and replay cursor and cannot be satisfied by learning a URL's JSON alone.
Metadata/labels remain untrusted data, and neither receipt claims trained
comprehension. Schema declarations are baked through the separate
`web-image-tools-1` capability pack without executing example host actions.

See [selected web images](web-image-inspection.md) for ownership, dormant
portability, decoder/platform limitations and exact scoped test evidence.

## Learn and Fact Check API

Fact Check remains a native chat/tool capability, not a standalone navigation
page. Saved fact-check tasks and their evidence/review controls remain available
through the unified Training & Evolution task list; pending fact-check questions appear in
Conversation. Removing the page does not remove this API or grant tool access.

`src/shared/learning.ts` defines `LearningApi`:

```ts
interface LearningApi {
  start(request: SpecificLearnRequest): Promise<LearningTask>;
  startFactCheck(request: FactCheckRequest): Promise<FactCheckWorkspace>;
  list(brainId: string): Promise<LearningTask[]>;
  get(brainId: string, taskId: string): Promise<LearningTask>;
  factCheck(brainId: string, taskId: string): Promise<FactCheckWorkspace>;
  pause(brainId: string, taskId: string): Promise<LearningTask>;
  resume(brainId: string, taskId: string): Promise<LearningTask>;
  stop(brainId: string, taskId: string): Promise<LearningTask>;
  steer(request: LearningTaskSteerRequest): Promise<LearningTask>;
  review(request: FactCheckReviewRequest): Promise<FactCheckWorkspace>;
  acceptCorrection(
    brainId: string,
    taskId: string,
    claimId: string,
    findingId: string
  ): Promise<LearningTask>;
  onEvent(listener: (event: LearningTaskEvent) => void): () => void;
}
```

### Input and ownership

`LearningInput` is one of:

- attributed text;
- a host-owned dataset manifest ID created by the file/folder picker;
- an HTTP(S) web source with reviewed crawl controls.

A generated path is never a filesystem grant. Tasks, claims, sources, native
turns, policy revisions, and evidence are bound to one instance. Reusing a
request ID with different content fails rather than overwriting the saved task.

The unified source-and-goal form is `LearningWorkspace` embedded in
`DataWorkspace`; its same instructions draft supplies the adjacent
`EvolutionWorkspace` objective. Real source tasks, runtime job events, dataset
coverage, resume cursors and candidate results remain distinct receipts, not
duplicate input forms. Visible source/resource ledgers and the next-file parser
picker are removed; their storage/accounting/provenance APIs remain.

`SpecificLearnRequest.quarantine?: boolean` is an optional web-source default;
an individual web input may supply its own boolean `quarantine`. Omission keeps
Specific Learn's existing direct-learning default. This is exposed as an
optional boolean on `data.learn.start` under its unchanged Ask permission and
the existing private typed-action schema. Strict host validation rejects a
non-boolean flag or applying the request option to a request without any web input.
The normalized flag is stored in the source specification and request hash, so
changing it is not an idempotent retry. It reaches the existing `crawlWeb` path,
not a second crawler. That direct crawler keeps its own quarantine-default-true
contract. Pause/restart/resume retain the same source/crawl identity, scope and
resource/permission checks. Quarantined traversal records `quarantined: true`
and never serves as a new neural-learning receipt or an unlearning operation.

### Web scope and evidence

Web policy supports allowed/excluded domains, optional subdomains, global date
bounds, and per-domain date overrides. Enforcement occurs before dispatch, on
redirect/browser requests, and at evidence commit. Publisher-declared dates are
recorded as declared or conflicting, never promoted to independent verification.

Fact Check stores supports/contradicts/context relations, evidence hashes, and
model or user findings. A model finding has `verifiedTruth: false`. Supported,
contradicted, and mixed findings must refer to actual captured evidence IDs.
Only `acceptCorrection` turns a reviewed proposal into an ordinary learning
task.

### Durability and lifecycle

Task state, source cursors, policies, evidence, question links, and receipts are
SQLite/file state—not cue-to-answer neural memory. Pause, Stop, Unload, restart,
fork, export/import, recovery, and full merge retain supported boundaries.
Copies are destination-owned and dormant until explicitly resumed. Missing
source bytes or a mismatched neural generation block continuation instead of
silently restarting from the beginning.

The supplementary Learn/Fact Check capability pack contains declared schemas
and examples and uses actual native candidate/rollback receipts. It never labels
placeholder arguments as observed host successes. Completion of the host task
or capability-teaching path does not prove comprehension or natural tool use.

## Native sparse transfer API

`window.omni.substrate` is a renderer-safe view of main-owned sparse-transfer
jobs:

| Operation | Contract |
| --- | --- |
| `jobs(brainId, query?)` | Paged current-owner summaries and cursor; no raw receipt JSON or authorization revision |
| `job(brainId, jobId)` | One exact owner-checked job summary |
| `plan(brainId, planId)` | One exact preview with operation/mode/counts and binding hashes; no native path |
| `stop(brainId, jobId)` | Trusted Studio control that stops this existing job only; it creates no apply/rollback authority |
| `onEvent(brainId, listener)` | Owner-filtered, revision-deduplicated durable job updates |

Subscription installs its renderer listener first. Main then registers the
sender/brain/subscription identity and takes an owner-validated initial job
page. Preload emits that page before the highest buffered revision for each job;
dispose-before-acknowledgement cannot revive the subscription. Older jobs remain
available through normal pagination.

Mutation stays in the `brain.substrate` ToolExecutor boundary: `inspect`,
`export`, `register-bundle`, `read`, `prepare-import`, `prepare-rollback`,
`apply`, `status`, `list`, and permissioned `cancel`. Apply is a detached job
that waits for the same loaded owner and current authority. A genuine receipt
wins over a late Stop; an effect that may have dispatched without a receipt is
uncertain and is never silently retried. Portable copies retain exact evidence
and lineage but no callback, approval, runtime grant, or automatically valid
plan. This API covers declared native sparse assemblies/contacts, not a whole
cortex, identity merge, or arbitrary tensor editor.

## Native improvement candidate tools

Loaded-brain adaptive learning/improvement availability is normalized on; it is
not a manual recursive-score or reassessment switch. This does not run an
Unloaded owner or grant source access. The host still reads the current
`source.self-modify` policy, exact approvals and immutable evaluation,
regression/retention/resource checks before promotion. Its typed action paths
remain available to permitted chat requests; the ordinary UI hides the obsolete
tool/recipe controls rather than removing those boundaries.

`source.self-modify.propose` routes `candidateKind: source` to the existing Git
workflow. Explicit `neural`, `data`, `substrate`, or `architecture` kinds enter
the existing protected native evaluator instead and cannot fall back to source
edits. Kind-specific operand validation occurs before a durable
`omni-native-candidate-tool-job` is queued.

`source.self-modify.native-status({jobId})` reads that exact owner job;
`native-stop({jobId})` stops the existing job without authorizing a new one.
Jobs report queued, load/boundary wait, running, stopping, complete, failed,
cancelled, uncertain, or dormant state and attach evaluator run/candidate IDs
only when those IDs actually exist. Main retains the current authority only in
memory, rechecks it at the writer and loaded supervisor boundaries, and never
starts/replaces a worker as a fallback. Copied/restarted pending records are
historical evidence, not runnable grants. A completed evaluator receipt is not
proof that the candidate improved the brain or that recursive improvement is
generally successful.

## Structured questions API

`src/shared/userQuestions.ts` defines `UserQuestionsApi`. Requests support
single choice, bounded multiple choice, and text. They are durable,
owner-bound, idempotent, and can be cancelled.

Automatic assumptions are user-controlled and off by default. Expiry records
its deadline and `timeout` provenance. Every answer has task-choice-only
authority and `permissionsGranted: false`; it cannot replace ToolExecutor
approval or authorize a destructive effect.

Question files are included and rekeyed in supported copy/import/full-merge
paths. Copied questions are suspended until explicit resume. Durable resolution
delivery is acknowledged only after the corresponding conversation/action
receipt exists.

The existing question panel is mounted in Conversation as a compact section
that opens for pending questions or service errors and is absent when there
are no questions, history or errors. Conversation excludes `learn` and
`fact-check` source records; those stay scoped to their task's question IDs.
Changing the current instance or API owner retires old reads, subscriptions
and pending response UI. Response receipts must name the same question and
instance. This renderer scope does not grant or revoke a host capability.

`studio.question.ask` returns its durable pending request; it does not await a
native decoder continuation. Main supplies its source action/turn identity,
overriding generated `source`, `brainId` and `requestId` operands.
`studio.question.status` reads the current owned record. A later answer,
timeout assumption or cancellation is staged by `UserQuestionDelivery` as
ordinary completed-action evidence before its delivery acknowledgement. The
selected instance's normal learning scheduler may consume that evidence.
There is no automatic hidden-prompt insertion, forced follow-up chat, direct
waiting-decoder wakeup or demonstrated learned use implied by this wire.

## Instance lifecycle API

`InstanceLifecycleApi` exposes:

```ts
lifecycle(brainId)
loadFit(brainId)
load({ brainId, config })
unload({ brainId })
setAutoLoad({ brainId, enabled })
resumePresentation(brainId)
resumeTurn({ brainId, checkpointId })
inheritedResumePresentation(brainId, executionId)
resumeInheritedTurn({ brainId, executionId })
onLifecycle(listener)
```

Auto-load is installation-local and off by default. Load validates current
resource fit and saved occupancy. Unload requests a supported native suspension
frontier; it is not cancellation or input replay. Resume uses the original turn
identity, authenticated prefix, cache/RNG state, action ledger, and effect inbox.
Already emitted effects are deduplicated.

External camera/microphone/screen/browser/MCP/process handles produce explicit
reconnect requirements. Resource subscriptions and uncertain remote effects are
not replayed. Private Ponder/Best Work waits for a supported boundary rather
than claiming arbitrary generator-frame persistence.

The production schema-1 host portability path and newer schema-2 components
have focused evidence. A fresh complete inherited schema-2 Worker path, trained
fluency, all external handles, and all-platform execution remain open. Existing
exact evidence is same-host CPU: the stored backend name is not a full
OS/ISA/kernel/GPU compatibility proof. New executable native frontiers already
require their runtime/kernel/hardware compatibility seal and paused
`ActivityClock` state, checked against the exact learned-state metadata before
cache adoption. Missing old seals remain explicitly unsupported, not fabricated.

The persisted owner activity clock excludes unloaded/suspended time from
transient decay and idle cadence. Cold working pages use that owner's activity
coordinates with a persisted authenticated checkpoint baseline for older rows;
current wall/file time does not create a new age or recency advantage. Borrowed
regional owners hydrate and re-persist their own paused source clocks. Audit
dates, provenance and user time/timer tools remain wall-clock based. Source
and focused evidence are recorded in [instance lifecycle](instance-lifecycle.md#october-11-active-time-runtime-seal-and-recovery-additions)
and its [cold-page correction](instance-lifecycle.md#october-13-cold-working-page-recency-correction).
Full current-format inherited-Worker and post-learning equivalence acceptance
remain open; clock source/component evidence is not cross-hardware or broad
learned-capability proof.

## Optional Ponder self-report stream

The native action head may leave Ponder private or, after at least one actual
recurrent Ponder pass, choose the private structural route
`brain.ponder/self-report`. No behavioral prompt, canned phase text, or second
learned module forces that choice. The host first publishes the matching Ponder
action and then accepts `chat-ponder-self-report` events containing a validated
`PonderSelfReportDelta`.

The full self-report is decoded before publication begins. Delta frames are
bounded UTF-8-safe delivery chunks, not inference-token streaming or access to
an unobserved chain of thought. Their action/report identity, phase, chunk
index, byte offset, and final byte/hash closure are checked in native and main.
A report can be emitted at most once per visible action; private branch actions
cannot publish it.

`PonderSelfReport` is persisted as `complete` only after every frame and final
hash arrive. Stop, failure, or detachment retains only an actually accepted
nonempty prefix as `partial`, without a final full-content hash or fabricated
remainder. Complete reports are projected onto the assistant message and
thought trace; partial custody remains on the matching action event. This is a
model-authored self-report, explicitly not certified truth or hidden
chain-of-thought. The renderer joins the worker's `actionId` to
`ActionEvent.neuralActionId`, not the distinct host event UUID. Action-bound
partial reports and message-bound complete reports reconcile without duplicating
the publication. The default-off **Readable speed** control paces Unicode
display only; terminal/hidden views flush already accepted text. It does not
throttle inference or fabricate a self-report for silent Ponder.

## Full merge API

The `brain` group exposes `startFullMerge`, list/status, pause, resume, cancel,
history, pending-ingestion continuation, and progress subscription.

`FullMergeStartRequest` requires:

- a caller-generated operation ID;
- at least two distinct native parent IDs;
- a destination name and reviewed `BrainConfig`;
- an explicit hardware tier;
- an access policy containing destination tool levels, `credentials: "none"`,
  and `mcp: "disabled"`;
- an optional installation-local spill selection token.

Parents are checkpointed and leased in deterministic order. Complete regular
files and consistent SQLite/WAL state are streamed into content-addressed
staging. Seals are rechecked before connected assembly and fresh verification.
Publication is atomic; cancellation removes only unpublished staging. Resuming
uses the same operation ID, parents, and reviewed settings.

The new instance is `merged-native`. It retains distinct registered regional
owners and adds packed regional exchanges; it does not average parent weights
or rerun the initial curriculum. Pending source cursors and supported suspended
turns remain explicit queues. New effects use destination permissions and are
never authorized by a parent receipt.

Native/host fixtures cover connected computation, nested merge, rollback,
portable state, selected ingestion continuation, and targeted inherited
execution. Useful integrated behavior, large trained parents, cross-device
continuation, and the newest complete production inherited-Worker path are not
yet established.

## Optional map imagery profiles

`maps.basemaps.list/save/remove` manages device-owned typed `MapBasemapProfile`
records with exact revision checks. The renderer receives no secret getter or
network-dispatch API. `MapBasemapSave.credential` is write-only: omission keeps,
null clears, and a new value replaces the private credential. Profiles, key
references and live authority remain outside brain archives.

The ordinary `maps.display` tool has `basemaps` and `basemap` actions in addition
to the offline `show` action. Actual imagery requires an explicitly enabled
exact profile revision, geographic bounds, requested viewport and current
instance permission. Keyless/keyed XYZ and explicitly contract-authorized
Mapbox static imagery are optional; no public service is a default. Requests
are paced, cancellable and admitted under the selected family RAM/disk policy.
No credential, renderer URL or server-supplied arbitrary fetch target is a
tool operand. Provider errors/redirects do not cause automatic retries.

`MapRasterResult` preserves real raster bytes, hashes, times, source revision,
attribution and the user's recorded retention/learning/export authorization.
That declaration is not a legal determination. Retained views verify checksum
and decoded dimensions without network access. XYZ extent is exact per tile;
Mapbox extent is explicitly `provider-fit-to-request`, not a certified image
edge mapping. Ordinary history/copy/export retains output data without granting
the destination any source connection. See [optional basemaps](map-basemaps.md).

## MCP registry and access

MCP server definitions live in one installation registry. Registrations can be
installation-wide or instance-only. Effective access combines:

1. installation master switch;
2. server default;
3. instance master switch;
4. per-server instance override;
5. the ordinary per-tool policy.

A broader Off wins. Access is checked when schemas are registered and again
immediately before transport dispatch. Disabling/removing a server aborts
pending requests, stops local stdio, removes schemas, and invalidates live
session/subscription state.

### Registration

`McpServerRegistrationRequest` supports:

```ts
{
  brainId: string;
  id: string;
  label: string;
  transport: "http" | "stdio";
  scope?: "all" | "instance";
  defaultEnabled?: boolean;
  disabledBrainIds?: string[];
  url?: string;
  command?: string;
  args?: string[];
  bearerToken?: string;
  environment?: Record<string, string>;
}
```

HTTP requires a validated HTTPS endpoint except supported loopback development.
Stdio requires an absolute executable and separate arguments. It never inherits
the parent application's environment wholesale. Secret values are stored in the
separate encrypted/session store; summaries expose environment names only.

Transport/schema guards currently include 16 MiB per MCP RPC message/event
frame, 4 MiB per tool input schema, and 64 MiB for the installation registry.
These protect the integration transport and registry; they are not model,
dataset, or whole-brain size limits.

Supported negotiated protocol revisions are `2025-11-25`, `2025-06-18`,
`2025-03-26`, and `2024-11-05`. Recognition of a revision is not a claim that
every optional feature in it is implemented.

## MCP authentication

HTTP connections support explicit bearer replacement/removal and OAuth/OIDC
authorization where advertised. The OAuth path implements protected-resource
and issuer discovery, exact resource/issuer binding, PKCE S256, state checking,
an exclusive loopback callback, code exchange, refresh, optional static or
dynamic client registration, disconnect, and revocation reporting.

Studio never invents hosted client metadata or a client ID. A 401/scope
challenge does not retry a tool effect or elevate access. Login opens the OS
browser; connected-app URL review does not expose credentials to an iframe.
Provider-specific and nonstandard auth adapters require separate support.

Credentials are encrypted through the OS store when available and otherwise
session-only. They are outside brain roots, prompts, traces, learned capability
documents, and `.omni` exports. Removing/replacing credentials stops the active
transport and requires explicit reconnection. Failed registry publication
restores the previous secret or reports that re-entry is required.

Generic protocol and UI fixtures exist, but no universal real-provider/account
interoperability claim is made.

## MCP capabilities

Remote tools preserve supported input/output schemas, title, description,
annotations, and metadata. Arguments are validated locally before dispatch.
Malformed schemas and unresolved references fail explicitly.

Declared resources, prompts, and completions create reserved client-owned
wrappers for:

- `resources/list`, `resources/templates/list`, `resources/read`;
- `resources/subscribe`, `resources/unsubscribe` when declared;
- `prompts/list`, `prompts/get`;
- `completion/complete`.

These wrappers still use `tool.execute`, calling-instance ownership, normal
permission levels, validation, cancellation, audit, and actual-outcome
learning. Server-supplied fields cannot forge a client operation.

Results are typed untrusted task data with provenance and
`hiddenPromptUsed: false`. Remote user/assistant roles remain data; no system or
persona role is created. Reading a server `file:` resource does not grant local
filesystem access. Pagination cursors are opaque, operation-specific,
connection-local, and not persisted as recovery tokens. Completion is bounded
to 100 suggestions per response.

Resource/list-change notifications and acknowledged subscriptions emit typed
capability events. Unload, credential replacement, disable/removal, or
connection loss invalidates live subscription/session authority. Reconnection
does not replay completed effects or subscriptions.

### Selected roots and operational diagnostics

`mcp.roots.status/select/review/clear` controls disclosure of explicitly selected
folder names and `file:` URIs. Renderer commands contain brain/server/request
identity, not paths; the main process obtains paths from its trusted picker.
`roots/list` returns only the exact currently reviewed selection, otherwise an
empty list. Server requests never open a picker or choose folders. Reviewed
roots are disclosure metadata, not a filesystem grant or an OS sandbox.

Sharing binds the exact current connection, launch/endpoint configuration,
selection and file-policy revision. Account replacement, reconnect, Unload,
selection change or Off retires live sharing. Only an already-ready exact
session receives `notifications/roots/list_changed`; it is not reconnected or
retried automatically. Choices and grants are device-owned, outside exports.

Negotiated `logging/setLevel` is a client-owned tool wrapper with the ordinary
permissions and argument checks. `notifications/message` from servers or exact
ready Apps enters redacted current-process diagnostics, not neural learning.
`mcp.diagnostics.page(brainId, serverId, afterSequence?, limit?)` returns retained
records with rejection, rate-limit and eviction counters. Retention is bounded
to 256 records / 4 MiB globally, with 64 KiB per record; these are operational
log bounds, not brain limits. Restart clears this log. Known secret redaction
does not establish universal PII or credential detection.

## MCP sessions and private reviews

Streamable HTTP supports background GET event streams, GET-only resumption,
explicit DELETE, event-ID deduplication, and legacy same-origin SSE. The original
POST is never resent during resume. If outcome is uncertain, Studio reports it
instead of repeating a possibly completed effect.

Server-initiated requests enter `McpInteractionPanel` as durable private
reviews:

- typed form elicitation;
- full-URL review, including error `-32042`;
- private sampling input review;
- sampling output review before sharing;
- connected-App tool, network, and link permission review.

The 120-second external network-wait deadline pauses while a real host review or
native sample owns the request. Elapsed review time never grants authority.
Pending reviews cancel after restart rather than replaying navigation or
sampling. Submitted form values are retained by hash; credentials belong in the
auth controls, not ordinary forms.

## MCP server-owned tasks

The genuine outbound task adapter is explicitly `mcp-2025-11-25`; the different
2026 Tasks extension draft is not implemented. A server must negotiate task
`tools/call` support, and each original remote tool's `execution.taskSupport`
is respected: `required` cannot use an ordinary synchronous call, `optional`
supports either form, and absent/`forbidden` does not gain a task-start action.
Studio does not advertise a client task sampling/elicitation receiver.

Task actions remain on the original remote tool ID and use normal
`tool.execute` with its current instance policy, exact Ask approvals, reviewer,
owner/account/session and schema checks:

| Action | Typed input | Meaning |
| --- | --- | --- |
| `call-task` | `{ arguments: originalToolArguments, ttl?: milliseconds }` | Request task creation; its ACK is not the original tool's completed result |
| `task-status` | `{ taskRef }` | Read the exact currently owned task's remote state |
| `task-result` | `{ taskRef }` | Receive the original tool result or read its committed source result |
| `task-cancel` | `{ taskRef }` | Separately permissioned cancellation, only when negotiated; no undo claim |

The reserved negotiated `tasks/list` wrapper uses its own ordinary Ask-level
client-operation policy. An account-scoped provider list is data, not adoption
of foreign task IDs or authority to manage them. Only a locally owned task
reference enters management, never an arbitrary remote ID from the renderer.

`mcp.tasks.localPage(brainId, after?, limit?)` reads local source-original
metadata with current-session/dormant labels and `networkRequestMade: false`.
It does not connect, poll, resume, rebind or grant authority.
`mcp.tasks.stopWait(brainId, taskRef)` aborts only that owned local management
wait, without remote cancellation, process restart or brain-wide cancellation.
All actual remote status/result/cancel requests still pass through `tool.execute`.

Creation intent and possible dispatch are committed before sending. Lost ACKs
and interrupted waits retain unknown outcomes; original calls are not resent.
Typed `ToolExecutionResult.mcpTaskReceipt` preserves real committed custody on
failed waits without turning failure into success or parsing error prose.
Known final payloads retain source IDs, hashes and stable
`taskResultReceiptId`; separately normalized `observedResult` supports the
existing host-owned App presentation without rewriting the raw decoded result.
Distinct explicit result reads keep their own action/execution/evidence IDs
and ordinary learning semantics, linked to that stable source receipt. They
are not another remote task execution.

Task-related input is bound to the actual related-task ID and independently
reviewed. History, exact outcome bytes and inert interrupted publications have
checksummed snapshot/copy/import/full-merge custody; no live session, credential,
approval or task-management grant is portable. Unload/recovery/exit retire local
authority and await custody writes. Historical receipts stay dormant unless
current authority is genuinely established; same endpoint alone is not account
verification and no automatic rebind is implemented.

See [the protocol/source/UI contract](mcp-durable-tasks.md) and
[portable task history](mcp-task-portability.md). Focused synthetic transport,
owned local stdio, filesystem and renderer checks are not real-provider/account,
trained-learning, all-platform or packaged acceptance.

## Same-brain MCP sampling

`sampling/createMessage` can use the already loaded OmniCortex after explicit
input review. It does not load another model, call an API teacher, use ordinary
chat, replay old dialogue, or apply the server's `systemPrompt`. Model
preferences remain visible advisory data.

Execution is private and read-only: authoritative packed cortex, weighted VSA,
liquid, LIF, and modality owners are shared while disposable activity/cache is
isolated. No message, episode, replay item, learned byte, counter, source, or
event is committed. Output receives a separate editable review before delivery.

Supported input is reviewed text and canonical inline image/audio blocks. Exact
bytes/order and raw-source hashes are verified across the TypeScript/Python
boundary. Positive safe-integer `maxTokens`, temperature, and stop sequences
remain subject to real resource admission; oversized input is rejected without
truncation. Sampling tools, extra server/all-server context, video input, and
generated media output are unsupported.

Private image/audio decoding uses bounded admitted RAM and authorized spill,
never a server path or URL. The same loaded brain's sensory regions process the
actual frames/windows. This is mechanics evidence, not useful vision, speech,
audio understanding, universal codec provisioning, or production throughput.

## Connected MCP Apps

A completed tool result may contain a main-owned opaque App descriptor in
`output._meta["omni/mcpApp"]`. `McpAppsApi` can open, relay reviewed RPC, and
close a view. A task-result wrapper keeps separately normalized presentation
under `output.observedResult._meta["omni/mcpApp"]`; its source result stays
immutable. The host reads the exact permitted `ui://` HTML resource and
serves it through a sandbox/proxy with:

- no preload/IPC bridge, Studio cookies, credentials, device grants, or direct
  local authority;
- exact HTTPS/WSS or loopback-development CSP origins; wildcards are rejected;
- tool/network/link actions returned to the private review surface;
- exact brain, source, connection epoch, phase, view token, and provenance
  checks;
- inline display and negotiated fullscreen in Studio's existing renderer.

`McpInteractionPanel` presents App permission records with exact arguments,
origins and URLs. The matching completed action card mounts `McpAppView`.
Opening is explicit, and Close/reopen uses a fresh handle without repeating
the original completed tool. Inert content-hashed call/result/HTML artifacts
survive supported copy/export/recovery/merge paths; interactive sessions,
approvals, tokens and network handles do not.

`onPresentation` and `acknowledgePresentation` carry main-owned exact-view
commands and actual renderer outcomes. Only a ready App that declares the
mode can request it. Fullscreen uses the HTML-dialog top layer, retaining the
same iframe, resource URL and App state. The host waits for the renderer's
acknowledgement before returning the displayed mode, then sends the context
notification. Escape/Return goes inline; Close and connection retirement end
the presentation lease. A missing acknowledgement restores the prior mode
after five seconds; a late acknowledgement cannot undo that restoration.

Reviewed `ui/message` and `ui/update-model-context` accept typed untrusted task
data with source/review hashes. They cannot forge a system/persona role,
ownership or approval. Pending data has a durable ordered admission ledger;
an accepted host queue receipt is not a neural-learning ACK. Copies preserve
dormant custody rather than importing authority. The ordinary loaded native
experience path acknowledges actual admission separately.

App logging is notification-only. Negotiated server tool/resource list changes
are relayed to the exact current-ready view; they never refresh schemas or
replay tools automatically. Not advertised: PIP, raw camera/
microphone/location/clipboard grants, view-exposed tools,
sampling tools/context/video/output media, or wildcard CSP domains. Optional
provider features and packaged-platform interoperability need separate support.

## Separate external advice

The additive `advice` preload API exposes settings/configure, exact preview,
assistant-only start, cancel, paged history, credential status/save/remove,
and request events. Reviewer execution and parsed-review retention are trusted
main-only operations, not renderer tool authority.

Channels have separate provider/model, context-selection and output budgets.
Adapters use OpenAI Responses with `store:false`, Anthropic Messages, Gemini
streaming content, OpenRouter chat, or an explicit loopback OpenAI-compatible
endpoint. Remote destinations are fixed; local endpoints reject embedded
credentials and non-loopback hosts. Redirects, provider tools, fallback models
and incomplete terminal protocols are rejected. The killable provider deadline
defaults to 120 seconds; action review has a shorter host deadline. A configured
model ID does not establish account/model compatibility.

`external-advice/workspace.sqlite3` retains complete request/context/reply history,
settings, parsed-review receipts, provenance and hashes. Keys use opaque references
in device-private storage. Grants, controllers and one-use approvals are not
portable. Known credentials are redacted from transmitted/recorded advisory text,
including split streaming echoes; other private history is not sanitized.
Export, copy, recovery and full merge preserve historical advice without replaying
provider calls or inheriting device consent.

Production advice writes use the shared disk-reserve admission before database
initialization and every mutation. Safe output streams to events immediately;
FULL-sync response checkpoints batch at 64KiB or 500ms observed at a delta,
with terminal/Stop flushes. Event persistence counts distinguish received text
from the committed prefix. History's `activeRequestIds` is current-process
authority, not a saved queued-state label. Orphaned requests retire without
replay; storage refusal leaves the last admitted prefix and a truthful warning.

The optional pre-dispatch reviewer binds the original and resolved operation,
arguments, permission revision, reviewer settings and current owner. A strict
observed-evidence verdict can require Ask, never override permissions. Completed-
effect delivery is not re-executed or reviewed as a new action. Consultation does
not automatically train or replace OmniCortex.

## API teachers

The teacher API supports OpenAI, Anthropic, Gemini, OpenRouter, and explicitly
local Ollama with a user-selected model ID and explicit questions. Hosted
providers use a user-supplied key. Ollama uses the fixed loopback teacher route
at `127.0.0.1:11434` without a key; Studio does not start the server, pull a
model, or fall back to cloud execution. Returned credential-redacted text enters
ordinary attributable learning. The provider does not answer as the OmniCortex
identity and is not a runtime fallback. Keys stay in the device secret boundary,
outside brain state and exports.

`teacher.discoverModels({provider, requestId, endpoint?})` uses fixed official
model-list endpoints, bounded responses, provider-correct pagination and exact
request cancellation. OpenRouter reads its account-filtered catalogue. Ollama
accepts only its explicit loopback endpoint, reads installed tags, and returns
known remote/cloud entries as inspectable exclusions rather than local choices.
The call returns metadata only, never credentials or auth headers, and reports
`generationCompatibilityVerified: false`. Credential change/removal retires an
in-flight result before publication. The manual model field remains unchanged. See
[API teacher model discovery](api-teacher-model-discovery.md).

`teacher.start()` assigns one stable submit identity, then persists a
credential-free request hash and one stable delivery ID per question in
`api-teacher/workspace.sqlite3`. Jobs expose `pause`, `resume`, `stop`,
`details`, and `resolveUncertain`. Committed provider responses and neural
learning receipts advance the durable cursor. A dispatching provider request or
unacknowledged neural update becomes `review-required`; neither is replayed
automatically. `resolveUncertain` requires an explicit `retry` or `skip`, and
the user surface warns that retry can repeat billing or learning. Restart
retires active transport, preserves admitted prefixes/receipts, and resumes only
after current device authority. Copy/import/full merge retains complete
credential-redacted question/response history and provenance as dormant state,
never a credential, process, or execution grant.

Focused provider and durable-job evidence uses fake HTTP/vault/native receipts
only. Live paid-provider/account interoperability, Ollama server/model behavior,
generation compatibility, learned quality and assistant/reviewer judgment
remain separate from protocol correctness. The optional advisory service never
becomes a teacher implicitly. Local GGUF teaching has its own bounded contract
and evidence.

The existing teacher model field stays manually editable. Its explicit chooser
searches the complete returned catalogue and windows options for display only;
no model is selected automatically. Provider/credential changes retire the view
owner. Cancellation updates are also request-bound, so an old Stop response
cannot change a newer discovery in the same view.

`localTeacher` is a separate typed bridge for installed GGUF teacher runtimes,
not a native-brain import. Main owns file selection/hash preparation, fit,
process ownership, attributable source custody and cancellation. See
[the local teacher contract](local-gguf-teacher.md) for supported runtimes,
transport framing, permission/resource boundaries and portability. Its Learn
controls preserve complete multiline lesson prompts and distinguish generated
examples from actual committed source-learning receipts.

## Custom extensions

The plugin SDK is immutable and versioned. It provides an
installation-private content-addressed catalog, JSON import/export, immutable
edit lineage, global and per-instance toggles, package documentation, static
HTML/CSS panels, device-local exact-hash theme Apply/Clear, built-in streaming-parser aliases
for custom filename extensions, and custom tools that delegate to one existing
typed host operation. Optional interactive panels add local controls/canvases
and one script inside a separate opaque browser sandbox; each panel declares
the exact package-local tool/actions it may request.

Every callable plugin action enters one ordinary ToolExecutor decision that
binds two scopes: its custom-tool permission and the resolved nested host
permission must both allow the same immutable package/action/argument revision.
Nested Off/Ask and risky Auto checks cannot be overridden by outer Full. The
Ask result includes the exact resolved operation for visible review.
Global Enable itself compare-and-sets the reviewed active content hash and
revision, preventing a stale toggle from authorizing a newly activated version.

The typed renderer bridge is `window.omni.plugins`; it has catalog/maker/access
operations plus per-open panel Open/Request/Review/Close calls, but no
filesystem handle, executor object, or credentials. Panel requests require a
fresh brain/hash/catalog-revision/instance-revision handle, a 256-bit nonce,
exact message source/schema, and a never-rebound request ID. Main resolves
them to a declared `plugin.*` action and routes that invocation through
ordinary ToolExecutor. The untrusted child has no preload and never receives
an Ask token; trusted Studio performs one exact Approve/Deny operation and
presents the resolved nested arguments outside the frame. Concurrent Ask
requests retain separate request-bound cards; stale open/review completions are
closed or ignored after unmount, identity change, or handle replacement.

The manager is mounted in Tools & permissions, while the exact resolved-Ask
view is mounted in chat, the manual console, and the interactive-panel host. A bounded real
Electron smoke rendered one catalog-backed static panel through the canonical
dedicated scheme and actual main-owned security helpers. That is not full
manager interaction, theme/accessibility, package, tool, model, network, or
cross-platform acceptance.

The existing manager's `PluginModuleAuthoring` edits package drafts with
executable Feature & menu, App styling, Custom tool and Training format starters.
JavaScript/CSS, module metadata and advanced declarations round-trip through
complete package JSON. Save reuses `installPackage` or immutable `edit`; it adds
no new execution IPC and performs no code evaluation. Explicit exact-version
access/activation remains required. The draft editor is not an unrestricted
Node/main-process editor or a permission bypass.

Custom-format aliases are bound into the immutable dataset manifest and reuse
the existing exhaustive parser/cursor over the original source bytes. A live
catalog/access revision aborts the cooperative request and leaves the
uncommitted suffix paused. Content deduplication also includes that parser
binding, so remapped bytes cannot borrow an old parser's receipt. Plugin
documentation, capability declarations and
action schemas enter a separate durable ordinary-learning queue only for an
already loaded enabled instance; only an exact returned neural source receipt
marks it complete. The queue row is atomic with executable access/global Enable
and restart reconciliation repairs installations from older builds.

Training & Evolution uses automatic detection without a next-file parser picker.
The underlying typed parser selection can still name exact reviewed
`{pluginId, contentSha256, formatId}` when required. Automatic preserves built-in
detection and unique custom-extension routing. MIME-only declarations and
known built-in overrides need an explicit choice and a matching extension or
observed MIME; an HTTP header is evidence, not permission to execute a parser.
The manifest/receipt bind source bytes, MIME provenance and parser identity;
start/resume rechecks current access and actual source evidence. The file/
folder picker does not itself start learning. Typed web/crawl requests also
carry the explicit selection where requested.

### Feature-runtime SDK and Studio bridge

`runtimeModules` declare immutable JavaScript, host event topics, allowed
UI kinds/slots and whether dynamic typed tools are permitted. A dedicated
hidden sandbox renderer uses a private ephemeral partition, its restricted
runtime preload, exact main-frame/session/nonce transport and hash-CSP document.
It receives neither Studio's preload nor Node, direct keys, filesystem,
network/device grants, raw renderer IPC or the Studio DOM. Importing a package
does not execute its modules; explicit Activate binds the current exact package
hash and catalog/instance revisions under resource admission.

The module receives frozen `globalThis.omniRuntime`: `context`,
`onActivate`/`onDeactivate`, `on`, `call`, `registerTool`, `ui.put`/`ui.remove`,
`state.get`/`state.read`/`state.put` and `receipts.read`. Register lifecycle
callbacks at top level; host effects belong in `onActivate`, not startup code.
Activation waits for the generated bootstrap's bounded ready/error handshake;
top-level errors or a missing acknowledgement never count as active. Large
completed tool outputs return `{receipt, state}` for paged `receipts.read`, not
truncated data. Supported local MCP-style JSON Schema dialects reject invalid
arguments without remote references, async validation, coercion or defaults.
Deactivation is best-effort, not a durable final-write guarantee. Currently
published `brain.lifecycle` and `chat.action` topics contain observed metadata,
not private thought or ambient chat/file access. UI callbacks use `ui:<event>`.
Styles, menus, buttons and inert panels are limited to their reviewed kinds
and current mounted `chat.toolbar`, `chat.before`, `chat.after` or `tools` slots.

Dynamic registrations have per-action JSON schemas and local handlers. The
host gives each one a source-bound `extension.<sha256>` ID; missing policy
starts at Ask and registration never raises an existing Off. Arguments are
validated before handler dispatch. Any nested `call` must additionally match
an enabled declared package capability and the nested tool's current policy.
Outer Full cannot bypass nested Off/Ask or risky Auto review. One-use approval
authority stays in trusted Studio, never in the module. Disabling/changing
access or Stop/Unload/recovery invalidates live tools, contributions and
approvals; dispatched or uncertain effects retain actual receipts, not replay.

The separate trusted `window.omni.pluginRuntime` bridge provides:

| Operation | Contract |
| --- | --- |
| `activate` / `stop` | Explicit exact-hash/revision module lifecycle; no implicit brain Load |
| `list` / `history` / `readHistoryEvent` | Bounded session/history pages and exact UTF-8 event bytes |
| `stateSources` / `stateConflicts` | Attributed inherited sources and unresolved keys, with pagination |
| `readSourceState` / `readSourceStateBytes` | Exact source/key inspection, using receipt descriptors/pages for large values |
| `selectSourceState` | Human-reviewed compare-and-set source selection while the module is stopped; not module RPC |
| `contributions` / `invokeContribution` | Current owned UI projection and callback dispatch, not raw host code |
| `approvals` / `review` / `onEvent` | Exact host-owned Ask review and lifecycle/contribution/tool status |

State writes use revision compare-and-set; transfers/read pages and disk writes
remain resource-admitted. Runtime state/history is complete unsanitized plugin
application data, not neural memory or fabricated internal reasoning.
Supported copy/export/import/recovery/full-merge paths preserve those records
and source hashes but not a live renderer, nonce, process or pending approval.
Copied sessions are dormant. Conflicting merged keys remain separate and need
explicit source inspection/selection; a module cannot choose a parent's state
or inherit its authority itself. See [the public SDK](plugins.md) for examples.

The dedicated Electron factory currently has injected lifecycle/security
checks, not real packaged renderer/OS hostile-code qualification. Resource
signals and bounded startup/Stop are enforced in source; forceful termination
requires exclusive actual renderer PID/creation-time ownership. Unverified
release is reported instead of killing a shared Studio or other process.

Executable custom formats use either the separate import-free WASM ABI or an
exact `runtime-stream` session documented in
[Executable training-format parsers](executable-training-parsers.md). Both bind
package/module/format/source/output hashes and current catalog/instance
revisions. Main alone opens and traverses the selected source with bounded
backpressure.

WASM remains a pure byte-to-JSONL boundary with no host imports. Runtime JS
registers exact `push`/`drain` handlers in a fresh training-parser session. It
may emit JSONL or declared typed image/audio/video asset pages and may call only
currently enabled package capabilities through ordinary nested ToolExecutor
review. It cannot mount UI, publish tools, persist general module state, receive
a raw source path, or approve its own Ask. Prepared raw media enters the normal
native archive record cursor; it is not a caption/text substitute. Initial
preparation and host review do not Load a neural worker, and complete byte
coverage is not a claim that later native learning succeeded.

Supported `.omni` modes now declare and checksum exact plugin dependencies,
documentation, reviewed selections, semantic receipts, and provenance. A
trusted local duplicate preserves its current selection under new independent
host-policy revisions. External imports retain those records as dormant and
install missing dependencies inactive; full merge retains distinct dormant
records for every parent. Neither route imports credentials, sessions,
approval tokens, or callable instance grants from host stores, and origin-only export excludes
later mutable plugin selections. `window.omni.plugins.portable(brainId)` exposes
the non-authorizing active/dormant lineage projection for inspection.

Static panels remain scriptless.
Interactive scripts run only under the dedicated hash-CSP opaque frame: no
same-origin, preload, Node, network, credentials, devices, Studio DOM, or
dynamic evaluation. Packages still cannot execute arbitrary native parser code.
The only executable parser boundaries are the import-free WASM ABI and the
exact `runtime-stream` training session above. Ordinary feature-runtime
activation does not itself authorize dataset parsing, and a training session
does not grant direct decoder internals, Node/native modules, unrestricted VM
hooks, or undeclared capabilities. See [the exact SDK contract](plugins.md).

## Saved chat HTML artifact API

`studio.artifact.create` saves canonical local markup, styles, an optional
isolated script, an exact tool/action allowlist, and a bounded presentation
height. Its action result contains a safe identity only. Main stores the raw
content below the instance's content-addressed `html-artifacts` workspace and
serves it through `omni-html-artifact:`; the Studio renderer never receives
that raw source.

`window.omni.artifacts` provides typed `open`, `activate`, `request`, `review`,
and `close` calls. An open launch is bound to the current brain, saved origin,
artifact identity, content hash, revision, random view handle, and random
nonce. The iframe is opaque and sandboxed, with hash-only script CSP and no
network, navigation, preload, Node, credentials, devices, frames, workers, or
dynamic evaluation. The renderer validates `event.source`, opaque origin,
nonce, schema, and size; main independently reserves request IDs and rechecks
the current saved revision after asynchronous authority reads.

Each requested effect re-enters the ordinary ToolExecutor. Main withholds Ask
tokens from the child, while trusted Studio reviews the exact resolved tool,
action, and arguments. Closing or revising the artifact cancels pending work
and invalidates approvals. Complete results remain truthful receipts.

The artifact registry participates in current workspace snapshots. Same-owner
recovery keeps content without live handles. A new-owner copy/import and full
merge keep canonical bytes and lineage but set interaction dormant; no pending
request or approval authority transfers. The exact format, API, evidence, and
remaining qualification limits are in [Saved HTML artifacts](html-artifacts.md).

## Verification boundary

Focused host, renderer, and small native fixtures cover the interfaces above,
including learning durability, questions, registry/access, OAuth mechanics,
MCP capabilities, review flows, private native sampling, connected graph
assembly, storage accounting, and supported continuation.

Still separate:

- useful learned behavior and natural integration selection;
- real-provider/account MCP and teacher interoperability;
- actual inline App placement/browser acceptance and App portability;
- the newest complete production inherited schema-2 Worker continuation;
- CUDA/NPU/multi-GPU and physical multi-drive performance;
- all-platform packaged execution, CI, signing, and release;
- long-corpus training, broad retention, useful merged capability, or AGI.

See [the feature matrix](FEATURE_MATRIX.md) for the full v2 workstream and the
specialized evidence notes in this directory for component-level details.
