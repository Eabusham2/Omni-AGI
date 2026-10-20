# Independent plan and verbatim completeness audit

Audit date: 2026-10-20. Source baseline:
`262915ff9326f5387932a002ef6770aa23bca16c` in the primary checkout.

Coordinator follow-up, October20: F1 and F2 now have reviewed, focused inert
dispatch/authority/helper checks in PRIMARY, not yet a published commit or live
provider/sensor/model proof. F3 now has reviewed source and scoped handoff,
retirement, unknown-effect, inherited-root and resume checks recorded in
[live-sensory-handoff.md](live-sensory-handoff.md). F4's redundant
preview/header labels have a narrow source-checked fix; new visual qualification
remains open. The historical findings and baseline below are preserved rather
than rewritten as if the omitted paths already existed. Current gate progress
is maintained in [FEATURE_MATRIX.md](FEATURE_MATRIX.md#current-source-and-remaining-gates--october20--40bed3f-plus-pending-integration).

## Result

**The entire plan is not yet proved complete.** The audit found three real
source/connection gaps, rather than treating every historical `open` row as
missing code. Those gaps were reported to the coordinator before this report
was finished and have separate implementation owners. Current all-appearance
acceptance, exact-source CI, packages and release are also unfinished gates.

The native packed brain, continuous memory mechanisms, connected full merge,
format2 custody, most integrations and the newer five-workspace interface are
present in source. That does **not** establish a trained human-equivalent brain,
perfect recall, broad provider/device compatibility, or that every visible
state is bug-free. The user explicitly deferred new-brain/full-folder learned
qualification; this audit did not perform or silently reschedule it.

This is a read-only implementation/evidence audit, apart from this report.
It did not create, load, train, quiz or run a model; open/inspect the user's
private preview; start another UI matrix; contact a provider; rerun a successful
test; cancel existing checks; change production source; commit; or publish.

## Authority and reading boundary

Read completely, in this order, without omitting either raw appendix:

| Document | Complete lines read | SHA-256 at audit start |
| --- | ---: | --- |
| `PLAN.md` | 318/318 | `0ebd1313d4fa7983fb6a0bc6b7066426cf2fa8b0a4bedeadaa3edfc4145152c2` |
| `REQUEST_VERBATIM.md` | 258/258 | `9cd061efaea8650642e3c30020434680af8625bd290ab5d6be93b66a8183495c` |
| `AMENDMENTS.md` | 188/188 | `f9b97da0bbb0cdc50a12c7e22252a1d5664734ae280793fcd0a6edb1fb70746a` |
| `CONTINUATION.md` | 2347/2347 | Historical and latest entries read; not treated as executable proof |

The newer human decisions override conflicting older text. The existing
FEATURE_MATRIX provides original-line IDs (`R...`) for traceability, not proof
that the implementation passes. Current code and the scope of retained results
were checked independently. The scope is the approved v2 plan, both verbatims
and their recorded amendments, not a claim that this audit reread every raw
Codex session ever written.

The coverage tables and exclusions explicitly map all239 distinct original
`R...` entries in the index, with no omitted ID in the read-only mapping check.
This is a completeness-of-mapping check, **not239 passing features**. Repeated
requests and analogy headings share their surviving mechanism/evidence row;
the newer Verbatim2 clauses have their own table below.

At the source read, the primary checkout also contained uncommitted work in
the teacher dispatcher, UI styles and CI/source assertions. Such work is not
in the baseline commit or evidence of GitHub delivery. Findings below distinguish
those in-flight fixes from committed source and from later verification.

Status vocabulary:

- **Source + scoped evidence:** a concrete consumer exists; named retained
  tests establish only their stated boundary, not universal acceptance.
- **Source / qualification open:** code exists, but the requested broader
  UI, runtime, service, device, performance or package result is not proved.
- **Gap / in flight:** a surviving requirement lacks a needed path on the
  baseline; newly assigned edits do not close it before review/evidence.
- **Superseded:** a newer decision deliberately removed or replaced the old
  request. It must not be put back on the not-done list.
- **Conditional research:** investigated/adapted/retained/deferred/rejected as
  the approved plan allows; not a claim that every named paper is implemented.
- **User-deferred:** not passed, but not a current mandate to create/train a
  replacement brain.

## Confirmed gaps and exact requested behavior

### F1 — API-teacher learning is not yet callable by the brain from chat

Authority: PLAN, Learning/teachers and Verbatim2's retained chat capabilities;
original R063/R172/R181/R239 and Verbatim2 “any of the things in tab should also
be able to be done in chat via request.”

Baseline evidence:

- `src/shared/types.ts:3283` exposes the renderer `teacher` API, including
  Start, Pause, Resume, Stop, details and uncertain-outcome resolution.
- `src/main/ipc.ts:1496` routes the renderer Start to the existing API-teacher
  service; `ApiTeacherPanel.tsx` is mounted in Tools.
- `src/main/brainService.ts:249`, `tools/catalog.json`, and
  `src/main/toolExecutor.ts:1359` expose **local GGUF** teaching, not API-teacher
  teaching, to the native structured action system.
- `src/shared/learning.ts` accepts text, dataset and web inputs, so `data.learn`
  is not a hidden API-teacher substitute. The runtime plugin host likewise
  calls declared ToolExecutor capabilities, not arbitrary renderer APIs.

Required completion: a typed, learned-capability path to the **existing**
API-teacher jobs, with configured provider/model selection, secure device-owned
keys, permission/origin revalidation, resource admission, durable job IDs,
Pause/Resume/Stop and honest unknown-ACK handling. No keyword router, foreign
brain replacement, second teacher backend or key supplied by model prose.

Disposition: **Gap / in flight**, owned by `media_teacher_wire_fix`.
The new untracked `apiTeacherTool.ts` and partial dirty dispatcher edits are
implementation in progress, not a finished native route or provider result.
At report finalization, the dirty source additionally contains typed
`data.distill-api` dispatch and `teacherStatsCapabilityPack.ts`. Those concrete
edits address this discovery, but reviewed integration and their affected
verification have not been supplied to this auditor as a completed result.

### F2 — Model-readable hardware/resource/sensor statistics lack a tool route

Authority: PLAN “Expose time/date/timezone, actual inference speed, hardware
capability, resource usage, and available sensors through tools”; original
R052/R092/R165.

Baseline evidence:

- The complete `tools/catalog.json` tool list has no hardware/resource/sensor
  read action.
- `src/main/clockService.ts:170` stores actual output-token speed history;
  its `performance` dispatch at line213 returns those rates, not machine RAM,
  VRAM, disk, temperatures or supported sensor status.
- `src/main/toolExecutor.ts:1545` `studio.settings/inspect-access` returns tool
  policies and approval timeout, not device statistics.
- `src/shared/types.ts:3428` and `src/main/ipc.ts:2328` expose hardware profile
  and topology to the application. Those renderer APIs are not native tools.
- Native chat uses `neuralToolSchemas`; its capability encoding has no separate
  live system-statistics channel. Capture configuration/snapshot tools do not
  provide that missing channel either.

Required completion: a typed read-only callable snapshot of **observed** current
machine/managed-family resources, supported capabilities and available sensors,
with timestamp, scope and explicit unavailable/unknown fields. Do not invent
temperatures or equate an allocation budget with actual usage. Keep inference
speed history as the distinct measured speed feature.

Disposition: **Gap / in flight**, assigned by the coordinator alongside F1.
The dirty source now includes `systemStatsService.ts`, `shared/systemStats.ts`,
a `system.stats/snapshot` dispatch and declared capability examples. Its
observed/cached/unavailable fields are appropriately distinct on inspection;
current integration/affected test evidence and GitHub delivery remain open.

### F3 — Live sensory input waits behind the current generation RPC

Authority: PLAN “Allow sensory input alongside generation”; original R190 and
the same-brain/real-time input requirements. The coordinator confirmed that
this means within-turn sensory conditioning, not merely a running capture UI.

Baseline evidence:

- `src/main/brainService.ts:11673` sends live packets with the normal
  `engine.request("observe_packet", ...)` path.
- `src/main/engineSupervisor.ts:1236` serializes ordinary requests.
- `engine/worker_transport.py:99` uses one normal-dispatch queue; its explicit
  in-flight control list does not include a live-packet admission operation.
- `engine/worker.py:4275` calls `brain.observe_live_packet` only when that normal
  dispatch executes. It therefore cannot condition the still-running chat RPC.
- `engine/omni_core/brain.py:14727` has a genuine within-generation observation
  callback for completed tool/web-image evidence, not a live sensory-packet
  queue consumer.

Required completion: resource-admitted, owner/sequence/permission-bound sensory
handoff at the existing safe neural boundaries, so newly captured input can
condition current output. Preserve backpressure, Stop/Unload/Steer ownership,
the one authoritative brain, and normal temporary activity/settling/learning.
Do not force every frame immediately into permanent parameters or introduce a
second model/concurrent unsafe writer. Native learned comprehension remains
the user-deferred quality gate after the source mechanism is verified.

Disposition: **Gap / design and implementation in flight**, owned by
`action_contract_finish`. Capturing frames while the reply runs is not, by
itself, closure of this gap.

### F4 — Observed appearance defects and current acceptance are not closed

The coordinator's saved current-cohort screenshots found the center background
seam retained by two exact gradients in `styles.css`, and nearly invisible
disabled-primary text. PRIMARY has narrow dirty fixes; they are not current
visual passes. Repeated disconnected-preview notices are also an outstanding
clutter observation to reconcile without concealing the no-engine boundary.
These findings implement original R064/R102/R114/R119/R209/R231 and Verbatim2,
not a request to remake the app. The full old-source cohort is preserved;
genuinely affected current-source coverage and direct screenshot review remain.

## Plan coverage: architecture, memory, research and identity

Each row below states the actual mechanism rather than treating its filename
or an old test count as complete-product proof. `engine/omni_core/` is abbreviated
to `core/` only in these tables.

| ID / original references | Surviving requirement | Current source inspected | Disposition and remaining evidence |
| --- | --- | --- | --- |
| A01 — R045/R058/R060/R142/R168/R246/R247/R250 | Traceable latest-wins checklist; no repeated resolved work; ask actual contradictions | PLAN, both raw appendices, AMENDMENTS, dated continuation, this report | Source/evidence audit done for these documents; not a historical-session reread or full-goal completion. No new fundamental conflict was identified beyond recorded conditional acronyms. |
| A02 — R054/R065/R204 | Authoritative native signs, discrete individual strengths/resistance, repeated contacts | `core/packed_connections.py:1`, `model.py:85`, `model.py:420`, `weighted_vsa_row.py:20`, `sparse_router_state.py:172` | Source + scoped evidence in weighted-substrate/memory/router reports. Signs are ternary; summed strength-weighted contributions need not be ±1. No learned floating master is needed. |
| A03 — R065/R155/R204 | Direct packed computation, not a fully decoded weight model | `model.py:542`, `packed_native_kernel.py:55`, `kernels/weighted_ternary.cpp:13`, `.metal` | CPU/Metal kernels read signs/state within arithmetic; unsupported acceleration uses admitted packed CPU tiles. Tiny CPU/Metal evidence is not full-size speed or native CUDA evidence. |
| A04 — R047/R118/R155 | From-scratch native origin; no Falcon/foreign backend or Q/A recall substitute | `core/brain.py:4916`, `:5072`, `ground_up.py:1`, `native_architecture.py:196`, `brainRepository.ts:1245` | Source rejects foreign/sequence-memory payloads. Runtime answer generation uses the native decoder/vector conditioning. Explicit response-length parsing near brain15207 is a token-budget constraint, not answer retrieval. Trained-quality proof remains deferred. |
| A05 — R054/R130/R204/R205 | All learned owners included in audit, accounting and persistence | `model.py:2915`, `brain.py::packed_runtime_audit`, `ternary_packing.py:407`, `parameter_diagnostics.py` | Audit rejects floating learned Parameters including frozen ones; six-root and recursively connected accounting exists. Actual parameter totals are not cumulative update counts. Full checkpoint/hardware coverage is distinct. |
| A06 — R054/R065/R211 | Honest storage and state cost | `model.py:155`, `packed_connections.py`, `nativeCoreInventory.ts`, `weighted_vsa_row.py` | Physical sign packing is2 bits plus strength/resistance/metadata overhead; this is not1.58 bits for the entire brain or a RAM guarantee. Existing source diagnostics account for that distinction. |
| A07 — R009/R054/R204/R217 | Training, growth, snapshot, import/export, map and merging carry new state | `ternary_packing.py:626`, `weighted_substrate_migration.py`, `persistence.py`, `full_merge.py`, `whole_brain_catalog.py` | Native unit-strength migration, exact current/origin state and repeat identity exist; scoped fixtures recorded. Current-format new-brain resume and semantic merged capability are user-deferred, not passed. |
| A08 — R066–R088/R152/R173/R214/R233 | One interconnected plastic brain; overlapping specialization, sparse routing, common workspace and modality/action regions | `model.py:3487`, `OmniDecoder`, `spiking.py:681`, `liquid.py:12`, `vsa.py:761`, `connected_regions.py:79`, `modalities.py:1159` | Source has shared latent communication and actual learned exchange bridges, not isolated foreign chat calls. Region analogy is not physical/biological equivalence. Actual useful simultaneous specialization/efficiency requires later measurements. |
| A09 — R073/R083/R091/R115/R118/R192/R207 | Ongoing working/afterimage/lasting movement, recurrence/reuse/salience/interference/stability/decay | `memory_lifecycle.py:40`, `brain.py:9098`, `:9241`, `:9392`, `online_replay.py`, `working_activity.py` | Continuous reversible settling is present, not a mandatory rigid three-stage march or keyed fact list. Fast neural admission, transient activity and queued slow replay have different timescales; not every observation becomes equally permanent. Quality/nonforgetting is not proved by a mutation counter. |
| A10 — R091/R115/R192/R211 | Temporary context and useful current activity retained, oversized input not silently truncated | `brain.py:15522`, `:15579`, `recent_token_activity.py`, `working_attention_paging.py`, `portableWorkingActivity.ts` | Current oversized message returns a fit refusal; older activity uses genuine retention/settling and the selected workspace. Persistence/Fresh have separate source contracts; no retrieved raw-source prompt is used as ordinary memory recall. |
| A11 — R022/R112/R176/R198 | Voluntary Ponder, steerable, no fixed thinking deadline or fabricated prose | `brain.py:14879`, `:14920`, `ponder_self_report.py`, `ponder_episode.py`, renderer `chatPonderGroups.ts:22` | Learned continuation/convergence/resource boundaries and optional real self-report exist. Silent computation is not displayed as invented reasoning. Current all-theme chronological composition remains U1. |
| A12 — R016/R098/R099 | Intrinsic improvement available and optional active execution | `brainRepository.ts:2347`, `config.py:547`, `idleCognitionScheduler.ts:270`, `evolutionController.ts`, `evolutionPolicy.ts:1` | Source availability is normalized on; Active mode selects optional idle execution, without forcing every turn or working while Unloaded. Existing tools/Off/Ask/immutable evaluator/rollback still govern effects. Broad successful RSI is unproved, not manufactured. |
| A13 — R122/R128/R138/R141/R145/R212 | Initial whole-data learning then chat, app/tool documentation and current declared curricula, no readiness quiz | `buildInitializationCoordinator.ts:142`, `brainService.ts:4995`, `documentationCurriculumPack.ts`, `supplementary_capabilities.py` | Completion requires every selected source, ordinary docs and capability learning; a failed held-out tool answer does not block chat. Documentation pack24 contains8 public manuals, not private audit/chat text. First natural user conversation is not replaced by a quiz. |
| A14 — R199/R200 | Optional resource-admitted Best Work, ordinary branch, diversity, max proposed temperature0.95, verification/fallback | `best_work.py:24`, `:67`, `:83`, `cooperative_generation.py` | Source + focused fixtures. Cooperative token interleaving, exact-output agreement and syntax/schema validation are not simultaneous GPU branches, calibrated confidence or correctness proof. No reward loop; actual trained benefit remains unqualified. |
| A15 — R012/R036/R038/R041/R044/R057/R061/R096/R185/R188/R207/R215 | Named learning methods evaluated/dispositioned, not blindly enabled | `research-methods.md`, `retention_comparison.py`, recorded24-trial JSON | PNN growth adapted; LwF/SI/EWC/Fisher/MAML/OGD/LoRA literal variants deferred; CLS/attention related existing paths retained; full predictive-coding hierarchy deferred; CLMR and FIM/OGP interpretations unconfirmed. These are allowed research dispositions, not missing mandatory algorithm switches. |
| A16 — R124/R132/R197/R201/R218 | Metal/MLX, CUDA, CPU, communication, distributed, NPU and DFlash2 investigation | `packed_native_kernel.py`, `device_topology.py`, `packed_collective.py`, `distributed_training.py`, `native_draft_evaluation.py` | Direct CPU/Metal and native canonical collectives exist. DFlash2 is evaluated/not enabled (retained report serial ratio0.535); MLX/vLLM/DeepSpeed/custom NPU remain conditional compatibility/benefit candidates. No unsupported native kernel or speedup is claimed. |
| A17 — R009/R048/R217 | Full merge2+parents, new instance, unchanged parents, complete state/history/contradictions/settings | `fullBrainMerge.ts:183`, `full_merge.py:255`, `:846`, `connected_regions.py:79`, `fullMergeHost.ts:89` | Source and tiny historical/native/host evidence exist. Complete state is preserved, shared byte-identical ancestry may deduplicate, distinct experiences remain. New bridges integrate real regions; not whole-model averaging or guaranteed increased intelligence. |
| A18 — R009/R217/R222 | Merge progress, transactional resume/Stop, pending learning and current/origin lineage | `fullBrainMerge.ts:239`, `:253`, `:587`, `fullMergeLearningWorkspace.ts`, `connected_runtime.py`, `connected_execution.py` | Durable coordinator and supported native frontiers exist. Exact missing-custody current-format neural fixture is explicitly deferred. Cancellation deletes only unpublished output; old failed proofs remain failed. |
| A19 — R048/R195 | Distinct duplicate, fork/subagent, reviewed overlay and full merge | `brainRepository.ts`, `ToolExecutor.agent:2325`, `mergeEvidenceStore.ts`, `FullMergePanel.tsx` | COW children use ordinary actual native action runner and resource preflight; workers currently execute serially. Overlay return does not average identities. Learned autonomous selection and better outcomes remain unqualified. |
| A20 — R071/R072/R075–R080/R082/R084/R086/R087/R149/R154/R214 | Honest brain analogy, language tokens vs assemblies, no universal knowledge claims | `USER_GUIDE.md:32`, `brain-versus-biology.md`, `tokenizer.py`, actual native mechanisms above | Documentation explains software arithmetic, byte/token language boundary, distributed assemblies, senses/training/memory/chemistry differences. A token is not literally a feeling or neuron; consciousness, perfect memory and universal correctness are not verified deliverables. |

## Plan coverage: learning, integrations and autonomous tools

| ID / original references | Surviving requirement | Current source inspected | Disposition and remaining evidence |
| --- | --- | --- | --- |
| L01 — R056/R104/R117/R120/R121/R123/R137/R150/R169/R172 | Specific Learn, full inputs/epochs, natural chat route, steering, no old file/chunk caps | `LearningCoordinator:193`, `datasets.py:874`, `:1931`, `distributed_training.py:2681`, `ingestion_schedule_v3.py`, `LearningWorkspace.tsx` | Actual jobs/cursors/traversal accounting and source changes exist; no claim that manyGB implies a genius or requires new parameters per record. Large-record/format source tests do not prove useful learned recall. |
| L02 — R123/R147/R152/R183 | Text/PDF/Office/EPUB/HTML/code/CSV/JSON/Parquet/Arrow/HF/WebDataset/archive/SQLite/media ingestion | `datasets.py:99`, `:114`, `:1320`, `:1470`, `:1703`, `:1931`, `:2234`, `:2356`, `:2497`; existing media ingestion | Declared readers, incremental hashes/provenance, rejection accounting and transactional record cursors are present. Unsupported/corrupt records stay explicit. The full user-folder run is deferred, not passed by supported-extension lists. |
| L03 — R118/R147/R150 | Actual native learning and lower-RAM/drive spill rather than fake fact insertion | `brain.py::learn_experience`, `model.py:420`, `vsa.py:1909`, `native_core_paging.py:225`, `outer_learning_pipeline.py` | Source performs discrete parameter/synapse changes and queued slow updates. Current v3 checkpoint binding at brain7481 uses immutable shards with `sqlite_backup=None`, so the old optional full-SQLite-backup prototype is not evidence of present per-record whole-DB copying. Low-end/full-run pressure and learned quality still need qualification. |
| L04 — R017/R056/R172/R223/R237 | Attach distinct from Learn, supported files/media/folders, output attachments | `TaskAttachmentService:31`, `taskFolderContainer.ts:10`, `UniversalSourcePicker.tsx`, `App.tsx:5878`, `ChatImaginationMedia.tsx` | Source + tiny real-files/SQLite/folder and renderer fixtures. Folder attachment is one immutable task container, not auto training. Explicit saved-copy review with file access Off is scoped to disclosed copies, not original paths or global authority. |
| L05 — R063/R127/R151/R158/R166/R171/R181/R239/R240 | Automated API/log/GGUF teachers, separate assistance/oversight, discovery and BYO keys | `teacherTraining.ts:157`, `ApiTeacherStore:162`, `apiTeacherProviders.ts`, `localTeacherService.ts:45`, `apiTeacherModelDiscovery.ts:211`, external advice services | Services, durability, typed media and local teacher source exist. API-teacher chat parity is F1. Real provider/GGUF inference/quality remains unverified; optional external models do not replace OmniCortex. |
| L06 — R024/R063/R151/R166/R183 | Typed text/image/audio/video teacher lessons and self-supervised unlabeled media | `apiTeacherMedia.ts`, `apiTeacherMediaArchive.ts`, `datasets.py:1233`, `modalities.py`, `speech_training.py` | Typed binary lessons reach real native-media records rather than treating captions as visual/audio learning; provider envelopes/provenance and unknown delivery remain distinct. Reconstruction/denoising objectives exist; no generated-label truth guarantee or new trained-quality claim. |
| L07 — R013/R147/R169/R172/R182/R184/R242 | Continuous crawler ordinary learning with domain/date rules, URLs, frontier/dedup, steer/pause/resume/Stop, scope document | `brainService.ts:9112`, `dataIngestion.ts:1926`, `webLearningPolicy.ts:12`, `LearningCoordinator:568`, `learningTaskStore.ts` | Persistent frontier, configured/unbounded page target and constraints exist. Optional quarantine is not a neural-learning receipt; direct Specific Learn default remains learning. DNS/TLS/authorization failures are not bypassed or silently redirected to a different target. |
| L08 — R013/R035/R166 | Chat fact checking, evidence conflicts/questions, corrections through normal learning, independent work | `LearningCoordinator:216`, `:447`, `:494`, `:533`, `learning.ts`, `ToolExecutor:1374` | Source captures scoped claims/evidence/findings; confirmed corrections create ordinary learning jobs. No complete extraction of “all facts” in weights or universal truth guarantee. Separate Fact Check page is superseded, not missing. |
| T01 — R007/R055/R092/R101/R226 | Local/remote MCP transports, negotiation/auth/resources/prompts/completion/notifications/elicitation/sampling/Apps/tasks | `mcpClient.ts:159`, `:1385`, `:1622`, `:1933`, `:2075`, OAuth/interactions/App/task services | Declared2025-11-25 and compatible older negotiated protocols have consumers and scoped transport/SQLite/inert-App evidence. All-auth/all-future-server universality is not proved; real accounts and package interoperability remain qualification. |
| T02 — R055/R101/R226 | Installation/server/instance master switches, inheritance/presets, structured capability learning | `mcpRegistry.ts:187`, `:206`, `mcpPresets.ts`, `supplementary_capabilities.py`, `BrainService.neuralToolSchemas` | Real precedence and schema paths exist. Presets are inert templates, not completed provider operations; Adobe Express template does not prove broad Adobe app document control. No hidden prose context grants. |
| T03 — R014/R039/R059/R093 | Adobe/mail/coding/browser/maps integrations using permitted APIs/MCP/desktop fallback | `mcpPresets.ts`, `mcpClient.ts`, `mapsService.ts`, `mapBasemapService.ts`, `ToolExecutor:1480`, `deviceInput.ts` | Adapters/templates and private Photon/offline/optionalXYZ/Mapbox exist; configured provider licenses/accounts are real prerequisites. Broader desktop/API compatibility and actual third-party operations are not certified by templates. |
| T04 — R053/R090/R093/R159/R163 | Files, whole/segmented/search reads, task materialization, own/host clipboard, shell/PowerShell | `FileToolService:79`, `ToolExecutor:1415`, `:1610`, `ClipboardService:52`, `fileToolHandoffCustody.ts` | Actual source and scoped file/clipboard custody tests. Current policies, exact copied task grants, cursor/hash fencing and nested effects remain; Linux selection-owner and non-text clipboard compatibility remain unqualified. |
| T05 — R025/R039/R040/R043/R093/R095/R162 | Assembly/OS-permitted process memory, sandbox/VM create/open/run/Stop, elevation boundary | `assemblyRuntime.ts`, `ProcessMemoryService:36`, `EnvironmentService:121`, `ToolExecutor:1408` | Real adapters and truthful unsupported/denied results exist; a harmless Mac assembly/sandbox check is not all-OS DMA/guest proof. macOS child memory access was denied, not a success; Windows guest I/O and some adapters remain unavailable. No protection bypass or unsafe host fallback. |
| T06 — R092/R097/R098/R099/R164/R165/R219 | Time/timezone, stopwatch/laps, speed history, durable timers/calendar/event wakeups/automations | `ClockService:44`, `clockRuntime.ts`, `AutomationService:29`, `automationRuntime.ts`, `clockStore.ts` | Independent receipts, scoped unknown-effect review and no automatic repeat of committed effects exist. Scheduling stays accessible from chat; no separate Schedule tab required. Live hardware statistics remain F2; real native wake/OS stimulus and third-party effect semantics are qualification. |
| T07 — R100/R126/R164 | Custom file/design/graph/PDF/interactive artifact tools and grid/freeform Bézier action editor | `ToolExecutor::files/code`, `studio.artifact`, `ActionCanvasEditor.tsx`, `actionCanvas.ts`, `htmlArtifactService.ts` | File/program tools can create actual formats; HTML/canvas and action-file persistence are real, not canned generated outputs. Source/host and bounded actual Electron evidence is scoped; arbitrary toolchain/PDF correctness or every pointer action is not proved. |
| T08 — R159/R167/R243/R248 | BetterDiscord-like packages, import/edit/export/toggles, JS/CSS/menu/panel/button/tools and training formats | `PluginModuleAuthoring.tsx:15`, `pluginModuleDraft.ts`, `PluginRuntimeService:177`, `:352`, `:385`, runtime parsers/WASM/media spools | Real executable hooks/host calls/dynamic typed tools and four mounted contribution slots exist, not just decorative theme controls. Exact activation, capability/policy checks and versioned docs remain. Pure owned-JS/parser smokes do not certify hostile arbitrary native/main code or all codecs/platforms. |
| T09 — R100/R244/R249 | Seamless custom HTML/interactive apps in chronological chat | `htmlArtifactService.ts`, `HtmlArtifactView.tsx`, `ChatActionCard`, `ToolExecutor:1439` | Actual opaque hash-CSP hosting, exact reviewed calls, custody and dormant-copy behavior exist. Raw markup does not become preload/FS authority. Local Electron evidence does not prove every malicious artifact or packaged platform. |
| T10 — R094/R127/R139/R151/R166/R236 | Optional separate local/API adviser/reviewer, selected context, one-second completed-review gate, intervention | `externalAdviceClient.ts:202`, `externalAdviceService.ts`, `ActionReviewer:137`, `:191`, `ToolExecutor` | Own keys/local selection, consent, streaming, budget preview, allow/Ask binding and decline/custom instruction are source-implemented. It reviews observed actions/self-reports, not guaranteed hidden intent; real judgment/provider outcomes are unqualified. No threat objective is added to the acting brain. |
| T11 — R035/R098/R139/R175 | Ranked single/multiple/free questions, min2 possible, Other/text, optional20min assumption, no permission grants | `userQuestions.ts:300`, `shared/userQuestions.ts:45`, `UserQuestionPanel.tsx:243`, `userQuestionDelivery.ts` | Durable question/answer/timeout scope and actual Conversation panel exist. Timeout is a task assumption only, never an approval/destructive grant. Untrusted page/tool data cannot replace host authority. |
| T12 — R023 | Inspect actual images discovered in web search/page results | `webImageService.ts`, `webImageReferences.ts`, `ToolExecutor:1448`, `core/web_image_observation.py` | Source-bound selected image bytes reach native perception and separately receipted learning, rather than image captions standing in for pixels. Scoped transport/files/constructor-free tests exist; real provider, pixel comprehension and natural learned image inspection are unqualified. |

## Plan coverage: lifecycle, capacity, chat, media and delivery

| ID / original references | Surviving requirement | Current source inspected | Disposition and remaining evidence |
| --- | --- | --- | --- |
| R01 — R011/R033/R211/R220/R221/R222 | Load first; Auto-load off; live fit/edit settings; exact supported Unload/resume | `InstanceLifecycleController:116`, `:153`, `brainService.ts`, `generation_cursor.py`, `execution_checkpoint.py` | Concrete custody/once-only frontiers and explicit external reconnect exist. Stored settings cannot silently shrink unreconciled occupancy. Current-format inherited neural proof is user-deferred; supported OS/runtime boundaries remain explicit. |
| R02 — R174/R211/R217/R218 | RAM/VRAM/multiple-drive topology, budgets, unified grouping, rates/links, spill and optional increases | `deviceResourcePlan.ts:104`, `DeviceResourceService`, `device_resources.py`, `device_topology.py`, `sharedResourceRegistry.ts` | Actual selections/admission and measured topology paths exist; shared pool uses largest declaration, not sum across minds. An allocation is cooperative admission/monitor/reclaim/spill, not impossible universal physical page-pinning. Representative devices/sensors remain qualification/F2. |
| R03 — R117/R150/R174/R211/R217/R221 | Actual retained context capacity and theoretical/manual limit, headroom, training/checkpoint/growth costs | `resourcePlanner.ts:430`, `:664`, `retainedContextCapacity.ts:107`, `working_attention_paging.py:558`, `native_core_paging.py:225` | Enabled distinct GPU page storage is a real consumer, not only diagnostic capacity. Metadata, owner/control minima, primary paused snapshot and extra spill are separate. Unknown hardware measurements are not invented. Manual maximum depends on device/model geometry, not a universal4M slider. |
| R04 — R089/R116/R124/R131/R174/R201/R217 | Useful Auto/speed improvements across actual request/reply/learning path | operation Auto registry, `whole_path_scheduler.py`, `reply_execution_planner.py`, `outer_learning_pipeline.py`, `mixed_region_native.py` | Thirteen homogeneous work kinds and six text mixed sequential consumers exist. Mutable effects remain pinned/once-only. No concurrent/cross-rank Auto/live-cache migration/directCUDA/NPU/MLX/vLLM/DeepSpeed performance is implied. Component timings are not whole-load/TPS evidence. |
| R05 — R015/R021/R032/R042/R104/R105/R106/R108/R109/R120/R130/R144/R157/R160/R161/R189/R205/R230 | Honest phase/progress/elapsed/ETA/total/rate/coverage/logs; actual parameter totals; no changed-param box | `trainingTelemetry.ts:512`, `:577`, `:1055`, BuildProgressStatus, `runtimeJobPresentation`, `App.tsx` | Measured counters, finalization below100 until durable ready, live estimating/countdown and precise hover totals exist. Unknown totals are not fake0/0; exact current multi-file/all-theme outcome remains U1/platform. More decimals cannot make estimates exact. |
| R06 — R017/R018/R019/R020/R027/R049/R129/R134/R135/R136/R174/R175/R177/R186/R187/R194/R202/R203/R208/R213/R229/R238 | One composer action row; Send/Queue/modifierSteer; one editable ordered stack/SendNow; steering preserves task; off-page work and terminal cleanup | `ComposerSubmitControl`, `ChatQueueStack`, `ChatDirectionStrip`, `chatActionController.ts`, `chatGenerationLifecycle.ts`, `App.tsx:5430` | Real handlers/current owner receipts and pre-registration Steer fix exist; no alternate fake router. Generic header Pause/Delete removed; Tools/Settings usable during response with actual safe-boundary exceptions. Full current interaction matrix/engine recovery are not proved by source assertions; the old post-reply remote-method error is not declared universally eliminated without the matching current runtime result. |
| R07 — R022/R028/R030/R111/R112/R133/R176/R198/R206/R223 | Chronological Ponder/action/media, nested actual children, TPS/tokens; Stop preserves committed work | `ponderEpisodeProvenance`, `chatPonderGroups`, `chatReplyProjection`, `ResponseOutputCounter`, `ChatImaginationMedia`, `chatSuspensionPresentation` | Host-bound event ancestry/public-prefix receipts avoid invented chain-of-thought or changing canonical text. Default-off readable buffer affects display only. Stop/cancelled state must reflect actual outcome, not remove learning/reply. Current U1 composition and loaded quality remain distinct. |
| R08 — R029/R050/R090/R110/R170 | Prompt available-token delivery and smoothing/scrolling without model throttling | `chatStreaming.ts:101`, chat viewport/follow-bottom logic, actual shared counters | Burst coalescing has no old96-character/frame ordinary throttle. Reading upward preserves viewport; follow-bottom/jump restores it. Best Work/cold load may genuinely delay first output; no fabricated instant response or guaranteed10TPS. |
| R09 — R024/R028/R088/R152/R173/R190/R223/R233 | Same-brain image/audio/video generation/input, direct assembly conditioning, real previews/finals | `modalities.py:732`, `:852`, `:994`, `:1159`, `brain.py::generate_modality`, media artifact/capture services | Native shared latent/region owners exist. Real binary observations and small weighted media reload evidence are scoped. Within-turn live sensory admission is F3; broad natural imaginative quality/speed is deferred, not proved by inline fixture images. |
| R10 — R026/R052/R093/R227/R228 | Top Call with adjacent chat, live/snapshot input, burst/custom resolution, AutoFPS/quality, speech defaults and manual routes | `CallLayout.tsx`, `LivePerceptionPanel.tsx:125`, `livePerception*`, `voiceRouting.ts:1`, `native_audio_chat.py` | Resource-scaled capture/preview and exact <=200B TTS/>200B native-output default with independent manual input/output are source-connected. Unavailable native route reports unavailable; real microphones/cameras/native speech/physical permissions remain unqualified. |
| R11 — R191/R192 | Requested imagination saved; spontaneous rolling up-to5min, retro Record and always-save off | `ImaginationCaptureService:85`, `:156`, `imaginationCaptureOrigin.ts`, replay/assembly services and existing chat Record drawer | Trusted origin controls retention, not a keyword guess. Retrospective fragments/codec assembly have scoped evidence; gaps/expired content remain honest. Native natural intent/continuous learned media quality and platform performance are not established. |
| R12 — R062/R130/R156/R174/R204/R210 | One complete connected2D/3D map, real parameters/ideas/neurons/contacts/strengths, multiresolution and observed firing | `WholeBrainMap:84`, `BrainMapSubstrateGraph:9`, `whole_brain_inspection.py:215`, catalog/projection/viewport helpers | Real source paging/anchors/LOD and whole graph views exist; Next/Previous map area can traverse beyond one bounded page. No fake token=neuron mapping or firing inferred from weights. Full384 no-API map screens alone do not prove actual populated2D/3D interaction, large scale or live firing. |
| R13 — R034/R046/R064/R102/R103/R107/R113/R114/R119/R125/R178/R193/R196/R209/R216/R224/R225/R231/R232/R235/R241 | Existing Electron UI retained; readable all themes/sizes/states, structural layouts/profiles/CSS, uncluttered notifications/overlays | `appearance.ts:82`, `appearanceWorkspace.ts:29`, App/styles, `ToastNotice`, `toastExpiry`, actual matrix specs | Source and finite saved screenshot evidence exist; current full384 and affected-source review remain open. F4 is concrete remaining appearance work. A cancelled/missing/error state is not to be hidden to claim a perfect UI. |
| R14 — R051/R093/R140/R146/R153/R234 | Apple Glass default, priority/sleep prevention, platform controls and TouchBar mirrors | `appearance.ts:82`, `OsWorkLeaseService:59`, `macTouchBar.ts`, touchBar bridge, mobile desktop definitions | Implemented using supported mechanisms with truthful unavailable paths. No guarantee the OS can never unload/kill the app. Physical TouchBar/OS control/native devices need their own permitted hardware evidence. |
| R15 — R006/R009/R037/R047/R179/R192/R217/R222/R237 | Complete unsanitized current/origin/reference archives, lineage and supported workspaces, import/fork/recovery isolation | `brainRepository.ts`, `wholeBrainCatalogArchive.ts:12`, `connectedBrainArchive.ts`, learning/plugin/teacher/clock/attachment/sparse portability adapters | Format2 complete catalog/current/origin custody and exact host file/SQLite fixtures exist. Keys/live grants stay device-owned; personal-memory sanitizer was cancelled. Actual native current-format inherited-resume qualification is deferred. |
| R16 — R037/R146 | Windows portable EXE/data folder, destination-local reserve and portable relocation | `PortableWorkspaceService`, `portableBootstrap`, `portableWorkspaceAuthority`, receipt scripts, Windows package hooks | Source and tiny PE/ASAR/host tests exist; the extra portable UI was removed, not the shipping/API contract. Real destination preparation/relocation/package launch needs platform evidence. Empty reserve is a destination allocation, not a padded20GB download. |
| R17 — R143/R146/R235/R250 | Windows/macOS/Linux x64/ARM64, companion packages, real worker health and ordinary launch/close | package/build scripts, five platform workflows, release workflow, packaged-health/smoke helpers | Required workflows/target metadata exist; actual current exact-source green and signed/unsigned package receipts remain root-owned unfinished gates. Health/stdlib/no-engine checks do not train or qualify a brain. |
| R18 — R145/R247/R250 | User/API/plugin docs, exact checklist/results, full required CI and released package provenance | public8manual curriculum, PLAN/REQUEST/AMENDMENTS, package2.0.0 metadata, `.github/workflows/release.yml` | Source preparation is not final release. Required tag/commit, checksums/manifest,17packages+8smoke inputs and final published asset verification remain open; sign/notarize only with credentials, otherwise truthful unsigned labels. |
| R19 — R245 | End reminder for next full `Downloads/ai train` task and rentedGPU path if measuredMac estimate >week | PLAN end reminder and explicit deferral | User-deferred training/learned capability; final reminder is still due at actual handoff. No model run is authorized merely to close this report. |

## Verbatim2 and latest amendment coverage

| Exact newer requirement group | Actual current source | Audit status |
| --- | --- | --- |
| Existing app, readable text, no redundant sections/single-choice pickers/random footnotes | Existing App/styles/components retained, advanced settings collapsed | Source present; F4 and full U1 acceptance open. The report does not treat “no errors ever” as an honest software guarantee. |
| Chat-first; remove FactCheck/Schedules/TraceJournal/Imagination pages, retain their work | App route list at417: chat/data/map/tools/settings; actual capability/catalog paths above | Source removal implemented. Standalone pages are **superseded**, not missing. API-teacher parity F1 and sensory timing F3 are genuine surviving paths. |
| Merge Learn+Evolution as Training & Evolution; no whatmaychange/maxcandidate/reassessment controls | DataWorkspace + LearningWorkspace + EvolutionWorkspace | Mounted shared goal/source form and existing jobs/results retained. Internal immutable regression/evaluator checks correctly remain; full UI acceptance open. |
| No visible encoded-source/resource ledgers/nextparser picker | Same mounted source; parser/accounting/provenance backend retained | Presentation cleanup implemented; dead/unmounted legacy component text alone is not a visible regression. |
| Intrinsic recursive improvement, existing permissions and full permitted tools | Repository/native config, native action/controller and normal ToolExecutor | Source implemented as availability, not new FullAuthority/default grants or forced per-turn activity. Broad successful learning/RSI remains deferred/unqualified. |
| Simple connected map2D/3D, no checkbox/router/source dashboards | WholeBrainMap + underlying exact query APIs | Source implemented; populated map and current-theme acceptance remain scoped, not closed by a no-API placeholder. |
| Separate Attach/Learn, unified supported selection, files/media/folders | App5878, UniversalSourcePicker, task-folder/learning services | Implemented; attachment never itself claims learning. Focused source/file tests do not certify useful trained understanding. |
| Import/export/duplicate onHome; Select/all/delete/export/merge; preserveparents/settings/confirmations | LibraryPage and FullMergePanel, librarySelection, storage/archive APIs | Source implemented, with compact distinct export modes and exact destructive targets. Portable UI clutter removed while portable Windows shipping retained. |
| BestWork/Active in chat; shorter settings and no residency/currentfit jargon | App5430–5500, DeviceRuntimeWorkspace advanced details | Source implemented; actual resource/runtime ownership and safety checks remain. |
| All Build modalities enabled, no experience question | SimpleBuildWizard initialization+submit | All vision/image/audio/video flags true; source gates retain hardware scale/permissions. |
| BetterDiscord-like feature authoring, not theme-only | PluginModuleAuthoring+generic runtime/host/parser+Tools mounts | Real author/import/export/version/activation/host calls exist. OS-protected arbitrary access is not silently promised. Actual full current UI/platform qualification remains. |
| Latest: “Senses”; no switches, disabled substitutes, fixed summaries/forced setting displays | Committed262915f Build/runtime/Fresh/map presentation | Source implemented on that commit with scoped tests reported; changed-source visual coverage remains separate from the frozen55b8116 matrix. |
| Keep current user preview only; don'tcancel ongoingchecks; fixes inPRIMARY | Coordinator-owned process/checkout custody | Honored by this auditor: no preview/process/CI interaction, frozen9eaa unchanged. Old private preview does not display new source until the user later requests an update. |

## Deliberately excluded and conditional items

These are not unfinished mandatory features:

- Personal-memory stripping/export sanitizer: **cancelled**; complete
  unsanitized archives remain, with operational keys/grants excluded.
- R031/R148: JEV, RLCD/RCLD reward training, externally rewarded RLVR, preference/reward
  loops: **excluded by later settled decisions**, despite earlier raw suggestions.
- Separate FactCheck, Imagination/Vision-generation, Schedule, Trace/Journal,
  Agents/Evolution/Learn navigation pages; recipes/forcedmodality/recursive
  controls: **replaced by latest chat/five-route UI**. Retained capabilities
  and saved evidence are not cancelled with those pages.
- A fixed4-million context-slider endpoint and universal illustrative32K/128K
  floors: **superseded** by actual model/device capacity and current headroom.
- Blocking first-chat tool quiz: **rejected**; finish initial learning then
  allow chat. Genuine action permissions still apply.
- R180: the older demand to remove all Cancel controls is superseded by the
  explicit Stop contract: Stop preserves committed work; file pickers and
  genuine long operations retain their scoped close/stop/cancel mechanisms.
- Foreign foundation brain/GGUF weight conversion: **not permitted**; GGUF/API
  support is an explicitly separate teacher or adviser.
- “Full tool access” does not override the later explicit **Keep existing
  permissions** answer or OS protections.
- Literal SI/EWC/LwF/PNN/MAML/OGD/FIM/CLMR/LoRA/predictive-coding algorithms and
  MLX/vLLM/DeepSpeed/NPU/DFlash2 switches: governed by the approved research
  disposition/compatibility/comparison conditions, not forcibly added by name.
- New-brain merged-resume proof and full training-folder/learned-capability/
  multimodal-quality qualification: **user-deferred**, not claimed successful.

The FIM/OGP/CLMR expansions remain recorded uncertainties within optional
research. They do not authorize silently adopting a different fundamental
learning design. No additional user policy decision is needed to implement
F1–F3 within the already-approved contracts.

## Completion evidence still required

1. Close F1–F3 at their real consumers and F4's exact presentation defects;
   review the resulting source and new affected non-neural/host contracts.
   Do not substitute a synthetic success, keyword intent parser or merely a
   progress bar for the requested mechanism.
2. Finish/preserve the existing full384 frozen cohort; retain failures.
   Qualify genuinely affected new-source UI coverage and directly inspect saved
   images. Historical screenshots and fixture callbacks have their named scope.
3. Bind current source to required hosted source contracts, worker health,
   ordinary packaged launch/close and required platform artifacts. A green
   old-source job or stale assertion edit is not current package acceptance.
4. Publish/verify the authorized v2 package release only after those gates,
   including exact main/tag, checksums, provenance, portable declaration,
   signing/notarization status and companion artifact labels.
5. Report service/device/hardware limits accurately and give the deferred
   full-folder/rented-GPU reminder. Do not run the deferred model goal here.

No claim of all-plan completion follows from this report. It preserves the
full scope and provides concrete closure work, without re-opening cancelled
features or repeating already sufficient evidence.
