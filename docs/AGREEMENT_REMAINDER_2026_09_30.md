# Current agreement audit and remaining implementation

**Implementation follow-up:** the user requested fixes after this report.
The [implementation closeout](AGREEMENT_IMPLEMENTATION_2026_09_30.md) records
the resulting mechanisms and verification. R1–R12 are checked off there at
source level. The sections below are the historical
findings, not the current unfinished checklist. The later actual RAM answer
selects allocation reservations, monitoring, reclaim/spill and pause. Own
observed speech/Ponder and the originally planned audio-code generator are
implemented under the request to complete the all-experience design.

## Result and scope

The chronological agreement audit has been performed, but **the code does not
fulfil every agreed detail yet**. This document identifies the remaining source
differences at application commit `20e36ee`. It does not count canceled choices,
replaced designs, user-closed engineering tests, or deferred native training as
unfinished work in this audit.

The historical full read covered entries 1–3,661. On this continuation the root
read the complete appended range 3,662–3,695, including the final commit/push
instructions and the current request, without a shortened display. The refreshed
snapshot has 655 user messages, 2,902 assistant replies, 60 question/answer
records, 75 plans and three attachments; extraction reports zero malformed
records. Repeated messages are retained in source order. Its source SHA-256 is
`a48c77705d46a8a2168719911a8263ad3c91fb128e4cfa1024bacfbb9d71af89`.

This is continuation of the recorded full read, not a claim that a new full
reread was performed in this turn or that every raw word survives compaction.
The complete private [ordered transcript](../work/requirements-audit-2026-09-28/FULL_CHAT_ORDERED.md)
and [ordered user messages](../work/requirements-audit-2026-09-28/USER_MESSAGES_ORDERED.md)
remain available locally. Their content is not being published with this report.
Historical screenshot references are not a claim to have their missing pixels.
Tool delivery acknowledgements are not user answers. Assistant proposals become
requirements only when the conversation supports user acceptance.

The root reconciled history and checked source. Three bounded read-only source
reviews covered memory/actions, data/evolution, and UI/media. There was no model
construction, training, quiz, app launch, test execution, CI, packaging, release,
or saved-data deletion in this continuation. The earlier scoped checks remain
recorded in the [agreement recheck](AGREEMENT_RECHECK_2026_09_30.md); final late
geometry/export changes were not rerun after the user explicitly stopped checks.

## Later decisions that override the old plans

| Subject | Agreement in force | Excluded earlier choice |
| --- | --- | --- |
| Origin | Own initialized and trained OmniCortex: Q0004/A00008, U0261–U0267, U0533–U0537. | Falcon/Llama/converted BitNet foundation or answer fallback. |
| Learned weights | Packed authoritative ternary synapses; no full floating learned master: U0419–U0422, U0558–U0569. | Early FP32/BF16 full masters. Temporary arithmetic, activations and eligibility are distinct from a second learned model. |
| Learning | Ordinary experience, overlapping activity, selective retention and actual neural updates: U0111–U0122, U0512, U0522. | Fact-command gating, cue/answer tables, exact answer sequences, hidden source-text answering prompts. |
| Setup | Data before resource sizing; initial training then chat, no quiz: U0170/U0203, actual U0632 answer. | Starter/Blank selection and blocking readiness quiz. |
| Memory controls | One ordinary memory policy; automatic settling; no visible rigid stages: U0106–U0113/U0120–U0122. | Manual Consolidate or ordinary memory-recipe chooser. |
| Resource controls | Manual RAM cap, storage pool and context remain requested: U0130/U0170. | Curiosity/noise/personality controls. Numeric resource sliders are not personality controls. |
| Context | Device-derived theoretical maximum; preserve the entire current message and block one that cannot fit: U0114, U0632, U0648. | Fixed 4M endpoint, unrelated character cap or suffix cropping. |
| RAM/disk | Total-capacity Auto with speed headroom; selected RAM ceiling; compression/spill allowed; shared pool is largest instance declaration: U0170/U0190/U0643. | Physical page pinning or adding a full pool for every identity. The separate repeated 20-GiB disk floor remains. |
| Ponder | Always available and voluntarily selected: U0079/U0094/U0116. | Forced Ponder, optional removal of the pathway or Thinking/Reasoning naming choice. |
| Modalities | Same neural identity and connected idea state; progressive media and separate waveform input/output switches: U0047/U0082/U0096/U0099/U0272. | A separate hidden media or coding assistant. |
| Video runtime | Automatic setup: actual U0632 answer. | Manual FFmpeg setup as the normal product path. |
| Portability | All saved-instance content unsanitized: actual U0626 answer. | Removing chat, pending learning or saved recovery content for a sanitized sharing projection. External OS-vault credentials are separately stored. |
| Platform | Windows/macOS/Linux; Android/iOS companions are accepted: U0042/U0185 and later companion agreement. | Windows-only scope or mandatory independent phone training. |
| Current verification | Code/transcript comparison, no brain testing, no CI/package/release; U0654 stopped additional checks. | Repeating closed feature tests or starting the next long native-training goal. |

## Remaining source differences

The entries below have direct code evidence. They are not inferred from an old
assistant completion claim or the existence of a test file. Related findings
are grouped to avoid duplicate tasks. They remain findings, not fixes made by
this documentation pass.

### R1 — Context cooling is not tied to completed cortical learning

**Request:** U0522: “the context moves through the phases in realtime … at end
they pass into parameters in realttime.” U0111/U0113 also require automatic
movement while running, current/important activity remaining active, and weak
older scratch influence fading. U0648 says older material makes room through
that process, while an oversized new message is blocked.

**Present:** real packed assembly/synapse/STDP updates, automatic overlapping
activity, internal-vector recall and Fresh preservation. The default answering
path does not replay a saved answer or inject retired source sentences.

**Difference:** cancellable production chat previews the human experience,
then durably commits fast learning only after owned reply completion
([brain.py:13965](../engine/omni_core/brain.py#L13965),
[14500](../engine/omni_core/brain.py#L14500)). Cortical updates are queued
separately ([6032](../engine/omni_core/brain.py#L6032)). Raw cooling
([8283](../engine/omni_core/brain.py#L8283),
[recent_token_activity.py:154](../engine/omni_core/recent_token_activity.py#L154))
does not require that queued cortical replay has completed. Continued foreground
work/resource pressure can delay it. Evicting words therefore cannot certify
that those words have already trained the language cortex. The requested
continuous transfer must not be described as fully satisfied by a smaller
context counter or a changed fast-state checksum.

### R2 — Completed tool evidence can miss learning after failure or Stop

**Request:** U0098 permits actions on the go; U0101 requires tools and their
experience to enter the brain. The ordinary-experience requirement applies
without a special “remember” command.

**Difference:** the zero-token Steer/native-stop branch drains completed
external results into learning, but the ordinary failure/cancellation branch
does not ([chatActionController.ts:1191](../src/main/chatActionController.ts#L1191),
[1230](../src/main/chatActionController.ts#L1230)). Completed effects can have
receipts while their results never reach that turn's structured learning drain.
An action already performed must not be executed again to recover its learning.
The prior report's unconditional fallback-learning statement was too broad.

### R3 — Text completion can still wait for a media preview

**Request:** U0094/U0186/U0464–U0466/U0490 require truthful completion and
usable message controls; U0047 requires concurrent creative output.

**Difference:** after `brain.chat()` completes, the worker waits for every
inline Auto/Full imagination job's first preview before returning the chat RPC
([worker.py:2680](../engine/worker.py#L2680)). The loop has no independent deadline
or cancellation check. Consequently the neural/write boundary and ordinary
next message can remain held by codec or first-preview readiness. Media should
keep its own lifecycle after committed text releases that boundary.

### R4 — Ask-permission imagination cannot begin during the reply

**Request:** U0047 asks for visualization to emerge while writing a story;
U0098 says permitted actions must be usable on the go. Ask remains the agreed
default access level.

**Difference:** worker-started mid-generation imagination accepts only Auto/Full
([worker.py:2598](../engine/worker.py#L2598)). The controller starts streamed
`tool` events early, but does not start `imagine` there
([chatActionController.ts:948](../src/main/chatActionController.ts#L948)). Ask media
reaches approval/execution only in the post-reply action loop. This is a missing
permissioned live path, not permission to bypass Ask.

### R5 — New width/head evolution is exposed through API, not natural chat/UI

**Request:** U0038: “Make sure it has recursive self improvement too”; accepted
U0042 includes model/data/source/architecture candidates. U0096/U0098 retain
voluntary improvement and natural permitted actions in the same chat.

**Difference:** the advertised neural schema permits only four `grow-*`
mutations ([brainService.ts:241](../src/main/brainService.ts#L241)). It omits
width/head and holdout fields. The action controller does not forward
`geometryHoldouts` ([chatActionController.ts:661](../src/main/chatActionController.ts#L661)),
although geometry requires them
([evolutionController.ts:1110](../src/main/evolutionController.ts#L1110)).
The normal architecture UI builds only `grow-experts`
([evolutionView.ts:49](../src/renderer/src/evolutionView.ts#L49)). The isolated
migration/evaluation implementation exists, but normal chat/UI cannot request
all those supported operations.

### R6 — Promotion does not require statistically supported improvement

**Request:** accepted U0042's controller promotes statistically supported
improvements and tests whether an improved brain improves its improvement
process. U0038/U0096 retain recursion.

**Difference:** geometry checks permit token/tool loss up to 5% worse and
modality loss up to 10% worse
([evolution.py:209](../engine/omni_core/evolution.py#L209)). The final conjunction
checks mutation, compatibility, resource limits and non-regression tolerances
([1337](../engine/omni_core/evolution.py#L1337)); it does not require a statistical
benefit estimate. A compatible changed candidate is not necessarily an
improvement. Recursive continuation rewrites the objective and omits the prior
geometry mutation/holdouts
([evolutionController.ts:2066](../src/main/evolutionController.ts#L2066)); an
architecture continuation then defaults to expert growth. A new objective
string is not an evaluation of improved improvement ability. Beneficial native
outcomes are separately deferred; the missing decision criteria are a source
gap now.

### R7 — Long one-record training can skip the middle tool curriculum

**Request:** U0282 says tools/actions must be trained at the middle and end,
not merely the start. U0170–U0172 require full training and truthful coverage.

**Difference:** a finite source with one record gives `middleWave=None`
([brain.py:18489](../engine/omni_core/brain.py#L18489)). Its finite branch has no
periodic fallback ([18827](../engine/omni_core/brain.py#L18827)), even if that record
contains many token/audio/video windows. Start and end rehearsal exist, and
this does not imply the record's content is skipped. The middle rehearsal must
also work for long single-record sources.

### R8 — Registered evaluation copies can exceed the shared pool and survive failure

**Request:** U0113/U0170 include training, checkpoints and growth in the
reserved disk budget; U0190: “if u have multiple ais, use the largest one to
prevent multple storage use.”

**Difference:** registration copies into model-owned `engine/evaluation/data`
after `require_disk`, without the separate aggregate `reserve_spill` lease/file
registration ([registered_geometry_holdouts.py:99](../engine/omni_core/registered_geometry_holdouts.py#L99)).
That disk watermark is not the configured pool quota. Cancellation or hash
failure during copying precedes the save-only rollback block; it does not remove
new copies ([103](../engine/omni_core/registered_geometry_holdouts.py#L103),
[121](../engine/omni_core/registered_geometry_holdouts.py#L121)). Repeated failed
registrations can leave unused owned files. This is a concrete uncovered writer,
not a claim about arbitrary files the user independently creates.

### R9 — Registration and portability disagree on allowed filename suffixes

**Request:** U0626 answers “all unsanitized”; the accepted identity/export/fork
requirements preserve valid saved state and checksums.

**Difference:** registration retains arbitrary source suffixes
([registered_geometry_holdouts.py:100](../engine/omni_core/registered_geometry_holdouts.py#L100)).
Fork/recovery/export only accept ASCII alphanumeric/underscore/hyphen suffixes
([brainRepository.ts:1780](../src/main/brainRepository.ts#L1780),
[168](../src/main/brainRepository.ts#L168)). A valid registered source with a
Unicode or punctuation suffix can therefore save successfully and later fail
portability. An internal safe generated name can preserve the original name as
metadata; the brain's contents need not be sanitized away.

### R10 — Some activity graphics are still fabricated presentation values

**Request:** U0101 asks to see actual connections and firing; U0170/U0193/U0239
require real counters and progress. The audit already rejected confidence or
decorative arithmetic being reported as neural firing.

**Difference:** Library bars are fixed arrays chosen by card index
([App.tsx:1049](../src/renderer/src/App.tsx#L1049)) but labeled “Recent neural
activity” ([1116](../src/renderer/src/App.tsx#L1116)). Workspace header bars are
another fixed array ([2617](../src/renderer/src/App.tsx#L2617)). Queried Brain Map
and parameter totals use separate real paths; these graphics are not evidence
that memory itself is an answer table. The graphics should use observed samples
or be explicitly decorative.

### R11 — Block-sparse routing was promised; the recurrent router remains dense

**Request:** accepted U0042 describes block-sparse synapses between regions.
U0101/U0159/U0170 require scalable connected state and model spill.

**Difference:** packed router weights, eligibility, stability and uses allocate
the logical full pre/post matrix
([spiking.py:115](../engine/omni_core/spiking.py#L115)). The tiled forward explicitly
keeps every dense edge ([336](../engine/omni_core/spiking.py#L336)). Paging and
tiling are substantive, but do not create block-sparse topology. Sparse concept
connections elsewhere do not fulfil this separate router claim. Dense routing
cost remains proportional to the logical edge matrix.

### R12 — The RAM cap remains cooperative rather than a never-exceed bound

**Request:** U0130 says the cap affects everything; U0643 clarifies “doesnt use
more ram thsn thaat percent,” with compression/spill allowed.

**Present:** shared atomic reservations, managed-process accounting, hot/cold
paging and pressure pauses. Auto starts from total capacity with headroom; the
shared pool is the largest identity declaration.

**Boundary:** the source explicitly reports `hardRssIsolation=False` and a
one-second sample cache ([offload.py:821](../engine/omni_core/offload.py#L821)).
Tracked reservations reduce races, but do not intercept every native/driver
allocation or OS page fault. Accelerator restoration and some downstream
outputs still need admitted whole tensors; native format decoding retains
allocation minima. The exact never-exceed promise is not fulfilled. Storage
spill cannot make every possible tensor operation executable at arbitrarily
small RAM. This is a real implementation/platform boundary, not a new request
to pin physical RAM or lower the user's chosen ceiling silently.

## Two unresolved design choices

### Own replies and Ponder as lasting experiences

U0111/U0522 describe human-like experience and reactivation. Earlier permission
to use visible chat/history is not itself a decision to train automatically on
every self-generated statement.

Current code sets generated-response supervision eligibility false
([brain.py:14519](../engine/omni_core/brain.py#L14519)); generation-bound Ponder
uses `learn=False` with restored scratch state
([13476](../engine/omni_core/brain.py#L13476)). Both can affect temporary activity,
but are not automatically lasting independent teaching evidence. The earlier
self-experience question Q0059 has no recorded user answer. It was presented
again during this audit. A display acknowledgement is not agreement.

The question is whether the brain should learn its own acts as experiences
without assuming their contents are confirmed facts, or retain the current
incoming-evidence policy. Neither is silently declared the user's choice.

### Planned audio code-token generator versus continuous latent generation

The first accepted plan U0007 specifies a residual-vector-quantized neural
codec and token generator. Later same-brain requirements do not explicitly
settle whether those generated tokens must be discrete codec IDs.

Current audio has a real two-stage residual VQ codec, but its `token_generator`
predicts continuous latent tensors
([modalities.py:556](../engine/omni_core/modalities.py#L556),
[581](../engine/omni_core/modalities.py#L581),
[594](../engine/omni_core/modalities.py#L594)). Quantization returns vectors and
loss, not code IDs; there is no discrete code-ID prediction objective or
autoregressive codec-token sampling on this path. This algorithmic difference
is separate from deferred speech quality. The user was asked whether to complete
the discrete codec-token design or retain the current continuous generator.
Until answered, it is a possible plan mismatch rather than an approved redesign.

## Current feature disposition

The entries below cover the remaining agreement groups without reviving old
settings. “Source present” means an executable path exists; it does not convert
historical hybrid tests into current trained-native proof. Exact historical
requests and bug evidence remain in the private local
[full agreement matrix](TRANSCRIPT_REQUIREMENTS_AUDIT.md), which is intentionally
excluded from the public repository with the raw transcript.

| Agreement group | Current source disposition |
| --- | --- |
| Own brain and ternary learning | Native origin/architecture checks, packed projections/embeddings/convolutions/learned controls and legacy answer/foundation rejection are present in `brain.py`, `model.py`, `ternary_packing.py` and repository validation. No full floating learned master was found in the inspected native answering path. |
| Spiking/liquid/compositional ideas/global integration | LIF/STDP, CfC/internal LTC, VSA binding/adaptive rows, connected assemblies and global latent conditioning are present. R11 distinguishes dense routing from promised block sparsity. |
| Working/scratch/active/lasting movement | Lifecycle scoring, exact episode-linked token cooling, hot/cold working pages, spreading recall and Fresh epochs are present. R1 and the unanswered self-experience policy limit the stronger completion claim. |
| Growth, stability and retention | Resource-admitted assemblies/regions/experts, packed row resistance, replay and decay are present. No arbitrary small `maxIdeas/maxConcepts/maxSynapses` product setting was found in the inspected production path. Perfect memory and beneficial trained growth remain unproved. |
| Initial training and capability foundations | Native initialization, selected-data progress, required training before chat and no blocking quiz are present. Start/middle/end rehearsal exists with the R7 single-record exception. |
| Entire datasets and distributed updates | Deterministic manifests, committed cursors, token/media windows, incomplete-resource reporting, real packed collective updates and canonical fast/media replay are present. No new obvious coverage or distributed regression was confirmed by the read-only review. Native decoder allocation limits remain R12; the large corpus run stays deferred. |
| File/data types and uploads | Build/chat/Data file/folder/image/audio/video routes; PDF/EPUB/Office/OpenDocument/HTML/Markdown/code/CSV/JSON/JSONL/Parquet/Arrow/HF/archives/WebDataset/SQLite/remote manifests are present in `datasets.py` and host ingestion. Historical file/media-input tests remain closed. |
| Crawling and licensing | Persistent paced parallel frontier, dedup, provenance, same-site default, wider-link/robots/quarantine options and learner-failure requeue are present. No arbitrary new page-count limit was found in the reviewed crawl path. Historical crawl/search tests remain closed. |
| Teacher APIs | Explicit OpenAI/Claude/Gemini teaching, keys in the external vault, progress/cancel and native ingestion are present. Actual paid requests remain deferred. A teacher is not the resident foundation. |
| Natural tools, schemas, MCP and permissions | Files/shell/code/web/browser/history, MCP configuration, system access, Off/Ask/Auto/Full, settings awareness, configurable waits and schema/argument heads are present. Local recursive JSON Schema validation and same-turn tool observations exist; remote reference retrieval is not implicit. R2/R4/R5 qualify the on-the-go and evolution paths. |
| Ponder and idle activity | Learned action selection and per-prefix reevaluation/internal text conditioning exist without a hidden persona or forced Ponder. Durable self-Ponder learning remains a question. |
| Imagination and generation | Actual image VQ/latent diffusion, residual-VQ audio and factorized spatial/temporal video with liquid gating share the brain's idea state. Progressive artifacts/gallery/cancel exist; R3/R4 and the audio choice remain. |
| Voice and live perception | Default STT/TTS, live/buffered pace and interruption controls, independent native input/output ticks, camera/screen/microphone, temporary quality/FPS priorities, bursts and selectable snapshots, keys/pointer/scroll are present. Useful neural speech/perception remains deferred. |
| Chat lifecycle | Continuous identity, optimistic receipts, conversation ledger, anchored smooth output, Queue, explicit Cmd/Ctrl+Enter Steer, warm handoff, off-page work, Stop and no-reply handling are present. R2/R3 are surviving failure/completion defects. |
| Counters, search, traces and viewer | Exact compact parameter count/hover, token/search/job events, source coverage and paged cortical/assembly/synapse/token inspection with observed-only values are present. R10 identifies remaining decorative activity claims. Generated explanations are not verified chain-of-thought. |
| Appearance and notifications | Four-stage data-first Build, theme/light-dark bridge, Apple fresh defaults, layouts/profile add-update-delete/dice, compact action cards and notification controls are present. RAM/storage controls remain agreed. Toast-driven job notifications do not cover successful reply completion; no explicit reply-completion notification promise was found, so that scope is not counted as an unmet agreement. No new screenshot audit was run. |
| Identity, duplicate, agents and delete | Immutable origin, COW blobs, separate mutable state, duplicate/fork, reviewed evidence/file/overlay merges and three-confirmation deletion are present. Earlier exact deletions and accepted engineering checks stay closed. |
| Recovery/export/import/catalog | Unsanitized saved neural/chat/working/pending/cursor/recovery content, checksums, safe non-pickle bundles and native catalog/recipe routes are present. R8/R9 qualify new geometry continuation. Missing old payloads cannot be reconstructed; external source datasets are explicitly referenced, not silently relocated. |
| Recursive improvement | Isolated model/data/source/substrate/architecture candidates, protected evaluators, lineage, permissions, side-by-side promotion and rollback exist. R5/R6 distinguish executable mechanisms from the full requested recursive process. |
| Windows/macOS/Linux/mobile | Desktop x64/ARM64 and paired Android/iOS source/package routes are present. Companion implementation is accepted; local phone training is not an unfinished requirement. Prior releases retain their recorded closed status. |
| Video runtime and licenses | Pinned automatic six-target codec setup and the separate runtime/source/license release are recorded. PolyForm/commercial and third-party notices are present. New code is not a newly published application package. |

## Bug-fix verdicts and evidence that remains outside this pass

Source corrections for message disappearance/receipts, warm Steer, oversized
input, Unicode/EOS, per-prefix Ponder, parameter accounting, empty-map totals,
coverage/progress, recovery continuity and queried cortical activity remain
present. This review does not declare every historical screenshot condition
fixed merely because a helper test once passed. R2/R3/R8/R9/R10 are surviving
source defects affecting result learning, completion, storage/portability and
truthful presentation.

U0471 closed MCP/media/agents/evolution/recovery/export/device/voice and
PDF/PNG/WAV/MP4/folder engineering checks; U0479 closed repeated crawl/search.
Duplicate/delete and earlier theme/map/direct-tool checks stay closed. Their
historical evidence is preserved rather than repeated or erased. Later source
rewrites can limit how far those old results prove current native behavior.

Explicitly deferred, **not unfinished chores in this audit**: full supplied-folder
native training; ordinary experience/slang learning followed by unrelated
conversation, Fresh/restart and verbal recall; useful natural tool choice;
meaningful image/video/audio/speech; beneficial recursive generations; actual
native multi-GPU throughput, physical peak RAM/TPS and connected device/provider/
phone qualification. The stronger human-like/AGI/perfect-memory aspirations are
not source-certified outcomes.

No new design contradiction was resolved by silently choosing the assistant's
preference. The two questions above remain pending. The report is the audit
deliverable; the concrete remaining source differences are still open.
