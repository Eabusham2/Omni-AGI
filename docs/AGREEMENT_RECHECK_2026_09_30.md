# OmniCortex agreement and implementation recheck

The [implementation closeout](AGREEMENT_IMPLEMENTATION_2026_09_30.md) supersedes
the open source-difference status recorded in this historical recheck and its
subsequent audit. Trained native qualification remains user-deferred.

**Current follow-up:** the source at `20e36ee` was pushed to `main`. The later
[current agreement remainder](AGREEMENT_REMAINDER_2026_09_30.md) supersedes
current-status wording below. It records additional concrete source gaps and
the appended chronological messages; source presence is not full completion.
The old continuation boxes preserve their original scoped evidence and do not
mean every agreement is now met.

The repeat review found real remaining implementation limits. The current code
contains the intended native ternary learning mechanisms, but neither that fact
nor source tests establish that everything works like a human brain. This report
compares the latest agreed requirements with source, distinguishes partial
implementation from deferred proof, and does not reopen canceled choices or
user-closed tests.

## Active implementation continuation

The user rejected stopping with the findings still open. The `da64cd5` audit
commit is a checkpoint, not completion of the ensure/fix task. The checked
items below mean that an implementation path is present in this working tree,
not that a trained brain, GUI session, or GitHub CI has passed. The main task
must record affected source/control verification separately. Its later commit/
GitHub instruction supersedes the earlier local-only handoff; CI, release and
package work remain out of scope.

- [x] Cooperative, cross-process atomic RAM reservations through
  `shared_resource_ledger.py`, `offload.py` and the trusted desktop registry;
  `hardRssIsolation=False` remains an explicit physical/platform limit.
- [x] Shared spill-pool lease/physical-file ledger for the integrated core,
  attention, router and rollback/snapshot writers, with bounded reconciliation;
  native writers outside these paths are not thereby universally proved bounded.
- [x] Tiled recurrent multiplication and exact tiled STDP over paged router
  controls in `spiking.py` and `router_state_paging.py`.
- [x] File-backed CPU saved-activation restoration in
  `working_attention_paging.py`; accelerator tensors and downstream compute
  outputs retain admitted whole-tensor minima.
- [x] Lazy nested Arrow values in the production `_bounded_columnar_rows()`
  path; native batch/decompression allocation is still estimate-admitted,
  not an allocator-enforced hard bound.
- [x] Isolated width/head candidate migration, protected parent state and
  mandatory recorded training/held-out evaluation gates in
  `architecture_migration.py`, `geometry_candidate_application.py` and
  `evolution.py`; no candidate benefit has been measured here.
- [x] Remove a duplicate cortical checksum pass and preserve bounded exact
  integrity/delta operations; required full-byte verification remains I/O.
- [x] Actual host-result, typed same-turn observation transport into unfinished
  native decoding; inbox acceptance alone is not neural use.
- [x] Reviewed `jsonschema` validation with recursive local references and
  no implicit remote retrieval or product depth cap; unavailable external
  references fail closed.
- [ ] Resolve the unanswered automatic self-experience policy before changing
  it. Trained-native context/scratch/learned-state movement is deferred proof,
  not a model run authorized for this pass.

These are scoped source findings, not full AGI, trained recall, resource peaks,
or screenshot/GUI proof. No model/app/CI run is asserted by this update.

Review date: 30 September 2026. The original repeat-audit baseline was
application source `5b15e22` plus workflow-only `b35cc7d`; the source status
above and below additionally inspects the current, uncommitted continuation.
None of that is an installed package or an application tag. No GitHub CI, new
release, native brain creation/training, model quiz, or live model/UI run is
part of this recheck. The earlier release workflow was canceled and this pass
did not redispatch or monitor it. The later user request asks for verification
and a GitHub commit/push, but still excludes CI, release and package work.

## Agreement authority and read coverage

The primary agent read the full chronological snapshot from the first original
prompt through the latest audit request: 3,645 entries, including 646 user
messages, 2,865 assistant replies, 75 displayed plans, 56 question/answer records
and three pasted attachments. Repeated messages were retained. Two shortened
tool displays were repaired by reading their complete missing ranges. Extraction
reported zero malformed records. Question-display acknowledgements were not
treated as user answers. Historical screenshot references are preserved, but
their missing pixels are not claimed as read. Private reasoning and ordinary
non-dialogue tool dumps are not conversation messages.

The complete ordered transcript and extraction manifest remain private local
audit artifacts, not public package contents. This is a completed reading of the
snapshot, not a promise of permanent verbatim retention after compaction. The
following command request and the actual current-input overflow answer were
also read. The refreshed appended range 3,646–3,661 was read in full: the final
read snapshot contains 648 user messages, 2,877 assistant replies, 58 question/
answer records, 75 plans and three attachments, with zero malformed records.
Later status commentary is not a new requirement or grounds to restart the
already completed chronological pass.

Later corrections override earlier plans. The final decisions are:

- Own initialized/trained OmniCortex, not Falcon/Llama/BitNet model weights or a
  converted language foundation: original Q0004/A00008, then U0261–U0267 and
  U0533–U0537.
- Packed authoritative learned ternary values, no complete floating master
  model: U0419–U0422 and U0558–U0569 supersede the earlier master-weight plan.
- Ordinary experience and connected neural state, not fact parsing, cue-answer
  tables or verbatim answer replay: U0353–U0361, U0494–U0496, U0512 and U0535.
- Overlapping working/scratch/active/lasting dynamics, automatic settling and
  reactivation, no manual consolidation or visible rigid stages: U0111–U0122,
  U0442–U0449 and U0522. Fast synaptic adaptation is primary; later cortical
  replay is refinement, not proof of instantaneous fluent recall.
- Device-derived context maximum, initial training then chat without a quiz,
  and automatic codec setup: actual answers in U0632.
- Unsanitized saved-instance exports: U0626. The separately stored OS credential
  vault is not saved brain state and remains excluded.
- Separate 20-GiB disk floor and device-aware RAM reserve: U0113, U0132, U0449
  and the history-directed answer U0626.
- Auto uses total selected device RAM, leaves speed headroom and avoids normal
  spill with other apps closed; the selected RAM percentage is a ceiling, with
  compression/spill permitted: U0170 and the explicit U0643 clarification.
- At the historical audit cutoff, code comparison and local commit without
  GitHub CI/application publication: U0646 overrode U0642. The later direct
  GitHub commit/push request changes the handoff, not the CI/release exclusion.
- Older context yields room through the existing memory process; never crop a
  new message or silently enlarge the selected window. If the new message
  itself is larger, block Send: the actual overflow answer U0648.

No unresolved contradiction requires reviving physical RAM pinning, fixed 4M,
Starter/Blank choices, a blocking tool quiz, personality sliders or Falcon
compatibility. The limitations below are not permission to silently change the
design or substitute a less capable mechanism.

## Historical gaps and current source boundaries

### 1. The RAM percentage is not an instantaneous hard usage limit

Request: U0130 asks for a system-wide RAM cap; U0175 says to lock it; U0643
clarifies “doesnt use more ram thsn thaat percent,” while allowing compression
and spill.
Your exact U0130 wording also says “this also effects everything.”

Current source: [`shared_resource_ledger.py`](../engine/omni_core/shared_resource_ledger.py)
atomically escrows cooperating RAM allocations in SQLite;
[`offload.py`](../engine/omni_core/offload.py) combines that escrow with verified
managed-family residency, and
[`sharedResourceRegistry.ts`](../src/main/sharedResourceRegistry.ts) publishes
the trusted selected ceiling. The sample epoch prevents a cached reading from
prematurely releasing newly allocated escrow. `crossProcessAtomicReservation`
is now true when the shared ledger is active.

Residual boundary: [`managed_process_memory.py`](../engine/omni_core/managed_process_memory.py)
still caches readings for up to one second and the policy still reports
`hardRssIsolation=False`. Cooperative reservations reduce races among covered
allocations; they cannot prevent untracked native/driver allocation or impose
universal OS RSS isolation. No strict never-exceed measurement is claimed.

### 2. Aggregate spill quota is implemented for integrated writers, not all native writes

Request: U0113/U0170 reserve model, full context, training and future growth;
U0190 says multiple identities share the largest pool rather than reserving it
repeatedly.
U0190: “if u have multiple ais, use the largest one to prevent multple storage use.”

Current source: [`shared_resource_ledger.py`](../engine/omni_core/shared_resource_ledger.py)
charges pending reservations and durable physical file identities against the
largest registered pool, reconciles missing/dead owners and deduplicates known
hardlinks. [`native_core_paging.py`](../engine/omni_core/native_core_paging.py),
[`working_attention_paging.py`](../engine/omni_core/working_attention_paging.py),
[`router_state_paging.py`](../engine/omni_core/router_state_paging.py),
[`slow_state_snapshot.py`](../engine/omni_core/slow_state_snapshot.py) and
[`parameter_diagnostics.py`](../engine/omni_core/parameter_diagnostics.py)
now use spill leases through `ResourcePolicy.reserve_spill()`; desktop saved
brain declarations flow through `sharedResourceRegistry.ts`. The separate
20-GiB free-space floor remains in force.

Residual boundary: the ledger accounts for integrated writers and registered
file identities, not a measured guarantee for every possible external/native
write or filesystem-level copy-on-write sharing. Its status conservatively
counts unverified COW files. Failures must pause without treating unremoved
backing files as freed pool space.

The current desktop startup also reconciles app-managed saved UUID owners
against the library before worker launch: a crashed deletion cannot leave an
absent brain's oversized pool declaration active. This never credits Python
leases or backing files, and invalid committed owner declarations fail closed.

### 3. The recurrent router is tiled and paged, but not block sparse

Request: U0159 explicitly includes the model itself in spill; accepted U0042
describes block-sparse connections; U0101/U0170 require growth without resident
whole-brain materialization or misleading low-memory claims.
U0159: “not just training use storage also like the model itself when it doesnt all fit on ram.”

Current source: [`spiking.py`](../engine/omni_core/spiking.py) now uses bounded
tiles for recurrent multiplication, exact causal/anti-causal STDP, decay and
explicit inspection. [`router_state_paging.py`](../engine/omni_core/router_state_paging.py)
backs packed synapses and dense control matrices with paged state and a bounded
mutation journal; [`brain.py`](../engine/omni_core/brain.py) reports
`routerControlStatePaging=True` on load. All logical edges still participate:
this is tiled dense semantics, not a new block-sparse topology.

Residual boundary: hot spike/trace vectors and a tile remain resident;
full-matrix inspection is explicitly refused above one tile. This source path
does not prove a trained large router's memory peak or throughput.

### 4. Saved activation paging does not make every computation pageable

Request: U0125–U0129/U0159/U0170 ask for low-RAM training and storage overflow,
not merely a larger context slider.
U0125: “when i tried training i ran out of memorey so see if u can fix that.”

Current source: `SavedActivityTensor.restore()` in
[`working_attention_paging.py`](../engine/omni_core/working_attention_paging.py)
builds a verified, leased file image in bounded page transfers and returns a
CPU tensor backed by a private mapping, avoiding a complete new anonymous CPU
copy. Saved-page lifetime remains tied to mapping ownership.

Residual boundary: accelerator restoration still allocates an admitted full
device tensor. Downstream operators can demand complete outputs/gradients and
fault mapped pages into RAM. Large paged inference context remains distinct
from unrestricted full-context training; a resource pause is not completed
dataset traversal.

### 5. Some native and nested dataset values still require whole allocation

Request: whole-dataset and low-RAM U0042/U0125–U0131/U0170–U0172; A00928
specifically promised splitting huge columnar records without allocating them
whole.
U0172: “fix it skiping rest of trainijg.” Resource rejection must remain visibly
incomplete, not masquerade as complete traversal.

Current source: [`datasets.py`](../engine/omni_core/datasets.py) now represents
nested list/map/struct children lazily in `_bounded_columnar_rows()` and
traverses selected content with bounded encoding/text leases. That production
Parquet/Arrow ingestion path no longer makes a whole nested `as_py()` row copy.
The older `_iter_columnar_rows()` helper still contains `as_py()` but has no
production call site in this module.

Residual boundary: PyArrow batch/dictionary/decompression and unknown extension
leaf allocations still depend on the estimates in
[`columnar_admission.py`](../engine/omni_core/columnar_admission.py), not an
allocator hard cap. Speech and other native formats retain their own minima.
No universal native-OOM immunity or completed full-dataset run is claimed.

### 6. Architecture evolution has a defined compatibility boundary

Request: U0038, accepted U0042 and U0096 retain recursive model/data/source/
architecture improvement, not merely manually added residual experts.
U0038: “Make sure it has recursive self improvement too.”

Current source: [`architecture_migration.py`](../engine/omni_core/architecture_migration.py)
now declares explicit `resize-width` and `repartition-heads` operations and
bounded coordinate-copy plans. [`geometry_candidate_application.py`](../engine/omni_core/geometry_candidate_application.py)
applies them only in an isolated candidate, checks parent ownership/aliases,
and preserves the original durable activity bytes. [`evolution.py`](../engine/omni_core/evolution.py)
requires recorded training, registered real held-out token/modality/tool and
resource measurements, a passing immutable evaluation and explicit geometry
authorization before promotion.

Residual boundary: changed normalization, head partitioning and RoPE make
function preservation false. Newly introduced coordinates require actual
learning, and no beneficial real candidate or unrestricted arbitrary geometry
migration has been demonstrated by this source review. The registered evaluator
uses all selected video frames but does not separately score embedded video
audio; that narrower boundary is not presented as full audiovisual quality.

### 7. Total-dependent integrity and scoring work remains

Request: U0127–U0129/U0191/U0412/U0461 ask for efficient operation, avoiding
excessive drive work and long small-turn delays.
U0127: “not to cook ssd and slow down mutch”; U0191: “its way to slow for its size.”

Current source: [`brain.py`](../engine/omni_core/brain.py) reuses an already
verified slow/cortical checksum for the same owners instead of immediately
hashing them twice. Router checksums now stream exact historical decoded
bytes without a whole decoded matrix; changed-byte journals and bounded
snapshots remain in [`parameter_diagnostics.py`](../engine/omni_core/parameter_diagnostics.py)
and [`slow_state_snapshot.py`](../engine/omni_core/slow_state_snapshot.py).

Residual boundary: `parameter_checksum()` and `_slow_parameter_checksum()`
still perform required total-byte integrity reads; exhaustive semantic
scoring may visit the whole learned state, and an inline-media snapshot copies
its selected region. Scratch-RAM bounds do not make these constant-I/O or
establish 10+ TPS or a 20-second cold start. Integrity checks are retained.

### 8. Typed tool results can condition the same unfinished reply; live competence is unproved

Request: the original in-chat tools requirement, accepted U0042's natural
action head and U0098 require the brain to use its actions on the go. U0512
explicitly rejects fact/answer stores as the learning substitute.
U0098: “ai should be able to use internet or all of its actions/tools onthe go
aslong as its permited by aceess.”

Current source: [`chatActionController.ts`](../src/main/chatActionController.ts)
offers the completed, permissioned host execution while its turn still owns
unfinished output. [`chatToolObservation.ts`](../src/main/chatToolObservation.ts),
[`worker.py`](../engine/worker.py) and
[`chat_tool_observation.py`](../engine/omni_core/chat_tool_observation.py)
bind/validate the exact brain, turn, native action, execution and result.
[`brain.py`](../engine/omni_core/brain.py) encodes the actual result into native
conditioning, and [`model.py`](../engine/omni_core/model.py) invalidates stale
cache state before subsequent forward use; the trace distinguishes inbox
acceptance from `nativeDecodingUsed`. Natural EOS can wait for an already
requested result within the configured wait, without forcing a reply.

Residual boundary: late, canceled or resource-paused observations need not
affect that turn. The later source review found that ordinary chat failure/Stop
lacks the completed-result learning drain present on zero-token Steer/native
Stop; completed results are therefore not unconditionally guaranteed to enter
durable learning. This is no evidence that an untrained action head chooses useful tools
or composes another useful same-turn decision. No fake human turn, answer
replay or hidden prose prompt is added.

### 9. Typed tool schemas have a compatibility boundary

Request: accepted U0042 and subsequent MCP/browser requirements retain genuine
structured schemas and model-produced arguments, not hard-coded word triggers.

Current source: [`schema_validation.py`](../engine/omni_core/schema_validation.py)
uses the reviewed `jsonschema` validators, preserves assertion structure and
recursive local references, and validates host-exact safe integers. The
[`native_action_protocol.py`](../engine/omni_core/native_action_protocol.py)
wrapper no longer imposes a product depth-32 or enum-product cap. The
dependency is pinned in [`requirements.txt`](../engine/requirements.txt) and
platform locks; this source update does not claim the locked installation was
run in every target environment.

Residual boundary: no unpermissioned network/file reference retrieval is
performed. An unavailable external dialect/reference fails closed; physical
interpreter recursion pressure is a recoverable pause, not silently weakened
validation. Historical user-closed MCP fixtures retain their recorded status.

## Memory movement checked in source

| Agreed behavior | Actual source and boundary |
| --- | --- |
| Current user input participates without “remember,” Q&A or fact grammar. | `chat()` previews current activity, recalls internal vectors and admits the genuine fast experience after a successful/correctly owned completion. Cancellation does not commit an incomplete turn. |
| Current material stays active; older scratch activity can fade with room still free. | `RecentTokenActivity.cool()` follows exact afterimage IDs and existing lifecycle signals, protecting current/unfinished/reused activity. Unlinked historical spans remain conservative rather than having fabricated episode bindings. |
| Recurrence, salience, unrelated interference, reuse, stability and decay influence retention. | `OrganicMemoryLifecycle` scores overlapping activity and selective replay. These are engineered dynamics, not measured biological equivalence. |
| Useful experience changes the same learned neural state used by generation. | Adaptive packed assembly/neuron rows and sparse synapses plus STDP condition the native decoder through vectors, not retrieved sentences or stored answer sequences. |
| Slow refinement must not block each chat turn. | Production worker sets `defer_slow_learning=True`; durable queued replay uses explicit source teaching evidence and changes cortical weights only when the job really completes. Queued is not completed. |
| Related memory can reactivate; retired raw text must not be pasted back. | Spreading neural recall and hot/cold state feed internal conditioning. Cooling never reconstructs raw spans from history or IDs. Explicit history-tool use remains allowed. |
| Fresh clears temporary attention, not history or established learned state. | Attention epochs, idempotent receipts and recurrent/scratch clearing remain distinct from parameter mutation. Correct post-Fresh verbal recall is still deferred native proof. |
| Older raw context makes room, without cropping a new message. | The complete current byte-token input has priority over older prompt words. Previously learned/scratch influence remains under the existing lifecycle. A current message plus its three role boundaries exceeding the selected window is blocked in the renderer, host and native entry; the setting is not automatically enlarged. |

This confirms executable mechanisms, not useful one-experience recall,
generalization, perfect retention or instantaneous training of every cortical
weight. Those outcomes cannot be certified by checksum changes alone.

### Extra context to parameter source check

The user's follow-up specifically asked for deeper checking of this flow. The
terminal-pasted transcript was compared entry by entry with the previously
read ordered source. Its 3,645 entries add no new historical instructions:
3,643 message bodies match, one old message lost carriage-return text in
terminal rendering, and the last reply has the terminal prompt appended.
The intact raw-log wording remains authoritative, not the shortened copy.

The actual production call chain is:

1. `Worker` calls `AdaptiveBrain.chat()` with deferred slow learning and
   cooperative cancellation. `_preview_chat_experience()` computes current
   neural conditioning without committing an unfinished/canceled pair.
2. After successful visible decoding reaches its owned completion boundary,
   `commit_fast_experience()` calls `learn_experience(..., steps=0)` on the
   complete human experience. `vsa.py` forms segment/whole assemblies,
   strengthens ternary pathways and adapts coactive packed neuron rows.
   `router.route(..., learn=True)` applies enabled STDP. This is real fast
   learned state, not a saved sentence selected as a response.
3. `_append_working_memory()` and `_settle_memory_automatically()` keep
   transient vectors/focus. `OrganicMemoryLifecycle.settle()` remeasures
   recurrence, salience, interference, reuse, stability and dormancy; it does
   not assign a permanent category or force visible Ponder. Unfinished
   afterimage activity can reenter working activity as a vector.
4. `RecentTokenActivity.cool()` follows exact episode IDs and live signals.
   Weak older raw spans can retire with space still free; current/unfinished/
   reused spans are protected. Capacity selection may omit older prompt
   words, but no retired sentence is reconstructed from an episode ID or
   history. The retained current input is never suffix-cropped.
5. `_enqueue_chat_slow_learning()` references authenticated human teaching
   evidence, not an answer key. The host schedules
   `consolidate_pending_chat_learning()` only when its serial worker can
   obtain background ownership. `_optimize_experience()` then traverses
   bounded token windows, performs gradients and commits packed cortical
   updates atomically with the job tombstone. Failed or canceled work remains
   pending or rolls back; it is not recorded as successful learning.
6. Later generation reads distributed learned vectors and recurrent spreading
   activation through internal conditioning. It does not paste stored source
   passages into the default answering prompt. `start_fresh_attention()`
   clears raw recent tokens, working/focus/afterimages, liquid and router
   activity, verifies learned-state preservation and saves an idempotent epoch.

The timing boundary matters: ordinary transactional chat admits fast experience
after its owned completion, not on every incoming keystroke. Cortical jobs may
wait behind continuous foreground work or resource pressure. Raw eviction does
not itself perform gradients or establish that every older token is already
encoded in cortical weights. The source therefore supports automatic overlapping
activity and real learning, **not a guaranteed realtime transfer of all context
into the language cortex**. That stronger claim would need additional scheduling
and trained-native evidence; this report does not quietly count it as fulfilled.

Own generated speech remains visible recent context, but the current code does
not automatically use it as an independently verified slow-training target.
This avoids treating a generated mistake as new teaching evidence; it does not
erase history or forbid the explicit history tool. It is not proof that all
self-reflection experiences are durably learned. Fast state changes, a queued
job, a completed cortical mutation, transient occupancy and successful recall
must remain separately reported.

Because automatic lasting learning from its own speech/Ponder was not explicitly
resolved in the history, a focused clarification was asked before changing that
policy: should those be learned as experiences without treating their claims as
true facts, or should current independent-evidence requirements remain? No
answer has been received at this report's cutoff. The current policy is
preserved, not declared user-approved or quietly replaced.

## Source defects corrected in the earlier scoped pass

- Foreground slow-replay admission scanned all assemblies just to find the
  current ID. It now uses `assembly_by_id`, including the paged exact index,
  without changing retention signals or creating an answer key.
- Mid-response Ponder previously changed only the cue for subsequent action
  selection. The correction carries refined same-cortex memory conditioning
  into subsequent text decoding and invalidates incompatible cached state.
  Repeated identical Ponder was also suppressed for the entire turn. A new
  genuine prefix decision may now select it again; replay of that exact
  decision stays idempotent, and external effects remain exactly-once.
- Native printable generation previously blocked initial EOS and non-ASCII
  text. The correction uses UTF-8 syntax boundaries, allowing an immediate
  learned EOS/no-reply and valid Unicode without manufacturing assistant words.
  NUL is excluded because existing saved-message validation forbids it; other
  scalar languages and ordinary wire controls are not preference-filtered.
- The decorative cortex activity display could average curiosity/coherence/
  novelty when no activation observations existed. It now reports no observed
  value in that case; a real mean is labeled as a mean of observed values, not
  the percentage of the entire brain firing.
- The evolution-controller comment had still called compatible depth/router
  changes unsupported after their implementation. It was corrected to match
  the width/head restriction at that earlier baseline; the current isolated
  geometry candidate path described above supersedes that restriction.
- The host's unrelated 100,000-character ceiling and native current-message
  suffix crop are removed. Exact UTF-8 capacity checks block a message that
  cannot fit, preserving the draft and running before new neural work. A
  matching already-completed receipt can still recover without another decode
  if context was reduced afterward; changed text cannot claim that receipt.
- Zero-output warm Steer now carries actual unfinished human input, bound to
  brain, original turn/hash, successor and native Fresh epoch. It uses only
  transient HUMAN role segments, not a synthetic assistant response, stored
  answer, behavioral prompt or history lookup. Completed partial turns are
  not duplicated; Fresh discards old carry. Complete carry must fit the
  selected window and admitted token RAM or explicitly pause without cropping.

Those fixes are source/control contracts, not proof of beneficial learned
Ponder, fluent Unicode generation, ordinary-experience recall or usable native
latency. No exact answer table or imported foundation was introduced.

## Current requirements with source paths present

These are code findings, not blanket behavior passes.

| Requirement group | Current path | What is not claimed |
| --- | --- | --- |
| Own brain and mandatory ternary learning | Native origin/architecture validation; packed linear, embedding, convolution, controls, liquid/router and shared substrate rows; tiled/paged recurrent router execution; legacy answer/foundation payload rejection. | Broad fluency, 1.58 bits for every runtime byte, or a measured large-router memory peak. |
| Spiking, liquid, VSA and whole-input processing | LIF/timed STDP, CfC/internal LTC, binding/bundling, causal global workspace and connected assemblies. | Human-equivalent cognition, consciousness or achieved AGI. |
| Growth and anti-forgetting | Resource-admitted regions/assemblies/experts, packed row resistance, recurrence/replay, no arbitrary idea eviction, and isolated width/head candidates behind training/evaluation gates. | Perfect non-forgetting, function preservation after geometry change, or unrestricted arbitrary migration. |
| Initial and whole-dataset learning | Required initial curriculum/selected-data progress; no first-chat quiz; start/mid/end capability rehearsal; exact committed coverage and resume. | A bundled broadly pretrained brain or a completed current full-folder run. |
| Uploads and formats | Chat/Build/Data file/folder/media pickers; PDF, EPUB, Office/OpenDocument, HTML, code, CSV/JSON, Parquet/Arrow with lazy nested values, archives/WebDataset, SQLite and remote manifests. | Every historical native-picker condition or universal native-decoder RAM bounds. |
| Web and teacher learning | Persistent paced same-site frontier, parallel fetch, stop/resume, external/robots/quarantine settings; explicit OpenAI/Claude/Gemini teachers. | An unperformed paid-provider call or pretrained weights silently becoming the resident AI. |
| Natural tools and MCP | Neural action/argument schemas, standards-based local JSON Schema validation, typed generic/MCP/browser payloads, same-turn actual-result observation into native decoding, permissions, settings awareness and configurable approval window. | Useful unseen-tool judgment, implicit remote schema retrieval, or trained same-turn tool competence. |
| Ponder and spontaneous actions | Always-available learned selection, repeated per-prefix Ponder and generation-bound text/action conditioning, no backend persona prompt/slash-only dependency. | That the untrained brain reliably chooses when pondering or tool use is useful. |
| Same-brain media and voice | Shared idea/sensory/motor regions, progressive image/audio/video, cancellation/gallery, paired own-speech conditioning; STT/TTS default and independent neural input/output ticks. | Meaningful imagination, intelligible neural speech, real-time quality at arbitrary resolution or a different hidden media assistant. |
| Live perception and devices | Permissioned camera/screen/microphone, source/resource-derived quality/FPS, priorities, snapshots/native/current/custom bursts, keys/pointer/scroll. | A new physical-device or driver-specific test in this audit. |
| Chat lifecycle | Continuous identity/history, optimistic receipts, anchored smooth output, ordinary Queue, explicit Cmd/Ctrl+Enter Steer, off-page work and exact cancellation ownership. | Fresh packed-native latency/navigation behavior proved by old hybrid results. |
| Counters and traces | Compact exact parameter breakdown, token/search/job activity, real committed sources, current context, output/commit separation and typed operational traces. | Cumulative updates as unique synapses, confidence as firing, or a prose self-report as verified chain-of-thought. |
| Viewer | Paged sparse map plus packed cortical byte/range/token-path drilldown, search/back/zoom/continuation and observed-only activation coverage. | A complete semantic explanation of thought or a new visual screenshot audit. |
| Setup and appearance | Data-first four stages, one ordinary memory policy, no personality/Starter controls; Default/Classic/Colorful/Liquid Glass, layout/profile CRUD/dice, Auto light/dark and Apple fresh defaults. | Every viewport, hover/highlight, scrolling and overlap condition freshly observed here. |
| Identity/storage/portability | Immutable origin, isolated COW fork/duplicate, reviewed merges, three-confirmation delete, complete unsanitized saved state and recovery references. | Process cloning, arbitrary external-data relocation or reconstructing already-missing historical files. |
| Recursive improvement | Isolated typed candidates including width/head geometry, protected evaluators/origin/permissions, real registered geometry holdout gates, lineage/archive, approved or authorized promotion and rollback. | Demonstrated beneficial or meta-recursive intelligence improvement. |
| Cross-platform product | Windows/macOS/Linux x64/ARM64 package routes; Android/iOS companions to the same identity; PolyForm/commercial and third-party notices. | Local phone training, new signed binaries without credentials, or a new application release in this pass. |

## Deferred proof and closed or replaced work

Deferred by the user, not unfinished chores in this audit: full supplied-corpus
native training; ordinary story/slang teaching followed by unrelated turns,
Fresh/restart and generated recall from changed weights; useful natural tools;
meaningful image/audio/video/speech; beneficial recursive improvements;
trained-native device/provider/companion sessions and real peak resource/TPS
qualification. Old Falcon or exact-sequence answers cannot qualify these.

U0471 closed historical MCP, media, agents/evolution, recovery/export,
device/voice and PDF/PNG/WAV/MP4/folder engineering checks; U0479 closed repeated
crawl/search checks. Duplicate/delete, basic map totals, direct safe tools and
prior theme/UI checks retain their recorded disposition. Later rewrites limit
their applicability but do not authorize repeating all of them now.

Replaced rather than missing: Windows-only scope; v2 naming; Blank/Starter and
personality recipes; exact-source retention choice; manual consolidation;
visible rigid memory stages; fact/sequence answer replay; imported Falcon
runtime/compatibility; fixed 4M slider; mandatory Ponder; blocking first-chat
quiz; independent code AI; requiring local neural phone execution; expensive
brain-training hosted CI. The v1.1.0 and separate codec releases remain
historical completed work. The newly canceled application CI/release is excluded
from this local handoff, not added as an unmet gate.

## Verification recorded for the earlier repeat audit

Only affected constructor-free/helper/controller checks and static verification
are authorized. No test result below is a trained-model capability verdict.
The completed-turn fixture creates only an isolated presentation repository;
it starts no neural worker, model or app and touches no user-saved instance.

Verified in that earlier scoped pass, before the current source continuation:

- Replay-index fixtures: 2 passed.
- Current-message capacity/idempotence fixtures: 5 passed.
- Native text boundary/Ponder protocol fixtures: 11 passed.
- Temporary Steer native input fixtures: 8 passed.
- Per-prefix Ponder identity fixtures: 3 passed.
- TypeScript input admission: 5 passed; measured-activation helper: 3 passed;
  temporary Steer controller: 4 passed; reduced-context completed-receipt
  recovery: 1 passed, with the other 8 historical receipt cases intentionally
  skipped in that focused invocation.
- Affected source-only GUI checks in that earlier pass: 13 defect
  contracts, 9 visual/CSS contracts and 12 no-reply presentation contracts
  passed. They were not rerun just to increase a progress count. Static CSS
  checks are not screenshot or computer-use verification.
- Both Node and renderer TypeScript checks passed then. Syntax compilation of
  the Python sources/tests changed in that pass and `git diff --check` passed
  before its staging; this is not a fresh check of the present working tree.

Repeated executions are not counted as new coverage: the distinct totals are
29 Python and 47 TypeScript contract checks. The scripted token loop uses
programmed logits and tiny control tensors, not learned forward inference.
No corpus training, brain quiz, native quality run, app/UI launch, kernel
benchmark, new GitHub CI, push or application release occurred in that earlier
pass. This sentence does not report the current commit/push outcome.

The nine historical implementation gaps above now have specific source
follow-ups and residual physical/behavioral boundaries. The prior audit is a
request-to-source comparison and scoped correction handoff, not a claim that
every feature is fully implemented or biologically equivalent. The checks
listed here predate the current continuation; its affected validation and
commit/push outcome must be reported by the main task, not inferred from
these earlier counts. Deferred native and GUI qualification stay separate.

In the current continuation, before the user's instruction to stop checks,
100 selected Python source/fixture checks, 41 selected desktop checks and 15
evolution-controller fixtures passed; both TypeScript projects and all five
engine dependency-lock manifests also passed their focused checks. The final
geometry registration/evaluator corrections and portable-holdout wiring were
committed without rerunning those checks at the user's explicit direction.
No native model, full dataset, app session, CI, package or release was run.
