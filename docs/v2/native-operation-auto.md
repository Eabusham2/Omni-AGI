# Current-owner operation Auto: perception, retained steps and native media artifacts

This is an incremental R1 implementation, **not R1 closure**. No brain, Load,
initialization, training, generation, neural calibration, hardware comparison,
provider call or device capture was run for this slice. The user's deferred
new-brain/merged-resume proof remains deferred. Existing prefill and one-shot
cached-decoder interfaces remain separate and working; they were not replaced.

## Actual source path and scope

`src/shared/nativeOperationAuto.ts` supplies the additive DTOs.
`engine/omni_core/operation_auto_policy.py` owns explicit current-session policy,
qualification and metadata-only controls. `operation_auto_native.py` supplies
the actual already-loaded owner's callbacks. `AdaptiveBrain`, `Worker` and
`worker_transport` provide serial loaded-owner and separate flags-only seams.
`ModalityHub.perception_embedding` and its connected override have real
decorated entry points. An unbound or unarmed hub calls the original method;
importing/opening an app performs no benchmark or model operation.

The three perception work kinds are exactly:

- `native.perception.image-embedding`;
- `native.perception.audio-embedding`;
- `native.perception.video-embedding`.

The internal Review/Status contract is now version2 and declares exact stage
coverage per supported work kind. The additive ordinary/connected retained-step
consumer below adds two work kinds; the prepared media consumer adds three
artifact kinds. Together with the prepared training wave, two full-prefix
consumers and two uncached full-forward consumers, the current registry has
thirteen individually declared kinds. Each owner advertises its actually
supported subset; an absent video codec does not require an invented video
route. This is not a product or `.omni` version.

The selected operation is the existing tensor-to-embedding computation, not
image recognition quality or a whole media task. The ordinary image closure is
the enabled vision encoder, or the image encoder plus projection; audio/video
use their existing encoders and projections. For connected perception, the
closure explicitly traverses every borrowed parent hub, its real resize
geometry and learned workspace-exchange bridges. `hub.named_modules()` alone
would omit borrowed connected parents, so it is not used as a false complete
owner inventory. Actual shared module objects are deduplicated while parent
call cardinality and original region/generation identities remain distinct.
Connected exchange output is preserved as it really is, not asserted to be a
normalized ordinary single-region output.

Raw file decoding, source preprocessing, lasting sensory learning, graph
updates, active/working memory, router recurrence and user/tool callbacks lie
outside this entry and are **not replayed**. Private nested connected calls
bypass policy re-entry; the complete entry has one controller. No alternate
model, learned floating master, output lookup, raw-text prompt retrieval or
external vision system is introduced.

## Explicit intent and reachable qualification

1. Prepare reviews the current loaded owner, actual compute topology, physical
   groups, selected device budgets and exact registered closure. It performs no
   perception or quiz. Reviewing over an active policy is refused: use explicit
   Stop first, rather than silently resetting it.
2. Arm consumes the main-owned exact review and operation ID, then awaits the
   next genuine ordinary perception input. It does not synthesize an input or
   start a model. Only an enabled, supported, float32, no-grad operation at a
   quiescent owner boundary is eligible. Current live cache/spill handles, pins,
   transactions, connected mutation holds and independent inline work are not
   cleared or terminated to force eligibility.
3. The actual input is guarded by its original tensor identity/version/layout,
   exact byte SHA, source geometry and current owner/policy/topology bindings.
   Borrowed storage and closure/version metadata are resource-admitted. There
   is no whole-model hash per frame or an unsaved-state blanket exclusion.
4. One excluded warm-up per candidate is followed by at least three real samples
   per candidate, round-robin, using the **same actual input and native state**.
   No sampling/noise draw is part of these pure encoders. The first current
   candidate produces the reference. Finite shape/dtype and **exact embedding
   equality** are required; this slice introduces no numerical tolerance.
5. The existing streamed whole-declared-path selector can choose only an
   admitted Auto candidate whose measured maximum latency is below the current
   candidate's measured minimum, using authenticated same-work receipts. That
   finite-sample condition is not statistical confidence or a neural-quality
   guarantee. CUDA requires its explicit packed-CPU-tile acknowledgment; it is
   never relabeled a direct native CUDA kernel. XPU/NPU are unavailable here.
6. An accepted route is held only as resource-admitted, process-private derived
   metadata. The next compatible input is **computed anew** with that route.
   Same operator/geometry/current participating learned state and policy are
   checked; the fresh result must have the checked output shape/dtype and be
   finite. The previous embedding is not retained in the plan or returned.

The reusable route is explicitly an **operator/geometry inference for unseen
inputs**, qualified on one input. It does not claim that a later input was
already measured, that exact equality was separately tested for every future
input, or that the selected route must stay fastest forever. The public plan
reports `qualificationWorkloads:1`, `unseenInputOrFrontierWasPreviouslyMeasured:false`
and `wholeBrainOrTrainingCovered:false`. A participating learned mutation
requires fresh qualification. Unrelated native owner updates need not change
this closure's learned epoch. A checked ordinary-route result is retained as a
conservative decision too, so the same rejected/inconclusive pool is not
benchmarked on every subsequent unchanged frame.

## Current learned ownership, restoration and physical resources

`scoped_owner_observation.py` and the narrow pager observer track only actual
selected packed owners. Persistent signs, strength/plasticity/stability and
repeat endpoint pages contribute their true version changes. Disposable sort
indices and known exact device/buffer representation moves do not masquerade
as learning. Independent observer leases cannot be retired by the older global
decoder observer. The unregistered pool's tiny CPU ordinal counter is hashed
as a small persistent control, not incorrectly asserted to be a pager owner.
The complete closure/config/region metadata and small non-paged controls are
also bound. This is an owned runtime mutation fence, not protection against
arbitrary privileged external memory tampering.

Candidate execution moves **only the declared selected closure's activity and
packed residency**. It does not call a global brain/model `.to()`, change
`brain.device`, alter saved hardware preferences, arm manual regional routes,
clear context or change module training flags. These declared encoders have no
dropout/batch-normalization/recurrent writes; normal no-grad perception may have
`Module.training=True`. Unknown classes, instance forward/encode overrides,
callbacks/global hooks, in-place SiLU and live learned transactions are refused
for comparison instead of being replayed speculatively.

Current packed-owner admission, physical-group budgets, shared RAM accounting,
actual transfer reservations and current read-only learned-write guards remain
in force. Transient demand includes actual convolution positions/patches,
encoder outputs, resized connected inputs/bridge sums and the existing weighted
kernel's **int64** row/concatenation buffers. It is a conservative declaration,
not an exact allocator peak. Repeated-contact pages retain their real existing
owner and cursor admission. CPU/Metal unified physical capacity is not counted
twice; unused distinct GPUs are not credited as consumed capacity.

Original selected packed residency and control references are restored before
delivery on success, failure or Stop. The pager's expendable cache may reclaim
unrelated unpinned owners under its current budget; this is not a claim to
restore the entire OS/global accelerator cache layout. No learned epoch is
reset to hide a real mutation. A genuine failed exact restoration marks the
owner paused for explicit Unload. The narrow `Worker.observe_packet` guard
preserves that current owner and reports typed `resourcePause`/`reloadRequired`
instead of its former broad pop/close/Load error branch. Ordinary private Stop
or pressure does not set this failed-restore flag.

Restoration first verifies the exact original owner object, loaded state and
persistent mutation tokens. If its actual canonical device/CPU-cold state is
already the recorded original, it is not needlessly evicted/readmitted. A real
change is restored; replacement/retirement fails closed. Control/compute
references are always restored. This removes redundant baseline work without
skipping learned verification or claiming a hardware speedup.

Stop retires policy authority immediately using only metadata/Event flags. It
does not set chat cancellation or terminate/restart the worker. Perception
hashing and existing compute/page boundaries observe private cancellation; no
instant kernel preemption is promised. A final delivery lock prevents a result
from being committed after an acknowledged private Stop. Ordinary fallback
runs once, after leaving all private/read-only scopes, and its genuine error
is not caught and replayed. If exact restore is unsafe, the live body is not
executed as a pretend-safe fallback. Serialized stopped cleanup releases the
derived leases/scopes; no Tensor or resource cleanup runs on the flags lane.

## Timing, evidence custody and bounded plan pages

The five stages, in order, are:

1. operation owner admission;
2. input activity transfer;
3. complete perception embedding, including connected resize/bridges;
4. output home transfer;
5. original selected-owner restoration.

Total monotonic observations include all declared work, its verification and
compute/home completion synchronization. Prior unrelated async work is drained
before timing; warm-up/backend-unknown observations are recorded as excluded.
Measured samples are initialized-backend selected-operation observations, not
cold Load/OS-cache or whole-brain/media latencies. Core mapped-read counters
remain **logical visits**, not physical disk traffic. Attention-file and actual
packed/activation transfer counters keep their separate basis. Family/allocator
resource peaks use fresh start/end and 50-ms sampling; they are sampled peaks,
not exact absolute peaks or an OS-enforced hard RSS cap.

Existing `engine/events.sqlite3` holds warm-up, measured, failed/cancelled,
selection and fresh-route-use receipts under `native-operation-auto`. No raw
source pixels/audio, embeddings, text or keys enter these timing rows. Per-row
size/disk/RAM admission and SHA-before-parse remain required. The iterator uses
the exact owner/job indexed sequence range; a live private issued head/count
must match, so imported self-checksummed JSON cannot authorize adoption. The
existing complete event-log portability retains history, **not** active plans,
controllers, captures, device leases or grants. A pressure failure may prevent
persisting its terminal diagnostic; that is not reported as a saved receipt.

Derived plan metadata has no fixed lifetime count. Its admission can evict older
derived entries under physical RAM pressure; no user source/history is removed.
If even one plan cannot fit, the policy stops qualifying until explicit new
review rather than repeatedly running an unstorable pool. Metadata pages default
to 32 and allow 1..128 records. Their <=512-character HMAC cursor binds the private
current session, snapshot revision, offset and exact filter. Changed, foreign,
tampered or out-of-snapshot cursors are refused. These are bounded transport
pages, not a cap on model regions, learned data or stored event history.

## Worker seam

Serial loaded-owner operations, none implicitly loading/creating a brain:

- `native_operation_auto_prepare({brainId,storagePath?,topology?,acknowledgeCudaCpuTileFallback?})`;
- `native_operation_auto_arm({brainId,storagePath?,operationId,reviewed,samplesPerCandidate?})`;
- internal `native_operation_auto_release_stopped({brainId,storagePath?,operationId?})`.

Flags-only controls:

- `native_operation_auto_status({brainId,storagePath?})`;
- `native_operation_auto_plans({brainId,storagePath?,after?,limit?})`;
- `native_operation_auto_stop({brainId,storagePath?,operationId?})`.

Main owns exact origin/config/storage identity, device topology and one-use
review authority, never the renderer. Native binding includes the live session,
current resource-policy/config hash, selected budgets, actual compute topology,
registered closure and home device. A queued exact Stop covers Arm's dispatch
race; unrelated old operation IDs do not retire a replacement. Prepare has its
own generation ticket, so a no-ID Stop during callbacks cannot publish a late
prepared review. Config replacement/close retires the policy; real closure,
geometry, learned-owner or topology changes invalidate subsequent consumption.
The separate manual/prefill/cached review paths cannot be armed over an active
operation policy. No imported preference automatically enables qualification.

## Additive complete retained-step consumer

`cached_whole_operation_auto.py` joins the SAME explicitly armed controller,
not another brain or UI. `cached_whole_operation_native.py` supplies the actual
loaded decoder callbacks. `OmniDecoder`'s real cached sampling boundary calls
it only when its existing retained cache is present and the current caller
permits measured cached execution. `ConnectedOmniDecoder` uses that inherited
boundary. The prior one-shot cached controller remains the unchanged fallback;
the two explicit review authorities cannot be armed over one another.

The two additional work kinds are:

- `native.decoder.cached-whole-step-sampling`;
- `native.connected-decoder.cached-whole-step-sampling`.

This scope is ONE complete real append frontier through the decoder and home
sampling, not an entire answer, working-memory migration, training, or a speed
claim. Initial prefill, shifted-window refill, resumed cursors and unknown or
unfenced owners retain their existing ordinary path. Preparing/arming does not
invent a prompt, start generation, synthesize a quiz, Load or train a model.

The closure includes embedding, causal global workspace and gains, memory
conditioning, every decoder block, dynamic experts/prototypes, final norm and
language readout. Connected traversal includes every actual borrowed parent,
its nested cache, graph recall projection and memory bridge, idea adapter,
liquid controller, LIF/CPU-backed sparse recurrence and workspace/cortical
exchange. Shared packed objects are deduplicated without flattening distinct
regional calls. Parent configuration, exchange widths, actual RoPE cache
geometry and computational inverse frequencies are bound. Unknown overrides,
hooks, cyclic decoder ownership, missing/foreign packed owners, unsafe live
transactions and incomplete retained frontiers cannot qualify.

Connected recall uses `ConnectedRecallBinding`, installed at the original
merge loader with the exact ROOT `config.working_memory_slots`. Ordinary calls
still invoke the original source with `record_activity=False`; its memory
learning callback is unchanged. Private trials require the declared native
source with current paged neuron, assembly/vector and synapse revisions.
Resident/opaque graph stores are honestly unavailable to this comparison, not
reported absent or broken. Private readouts share authoritative stores but
own their derived frontier/graph and resource-admitted immutable attention
membership. Stop reaches paged read sessions and existing scratch resource
boundaries; no instantaneous SQLite or kernel preemption is promised.

`cached_operation_snapshot.py` recursively forks the actual ordinary/connected
cache shape. Immutable K/V and workspace pages are refcount-shared, while
running sums, memory/neural state and every connected temporal triple are
independent admitted copies. Mutable cache aliases/cycles and incomplete
triples refuse. No decoder deepcopy, learned-weight copy, source-text replay,
whole-prefix refill or cache clearing is used to manufacture a safe trial.
Every candidate must advance each actual decoder cache exactly one frontier.

The current-home reference preserves its existing control and compute-device
references. It does not relocate a CPU RoPE cache first and call that a normal
reference. Alternate candidates place the complete declared closure, keep the
real CPU-backed recurrent owner, and count its actual transfer roundtrip.
Selected packed residency and exact control/callback/observer references return
before home sampling. Disposable pager caches may reclaim unpinned pages, but
the source cache and learned state are never replaced or relabeled current to
hide a write. A genuine failed reference/packed restoration pauses for explicit
Unload; there is no worker kill, reload or pretend-safe live fallback.

Warmup and measured trials compare the complete logical cache, connected/native
activity, selected token, exact float64 entropy/log-probability representation
and post-sampling home RNG state. A temporary home generator makes the same
draw from the captured original state without advancing the real generator.
There is no numeric tolerance. The original world callback is outside all
private trials and runs once at delivery. If it changes the home RNG, the
ordinary computation consumes the NEW RNG without replaying that callback.
Its real exception is not swallowed/replayed. Private Stop before delivery
retains the original chat once; after the real boundary linearizes, it retires
future policy work without undoing an already accepted ordinary action.

Live cortical viewport observations are private per trial. Only an accepted
trial's exact current viewport scope publishes its actually computed rows.
Warmups/rejected candidates cannot appear as user-visible firing; old live
rows are restored on fallback. A live independent recurrent transmission
observer instead keeps the ordinary route, preserving its genuine events
rather than replaying or silently dropping an uncloneable observer.

For unseen token frontiers, `cached_operation_layout.py` derives ranges from
the ACTUAL next-append page/coalescing geometry and exact q=1 attention
halving/tile-count transitions, intersected across real owners and parent
capacities. It does not select arbitrary 128/1k context buckets. The current
sampling law and exact UTF-8 allowed mask are bound separately from a changing
unused remaining-token allowance. Fresh values inside that same validated
layout range execute anew. Page/tile, owner, policy or computational-control
changes require fresh qualification. Per-frontier physical admission always
runs; a range is not a reservation for its largest context or a proof that
every future input was already measured or remains fastest.

Cached plan metadata explicitly reports
`numericGate:exact-cache-connected-activity-neural-sampling-equality`,
`reuseBasis:same-validated-operator-page-layout-range-current-owner-policy-inference`
and a `contextRange` with its real layout SHA. All plans continue to report
`unseenInputOrFrontierWasPreviouslyMeasured:false` and
`wholeBrainOrTrainingCovered:false`. They retain no old output token/cache or
source data. A rejected ordinary route is retained too, so unchanged rejected
pools are not rerun at every token. If even derived metadata cannot fit, the
policy stops qualifying until an explicit fresh review.

The eight cached coverage checkpoints are frontier COW/owner admission;
input/cue/cache transfer; complete causal workspace conditioning; all decoder
regions/recurrent exchange; experts/norm/readout; cache/neural output home
return; original-owner restoration; and same-home sampling. The three internal
forward-coverage checkpoints are reported after the complete forward returns;
they are NOT separately timed nested kernel phases. Total observations include
that whole operation, transfers, exact verification and restoration. They
remain scoped runtime receipts, not whole-answer/whole-brain benchmark proof.

## Additional retained-step source evidence

The new focused cases are distinct from the already closed one-shot/perception
checks and were run in small affected selections, not as a broad model suite:

- `test_cached_operation_layout.py`:4 recorded pure geometry cases.
- `test_cached_operation_snapshot.py`:4 recorded pure recursive ownership cases.
- `test_cached_whole_operation_auto.py`:10 inert-controller cases. The first8
  passed in38ms;2 new pressure/failed-restore cases passed within a15ms affected
  selection. Exact mismatch, fresh range reuse, real page/epoch transition,
  Stop, one-time world delivery and no unstorable-pool loop are covered.
- `test_cached_whole_operation_boundaries.py`:17 distinct constructor-free
  owner/raw CPU/source-callback cases. They cover actual closure validation,
  parent/control bindings, independent recursive state, cancellation, native
  restoration, actual-source write guards, ordinary/connected inert step
  sampling, private graph callback/membership custody and viewport adoption.
  The first batch exposed one fixture-only missing `brain.config`; that fixture
  was corrected and the one affected case passed in13ms. Later affected
  selections passed without a learned module/brain constructor or packed
  kernel. Repeated affected cases are not counted as new evidence.

The internal v2 plan field rename also updated/rechecked the one genuinely
affected prior perception plan-assertion case. That is not another new case.
Changed Python sources compile. No brain, Load, training, real token generation,
native packed kernel, actual accelerator/CPU speed comparison, provider or
device operation was run. The raw/inert tests establish only their stated
ownership/controller/source boundaries, not learned capability or performance.

## NEW source evidence and still-open work

20 distinct new cases:

- `test_scoped_owner_observation.py`: 3/3 in 35ms, constructor-free pager epochs,
  independent leases, replacement/pressure and no leaked ownership metadata.
- `test_operation_auto_policy.py`: initial 8/8 in 104ms, then one distinct
  no-metadata-room fallback case in 10ms. Two affected Stop/delivery cases were
  rerun in 2ms+6ms after adding the final delivery lock; this is not 11 distinct
  policy cases. Fake clocks/hosts are not hardware speed measurements.
- `test_operation_auto_boundaries.py`: 8 distinct actual-source seams and tiny
  raw CPU tensor cases. Three worker cases initially passed in the 155ms batch;
  three fixture-only weak-reference failures were corrected and only those
  affected cases passed in 14ms/1ms/8ms. The new real public-hook/nested-bypass
  case passed in 9ms. No learned module or brain constructor/native kernel ran.
  The later unchanged-residency/changed-restore/replaced-owner case passed in
  <1ms; only its genuinely affected existing stage/restore case was rerun in
  15ms. Both use fake pager/device metadata, not accelerator allocation.

System Python initially lacked `jsonschema` before these boundary test bodies;
repo `.venv/bin/python` supplied existing dependencies without installation.
Changed Python sources compile; scoped whitespace is clean. Closed
prefill/cached/Load/model/hardware/parser suites were not restarted for this
work. Main/API/UI/CI/package evidence is owned and recorded separately.

Surviving **missing implementation**, not relabeled mere evidence: generalized
retained-KV whole-turn scheduling beyond the complete-step source consumer above,
opaque/unfenced connected recall/current owners, fast-memory/STDP learning,
graph/working-memory and VRAM-backed context
placement, whole imagination/preview/codec/artifact paths, training/backward/
optimizer placement, concurrent cross-device schedules and direct CUDA/XPU/NPU
packed kernels. The approved full R1 contract remains intact. Actual supported
perception multi-device behavior/speed, broad neural capability and platform
performance remain unverified under the present no-model scope.

## Prepared complete-media source consumer (October 17, in composition)

The actual `AdaptiveBrain.generate_modality` path now prepares its real output
geometry/cadence and optional companion-audio plan before entering one reusable
producer. It is not a separate model or a stand-alone timing callback. Ordinary
generation still publishes its genuine previews and artifact once. The public
event/performance/learning handoff remains outside the private producer.
Standalone native-unit timings choose output size only; they never authorize a
device route or count as complete-media placement evidence.

`media_whole_operation_auto.py` retains only the admitted current cue and prepared
operation after its genuine result. The existing serial stdio owner may then
visit one complete private warm-up/sample at a time, prioritizing new normal
requests. The reader retires that work using flags/integers only; tensors, files
and leases are released on the serial owner. Explicit policy Stop does not
cancel/replay/erase a delivered user generation. An optional proof-budget or
post-result fence refusal does not turn a completed real artifact into an error.

`media_whole_operation_native.py` closes over registered packed generators,
codec tables, native recurrent media computation and, for connected brains,
every borrowed parent pack plus actual selector/workspace exchange. Learned
owners stay read-only and shared. Temporary compute targets, small control
references, eval flags and viewer observations are restored exactly; no global
brain/device/context switch, learned-epoch reset, worker replacement or copied
model is used. Native rows remain ternary/integer-strength learned owners; the
temporary cue/output/codec/RNG activity is not another learned model.

Random draws and codec-token sampling occur on the original home device under
the same seeded generator law. The exact ordered generator/draw/post-state,
native float activity, complete preview sequence, media timeline and encoded
artifact checksum form one semantic contract. Comparisons use a genuine
first-result reference, followed by private same-geometry artifact sinks; they
never replay the original callback, user event or learning receipt. Future
compatible cues/seeds are computed anew, explicitly as operator/layout inference
rather than previously measured or cached content.

Private media uses a correlated already-pinned codec scope: it may reverify that
binary but cannot discover/install a codec, prompt the user, change selection or
fall back to an unowned executable. Missing/unpinned/nondeterministic codecs and
unsupported owners retain normal functioning generation. Exact equality and
matched admitted whole-operation samples remain required for a new route; a
warm-up or an inconclusive/failed pool is not mislabeled three qualified samples.

### Actual streaming output, not only planning

Large admitted image/audio/video plans now use file-backed overlap numerators
and blend weights via the existing selected-drive/shared-quota protocol.
Bounded native tiles/chunks/windows still call the same generator. Spatial
ordering uses an owned SQLite index rather than an unbounded coordinate list;
the first center tile can be produced before indexing the rest. All planned
units are traversed. Small native output retains its in-memory tensor path.

PNG/APNG frames, WAV/RF64 samples, MP4 raw RGB/companion audio and final artifact
copies can stream in bounded windows. MP4 is streamed for small output too so
Stop/resource pressure need not await a whole-movie `communicate` or a fixed
120-second timeout. Diagnostics use a bounded pipe tail, not an unlimited disk
log. Checksums and opened-file identities reject replaced payloads; cleanup
unlinks only its own exact activity inode. Private preview/final files have
independent spill escrow/identities and are removed before releasing their
leases. Format integer bounds are reported as container constraints, not hidden
model-size/resolution caps or a resolution downgrade.

### Recorded scoped evidence and remaining boundary

The new evidence consists of9 canvas/planning/RNG cases already recorded,9
inert policy cases,9 actual-body/metadata/native-effect cases,2 stdio/worker
handoff cases and4 fake-Popen codec-protocol cases:33 distinct new cases, run in
affected selections. Fixtures use tiny raw CPU tensors, real temporary files or
SQLite, fake resource leases and inert native decoders. Fixture-only missing
globals/cancellation-class/old-Python unittest helpers were fixed; only affected
cases were repeated. Actual first/private producer contracts match for the
image/audio/video fixture, including once-only public events and artifact
isolation. Replacement, Stop, pressure, fresh-cue/seed inference, current epochs,
borrowed connected owners and private observation/control restoration have
source-boundary evidence. Fake codec output is not playback or codec quality
proof. Changed sources compile; there was no brain/learned-module construction,
Load, training, native packed kernel, actual FFmpeg execution, accelerator test
or neural-quality/performance qualification.

Root approved descriptorA. Native registry now declares the3 media work kinds
and6 exact stages alongside the unchanged original5 kinds; its one new registry
contract case passed without rerunning the33 closed cases (34 distinct media
cases total). Only the obsolete blanket whole-imagination unsupported marker
was replaced by the opaque/unpinned/nondeterministic-codec fallback marker.
The matching main/shared/renderer contract composition is recorded below.
The emitted `mediaLayout` is exactly `{modality,shapeSha256}`; the complete frozen
primary/companion geometry, settings, codec and preview cadence remain bound in
the internal operation/geometry hashes. This source consumer is not full
R1: training/backward/optimizer private overlays, fast-memory/STDP placement,
general whole-turn/context migration, concurrent mixed-device schedules and
direct CUDA/XPU/NPU kernels remain separate unfinished implementation.

### Main-owned media contracts and labels

`nativeOperationAuto.ts` and `nativeOperationAutoService.ts` now compose these
exact native declarations into the existing Prepare/Arm/Status/plan-page path:

- `native.media.image-artifact`, with `mediaLayout.modality: "image"`;
- `native.media.audio-artifact`, with `mediaLayout.modality: "audio"`;
- `native.media.video-artifact`, with `mediaLayout.modality: "video"`.

All three require `mediaLayout: {modality,shapeSha256}`, a lowercase 64-character
SHA-256 identity, the numeric gate
`exact-media-neural-timeline-rng-and-encoded-artifact-equality`, and the reuse
basis `same-validated-media-operator-output-layout-current-owner-policy-inference`.
The six ordered stages are exactly:

1. `prepared-media-owner-admission`;
2. `home-RNG-and-activity-transfer`;
3. `all-native-media-regions`;
4. `overlap-composition-and-preview`;
5. `deterministic-codec-artifact-checksum`;
6. `original-owner-restore`.

Missing or additional layout fields, cross-modality layouts, extra paths,
weaker gates and invented stages are rejected rather than passed through.
`contextRange` is forbidden on media plans; `mediaLayout` remains forbidden on
the unchanged five perception/cached-step contracts. Reviews may list only the
native owner's supported subset, but every declared kind needs its complete
exact stage coverage. These plans still report one qualification workload,
unseen-input reuse as inference, and no whole-brain or training measurement.

The existing renderer gives the new routes readable image/audio/video artifact
labels and no longer incorrectly excludes all media generation from this
control. This is a narrow addition, not a UI replacement or automatic Load.

Nine new inert host/presentation cases passed: all three modality layouts,
malformed/foreign/missing layout refusal, weakened gate/scope refusal,
cross-contract field refusal, supported-subset stage review, owner-bound
status/pagination validation, and the three labels. The literal test filter
also selected one previously passed Stop case because “immediately” contains
“media”; the tool reported ten passed and nine skipped, not ten new cases.
No native media suite, model, Load, training, FFmpeg, actual hardware placement
or media-quality/performance test was run for this contract composition.

## Actual prepared native learner-wave consumer (October 17)

The app's production dataset path is `BrainService.ingestManifest` → worker
`ingest` → `AdaptiveBrain._optimize_streaming_experience_batch`. The separate
distributed invocation builder has no production app caller and is not counted
as app integration. Worker ingestion now captures the existing explicitly armed
loaded-owner Auto review as source-free per-attempt job provenance. Unarmed jobs,
unreviewed wave kinds, absent job identities, stale owners/policies and cross-rank
worlds remain ordinary. The saved JSON records exact owner/session/operation,
review, selected device/policy/topology/budget and source-manifest hashes; reading
it after restart cannot restore authority. No new basic mode/checkbox was added.

The first unfrozen learner schedule remains ordinary. A later requested local
wave is eligible only at its actual locked serial boundary. Its real token
windows and neural cues are prepared once, with RAM escrow and incremental,
checksummed SQLite spill rather than repeated source traversal. Every private
candidate sees identical row geometry, physical batch, sequence size and logical
accumulation. No standalone projection/unit timing authorizes a route.

`TrainingNamespace` binds independent dictionaries on the SAME existing module
objects. Touched packed native fields copy on first use into private, resource-
admitted byte maps; controls preserve storage/view aliases. Unexecuted modality
activity and fixed sparse recurrent controls remain read-only instead of copying
an unrelated model. New native contact pages belong to private registries/pagers.
Declared connected graph caches/readout activity are private, SQLite snapshots
open originals read-only and reject writes during this text-wave read closure,
and original router operation fences keep its representation stable throughout
the pool. This is not a live rollback of originals labeled as private training.

Private parameter observers reject an original-storage write before the user's
delta journal can see it. Effects/cursor scratch uses a separate SQLite namespace
with spill escrow. Original event callbacks, source admission, STDP, dataset
cursors and checkpoints are not replayed. Process RNG is not reset by trials:
independent generators start at the real home state; scoped native stochastic
updates draw there and transfer actual activity. Checkpointed declared blocks
do not fork/reset process RNG in this scoped path. Ordinary unscoped calls retain
their existing RNG behavior. Explicit already-advanced home generators preserve
their actual state; a foreign generator is not equated by its seed.

Complete private samples include owner admission/COW, all actual region forward
and backward computation, native mutation, transfer, complete state/effects/RNG
proof and cleanup. Exact packed/control state, graph contents/hot overlays,
metrics and home-RNG results must agree. Physical resource observations and
runtime-issued whole-path receipt custody remain required. Warm-up alone is not
qualification. After current owner/policy/input fences pass again, the selected
candidate target runs ONE genuine update through the existing learner. Its real
effects are forwarded once and its RNG state published once. The ordinary outer
maintenance, checkpoint and cursor publication follow once. An error after that
genuine update begins never falls through to a second update. Stop Auto during a
private sample retains one ordinary fallback; Stop Auto after genuine start does
not cancel or repeat that canonical update. Private cleanup trouble cannot trap
the controller in a running state or undo an already verified real update.

The ninth declared kind is `native.training.prepared-whole-wave`, with exactly:

1. `prepared-wave-owner-and-policy-admission`;
2. `private-native-and-graph-COW`;
3. `same-home-RNG-input-and-activity-transfer`;
4. `all-region-forward-backward-and-native-update`;
5. `exact-complete-wave-state-and-effects-proof`;
6. `original-owner-and-RNG-fence`.

Its plan carries only `trainingLayout: {shapeSha256}`, the numeric gate
`exact-prepared-wave-native-state-rng-and-effects-equality`, and reuse basis
`same-prepared-wave-current-owner-policy-no-cross-wave-inference`. The hash binds
actual whole-wave geometry, its frozen schedule and declared native closure.
There is intentionally NO cross-wave route reuse yet: a changed learned state
needs a new real-wave qualification. The original `wholeBrainOrTrainingCovered:
false` scope means no whole training PIPELINE claim, not that this wave is a
primitive timing. Main/shared/renderer composition is owned/recorded separately.

### Evidence and surviving R1 boundary

26 distinct new scoped cases pass across affected selections: unload-safe owner
references, raw uint8 COW and
aliasing, source observer rejection, real SQLite copy isolation/read restrictions,
bounded prepared input/coverage, current job provenance, cross-rank/stale-owner
refusal, actual learner/worker source seams, inert selected-target/once-only
delivery, Stop/failed-genuine/cleanup behavior, sparse-router ownership fences,
home RNG/advanced-generator state, typed transfer endpoints and the ninth-kind
registry. Fixtures are raw CPU tensors, files/SQLite, fake owners and inert
callbacks; their fixed losses/receipts are NOT trained answers or hardware proof.
The earlier closed cached35 and media34 cases were not rerun. No model/brain
construction, Load, neural forward/backward/training, real GPU or benchmark ran.

No total speedup is claimed: qualification, copying and exact proof can outweigh
the eventual selected operation. This is not complete R1. Full training aggregate
placement, lasting graph/STDP updates, multimodal training objectives, distributed
all-rank protocol/custody, reusable changed-state proofs, general context migration,
concurrent mixed-device schedules and direct CUDA/XPU/NPU native kernels remain
separate work. Unsupported closures keep normal learning; they are not disabled
or relabeled as optimized whole-brain execution.

### Main/shared/renderer prepared-wave composition (October 18)

The existing `NativeOperationAutoService` now accepts the ninth native kind
through the same loaded-owner Prepare/Arm/Status/plan-page control. The shared
plan union requires exactly `trainingLayout: {shapeSha256}` with the exact
native prepared-wave numeric gate and no-cross-wave reuse basis above. Main
decoding rejects missing, additional or malformed layout fields, media layouts
and cached context ranges on training plans. A training layout is also forbidden
on each of the unchanged eight prior discriminators. The six-stage native map
is matched exactly, not shortened or borrowed from perception/media.

The existing renderer labels it **Prepared training wave**, marks its plan as
“this wave only,” and says a future wave needs a new check. Measurement scope
distinguishes individually prepared waves from full training pipelines,
distributed training, whole-brain and complete-reply placement. It does not
claim cross-wave reuse or total speedup. This adds no mode, automatic workload,
new permission, implicit Load, or new app layout.

Twelve NEW inert host/presentation cases in
`tests/nativeTrainingWaveAutoContract.test.tsx` passed in14ms: exact plan/deep-copy,
malformed fields, foreign layout/authority fields, weaker gate/overclaim refusal,
training-layout exclusion on prior kinds, metadata-only supported-subset
Prepare/Arm, complete stage coverage, owner-bound status/pagination, status scope
refusal, native declaration text parity, wave-only presentation and unchanged
collapsed control/no-host rendering. Native declaration parity reads Python
source text only; it does not execute Python. No native26/media/cached case,
worker/model/Load/training, real hardware placement, browser flow or performance
check ran. Overall TypeScript/UI compilation and goal closure remain root-owned.

### Actual full-prefix consumer: native source closure (October 18)

`prefill_whole_operation_auto.py` and `prefill_whole_operation_native.py` now join
the same explicit loaded-owner controller at the existing
`OmniDecoder._generate_scoped_iter` sampling boundary. This handles a genuine
first prefix or a naturally required refill under a changed cue, shifted window,
or stale old cache. It does not clear context, manufacture an input, restart a
worker, construct/copy a brain, or replace the previously closed retained-step
consumer. A still-valid retained step takes the unchanged cached path. Explicit
`use_cache=False` remains the separate exact full-forward numerical reference;
it is not claimed as covered by this new cache-producing operation.

Each trial invokes the actual ordinary/connected `_generation_step_bounded`
with `cache=None`. Its entire prefix traverses the actual streamed query blocks,
workspace summaries, every connected region/nested recurrence, attention pages,
experts, normalization and language readout. Every trial owns only its new cache
pages/activity. Learned packed/control owners, current typed read-only connected
recall, graph ownership fences, private viewport observations and original-device
restoration use the existing declared complete text closure. The caller's old
cache is left unchanged until genuine adoption, or ordinary fallback closes it
normally. It is never relabelled valid to authorize a refill.

Whole-prefix receipts measure the real full operation, including input/cue and
activity transfers, new-cache home return, restoration and original-home token
sampling. Qualification compares complete recursive cache contents, connected
activity, neural output, token/log probability/entropy and advanced home RNG.
The selected private result is delivered through the actual `record_forward`
boundary once; normal observation binding, cancellation, subsequent action
selection, token emission and genuine learning are not replayed. No faster
candidate means the exact measured home result is adopted, not an additional
synthetic/repeated ordinary prefix. Stop or invalidation before publication
retains ordinary execution once. Stop after publication cannot undo or repeat
the actual boundary. A changed home RNG at that boundary invokes the existing
no-second-callback fallback. Cleanup failures clear running flags and never
replay already performed real work.

Declared kinds are `native.decoder.full-prefix-sampling` and
`native.connected-decoder.full-prefix-sampling`, with eight full-prefix stages.
`prefillLayout` contains exactly `{shapeSha256}`; its hash binds complete current
prefix/query-block/page/tile/nested geometry internally. The numeric gate is
`exact-prefix-cache-connected-activity-neural-sampling-equality`, and reuse basis
is `same-prepared-prefix-current-owner-policy-no-cross-prefix-inference`.
No other prefix, changed RNG/state or same-shaped future input reuses its proof.
At native source freeze both kinds were deliberately absent from
`STAGE_COVERAGE` pending strict main/shared/renderer composition, completed in
the following section. Binding the native callback alone cannot enlarge an
existing nine-kind review or grant implicit authority.

Nineteen NEW scoped cases passed across affected selections: thirteen inert
policy/source cases and six raw CPU tensor/fake-owner boundary cases. They cover
old-review gating, actual caller binding, full-prefix/no-cross-prefix behavior,
ordinary/connected metadata, new-cache custody, six exact-result mismatches,
Stop/owner/policy changes, once-only world/RNG boundaries, current geometry,
manual-review exclusion, release failures and no rejected-cache relabelling.
The execution fixture is an explicitly inert callback, not a learned forward.
No model/decoder/brain constructor, native forward/backward, Load, training,
real hardware route, benchmark, provider or closed cached/media/training case
was run. Python compilation passed for the affected files; no total speedup is
claimed, since complete qualification may cost more than the selected prefix.

The finite R1 source gaps remain explicit rather than hidden under test status:

1. Complete prefix coverage: the cache-producing first/refill consumer is
   source-composed with strict host contracts below. The additional actual
   uncached full-forward/action-logit consumer is recorded below; its public
   host composition is a separate gate.
2. Whole reply: the initial full routing forward has an additive native
   consumer below. Later action/Ponder orchestration, real provider/world
   callbacks and complete reply orchestration remain outside a privately
   qualified whole-reply consumer.
3. Working/lasting memory: the actual mutable closure now stays on its real
   owner and participates once through the request-scoped aggregate scheduler
   documented below. General private relocation of this mutable closure is
   not implemented; its ordinary learning and state movement are retained.
4. Outer training: direct `train`, media objectives, dialogue/replay/evolution
   learners and aggregate ingestion/checkpoint/cursor publication are not the
   prepared text-wave consumer. Cross-rank authority still requires real
   all-rank protocol/custody, not a local receipt or primitive timing.
5. Mixed placement: candidates still name one execution device per selected
   operation. Automatic multi-region/concurrent schedules, actual cross-device
   transfer paths, GPU context/cache migration and unsupported accelerator
   kernels require additional real consumers. Physical memory groups/budgets,
   current context, slow-storage opt-in and ordinary fallbacks must remain
   enforced; nine or eleven scoped kinds do not constitute complete R1.

### Full-prefix native/main/shared/renderer composition (October 18)

The two frozen full-prefix kinds are now added to native `STAGE_COVERAGE` and
the strict shared work-kind union, making eleven individually declared kinds.
The actual loaded owner still advertises only its supported ordinary or
connected closure. An existing review cannot acquire either new authority:
Prepare/Arm must contain the current native registry, owner, policy, topology,
budgets and exact eight-stage declaration. No new permission, mode, checkbox,
implicit Load, model workload or alternate UI was added.

Main decoding accepts `prefillLayout` only on the two full-prefix plans, requires
exactly its lowercase SHA-256 field and agreement with the native plan's
`geometrySha256`, and enforces the exact prefix numeric gate and no-cross-prefix
reuse basis. Cached `contextRange`, `mediaLayout`, `trainingLayout`, source paths
and authority fields are forbidden on prefix plans. Conversely, every prior
nine-kind plan forbids `prefillLayout`. Global placement and whole-brain/training
measurement scope remain false. Existing loaded-owner control and metadata page
custody are unchanged.

The existing collapsed control labels the kinds **Full decoder prefix** and
**Full connected prefix**. Active status says another prefix needs a new check;
qualified routes say **this prefix only**. Measurement details continue to
exclude uncached full-forward/reply orchestration, fast plasticity, complete
training pipelines, mixed-device policy and any total-speedup claim.

Thirteen NEW inert host/label/source-parity cases in
`tests/nativePrefillOperationAutoContract.test.tsx` passed in20ms (504ms total):
both exact plan discriminators/deep-copy, malformed or foreign layouts/hash
mismatch, weaker gates/reuse/measurement claims, exclusion on all prior kinds,
metadata-only Prepare/Arm, exact stage refusal, owner-bound status and pagination,
native declaration/gate/reuse source parity, prefix-only labels and unchanged
collapsed rendering without a host call. Python is read as source, not executed.
The earlier nineteen native prefix cases and closed cached/media/training cases
were not rerun. Only now-obsolete registry cardinality/old-review gating
assertions were updated for the approved composition. No model constructor,
native forward/backward, Load, training, hardware route or benchmark ran.
Overall TypeScript compilation/UI and complete R1 closure remain root-owned.

### Actual uncached full-forward consumer: native source closure (October 18)

`forward_whole_operation_auto.py` and `forward_whole_operation_native.py` bind
the same loaded-owner controller to the actual public `OmniDecoder.forward`.
The connected decoder inherits that boundary. This covers its real eval,
no-grad, `labels=None` computation, including `brain.chat`'s initial global
routing forward and the explicit `use_cache=False` generation reference.
The complete existing output-window admission remains inside the invoked
ordinary entry. A larger prefix is neither truncated nor substituted with
last-token-only output to force admission. Training, gradients, labels, manual
leases, unknown hooks/overrides and opaque transmission observers retain their
ordinary paths.

The text closure adds the actual packed `ActionPolicyHead` projections and
normalization, exact borrowed connected wrappers and their regional identities,
and the real `NativeRegion.liquid_state` initial recurrent buffer. Borrowed
views cannot silently point at different heads, widths, exchanges or sequences.
All ordinary/connected workspace, regional recurrent/recall/exchange, experts,
normalization, language and action readout run through the actual forward.

Each trial owns only its returned activity and temporary placement. It returns
the complete actual output dictionary: logits, hidden state, action logits,
requested whole workspace and any real expert routing. Storage views and
aliases between connected activity and returned hidden tensors are retained.
Private viewport observations and actual connected `last_region_activity`
publish once after current input/owner/policy fences. Full forward does not
set `last_generation_neural_state`, create a KV cursor, draw a sampling token,
or manufacture those effects for its proof. The surrounding real caller keeps
its own generation/sampling and learning behavior.

The declared eval closure has no RNG draw. Original CPU and participating
backend process RNG frontiers are captured and compared before/after each
trial; there is no process reset, reseed or seed-equivalence assertion.
Changed or unprovable RNG/native controls require explicit Unload rather than
retrying a real operation against altered state. Exact mismatch, ordinary
pressure or Stop with intact original frontiers leaves one ordinary fallback.
Publication, real fallback failure and private release failures cannot cause
the public forward to be replayed.

The two native declarations are:

- `native.decoder.uncached-full-forward`;
- `native.connected-decoder.uncached-full-forward`.

Their seven ordered coverage checkpoints are:

1. `forward-owner-and-policy-admission`;
2. `input-cue-mask-activity-transfer`;
3. `complete-causal-and-requested-whole-workspace`;
4. `all-decoder-connected-recurrent-exchange`;
5. `experts-norm-language-action-readout`;
6. `all-outputs-and-activity-home-return`;
7. `original-owner-restore`.

The middle three are completed coverage checkpoints after actual full forward,
not separately timed inner kernels. Matched observations include its complete
input/cue/mask transfer, output/activity home return, proof and restoration.
The plan requires exactly `forwardLayout: {shapeSha256}`, numeric gate
`exact-full-forward-output-workspace-action-activity-equality` and reuse basis
`same-prepared-full-forward-current-owner-policy-no-cross-input-inference`.
The known absence of retained KV is explicitly hashed in the logical-cache
field; it is not an unknown cache asserted eligible. A plan is an inspection
receipt for this actual prepared input only. A new call, even with the same
shape, needs its own qualification; prior output or route proof is not reused.

At the native source freeze the kinds remained absent from `STAGE_COVERAGE`
pending the strict shared/main/renderer composition recorded next. Binding the
new callback cannot enlarge an older eleven-kind review. No new UI, permission, workload,
Load, brain, manual route or default is introduced by this binding.

Twenty distinct new cases in `test_forward_whole_operation_auto.py` and
`test_forward_whole_operation_boundaries.py` pass across affected selections.
The initial17-case run exposed fixture-only missing connected configuration
fields and a synthetic-timing proposal expectation; only the affected five
cases were rerun (46ms). Two new capture/unprovable-owner cases and their
affected RNG case passed17ms; one new explicit selector-stub case plus affected
AST/capture cases passed362ms. Inert receipt origin correctly cannot authorize
a real hardware comparison; the selector stub proves only that the controller
takes a supplied proposal's exact chosen output once. Cases also cover actual
source placement, ordinary/connected typed closure, action ownership, liquid
initial state, mask/cue/current geometry, real output/activity aliases, every
output mismatch, old-review refusal, Stop, pressure, no-draw frontiers, cleanup
and real-publication/fallback exceptions.

Fixtures are metadata-only owners created without learned constructors, tiny
raw CPU tensors and explicitly inert forward callbacks. No decoder/model/brain
constructor, neural forward/backward, Load, training, native packed kernel,
hardware route/benchmark, provider or closed cached/media/training/prefix case
ran. Changed Python source compiles. Complete reply orchestration, fast
plasticity, working/lasting-memory placement, complete training pipelines,
multi-device schedules and direct unsupported kernels remain explicit R1 work.

### Uncached full-forward native/main/shared/control composition (October18)

The exact two uncached full-forward declarations and seven stages now compose
through shared types, strict main validation and native `STAGE_COVERAGE`/
`WORK_KINDS`, for thirteen supported individual operation kinds. A current owner
still advertises only its actually valid closure. The complete output/input
geometry SHA must equal the sole `forwardLayout.shapeSha256` field. Main rejects
additional layout fields, other operation layouts, source paths, injected
authority, weaker gates, cross-input reuse and whole-reply claims. Each prior
eleven-kind discriminator explicitly forbids `forwardLayout`.

Prepare requires every advertised kind's full ordered stage declaration and
matching status/owner. Arm cannot enlarge or replace the review's supported
kinds. Status and plan pages reject an operation outside their current
advertised reviewed kinds, while existing one-use authority, owner/config/
session fences and metadata pagination remain in place. This adds no model
operation to rendering or to reviewing/arming metadata.

The existing collapsed control labels the operations **Full decoder forward**
and **Full connected forward**. It describes a complete forward check and
marks its receipt **this input only**; another input needs a new check. The
measurement text continues to exclude complete replies, whole-brain placement,
full/distributed training and total-speedup claims. Its hooks, layout, controls
and resource review remain the existing implementation.

Eleven NEW inert host/presentation/source-parity cases in
`tests/nativeForwardOperationAutoContract.test.tsx` passed17ms (433ms total).
They cover both exact/deep-copied plan variants, geometry and foreign-field
refusal, gate/scope refusal, exclusion on every prior valid plan, metadata-only
Prepare/Arm, all seven ordered stages, old-review expansion refusal, owner-bound
status/pages, thirteen-kind native source parity and the existing closed-panel
render/labels without a host call. The React review leaves the current owner
and request lifecycle intact; this slice only adds labels/scope expressions.
Three older test fixtures received only their now-obsolete cardinality/filter
updates; their closed cases and the20 native forward cases were not rerun.
Python registry/policy compilation and affected whitespace checks pass. No
Python/neural execution, constructor, worker, Load, training, device benchmark,
browser interaction or hardware proof ran. Overall TypeScript compilation,
UI acceptance, full R1 and release closure remain root-owned.

### Actual fixed-memory aggregate scheduler (October18)

`whole_path_scheduler.py` and `whole_path_native.py` now join the real
`Worker.dispatch` request and annotated memory/STDP bodies to the existing
thirteen complete-operation selectors. Mutable memory/graph/SQLite/replay/
workspace/STDP stays on its actual owner and executes once. Nested phases do
not inflate the fixed duration. The current request's observed fixed prefix
plus authenticated whole-compute samples drives conditional route selection;
this is not a fourteenth work kind or a whole-reply benchmark. Current owner,
policy, context, physical budgets and slow-storage consent remain rechecked.

The native resource sampler now includes every actual home/target participant,
including a distinct home GPU that retains memory and receives outputs.
Physical RAM/Metal groups remain counted once; unused GPUs are excluded.
Unproven overlapping non-RAM allocator views cannot authorize a route.
Twelve new raw/inert/real-SQLite/source cases have scoped evidence; no closed
operation, model, training or hardware suite was run. See
[whole-path-memory-scheduler.md](whole-path-memory-scheduler.md) for exact
custody, verification and the remaining implementation/evidence distinction.

The actual reply-level consumer is now wired across `AdaptiveBrain.chat` and
`OmniDecoder._generate_scoped_iter`, including neural action/Ponder/tool-route/
argument decisions and pinned world callbacks. It dispatches the semantic-
compatible existing complete consumer at each current frontier, consumes its
actual measured selection and admits/accounts for non-migratable owner phases.
Exact Worker turn-local Event custody keeps private cancellation out of opaque
world callbacks; all real callbacks/publications remain once-only. Sixteen new
stdlib/inert/SQLite/AST cases have affected source evidence, not model or
hardware proof. See [reply-execution-planner.md](reply-execution-planner.md).
Subsequent source composition adds mixed sequential decoder regions for six
text kinds only and actual outer modality/replay/evolution owner-phase
orchestration with once-only canonical update/publication. See
[mixed regions](mixed-region-complete-operations.md) and
[outer learning](outer-learning-pipeline.md). This keeps thirteen work kinds;
other consumers do not acquire mixed support. Concurrent/cross-rank execution,
retained device-cache/context migration and unsupported direct accelerators
remain outside that source coverage. Global whole-reply speed, private replay
of mutable pipelines/world I/O, hardware performance and learned quality are
not claimed.
