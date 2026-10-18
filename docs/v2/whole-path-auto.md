# Measured decoder Auto: exact isolated-prefill source slice

This is an incremental R1 implementation, **not R1 completion**. No brain,
Load, training, generation, hardware benchmark or neural calibration was run
for this source slice. The user-deferred merged/native test remains deferred.

## Callable production path

`src/shared/decoderExecution.ts` defines the public host DTOs.
`engine/omni_core/decoder_path_execution.py` owns process-local orchestration;
`decoder_execution_native.py` supplies the actual already-loaded native owner
callbacks. `AdaptiveBrain`, `Worker` and `worker_transport` provide the narrow
loaded-owner serial/control seams. The model has real capture and consumption
hooks, not only a recorder exported for a future caller.

The current work kind is exactly
`native.decoder.prefill-next-logits-isolated-cache`:

1. Explicitly arm capture, then retain the next **actual ordinary user-turn**
   prefill input/cue. No synthetic prompt, BOS quiz, source-text injection or
   new model is generated. Admission/capture failure preserves the normal
   body exactly once; it does not cancel or retry that chat.
2. At a genuinely quiescent loaded-owner boundary, admit an exact CPU copy of
   that input/cue and bind live packed-owner epoch, non-pager controls, real
   registry/geometry, current policy, selected budgets and compute topology.
3. Verify against the existing resource-admitted full-forward next logits and
   normalized last neural state for that exact input. If that full-output
   reference cannot fit, refuse calibration; do not truncate or expand the
   original forward's resource limits.
4. Run one recorded/excluded warm-up per candidate, then at least three real
   samples per candidate, round-robin. Every sample uses isolated empty
   workspace/attention caches and the same live learned/control state.
5. Stream checked observations from the owner's existing EventLog and compare
   exact same-work receipts. Propose only an admitted supported candidate whose
   observed maximum latency is below the current candidate's observed minimum.
   This finite-sample rule is not statistical confidence or a quality claim.
6. Apply consumes one live proposal authority and installs **only an exact
   isolated-operation lease**. It changes no saved setting/global backend and
   does not arm portable/manual regional preferences.

Actual consumption is at `OmniDecoder._isolated_generation_logits`, the existing
`use_cache=False` generation boundary. Only identical input/cue, owner state,
geometry, topology and budget can use the checked prefill. Everything else
retains the original full-forward path. Default retained-KV chat, connected
regions, media and whole-brain continuation **do not adopt this result**.
Standard host chat currently uses retained caches; this slice therefore does
not claim that ordinary cached chat has become fastest-device Auto. Broader
reachable cached/all-region consumers and their measurements remain work.

## Exact scope and timing

Declared stages, in order:

- exact placement admission;
- input/cue transfer;
- native prefix-causal workspace/decoder/expert/norm/language-head prefill;
- exact home-output transfer;
- private cache release;
- original home placement restoration.

Total monotonic nanoseconds include all declared native stages, verification
work inside the observation and completion synchronization. They exclude
preceding unrelated asynchronous work (drained before timing), host/network
and user wait, the separate full-forward reference, and excluded warm-ups.
Every used async compute **and home-output transfer** device is included.
A CPU candidate returning to a Metal home therefore includes Metal completion;
it does not pretend to perform CPU inference on Metal. CPU/Metal numerical
differences are checked using the explicit existing float32 bounds
`rtol=2e-5, atol=1e-4`; tolerances are not loosened after a failure.

The output contract hashes reference bytes, exact shape/dtype/finite checks and
tolerances. It is not a false raw-logit-byte equality claim. Observations retain
actual output hashes, maximum logit/state/probability differences and argmax
agreement. No sampling/noise/generator draw is performed by calibration. The
normal caller's sampling algorithm and home device remain unchanged; identical
sampled tokens across different floating-point devices are not claimed.

Cold/backend-unknown warm-ups are recorded as such and ineligible. Measured
samples are initialized-backend, empty-private-activity operations; no cold
startup, retained-KV continuation, physical OS cache state, full reply, full
brain or multimodal latency is inferred from them.

## Learned custody and resource behavior

- The actual packed signs, strength/plasticity/stability and repeat metadata
  stay in their current authoritative owners. Native read-only parameter guards
  reject learned writes/growth/retirement before they occur. No alternate
  decoder, floating learned master, model deepcopy or shadow weights are made.
- A process-local persistent-buffer epoch includes real writes and unexpected
  replacements, but excludes disposable sort indices and known exact pager
  moves. Non-pager controls are checked with bounded copies; registry metadata
  is streamed per real module. This is a supported live-owner mutation fence,
  not a cryptographic claim about arbitrary external process-memory tampering.
- Live epoch metadata is opt-in and resource-admitted. Default forwards do not
  build those extra version dictionaries. Stop/retirement releases observation
  metadata. Logical counters are monotonic; mapped owner visits/dirty flushes
  are **not physical disk I/O**. Actual attention-file reads/writes and packed/
  activation transfers retain their separate counter basis.
- Fresh shared-family/device admission is retained before allocation/dispatch.
  Unified RAM/Metal capacity is not summed. Operation resources use fresh
  start/end and 50-ms background samples; these are sampled family/allocator
  peaks, **not exact absolute physical peaks** or an OS hard limit.
- CPU/Metal direct routes remain distinct from explicitly acknowledged CUDA
  packed-CPU-tile fallback. CUDA is not relabeled native. XPU/NPU remain
  unsupported here. Unknown/disabled capacity and a live manual regional lease
  are refused; no source preference is silently remapped.
- Live cache/spill handles, pins, training transactions, connected mutation
  holds and another active inline job are refused, never cleared. Training
  flags, original registered controls/state references, placement metadata and
  activity budgets are restored on success, failure and Stop. A genuine failed
  packed restore is left explicitly paused for Unload, not automatically
  restarted or presented as safe continuation.

Receipts use existing `engine/events.sqlite3` (full portable custody already
exists), with no raw input/cue/text/API keys. Each row is size-admitted and
SHA-checked before parse. The source iterator uses the actual operation's
indexed sequence range, not a full history scan or ever-growing receipt Map.
The checked stream head/count must match the live private issued evidence
head/count; recomputing a public JSON checksum cannot authorize imported data.
Warm-up, failed/cancelled observations, numerical checks and selection are
retained. A disk/RAM refusal may prevent writing a failure diagnostic; it is
not relabeled as a persisted receipt. Copy/import/recovery cannot restore an
active input, reviewer token, proposal, controller or placement lease.

## Worker/host seam

Serial loaded-owner methods (none implicitly loads/creates a brain):

- `decoder_execution_capture_next({brainId, acknowledgeCudaCpuTileFallback?})`
- `decoder_execution_prepare({brainId, topology?})`
- `decoder_execution_calibrate({brainId, operationId, reviewed, samplesPerCandidate?})`
- `decoder_execution_apply({brainId, proposalId, proposalSha256})`
- internal `decoder_execution_release_stopped({brainId, operationId?})`

Flags-only controls:

- `decoder_execution_status({brainId})`
- `decoder_execution_stop({brainId, operationId?})`

Main adds/resolves its one-use `reviewId`; renderer hashes/topology are never
authority. Calibration requires a main-seeded operationId and owns a private
Event. Status/Stop do no tensor/model/policy inspection or `_get`/Load. An exact
Stop cannot cancel a different operation or the supplying normal chat. A
one-slot exact queued cancellation covers the dispatch race without a lifetime
ID set. Stop immediately retires authority; tensor input release happens on
serialized calibration completion or the internal stopped-input cleanup.
That cleanup refuses a new capture/operation. Cancellation does not use a
generic supervisor grace-kill/restart. A currently executing native projection
finishes to its existing safe tile/page boundary; instant kernel preemption is
not promised.

## New scoped evidence and remaining gates

No closed model/Load/parser/manual-region/research suites were rerun.

- `test_decoder_path_execution.py`: initial8 inert cases passed34ms; two distinct
  async-home/capture-refusal cases passed1ms; two distinct serialized cleanup
  cases each passed1ms.
- `test_decoder_execution_boundaries.py`: initial4 constructor-free actual
  source cases passed128ms; three distinct candidate/registry/inline-boundary
  cases passed46ms; opt-in metadata case passed15ms. The one affected learned
  epoch case was rerun after opt-in gating and passed9ms.
- Twenty distinct new cases in total; callback timestamps are fake fixture
  values, never hardware performance evidence. Syntax parsing is separate.

Still unverified: actual CPU/Metal/CUDA loaded-owner numerical/resource/Stop
behavior, usable measured speedup, current packaged host/UI behavior. Still
unimplemented broader R1 work: general retained-KV and connected/full-region
whole-path Auto, whole multimodal operation coverage, VRAM-backed context,
additional direct packed backends and relevant hardware qualification. These
are not closed by the pure helper, source adapter or control UI.
