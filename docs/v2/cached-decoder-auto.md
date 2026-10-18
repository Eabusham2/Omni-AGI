# Retained-cache decoder Auto: exact one-shot source consumer

Internal source evidence, October 15, 2026. This extends, but does not replace,
the separate isolated-prefill adapter in `whole-path-auto.md`. It is not full R1
closure, an actual neural/hardware comparison, or a performance claim.

## Reachable operation and scope

`cached_decoder_execution.py` and `cached_decoder_native.py` implement the
distinct work kind **`native.decoder.cached-next-token-sampling`**. A user
explicitly reviews current hardware and arms one operation before a normal
chat. Prepare/Arm do not synthesize a prompt, quiz, forward, or training task.
The ordinary visible single-pass chat path opts into the actual sampling-loop
hook. Its first prefill is unchanged. The next valid retained-KV frontier can
perform the authorized private comparison. Invalid frontiers retain the normal
step; there is no automatic context clear or prefix replay to force eligibility.

The comparison uses the same already-loaded decoder and packed learned owners,
not a second model. Actual registered decoder blocks may route to admitted CPU,
Metal, or explicitly acknowledged CUDA **packed CPU-tile fallback**. Workspace,
readout, sampling, and returned cache stay on the original home device. Direct
CUDA/XPU/NPU kernels are not added. Connected decoders, Best Work/private branch
generation, resume-specific paths, arbitrary queued operations and full media
execution do not acquire this new authority.

No model was created, loaded, trained, calibrated or invoked to verify this
slice. The CPU fixtures below use tiny raw tensors and fake forward callbacks.

## Exact identity, comparison and adoption

- The review binds current process owner/session, complete current configuration
  and resource policy, actual registered module geometry, selected device budgets
  and physical topology. A saved/imported preference or receipt is not a lease.
- The actual latest cache/input/cue/RNG and learned/control mutation fence are
  captured at the real frontier, rather than binding a pre-chat learning epoch
  that normal experience processing would immediately invalidate.
- Existing exact packed-owner learning epochs distinguish known representation
  moves/disposable sort caches from actual persistent sign, strength, stability
  and repeated-contact changes. They are neither reset nor replaced by a
  manufactured checkpoint hash. Non-pager controls also participate.
- Each candidate gets one excluded warmup and at least three measured samples,
  round-robin. Sample count is resource-admitted, not a product lifetime cap.
- Acceptance requires exact logical cache bytes, sampled token, float64
  entropy/log-probability representation, and post-sample home RNG state. A
  numerically similar CPU/Metal result is insufficient for this retained-cache
  contract and may be rejected. No unsupported fallback is mislabeled native.
- Selection uses matched complete receipts and nonoverlapping observed latency
  ranges. It additionally requires the target's current allocation to permit
  Auto. No invented throughput/confidence or primitive-bandwidth extrapolation
  chooses the winner.
- The winner can be consumed only for this exact one operation. Its checked
  cache and exact sampling result are adopted once, at the real caller boundary;
  the actual generator advances to the checked post-state once. There is no
  global placement mutation or reusable plan for a different future token.

### Declared timing coverage

The ordered stages are:

1. `cache-cow-snapshot`
2. `cached-workspace`
3. `routed-decoder-blocks`
4. `home-readout`
5. `same-home-sampling`
6. `retained-cache-return`

The total is measured with actual paging/transfer counters, target **and home**
asynchronous completion, and restoration of original packed residency. The
workspace/block/readout stage markers certify coverage, not independently
measured per-stage durations. Raw page verification inside a trial also costs
real elapsed time and increments logical read bytes. Capture/review overhead,
the whole calibration pool, host/network/user waits, a full reply, initial
Load, connected execution and media are not claimed by this work kind.

The managed logical cache is matched exactly. OS file-cache temperature and
driver state are not asserted identical or reset. Warmups are excluded and this
is explicitly warm retained-cache evidence, not cold-start evidence. Sampled
family RSS/allocator readings are not an exact absolute simultaneous peak;
logical read/write counters are not physical disk-I/O measurements.

## Copy-on-write activity and real resource boundaries

`cached_activity_snapshot.py` copies only mutable cache containers and small
running-statistic tensors. `working_attention_paging.py` provides independent
append cursors over immutable activity pages. Reference counts do not double
charge physical shared pages. Tail coalescing creates a new page; the original
cache's page bytes survive until its final owner releases them.

Metadata, small tensor copies, transfers, compute, retained outputs, receipt
serialization and spill allocations use current resource admission. Failed
last-owner unlink keeps the backing file and quota tracked; it cannot report
the bytes freed. A first CPU saved-activation restore now correctly builds its
backing image, while exported CPU tensors use private writable mappings and
keep that immutable image alive until their last storage view is released.

Page-integrity evidence is stored on the existing activity page object, not an
unbounded cache or second authoritative database. Hot proofs bind actual tensor
identity, version, storage, shape and dtype. Cold proof reuse on POSIX requires
an exact opened no-follow regular file, device/inode/size/mtime/ctime and exact
source digest, with identity checked before and after. It additionally uses the
existing conservative `_identity_proof_supported` eligibility helper: reviewed
local APFS with high-resolution change times only. Replacement forces
reverification; modified bytes or symlinks fail. Windows, network, Linux and
unknown paths hash the exact opened bytes again; Windows does not claim POSIX
atomic no-follow protection. The new FD case injects both eligible and ineligible
classification without running a hardware/filesystem benchmark.
Cancellation is checked before, during and after bounded hashing.

Temporary routing supplies per-device packed-owner budgets through the existing
pager. It does not arm saved manual routes or globally migrate controls, working
attention/KV or the decoder. Restoring original packed residency still requires
current resources; genuine refusal is an error, never a silent warm-worker
restart, learned-state rollback or weakened budget.

## Stop, world effects and durable evidence

Status/Stop use metadata-only worker controls. They do not call `_get`, initialize
a controller, inspect Torch state, load/restart a worker, or wait behind the
private comparison. Prepare/Arm stay serialized loaded-owner operations.

A preparation-generation ticket is registered before any metadata callback.
Current/no-ID Stop or close retires it; a late callback cannot publish a new
prepared review. An unrelated old operation ID cannot retire replacement
authority. Exact queued Stop prevents only its own later Arm. Import/Unload,
configuration replacement and owner close do not preserve active authority.

Private Stop cancels only the comparison. Trials use flags-only cancellation,
private home RNGs and no stream/activity/world/conditioning callbacks. A stopped,
stale, inexact or nondominant comparison releases its trials and runs the normal
original step once, with original context and unadvanced RNG. Real chat Stop,
Steer and suspension remain the caller's existing behavior. If the genuine
world boundary changes RNG, original computation uses that new state without
resetting it or replaying the world boundary. If the genuine boundary throws,
its exception propagates; it is not caught and replayed as a private failure.

Actual observation/selection/failure records use the existing paged EventLog,
exact source-row hashes, current operation sequence and brain ID. A private
issued receipt head/count authenticates selection; public hashes or imported
historical receipts alone cannot authorize it. Receipts contain identity hashes,
not raw user text, credentials or hidden prompt memory. Failure receipts remain
failures. Resource refusal may itself prevent an optional diagnostic write.

## New focused evidence only

**22 distinct passing cases**, not 22 neural runs:

- `test_cached_decoder_execution.py`: 10 pure injected-controller cases; final
  affected Prepare-fence pass **0.066 s**. Exact adoption, cache/token/RNG
  mismatch, learned/hardware changes, private Stop, queued Stop, boundary RNG
  change/exception, pager cancellation and late Prepare publication are covered.
- `test_cached_activity_ownership.py`: 8 tiny CPU cases using actual AST-loaded
  pager/cache classes, owned small raw files and injected policy. Five original
  cases passed in the initial **0.017 s** six-case run; first restore failed with
  `None.path`. The surgical indentation fix then passed its affected case in
  **0.053 s**. New FD identity/tamper/cancellation and transient dispatch cases
  passed separately in **0.016 s** and **0.000 s**. The original failure is not
  counted as a pass. Sampling matches the prior law and exact RNG/draw behavior.
- `test_cached_decoder_boundaries.py`: 4 constructor-free source cases,
  **0.521 s**. Actual worker methods keep exact loaded identity and refuse an
  existing independent inline job; flags controls bypass neural getters. The
  actual generator loop with fake forward reaches the explicit retained hook,
  leaves the default path unchanged and serves the world boundary once/token.
  Only visible single-pass chat opts in, using raw Stop/Steer/suspend flags.
- All 11 affected Python production sources parsed successfully without imports
  or execution. Main service/supervisor/preload and GUI evidence are maintained
  by their owners, separately from these native-source results.

The earlier isolated-prefill20 cases and unrelated old suites were not rerun.
The initial new ownership-run handle was not retained; it was confirmed exited
and its result was not used. The reported six-case diagnostic is the captured
run above. No missing result was silently inferred successful.

## Still open

Actual same-brain cached neural/device performance and retention proof is
unverified under the user's current deferral. Exact-cache cross-device equality
can reject candidates. General reusable retained-KV placement, all-region and
connected execution, regional training, VRAM-backed context, whole-brain/media
measured Auto and platform qualification remain distinct R1 work. This one-shot
consumer must not be presented as their completion or as an automatic speedup.
