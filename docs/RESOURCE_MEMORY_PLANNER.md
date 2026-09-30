# Hardware memory and storage planner

Candidate v1.1.1 sizes a fresh OmniCortex from the selected total-device RAM
envelope, with room for runtime, activity and bounded training work. Auto aims
to keep its normal baseline in RAM when competing applications are closed. It
does not choose the largest model merely because some bytes can fit on disk.
Existing learned geometry is never silently resized by runtime settings.

## Auto and manual capacity

Fresh native width/depth, workspace and context are jointly admitted against
packed weights, actual nonweight state, attention, indexes and working
reserves. These are shape inventories and conservative allocation estimates,
not measured intelligence or a guarantee of neural tokens per second.
Optional small-kernel profiling does not gate Build or set a hidden TPS ceiling.

The recommendation comes from total RAM and measured memory/storage properties,
not transient free RAM while another application is busy. Current pressure can
produce a close-other-programs warning or a recoverable pause; it does not shrink
the chosen baseline or its learned parameter tables.

Auto selects the resident working region. Extended and manual settings can use
the designated pool with a visible slowdown estimate. Manual context maximum
comes from physical RAM, indexes and reserved storage rather than a fixed
32,768 or 4-million ceiling. Cold attention really can page: exact tiled causal
computation preserves the logical window instead of calling truncated history
extended context. Resident token/metadata and live execution minima still have
to fit. Bigger context is not automatically better or faster.

All learned weights remain packed ternary; activations, fixed controls, timing
and eligibility are accounted separately. Packed weights are four trits per
byte, not a claim that the entire process is physically 1.58-bit. No full
floating cortical master or packed-weight Adam mirror is budgeted or retained.

## RAM usage ceiling

The chosen RAM share is a ceiling, not a promise to fill that amount and not
physical page pinning. OS compression and cold drive spill remain permitted.
Auto no longer raises its allowance simply because a new allocation asks for
more memory. Runtime resource updates preserve learned shapes and refresh the
real core/activity pager policy.

The trusted launcher supplies the app-family owner. Admission uses measured
managed-family RSS, a current-worker footprint floor and projected next
allocation bytes. Desktop/renderer/GPU helpers, neural/inspection workers and
managed codec descendants are included; unrelated user applications are not.
Shared pages may be conservatively counted more than once. Family reads are
bounded and cached for at most one second; sampling age and duration remain
visible. Unknown usage is not treated as zero. Warm operations reclaim and
recheck, or pause when unavoidable live state still cannot fit.

This is sampled cooperative admission, not hard OS RSS isolation or an atomic
cross-process allocation reservation. Native allocators and concurrent activity
can change residency between samples. Standalone CLI covers itself and children;
cross-rank CLI accounting needs a trusted common owner supplied by its launcher,
not an inferred terminal/SSH ancestor.

## Designated storage and reserves

Every identity shares the largest required pool reservation; quotas are not
summed as separate empty allocations. The desktop disk floor is 20 GiB, distinct
from the adaptive OS RAM reserve. Too little space pauses rather than weakening
that floor. Model/current/origin/candidate copies, growth, training scratch and
the selected cold backing all participate in preflight.

Selected source files already occupying disk are reported, not subtracted from
free space twice. An Auto pool grows to the admitted obligation; a too-small
manual pool reports its actual blocker instead of selecting a tiny model to
conceal it. Free space is rechecked before durable publication and during long
work. A pool is not a guarantee against unrelated future disk consumption.

## Training and hot state

Resource-derived physical microbatches and accumulation preserve requested
logical work and actual labelled targets. Committed source/window cursors are
bound to source identity and policy. Failed windows remain uncommitted; decoder
or allocation failure never means complete traversal or a silently skipped row.

The configured logical optimizer target is
`train_batch_size * gradient_accumulation`. For an explicitly selected four-row
physical start, `4 → 2 → 1` is one possible safe retry order, not an Auto parallel
ceiling. Dataset ingestion retains its default cadence of 512 committed source records
and the final partial group; publication binds actual committed coverage/state.
Physical batching and checkpoint cadence do not shorten the requested dataset.

Packed core owners and activity pages keep hot state in admitted RAM and use
private cold storage when necessary. Sparse recall/frontier, idle queries and
inspection use bounded pages rather than full entity lists. Exact whole-memory
scoring can still require exhaustive work, and weak filesystem identities still
require cryptographic reads. Dirty checkpoints reuse unchanged verified groups.
Rollback journals first-original learned bytes rather than writing an entire
core before every small change; nonweight/resistance baselines remain exact.

## Verification boundary

Arithmetic, injected resource/process, file/storage, typed controller and paging
fixtures verify these mechanisms. Package CI runs health/UI contracts without
creating a brain. Full native training, actual large-device performance,
parameter-only recall, intelligible speech and beneficial recursive improvements
remain separate, user-deferred qualification. See
[implementation status](IMPLEMENTATION_STATUS.md).
