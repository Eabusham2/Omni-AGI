# Hardware-safe memory and storage planner

Stable `v1.1.0` checks the live device before a mind is built and again before
its neural worker starts. The check is a physical resource boundary, not a
personality setting.

## What is sized

The public capacity represents recurrent/paged working-memory items: recent
sensory patterns, active assemblies, and a continuously rescored neural
afterimage trail. It does **not**
claim that millions of items are placed in one dense-attention matrix. The
decoder's active token window is a separate, resident allocation. A
conservative tier floor is the only fixed target. Auto measures live safe RAM
after the native core/runtime, estimates KV and activation bytes per token,
checks the product context limit and accelerator availability, and incorporates
the measured storage class because faster storage can move colder competing
state out of RAM. Active context itself is never paged. Longer activity flows
through recurrent state, neural assemblies, and typed tensor pages.

Auto chooses the hardware-suitable green region. Extended chooses the higher
yellow region. Manual exposes the smaller of the live resident-memory maximum
and the model's own context limit. The slider's black/red/orange/yellow/green
bands are positioned from that measured range rather than centered at a fixed
percentage. A typed decimal has no UI-defined ceiling, but the same physical
and model preflight rejects a value that cannot stay resident.

For a new Build, the model input is not a pretrained checkpoint-size lookup.
The planner reads the versioned project-owned profile in
`architecture/omnicortex-ground-up-v1.json` and calculates logical parameter
count from hardware tier plus global-workspace latents. It budgets packed
two-bit projection, embedding, codebook, and adaptive-control bytes separately
from higher-precision fixed buffers, activations, liquid/eligibility state,
bounded update scratch, and checkpoint headroom. It does not charge a full FP32
cortical master, gradient, or Adam moment copy for packed synapses. An accepted imported native
OmniCortex remains sized from its own files and verified ground-up profile;
foundation-backed checkpoints are rejected rather than relabeled.

## Mandatory reserve equation

Before Build/start, the planner reserves:

- a device-adaptive OS/recovery reserve (up to the recommended 20 GiB on
  capable desktop volumes, proportionally smaller on constrained/mobile-class
  storage);
- 20% of the disclosed checkpoint-tensor budget for candidate headroom;
- a conservative working set derived from the native learning state (or from
  measured bytes for an accepted native OmniCortex import);
- the selected memory's packed spill bytes; and
- an adaptive device/OS RAM reserve that scales down on mobile/constrained
  systems and up on desktops.

Active context bytes are reserved before resident recurrent items are counted.
If the selected colder recurrent workspace cannot remain in safe RAM, storage
offload is required. The UI always says that storage is in use and shows a
slowdown estimate. The estimate comes from a cached bounded durable-write probe
and an in-memory copy probe. The app never writes a large/destructive benchmark
file.

The same projection is presented as a space-left report, with independent
rows for measured disk total/free, the device-adaptive reserve, selected
dataset bytes, native model bytes, current/origin/candidate checkpoint bytes,
maximum selected cold-memory spill, training scratch, future neural growth,
projected disk remaining, and projected space above the reserve. Selected
sources are referenced in place, so their already-allocated bytes are reported
but not falsely subtracted from current free space a second time.

Live free space is authoritative. Build and restart recompute the report rather
than trusting an old measurement. Long training and storage operations recheck
at bounded progress intervals and before atomic promotion; reaching the reserve
pauses safely, while more available space is not blocked by an arbitrary
higher threshold.

## Training shape and durable cadence

Auto preserves the configured logical optimizer target
`train_batch_size * gradient_accumulation`. It considers at most four physical
rows at once, clamps that candidate to live CPU/accelerator headroom, then picks
the largest value that exactly divides the logical target. Accumulation is the
exact quotient. For a target of 16, the safe shapes are therefore `4 × 4`,
`2 × 8`, and `1 × 16`; resource pressure changes allocation shape, not the
number of rows/windows represented by the logical group.

Before the first successful mutation in a new ingestion transaction, an
allocator refusal restores the saved RNG state and retries the entire logical
group with the next divisor (`4 → 2 → 1` for the standard Auto candidate).
Once a step succeeds, its physical size, accumulation, and token-window size
are hash-bound into the resumable learning schedule. Later pressure cannot
silently replace that committed trajectory: a failure after slow importance or
optimizer mutation rolls back through the worker's last atomic generation,
while an allocation failure on a frozen schedule becomes a recoverable pause.

Dataset learning publishes its full mutable generation at a default cadence of
512 committed source records and once more for the final partial group. The
generation includes parameters, safe-tensor Adam state, slow anchors and
importance, replay state, coverage, and the source-free schedule/cursor. It
contains no source text, token IDs, or embeddings. This cadence limits replay
after interruption without treating SSD scratch as per-step virtual RAM.
Emergency optimizer/activation scratch is also sequential, restart-safe, and
rate-limited from the measured storage class (five minutes to one hour, or a
conservative thirty minutes when throughput is unknown).

Training progress carries the same disk total/free/reserve/projected fields as
Build. This keeps a long Parquet/media/web run honest as checkpoints, replay,
and cold state consume space instead of showing only RAM/VRAM telemetry.

## Hot and cold neural state

RAM residency is continuously reprioritized. Currently firing, recently read,
frequently used, high-retention/rooted, and unfinished assemblies and synapses
rank hotter; the ranking can reverse as activity and access change. The sparse
substrate itself is currently RAM-resident, so its hot/cold result is explicitly
reported as classification and spill eligibility rather than a fictitious
physical page operation. Cold afterimages, replay batches, optimizer moments,
and inactive working patterns spill first.

Cold working patterns use restart-safe SQLite typed tensor pages with explicit
dtype, shape, metadata, and checksum; pickle is never loaded. Those pages are
addressable and runtime-readable. A read verifies the payload without removing
it; a page-in verifies and transactionally removes it from the cold store only
after successful decoding. Current focus and unfinished activity can request
ranked page-ins, and runtime status reports real reads, misses, bytes read,
page-outs, and page-ins. Physical-window trimming removes the lowest live
priority rather than assuming the oldest page is always coldest.

Learned neural assemblies and synapses remain authoritative. Expiring a
temporary cold working page does not delete what slow learning already made
durable.

## Acceptance evidence

- `tests/resourcePlanner.test.ts` verifies the reserve equation, physical-only
  manual sizing, offload decision, slowdown visibility, and Build lock.
- `engine/tests/test_record_checkpoint_resume.py` verifies Auto's exact divisor
  schedule, first-OOM schedule freeze, 512-record cadence, and hash-bound resume.
- `engine/tests/test_allocator_oom_recovery.py` verifies complete logical-group
  retry, RNG restoration, downgrade ordering, and atomic worker rollback.
- `engine/tests/test_state_offload.py` verifies adaptive/large-volume reserve boundaries, hot-state
  ordering, disk-backed working pages, restart, replay, optimizer spill, and
  corruption recovery.
