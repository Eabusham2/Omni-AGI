# Retained context: real RAM/VRAM/storage ownership

October18 source slice. This is not a model-quality, GPU-throughput, training,
Load, or full hardware qualification result.

## Actual consumer and capacity

`brain.py::_configure_working_attention_resources` now derives storage pools
from the current `DeviceResourceManager` topology and selected budgets.
`retained_context_capacity.py` / `retainedContextCapacity.ts` share the literal
`tests/fixtures/retained-context-capacity.json` contract. Eligible distinct CUDA
and discrete-MPS tensor-storage interfaces need current total/free capacity,
exact device/physical-group identity and an enabled selected allocation.
Unified/RAM aliases add no bytes. Unknown XPU/NPU and unmeasured capacity do not
receive a grant. Unknown copy timings remain unmeasured, not a speed estimate.

`WorkingAttentionPager.save_page(retained_context=True)` actually allocates
immutable K/V pages on admitted device pools. Other saved training activations
retain their CPU/disk path. CPU producers remain RAM-first; an already-resident
producer can avoid a needless host roundtrip. Overflow uses the selected
distinct pools before authorized disk. Every new GPU allocation uses fresh
owner-policy/allocator admission, so another branch, model or process cannot
turn the per-pager capacity count into extra physical GPU bytes. Per-device
resident counts, immutable-page refs and real cold backing are separate.

Main's `ResourcePlanner.plan` consumes these eligible pool budgets, not
`projectedContextMaximumTokens`. Host indexes/page metadata are charged first;
hot K/V can then occupy RAM plus distinct VRAM, with only the remaining cold
K/V charged to optional disk. Each device is rounded to actual page allocation
granularity: several small GPUs cannot combine to hold one indivisible page.
Cold disk rounds conservatively by complete attention-page groups. The
separate full paused K/V/token/index snapshot includes the existing64-KiB
safe-tensor admission allowance per page and workspace state. Its primary-drive
capacity and the combined snapshot-plus-live-spill quota are both checked; the
same disk bytes are not credited twice. The configured selected window is not
silently reduced when currently free memory changes.

Selected spill folders need not include the checkpoint volume. Their local
quota constrains cold pages only; it must not cap a larger primary snapshot.
The joint quota uses the already-deduplicated primary-plus-extra physical
union, with independent primary-snapshot and authorized-cold-placement bounds.
The host regression also checks that a smaller explicit shared quota still
refuses the selection and a selected primary-volume alias adds no capacity.

## Auto and Manual

Auto retains the existing55/30/15 core/activity/transfer recommendation. Manual
is not confined to a30% activity partition: current complete packed-owner
sizes, nonpacked control costs and the admitted bounded transfer reservation
determine its usable retained pool. Main obtains packed-owner minima from the
trusted constructor inventory or complete direct-owner safetensor headers,
including the nonpersistent trigger; it does not substitute the largest sign
tensor or a guessed smaller layer fraction. Nonpacked header/inventory costs
are conservative fixed reservations. Missing ownership minima retain the
conservative Auto partition and are identified as missing, not called an exact
larger maximum.

The native core budget and `NativeCorePager.reclaim_device_for_context` consume
that Manual policy. Unpinned packed caches use their existing bounded
writeback/cold maps; a complete necessary packed owner and recurrent/transfer
scratch remain admitted. Smaller selected windows retain fitting hot model
bytes instead of forcing unnecessary spill. Pins, learning transactions,
Stop, unavailable backing or failed copies refuse/pause. Failed CPU cold-copy
staging no longer leaves an unwritten mapping registered as completed. New
manual RAM demand does not change learned shapes, reset any RNG, reinitialize
weights or drop the active attention cache.

The existing device UI allows explicit verified CUDA **storage** independently
of its unsupported native packed arithmetic. This does not enable a compute
route or remove its separate packed-CPU-tile acknowledgement. Default
selections remain unchanged. New/unverified devices stay unavailable.

## Exact boundaries

- Tail append stages all replacement pages before retiring the old page. A
  uniquely owned tail may use bounded temporary transfer credit; a shared
  fork's page cannot. Stop/failure leaves the original sequence usable.
- Each read fragment converts to the requested device **before** concatenation.
  Checkpoints receive canonical CPU pages one at a time, not a host copy of the
  whole window. Reload re-admits every page and preserves tail-page geometry.
- Pressure demotion publishes a CPU copy or verified disk page before releasing
  the old GPU owner. Failure keeps its bytes and charge. Physical removal or
  cross-hardware exact resume is not established by these contracts.
- Generation, prefill, complete-forward and cached-operation shape/policy
  signatures bind the pool identities, page size, core/control/transfer split
  and budgets. Cached checksums stage only bounded CPU fragments. Isolated
  placement restores the original pool controls; old CPU receipts do not grant
  new GPU storage authority.
- A changed resource/context policy with live K/V or packed pins refuses before
  session retirement, rotary changes or config mutation. Exact Unload/save and
  subsequent admitted restore are the supported boundary; the app does not
  close/discard live pages to force settings through.

## Evidence and remaining qualification

- `engine/tests/test_retained_context_capacity.py`:18 new stdlib/inert cases;
  byte-only simulated GPU adapters, exact AST page/codec/core-policy seams,
  literal main/Python capacity parity, unified aliases, simultaneous
  RAM+VRAM+disk, indivisible pages, pressure, Stop, failed copies, refs and reload.
  This file is suitable for the fixed **non-neural** CI contract runner.
- `tests/retainedContextCapacity.test.tsx`:7 new pure-main/component cases;
  the same literal math, owner-header accounting, actual `ResourcePlanner.plan`
  consumption (including primary snapshots larger than extra-only spill
  capacity), and real card element props/callbacks. Not a mounted live GPU UI
  or hardware allocation proof. The prior regional allocation assertion is
  updated because storage permission no longer requires compute permission.
- `engine/tests/test_retained_context_tensors.py`:one separate local check with
  real tiny CPU tensors and bounded safetensor I/O: noncontiguous input, partial
  tails, ordered fragment reads, spill checksum, refs and exact paged restore.
  Activations remain below64 KiB. Torch is deliberately absent from the fixed
  CI suite; no neural model/brain/decoder is constructed.

Actual GPU memory pressure, transfers, concurrent processes/devices, drive
failure, latency and supported-platform package behavior still require their
honest hardware qualification. This slice adds no direct CUDA/XPU/NPU packed
kernel, asynchronous stream scheduler, cross-rank Auto or vendor-framework
adapter, and makes no whole-model speed claim. Deferred full-folder training
remains deferred.
