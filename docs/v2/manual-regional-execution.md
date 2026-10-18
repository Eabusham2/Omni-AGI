# Manual decoder-region execution: internal source evidence

2026-10-15. This is an incremental R1 consumer, not R1 closure or a replacement
for whole-path Auto, all-region placement, regional training or VRAM-backed
context. No brain/model, hardware primitive or training was run for this slice.

## Actual consumer

`regional_execution.py::regional_block_call` wraps the existing
`DecoderBlock.forward` and `forward_cached`. A current reviewed process-local
lease selects the exact existing registered block, including nested connected
parents. The caller's hidden tensor is resource-admitted and synchronously
transferred to that device; the same block and its ordinary packed owners run
there. Its hidden result is admitted/transferred back to the caller/joint
exchange device. A cached attention object is retained, not recreated or
cleared. No alternate model, floating learned master, expanded weight matrix,
new learning objective or fabricated activation is introduced.

The current path makes actual caller-to-region-and-back transfers; it does not
pretend to use a direct GPU-peer pipeline. Receipts count observed transfer
bytes and completed region calls, monotonic elapsed duration, both directional
link estimates where the main-observed rates exist, and unknown costs otherwise.
These estimates are not neural speed or a reason to silently enable Auto.

`NativeCorePager.use` consumes per-device hot-cache limits from the active
consumer. It checks the target's limit and aggregate use of the same physical
group, evicting unpinned owners from that group when needed. Independent device
budgets no longer collapse into the old single global accelerator limit in this
manual path. Writeback, mutation observation, pins, partial-copy rollback and
immutable checkpoints use the existing packed-owner mechanism. Repeated-contact
pages stay CPU/paged with their existing signs/state/endpoints/cache accounting;
their bytes are not silently treated as GPU base projections.

Working-attention admission uses the regional target's fresh allocator/capacity
check instead of an unrelated root GPU's free-byte reading. CUDA compute bytes
are not miscounted as an additional anonymous RAM allocation; CPU/unified-memory
transfers still use the shared RAM envelope. Unified CPU/Metal capacity is not
summed. The derived one-element RMSNorm gain index follows its input device; no
learned gain is replaced. The single-device/off path remains the default.

CPU/Metal direct packed routes require actual native support. CUDA routes must
be explicitly reviewed as `packed-cpu-tile-fallback` with their own acknowledgment;
`nativeComputeSupported` remains false. This permits GPU activity/caching with
the existing admitted CPU packed projection tiles, not native CUDA dot products
or a speed claim. Unimplemented XPU/NPU routes remain refused.

## Review, lifecycle and custody

Optional `DeviceResourceSettings.regionalExecution` stores only versioned manual
preferences. Absent preferences preserve the original settings shape. Enabled
routes require explicitly enabled allocations. Neither saved settings, Load,
copy, import, module deepcopy nor recovery arms an execution lease. The global
block-to-consumer registry is weak/process-local rather than a module field that
could be copied into a snapshot.

The main-only loaded-owner serial methods are:

- `regional_execution_prepare`: descriptor pages plus policy/topology/registry
  hashes. Optional main topology is checked against current native compute
  interfaces/links; mutable free bytes and APFS storage-group rewrites are not
  stable compute identity. The returned snapshot retains main's measured rates.
- `regional_execution_activate`: exact `reviewed` policy/topology/registry hashes
  and topology; current config supplies saved routes. Activation mints a new
  in-process epoch. It does not change or rebuild learned state.
- `regional_execution_status`: actual operational state/counts, `leaseValid`,
  training inapplicability and the bounded diagnostic trace.
- `regional_execution_deactivate`: optional matching epoch, safe boundary,
  ordinary packed writeback. It never clears active context to make room.

All use `_get` with its no-load default; inspection workers and unknown fields
are rejected. Main's separate review-token service owns external authority.
Policy object/settings/RAM-envelope identity, registered object/geometry and
compute interface/physical-group/peer identity bind each lease. Native topology
uses the existing bounded observation cache; actual selected-device capacity is
rechecked at dispatch. An observed identity change invalidates dispatch rather
than silently remapping. ResourcePolicy/context replacement retires the lease
at quiescence; live attention pages, pins, online transactions or unresolved
inherited execution prevent changes without deleting their state. Cache identity
includes the manual epoch. Close retires only process-local execution authority.

Training remains the original single-device path and reports regional
inapplicability, including when the unused inference lease became stale. This
first slice does not claim regional training or enable a new default backend.

## New evidence only

- `engine/tests/test_manual_regional_execution.py`: **9/9, 0.020 s**, inert
  tensors/blocks, actual decorator and extracted real pager `use` transitions.
  Covers two distinct target IDs, exact roundtrips/bytes, stable learned fixture
  bytes, preserved cache identity, pressure/cancel/partial-copy failure, stale
  lease refusal, direct/fallback distinction, dormant deepcopy and shared-group
  admission. No Torch import or neural object was constructed.
- `engine/tests/test_manual_regional_boundaries.py`: first **3/3, 0.046 s**;
  separate new stale-inference/training case **1/1, 0.023 s**. Constructor-free
  extracted production methods cover actual target-GPU admission, Metal index
  normalization, loaded-only worker delegation and training inapplicability.
  The earlier three were not rerun for the separate case.
- `tests/manualRegionalResourceSettings.test.ts`: **2/2, 13 ms**, fake topology
  and pure settings/planning only. Unsupported native CUDA remains false; no
  actual context capacity is credited by a diagnostic projection.
- Seven changed Python entries parsed, actual block decorators/imports checked;
  three owned TypeScript entries syntax-transpiled without diagnostics. These
  are not a coordinated whole-project typecheck, app or neural/hardware proof.

Main/preload/IPC/advanced UI composition is now source-integrated; its separate
focused host/control tests and bounded synthetic Electron flow are recorded in
`manual-regional-host-ui.md`. This does not supply native hardware proof. The
new-brain/merged-resume proof remains deferred by the user. Broader R1 requirements
and actual hardware/service qualification remain in `device-source-readiness.md`.
