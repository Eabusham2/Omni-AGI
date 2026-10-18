# Device execution: source readiness and remaining R1 work

Internal evidence, 2026-10-15. This is not a hardware qualification or an R1
completion claim. The resource contract in `PLAN.md` and the detailed R1 section
of `FEATURE_MATRIX.md` remains unchanged. No brain, model, training job, device
benchmark or external service was started for this source slice.

## Existing implementation retained

| Path | Actual source contract | Boundary |
| --- | --- | --- |
| Topology and device budgets | `engine/omni_core/device_topology.py` enumerates RAM, CUDA devices/peer links, Metal and discoverable XPU/NPU interfaces. `device_resources.py` applies per-device ceilings and allocator admission. The host `deviceResourceService.ts` and `deviceResourcePlan.ts` preserve physical memory groups and current capacities; unified memory counts once. | Discovery does not establish native compute support. XPU/NPU capacity/backend unknowns remain explicit; unsupported devices cannot silently become usable pools. |
| Explicit measurements | `device_topology.py::_measure`, called by `collect_device_topology(measure=True)`, measures bounded memory, drive and transfer operations, including CUDA peer routes where available. `deviceResourcePlan.ts` orders diagnostic placements by measured local rate and RAM-to-device transfer cost. | Copy bandwidth is not neural throughput. The diagnostic order is not a whole-model or regional-placement executor. |
| Direct packed execution | `packed_native_kernel.py::_check` admits direct CPU/Metal kernels only. Bundled CPU code and the Metal shader consume packed signs/strengths inside computation; integer forward accumulation/output is 64-bit. | No direct native CUDA/XPU/NPU packed kernel is implemented here. Source support is not a speed guarantee on every device or packaged platform. |
| Explicit packed CPU-tile fallback | `model.py::_native_packed_projection` handles an unavailable non-CPU direct kernel with resource-admitted packed CPU tiles, returning results to the activity device. `worker.py::_builder_config` does not label generic CUDA/DirectML availability native acceleration. | This is CPU work plus transfers, not native CUDA/NPU acceleration. It does not create a full floating learned weight copy. |
| Current interactive placement | Single-device execution remains the default. The explicitly armed manual decoder-region consumer now dispatches real existing blocks through admitted activation roundtrips and per-device packed-owner limits; see `manual-regional-execution.md`. Runtime status distinguishes dormant/default placement from the armed manual path. | Source/inert-fixture evidence is not an actual multi-GPU neural run. Automatic whole-path placement, other regions, regional training and combined VRAM-backed context remain open. |
| Distributed training | `engine/distributed_train.py` has a real torchrun/rank launch path. `distributed_training.py::_resolve_strategy` selects canonical `packed-collective` synchronization for multi-rank native packed updates; incompatible DDP/FSDP packed updates are refused. `_bind_actual_native_device` requires CPU/Gloo relaunch if a native fallback leaves an NCCL launch on CPU. | Data-parallel ranks and canonical update synchronization are not regional interactive inference sharding. No distributed or training fixture was rerun for this audit. |
| Drive placement and fit | Authorized ephemeral native-core, attention and training/saved-activation pages use the existing drive-pool placement/admission paths described in `multi-drive-spill.md`. The host planner keeps projected multi-device capacity separate from the actually admitted RAM/spill context. | Persistent checkpoints remain on the primary volume. A projected sum is not credited as working GPU model/context capacity without an execution consumer. |

## Confirmed correction: stale transfer-route evidence

Previously, a discovery-only refresh reused a stored link measurement using only
its `from`/`to` IDs. A changed peer-access route or endpoint backend/memory group
could therefore retain a rate measured for a different physical interface.

`src/main/deviceResourceService.ts` now reuses an absent link measurement only
when both endpoint IDs still resolve to the same kind, backend, physical memory
group and unified-memory status, and the link's peer-access and unavailable
status are unchanged. Fresh capacity/allocated readings are not overwritten;
unchanged routes retain their existing rates. A newly supplied measurement is
not replaced. This is conservative measurement binding, not a new hardware
measurement, expiration policy or placement scheduler.

One **new fake-only case** in `tests/deviceTransferRateBinding.test.ts` covers
peer-route, backend and physical-group changes, an unchanged reverse route,
unchanged RAM-to-GPU rate reuse, and current free-capacity readings. The injected
engine returns synthetic topology; no worker or accelerator is launched.

Command: `vitest run tests/deviceTransferRateBinding.test.ts --maxWorkers=1 --testTimeout=10000`.
Result: **1/1 passed, 149 ms test time**. Previously closed resource/kernel tests
were not rerun. This result proves only the host measurement-reuse contract.

## Confirmed correction: weighted int64 profile and forward reserves

The original primitive profile declared int32 results and a 5,124-byte payload,
although the current forward retains int64 row blocks and concatenates them into
a second int64 result. It also omitted the 64×64 uint8 contact-state plane and
profiled `connection_state=None` rather than the current weighted input layout.

`native_compute_profile.py` and `nativeComputeMeasurement.ts` now use profile
format 2 and a new weighted/int64 kernel revision. Old-format/int32/missing-state
measurements fail validation; the main cache key includes the kernel revision.
Zero contact-state bytes retain unit strength and the original synthetic
numerical result; the production kernel/computation is unchanged.

The mirrored pure byte helper declares the persistent sample separately from
the known forward buffers:

| Declared 8×64×64 sample component | Bytes |
| --- | ---: |
| Float32 input + packed signs + uint8 contact state + scale | 7,172 |
| Quantized input + activation scales | 544 |
| Retained int64 blocks + int64 concatenated result | 8,192 |
| Float32 scaling and quantization working buffers + backend descriptor | 8,204 |
| Separate initialization/allocator allowance | 1,048,576 |
| Total admission | 1,072,688 |

Known forward buffers sum conservatively to **16,940 bytes**; this is not an
exact simultaneous allocator/RSS peak. The existing 1 MiB allowance is retained,
not repurposed to conceal the added state/result bytes. The old 1,053,700-byte
profile envelope no longer admits the new declaration, even from main's cache.

The profile explicitly records **zero repeated contacts/pages and no measured
repeat timing**. Actual repeated pages remain accounted by their runtime owner:
each 1,024-slot page has 256 packed-sign bytes, 1,024 state bytes, 10,240 endpoint
bytes and a 2,048-byte sort cache, plus the owner's 8-byte identity counter and
registry allowance. Allocated/tombstoned slots still cost their whole pages; live
count is not used as a substitute for physical page allocation. No repeat-page
storage or learning algorithm was changed.

`nativeCoreInventory.ts` provides a trusted optional projection-layout observer
without adding/changing immutable inventory fields. `omniArchitectureProfile.ts`
uses actual declared projection geometries, including modality output heads, to
reserve the largest 8-sample known forward workspace. It does not multiply every
embedding-table row on lookup or infer learned-repeat throughput from this
zero-repeat primitive. The reserve is added to transfer/training sizing and
`resourcePlanner.ts` resident/minimum-layer/device-fit arithmetic; previous
scratch, returned-output and headroom allowances remain. For the unchanged fixed
Personal geometry, the added reserve is 201,260 bytes. It is transient activity,
not a learned FP32 master or checkpoint payload. Larger execution batches and
dynamic repeat/page costs still require their existing runtime admission.

New evidence only: `tests/nativeComputeTensorBytes.test.ts` **3/3, 13 ms** and
`engine/tests/test_native_compute_tensor_bytes.py` **3/3, 0.006 s**. These use pure
arithmetic, injected timing/RPC and a constructor-free extraction of the actual
repeat-page inventory method. No Torch import, hardware primitive, model,
neural worker or training ran in the Python fixture. Existing profile fixture
constants were mechanically updated to format 2; their old suites were not
rerun. These checks establish byte/admission/consumer contracts, not measured
hardware memory peaks, backend speed or neural quality.

## Still-open source work, not silently narrowed

1. **Complete regional placement:** the new manually activated decoder-block
   consumer is a real source implementation, not diagnostic capacity projection.
   It does not complete automatic/all-region scheduling, regional training,
   cross-region concurrent execution, or GPU-resident context/KV placement. Its
   actual neural/hardware behavior remains unverified under the user's deferred
   new-brain proof. CUDA packed work is still explicit CPU-tile fallback, not a
   new native CUDA kernel. See `manual-regional-execution.md` for precise scope.
2. **Whole-path measured Auto competition:** current Auto chooses supported
   backends and safe quiescent migration; it does not compare complete native
   CPU/Metal/multi-device execution with actual geometry, paging, transfer and
   synchronization costs. `nativeComputeMeasurement.ts` is an explicit packed
   primitive profile; its optional `balanced-measured` sizing is not this missing
   end-to-end selector.
   A new callable **exact isolated decoder-prefill** source adapter now captures
   real normal input, compares admitted CPU/Metal/explicit-CUDA-fallback paths,
   authenticates current-owner receipts and can install an operation-only lease
   for the matching no-cache decoder boundary. It does NOT globally choose
   cached-chat/connected/full-brain/media placement. No neural/hardware run was
   performed; broader Auto remains open. See `whole-path-auto.md` for the exact
   stage coverage, source callbacks, new inert tests and still-missing consumer
   and hardware gates.
   An additional **ordinary retained-cache next-token** source consumer now
   exists, explicitly armed and one-shot. It compares private COW caches using
   the same loaded packed decoder, home sampler/RNG, all six declared stages,
   current physical budgets and exact cache/token/post-RNG equality. A checked
   winner can be consumed for that exact real visible single-pass chat step;
   it is not a reusable global route or full reply/media Auto. Private Stop and
   mismatch preserve the original normal step. Its22 distinct pure/tiny-CPU/
   constructor-free cases are source evidence only; no neural or hardware
   calibration ran. See `cached-decoder-auto.md` for custody, timing and precise
   limitations. Wider reusable cached/all-region/connected Auto remains open.
   A third callable source consumer now explicitly arms **pure perception
   operation Auto**. It qualifies actual ordinary or connected image/audio/video
   encoder/workspace closures, holds only admitted current-owner route metadata,
   and computes fresh compatible inputs on a qualified route. Exact equality on
   the qualification input is required; reuse on unseen inputs is explicitly an
   operator/geometry inference, not a previously measured output or global
   fastest-device guarantee. It preserves ordinary fallback once, scoped learned
   epochs, selected-owner restoration, current physical budgets and private
   Stop. Its 20 new pure/constructor-free/tiny raw-CPU cases are source evidence,
   not actual neural/hardware comparison. Full retained-turn, connected text/
   recurrent, learning, context, imagination/training and concurrent placement
   still lack generalized consumers. See `native-operation-auto.md`.
3. **Additional direct packed backends:** CUDA/XPU/NPU have no direct packed
   kernel in the current dispatcher. Preserve explicit fallback/unavailable
   status. NPU use remains conditional on supported operations and demonstrated
   benefit; discovery alone is neither a missing-hardware failure nor proof
   that it should be enabled.

Actual multi-GPU/NPU, separate physical-drive and packaged-platform performance
qualification is also outstanding evidence. That is distinct from the confirmed
missing executor/selector above: the absence of those devices on this Mac is not
itself a source defect. Root owns the next implementation choice and the shared
requirements ledger; this note does not mark R1 or its broader gaps complete.
