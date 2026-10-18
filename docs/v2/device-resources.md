# Device resource planning and admission

## Implemented contract

The desktop requests topology from the neural worker's actual OS/Torch APIs.
Renderer input contains selections, never hardware capacities. Discovery does
not construct a model or allocate benchmark tensors. Explicit measurements use
at most 4 MiB per tensor/file payload, within an admitted probe budget. They
record copy/transfer/write throughput and sample scope; these measurements are
not neural throughput claims.

The topology includes system RAM, enumerated CUDA devices and peer links, Metal,
visible XPU/NPU backends, and explicitly selected drive directories. A backend
without a native packed implementation or measured capacity is labeled
unavailable for native compute rather than silently treated as supported.

Apple Silicon CPU/Metal share one physical memory group. Selected APFS volumes
are resolved to their container. If any selected macOS storage identity cannot
be verified, all selected storage is conservatively counted once; a failed
identity query cannot invent extra free capacity. Multiple directories in one
verified group do not multiply that group's budget.

Validated per-device settings persist in the instance config. RAM keeps the
existing adaptive reserve and process-family ceiling. GPU ceilings use the
selected share after their reserve and are further bounded by current free
memory. Metal also subtracts the entire managed app family's residency from
the host envelope. New accelerator allocations force a fresh allocator
reading. A failed reading pauses admission instead of reusing a cached success.
Supported Torch process-allocation fractions are reapplied at allocation
boundaries because another instance can have changed that process-global
setting. These are allocator/cooperative limits, not OS physical-page locks.

Native core paging consumes the selected execution-device budget. Training
planning uses its actual remaining bytes to reduce physical windows/batches or
pause while preserving the full source traversal schedule. It does not raise
the configured ceiling. Stored pool quotas do not shrink merely because other
programs temporarily occupy the drive; every real write still checks free
space. Automatic quota increases require the corresponding memory/storage
choice rather than being silently enabled.

The host keeps its **projected** multi-device total separate from admission.
Admitted context uses the real host RAM/index/workspace envelope plus locally
authorized selected-drive quotas consumed by the pagers. It respects both the
largest shared pool across instances and deduplicated per-volume quotas.
Temporary drive occupancy, pending writes and sparse promises are reported
separately: they pause admission instead of shrinking a saved window. A sum of
GPU capacities is still not credited without an actual multi-GPU consumer.

## Focused verification

### Build and runtime controls

`DeviceAllocationSelector.tsx` is mounted in Build and Device & runtime. It
shows observed RAM, GPU/VRAM, unavailable accelerators, drive identities and
shared physical groups without adding device capacities into a fit claim.
Refresh performs discovery; Measure requests bounded transfer/copy/write
observations. Missing rates/capacity remain unknown, and unsupported devices
cannot be enabled.

Eligible allocations have enabled, Auto/manual percentage and automatic-budget
increase controls. RAM is synchronized with the existing RAM selector in both
directions, so a saved per-device override cannot silently defeat that control.
The primary checkpoint/spill volume is read-only observation until its folder is
explicitly selected; only selected directories have the per-drive placement
consumer. The global overflow and storage-increase choices remain separate.
Selected drive cards show the host's applied shared-volume ceiling by physical
group, not the diagnostic free-space projection. Missing applied quotas stay
unreported. Charged bytes, pending writes and unmaterialized sparse promises
are available in the card details.

New Build keeps the native-picker receipt transient and passes it to both
preflight and creation. Runtime selection supplies the current brain identity
to the picker, then saves only the reviewed `deviceResources` settings. Imported
strings never create drive grants. Existing controls and the Best Work/Active
Mode ordering remain intact. Only the runtime-admitted context maximum is shown;
the projected GPU/device sum is not promoted to an admitted capacity.

Five focused selector/wiring cases cover shared groups, unknown/unsupported
devices, the unselected-primary boundary, RAM synchronization and transient
proof forwarding, including applied-versus-projected quota separation. A real
rendered Electron fixture with synthetic observations
checks CPU/GPU percentage controls, bidirectional RAM edits, Build versus runtime
picker ownership, measurement requests and narrow/light overflow safety.
Screenshots were inspected. These UI fixtures never construct or train a brain
and do not prove native multi-GPU execution.

### Native and host contracts

- 9 runtime tests: strict settings/config round trip, shared RAM/Metal accounting, fresh GPU
  admission and failed reads, disabled allocators, process-global limit
  reassertion, storage-quota increase choices, and training-window consumption.
- 6 topology tests: no allocations on discovery, multiple GPUs/peer links,
  unsupported NPU reporting, fresh free-memory reads per GPU probe, bounded
  payloads, cancellation cleanup, and retained disk reserve.
- 10 host tests: exact arithmetic, zero-context model fit, deduplicated physical
  pools, measured placement order, invalid/missing identities, independent
  measurement requests, APFS failure behavior, and fresh capacity with retained
  rate measurements.
- 2 host wiring tests: persisted per-device selections and the current pager's
  admission ceiling despite larger projected multi-GPU capacity.

All fixtures are numeric or tiny copy/file fixtures. They do not create a user
brain, train a corpus, or prove multi-GPU neural throughput.

The Node typecheck and changed Python modules compile successfully. A separate
real-Mac fixture applied the selected Metal allocator fraction through
`ResourcePolicy.require_device`, allocated exactly 8 KiB of floating activity
(not learned weights), synchronized it, and verified its sum. Its admission
reported measured capacity and shared RAM correctly; the isolated process then
released the tensor and exited. This validates that API path, not a trained
model's performance or a universal physical-RAM lock.

## Still required for the full v2 requirement

- Broaden the applied-drive-quota snapshot and reviewed limits into actual
  placement and live allocation-change diagnostics. Build/runtime selectors and
  explicit Refresh/Measure are connected with the scoped UI checks above.
- Implement and verify multi-GPU regional placement and cross-device execution.
  The present runtime admission protects the current operation's device; it
  does not aggregate several GPUs into an already working shared tensor pool.
- Native-core, router/control, attention and saved-activation ephemeral pages now have actual
  selected-drive placement, per-volume/shared quotas, native-picker authority,
  and cleanup tests: see [multi-drive-spill.md](multi-drive-spill.md). Persistent
  checkpoint relocation remains separate. Checkpoints/training/recovery/growth
  space is still reserved on the primary volume; only supported ephemeral
  placement adds external-drive capacity to admitted context.
- Per-drive automatic quota increases now reach the ephemeral placement
  consumer. Finish broader automatic placement/expansion and expose
  applied-versus-requested changes in the UI.
- Validate CUDA, NPU, multiple physical GPUs/drives, and packaged Windows/Linux
  behavior on available hardware. Copy-rate evidence alone is insufficient to
  select a faster neural backend.

## Reviewed RAM selection without a hypothetical model

`ResourcePlanner.reviewedRamEnvelope(config, hardwareTier?)` selects the exact
reviewed family ceiling before connected-merge worker admission. It honors the
RAM-device override or manual percentage; Auto matches the native worker's
tier and stored measured-drive-speed rule. Capacity uses physical RAM after
the OS reserve, while current available RAM is reported separately. It neither
constructs a model nor substitutes a summed-width plain-stack estimate for a
connected architecture. `modelFitEvaluated` is explicitly false. Three focused
fixtures cover manual/Auto/override behavior and unchanged capacity under
temporary memory pressure. This budget-only helper is not itself connected
geometry preflight; the saved-region reader below supplies that geometry.

## Saved connected-region capacity

Saved root safetensor headers are read once for actual packed/strength/control
bytes, metadata overhead and largest packed update scratch. Connected regions
use current `connected_layout` depth/expert declarations and actual current
leaf attention and workspace tensor shapes. Nested regions are traversed;
parent aliases do not multiply physical state. K/V bytes sum each leaf's own
width times its current depth, never composite width times summed depth.
Workspace tables and connected recurrent/exchange activity are budgeted too.

Descriptor and original metadata hashes, regular-file paths, owner coverage,
grown layers/experts and nested counts are checked. Missing or inconsistent
data fails closed rather than falling back to an ordinary decoder estimate.
This planning path does not reread every parent weight payload; full payload
integrity remains the loader/importer's responsibility. Runtime/control and
update scratch allowances are conservative estimates, not exact OS RSS.

Five tiny metadata/header fixtures cover grown/nested heterogeneous regions,
alias deduplication, invalid/missing owners and provenance, preserved Python
numeric hash lexemes, and the saved-instance preflight. Seven arithmetic cases
cover actual drive/global capacity, primary reserve separation, pressure,
unchanged RAM-first Auto and disabled spill. No neural training is involved.
Main's former model-fit Auto percentage escalation was removed: it could
approve context against a higher percentage than the worker actually enforced.
Main, merge admission and native runtime now use the same tier/measured-drive
base policy; larger models require supported spill or an explicit new cap.
