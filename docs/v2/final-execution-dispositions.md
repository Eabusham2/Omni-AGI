# Execution breadth: plan fidelity and remaining source

Source-only reconciliation, October18. This does not narrow the approved plan,
claim hardware speed, run a model, or turn unsupported routes into completed
features. Later implementation/evidence entries take precedence over historical
incremental-slice wording in the earlier Auto documents.

## Governing requirements

[PLAN.md](PLAN.md#research-methods-and-speed) requires investigation of Metal/
MLX, CUDA, CPU packed kernels, multi-GPU communication, NPU and compatible
distributed execution, selecting supported backends using whole-path evidence.
It does not require adopting every suggested framework or an unsupported
direct kernel. The same plan's [resource contract](PLAN.md#instance-lifecycle-and-resource-planning)
and [compound R1](FEATURE_MATRIX.md#r1--resource-placement-r174-r211-r217r222)
do require enabled per-device memory budgets, correct unified-memory grouping,
actual automatic placement, resource-derived context capacity and preserved
working state. Research/measurement conditions are not permission to omit a
required memory consumer or claim an unmeasured backend is faster.

## Precise dispositions

| Item | Current source and disposition |
| --- | --- |
| **Enabled VRAM as retained-context capacity** | **Required missing source path.** Production `AdaptiveBrain._configure_working_attention_resources` always binds `WorkingAttentionPager`. Its `save_page` copies retained activity to CPU and evicts to selected disk; `PagedTensorSequence` retains CPU pages and transfers compute tiles. `resourcePlanner.ts::planWorkingMemory` derives admitted context from RAM/spill, and `ResourcePlanner.plan` sets `selected.admittedContextMaximumTokens = base.context.maximumTokens` rather than crediting a projected GPU sum. This is conservative, not fake capacity, but it does not yet implement the requested usable VRAM fast tier. An actual retained-page device owner, admission/accounting, fallback/serialization and planner consumer are needed before that extra capacity can be admitted. A context-number increase alone is not a fix. |
| **Moving an existing live cache subtree between accelerators** | **Unsupported optional optimization, not the definition of VRAM capacity.** New retained pages can use admitted VRAM without promising arbitrary mid-operation migration. Current mixed paths explicitly require owned paged K/V and return ordinary outputs home (`mixed_region_native.py`, `cached_whole_operation_native.py`). A future migration must preserve current signatures, exact bytes, owner/RNG/cache identity, cancellation and rollback. Do not call it implemented, or use its absence to erase the mandatory capacity path above. |
| **Concurrent multi-device interactive computation** | **Unsupported candidate optimization.** Current `NativeMixedRegionPool` and `regional_block_call` execute actual sequential block roundtrips within complete measured operations. Best Work's `CooperativeNativeGeneration` interleaves admitted branches while preserving their state; it is not simultaneous GPU streams. The approved plan requires useful supported placement and preserves ordinary behavior, not a universal concurrent-stream implementation. A concurrent candidate still needs real complete-operation comparison, transfer/completion/resource custody and exact gates before adoption. Existing sequential support is not concurrent speed evidence. |
| **Cross-rank execution/collective custody** | **Existing native distributed training, not wholly missing code.** `distributed_runtime.initialize_distributed` initializes Gloo/NCCL; `distributed_training._resolve_strategy` selects `packed-collective` for native packed owners. `_train_wave` binds `PackedCollectiveController`, whose `commit` reduces bounded derivatives, performs a canonical seeded rank-zero update and broadcasts authoritative packed/state rows. `_phase`, rollback, `synchronize_control_state`, rank cursors, native checkpoint seals and publication are real source paths. Separate cross-rank **Auto comparison** is unsupported: `NativeTrainingWaveOps._closure` refuses world sizes other than1. Local receipts cannot authorize collective reordering or distributed Auto. Additional cross-rank Auto is conditional research; real multi-rank correctness/pressure/performance remains required qualification for any advertised distributed support. |
| **Direct CUDA/XPU/NPU packed kernels** | **Explicit unsupported backends, not required unconditional kernel additions.** `packed_native_kernel._check` supports CPU/Metal; `device_topology.collect_device_topology` reports native CUDA false and XPU/NPU unverified. `decoder_execution_native.candidates` requires explicit CUDA packed-CPU-tile acknowledgement; `regional_execution` refuses unsupported native devices. Keep these truthful distinctions. The investigation/if-beneficial condition does not authorize relabeling CPU tiles as native CUDA/NPU, nor does it require inventing every direct kernel without compatibility/benefit evidence. |
| **Broader MLX, vLLM, DeepSpeed and NPU integration** | **Research follow-through still pending, not silently adopted or cancelled.** The current source has direct packed Metal and canonical native collectives, but no integrated adapter for those vendor execution systems. The original `Vllm?`/`Deepspeed?` suggestions and NPU-if-beneficial request are governed by the plan's native-state and measurement conditions. A specific compatibility record and relevant whole-path comparison are needed before a positive adoption/default claim. This source review does not supply that benchmark or prove such adapters impossible. |

An unpaged low-level attention helper can retain tensors on its caller device
(`model.py::CausalSelfAttention.forward_cached`), but production's configured
pager takes the paged branch. That reference helper does not provide an
aggregate budgeted RAM/VRAM context pool or fix production capacity by itself.

Mixed consumers outside the six supported text kinds are likewise not yet
implemented. Current perception, media and prepared training retain their
complete homogeneous alternatives plus pinned once-only owner phases. Do not
claim mixed regional coverage for those kinds. Whether another mixed candidate
is useful is an operation-specific supported/measurement decision, not grounds
to privately repeat a real tool or learning effect.

## Required next action and evidence boundary

Keep the VRAM retained-context consumer open until its actual storage ownership
and admitted capacity are implemented; root has assigned that source work next.
Preserve the existing real distributed/collective code and the already composed
reply, memory, outer-learning and six-text-kind mixed consumers. Do not rebuild
them merely because older slice notes still describe their earlier absence.

Hardware qualification remains distinct: actual complete-operation latency,
transfer/pressure, multidevice/drive behavior and cross-rank results cannot be
inferred from primitive rates, source checks, inert fixtures or a package health
probe. No brain, Load, training, GPU test or frozen-suite rerun was performed for
this reconciliation. Full-folder learned-capability work remains user-deferred.
