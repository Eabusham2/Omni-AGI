# Inline isolation and slow integrity snapshots

The inline worker previously used ordinary module deepcopy before moving the
copy off MPS/GPU, and returned `None` for preparation failures. Generic parent
modules and unregistered tensor attributes could therefore duplicate state on
the original device and fail without a truthful artifact error.

Inline image/audio/video now clones the existing architecture as a constructor-free
CPU skeleton, copying state in admitted chunks. It never creates/loads a second
brain or initializes a new model. The selected region uses actual remaining
shared CPU admission first; packed owners spill to private mapped pages only
under pressure. There is no independent parallel RAM budget. Float/control,
nonpersistent activity, idea and liquid-state resident minima are explicitly
admitted. Capture is CPU-only even when the authoritative brain uses an accelerator.
Unregistered tensor-bearing custom control objects are refused, not deepcopyed
on an unknown device.

Exact independent region isolation still copies the full selected region state;
if RAM cannot fit it, cold packed bytes require storage. This is not a claim of
zero-copy isolation. Nonweight state and later codec activations retain real
resident/computation minima. Preparation failure retains the exact owned failed
future with a typed resource pause/error. It is not silently dropped as `None`,
nor regenerated as a new unowned artifact. Decoder state closes only after that
decoder finishes or before a cancelled/unstarted job is removed.

The adjacent slow transaction snapshot previously cloned all module state into
CPU RAM, plus optimizer/anchors/importance. It now retains module/tensor refs
until admitted bounded serialization of nonweight/control state. Only actual
authoritative learned packed identities use first-original changed-byte COW
rollback, not an upfront whole-core disk copy. Uint8 resistance, counters and
other control state retain an exact bounded baseline; dtype alone never exempts
them. No/few packed mutations cause no upfront full-core packed snapshot I/O.
Real full-change cost remains proportional to bytes touched plus sparse-index
overhead. Common mutation hooks also observe row resistance; replica install and
rollback cannot bypass exact enclosing diagnostics.

Recent token lists, mutable context, optimizer structure and metadata are admitted
before deepcopy. Restore adopts only admitted control minima and resolves live
registered module targets after exact topology restoration. Nested guards close
snapshots after real restore/commit use in chat, supervised dialogue and deferred
slow consolidation. Existing parameter checksums, training modes, counters, RNG,
optimizer settings and stability rollback remain authoritative.
Exact integrity checksums still stream the complete core: this removes mandatory
full-core snapshot writes, not all O(parameter-count) reads or integrity work.

Validation is synthetic constructor-free skeleton, primitive tensor, storage,
control and lifetime fixtures. No model/brain constructor, model inference,
backward, optimizer step, training, app or CI run was performed. Large-model I/O,
accelerator behavior and generated media quality remain live acceptance gates.
