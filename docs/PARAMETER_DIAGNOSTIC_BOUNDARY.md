# Parameter diagnostics and rollback resource boundary

Discovery: `_parameter_copy` cloned every learned packed tensor from the
decoder, memory bridge, idea adapter and liquid modules into CPU RAM. Its actual
production callers were chat and idle cognition; feedback used module checksums,
not that delta copy. The streaming logical-batch caller also omitted the existing
packed snapshot disk fallback and never closed its rollback snapshot. Exact
checksum hashing already streamed contiguous values, but made a whole contiguous
copy for strided values.

## Exact, explicitly scoped core delta

The production baseline now captures stable owner/schema metadata and admitted
residual control Parameters, never a full packed-weight copy. Actual before-write
hooks retain only first-original changed bytes in a private sparse disk journal,
in bounded blocks with RAM scratch and disk admission. Final actual codes produce
signed-ternary-level net L2. Repeated flips, reversals, interrupted writes and
logical-batch rollback are not mistaken for cumulative net parameter change.

Hooks follow collective derivative deferral and wrap actual canonical/replica
installs, bulk fill/copy and rollback. Registered module/buffer names survive
paging replacement. Newly added expert owners contribute their actual new codes;
replacement/removal of an existing topology is refused instead of silently
claiming an exact metric. Such router/depth architecture migrations occur only in
isolated evolution proposals, outside the per-chat/idle diagnostic scope.

Journals close on normal return, early Steer/EOS or exception. Their storage cost
is proportional to bytes touched during the scope plus sparse-index/SQLite
overhead; a scope touching every weight can still be expensive. This is not a
claim that a fully changing large model has a constant-size exact delta history.
It avoids unconditional full-RAM/full-SSD model snapshots for each turn.

The numeric delta covers only decoder, memory bridge, idea adapter and liquid
weights/controls. VSA neuron-vector and sparse-edge net deltas remain explicitly
unmeasured/null, not zero. Module checksums include their declared module learned
tensors but do not include all substrate parameters. Trace/UI wording identifies
these scopes and labels STDP as activity, not an all-substrate net weight metric.
A core delta of zero does not establish that no neural learning occurred.

## Real logical-batch rollback

Real rollback remains an integrity snapshot of authoritative uint8 packed codes,
row resistance and tiny event state—not a floating weight master. Full CPU-copy
admission failure uses the designated private disk directory and disk reserve;
bounded transfer/header RAM must also be admitted. Restore resolves registered
targets after paging and checks their shape/dtype. A same-operation rollback uses
the existing diagnostic originals and never demands a second diagnostic journal
under pressure. Snapshot lifetime spans every retry of that logical batch, then
closes before final resource maintenance and on final failure.

Exact SHA-256 checksum format remains shape prefix, dtype prefix, then historical
logical C-order tensor bytes. GPU/strided input is streamed in bounded chunks;
mapped CPU chunks are released after hashing. Exact hashing still costs O(P)
work, but no whole-owner CPU clone or whole strided-owner copy.

Validation is constructor-free synthetic tensor/state/storage fixtures and source
integration checks. No brain/model constructor, backward, optimizer step,
training, app or CI run was performed. Actual large-model throughput, hardware
allocator behavior and learned language quality remain separate acceptance gates.
