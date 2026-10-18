# Native v1 to weighted-v2 checkpoint migration

Native checkpoint conversion preserves the original learned bytes before adding
unit-strength state. It does not import a foreign model, relearn a dataset, or
advance a source cursor.

## Source proof

The loader first recovers the generation named by the committed `brain.json`.
The mutable-state store verifies its manifests/blobs and restores compatibility
files from those immutable bytes. Bounded loading then fills the actual native
owners.

When representation changes, `_verify_native_representation_migration` rebuilds
the original learned checksum stream in its original order:

1. Visit decoder, memory bridge, idea adapter, router, liquid controller and
   modalities in the same order as the native learned checksum.
2. Visit parameters, then canonical module getters and their declared tensor
   order. Resolve each tensor's actual registered buffer path, including a
   getter that exposes a child's buffer. Safe-tensor header sorting is not a
   substitute for the original learned order.
3. Compare every preserved learned tensor against its saved field in bounded
   chunks, checking shape, dtype and complete coverage.
4. Permit a missing strength field only for a recognized native projection or
   sparse block, and only when all its newly loaded bytes are zero: unit
   strength and zero new resistance.
5. Permit a missing contact-pool counter only for a recognized empty pool with
   a zero counter. Missing saved signs, unexpected learned fields, or nonzero
   defaults fail validation.
6. For dense-native recurrence, place the original packed matrix at its original
   stream position and use the verified bounded dense-to-sparse equivalence
   proof. No extra contacts or nonunit defaults may appear during conversion.

Source file identities are checked again after the pass. The reconstructed
checksum must match `parameterChecksum` in the already verified immutable
generation. A checkpoint without that required migration evidence is retained
and rejected; it is not silently rewritten.

Already-weighted v2 loads take a faster path when the verified saved-generation
checksum exactly matches the loaded learned checksum and the router needs no
conversion. They do not repeat the migration source scan.

## Cursor and state preservation

Root memory must retain the exact `brain.json` substrate pointer while it loads.
Its hash must match the substrate hash recorded in the recovered immutable
neural generation. A pending migration also captures the loaded state revision;
an intervening memory change invalidates that migration boundary.

Original ingestion cursors are validated against the reconstructed source hash
before any rebinding:

- v2 cursors must match the original neural generation, replay count/high-water
  mark/hash, working-memory page count/high-water mark/hash, and substrate hash;
- v3 cursors must match the original paged binding and the recovered joint
  generation's cursor and coverage;
- record/byte-window positions, source identities, coverage, schedules, epochs
  and commit sequence stay unchanged.

Saved CPU and accelerator RNG are read before publishing converted files. This
avoids invalidating a lazy reader's old offsets and avoids saving construction
RNG in place of the stored sequence.

## Atomic publication and recovery

The normal save transaction stages new neural/substrate/joint generations and
commits them through `brain.json`. Only the neural checksum and generation
bindings change for active cursors. The immutable origin is untouched.

Before that JSON commit, interruption leaves the old generation authoritative;
recovery restores its files and original cursor. After that commit, interruption
leaves the new generation authoritative. An optional publication or maintenance
failure cannot undo a completed JSON commit. Retry therefore starts from one
complete generation rather than mixing old offsets with new tensors.

Packed inference shards are regenerated from the committed current state when
their hash/layout has become stale. They are not used to justify changing the
authoritative learned checkpoint.

## Weighted memory integration

The v2 memory writer keeps the outer substrate shard format at version 3. Its
`native_weighted_migration_required` property identifies missing weighted-state
representation from the current committed generation, rather than a one-shot
in-memory flag.

The loader includes memory-only conversion in the migration transaction even
when the six-module core checksum stays unchanged. During save, an exact copy of
the original substrate pointer is retained. After weighted memory shards are
staged, and before neural staging or `brain.json` publication,
`NeuralSubstrate.verify_saved_weighted_generation` must prove matching source and
target identities, unchanged record counts/signs/counters/existing learned
state, unit defaults for missing companions and no added contacts. The root
rejects incomplete or mismatched proof receipts.

The normal save rollback restores the original pointer after an interrupted
stage, so retry still derives migration requirements from that original
generation. Active dataset positions, coverage and schedules stay fixed while
the joint substrate/neural generation is rebound.

The weighted writer's artifact comparison now has real resident and paged
generation fixtures. It checks every original record and contact, rejects
non-unit additions or changed native state, and rechecks artifact identities
before completing. Clean old native groups are rewritten with their new
companions without advancing learning. See [weighted native memory](weighted-memory.md)
for the representation, streaming interfaces, resource accounting and exact
verification scope.

## Focused evidence

`engine/tests/test_native_representation_migration.py` covers original tensor
order across core and sparse/dense recurrence, child-owned buffers, additive
defaults, source/live tampering, current pooled-contact reload ordering, source
hash anchoring, cursor refusal, v2 replay/page bindings, real v3 joint rebinding,
and actual mutable-store recovery before/after interrupted publication.

The fixtures use small tensors, short native projection owners, and generated
local checkpoint files. They never construct or train an `AdaptiveBrain`, use
the user corpus, or establish broad learned capability.
