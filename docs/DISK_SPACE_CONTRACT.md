# Disk space-left contract

All write-producing Omni operations share one safety invariant: the projected
free space after the next bounded write may not enter the device-adaptive
OS/recovery reserve. Capable desktop volumes use up to the recommended 20 GiB;
small Linux, Android, and iOS volumes use proportional bounded reserves so the
policy itself does not make the device unusable. A higher feature-specific
cutoff must not reject an otherwise safe operation, and a lower cutoff must not
consume the reserve.

## One report shape

Build, training, and storage-operation progress use the same report fields:

- measured time, disk total, and current disk free;
- platform/device-derived mandatory reserve and desktop recommendation;
- selected dataset/source bytes;
- native model bytes;
- current/origin/candidate checkpoint bytes;
- maximum selected cold working-memory spill;
- training/staging/operation write bytes;
- future neural-growth headroom;
- projected disk remaining and projected bytes above the reserve;
- whether the operation is paused at the reserve.

Selected local datasets are normally referenced in place. Their size is shown
for truthful workload context but is not subtracted from current free space a
second time. Scratch, downloaded/crawled bytes, extracted archive bytes,
missing content-addressed blobs, copy fallbacks, checkpoints, and newly packed
state are physical writes and are subtracted.

## Recheck rules

A preflight report is a snapshot, not a permanent promise. Live free space is
re-read:

- before Build allocation and immutable-origin publication;
- before each checkpoint/candidate/packed-state promotion;
- at bounded byte or record intervals in long dataset, media, crawl, upload,
  import, and export streams;
- before a hard-link fallback makes a physical copy;
- before the final atomic rename makes a new brain or artifact visible.

A reserve pause keeps the last committed neural cursor and removes or
quarantines incomplete temporary output. Retry recomputes current free space
and continues from the durable boundary; it does not silently shrink the
selected brain or claim skipped source coverage.

## Copy and archive accounting

Duplicate distinguishes logical bytes from new physical bytes. Existing
content-addressed blobs and successful hard links do not consume their logical
size again; a copy fallback does. Export measures the destination volume.
Import budgets the compressed staging file, expanded entries, missing CAS
objects, and link/copy materialization so its temporary peak is not presented
as only the archive's compressed size.

The storage-operation event is monotonic in completed bytes and preserves the
same disk report schema used by Build and training. Cancellation never leaves
a partially installed brain visible.
