# Paged dirty journal and incremental v3 checkpoint boundary

The shared SQLite neuron/packed-vector/assembly cache is a live, derived
working cache. Authoritative neural state remains the checksummed packed v3
generation selected by the atomic `brain.json` commit. Neither SQLite nor the
substrate convenience `manifest.json` may select recovery state.

## Production writer path

- `install_generation_journal(sqlite_path, store_root, committed_pointer)`
  requires one shared three-store SQLite cache cleanly bound to the committed
  v3 generation. It installs triggers and imports prior ID-to-bucket/part
  membership from checked shards in windows of at most 512 IDs. A crash or
  reserve pause leaves state `building`; triggers preserve subsequent source
  mutations and a retry resumes that same import.
- Source row triggers record dirty IDs and a journal mutation revision in the
  source transaction; rollback rolls both back. Global decay metadata triggers
  mark every neuron logically dirty without materializing every ID.
- `DirtyShardPlan` stages membership deltas and complete changed groups in
  SQLite. Existing IDs keep their bucket/part; new IDs append to the bucket's
  last bounded part, allocating another part when full. Deletions rewrite
  their old complete group without moving later records. The 512-record bound
  is an I/O window, not a learned-neuron or recall limit.
- `write_paged_substrate_generation`, selected by the paged
  `NeuralSubstrate.save_sharded` path, reads only changed complete
  neuron/assembly groups from a ready shared journal and reuses untouched
  descriptors. Missing, old, building, separate or unbound journals retain the
  exact full-scan fallback; they are never used to omit valid records.
- Transaction-maintained row and linked-vector counters avoid `COUNT(*)`
  corpus scans in paged length/status queries. A SHA256 Patricia-Merkle
  assembly ID-set root updates from same-transaction membership deltas, with
  at most 256 path nodes visited per changed ID independent of cardinality.
  The derived forward descriptor index explicitly marks version 4 and its
  checksum algorithm. Legacy v1/v2/v3 inline indexes remain checked at migration
  boundary. Authoritative v3 generation/blob hashes do not change.
- Committed-cache rebuild binds paged neuron metadata in the same database.
  Native deferred cold-load completion installs the ready journal after
  verified bindings, so the next small update does not require another full
  membership import.
- `paged_synapse_endpoints.py` indexes complete source/target membership for
  every synapse group, including zero-weight cold edges. Group authentication
  and a generation-bound Patricia-Merkle map prove complete lookups and
  absence; a missing row/leaf is not treated as "no incident edge". Hot
  membership changes reread only their incident old groups. Truly new
  endpoints read no unrelated old groups, and all unchanged descriptors and
  compact forward entries are reused.
- Cold reconstruction checks every compact forward entry against actual
  checked packed shard contents. A missing, malformed or incomplete derived
  forward cache is reconstructed from those contents, not accepted merely
  because its self-reported checksum is internally consistent. Endpoint
  generation/authentication failure forces checked rebuild/full fallback.
- Derived cache v4 keeps exact packed forward signs and hot locators in
  immutable bounded per-group blobs. Its manifest holds only descriptors;
  unchanged groups reuse those descriptors. The runtime pages one group and
  uses authenticated complete endpoint queries instead of retaining every
  forward tuple/hot locator in Python. Large old inline caches rebuild from
  checked source rather than allocate a whole JSON tree. Canonical neural
  generation format remains v3, unchanged. Existing checkpoint GC reclaims old
  derived group blobs and obsolete inspection-query generations.
- Immutable identity proofs live in process-HMAC-authenticated SQLite rows,
  not a fixed 65,536-entry Python LRU. A resource-sized SQLite page window
  bounds working memory; corpus-sized proof population remains on disk.
  Optional proof-write refusal falls back to actual cryptographic reads.
  Existing unreachable-blob GC removes only proof rows for already deleted
  files, so checkpoint history does not indefinitely duplicate proof metadata.

## Authoritative commit and recovery

The `AdaptiveBrain.save` coordinator calls
`commit_paged_substrate_generation(substrate, store_root, brain_json_path)`
immediately after the real atomic `brain.json` write. It independently rereads
that file, verifies the pending substrate pointer and source revisions, then
rebases membership and changed endpoint groups, clears dirty IDs, and advances
all three store bindings in one SQLite transaction. Revision, identity, decay
epoch, plan nonce or
concurrent brain-pointer drift fails closed.
Lazy synapse persistence revisions cover metadata mutations such as uses,
eligibility and plasticity separately from forward-graph revisions. Neither
can change after publication and still allow the cache to be marked clean.

The first/full commit initializes a journal from committed shards. A reserve
pause or interruption leaves a missing/building/stale journal safe for full
fallback; it does not erase later source mutations. Post-commit rebase failure
is optional derived-cache maintenance failure, not loss of the already
committed neural checkpoint. Recovery rebuilds a fresh cache from `brain.json`,
never from a pending plan, dirty old cache, or ahead convenience pointer.

## Required I/O costs and platform security boundary

- Global decay faithfully rewrites every affected neuron group, or pauses at
  the reserve. No neuron silently remains at the prior logical epoch.
- Incident old groups are read in bounded pages when their hot membership
  changes. Non-incident old synapse groups are not reread. A first migration,
  stale/unauthenticated index, or committed cold recovery still verifies the
  authoritative source once before reuse; no stale cache can replace that
  verification.
- The v3 manifest retains O(number of shards) descriptor work, not an
  O(changed-records) manifest format.
- Unchanged blob bodies avoid repeat hashes only with authenticated checked
  file
  identity proofs on recognized local APFS strong-change-time filesystems.
  Proofs bind device/inode/size/mtime/change-time and reject tamper followed by
  restored mtime. Windows birth-time metadata, Linux without a verified
  change-cookie contract, and coarse-time/network/unknown filesystems require
  actual cryptographic body reads. This security boundary is intentional;
  Windows birth-time metadata is never used as a protected change cookie.
  Proofs from a prior process are not trusted: no authentication key is
  persisted, and cold recovery primes new proofs from actual checked reads.

## Whole-memory readout, inspection and argument views

- Exact assembly scoring/novelty scans packed assembly rows in bounded paired
  index/vector pages; it does not decode every assembly's metadata or treat
  ordinary neuron vectors as assemblies. Exhaustive exact work remains visible,
  not replaced by an approximate nearest-neighbor or count-truncated result.
- Recurrent recall uses an indexed disk graph and a disk active frontier. Every
  eligible signed contribution participates until the existing 0.52 contraction
  settles; there is no hop, active-node or top-k ceiling. Authenticated dirty
  sets update changed/incident graph rows only; missed revisions require a
  checked rebuild. Readout remains the continuous activity-weighted mean.
- Idle workspace selection uses exact scheduling components, safe group upper
  bounds and authenticated dirty-ID sets. Source transaction callbacks cannot
  reject valid learning if the optional cache pauses. Actual source records
  rebuild only changed groups. Global activation/uncertainty decay uses the
  original anchors; it never silently omits logically changed rows. The
  hardware-derived workspace limits current attention, not learned capacity.
- Sparse detail inspection builds a checked immutable per-generation SQLite
  query index once. A process-HMAC descriptor authenticates source provenance;
  a self-consistent forged public checksum is insufficient. Warm detail pages
  decode only requested records. New-process/tampered caches rebuild. Assembly
  confidence is separate from actual saved activation and observation state.
- Recall results/ID action inputs are lazy sequences. JSON reports carry byte-
  budgeted pages, exact totals and continuation data, never pretend the page
  was the full active set. Large media arguments use an immutable owned
  structural ID view with complete count/byte/hash and brain/turn ownership.
  The main/tool/worker path resolves every ID and streams unique vector sums;
  diagnostic page IDs never substitute for full neural input. The view contains
  no text, answers, weights or behavioral context and selects no recovery state.
- Saved snapshots/copies/.omni archives preserve structural reference files.
  Historical read-only queries retain original brain/turn headers and report
  explicit source provenance; inspecting a current-owned ancestral copy never
  grants permission to replay it as a newly generated current-brain action.
- Sparse dynamic packing and module export/verification stream bounded exact
  trit pages, retaining zero edges, canonical order hashes, global encoding,
  padding and both packed/logical checksums. Production load verification
  retains no full decoded tensor. Explicit legacy callers may still request
  decoded tensors; production export/import validation does not use that path.

## Validation boundary

`engine/tests/test_incremental_paged_checkpoint.py` uses a constructor-free
substrate storage shell and pure SQLite/filesystem fixtures. Coverage includes
complete-group replacement, stable births/deletions, global decay, no all-ID
walk on assembly append, Merkle corruption/order/deletion/rollback, zero-weight
incident synapse indexing, interrupted-pointer recovery, revision drift,
reserve refusal, counters, checked immutable reuse and mtime-restored tamper.
Post-commit rebase reserve refusal is retryable, and a neuron-only mutation is
not accepted as a clean three-store cache binding.
Further fixtures cover many disjoint zero-weight groups with exact selective
incident reads, authenticated empty lookups, missing members/proof leaves,
endpoint rebase rollback with journal rollback, stale-index rebuild, cold
proof priming, corrupted self-consistent/missing forward-cache recovery,
late synapse metadata mutation, 70,000 disk-backed proof entries, forged
proof rejection, ephemeral-session rejection, optional proof reserve refusal,
and proof cleanup after existing authoritative GC.
These checks do not construct, train, run or validate an Omni brain/model/app;
they do not establish natural recall, learning quality or release completion.
