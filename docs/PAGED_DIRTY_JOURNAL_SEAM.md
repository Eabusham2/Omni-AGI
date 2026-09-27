# Paged dirty journal seam (not yet an incremental checkpoint)

`engine/omni_core/paged_dirty_journal.py` is a fail-closed storage primitive for
the shared SQLite paged cache. It is not a second neural checkpoint and does
not change `brain.json` authority.

## What exists

- `install_generation_journal(sqlite_path, store_root, committed_pointer)`
  requires one SQLite file containing paged neuron metadata, packed vectors,
  and assembly metadata. All three store bindings must be clean and match the
  committed v3 generation. It installs triggers and imports prior
  `(kind, record ID) -> (hash bucket, part)` membership from checksummed record
  shards in windows of at most 512 IDs. A crash or reserve pause leaves state
  `building`; triggers remain active, and a retry resumes the same base import.
- Row INSERT/UPDATE/DELETE triggers record dirty IDs inside the source
  transaction. Rollback rolls back the journal entry. The neuron
  `decay_epoch` metadata trigger sets `fullNeuronsDirty` for a logical
  all-neuron mutation without materializing all IDs.
- `journal_status` validates the trigger SQL, store identities, and base
  generation. `iter_dirty_ids` exposes prior placement for changed IDs; new
  IDs have no placement. It refuses a building journal or a changed global
  decay epoch.

## Still required before the writer may skip a full scan

1. Assign newly learned IDs to stable bounded bucket parts, and enumerate
   complete changed groups from the persisted membership table. A dirty ID
   alone is not a complete replacement shard.
2. Emit changed v3 JSON/safetensors groups and reuse untouched descriptors,
   verifying source revisions before publication. The v3 generation manifest
   itself still has one descriptor per shard, so even this would retain
   O(number of shards) manifest work.
3. After `brain.json` commits the new pointer, atomically rebase membership and
   clear dirty IDs against that generation. If interrupted, detect the stale
   base on recovery and rebuild from the committed generation; never clear
   dirty state merely because the substrate convenience pointer advanced.
4. Handle global neuron decay by rewriting all affected neuron shards or by
   adding an explicit generation-level epoch/anchor format. Until then the
   full-neuron flag must force a full rewrite or pause.
5. Wire the live paged neuron metadata store into cold-load/rebuild and bind it
   to the committed generation. Journal installation currently refuses a
   separate or unbound neuron store.

The current paged v3 writer remains bounded in RAM and reuses identical blobs,
but it still streams all neuron and assembly records on each checkpoint. Do
not describe frequent small ingestion commits as low-I/O or low-wear yet.
