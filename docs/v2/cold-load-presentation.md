# Cold Load: discarded presentation and derived-copy IO

## Retained evidence first

The earlier connected current-seal capture took 1,023.891661 s; its subsequent
bounded Worker Load stopped at 113.192 s without reaching reference/decode.
Those remain failed/open latency evidence. The completed read-keeper,
endpoint-membership comparisons and persisted-recall finding were not rerun or
multiplied into a whole-Load speed claim. Current Load already avoids eagerly
constructing the temporary recall graph.

Before this work, process inspection found no Omni model or Electron app. Root's
Vite PID 52944 on 5189 and CUA processes were retained. The 16 GiB host reported
37% memory free, roughly 34 GiB disk available and 4.54 GiB swap used. Those are
point-in-time observations, not reserved capacity or a platform guarantee.

## Confirmed source waste removed

1. The unpublished Load phase already computes the full native learned checksum.
   It now uses that same value for the later packed-copy comparison, with no
   intervening learned mutation. Source/generation, migration, cursor and full
   existing sidecar integrity verification remain. No persistent hash cache or
   cross-generation trust was added.
2. `Worker.load(config)` previously discarded `update_config`'s full summary,
   then constructed another one. Its internal `include_summary=False` route
   keeps all configuration validation, admission, saves and audit events; only
   the unused final presentation is omitted. The one final returned summary and
   warm cancellation check remain.
3. One summary previously called authoritative parameter accounting three
   times, including an eagerly evaluated `dict.get` default. It now measures
   once synchronously for metrics/runtime card and copies each returned dict.
   The next summary remeasures; no inventory is cached across neural changes.

Six new constructor-free cases passed in **15 ms**. They establish these source
and call-count contracts, not a numeric whole-Load speedup.

## One bounded actual random-native attempt — not accepted as a pass

The single new attempt is retained at
`/private/tmp/omni-native-cold-load-1X9Hkx`. It initialized random CPU native
weights, saved/exported them, closed that original owner, then called actual
`Worker.load` with a reviewed unchanged context. Width 8, one layer and four
router channels produced **223,187 logical parameters**, including native
modality owners. It performed no curriculum, corpus, ingest, chat or generation.
Forbidden-method spies guarded those operations.

- Initialization marker: 1.207073 s.
- Save/export/original-close marker: 3.521659 s.
- Total supervised attempt: **9.116288 s**, exit 1, no cap kill.
- Monitored process-family peak RSS: **247,513,088 B**.
- Observed allocated fixture peak: **3,895,296 B**.
- Bounds: 120 s, 512 MiB monitored family RSS and 128 MiB allocated fixture,
  sampled every 0.5 s. This is a watchdog, not an OS-enforced RSS limit.

Load returned, but the combined learned/source preservation assertion failed.
The original harness did not save each pre/post clause before raising; no
missing per-clause receipt or exact Load-only duration is reconstructed as fact.
Its supervisor receipt remains unchanged and the attempt stays **failed**.
The helper now retains pre/post observations before assertions for a future
separately authorized run; **this failed attempt was not rerun**.

Read-only inspection after termination established that current compatibility
core/plasticity files still match their original immutable generation roles:

- Core: `702edbc0fa0f9ad2fb16d0c89353bb9870173ce74c41221d023a5506fae868d2`.
- Plasticity: `062d34dd1f8cdb4163b942422e3f20a569d59b775f2aec56f31e7dd2aac24864`.
- The original generation and current packed copy both declare native checksum
  `00a310dbecd098ab9463ccbe9d9d2b88552b20ce39a3119f6c2373265fbf2c67`.
- Saved experiences, training and inference counters all remain zero.

Current source also confirms that in-memory record ordering and restored
shard-record ordering use different descriptor identities. An unchanged native
Load can therefore refresh the derived sidecar and rewrite `brain.json`. This
is a supported explanation for the overly broad metadata assertion, not an
invented retroactive pass or proof of the lost in-memory clause values.

## Derived inference sidecar: verify on Load, rebuild on explicit export

Runtime computes directly from authenticated native packed core/substrate
state. The optional `packed-ternary/` directory is a derived inference export;
it is not another authoritative memory store.

Cold Load still fully verifies an existing sidecar's bytes, layout, provenance
and native/graph bindings. Corrupt files do not become acceptable because the
copy is stale. If a verified copy no longer matches the current committed
layout/order, Load withholds its current descriptor and reports
`packedInferenceSidecar.refreshRequired`. It does **not** rebuild every shard,
write paused metadata/RNG, change a neural generation or consume stale data.
Missing/unknown copies are not advertised as current either.

The diagnostic binds the last checked committed mutable/substrate/connected
graph generation. A later commit invalidates its descriptor; it is explicitly
not an in-flight weight-readiness proof. A later ordinary Save may persist this
status, but Load itself never rewrites a sealed frontier for this purpose.

Explicit native export and checkpoint still build and verify the current copy
through the existing resource-admitted staged publication. Normal public
current/private/referenced `.omni` export flushes that checkpoint/repack and
verifies its receipt/files before archive creation. Export to an external
directory does not falsely mark the managed directory refreshed. Failed final
metadata publication restores the prior in-memory descriptor/status.

A narrow repository guard rejects **explicitly marked** stale/unverified or
generation-mismatched copies for low-level archive callers. Unmarked historical
metadata cannot reveal process-local stale findings: the guard does not claim
to establish its freshness. The normal public checkpoint/repack path is the
stronger boundary. Immutable origins and their original bytes are untouched.

Four new sidecar source/diagnostic cases passed in **6 ms**. Three new host
export-guard/source-order cases passed in **4 ms** (444 ms runner). The earlier
six cases and actual native probe were not repeated.

## Separate authorized private-copy Load-only retry — October 13

This is a **new affected-path attempt**, not a reinterpretation or rerun of the
old initialization fixture. Its preserved source remained
`/private/tmp/omni-native-cold-load-1X9Hkx`; output and complete receipts are at
`/private/tmp/omni-native-cold-load-retry-kFufNq`. The source is an already-created
random native checkpoint: width8, one layer, four router channels and **223,187
logical parameters**. It has no `origin/` directory or initial-training manifest,
so it qualifies only for checkpoint Load—not trained-origin restoration,
capability, connected-parent or exact inherited-cursor acceptance.

The reviewed command was:

```sh
PYTHONPATH=engine OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -u engine/tests/probe_tiny_load_summary.py --root /private/tmp/omni-native-cold-load-retry-kFufNq --source-root /private/tmp/omni-native-cold-load-1X9Hkx
```

The standard-library custody helper made an exact independent-file copy of
`brain/`, with no source hardlinks retained. It opened **only the copy's** SQLite
databases. The original entire fixture's file hashes, sizes, inode/device,
mtime/ctime and link counts were observed before copy, after copy, after Load and
after child termination; all stayed unchanged. No source metadata, obsolete
execution seal, origin or training receipt was created or patched to qualify it.

Only the deferred `_loading_checkpoint=True` constructor for the copied identity
was allowed. Spies rejected fresh initialization, `Worker.create`, any module
forward, curriculum, train, ingest, chat, modality generation, Save and repack.
One actual `Worker.load` used the unchanged32-token context. Production source
and kernel-code hashes were pinned before/after; no runtime source changed.
Four **new** small constructor-free custody cases passed in **0.026 seconds**
before the attempt, covering copy independence, symlink rejection, SQLite row
tampering and the fact that clock/schema/authoritative metadata are not exempt.
Changed harness/helper sources compiled; no old fixture suite was repeated.

`preload-observation.json` and `postload-observation.json` retain each observation
and **all20 clauses before asserting**. All clauses passed:

- Returned and live learned checksums equal the saved generation's
  `00a310dbecd098ab9463ccbe9d9d2b88552b20ce39a3119f6c2373265fbf2c67`.
  Logical count, mutable/substrate pointers, all saved counters and newly added
  counters stayed unchanged/zero.
- The copy's core, plasticity, `brain.json`, immutable generation files and
  existing packed sidecar bytes stayed exact. Only known SQLite physical
  WAL/SHM/header mechanics and explicitly derived per-owner live-cache files are
  classified operational; **every SQLite schema and logical row** stayed exact.
  There is no blanket learned-state exception. All excluded paths remain in the
  raw pre/post inventories rather than disappearing from the receipt.
- One checkpoint-load owner, one final summary and one parameter-accounting
  call were observed; two checksum calls occurred during Load. Metrics/runtime
  accounting matched. No forbidden method was called and cleanup succeeded.
- Saved clock bytes and activity/idle markers stayed exact. Ordinary
  `Worker.load` intentionally resumed the **live** clock from2.1330845002 to
  2.5773760830 active seconds; its advance was within the measured Load interval,
  not the offline wall interval. The new working-page clock bound read-only at
  the saved active/wall baseline, without adding a baseline table or changing
  page payloads. This is not an exact suspended-execution clock claim.
- The existing sidecar was fully verified and reported a current committed
  copy for its exact generations; runtime consumption remained false. This case
  did not exercise stale-sidecar regeneration or external export.

Actual `Worker.load` took **2.069354 seconds**; the full supervised attempt took
**4.260830 seconds**, exit0, no cap kill. Sampled whole-process-family peak RSS
was **246,562,816 B** (235.14MiB); allocated new-fixture peak **3,198,976 B**.
Minimum observed free disk was **34,241,748,992 B**, above the actual20GiB floor.
Bounds were120 seconds,512MiB sampled family RSS and128MiB allocated output,
checked every0.5 seconds. These are monitored watchdog bounds, not an OS hard
RSS guarantee or a claim to observe sub-sample peaks. Parent and child fully
exited; a subsequent process check found no retained probe/native Worker.

This accepts the **tiny existing-native private-copy Load gate only**. It is not
a paired speedup: the old9.116288-second attempt included initialization and
failed, and its actual Load-only time was not recorded. Neither this tiny model
nor that old run measures a3B/connected/frontier Load, TPS, trained knowledge,
generation, packaged UI or cross-platform behavior. Both private fixtures and
the old failed supervisor receipt are retained unchanged.

## Remaining boundary

The tiny private-copy Load-only preservation gate above passed. The separate
connected cold-Load/current-seal lifecycle gate remains **open**. No new
connected native frontier, large model, trained recall, device capture, build,
CI, package or release proof follows from these changes. Root owns the next
explicitly bounded integration decision; do not repeat retained long fixtures.
