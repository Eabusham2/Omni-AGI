# Complete multi-parent merge

## Contract

Full merge creates a new instance from at least two selected native parents. Parents remain unchanged. A reviewed branch overlay remains a separate operation; it is not full merge.

The new graph preserves each distinct parent's learned decoder, action/argument heads, memory bridge, idea adapter, spiking router, liquid controller, and image/audio/video regions. It preserves original snapshot bytes as provenance and initializes new packed connections between regions. It does not average parent weights, import a foreign foundation, turn memories into question/answer text, or start independent parent chat loops.

Complete stored state and useful integration are different outcomes. Preserved weights do not establish improved fluency, perfect retention, or immediate semantic alignment of independently trained dimensions. The future full training/behavior goal still owns that qualification.

## Implemented foundation

### Connected neural computation

`engine/omni_core/connected_regions.py` contains the registered neural graph:

- A single byte-token input advances heterogeneous parent cortical regions. Their causal workspace summaries, memory recall vectors, liquid recurrence, sparse router activity, and cortical hidden states participate in the same graph.
- Directed packed `RegionExchange` maps connect regional hidden activity. They begin at zero, preserving parent learned bytes, and receive real gradients through the joint objectives. An activity-alignment objective changes only the new bridges.
- Language and action readouts consume connected activity. Typed argument generation combines the original condition, embedding, transition, and output owners and exchanges recurrent hidden activity. This is one sampler/typed action result, not separate chat answers or answer voting.
- Parent tool schema registries are combined by exact structural identity. Distinct route and grounded-argument learning remains in its named parent owners.
- Parent media encoders feed the common neural workspace. Native generators take connected ideas; their outputs enter a learned same-brain regional synthesis. The ordinary modality interfaces and scaled tile/chunk generation remain available.
- Generation advances one causal token frontier. Parent KV and workspace statistics use the shared attention pager, avoiding a full-prefix hidden/logit allocation. Changed conditioning, learned state, batch, or shifted context invalidates the cache.
- Registered state contains every parent owner exactly once. Convenience views borrow modules without registering duplicate tensor paths. New bridge/scalar/control storage is counted as real overhead.
- Once an ordinary experience is admitted, its current neural activity can update every retained regional graph/router/liquid pathway. A prefill/retry does not itself commit recurrent timing state.
- Additional experts and layers retain per-region topology in `connected_layout`; rollback/load cannot remove original parent owners.
- An already merged parent remains a nested connected region. Its internal exchange matrices, heterogeneous cortical blocks, sparse routers, argument heads, modalities, and neural graphs remain active. Forward, cached decoding, safe-tensor restoration, attention clearing, graph publication, and pointer rollback recurse through the nested owners.

### Bounded assembly and persistence

`engine/omni_core/full_merge.py` verifies native snapshot metadata and tensor hashes, loads directly into final packed owners, attaches neural-only graph recall, and stages regional graph generations. It keeps graph pointers bound to the root commit. Loading a prior committed root restores its regional pointer mirrors after an interrupted save.

`AdaptiveBrain` now accepts a connected descriptor in its construction/loading path, selects the connected decoder/modality factories, restores nested sparse routers, records topology and graph pointers, includes regional graph parameters in accounting, and clears regional attention when Fresh is requested. The ordinary single-parent constructor path is unchanged.

Original source snapshots are distinct from mutable regional graphs under `engine/connected-state/<region-id>`. Immutable graph blobs may share storage; mutable pointers and tensor owners remain private. Saved parent histories are not consumed as hidden generation prompts.

### Common runtime adoption

`engine/omni_core/connected_runtime.py` now adopts actual parent state, not summaries:

- Verify every source conversation hash-chain row and its committed head. Preserve distinct/conflicting history; deduplicate only exact matching source payloads. A destination SQLite provenance index retains all source sequence/hash/identity mappings.
- Carry latent replay into the corresponding regional coordinate range. Destination adoption IDs make copying retry-safe without duplicating the same committed replay row.
- Preserve hot and cold working vectors, lifecycle afterimages/focus, their timing, and the links from current token spans to their afterimages. Only already-present parent token windows become the common temporary window. If those complete windows exceed the reviewed capacity, assembly rejects the settings instead of dropping the beginning or retrieving history into a prompt.
- Preserve pending chat-learning evidence, exact replay cursor, and completed-pass position. The new job references an authenticated message in the common conversation ledger.
- Preserve parent dataset cursors in an explicit queue. `full_merge_activate_ingestion` verifies the exact source bytes/name, then commits one cursor against the merged root's native generation while retaining its record/window position and learning schedule. Other cursors stay queued until the active transaction actually completes. Actual resume proof is tracked separately below.
- The merge origin records the assembled birth state and parent provenance. It explicitly says that no initial curriculum was rerun; it does not manufacture a new ground-up training receipt.
- The origin marker binds the actual metadata/core/plasticity/packed-manifest hashes. Native origin checks recognize this proof; inherited action-origin proof is verified independently against the real parent origins, including nested ones.
- Suspended parent computations now have an explicit executable queue and borrowed registered-region owner rather than archive-only retention. See [exact inherited execution](connected-execution.md) for the source contract, actual save-lock discovery, affected repair and precise proof status; a parent cursor is never sent to the new joint sampler.
- `connected_transaction.py` wraps regional admitted-experience updates in one mutation scope. Existing per-row SQL writes become savepoints, lazy-synapse before-images contain only touched rows, and successful sparse-router tile journals remain available until the outer admission completes. Failure restores regional vector/neuron/assembly/synapse state, topology, attention, liquid and firing activity. A previous successful admission is not undone by a later Stop/failure. An interrupted multi-file commit or failed enclosing root update is marked for recovery from the last atomic root checkpoint before another worker request can use it.

### Host transaction

`src/main/fullBrainMerge.ts` implements `FullBrainMergeCoordinator`:

1. Validate the complete reviewed settings and at least two distinct parent IDs.
2. Freeze parents under a caller-provided checkpoint/write lease and record exact native generation seals.
3. Stream the complete file inventory to SQLite. Consistent incremental SQLite backup captures WAL rows; source sidecars are not copied as independent data.
4. Copy every file into a content-addressed snapshot store. Use isolated reflinks where supported, otherwise a bounded cancellable stream. Deduplicate only byte-identical blobs.
5. Recheck parent seals, then run the connected assembler and verifier. The assembler receives every distinct history/artifact/pending-work file, not summaries.
6. Verify all snapshot blob checksums. Publish the new directory atomically only after the connected engine and UI metadata exist and verification succeeds.
7. Resume the same operation and settings from its durable cursor. A crash after publication recovers the same target. Explicit cancellation deletes only that unpublished staging directory.

Physical allocation checks are injectable from the existing shared resource controller. Source IDs/settings cannot silently change between retries. Failure never replaces a parent or exposes a partially built library entry.

## Runtime integration interfaces

`BrainService` now supplies the production coordinator callbacks; renderer access is typed through shared interfaces, preload, and IPC:

- `withParentsFrozen(ids, action, signal)`: checkpoint then lease all parent instances in deterministic order.
- `sealParent(id)`: verify the complete committed native state and immutable ancestry, including active source cursors.
- `validateSettings(config)` and `reserve(bytes, purpose)`: the reviewed destination permission/resource policy.
- `assemble(input)`: materialize the exact snapshot mappings into the new connected engine, combine history/provenance/pending work, create its origin, and save.
- `verify(input)`: reload and inspect the actual connected engine and preserved state before publication.

Python factory inputs:

- `inspect_parent_snapshot(new_instance_root, parent_id, engine_relative_path, generation_sha256)` binds exact local snapshot bytes.
- `connected_descriptor(parents)` records all parent provenance and shares a neural region only if the complete neural state/configuration is identical. Common ancestry alone does not authorize dropping divergent state.
- `merged_config(descriptor, settings=reviewed_settings)` preserves all regional widths and chooses the composite workspace dimensions. A simple single-stack native descriptor must not be used as its memory inventory.
- `AdaptiveBrain(..., _connected_descriptor=descriptor)` instantiates the actual connected path. Future loads read the committed metadata automatically.
- Root metadata includes `connected_regions`, `connected_layout`, and `connected_graph_states`. Every portable/snapshot path must include the files those fields reference.

Native worker endpoints now exist:

- `full_merge_assemble({brainId, storagePath, parents:[{brainId, enginePath, generationSha256}], config, streamId?})` operates only on isolated parent snapshots beneath the unpublished destination, saves the real connected engine, adopts runtime state, emits packed inference state, and creates its immutable merge origin.
- `full_merge_verify({brainId, storagePath})` performs a fresh native load, verifies registered owners, graph/runtime adoption, conversation/replay, packed execution, and the merge-origin binding.
- `full_merge_activate_ingestion({brainId, storagePath, pendingId, sourcePath, sourceName})` selects one retained source transaction. The host then calls normal ingestion with the preserved policy/epoch/transaction identity.
- `learn_action_result` accepts the closed `mergeContinuation: {pendingId, originalEvidenceId, parentId}` selector for an already completed external result. It verifies the exact inherited source, rebinds a moved owned spool to its latest committed cursor, preserves the original structural record names across renamed evidence IDs, and returns destination-local receipt IDs. It never reruns the external action.
- Assembly and verification lease the destination and close it afterward; they do not leave another live model installed. Cancellation uses the ordinary cooperative protocol and reports a safe boundary.
- `copy_connected_graph_snapshot(sourceInstanceRoot, graphStates, destinationEngine, reserve=...)` copies all committed regional generations, including nested merges. Portable files must also include the immutable source snapshot paths referenced by the descriptor and `engine/connected-runtime.sqlite3`.

### Production host adoption and lifecycle

- `BrainService.startFullMerge` requires a complete reviewed configuration, hardware tier and explicit destination tool policy. MCP starts disabled with every server override off; no previous-instance default or credential authority is inherited. A selected spill drive needs a destination-local native-picker grant. The reviewed RAM envelope is selected separately from native allocation: it is not presented as a composite-model fit calculation.
- Parent writes are leased in deterministic order, optional background learning is quiesced, and each parent is checkpointed and unloaded before copying. Seals cover all retained files and consistent SQLite snapshots, including WAL rows; process-local SHM state is not neural history. A paused partial copy does not re-checkpoint or rewrite its source ledgers before comparing the exact saved seals.
- `fullMergeHost.ts` verifies complete host hash chains, preserves original payload/provenance in a cursor-indexed history database, and projects messages, actual emitted Ponder, traces and distinct/conflicting activity into common ledgers. Native message IDs are retained where native adoption provides a mapping. Artifact indexes point to isolated, verified destination files; shared source blobs are preserved under `merge/shared-blobs`.
- Question/permission/live-observation/tool journals remain complete historical records; old approvals and unfinished effect proposals do not become destination grants or retryable commands. `fullMergeActionEvidence.ts` separately rehomes **learning-only** completed-result jobs, preserving raw outcome bytes, receipts, route-learning status and native partial-source selectors. Remapped learning IDs retain original execution provenance. Completed learning is not repeated; captured orphan outcomes and lost host acknowledgements are reconciled from their actual stored evidence.
- `start/list/status/pause/resume/cancel`, progress events, historical-record pagination and retained-source list/resume are exposed through the production bridge. Status includes the reviewed settings but not the opaque native-picker capability token. Cancel removes only unpublished output and retains a small cancellation receipt. Published retries recover the same identity without reassembling it.
- The host origin is `merged-native`, with all parent IDs and `initialCurriculumRerun: false`. Origin verification permits a fork/import to retain its authenticated ancestral owner, and rejects modified origin bytes. It does not create an initial-curriculum receipt or claim trained capability.

Scoped host verification includes four production-service/transport fixtures (history/artifacts/policy adoption, exact-copy pause/resume, early cancellation and durable cancel receipt, origin corruption), seven coordinator fixtures, and the completed-result adoption fixture with a second merge of already remapped state. The real native host fixture and portability expansion are recorded below. The native **partially learned action-result relocation** fixture is tracked separately; a completed host queue projection is not substituted for that learning proof.

### Library settings and progress UI

`FullMergePanel.tsx` is inserted into both Library entry points. It selects two or more parents, reviews the destination name/capacities/RAM/storage/learning/Best Work choices, exposes the complete configuration in an editable JSON review, and requires a separate explicit tool-access acknowledgement. Parent settings can be used as a starting point without copying their credential or drive grants. External spill roots come only from the native picker. A published receipt opens the actual new identity even before the Library refresh returns.

Durable operation cards display actual snapshot counts and native phases, with Pause, same-operation Resume, and a scoped Stop/discard confirmation. A deliberate pause or stop is presented as its receipt state, not a red failure alert. Settings and provenance are expandable; opaque picker tokens are not displayed. Existing parent cards remain available.

Three markup/config tests and an isolated Electron UI transport fixture pass. The latter exercised explicit review, exact request fields, invalid-JSON repair, pause, resume, receipt recovery after reload, discard confirmation, and unchanged parent entries. Dark desktop, narrow light, paused and stopped screenshots were inspected; the narrow layout had no horizontal overflow and hover text remained readable. This is a real rendered UI with a synthetic transport, not a second native merge/learning proof or an exhaustive all-theme product audit.

## Evidence so far

### Connected instance portability (October 7)

`src/main/connectedBrainArchive.ts` and the real `BrainRepository` lifecycle
now preserve the connected dependency closure in fork, ordinary recovery,
current export, origin export, private archive, and referenced-local export:

- Current and immutable-origin regional generations, nested graph owners,
  `connected-runtime.sqlite3`, `connected-assembly.json`, authenticated origin
  markers, and engine-local `connected-inputs` aliases remain available at the
  paths consumed by the native loader. Graph pointer mirrors are synthesized
  from the committed root metadata, not orphaned newer mirrors.
- Complete `merge/parents` trees, `full-merge.json`, `merge/parents.jsonl`,
  `merge/host-history.sqlite3`, and verified shared source blobs are retained.
  Original code/files use opaque content-addressed archive payloads rather than
  executable ZIP entries. Import does not execute them or grant access to saved
  drive paths; installation-local credentials and spill grants are not collected.
- Parent metadata/core/plasticity hashes, parent-index records, payload hashes,
  SQLite integrity, and dependency paths are checked. Mutable state receives
  isolated inodes/reflinks; read-only SHM lock tables are not saved state. Live
  runtime SQLite is captured consistently including committed WAL rows. A
  nonempty WAL/journal under an immutable parent fails rather than losing rows.
- Fork/import/recovery identity changes preserve untouched native JSON numeric
  lexemes, so Python `1.0` values inside hash-bound descriptors do not become
  JavaScript `1`. Immutable origin metadata and its original owner are not rekeyed.
- Ordinary recovery restores the current checkpoint and host history while
  preserving the instance's existing immutable origin. It is **not** a standalone
  disaster backup for a separately deleted origin; complete export includes that
  origin. Cancellation and failed reload roll back unpublished output/current
  host promotion; original parent files remain unchanged.

The actual opt-in `tests/fullMergeNativeHost.test.ts` passed in 331.947 seconds:
real two-parent assembly through `BrainService`, native fresh-load baseline,
host snapshot, native-loadable fork, all four export/import modes, and an actual
fresh worker load after restoring each imported recovery point. Parameter
checksums matched the baseline, and source/root plus immutable parent metadata
bytes remained unchanged. These are tiny native filesystem/runtime fixtures,
not trained retention, useful semantic integration, Windows execution, or large
user-data qualification. The synthetic repository tests separately exercise
nested closure, opaque source-code preservation, WAL capture, malformed/missing
payloads, path escape/symlink rejection, private mutation isolation, cancellation,
shared-blob retention, and host rollback.

Tiny component/native fixtures only; no saved user brain or training-folder run:

- Eight native core tests passed before the production `AdaptiveBrain` hook: heterogeneous widths, causal influence between regions, all auxiliary owners present/active, no floating learned parameter copy, bridge-only learning with unchanged parent tensors, safe-tensor reload, combined action/perception/audio, and exact snapshot preservation/deduplication.
- One additional token-frontier cache fixture passed after cached generation was added.
- Seven host transaction tests passed: new identity, content deduplication and conflicting history preservation, copy pause/resume, changed source/settings rejection, SQLite WAL capture, failed assembly/cancel isolation, receipt idempotency, and corruption rejection.
- TypeScript node typecheck and affected Python compilation passed at those boundaries.
- The full production two-region constructor/save/reload fixture passed after the weighted-substrate cache-reader migration (28.312 seconds).
- The actual native assembler, complete runtime adoption, packed export, merge-origin creation, and fresh verification/load fixture passed (20.302 seconds), retaining both original source snapshots byte-for-byte.
- The expanded double-merge lifecycle passed (136.939 seconds): merge two parents, merge that result with a third, reload, retain three histories and the nested graph owners, verify packed/origin state, and compute finite shared logits. This run preceded the root lane's later neuron-vector export additions; the final affected native fixture must cover those additions as well.
- One nested computational fixture passed (2.042 seconds), proving that changing an inner retained exchange affects the new outer decoder; cached decoding, native tool routing, and safe-tensor reload use those same nested owners.
- Four runtime-adoption fixtures passed (0.131 seconds), covering exact dedup, conflicting experiences, pending chat positions, afterimage/context links, unchanged parent bytes, undersized working-window rejection, and corrupt conversation/receipt rejection.
- Three worker transport fixtures passed (0.006 seconds), covering exclusive destination ownership, cooperative cancellation/lease cleanup, no additional live model after verification, and rejection from inspection workers. These are transport fixtures, not learned capability evidence.
- The real three-record partial-source adoption/resume fixture passed (120.863 seconds), with checkpointing enabled. It pauses genuine native learning, merges, activates the exact retained record/window/schedule, restarts, continues the suffix, reaches 3/3 coverage, and marks the queue complete. It includes the current recursive graph/neuron-vector packed export. This fixture exposed and led to fixes for checkpoint recomputation's pager context, valid lifecycle-field cache loading, and duplicate sparse-router loading; none was bypassed.
- One regional rollback fixture passed (8.164 seconds): a failure in the second region restores the already-changed first region's exact graph/control state and adds no lasting topology. A subsequent committed update survives another failed update. The shared sparse-router tests passed 42 cases after the retained-journal change (6.602 seconds).
- The final affected double-merge fixture passed (158.307 seconds): resume after immutable-origin creation, current neuron-vector export, and nested failure rollback.
- The standalone native snapshot fixture passed (168.715 seconds): engine → stable snapshot → isolated engine without an external `merge/parents` tree, plus inherited queued/active source cursors and actual continuation. Host full-history portability remains the separately recorded host fixture above.
- The relocated three-record completed-action cursor fixture passed (179.936 seconds): original committed boundary, merge, active owned-spool relocation to the new evidence ID, restart, suffix-only learning, 3/3 receipt, and duplicate retry with unchanged checksum. Source learning, traversal, rebinding and atomic saves are real. The separate capability-finalization boundary is explicitly mocked because these cursor parents use low-level random constructors, not claimed initialized origins; this is cursor/receipt evidence, not a capability qualification.
- The borrowed nested action-owner snapshot fixture passed (0.136 seconds). The failure/regression transaction fixture passed (0.129 seconds), restoring exact packed state and counters after an exception or rejected declared-data candidate.
- Expanded connected retention fixtures cover topology growth, pending stability events, exact counter keys, mixed training modes and individual-record regressions; the focused pair passed (0.104 seconds). A separate confirmed-experience-owner fixture passed (0.113 seconds): generic replay cannot overwrite grounded argument or confirmed route owners.
- A genuine tiny `AdaptiveBrain.create` completed its full current initial curriculum (364.27 seconds), with no readiness quiz. A distinct private graph fork plus actual connected assembly completed (445 seconds in the later reuse run). The resumed authenticated-load/one-trajectory/update/save/reload fixture passed (487.655 seconds), reusing those exact outputs without repeating training or assembly. It checks actual native head-byte change, working/router/liquid preservation, no duplicate feature forward, inherited eligibility after restart, and unchanged original parent files. This is one selected trajectory, not full 35-row replay or trained competence.
- The measured cold-load attention-membership hotspot now groups requested IDs by existing checked bucket/range descriptors, reads each required shard once with unchanged validation, and admits group/decoded-cache scratch before allocation. Focused tests passed (2.625 seconds, alongside two retention tests), including new/dirty/deleted IDs, missing IDs, unchanged hash-corruption rejection and RAM rejection before reads. This read-count proof is not a claim about overall model throughput.

### Inherited capability retention boundary

The connected loader resolves exact supported curriculum declarations through
the authenticated original parent origins, recursively. The root does not
receive a fictional initial-training manifest. Startup authenticates eligibility
without curriculum forwards. Actual joined language/internal features are
computed once inside the learning transaction, not restored from stale
parent-width caches. Direct learning snapshots traverse real
borrowed regional heads, including nested packed strengths and repeated links.

Connected start/middle/final replay uses declared action/route/argument records,
records measured before/candidate/after values and per-record retention, and
restores exact head state when a candidate regresses or is interrupted. Generic
maintenance does not overwrite confirmed route/grounded argument owners. Feature
extraction preserves live working owners, attention audits, liquid/router
activity and module modes. It neither forces a fresh head
reset nor blocks on a perfect tool quiz. This does not claim useful tool
competence or initial curriculum rerunning. The separate single-parent maintenance
path has also been repaired: no float-field seed reset, forged episodes/time, or
perfect public quiz; measured per-record retention and exact native-owner rollback
replace the old bypass. Its private initialized-parent two-step/save/reload proof
passed in 221.689 seconds, declining a candidate that improved aggregate losses
but failed a per-record condition. Maintenance-only protection leaves confirmed
grounded argument and independent-experience route owners read-only, without
disabling ordinary schema/experience learning. See the exact scope in
[research-methods.md](research-methods.md#canonical-replay-safety-repair-october-7).
The actual **connected** initialized-parent one-trajectory load/update/reload
fixture passed as recorded above; the single-parent result is not substituted
for it. Full 35-row connected maintenance remains performance-unqualified.

The short physical proof selects one declared trajectory for real neural
execution. Full 35-row maintenance keeps its default coverage, but its 35 × 32
tokens × two regions = 2,240 real graph recalls remain performance-unqualified.
That entire pass is no longer performed during startup. The complete
initial-curriculum fixture is opt-in (`OMNI_RUN_NATIVE_CURRICULUM=1`); saved
authentic fixtures can resume the smaller load/update/reload proof without
repeating training or merge.

## Required remaining integration

These remain open and must not be hidden by the foundation tests:

- Broader historical-source inspection inside the merged workspace and the remaining all-theme/all-layout application audit. The Library settings/progress flow is implemented and has the scoped rendered-UI proof above; its transport fixture is not substituted for the separate actual-native integration tests.
- The relocated completed-action cursor/receipt and authentic inherited-origin one-trajectory contracts have the scoped proofs above. Full 35-row connected maintenance performance remains open. Broader presentation/review of inherited historical question/observation records remains UI work, not restored execution authority.
- Connected import/export/fork/ordinary-recovery closure now has the scoped proofs above. Host recognition authenticates the merged origin rather than requiring a fictional initial-training receipt; imported/forked current identities preserve that authenticated ancestral owner. Cross-platform execution and larger-scale portability remain separate qualification.
- Resource-planner/descriptor reporting using actual regional owners plus new bridges and graph dimensions, not the fictitious cost of a simple stack of the summed width.
- Regional Brain Map queries and complete preservation/reconciliation of live observations/approval/question/tool receipts under the new identity.
- The actual host/native merge, restart, fork and all export/import/recovery modes have the scoped proof above. Continuing-learning/Fresh/settings-change combinations and user-facing cancellation interactions still require their separately scoped fixtures or UI checks.
- Geometry evolution beyond the implemented expert/depth topology needs explicit composite shape handling; a simple-stack width/head migrator must not silently rewrite the connected graph.
- Useful semantic integration/retention and large trained multimodal quality remain part of the separately scheduled full-corpus goal.
