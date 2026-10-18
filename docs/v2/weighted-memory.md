# Weighted native memory

Updated 2026-10-06. The memory substrate uses the same native ternary signs for learning and recall, with discrete connection strength and independently represented repeated contacts. This covers memory storage, recurrence, paging, and native representation migration. It does not establish large trained-model capability.

## Learned representation

Each neuron and assembly shares one authoritative vector row. A row contains:

- Canonical two-bit signs, with values exactly −1, 0, and +1.
- One byte per base coordinate: low four bits are strength minus one, giving strengths 1–16; high four bits are resistance 0–15.
- An eight-byte deterministic adaptation counter.
- Explicit repeated contacts, each encoded as a little-endian eight-byte coordinate, one padded sign byte, and one strength/resistance byte. Record order identifies independent contacts, including opposing signs at the same coordinate.

`weighted_vsa_row.py` computes each coordinate as the base sign times its strength plus its repeated contacts. Normalized floating activity is transient; the durable learned row contains no floating master or stored answer text. Seeded adaptation changes signs, strengths, resistance, and, where required, explicit repeated contacts. Safetensor format 2 carries all these fields and validates their hashes, geometry, padding, and offsets.

Graph synapses remain individual contacts. Their sign is exactly ternary and their magnitude is a packed four-bit strength. Their existing eligibility, plasticity, use count, stability, and timing fields remain the sole control state. Strength saturation can create another independently represented contact rather than clipping the endpoint's total contribution indefinitely. Every contact participates in recurrence; fan-in normalization uses the sum of absolute integer contributions to preserve contraction.

These changes do not introduce Q/A matching, source-text injection, or a second fact store. They preserve the ordinary experience, coactivity, interference, salience, stability, decay, and working-memory admission paths.

## Storage and resource costs

For a vector with D coordinates, base learned storage is `ceil(D/4) + D + 8` bytes, plus 10 bytes per repeated contact. Paged working rows add a 12-byte envelope. Filesystem, SQLite, identifier, metadata, cache, and temporary-computation costs remain additional costs, not part of a claimed 1.58 bits per complete connection.

Fresh-memory admission includes the strength bytes and row envelope. Paged imports, reads, similarity, and adaptation reserve their actual payload and temporary computation. Batch admission charges the complete weighted row, including repeated contacts; a refusal rolls back the transaction without advancing the row counter.

Both resident and paged vector stores expose:

- `row_payload(id)`: complete signs, strengths, resistance, and repeated-contact bytes.
- `packed_row(id)`: base sign plane only, for explicit sign inspection.
- `connection_state(id)`, `parallel_records(id)`, and `update_count(id)`.
- `weighted_levels(id)`: the summed integer contribution.
- `current_snapshot()`: a generation-bound streaming guard. The resident token uses a monotonic runtime revision and does not alter saved checksums.

`PagedAssemblyIndex.get_weighted_vector(id)` returns the complete payload. Its original `get_packed_vector(id)` remains explicitly sign-only. Exact similarity pages and recurrence consume weighted payloads, not that inspection-only sign accessor.

## Incremental accounting and inspection

Resident vector stores keep an exact repeated-record byte counter across assignment, adaptation, append, merge, deletion, and reload. Paged stores maintain payload bytes, weighted-row count, and repeated-contact count through SQLite triggers in the same transaction as the row mutation. An old working cache reconstructs these counters once. Missing or changed triggers fail validation; ordinary telemetry reads indexed metadata rather than scanning the vector table.

`parameter_accounting()` reports base coordinates, repeated contacts, their combined logical count, learned bytes, structural bytes, allocated payload bytes, and row envelopes. The brain aggregates these once across its root and recursively connected regional stores. `substrateVectorParameters` includes repeated contacts. `substrateRepeatedVectorContacts` names that component explicitly; `nativeRepeatedConnections` keeps its existing core-only meaning so initial-core receipt comparisons remain valid.

Packed byte totals cover core native buffers and substrate vector payloads. They exclude sparse graph database files and host object overhead; the returned `packedByteAccountingScope` names that boundary. They are not a substitute for measured process RSS or disk allocation.

Live and persisted graph inspection exposes exact `connectionStrength`, `effectiveContribution`, `isRepeatedContact`, `parallelRootId`, and `parallelOrdinal`. Derived inspection indexes rebuild under their new version so old sign-only records cannot masquerade as current weighted data. The cortical viewport reads gain/resistance bytes beside selected signs and exposes actual pooled repeated-contact pages. It skips unused/tombstoned slots while advancing the cursor through scanned slots and validates live endpoints against their parent owner. No activation is inferred from strength or sign. Page capacity is not mislabeled as live contact count; whole-model totals remain available through runtime accounting.

The paged assembly schema also accepts the existing, validated numeric lifecycle state written by automatic settling: activity, retention, plasticity, reinforcement, unfinished state, finite signal values, and its timestamp. Source text remains rejected. This repairs cold loading of genuinely learned parents without discarding their working-to-memory dynamics.

## Save and load

The outer immutable substrate format stays version 3. New generation manifests declare `nativeWeightedRepresentation: 2`. Earlier native sign-only memory receives unit strengths and empty repeated-contact state without retraining.

`NeuralSubstrate.native_weighted_migration_required` derives conversion requirements from the committed generation. Its cached result is keyed to that generation, so restoring an original pointer after an interrupted save also restores the pending conversion requirement.

`verify_saved_weighted_generation(root, target, source_pointer=source)` authenticates both generations and compares every original neuron, assembly, and synapse through bounded record hashes. It verifies all signs, counters, controls, metadata, preexisting strengths and contacts; missing old companions must become unit defaults and empty contacts. It rejects added, removed, changed, duplicated, or incomplete records. File identities are rechecked before the proof completes. Its temporary SQLite table contains only comparison hashes and is removed afterward; it is not learned memory.

The brain invokes this proof after staging weighted memory and before publishing its neural/joint generation through `brain.json`. The migration proof does not advance source positions, coverage, epochs, or schedules.

Cold cache rebuilding now includes actual graph strengths in its forward index. Derived indexes declare strength coverage; an old index without that contract is rebuilt once from authenticated neural shards. Warm indexed recurrence continues to avoid paging every cold record. Explicit scrub rejects omitted non-unit strengths.

## Packed export interface

`LazyPersistedSynapses.stream_dynamic_connection_state(...)` returns:

```python
(ConnectionStateSource(shape=(1, count), pages=...), count, order_sha256, order_basis)
```

Its order identity matches `stream_dynamic_pack_state(...)`. Each real contact, including a repeated graph contact, occupies one logical entry. The low nibble carries gain minus one; high resistance bits stay zero because graph control state is preserved separately in substrate shards. Streams reject structural changes that have not been checkpointed and any revision change during traversal.

The root export implementation owns binding these streams into `.omni` packages, connected-parent exports, complete runtime accounting, and the viewer. The stream interfaces do not claim those integrations are automatically complete.

## Focused verification

The following scoped checks passed on this host; overlapping totals are not additive:

- `test_weighted_memory_migration`: 10 cases, including real resident and paged conversion, exact preexisting weighted-state preservation, missing companions, changed gains, new contacts, artifact mutation, old derived-index rebuilding, and streamed export order/mutation guards.
- `test_weighted_memory_substrate`, `test_paged_packed_vectors`, and `test_paged_vector_scoring`: 25 cases, including actual integer gains/opposing contacts in activity and similarity, deterministic resistance updates, recurrence contraction, paging, and exact refusal rollback.
- `test_paged_assembly_index`, `test_live_paging_migration`, and `test_native_representation_migration`: 62 cases, including full-payload migration, unchanged cursor positions, additive-only conversion, and interrupted atomic-generation recovery.
- `test_weighted_memory_migration`, `test_lazy_substrate_loading`, and `test_committed_paged_cache`: 29 cases after the derived-strength contract was added.
- `test_packed_vsa_vectors`: 7 cases after adding runtime revision guards. Checkpoint metadata and learned checksums do not contain that runtime revision.

The subsequent accounting/inspection run passed **56 tests** across `test_cortical_inspection`, `test_persisted_substrate_inspection`, `test_substrate_queries`, `test_vector_parameter_accounting`, and `test_paged_assembly_index`. The four new accounting fixtures verify transaction rollback, deletion/reopen, reconstructible counters, no corpus-table reads during repeated telemetry refreshes, and connected-store deduplication. New cortical fixtures read real gains and repeated-contact endpoints and cross an empty/tombstoned viewport. Renderer typechecking and eight focused presentation/source-contract tests passed. Full visual UI review remains a separate gate.

Existing ordinary-experience vector learning, paged metadata/writer, and memory-scaling cases passed. The separate restart-through-curriculum fixture subsequently passed in 275.237 seconds after the receipt lane corrected initial-count comparison for 576 genuinely added recurrent contacts. Its checksum, tensor, and coverage checks remained intact; five constructor-free tamper regressions also passed. See [initial receipt accounting](initial-receipt-accounting.md) for that separate fix and evidence.

For the direct packed cosine primitive, six alternating same-process measurements of 2,000 calls on a 256-coordinate weighted row gave median 0.2185 seconds for the prior tensor-expanded implementation and 0.1967 seconds for the direct integer reader, with the same score. This limited comparison does not establish whole-brain throughput or large-memory paging performance.
